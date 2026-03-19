import os
import re
import warnings

import joblib
import contractions
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sentence_transformers import SentenceTransformer

_sbert = SentenceTransformer("all-MiniLM-L6-v2")


class SentimentAnalyser:
    def __init__(self, csv_folder: str = None, args: list[str] = None):
        path = "csv"
        if csv_folder is not None:
            path = os.path.join(csv_folder, path)

        model_path = f"{path}/sentiment_model.pkl"
        tfidf_path = f"{path}/tfidf_dump.pkl"

        if args is not None and "train-for-me=true" in args:
            if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
                import common.train as train

                warnings.warn(
                    "Training model due to `train-for-me=true` arg being enabled"
                )
                train.train(csv_folder)
            else:
                warnings.warn(
                    "`train-for-me=true` arg is enabled, however model has been located therefore no need to train"
                )
                print("Running!")

        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)

    @staticmethod
    def __clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"&\w+;", "", text)
        text = contractions.fix(text)
        text = text.lower().strip()
        return text

    def predict(self, text: str, enable_plt: bool = True) -> dict:
        """
        Predict sentiment and include explainability data for TF-IDF + SBERT.
        Returns website-friendly visual data as well.
        """
        cleaned = self.__clean_text(text)

        x_tfidf = self.tfidf.transform([cleaned])
        sbert_full = _sbert.encode([cleaned], convert_to_numpy=True).astype(np.float32)
        sbert_128 = sbert_full[:, :128]

        model_n_features = int(self.model.coef_.shape[1])
        tfidf_n_features = int(x_tfidf.shape[1])

        if model_n_features == tfidf_n_features + 128:
            mode = "hybrid"
            x_features = sp.hstack([x_tfidf, sp.csr_matrix(sbert_128)], format="csr")
        elif model_n_features == 128:
            mode = "sbert_only"
            x_features = sp.csr_matrix(sbert_128)
        elif model_n_features == tfidf_n_features:
            mode = "tfidf_only"
            x_features = x_tfidf
        else:
            raise ValueError(
                f"Model expects {model_n_features} features, but got TF-IDF={tfidf_n_features} and SBERT=128."
            )

        prediction = int(self.model.predict(x_features)[0])
        probability = self.model.predict_proba(x_features)[0]

        id_to_name = {0: "Negative", 1: "Positive"}
        id_to_colour = {0: "#e74c3c", 1: "#2ecc71"}

        intercept = float(self.model.intercept_[0])
        coef = self.model.coef_[0]

        tfidf_logit = 0.0
        sbert_logit = 0.0
        if mode in ("hybrid", "tfidf_only"):
            coef_tfidf = coef[:tfidf_n_features]
            tfidf_logit = float(x_tfidf.dot(coef_tfidf)[0])
        if mode in ("hybrid", "sbert_only"):
            start = tfidf_n_features if mode == "hybrid" else 0
            coef_sbert = coef[start : start + 128]
            sbert_logit = float(np.dot(sbert_128[0], coef_sbert))

        total_logit = intercept + tfidf_logit + sbert_logit
        pos_prob = float(probability[1])

        words = cleaned.split()
        tfidf_feature_names = np.array(self.tfidf.get_feature_names_out())
        word_weights = {}

        if mode in ("hybrid", "tfidf_only"):
            coef_tfidf_words = coef[:tfidf_n_features]
            for word in words:
                matches = np.where(tfidf_feature_names == word)[0]
                if len(matches) > 0:
                    idx = matches[0]
                    word_weights[word] = float(coef_tfidf_words[idx] * x_tfidf[0, idx])
                else:
                    word_weights[word] = 0.0
        else:
            for word in words:
                word_weights[word] = 0.0

        all_vals = list(word_weights.values())
        max_abs = max(abs(v) for v in all_vals) if all_vals else 1.0
        if max_abs == 0:
            max_abs = 1.0
        word_weights_norm = {w: v / max_abs for w, v in word_weights.items()}

        token_weights = []
        for orig in text.split():
            clean = self.__clean_text(orig)
            token_weights.append({"word": orig, "weight": round(word_weights_norm.get(clean, 0.0), 4)})

        top_dims = []
        if mode in ("hybrid", "sbert_only"):
            start = tfidf_n_features if mode == "hybrid" else 0
            coef_sbert = coef[start : start + 128]
            dim_contrib = sbert_128[0] * coef_sbert
            top_idx = np.argsort(np.abs(dim_contrib))[-8:][::-1]
            top_dims = [
                {
                    "dimension": int(i),
                    "value": float(sbert_128[0, i]),
                    "coef": float(coef_sbert[i]),
                    "contribution": float(dim_contrib[i]),
                }
                for i in top_idx
            ]

        if enable_plt:
            labels = [id_to_name[0], id_to_name[1]]
            colours = [id_to_colour[0], id_to_colour[1]]

            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)

            bars = ax1.barh(labels, probability * 100, color=colours)
            bars[prediction].set_edgecolor("black")
            bars[prediction].set_linewidth(2)
            for bar, prob in zip(bars, probability):
                ax1.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{prob * 100:.1f}%",
                    va="center",
                    fontsize=10,
                )
            ax1.set_xlim(0, 115)
            ax1.set_xlabel("Confidence (%)")
            ax1.set_title("Prediction Confidence")

            x = np.linspace(-8, 8, 300)
            sigmoid = 1 / (1 + np.exp(-x))
            sigmoid_x = np.log(pos_prob / (1 - pos_prob + 1e-9))
            sigmoid_x = np.clip(sigmoid_x, -8, 8)
            ax2.plot(x, sigmoid, color="steelblue", linewidth=2, label="Sigmoid curve")
            ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision boundary (0.5)")
            ax2.axvline(
                sigmoid_x,
                color=id_to_colour[prediction],
                linestyle="--",
                linewidth=1.5,
                label=f"This input (logit={sigmoid_x:.2f})",
            )
            ax2.scatter(
                [sigmoid_x],
                [pos_prob],
                color=id_to_colour[prediction],
                zorder=5,
                s=80,
                label=f"P(Positive)={pos_prob:.2f}",
            )
            ax2.set_xlabel("Logit (log-odds)")
            ax2.set_ylabel("P(Positive)")
            ax2.set_title("Sigmoid Logistic Regression")
            ax2.set_ylim(-0.05, 1.05)
            ax2.legend(fontsize=8)

            sorted_words = sorted(word_weights.items(), key=lambda kv: kv[1])
            w_labels = [w for w, _ in sorted_words]
            w_values = [v for _, v in sorted_words]
            bar_colours = [id_to_colour[1] if v >= 0 else id_to_colour[0] for v in w_values]
            ax3.barh(w_labels, w_values, color=bar_colours)
            ax3.axvline(0, color="black", linewidth=0.8)
            ax3.set_xlabel("Weight Contribution")
            ax3.set_title("TF-IDF Token Influence")

            contrib_labels = ["Intercept", "TF-IDF", "SBERT"]
            contrib_values = [intercept, tfidf_logit, sbert_logit]
            contrib_colours = ["#95a5a6", "#3498db", "#9b59b6"]
            ax4.bar(contrib_labels, contrib_values, color=contrib_colours)
            ax4.axhline(0, color="black", linewidth=0.8)
            ax4.set_ylabel("Logit contribution")
            ax4.set_title("Feature Group Contributions")

            fig.suptitle(
                f'Sentiment Prediction: "{text}"\nPredicted: {id_to_name[prediction]} ({mode})',
                fontsize=11,
            )
            plt.tight_layout()
            plt.show()

        return {
            "sentiment": id_to_name[prediction],
            "confidence": round(max(probability) * 100, 2),
            "token_weights": token_weights,
            "sbert_summary": {
                "mode": mode,
                "embedding_norm_l2": float(np.linalg.norm(sbert_128)),
                "tfidf_logit": round(tfidf_logit, 6),
                "sbert_logit": round(sbert_logit, 6),
                "intercept_logit": round(intercept, 6),
                "total_logit": round(total_logit, 6),
                "prob_positive": round(pos_prob, 6),
                "top_sbert_dimensions": top_dims,
            },
            "visual_data": {
                "class_probability": {
                    "negative": float(probability[0]),
                    "positive": float(probability[1]),
                },
                "modality_contributions": [
                    {"name": "Intercept", "value": float(intercept)},
                    {"name": "TF-IDF", "value": float(tfidf_logit)},
                    {"name": "SBERT", "value": float(sbert_logit)},
                ],
                "top_sbert_dimensions": top_dims,
            },
        }