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

    def predict(self, tweet: str, enable_plt: bool = True) -> dict:
        cleaned = self.__clean_text(tweet)

        x_tfidf = self.tfidf.transform([cleaned])
        sbert_vec = _sbert.encode([cleaned], convert_to_numpy=True).astype(np.float32)
        x_features = sp.hstack([x_tfidf, sp.csr_matrix(sbert_vec)], format="csr")

        prediction = self.model.predict(x_features)[0]
        probability = self.model.predict_proba(x_features)[0]

        id_to_name = {0: "Negative", 1: "Positive"}
        id_to_colour = {0: "#e74c3c", 1: "#2ecc71"}

        words = cleaned.split()
        tfidf_feature_names = np.array(self.tfidf.get_feature_names_out())

        n_tfidf = x_tfidf.shape[1]
        coef_tfidf = self.model.coef_[0, :n_tfidf]

        word_weights = {}
        for word in words:
            matches = np.where(tfidf_feature_names == word)[0]
            if len(matches) > 0:
                idx = matches[0]
                word_weights[word] = float(coef_tfidf[idx] * x_tfidf[0, idx])
            else:
                word_weights[word] = 0.0

        all_vals = list(word_weights.values())
        max_abs = max(abs(v) for v in all_vals) if all_vals else 1.0
        if max_abs == 0:
            max_abs = 1.0
        word_weights_norm = {w: v / max_abs for w, v in word_weights.items()}

        token_weights = []
        for orig in tweet.split():
            clean = self.__clean_text(orig)
            token_weights.append({"word": orig, "weight": round(word_weights_norm.get(clean, 0.0), 4)})

        if enable_plt:
            labels = [id_to_name[0], id_to_name[1]]
            colours = [id_to_colour[0], id_to_colour[1]]

            fig = plt.figure(figsize=(18, 8))
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

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
            pos_prob = probability[1]
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

            sorted_words = sorted(word_weights.items(), key=lambda x: x[1])
            w_labels = [w for w, _ in sorted_words]
            w_values = [v for _, v in sorted_words]
            bar_colours = [id_to_colour[1] if v >= 0 else id_to_colour[0] for v in w_values]
            ax3.barh(w_labels, w_values, color=bar_colours)
            ax3.axvline(0, color="black", linewidth=0.8)
            ax3.set_xlabel("Weight Contribution (model coef scale)")
            ax3.set_title("Word Influence on Prediction")
            for i, (val, _) in enumerate(zip(w_values, w_labels)):
                ax3.text(
                    val + (0.002 if val >= 0 else -0.002),
                    i,
                    f"{val:.3f}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8,
                )

            fig.suptitle(
                f'Sentiment Prediction: "{tweet}"\nPredicted: {id_to_name[prediction]}',
                fontsize=11,
            )
            plt.tight_layout()
            plt.show()

        return {
            "sentiment": id_to_name[prediction],
            "confidence": round(max(probability) * 100, 2),
            "token_weights": token_weights,
        }