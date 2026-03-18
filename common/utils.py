import os
import re
import warnings

import joblib
import contractions
import matplotlib.pyplot as plt
import pandas as pd
from afinn import Afinn
import scipy.sparse as sp

_afinn = Afinn()

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
                warnings.warn("Training model due to `train-for-me=true` arg being enabled")
                train.train(csv_folder)
            else:
                warnings.warn("`train-for-me=true` arg is enabled, however model has been located therefore no need to train")
                print("Running!")

        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)

    @staticmethod
    def __clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'&\w+;', '', text)
        text = contractions.fix(text)
        text = text.lower().strip()
        return text

    def predict(self, tweet: str, enable_plt: bool = True) -> dict[str, str]:
        cleaned = self.__clean_text(tweet)
        x_tfidf = self.tfidf.transform([cleaned])
        
        afinn_score = sp.csr_matrix([[_afinn.score(cleaned)]])
        x_features = sp.hstack([x_tfidf, afinn_score], format="csr")

        prediction = self.model.predict(x_features)[0]
        probability = self.model.predict_proba(x_features)[0]

        id_to_name = {0: 'Negative', 1: 'Positive'}
        id_to_colour = {0: '#e74c3c', 1: '#2ecc71'}

        labels = [id_to_name[0], id_to_name[1]]
        colours = [id_to_colour[0], id_to_colour[1]]

        if enable_plt:
            _fig, ax = plt.subplots(figsize=(8, 4))

            bars = ax.barh(labels, probability * 100, color=colours)

            bars[prediction].set_edgecolor('black')
            bars[prediction].set_linewidth(2)

            for i, (bar, prob) in enumerate(zip(bars, probability)):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{prob * 100:.1f}%', va='center', fontsize=10)

            ax.set_xlim(0, 115)
            ax.set_xlabel('Confidence (%)')
            ax.set_title(f'Sentiment Prediction: "{tweet}"\nPredicted: {id_to_name[prediction]}')
            plt.tight_layout()
            plt.show()

        return {'sentiment': id_to_name[prediction], 'confidence': round(max(probability) * 100, 2)}
