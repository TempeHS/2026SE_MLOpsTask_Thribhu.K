import re
import joblib
import contractions
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

model = joblib.load('csv/sentiment_model.pkl')
tfidf = joblib.load('csv/tfidf_dump.pkl')

def clean_tweet(text):
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

def predict_sentiment_graph(tweet):
    cleaned = clean_tweet(tweet)
    X_tfidf = tfidf.transform([cleaned])

    X = X_tfidf

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    id_to_name = {0: 'Neutral', 1: 'Negative', 2: 'Positive', 3: 'Irrelevant'}
    id_to_colour = {0: '#95a5a6', 1: '#e74c3c', 2: '#2ecc71', 3: '#3498db'}

    class_ids = list(model.classes_)
    labels = [id_to_name[c] for c in class_ids]
    colours = [id_to_colour[c] for c in class_ids]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, probability * 100, color=colours)

    pred_idx = class_ids.index(prediction)
    bars[pred_idx].set_edgecolor('black')
    bars[pred_idx].set_linewidth(2)

    for bar, prob in zip(bars, probability):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=10)

    ax.set_xlim(0, 115)
    ax.set_xlabel('Confidence (%)')
    ax.set_title(f'Sentiment Prediction: "{tweet}"\nPredicted: {id_to_name[prediction]}')
    plt.tight_layout()
    plt.show()

    return {'sentiment': id_to_name[prediction], 'confidence': round(max(probability) * 100, 2)}