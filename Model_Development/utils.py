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

    prediction = model.predict(X_tfidf)[0]
    probability = model.predict_proba(X_tfidf)[0]

    # Binary classification: only 0 (Negative) and 1 (Positive)
    id_to_name = {0: 'Negative', 1: 'Positive'}
    id_to_colour = {0: '#e74c3c', 1: '#2ecc71'}
    
    labels = [id_to_name[0], id_to_name[1]]
    colours = [id_to_colour[0], id_to_colour[1]]

    _fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, probability * 100, color=colours)

    # Highlight predicted class
    bars[prediction].set_edgecolor('black')
    bars[prediction].set_linewidth(2)

    for i, (bar, prob) in enumerate(zip(bars, probability)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=10)

    ax.set_xlim(0, 115)
    ax.set_xlabel('Confidence (%)')
    ax.set_title(f'Sentiment Prediction: "{tweet}"\nPredicted: {id_to_name[prediction]}')
    plt.tight_layout()
    plt.show()

    return {'sentiment': id_to_name[prediction], 'confidence': round(max(probability) * 100, 2)}