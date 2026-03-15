import mlcroissant as mlc
import pandas as pd
from datasets import load_dataset
import os
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
import contractions
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

n_jobs = os.cpu_count()

def train(csv_folder: str = None):
    path = "csv"
    if csv_folder is not None:
        path = os.path.join(csv_folder, path)
    os.makedirs(path, exist_ok=True)

    stages = [
        "Loading datasets",
        "Cleaning data",
        "Preprocessing text",
        "Tokenising",
        "Vectorising (TF-IDF)",
        "Training model",
        "Saving files",
    ]

    with tqdm(total=len(stages), desc="Training pipeline", unit="stage") as pbar:
        pbar.set_description("Loading datasets")
        croissant_dataset = mlc.Dataset(
            'https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/croissant/download')
        record_sets = croissant_dataset.metadata.record_sets
        df1 = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))
        ds = load_dataset("NNEngine/Sentiment-Analysis-Complex", split="train")
        df2 = ds.to_pandas()
        pbar.update(1)

        pbar.set_description("Cleaning data")
        df = df1.drop(columns=['twitter_training.csv/2401', 'twitter_training.csv/Borderlands'])
        df = df.drop(index=0).reset_index(drop=True)
        df.columns = ['Sentiment', 'Text Content']
        df['Sentiment'] = df['Sentiment'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        df['Text Content'] = df['Text Content'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df[~(df["Text Content"].fillna("").str.strip().eq(""))].reset_index(drop=True)
        df = df[df['Text Content'].str.strip().str.len() > 0].reset_index(drop=True)
        df = df.drop_duplicates(subset='Text Content').reset_index(drop=True)
        df['Text Content'] = df['Text Content'].str.replace('<unk>', '', regex=False)
        df['Text Content'] = df['Text Content'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df = df[df['Text Content'].str.strip().str.len() > 0].reset_index(drop=True)
        df['Sentiment'] = df['Sentiment'].str.lower()
        df2 = df2.rename(columns={'label': 'Sentiment', 'text': 'Text Content'})
        df2 = df2.drop(columns=['id'])
        df = pd.concat([df, df2], ignore_index=True)
        df.to_parquet(f'{path}/all_data.parquet', index=False, engine='pyarrow', compression='snappy')
        pbar.update(1)

        pbar.set_description("Preprocessing text")
        nltk.download('stopwords', quiet=True)
        nltk.download('twitter_samples', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        stop_words = set(stopwords.words('english'))
        negations = {'no', 'not', 'nor', 'neither', 'never', 'none',
                     "don't", "won't", "can't", "isn't", "aren't", "wasn't"}
        stop_words = stop_words - negations

        def clean_text_content(text):
            if pd.isna(text):
                return ""
            text = str(text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            text = re.sub(r'http\S+|www\.\S+', '', text)
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'&\w+;', '', text)
            text = contractions.fix(text)
            return text.lower().strip()

        def process_chunk(chunk):
            tqdm.write(f"  cleaning chunk {chunk.index[0]}–{chunk.index[-1]}")
            return chunk.apply(clean_text_content)

        df = df.reset_index(drop=True)
        chunks = [df['Text Content'].iloc[i::4] for i in range(4)]
        results = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(chunk) for chunk in chunks)
        df['Mutated Text Content'] = pd.concat(results).sort_index().reset_index(drop=True)
        pbar.update(1)

        pbar.set_description("Tokenising")
        tokeniser = TweetTokenizer(strip_handles=True, reduce_len=True)

        def tokenise_text_content(text):
            if not text:
                return []
            tokens = tokeniser.tokenize(text)
            tokens = [re.sub(r'[^\w\s]', '', w) for w in tokens]
            return [w for w in tokens if w and w not in stop_words]

        def process_token_chunk(chunk):
            tqdm.write(f"  tokenising chunk {chunk.index[0]}–{chunk.index[-1]}")
            return chunk.apply(tokenise_text_content)

        chunks = [df['Mutated Text Content'].iloc[i::4] for i in range(4)]
        results = Parallel(n_jobs=n_jobs)(delayed(process_token_chunk)(chunk) for chunk in chunks)
        df["Text Tokens"] = pd.concat(results).sort_index().reset_index(drop=True)

        df = df[df["Text Tokens"].apply(len) > 0].reset_index(drop=True)
        df = df[~df['Sentiment'].isin(['neutral', 'irrelevant'])]
        df['Sentiment'] = df['Sentiment'].map({'positive': 1, 'negative': 0})
        df.to_parquet(f'{path}/preprocess.parquet', index=False, engine='pyarrow', compression='snappy')
        pbar.update(1)

        pbar.set_description("Vectorising (TF-IDF)")
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        X_tfidf = tfidf.fit_transform(
            tqdm(df['Mutated Text Content'], desc="  TF-IDF rows", leave=False)
        )
        X = X_tfidf
        y = df['Sentiment']
        joblib.dump(tfidf, f'{path}/tfidf_dump.pkl')
        sp.save_npz(f'{path}/x_features.npz', X)
        y.to_csv(f'{path}/y_sentiment_labels.csv', index=False)
        pbar.update(1)

        pbar.set_description("Training model")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(class_weight='balanced', solver="saga", verbose=1)
        model.fit(X_train, y_train)
        pbar.update(1)

        pbar.set_description("Saving files")
        sp.save_npz(f'{path}/X_test.npz', X_test)
        y_test.to_csv(f'{path}/y_test.csv', index=False)
        joblib.dump(model, f'{path}/sentiment_model.pkl')
        pbar.update(1)

    print("Training complete.")
    return model, tfidf