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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from afinn import Afinn
tqdm.pandas()

n_jobs = 2

_PUNCT = re.compile(r"[^\w\s]")
_afinn = Afinn()


def _tokenise_row(text, stop_words):
    try:
        if not text:
            return []
        tokeniser = TweetTokenizer(strip_handles=True, reduce_len=True)
        tokens = tokeniser.tokenize(text)
        return [
            w for w in (_PUNCT.sub("", t) for t in tokens) if w and w not in stop_words
        ]
    except Exception:
        return []


def _afinn_row(text):
    return _afinn.score(text if text else "")


def train(csv_folder: str = None):
    path = csv_folder or "csv"
    if os.path.basename(os.path.normpath(path)) != "csv":
        path = os.path.join(path, "csv")
    os.makedirs(path, exist_ok=True)

    CP_ALL_DATA    = f"{path}/all_data.parquet"
    CP_PREPROCESS  = f"{path}/preprocess.parquet"
    CP_TFIDF       = f"{path}/tfidf_dump.pkl"
    CP_X_FEATURES  = f"{path}/x_features.npz"
    CP_Y_LABELS    = f"{path}/y_sentiment_labels.csv"
    CP_X_TEST      = f"{path}/X_test.npz"
    CP_Y_TEST      = f"{path}/y_test.csv"
    CP_MODEL       = f"{path}/sentiment_model.pkl"

    stages = [
        "Loading datasets",
        "Cleaning data",
        "Preprocessing text",
        "Tokenising",
        "Vectorising (TF-IDF + AFINN)",
        "Training model",
        "Saving files",
    ]

    try:
        with tqdm(
            total=len(stages),
            desc="Training pipeline",
            unit="stage",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ) as pbar:

            # load and clean
            if os.path.exists(CP_ALL_DATA):
                tqdm.write(f"[checkpoint] Skipping load+clean, reading {CP_ALL_DATA}")
                df = pd.read_parquet(CP_ALL_DATA)
                pbar.update(2)
            else:
                pbar.set_description("Loading datasets")
                croissant_dataset = mlc.Dataset(
                    "https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/croissant/download"
                )
                record_sets = croissant_dataset.metadata.record_sets
                df1 = pd.DataFrame(
                    croissant_dataset.records(record_set=record_sets[0].uuid)
                )
                ds = load_dataset("NNEngine/Sentiment-Analysis-Complex", split="train")
                df2 = ds.to_pandas()
                pbar.update(1)

                pbar.set_description("Cleaning data")
                df = df1.drop(
                    columns=[
                        "twitter_training.csv/2401",
                        "twitter_training.csv/Borderlands",
                    ]
                )
                df = df.drop(index=0).reset_index(drop=True)
                df.columns = ["Sentiment", "Text Content"]
                df["Sentiment"] = df["Sentiment"].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )
                df["Text Content"] = df["Text Content"].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )
                df = df.sample(frac=1).reset_index(drop=True)
                df = df[~(df["Text Content"].fillna("").str.strip().eq(""))].reset_index(drop=True)
                df = df[df["Text Content"].str.strip().str.len() > 0].reset_index(drop=True)
                df = df.drop_duplicates(subset="Text Content").reset_index(drop=True)
                df["Text Content"] = df["Text Content"].str.replace("<unk>", "", regex=False)
                df["Text Content"] = (
                    df["Text Content"].str.replace(r"\s+", " ", regex=True).str.strip()
                )
                df = df[df["Text Content"].str.strip().str.len() > 0].reset_index(drop=True)
                df["Sentiment"] = df["Sentiment"].str.lower()
                df2 = df2.rename(columns={"label": "Sentiment", "text": "Text Content"})
                df2["Sentiment"] = df2["Sentiment"].astype(str).str.lower()
                df2 = df2.drop(columns=["id"])
                df = pd.concat([df, df2], ignore_index=True)
                df = df.drop_duplicates(subset="Text Content").reset_index(drop=True)
                df.to_parquet(CP_ALL_DATA, index=False, engine="pyarrow", compression="snappy")
                pbar.update(1)

                del df1, df2, ds

            # preprocess and tokenise
            if os.path.exists(CP_PREPROCESS):
                tqdm.write(f"[checkpoint] Skipping preprocess+tokenise, reading {CP_PREPROCESS}")
                df = pd.read_parquet(CP_PREPROCESS)
                df["Text Tokens"] = df["Text Tokens"].apply(
                    lambda x: x if isinstance(x, list) else list(x)
                )
                pbar.update(2)
            else:
                pbar.set_description("Preprocessing text")
                nltk.download("stopwords", quiet=True)
                nltk.download("twitter_samples", quiet=True)
                nltk.download("wordnet", quiet=True)
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                stop_words = set(stopwords.words("english"))
                negations = {
                    "no", "not", "nor", "neither", "never", "none",
                    "don't", "won't", "can't", "isn't", "aren't", "wasn't",
                }
                stop_words = stop_words - negations

                cleaning_steps = [
                    ("Removing non-ASCII",    lambda s: s.str.replace(r"[^\x00-\x7F]+", "", regex=True)),
                    ("Removing URLs",         lambda s: s.str.replace(r"http\S+|www\.\S+", "", regex=True)),
                    ("Removing hashtags",     lambda s: s.str.replace(r"#\w+", "", regex=True)),
                    ("Removing HTML ents",    lambda s: s.str.replace(r"&\w+;", "", regex=True)),
                    ("Expanding contractions",lambda s: s.apply(contractions.fix)),
                    ("Lowercasing",           lambda s: s.str.lower().str.strip()),
                ]
                col = df["Text Content"].fillna("")
                for _, fn in tqdm(cleaning_steps, desc="  Cleaning text", position=1, leave=False):
                    col = fn(col)
                df["Mutated Text Content"] = col
                df = df.drop(columns=["Text Content"]).reset_index(drop=True)
                pbar.update(1)

                pbar.set_description("Tokenising")
                CHUNK = 50_000
                texts = df["Mutated Text Content"].tolist()
                results = []
                for i in tqdm(
                    range(0, len(texts), CHUNK),
                    desc="  Tokenising chunks",
                    position=1,
                    leave=False,
                ):
                    chunk = texts[i : i + CHUNK]
                    chunk_results = Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(_tokenise_row)(t, stop_words) for t in chunk
                    )
                    results.extend(chunk_results)

                df["Text Tokens"] = results
                df = df[df["Text Tokens"].apply(len) > 0].reset_index(drop=True)
                df = df[df["Sentiment"].isin(["positive", "negative"])].reset_index(drop=True)
                df["Sentiment"] = (
                    df["Sentiment"].map({"positive": 1, "negative": 0}).astype("int8")
                )
                df.to_parquet(CP_PREPROCESS, index=False, engine="pyarrow", compression="snappy")
                pbar.update(1)

            # vectorise
            if os.path.exists(CP_X_FEATURES) and os.path.exists(CP_TFIDF) and os.path.exists(CP_Y_LABELS):
                tqdm.write(f"[checkpoint] Skipping vectorising, reading {CP_X_FEATURES}")
                X = sp.load_npz(CP_X_FEATURES)
                tfidf = joblib.load(CP_TFIDF)
                y = pd.read_csv(CP_Y_LABELS).squeeze()
                pbar.update(1)
            else:
                pbar.set_description("Vectorising (TF-IDF + AFINN)")
                tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
                texts_for_tfidf = df["Text Tokens"].apply(" ".join)
                X_tfidf = tfidf.fit_transform(texts_for_tfidf)

                afinn_texts = df["Mutated Text Content"].fillna("").tolist()
                afinn_results = [
                    _afinn_row(t)
                    for t in tqdm(afinn_texts, desc="  AFINN rows", position=1, leave=False)
                ]
                afinn_scores = np.array(afinn_results, dtype=np.float32).reshape(-1, 1)
                X_afinn = sp.csr_matrix(afinn_scores)

                X = sp.hstack([X_tfidf, X_afinn], format="csr")
                del X_tfidf, X_afinn

                y = df["Sentiment"]
                joblib.dump(tfidf, CP_TFIDF)
                sp.save_npz(CP_X_FEATURES, X)
                y.to_csv(CP_Y_LABELS, index=False)
                pbar.update(1)

            # train
            if os.path.exists(CP_MODEL) and os.path.exists(CP_X_TEST) and os.path.exists(CP_Y_TEST):
                tqdm.write(f"[checkpoint] Skipping training, reading {CP_MODEL}")
                model = joblib.load(CP_MODEL)
                pbar.update(1)
            else:
                pbar.set_description("Training model")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                model = LogisticRegression(class_weight="balanced", solver="saga", verbose=1)
                model.fit(X_train, y_train)
                pbar.update(1)

                pbar.set_description("Saving files")
                sp.save_npz(CP_X_TEST, X_test)
                y_test.to_csv(CP_Y_TEST, index=False)
                joblib.dump(model, CP_MODEL)
                pbar.update(1)

        tqdm.write("Training complete.")
        return model, tfidf
    except Exception:
        import traceback
        tqdm.write(traceback.format_exc())
        raise