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
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console
from sentence_transformers import SentenceTransformer
import gc

console = Console()

n_jobs = 2

_PUNCT = re.compile(r"[^\w\s]")
_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# change me for a higher sample rate from source 2 (hugging face)
SOURCE2_SAMPLE = 500_000

TFIDF_CHUNK = 50_000
SBERT_BATCH = 256


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


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def train(csv_folder: str = None):
    path = csv_folder or "csv"
    if os.path.basename(os.path.normpath(path)) != "csv":
        path = os.path.join(path, "csv")
    os.makedirs(path, exist_ok=True)

    CP_ALL_DATA = f"{path}/all_data.parquet"
    CP_PREPROCESS = f"{path}/preprocess.parquet"
    CP_TFIDF = f"{path}/tfidf_dump.pkl"
    CP_SBERT = f"{path}/sbert_embeddings.npz"
    CP_X_FEATURES = f"{path}/x_features.npz"
    CP_Y_LABELS = f"{path}/y_sentiment_labels.csv"
    CP_X_TEST = f"{path}/X_test.npz"
    CP_Y_TEST = f"{path}/y_test.csv"
    CP_MODEL = f"{path}/sentiment_model.pkl"

    stages = [
        "Loading datasets",
        "Cleaning data",
        "Preprocessing text",
        "Tokenising",
        "Vectorising (TF-IDF + SBERT)",
        "Training model",
        "Saving files",
    ]

    try:
        with _make_progress() as progress:
            pipeline = progress.add_task("[bold green]Training pipeline", total=len(stages))

            if os.path.exists(CP_ALL_DATA):
                console.log(f"[checkpoint] Skipping load+clean, reading {CP_ALL_DATA}")
                df = pd.read_parquet(CP_ALL_DATA)
                progress.advance(pipeline, 2)
            else:
                progress.update(pipeline, description="Loading datasets")
                croissant_dataset = mlc.Dataset(
                    "https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/croissant/download"
                )
                record_sets = croissant_dataset.metadata.record_sets
                raw_records = croissant_dataset.records(record_set=record_sets[0].uuid)
                kaggle_task = progress.add_task("  Loading Kaggle records", total=None)
                records = []
                for r in raw_records:
                    records.append(r)
                    progress.advance(kaggle_task)
                progress.remove_task(kaggle_task)
                df1 = pd.DataFrame(records)
                del records
                console.log(f"  Source 1 (Kaggle) raw rows: {len(df1):,}")

                ds = load_dataset(
                    "NNEngine/Sentiment-Analysis-Complex",
                    split=f"train[:{SOURCE2_SAMPLE}]",
                )
                df2 = ds.to_pandas()
                del ds
                console.log(f"  Source 2 (HuggingFace) loaded rows: {len(df2):,}")
                progress.advance(pipeline)

                progress.update(pipeline, description="Cleaning data")

                df = df1.drop(
                    columns=[
                        "twitter_training.csv/2401",
                        "twitter_training.csv/Borderlands",
                    ]
                )
                df = df.drop(index=0).reset_index(drop=True)
                df.columns = ["Sentiment", "Text Content"]
                for col in ["Sentiment", "Text Content"]:
                    df[col] = df[col].apply(
                        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                    )
                del df1

                df2 = df2.rename(columns={"label": "Sentiment", "text": "Text Content"})
                df2["Sentiment"] = df2["Sentiment"].astype(str).str.lower()
                df2 = df2.drop(columns=["id"])

                df = pd.concat([df, df2], ignore_index=True)
                del df2
                gc.collect()

                console.log(f"  Combined rows before dedup: {len(df):,}")
                df["Sentiment"] = df["Sentiment"].str.lower()

                combined_cleaning_steps = [
                    ("Stripping <unk>", lambda s: s.str.replace("<unk>", "", regex=False)),
                    ("Collapsing whitespace", lambda s: s.str.replace(r"\s+", " ", regex=True)),
                    ("Stripping edges", lambda s: s.str.strip()),
                ]
                clean_task = progress.add_task("  Cleaning combined text", total=len(combined_cleaning_steps))
                col = df["Text Content"]
                for desc, fn in combined_cleaning_steps:
                    progress.update(clean_task, description=f"  {desc}")
                    col = fn(col)
                    progress.advance(clean_task)
                progress.remove_task(clean_task)
                df["Text Content"] = col
                del col

                df = (
                    df[df["Text Content"].str.len() > 0]
                    .drop_duplicates(subset="Text Content")
                    .sample(frac=1)
                    .reset_index(drop=True)
                )
                console.log(f"  Combined rows after dedup+shuffle: {len(df):,}")
                df.to_parquet(CP_ALL_DATA, index=False, engine="pyarrow", compression="snappy")
                progress.advance(pipeline)
                gc.collect()

            if os.path.exists(CP_PREPROCESS):
                console.log(f"[checkpoint] Skipping preprocess+tokenise, reading {CP_PREPROCESS}")
                df = pd.read_parquet(CP_PREPROCESS)
                df["Text Tokens"] = df["Text Tokens"].apply(
                    lambda x: x if isinstance(x, list) else list(x)
                )
                progress.advance(pipeline, 2)
            else:
                progress.update(pipeline, description="Preprocessing text")
                nltk_resources = [
                    "stopwords",
                    "twitter_samples",
                    "wordnet",
                    "averaged_perceptron_tagger_eng",
                    "punkt_tab",
                ]
                nltk_task = progress.add_task("  Downloading NLTK data", total=len(nltk_resources))
                for resource in nltk_resources:
                    nltk.download(resource, quiet=True)
                    progress.advance(nltk_task)
                progress.remove_task(nltk_task)

                stop_words = set(stopwords.words("english")) - {
                    "no", "not", "nor", "neither", "never", "none",
                    "don't", "won't", "can't", "isn't", "aren't", "wasn't",
                }

                cleaning_steps = [
                    ("Removing non-ASCII", lambda s: s.str.replace(r"[^\x00-\x7F]+", "", regex=True)),
                    ("Removing URLs", lambda s: s.str.replace(r"http\S+|www\.\S+", "", regex=True)),
                    ("Removing hashtags", lambda s: s.str.replace(r"#\w+", "", regex=True)),
                    ("Removing HTML entities", lambda s: s.str.replace(r"&\w+;", "", regex=True)),
                    ("Expanding contractions", lambda s: s.apply(contractions.fix)),
                    ("Lowercasing", lambda s: s.str.lower().str.strip()),
                ]
                text_task = progress.add_task("  Cleaning text", total=len(cleaning_steps))
                col = df["Text Content"].fillna("")
                for desc, fn in cleaning_steps:
                    progress.update(text_task, description=f"  {desc}")
                    col = fn(col)
                    progress.advance(text_task)
                progress.remove_task(text_task)
                df["Mutated Text Content"] = col
                df = df.drop(columns=["Text Content"]).reset_index(drop=True)
                del col
                gc.collect()
                progress.advance(pipeline)

                progress.update(pipeline, description="Tokenising")
                CHUNK = 50_000
                texts = df["Mutated Text Content"].tolist()
                results = []
                chunk_task = progress.add_task("  Tokenising chunks", total=len(texts))
                for i in range(0, len(texts), CHUNK):
                    chunk = texts[i : i + CHUNK]
                    row_task = progress.add_task("    Rows in chunk", total=len(chunk))
                    batch = Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(_tokenise_row)(t, stop_words) for t in chunk
                    )
                    results.extend(batch)
                    progress.advance(chunk_task, len(chunk))
                    progress.remove_task(row_task)
                progress.remove_task(chunk_task)
                del texts
                gc.collect()

                df["Text Tokens"] = results
                del results

                df = (
                    df[df["Text Tokens"].apply(len) > 0]
                    .loc[df["Sentiment"].isin(["positive", "negative"])]
                    .reset_index(drop=True)
                )
                df["Sentiment"] = (
                    df["Sentiment"].map({"positive": 1, "negative": 0}).astype("int8")
                )
                df.to_parquet(CP_PREPROCESS, index=False, engine="pyarrow", compression="snappy")
                progress.advance(pipeline)
                gc.collect()

            if (
                os.path.exists(CP_X_FEATURES)
                and os.path.exists(CP_TFIDF)
                and os.path.exists(CP_Y_LABELS)
            ):
                console.log(f"[checkpoint] Skipping vectorising, reading {CP_X_FEATURES}")
                X = sp.load_npz(CP_X_FEATURES)
                tfidf = joblib.load(CP_TFIDF)
                y = pd.read_csv(CP_Y_LABELS).squeeze()
                progress.advance(pipeline)
            else:
                progress.update(pipeline, description="Vectorising (TF-IDF + SBERT)")

                texts_for_tfidf = df["Text Tokens"].apply(" ".join).tolist()

                fit_task = progress.add_task("  TF-IDF fitting vocab", total=None)
                tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
                tfidf.fit(texts_for_tfidf)
                progress.remove_task(fit_task)

                tfidf_chunks = []
                tfidf_task = progress.add_task("  TF-IDF transform", total=len(texts_for_tfidf))
                for i in range(0, len(texts_for_tfidf), TFIDF_CHUNK):
                    chunk = texts_for_tfidf[i : i + TFIDF_CHUNK]
                    tfidf_chunks.append(tfidf.transform(chunk))
                    progress.advance(tfidf_task, len(chunk))
                progress.remove_task(tfidf_task)
                X_tfidf = sp.vstack(tfidf_chunks, format="csr")
                del tfidf_chunks, texts_for_tfidf
                gc.collect()
                console.log(f"  TF-IDF shape: {X_tfidf.shape}")

                raw_texts = df["Mutated Text Content"].fillna("").tolist()
                del df
                gc.collect()

                if os.path.exists(CP_SBERT):
                    console.log(f"[checkpoint] Loading SBERT embeddings from {CP_SBERT}")
                    X_sbert = sp.load_npz(CP_SBERT)
                else:
                    console.log("  Computing SBERT embeddings (this may take a while)...")
                    sbert_task = progress.add_task("  SBERT encoding", total=len(raw_texts))
                    sbert_vecs = []
                    for i in range(0, len(raw_texts), SBERT_BATCH):
                        batch = raw_texts[i : i + SBERT_BATCH]
                        sbert_vecs.append(
                            _sbert.encode(
                                batch,
                                batch_size=SBERT_BATCH,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                            )
                        )
                        progress.advance(sbert_task, len(batch))
                    progress.remove_task(sbert_task)
                    sbert_vecs = np.vstack(sbert_vecs)
                    X_sbert = sp.csr_matrix(sbert_vecs[:, :128].astype(np.float32))
                    del sbert_vecs
                    gc.collect()
                    console.log(f"  SBERT shape: {X_sbert.shape}")
                del raw_texts
                gc.collect()

                X = sp.hstack([X_tfidf, X_sbert], format="csr")
                console.log(f"  Combined feature shape: {X.shape}")
                del X_tfidf, X_sbert
                gc.collect()

                y = pd.read_parquet(CP_PREPROCESS, columns=["Sentiment"]).squeeze()
                joblib.dump(tfidf, CP_TFIDF)
                sp.save_npz(CP_X_FEATURES, X)
                y.to_csv(CP_Y_LABELS, index=False)
                progress.advance(pipeline)

            if (
                os.path.exists(CP_MODEL)
                and os.path.exists(CP_X_TEST)
                and os.path.exists(CP_Y_TEST)
            ):
                console.log(f"[checkpoint] Skipping training, reading {CP_MODEL}")
                model = joblib.load(CP_MODEL)
                progress.advance(pipeline, 2)
            else:
                progress.update(pipeline, description="Training model")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                del X, y
                gc.collect()

                model = LogisticRegression(
                    class_weight="balanced",
                    solver="saga",
                    max_iter=1000,
                    verbose=0,
                )
                fit_task = progress.add_task("  Fitting logistic regression", total=None)
                model.fit(X_train, y_train)
                progress.remove_task(fit_task)
                del X_train, y_train
                gc.collect()
                progress.advance(pipeline)

                progress.update(pipeline, description="Saving files")
                saves = [
                    ("X_test.npz", lambda: sp.save_npz(CP_X_TEST, X_test)),
                    ("y_test.csv", lambda: y_test.to_csv(CP_Y_TEST, index=False)),
                    ("sentiment_model.pkl", lambda: joblib.dump(model, CP_MODEL)),
                ]
                save_task = progress.add_task("  Saving artefacts", total=len(saves))
                for name, fn in saves:
                    progress.update(save_task, description=f"  Saving {name}")
                    fn()
                    progress.advance(save_task)
                progress.remove_task(save_task)
                progress.advance(pipeline)

        console.log("[bold green]Training complete.")
        return model, tfidf

    except Exception:
        console.print_exception()
        raise