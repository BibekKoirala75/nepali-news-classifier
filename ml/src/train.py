"""
Training pipeline for the Nepali News Classifier.

Loads the consolidated CSV produced by ml/data/prepare_dataset.py, cleans
the text with ml.src.preprocessing.clean_text, splits it into a stratified
train/test set, fits a TF-IDF vectorizer, trains several candidate linear
models, evaluates each on the held-out test set, and saves the
best-performing model + vectorizer + label encoder to ml/models/.

Usage:
    python ml/src/train.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # allow `import preprocessing`, `import features`

from preprocessing import clean_text  # noqa: E402
from features import build_vectorizer  # noqa: E402

DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "16nepalinews.csv")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
RANDOM_SEED = 42
TEST_SIZE = 0.2
MIN_CLASS_COUNT = 2  # classes with fewer examples than this cannot be stratified
MIN_TEXT_CHARS = 20  # drop near-empty documents after cleaning

# Averaging method used for multiclass precision/recall/F1. "macro" gives
# every category equal weight regardless of how many articles it has, which
# matters here because the corpus is heavily imbalanced (National News has
# 7,452 articles vs. 16 for Literature) -- accuracy alone would hide poor
# performance on the small classes.
AVERAGING = "macro"


def load_and_clean_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python ml/data/prepare_dataset.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} raw records.")

    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() >= MIN_TEXT_CHARS]
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"Dropped {before - len(df)} exact duplicate articles after cleaning.")

    # Drop categories too small to be represented in both a stratified train
    # and test split (this dataset has none below 2, but this guards future
    # dataset swaps).
    counts = df["category"].value_counts()
    rare = counts[counts < MIN_CLASS_COUNT].index.tolist()
    if rare:
        print(f"Dropping categories with < {MIN_CLASS_COUNT} examples: {rare}")
        df = df[~df["category"].isin(rare)]

    print(f"Final cleaned dataset: {len(df)} records, {df['category'].nunique()} categories.")
    return df.reset_index(drop=True)


def evaluate_model(name, pipeline, X_test, y_test, label_encoder):
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=AVERAGING, zero_division=0)
    recall = recall_score(y_test, y_pred, average=AVERAGING, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=AVERAGING, zero_division=0)

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision ({AVERAGING}): {precision:.4f}")
    print(f"Recall ({AVERAGING}):    {recall:.4f}")
    print(f"F1 ({AVERAGING}):        {f1:.4f}")

    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )

    return {
        "name": name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": report,
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_and_clean_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    X = df["text"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_SEED),
            method="sigmoid",
            cv=3,
        ),
        "naive_bayes": MultinomialNB(),
    }

    results = []
    fitted_pipelines = {}

    for name, clf in candidates.items():
        vectorizer = build_vectorizer()
        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])
        start = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"\nTrained {name} in {elapsed:.1f}s")

        metrics = evaluate_model(name, pipeline, X_test, y_test, label_encoder)
        metrics["train_seconds"] = round(elapsed, 2)
        results.append(metrics)
        fitted_pipelines[name] = pipeline

    # Select the model with the highest macro F1 -- appropriate here because
    # the classes are heavily imbalanced and macro F1 does not let a model
    # win purely by being good at the majority "National News" class.
    best = max(results, key=lambda r: r["f1_macro"])
    best_name = best["name"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n{'=' * 60}\nSelected model: {best_name} (macro F1 = {best['f1_macro']:.4f})\n{'=' * 60}")

    # Persist artifacts
    joblib.dump(best_pipeline.named_steps["tfidf"], os.path.join(MODELS_DIR, "vectorizer.joblib"))
    joblib.dump(best_pipeline.named_steps["clf"], os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

    cm = confusion_matrix(y_test, best_pipeline.predict(X_test))
    metadata = {
        "selected_model": best_name,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "averaging_method": AVERAGING,
        "categories": label_encoder.classes_.tolist(),
        "train_examples": len(X_train),
        "test_examples": len(X_test),
        "model_comparison": [
            {k: v for k, v in r.items() if k != "classification_report"}
            for r in results
        ],
        "selected_model_report": best["classification_report"],
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(MODELS_DIR, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nSaved model artifacts and training_metadata.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
