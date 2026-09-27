"""
Re-evaluate the currently saved model artifacts against a fresh stratified
test split (same random seed as train.py, so this reproduces the same
split) and render a confusion matrix heatmap.

Usage:
    python ml/src/evaluate.py
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from preprocessing import clean_text  # noqa: E402
from train import DATA_PATH, MODELS_DIR, RANDOM_SEED, TEST_SIZE, MIN_TEXT_CHARS  # noqa: E402

FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "notebooks", "figures")


def main():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
    model = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "category"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() >= MIN_TEXT_CHARS]
    df = df.drop_duplicates(subset=["text"])
    df = df[df["category"].isin(label_encoder.classes_)]

    y = label_encoder.transform(df["category"])
    X = df["text"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
