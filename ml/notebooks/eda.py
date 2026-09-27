"""
Exploratory Data Analysis for the 16NepaliNews corpus.

Run after ml/data/prepare_dataset.py has produced ml/data/raw/16nepalinews.csv.
Prints a text summary to stdout and writes charts (PNG) to ml/notebooks/figures/.

Usage:
    python ml/notebooks/eda.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw", "16nepalinews.csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")

sns.set_theme(style="whitegrid")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python ml/data/prepare_dataset.py` first."
        )
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total records:      {len(df)}")
    print(f"Categories:         {df['category'].nunique()}")
    print(f"Missing text:       {df['text'].isnull().sum()}")
    print(f"Missing category:   {df['category'].isnull().sum()}")
    print(f"Duplicate articles: {df.duplicated(subset=['text']).sum()}")
    print()

    print("Category distribution:")
    print(df["category"].value_counts())
    print()

    df["char_len"] = df["text"].astype(str).str.len()
    print("Article length (characters) summary:")
    print(df["char_len"].describe())
    print()

    # --- Chart 1: category distribution ---
    plt.figure(figsize=(10, 6))
    order = df["category"].value_counts().index
    sns.countplot(data=df, y="category", order=order, color="#2563eb")
    plt.title("Article Count by Category (16NepaliNews)")
    plt.xlabel("Number of Articles")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "category_distribution.png"), dpi=150)
    plt.close()

    # --- Chart 2: article length distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df["char_len"], bins=60, color="#16a34a")
    plt.xlim(0, df["char_len"].quantile(0.99))
    plt.title("Article Length Distribution (characters, 99th percentile clipped)")
    plt.xlabel("Character count")
    plt.ylabel("Number of articles")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "length_distribution.png"), dpi=150)
    plt.close()

    # --- Chart 3: train/test split composition ---
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="split", color="#f59e0b")
    plt.title("Predefined Train/Test Split Sizes")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "split_distribution.png"), dpi=150)
    plt.close()

    print(f"Charts written to {FIG_DIR}")


if __name__ == "__main__":
    main()
