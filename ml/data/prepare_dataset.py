"""
Dataset preparation script for the Nepali News Classifier.

This script downloads the "16NepaliNews" corpus (sndsabin/Nepali-News-Classifier,
GitHub, licensed GPLv3), extracts it, and consolidates the raw .txt documents
(one file per article, grouped into folders by category) into a single CSV file
at ml/data/raw/16nepalinews.csv with columns: text, category, split.

The corpus is NOT committed to this repository (see .gitignore) because it is
~40 MB compressed / ~75 MB as CSV. Run this script once before training.

Usage:
    python ml/data/prepare_dataset.py
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile

REPO_URL = "https://github.com/sndsabin/Nepali-News-Classifier.git"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
OUTPUT_CSV = os.path.join(RAW_DIR, "16nepalinews.csv")


def _run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def download_and_extract(tmp_dir: str) -> str:
    """Clone the corpus repo and extract the .rar archive. Returns the path
    to the extracted '16719' directory containing train/test/raw splits."""
    repo_dir = os.path.join(tmp_dir, "Nepali-News-Classifier")
    _run(["git", "clone", "--depth", "1", REPO_URL, repo_dir])

    rar_path = os.path.join(repo_dir, "16NepaliNews.rar")
    if not os.path.exists(rar_path):
        raise FileNotFoundError(f"Expected archive not found at {rar_path}")

    if shutil.which("unrar") is None:
        raise EnvironmentError(
            "The 'unrar' utility is required to extract the dataset. "
            "Install it with: sudo apt-get install unrar"
        )

    _run(["unrar", "x", "-y", rar_path], cwd=repo_dir)

    extracted_root = os.path.join(repo_dir, "16NepaliNews")
    # There is a single numbered subdirectory (e.g. '16719') under this root.
    subdirs = [d for d in os.listdir(extracted_root)
               if os.path.isdir(os.path.join(extracted_root, d))]
    if not subdirs:
        raise FileNotFoundError("Could not locate extracted dataset directory")
    return os.path.join(extracted_root, subdirs[0])


def consolidate_to_csv(dataset_dir: str, output_csv: str) -> int:
    """Walk the train/ and test/ split folders and write a single CSV with
    columns: text, category, split. Returns the number of rows written."""
    rows = []
    for split in ("train", "test"):
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for category in sorted(os.listdir(split_dir)):
            cat_dir = os.path.join(split_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                fpath = os.path.join(cat_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                except OSError:
                    continue
                if not text:
                    continue
                rows.append({"text": text, "category": category, "split": split})

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"Dataset already exists at {OUTPUT_CSV}. Delete it to re-download.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_dir = download_and_extract(tmp_dir)
        count = consolidate_to_csv(dataset_dir, OUTPUT_CSV)

    print(f"Wrote {count} articles to {OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
