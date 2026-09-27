"""
TF-IDF feature engineering for the Nepali News Classifier.

Word-level TF-IDF with (1,2) n-grams is used as the primary representation.
Character n-grams were evaluated during model selection (see README /
docs/model_card.md for the comparison) but did not justify the extra
memory and inference latency on this dataset, so they are not used by
default. The function below is kept generic so a different ngram/analyzer
setting can be swapped in without touching the training pipeline.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(
    max_features: int = 40000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
) -> TfidfVectorizer:
    """Construct the TF-IDF vectorizer used for both training and inference.

    Note: `preprocessor`/`tokenizer` are intentionally left as scikit-learn
    defaults (whitespace + regex token pattern) because the text handed to
    this vectorizer has already been through
    ml.src.preprocessing.clean_text, which normalizes Unicode and strips
    punctuation/URLs. Keeping cleaning and vectorization as separate steps
    means the exact same cleaning function can be reused unmodified by the
    FastAPI inference service.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
