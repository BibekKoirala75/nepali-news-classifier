"""
Nepali text preprocessing.

This module is imported by BOTH the training pipeline (ml/src/train.py) and
the FastAPI prediction service (backend/app/services/preprocessing.py) so
that the exact same cleaning logic is applied at training time and at
inference time. Never duplicate this logic elsewhere.
"""

import re
import unicodedata

# Devanagari Unicode block used by Nepali script: U+0900 - U+097F
_DEVANAGARI_RANGE = r"\u0900-\u097F"

# Matches http(s):// and www. URLs.
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")

# Matches one or more whitespace characters (spaces, tabs, newlines).
_WHITESPACE_RE = re.compile(r"\s+")

# Keep: Devanagari script, basic Latin letters, digits (both Devanagari and
# Latin), and spaces. Drop everything else (punctuation, symbols, emoji,
# stray control characters) since it carries no discriminative signal for
# a bag-of-words / TF-IDF classifier and can otherwise dominate n-gram
# features on noisy scraped text.
_KEEP_CHARS_RE = re.compile(
    rf"[^{_DEVANAGARI_RANGE}a-zA-Z0-9\s]"
)


def clean_text(text) -> str:
    """Clean a single raw text string for TF-IDF feature extraction.

    Steps (in order):
      1. Handle missing/non-string input -> empty string.
      2. Unicode-normalize (NFC) so visually identical Devanagari glyphs
         formed from different combining-character sequences compare equal.
      3. Strip URLs.
      4. Drop punctuation/symbols while explicitly preserving Devanagari
         and Latin alphanumerics.
      5. Collapse repeated whitespace and trim.

    This function must remain side-effect-free and deterministic so that
    training and API inference stay in sync.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub(" ", text)
    text = _KEEP_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_valid_text(text: str, min_chars: int = 3) -> bool:
    """Return True if the cleaned text has enough content to classify."""
    return isinstance(text, str) and len(text.strip()) >= min_chars


def clean_batch(texts):
    """Vectorized helper: clean an iterable of raw texts."""
    return [clean_text(t) for t in texts]
