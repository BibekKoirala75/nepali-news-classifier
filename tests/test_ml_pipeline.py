"""
Unit tests for the ML preprocessing and prediction pipeline. These do not
require PostgreSQL or the API -- they exercise ml/src directly.

Run with (from the project root):
    pytest tests/test_ml_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml", "src"))

import pytest
from preprocessing import clean_text, is_valid_text, clean_batch


def test_clean_text_strips_urls():
    text = "यो समाचार हेर्नुहोस् https://example.com/news/123 धन्यवाद"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "example.com" not in cleaned


def test_clean_text_preserves_devanagari():
    text = "सरकारले नयाँ शिक्षा नीति सार्वजनिक गरेको छ ।"
    cleaned = clean_text(text)
    assert "शिक्षा" in cleaned
    assert "सरकार" in cleaned


def test_clean_text_removes_punctuation():
    cleaned = clean_text("नमस्कार!!! कस्तो *&^% हुनुहुन्छ???")
    assert "!" not in cleaned
    assert "*" not in cleaned
    assert "&" not in cleaned


def test_clean_text_handles_none_and_non_string():
    assert clean_text(None) == ""
    assert clean_text(12345) == "12345"


def test_clean_text_collapses_whitespace():
    cleaned = clean_text("नेपाल    धेरै\n\nसुन्दर    छ")
    assert "  " not in cleaned


def test_is_valid_text():
    assert is_valid_text("नेपाल राम्रो छ") is True
    assert is_valid_text("") is False
    assert is_valid_text("   ") is False
    assert is_valid_text("अ") is False  # below default min_chars


def test_clean_batch():
    texts = ["नेपाल!", None, "  भारत  "]
    result = clean_batch(texts)
    assert result[0] == "नेपाल"
    assert result[1] == ""
    assert result[2] == "भारत"


@pytest.fixture(scope="module")
def classifier():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml", "src"))
    from predict import NewsClassifier
    models_dir = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
    if not os.path.exists(os.path.join(models_dir, "model.joblib")):
        pytest.skip("Trained model artifacts not found -- run ml/src/train.py first.")
    return NewsClassifier(models_dir=models_dir)


def test_model_loads(classifier):
    assert len(classifier.categories) > 0


def test_predict_returns_valid_category(classifier):
    result = classifier.predict("नेपाली क्रिकेट टिमले आजको खेलमा जित हासिल गर्यो")
    assert result["category"] in classifier.categories
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_rejects_empty_text(classifier):
    with pytest.raises(ValueError):
        classifier.predict("")


def test_predict_rejects_whitespace_only(classifier):
    with pytest.raises(ValueError):
        classifier.predict("   ...   ")
