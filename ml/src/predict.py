"""
Prediction service: loads the trained TF-IDF vectorizer, classifier and
label encoder produced by train.py, and exposes a single `predict()`
function used both by this CLI and by the FastAPI backend.

Usage (CLI):
    python ml/src/predict.py "सरकारले नयाँ शिक्षा नीति सार्वजनिक गरेको छ ।"
"""

import os
import sys

import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from preprocessing import clean_text, is_valid_text  # noqa: E402

MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")


class NewsClassifier:
    """Loads model artifacts once and reuses them for every prediction.

    Raises FileNotFoundError with a clear message if artifacts are missing,
    and a dedicated ValueError for corrupted/incompatible artifacts, so
    callers (e.g. the FastAPI startup hook) can surface a useful error
    instead of a raw stack trace.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self._load()

    def _load(self):
        paths = {
            "vectorizer": os.path.join(self.models_dir, "vectorizer.joblib"),
            "model": os.path.join(self.models_dir, "model.joblib"),
            "label_encoder": os.path.join(self.models_dir, "label_encoder.joblib"),
        }
        missing = [name for name, p in paths.items() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing trained model artifact(s): "
                + ", ".join(missing)
                + f". Run `python ml/src/train.py` first to generate them (expected in {self.models_dir})."
            )
        try:
            self.vectorizer = joblib.load(paths["vectorizer"])
            self.model = joblib.load(paths["model"])
            self.label_encoder = joblib.load(paths["label_encoder"])
        except Exception as exc:
            raise ValueError(f"Failed to load model artifacts (possibly corrupted): {exc}") from exc

    @property
    def categories(self):
        return self.label_encoder.classes_.tolist()

    def predict(self, raw_text: str) -> dict:
        """Predict the category of a single raw (uncleaned) news text.

        Returns a dict: {"category": str, "confidence": float,
        "cleaned_text": str}. Raises ValueError if the text is empty or
        too short after cleaning to classify meaningfully.
        """
        cleaned = clean_text(raw_text)
        if not is_valid_text(cleaned):
            raise ValueError("Text is empty or too short after cleaning to classify.")

        X = self.vectorizer.transform([cleaned])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
        else:
            # Fallback for classifiers without predict_proba (e.g. a raw
            # LinearSVC): convert the decision function into a
            # softmax-normalized confidence score. This is explicitly NOT a
            # calibrated probability -- it is documented as such wherever
            # it is surfaced (API response, frontend).
            scores = self.model.decision_function(X)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

        category = self.label_encoder.inverse_transform([pred_idx])[0]
        return {
            "category": category,
            "confidence": round(confidence, 4),
            "cleaned_text": cleaned,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ml/src/predict.py \"<Nepali news text>\"")
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    clf = NewsClassifier()
    result = clf.predict(text)
    print(f"Predicted Category: {result['category']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")


if __name__ == "__main__":
    main()
