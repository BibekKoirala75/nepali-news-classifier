"""
Loads the trained model once at application startup and exposes it as a
module-level singleton so every request reuses the same in-memory model
instead of re-loading/re-training it per request.
"""

import logging
import os
import sys

from app.config import get_settings

logger = logging.getLogger("uvicorn.error")

_ML_SRC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "ml", "src"
)
_ML_SRC_DIR = os.path.normpath(_ML_SRC_DIR)
if _ML_SRC_DIR not in sys.path:
    sys.path.insert(0, _ML_SRC_DIR)

from predict import NewsClassifier  # noqa: E402

_classifier: NewsClassifier | None = None
_load_error: str | None = None


def load_classifier() -> None:
    """Called once from the FastAPI startup event."""
    global _classifier, _load_error
    settings = get_settings()
    try:
        _classifier = NewsClassifier(models_dir=settings.ml_models_dir)
        logger.info(
            "Loaded news classifier with %d categories from %s",
            len(_classifier.categories), settings.ml_models_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        # Do not crash the whole app on startup: /health should still report
        # the problem clearly instead of the process failing to boot with a
        # cryptic traceback.
        _classifier = None
        _load_error = str(exc)
        logger.error("Failed to load model artifacts: %s", exc)


def get_classifier() -> NewsClassifier:
    if _classifier is None:
        detail = _load_error or "Model is not loaded."
        raise RuntimeError(detail)
    return _classifier


def classifier_status() -> dict:
    if _classifier is not None:
        return {"loaded": True, "categories": _classifier.categories}
    return {"loaded": False, "error": _load_error}
