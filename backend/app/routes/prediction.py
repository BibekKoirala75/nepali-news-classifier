import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import NewsPrediction
from app.schemas import PredictRequest, PredictResponse
from app.services.classifier import get_classifier

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    text = payload.text.strip()

    if len(text) < settings.min_text_length:
        raise HTTPException(
            status_code=422,
            detail=f"Text is too short. Minimum length is {settings.min_text_length} characters.",
        )
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=422,
            detail=f"Text is too long. Maximum length is {settings.max_text_length} characters.",
        )

    try:
        classifier = get_classifier()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    try:
        result = classifier.predict(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.") from exc

    try:
        record = NewsPrediction(
            text=text,
            predicted_category=result["category"],
            confidence=result["confidence"],
        )
        db.add(record)
        db.commit()
    except Exception:
        # A DB outage should not prevent the caller from getting their
        # prediction back -- log and continue rather than failing the request.
        logger.exception("Failed to persist prediction to history")
        db.rollback()

    return PredictResponse(
        category=result["category"],
        confidence=result["confidence"],
        available_categories=classifier.categories,
    )


@router.get("/categories")
def get_categories():
    try:
        classifier = get_classifier()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc
    return {"categories": classifier.categories}
