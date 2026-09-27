from fastapi import APIRouter

from app.schemas import HealthResponse
from app.services.classifier import classifier_status

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@router.get("/health/model")
def model_health():
    """Extra diagnostic endpoint: whether the ML model actually loaded."""
    return classifier_status()
