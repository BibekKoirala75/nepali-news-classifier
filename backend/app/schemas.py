"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw Nepali news article or headline text.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if v is None or not v.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return v


class PredictResponse(BaseModel):
    category: str
    confidence: float = Field(..., description="Model confidence in [0, 1] for the predicted category.")
    available_categories: list[str]


class HistoryItem(BaseModel):
    id: int
    text: str
    predicted_category: str
    confidence: float
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
