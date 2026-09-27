"""
FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000
(from inside the backend/ directory, with the virtualenv active)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.routes import health, history, prediction
from app.services.classifier import load_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception:
        logger.exception(
            "Database initialization failed. History endpoints will not work "
            "until PostgreSQL is reachable and DATABASE_URL is correct."
        )
    load_classifier()
    yield


app = FastAPI(
    title="Nepali News Classifier API",
    description="Predicts the category of a Nepali news article using a TF-IDF + ML pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(history.router)


@app.get("/")
def root():
    return {
        "service": "Nepali News Classifier API",
        "docs": "/docs",
        "health": "/health",
    }
