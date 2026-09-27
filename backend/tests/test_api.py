"""
API tests. Uses an isolated SQLite database file (never the production
PostgreSQL DB) so tests are self-contained and safe to run anywhere,
including CI.

Run with (from backend/):
    pytest
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_history.db")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    db_file = "./test_history.db"
    if os.path.exists(db_file):
        os.remove(db_file)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_loaded(client):
    resp = client.get("/health/model")
    assert resp.status_code == 200
    body = resp.json()
    assert body["loaded"] is True
    assert isinstance(body["categories"], list)
    assert len(body["categories"]) > 0


def test_predict_valid_text(client):
    resp = client.post(
        "/api/predict",
        json={"text": "नेपाली क्रिकेट टिमले आजको खेलमा जित हासिल गर्यो"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "category" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["available_categories"], list)


def test_predict_empty_text_returns_422(client):
    resp = client.post("/api/predict", json={"text": "   "})
    assert resp.status_code == 422


def test_predict_missing_field_returns_422(client):
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 422


def test_predict_too_long_returns_422(client):
    resp = client.post("/api/predict", json={"text": "क " * 15000})
    assert resp.status_code == 422


def test_history_after_predict(client):
    client.post("/api/predict", json={"text": "शिक्षा मन्त्रालयले नयाँ नीति ल्यायो"})
    resp = client.get("/api/history")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) >= 1
    assert "predicted_category" in items[0]


def test_delete_history_item(client):
    resp = client.post("/api/predict", json={"text": "अर्थतन्त्रमा सुधार आएको छ"})
    history = client.get("/api/history").json()
    target_id = history[0]["id"]

    del_resp = client.delete(f"/api/history/{target_id}")
    assert del_resp.status_code == 204

    del_again = client.delete(f"/api/history/{target_id}")
    assert del_again.status_code == 404


def test_categories_endpoint(client):
    resp = client.get("/api/categories")
    assert resp.status_code == 200
    assert "categories" in resp.json()
