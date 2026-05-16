"""FastAPI contract tests using `TestClient`."""

from __future__ import annotations

import pytest

from src.paths import BEST_MODEL_PATH, FEATURES_PARQUET


@pytest.mark.skipif(
    not BEST_MODEL_PATH.exists() or not FEATURES_PARQUET.exists(),
    reason="Requires trained model + features.",
)
def test_health_returns_ok_and_metadata() -> None:
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "model_name" in body
        assert "residual_std" in body


@pytest.mark.skipif(
    not BEST_MODEL_PATH.exists() or not FEATURES_PARQUET.exists(),
    reason="Requires trained model + features.",
)
def test_predict_endpoint_returns_valid_result() -> None:
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        r = client.get("/predict/IA/19169/2024")
        assert r.status_code == 200, r.text
        body = r.json()
        assert 50 < body["predicted_yield_bu_per_acre"] < 300
        assert len(body["top_drivers"]) == 3
        assert body["county_fips"] == "19169"


@pytest.mark.skipif(
    not BEST_MODEL_PATH.exists() or not FEATURES_PARQUET.exists(),
    reason="Requires trained model + features.",
)
def test_predict_404_on_unknown_county() -> None:
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        r = client.get("/predict/IA/99999/2024")
        assert r.status_code == 404


@pytest.mark.skipif(
    not BEST_MODEL_PATH.exists() or not FEATURES_PARQUET.exists(),
    reason="Requires trained model + features.",
)
def test_counties_endpoint() -> None:
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        r = client.get("/counties/IA")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 50
