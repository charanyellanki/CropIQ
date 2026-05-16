"""
Concept: PredictionService (local-only HTTP adapter)
Purpose: Expose `PredictionEngine` over HTTP for local development and contract testing.
State: Reference to PredictionEngine loaded at startup via FastAPI lifespan.
Actions: GET /health, GET /counties/{state}, GET /predict/{state}/{county_fips}/{year}.
Operational principle: GET /predict/IA/19169/2024 calls PredictionEngine.predict(...)
    and returns JSON.

This service is NOT deployed to HF Spaces (CLAUDE.md §1.10 / §12.2). It exists for:
- Local development.
- Demonstrating API contract design.
- Reusability in a hypothetical microservices deployment.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_engine, lifespan
from src.api.schemas import County, ModelMetadata, PredictionResult
from src.inference.prediction_engine import PredictionEngine

app = FastAPI(
    title="CropIQ Prediction Service",
    description="Local-only HTTP adapter over PredictionEngine.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:*"],
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=dict, tags=["meta"])
def health(engine: PredictionEngine = Depends(get_engine)) -> dict:
    """Return service status and model metadata."""
    metadata = engine.get_metadata()
    return {"status": "ok", **metadata.model_dump()}


@app.get("/counties/{state}", response_model=list[County], tags=["lookup"])
def list_counties(state: str, engine: PredictionEngine = Depends(get_engine)) -> list[County]:
    """List all counties available for the given two-letter state code."""
    try:
        return engine.list_counties(state)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.get(
    "/predict/{state}/{county_fips}/{year}",
    response_model=PredictionResult,
    tags=["inference"],
)
def predict(
    state: str,
    county_fips: str,
    year: int,
    engine: PredictionEngine = Depends(get_engine),
) -> PredictionResult:
    """Predict yield for (state, county_fips, year)."""
    try:
        return engine.predict(state, county_fips, year)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )


@app.get("/", tags=["meta"])
def root() -> dict:
    """Friendly index linking to docs."""
    return {
        "name": "CropIQ Prediction Service",
        "docs": "/docs",
        "endpoints": ["/health", "/counties/{state}", "/predict/{state}/{county_fips}/{year}"],
    }
