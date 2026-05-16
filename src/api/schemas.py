"""Pydantic v2 schemas for the FastAPI service.

The on-the-wire response types are the same dataclasses produced by `PredictionEngine`
(re-exported here so OpenAPI surfaces them on the FastAPI side).
"""

from __future__ import annotations

from src.inference.prediction_engine import County, Driver, ModelMetadata, PredictionResult

__all__ = ["County", "Driver", "ModelMetadata", "PredictionResult"]
