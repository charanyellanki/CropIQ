"""FastAPI lifespan hook + dependency providers.

Loads `PredictionEngine` ONCE at startup (gotcha #8) and exposes it via
`app.state.engine`. The dependency `get_engine` reads from app.state.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from loguru import logger

from src.config import settings
from src.inference.prediction_engine import PredictionEngine
from src.logging_setup import configure_logging
from src.paths import BEST_MODEL_PATH, FEATURES_PARQUET


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize PredictionEngine at startup and clean up at shutdown."""
    configure_logging()
    logger.info("FastAPI startup: loading PredictionEngine…")
    app.state.engine = PredictionEngine(BEST_MODEL_PATH, FEATURES_PARQUET)
    app.state.settings = settings
    try:
        yield
    finally:
        logger.info("FastAPI shutdown.")
        app.state.engine = None


def get_engine(request: Request) -> PredictionEngine:
    """FastAPI dependency: return the cached `PredictionEngine` from app.state."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise RuntimeError("PredictionEngine not initialized — lifespan hook missing.")
    return engine
