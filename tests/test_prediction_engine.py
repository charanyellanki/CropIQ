"""End-to-end test of `PredictionEngine` using on-disk model + features."""

from __future__ import annotations

import pytest

from src.paths import BEST_MODEL_PATH, FEATURES_PARQUET


@pytest.mark.skipif(
    not BEST_MODEL_PATH.exists() or not FEATURES_PARQUET.exists(),
    reason="Requires trained model + features (Phase 4/5 not yet run).",
)
def test_engine_predicts_for_story_county_2024() -> None:
    from src.inference.prediction_engine import PredictionEngine

    engine = PredictionEngine(BEST_MODEL_PATH, FEATURES_PARQUET)
    result = engine.predict("IA", "19169", 2024)
    assert 50 < result.predicted_yield_bu_per_acre < 300
    assert len(result.top_drivers) == 3
    lo, hi = result.confidence_interval_80pct
    assert lo < result.predicted_yield_bu_per_acre < hi


@pytest.mark.skipif(
    not BEST_MODEL_PATH.exists() or not FEATURES_PARQUET.exists(),
    reason="Requires trained model + features.",
)
def test_engine_metadata_shape() -> None:
    from src.inference.prediction_engine import PredictionEngine

    engine = PredictionEngine(BEST_MODEL_PATH, FEATURES_PARQUET)
    meta = engine.get_metadata()
    assert meta.feature_count > 5
    assert meta.residual_std > 0


@pytest.mark.skipif(
    not BEST_MODEL_PATH.exists() or not FEATURES_PARQUET.exists(),
    reason="Requires trained model + features.",
)
def test_engine_lists_counties_for_iowa() -> None:
    from src.inference.prediction_engine import PredictionEngine

    engine = PredictionEngine(BEST_MODEL_PATH, FEATURES_PARQUET)
    counties = engine.list_counties("IA")
    assert len(counties) > 50  # Iowa has 99 counties
    assert all(c.county_fips.startswith("19") for c in counties)
