"""Build `data/processed/features.parquet` from the three raw parquets.

Usage:
    python -m src.data.build_dataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.concepts.feature_matrix import build, save, validate_schema
from src.config import settings
from src.logging_setup import configure_logging
from src.paths import DATA_RAW, FEATURES_PARQUET


def main() -> Path:
    """Assemble the feature matrix and persist to `data/processed/features.parquet`."""
    configure_logging()
    yields = pd.read_parquet(DATA_RAW / "nass_yields.parquet")
    weather = pd.read_parquet(DATA_RAW / "noaa_weather.parquet")
    soil = pd.read_parquet(DATA_RAW / "ssurgo_soil.parquet")

    state_fips_to_alpha = {v: k for k, v in settings.state_fips.items()}

    features = build(
        yields=yields,
        weather=weather,
        soil=soil,
        state_fips_to_alpha=state_fips_to_alpha,
        one_hot_states=True,
    )
    validate_schema(features)
    save(features, FEATURES_PARQUET)
    logger.info(f"Final feature matrix: {len(features)} rows × {len(features.columns)} cols")
    return FEATURES_PARQUET


if __name__ == "__main__":  # pragma: no cover
    main()
