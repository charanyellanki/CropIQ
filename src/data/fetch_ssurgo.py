"""Fetch USDA SSURGO county-level soil aggregates → `data/raw/ssurgo_soil.parquet`.

Usage:
    python -m src.data.fetch_ssurgo
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.concepts.soil_profile import load_ssurgo
from src.config import settings
from src.logging_setup import configure_logging
from src.paths import DATA_RAW


def main() -> Path:
    """Run the SSURGO fetch CLI and write `data/raw/ssurgo_soil.parquet`."""
    configure_logging()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / "ssurgo_soil.parquet"

    df = load_ssurgo(
        state_alphas=settings.states,
        state_alpha_to_fips=settings.state_fips,
    )
    df.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    main()
