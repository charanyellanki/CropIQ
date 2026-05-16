"""Fetch USDA NASS Quick Stats corn yields → `data/raw/nass_yields.parquet`.

Usage:
    python -m src.data.fetch_nass
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.concepts.yield_observation import fetch_from_nass
from src.config import settings
from src.logging_setup import configure_logging
from src.paths import DATA_RAW


def main() -> Path:
    """Run the NASS fetch CLI and write `data/raw/nass_yields.parquet`."""
    configure_logging()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / "nass_yields.parquet"

    df = fetch_from_nass(
        api_key=settings.nass_api_key,
        state_alphas=settings.states,
        year_min=settings.year_min,
        year_max=settings.year_max,
        commodity=settings.commodity,
    )
    df.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    main()
