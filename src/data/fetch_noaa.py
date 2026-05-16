"""Fetch monthly county weather → `data/raw/noaa_weather.parquet`.

Usage:
    python -m src.data.fetch_noaa
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.concepts.weather_series import fetch_from_nclimgrid
from src.config import settings
from src.logging_setup import configure_logging
from src.paths import DATA_RAW


def main() -> Path:
    """Run the NOAA weather fetch CLI and write `data/raw/noaa_weather.parquet`."""
    configure_logging()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW / "noaa_weather.parquet"

    df = fetch_from_nclimgrid(
        state_alphas=settings.states,
        state_alpha_to_fips=settings.state_fips,
        year_min=settings.year_min,
        year_max=settings.year_max,
    )
    df.to_parquet(out_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_path}")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    main()
