"""Precompute predictions for every (state, county, test_year) → CSV cache.

Optional helper for the dashboard. Reading the cached CSV is faster than running
SHAP for every county on the choropleth. Writes:

    data/processed/predictions_cache.csv

Columns: state, county_fips, year, predicted, ci_low, ci_high, top_drivers (JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.config import settings
from src.inference.prediction_engine import PredictionEngine
from src.logging_setup import configure_logging
from src.paths import BEST_MODEL_PATH, DATA_PROCESSED, FEATURES_PARQUET


def main() -> Path:
    configure_logging()
    engine = PredictionEngine(BEST_MODEL_PATH, FEATURES_PARQUET)
    features = pd.read_parquet(FEATURES_PARQUET)
    features["county_fips"] = features["county_fips"].astype(str).str.zfill(5)
    features["year"] = features["year"].astype(int)

    years = list(settings.test_years)
    sub = features.loc[features["year"].isin(years)].copy()

    rows = []
    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="precompute"):
        try:
            r = engine.predict(row["state"], row["county_fips"], int(row["year"]))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed for {row['county_fips']}/{row['year']}: {exc}")
            continue
        rows.append(
            {
                "state": r.state,
                "county_fips": r.county_fips,
                "year": r.year,
                "predicted": r.predicted_yield_bu_per_acre,
                "ci_low": r.confidence_interval_80pct[0],
                "ci_high": r.confidence_interval_80pct[1],
                "top_drivers": json.dumps([d.model_dump() for d in r.top_drivers]),
            }
        )

    out = DATA_PROCESSED / "predictions_cache.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"Wrote {len(rows)} predictions to {out}")
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
