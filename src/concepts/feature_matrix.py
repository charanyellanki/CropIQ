"""
Concept: FeatureMatrix
Purpose: Produce a model-ready feature matrix by joining domain concepts on
    (county_fips, year).
State: One row per (county_fips, year) with the target column and all engineered features
    defined in CLAUDE.md §9.
Actions: build(yields, weather, soil), validate_schema(df), save(df, path), load(path).
Operational principle: Building for IA/IL/NE 2010–2024 produces ~5K rows with 20+ columns
    and zero NaN in required columns.

Synchronization: This module is the single place that calls `GrowingSeason.compute_*` and
joins them with `YieldObservation`, `WeatherSeries`, and `SoilProfile` (CLAUDE.md §6).

Gotchas applied:
- #1, #11: county_fips coerced to 5-char string on every input.
- #4: .loc[]-based assignment throughout.
- #12: dtypes set explicitly before to_parquet.
- #17, #20: get_dummies on full set; year as int.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.concepts.growing_season import (
    CORN_STAGES,
    compute_days_above_30c,
    compute_gdd_for_window,
    compute_precip_for_window,
)

# Static county centroid lookup (lat, lon) — embedded for IA/IL/NE so the pipeline does
# not require geopandas (gotcha #28). Source: US Census Gazetteer Files.
# Provided lazily via `_load_centroids` — values live in data/processed/county_centroids.csv
# if present; otherwise we leave lat/lon NaN and the model can still train without them.

REQUIRED_COLUMNS: list[str] = [
    "county_fips",
    "year",
    "yield_bu_per_acre",
    "gdd_total",
    "gdd_v6_to_vt",
    "precip_total_growing_season_mm",
    "precip_july_mm",
    "precip_critical_window_mm",
    "days_above_30c_july_aug",
    "tmax_mean_july_c",
    "soil_organic_matter_pct",
    "soil_ph_mean",
    "soil_awc_mean",
    "pct_well_drained",
    "pct_prime_farmland",
    "yield_lag1",
    "yield_rolling_mean_3yr",
    "yield_rolling_std_3yr",
    # Weather anomalies (departures from the 2010–train_year_max county climatology)
    # let the tree model focus on the *deviation* signal rather than re-learning the
    # baseline. They are also key inputs for SHAP-driven agronomic interpretation.
    "precip_july_anomaly_mm",
    "tmax_july_anomaly_c",
    "gdd_v6_to_vt_anomaly",
]


def _ensure_str_fips(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["county_fips"] = out["county_fips"].astype(str).str.zfill(5)
    return out


def _weather_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute the full set of GDD / precipitation / heat features."""
    monthly = _ensure_str_fips(monthly)
    monthly["year"] = monthly["year"].astype(int)
    monthly["month"] = monthly["month"].astype(int)

    pieces = [
        compute_gdd_for_window(monthly, CORN_STAGES["growing_season"]).rename(
            columns={"gdd_growing_season": "gdd_total"}
        ),
        compute_gdd_for_window(monthly, CORN_STAGES["v6_to_vt"]),
        compute_precip_for_window(monthly, CORN_STAGES["growing_season"]).rename(
            columns={"precip_growing_season_mm": "precip_total_growing_season_mm"}
        ),
        compute_precip_for_window(monthly, CORN_STAGES["july"]).rename(
            columns={"precip_july_mm": "precip_july_mm"}
        ),
        compute_precip_for_window(monthly, CORN_STAGES["critical_window"]),
        compute_days_above_30c(monthly, (7, 8)),
    ]

    july = monthly.loc[monthly["month"] == 7, ["county_fips", "year", "tmax_c"]].copy()
    july = july.rename(columns={"tmax_c": "tmax_mean_july_c"})
    pieces.append(july)

    weather = reduce(
        lambda a, b: a.merge(b, on=["county_fips", "year"], how="outer"),
        pieces,
    )
    return weather


def _temporal_features(yields: pd.DataFrame) -> pd.DataFrame:
    """Compute per-county temporal lag and rolling-statistic features."""
    df = _ensure_str_fips(yields)
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["county_fips", "year"]).reset_index(drop=True)
    grp = df.groupby("county_fips", group_keys=False)
    df["yield_lag1"] = grp["yield_bu_per_acre"].shift(1)
    df["yield_rolling_mean_3yr"] = (
        grp["yield_bu_per_acre"].shift(1).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["yield_rolling_std_3yr"] = (
        grp["yield_bu_per_acre"].shift(1).rolling(window=3, min_periods=2).std().reset_index(level=0, drop=True)
    )
    return df.loc[:, ["county_fips", "year", "yield_lag1", "yield_rolling_mean_3yr", "yield_rolling_std_3yr"]]


def _attach_state(df: pd.DataFrame, state_fips_to_alpha: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["state"] = out["county_fips"].str[:2].map(state_fips_to_alpha)
    return out


def _add_weather_anomalies(df: pd.DataFrame, climatology_year_max: int) -> pd.DataFrame:
    """Add anomaly columns (raw value - county climatology mean over baseline years)."""
    out = df.copy()
    base = out.loc[out["year"] <= climatology_year_max]
    norms = base.groupby("county_fips")[["precip_july_mm", "tmax_mean_july_c", "gdd_v6_to_vt"]].mean()
    norms.columns = ["_norm_precip_july", "_norm_tmax_july", "_norm_gdd_v6_to_vt"]
    out = out.merge(norms, left_on="county_fips", right_index=True, how="left")
    out["precip_july_anomaly_mm"] = out["precip_july_mm"] - out["_norm_precip_july"]
    out["tmax_july_anomaly_c"] = out["tmax_mean_july_c"] - out["_norm_tmax_july"]
    out["gdd_v6_to_vt_anomaly"] = out["gdd_v6_to_vt"] - out["_norm_gdd_v6_to_vt"]
    out = out.drop(columns=["_norm_precip_july", "_norm_tmax_july", "_norm_gdd_v6_to_vt"])
    return out


def build(
    yields: pd.DataFrame,
    weather: pd.DataFrame,
    soil: pd.DataFrame,
    state_fips_to_alpha: dict[str, str],
    one_hot_states: bool = True,
    climatology_year_max: int = 2020,
) -> pd.DataFrame:
    """Join yield / weather / soil into the model-ready feature matrix.

    Args:
        yields: Output of NASS fetcher; one row per (county_fips, year).
        weather: Monthly weather; many rows per (county_fips, year).
        soil: Static soil features; one row per county_fips.
        state_fips_to_alpha: Mapping like {"19": "IA", "17": "IL", "31": "NE"}.
        one_hot_states: Whether to one-hot encode the `state` column. CLAUDE.md §9
            requires this before model training (gotcha #17).

    Returns:
        DataFrame containing REQUIRED_COLUMNS plus optional one-hot state columns.
    """
    yields = _ensure_str_fips(yields)
    yields["year"] = yields["year"].astype(int)
    soil = _ensure_str_fips(soil)

    logger.info(f"FeatureMatrix.build: yields={len(yields)} weather_rows={len(weather)} soil={len(soil)}")

    weather_summary = _weather_summary(weather)
    temporal = _temporal_features(yields)

    df = yields.merge(weather_summary, on=["county_fips", "year"], how="left")
    df = df.merge(soil, on="county_fips", how="left")
    df = df.merge(temporal, on=["county_fips", "year"], how="left")
    df = _attach_state(df, state_fips_to_alpha)

    # Drop rows without lag/rolling features — first year per county has no history.
    df = df.dropna(subset=["yield_lag1"]).copy()
    # Year-2 per county lacks enough history for a rolling std; fill with 0 (no observed
    # variance) so that downstream models can still use the row.
    df["yield_rolling_std_3yr"] = df["yield_rolling_std_3yr"].fillna(0.0)

    # Weather anomalies vs county climatology (computed only on training-year rows so
    # we don't leak test info into the climatology baseline).
    df = _add_weather_anomalies(df, climatology_year_max=climatology_year_max)

    if one_hot_states:
        # gotcha #17: apply get_dummies BEFORE any split so columns are stable.
        dummies = pd.get_dummies(df["state"], prefix="state", dtype=float)
        df = pd.concat([df, dummies], axis=1)

    df = df.loc[:, _ordered_columns(df, one_hot_states)].reset_index(drop=True)
    df["year"] = df["year"].astype(int)
    df["county_fips"] = df["county_fips"].astype(str)
    return df


def _ordered_columns(df: pd.DataFrame, one_hot_states: bool) -> list[str]:
    base = list(REQUIRED_COLUMNS)
    extras = ["state"]
    if one_hot_states:
        extras.extend(sorted(c for c in df.columns if c.startswith("state_")))
    return [c for c in base + extras if c in df.columns]


def validate_schema(df: pd.DataFrame, *, tolerance_nan_pct: float = 0.05) -> None:
    """Assert required columns exist and have < `tolerance_nan_pct` missing values.

    Raises:
        AssertionError on any violation.
    """
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing required column: {col}"
        nan_frac = df[col].isna().mean()
        assert nan_frac < tolerance_nan_pct, (
            f"column {col} has {nan_frac:.1%} NaN (max {tolerance_nan_pct:.1%})"
        )
    assert df["county_fips"].str.len().eq(5).all(), "county_fips must all be 5-char strings"
    assert df["county_fips"].dtype == object, "county_fips must be object/str dtype"
    assert (df["yield_bu_per_acre"] > 0).all(), "yield must be positive"
    assert df["year"].dtype.kind in {"i", "u"}, "year must be integer dtype"


def save(df: pd.DataFrame, path: Path | str) -> Path:
    """Persist the feature matrix to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"FeatureMatrix: wrote {len(df)} rows × {len(df.columns)} cols → {path}")
    return path


def load(path: Path | str) -> pd.DataFrame:
    """Read the feature matrix back from parquet, restoring dtypes."""
    df = pd.read_parquet(path)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["year"] = df["year"].astype(int)
    return df
