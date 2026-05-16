"""Tests for `FeatureMatrix.build` and schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.concepts.feature_matrix import REQUIRED_COLUMNS, build, validate_schema


def _yields_for(counties: list[str], years: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    rows = []
    for c in counties:
        for y in years:
            rows.append({
                "county_fips": c,
                "year": y,
                "commodity": "CORN",
                "yield_bu_per_acre": float(170 + rng.standard_normal() * 15),
                "source": "TEST",
            })
    return pd.DataFrame(rows)


def _weather_for(counties: list[str], years: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(seed=7)
    rows = []
    for c in counties:
        for y in years:
            for m in range(1, 13):
                base = 22.0 if m in (6, 7, 8) else 8.0 if m in (3, 4, 5, 9, 10, 11) else -5.0
                tavg = base + rng.standard_normal()
                rows.append({
                    "county_fips": c,
                    "year": y,
                    "month": m,
                    "tmax_c": tavg + 6.0,
                    "tmin_c": tavg - 6.0,
                    "prcp_mm": max(0.0, 70.0 + rng.standard_normal() * 30.0),
                })
    return pd.DataFrame(rows)


def _soil_for(counties: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "county_fips": c,
            "soil_organic_matter_pct": 2.5,
            "soil_ph_mean": 6.4,
            "soil_awc_mean": 0.18,
            "pct_well_drained": 55.0,
            "pct_prime_farmland": 35.0,
        }
        for c in counties
    ])


def test_build_produces_required_columns_and_passes_validation() -> None:
    counties = ["19169", "17031", "31109"]
    years = list(range(2018, 2024))
    df = build(
        _yields_for(counties, years),
        _weather_for(counties, years),
        _soil_for(counties),
        state_fips_to_alpha={"17": "IL", "19": "IA", "31": "NE"},
        one_hot_states=True,
    )
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, col
    validate_schema(df)
    assert df["county_fips"].str.len().eq(5).all()
    assert df["year"].dtype.kind in {"i", "u"}
    # One-hot state columns
    assert any(c.startswith("state_") for c in df.columns)
