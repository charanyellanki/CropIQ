"""Tests for `GrowingSeason` window computations."""

from __future__ import annotations

import pandas as pd
import pytest

from src.concepts.growing_season import (
    CORN_STAGES,
    compute_days_above_30c,
    compute_gdd_for_window,
    compute_precip_for_window,
)


def _two_counties_one_year() -> pd.DataFrame:
    rows = []
    # County A: warm and wet July
    for month in range(1, 13):
        tavg = -5.0 if month in (1, 2, 12) else 22.0 if month == 7 else 15.0
        rows.append({
            "county_fips": "19169", "year": 2023, "month": month,
            "tmin_c": tavg - 5.0, "tmax_c": tavg + 5.0,
            "prcp_mm": 120.0 if month == 7 else 30.0,
        })
    # County B: cool and dry July
    for month in range(1, 13):
        tavg = -7.0 if month in (1, 2, 12) else 18.0 if month == 7 else 13.0
        rows.append({
            "county_fips": "17031", "year": 2023, "month": month,
            "tmin_c": tavg - 5.0, "tmax_c": tavg + 5.0,
            "prcp_mm": 60.0 if month == 7 else 30.0,
        })
    return pd.DataFrame(rows)


def test_gdd_growing_season_positive_and_warm_county_higher() -> None:
    df = _two_counties_one_year()
    out = compute_gdd_for_window(df, CORN_STAGES["growing_season"]).sort_values("county_fips")
    iowa_gdd = float(out.loc[out["county_fips"] == "19169", "gdd_growing_season"].iloc[0])
    illinois_gdd = float(out.loc[out["county_fips"] == "17031", "gdd_growing_season"].iloc[0])
    assert iowa_gdd > 0
    assert iowa_gdd > illinois_gdd  # warmer county has more GDD


def test_precip_july_window_isolates_july() -> None:
    df = _two_counties_one_year()
    out = compute_precip_for_window(df, CORN_STAGES["july"])
    iowa = float(out.loc[out["county_fips"] == "19169", "precip_july_mm"].iloc[0])
    illinois = float(out.loc[out["county_fips"] == "17031", "precip_july_mm"].iloc[0])
    assert iowa == pytest.approx(120.0, rel=0.01)
    assert illinois == pytest.approx(60.0, rel=0.01)


def test_days_above_30c_zero_when_cool() -> None:
    rows = []
    for month in range(1, 13):
        rows.append({
            "county_fips": "19169", "year": 2023, "month": month,
            "tmin_c": 10.0, "tmax_c": 25.0, "prcp_mm": 50.0,
        })
    df = pd.DataFrame(rows)
    out = compute_days_above_30c(df, (7, 8))
    assert float(out["days_above_30c_july_aug"].iloc[0]) == 0.0
