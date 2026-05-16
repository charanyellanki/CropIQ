"""Unit tests for `YieldObservation` cleaning logic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.concepts.yield_observation import YieldObservation, _clean_nass_rows


def _row(state_fips: str, county_code: str, year: int, value: str, *, prodn: str = "ALL PRODUCTION PRACTICES", util: str = "GRAIN", ref: str = "YEAR") -> dict:
    return {
        "state_fips_code": state_fips,
        "county_code": county_code,
        "year": year,
        "Value": value,
        "reference_period_desc": ref,
        "prodn_practice_desc": prodn,
        "util_practice_desc": util,
    }


def test_clean_rows_zero_pads_fips_and_drops_rollups() -> None:
    raw = pd.DataFrame(
        [
            _row("19", "1", 2020, "200"),     # Adair County, IA → 19001
            _row("19", "998", 2020, "190"),   # state rollup → dropped
            _row("17", "037", 2021, "210"),   # DeKalb County, IL → 17037
        ]
    )
    out = _clean_nass_rows(raw, year_min=2010, year_max=2024, commodity="CORN")
    assert list(out["county_fips"]) == ["17037", "19001"]
    assert out["county_fips"].dtype == object
    assert out["county_fips"].str.len().eq(5).all()


def test_clean_rows_drops_suppression_markers() -> None:
    raw = pd.DataFrame(
        [
            _row("19", "1", 2020, "(D)"),
            _row("19", "3", 2020, "180.5"),
        ]
    )
    out = _clean_nass_rows(raw, year_min=2010, year_max=2024, commodity="CORN")
    assert len(out) == 1
    assert out.iloc[0]["yield_bu_per_acre"] == pytest.approx(180.5)


def test_clean_rows_filters_to_all_production_practices() -> None:
    raw = pd.DataFrame(
        [
            _row("31", "1", 2020, "180", prodn="IRRIGATED"),
            _row("31", "1", 2020, "150", prodn="NON-IRRIGATED"),
            _row("31", "1", 2020, "165", prodn="ALL PRODUCTION PRACTICES"),
        ]
    )
    out = _clean_nass_rows(raw, year_min=2010, year_max=2024, commodity="CORN")
    assert len(out) == 1
    assert out.iloc[0]["yield_bu_per_acre"] == pytest.approx(165.0)


def test_clean_rows_filters_to_year_range() -> None:
    raw = pd.DataFrame(
        [
            _row("19", "1", 2025, "200"),
            _row("19", "1", 2020, "180"),
        ]
    )
    out = _clean_nass_rows(raw, year_min=2010, year_max=2024, commodity="CORN")
    assert list(out["year"]) == [2020]


def test_yield_observation_dataclass_validates() -> None:
    YieldObservation(county_fips="19001", year=2020, commodity="CORN", yield_bu_per_acre=180.0, source="NASS_SURVEY")
    with pytest.raises(ValueError):
        YieldObservation(county_fips="1900", year=2020, commodity="CORN", yield_bu_per_acre=180.0, source="x")
    with pytest.raises(ValueError):
        YieldObservation(county_fips="19001", year=2020, commodity="CORN", yield_bu_per_acre=-1.0, source="x")
