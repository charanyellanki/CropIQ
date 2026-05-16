"""
Concept: YieldObservation
Purpose: Record an observed annual yield for a county-commodity pair.
State: county_fips (str, 5-char), year (int), commodity (str), yield_bu_per_acre (float),
    source (str). The DataFrame returned by `to_dataframe` carries one row per observation.
Actions: fetch_from_nass(state, year_range), validate(row), to_dataframe().
Operational principle: Fetching corn yield for IA 2023 returns ~99 rows (one per Iowa
    county) with non-null yields and 5-char county FIPS strings.

Notes / gotchas applied:
- #1: county_fips is constructed as string and zero-padded to 5 chars; never int.
- #2: NASS HTTP calls are wrapped in tenacity exponential backoff.
- #3: NASS suppression markers like "(D)" are coerced to NaN and filtered out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
SUPPRESSION_TOKENS = {"(D)", "(Z)", "(NA)", "(X)", "(S)", ""}


@dataclass(frozen=True)
class YieldObservation:
    """One county-year corn yield observation.

    Attributes:
        county_fips: 5-char zero-padded state+county FIPS string.
        year: Crop year as int.
        commodity: e.g. "CORN".
        yield_bu_per_acre: Reported yield, bushels per acre.
        source: Origin tag, e.g. "NASS_SURVEY".
    """

    county_fips: str
    year: int
    commodity: str
    yield_bu_per_acre: float
    source: str

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if not (isinstance(self.county_fips, str) and len(self.county_fips) == 5 and self.county_fips.isdigit()):
            raise ValueError(f"county_fips must be 5-digit string, got {self.county_fips!r}")
        if self.yield_bu_per_acre <= 0:
            raise ValueError(f"yield_bu_per_acre must be > 0, got {self.yield_bu_per_acre}")


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((requests.RequestException, ValueError)),
)
def _nass_request(params: dict) -> list[dict]:
    """Perform a single NASS Quick Stats GET with retry. Returns the `data` list.

    Args:
        params: Query parameters (must include `key`, `commodity_desc`, etc.).

    Returns:
        List of raw row dicts as returned by the NASS API.

    Raises:
        requests.RequestException on irrecoverable HTTP failure after retries.
        ValueError if the response body is unparseable JSON.
    """
    response = requests.get(NASS_BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if "data" not in payload:
        raise ValueError(f"NASS response missing 'data' field: {payload!r}")
    return list(payload["data"])


def fetch_from_nass(
    api_key: str,
    state_alphas: Iterable[str],
    year_min: int,
    year_max: int,
    commodity: str = "CORN",
) -> pd.DataFrame:
    """Fetch county-year corn yield from USDA NASS Quick Stats.

    The query is chunked by state to stay within the 50,000-cell rate-limit.

    Args:
        api_key: USDA NASS Quick Stats API key.
        state_alphas: Iterable of two-letter state codes (e.g., ["IA", "IL", "NE"]).
        year_min: Inclusive lower year bound.
        year_max: Inclusive upper year bound.
        commodity: NASS `commodity_desc`. Default "CORN".

    Returns:
        DataFrame with columns county_fips (str, 5-char), year (int), commodity (str),
        yield_bu_per_acre (float), source (str). Only finalized annual reports are kept
        (reference_period_desc == "YEAR"); suppressed/missing values are dropped.

    Raises:
        ValueError if no rows survive cleaning.
    """
    if not api_key:
        raise ValueError("NASS_API_KEY is empty — populate .env (see .env.example).")

    frames: list[pd.DataFrame] = []
    for state in state_alphas:
        logger.info(f"NASS: fetching {commodity} yield for {state} {year_min}-{year_max}")
        params = {
            "key": api_key,
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "commodity_desc": commodity,
            "statisticcat_desc": "YIELD",
            "agg_level_desc": "COUNTY",
            "state_alpha": state,
            "year__GE": str(year_min),
            "year__LE": str(year_max),
            "unit_desc": "BU / ACRE",
            "format": "JSON",
        }
        rows = _nass_request(params)
        logger.info(f"NASS: {state} returned {len(rows)} raw rows")
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise ValueError("NASS returned zero rows across all states.")

    raw = pd.concat(frames, ignore_index=True)
    return _clean_nass_rows(raw, year_min=year_min, year_max=year_max, commodity=commodity)


def _clean_nass_rows(raw: pd.DataFrame, year_min: int, year_max: int, commodity: str) -> pd.DataFrame:
    """Normalize raw NASS rows into the YieldObservation schema."""
    df = raw.copy()

    # Keep only finalized annual reports.
    df = df.loc[df["reference_period_desc"].astype(str).str.upper() == "YEAR"].copy()

    # Use the all-practices aggregate yield to avoid double-counting irrigated /
    # non-irrigated variants (which exist for Nebraska in particular).
    if "prodn_practice_desc" in df.columns:
        df = df.loc[
            df["prodn_practice_desc"].astype(str).str.upper() == "ALL PRODUCTION PRACTICES"
        ].copy()
    if "util_practice_desc" in df.columns:
        df = df.loc[df["util_practice_desc"].astype(str).str.upper() == "GRAIN"].copy()

    # Coerce year to int, filter to requested range.
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.loc[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
    df["year"] = df["year"].astype(int)

    # Suppression markers → NaN, then drop.
    value_series = df["Value"].astype(str).str.replace(",", "", regex=False).str.strip()
    value_series = value_series.where(~value_series.isin(SUPPRESSION_TOKENS))
    df["yield_bu_per_acre"] = pd.to_numeric(value_series, errors="coerce")
    df = df.loc[df["yield_bu_per_acre"].notna()].copy()

    # county_fips = state_fips_code (2) + county_code (3), zero-padded, as STRING.
    state_fips = df["state_fips_code"].astype(str).str.zfill(2)
    county_code = df["county_code"].astype(str).str.zfill(3)
    df["county_fips"] = (state_fips + county_code).str.zfill(5)
    df = df.loc[df["county_fips"].str.len() == 5].copy()
    # Drop pseudo-counties (county_code == "998" or "999" for state-level rollups).
    df = df.loc[~df["county_fips"].str.endswith(("998", "999"))].copy()

    df["commodity"] = commodity
    df["source"] = "NASS_SURVEY"

    out = df.loc[:, ["county_fips", "year", "commodity", "yield_bu_per_acre", "source"]].copy()
    # Deduplicate — NASS occasionally returns multiple rows per county-year (different statisticcat).
    out = out.sort_values(["county_fips", "year"]).drop_duplicates(["county_fips", "year"], keep="first")
    out = out.reset_index(drop=True)

    if out.empty:
        raise ValueError("After cleaning, NASS yields DataFrame is empty.")
    logger.info(f"NASS: cleaned to {len(out)} county-year rows")
    return out
