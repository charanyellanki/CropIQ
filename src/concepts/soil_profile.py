"""
Concept: SoilProfile
Purpose: Provide static soil characteristics for a county.
State: county_fips, organic_matter_pct, ph_mean, awc_mean, pct_well_drained,
    pct_prime_farmland.
Actions: load_ssurgo(state_alphas), to_dataframe().
Operational principle: Loading IA SSURGO produces one row per county with all five soil
    features populated.

Data source: USDA SDA Tabular Service (https://sdmdataaccess.nrcs.usda.gov/), which exposes
SSURGO via SQL POST queries. We aggregate component-level properties to the survey-area
level (which corresponds 1:1 with county FIPS in IA/IL/NE: areasymbol = state_alpha +
zero-padded county_code → county_fips = state_fips + county_code).

Fields used (per SSURGO `muaggatt`/`component`/`chorizon`):
- om_r (organic matter %)
- ph1to1h2o_r (1:1 water pH)
- awc_r (available water capacity)
- drclassdcd (drainage class)
- niccdcd (non-irrigated capability class)
- mu.muareaacres + ma.mukey weights aggregations to map unit acreage.

Gotchas applied:
- #1: county_fips constructed as 5-char string.
- #2: tenacity retry on SDA HTTP errors.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"

_PROPERTY_QUERY_TEMPLATE = """
SELECT
    l.areasymbol,
    SUM(CAST(mu.muacres AS float)) AS total_acres,
    SUM(CASE WHEN ma.drclassdcd IN ('Well drained','Somewhat excessively drained','Excessively drained')
             THEN CAST(mu.muacres AS float) ELSE 0 END) AS acres_well_drained,
    SUM(CASE WHEN mu.farmlndcl LIKE 'All areas are prime farmland%'
             THEN CAST(mu.muacres AS float) ELSE 0 END) AS acres_prime_farmland
FROM legend l
INNER JOIN mapunit mu ON mu.lkey = l.lkey
INNER JOIN muaggatt ma ON ma.mukey = mu.mukey
WHERE l.areasymbol IN ({areasymbol_list})
  AND mu.muacres IS NOT NULL
GROUP BY l.areasymbol
"""

_HORIZON_QUERY_TEMPLATE = """
SELECT
    l.areasymbol,
    SUM(CAST(mu.muacres AS float) * COALESCE(c.comppct_r, 0) * COALESCE(ch.om_r, 0)) AS om_w,
    SUM(CAST(mu.muacres AS float) * COALESCE(c.comppct_r, 0) * COALESCE(ch.ph1to1h2o_r, 0)) AS ph_w,
    SUM(CAST(mu.muacres AS float) * COALESCE(c.comppct_r, 0) * COALESCE(ch.awc_r, 0)) AS awc_w,
    SUM(CAST(mu.muacres AS float) * COALESCE(c.comppct_r, 0)
         * CASE WHEN ch.om_r IS NOT NULL THEN 1 ELSE 0 END) AS om_wt,
    SUM(CAST(mu.muacres AS float) * COALESCE(c.comppct_r, 0)
         * CASE WHEN ch.ph1to1h2o_r IS NOT NULL THEN 1 ELSE 0 END) AS ph_wt,
    SUM(CAST(mu.muacres AS float) * COALESCE(c.comppct_r, 0)
         * CASE WHEN ch.awc_r IS NOT NULL THEN 1 ELSE 0 END) AS awc_wt
FROM legend l
INNER JOIN mapunit mu ON mu.lkey = l.lkey
INNER JOIN component c ON c.mukey = mu.mukey
INNER JOIN chorizon ch ON ch.cokey = c.cokey
WHERE l.areasymbol IN ({areasymbol_list})
  AND ch.hzdept_r <= 30
  AND mu.muacres IS NOT NULL
GROUP BY l.areasymbol
"""


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(requests.RequestException),
)
def _sda_query(sql: str) -> pd.DataFrame:
    """POST a SQL query to the USDA SDA tabular service and return rows.

    Args:
        sql: SQL string compatible with SSURGO schema.

    Returns:
        DataFrame using the first row of the response as column headers.
    """
    response = requests.post(SDA_URL, json={"query": sql, "format": "JSON+COLUMNNAME"}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    table = payload.get("Table")
    if not table:
        return pd.DataFrame()
    header, *rows = table
    return pd.DataFrame(rows, columns=header)


def _list_areasymbols(state_alphas: Iterable[str]) -> list[str]:
    """List SSURGO areasymbols for the given states (e.g., IA001, IA003, ...)."""
    state_in = ", ".join(f"'{s}'" for s in state_alphas)
    sql = f"SELECT areasymbol FROM legend WHERE LEFT(areasymbol, 2) IN ({state_in})"
    df = _sda_query(sql)
    if df.empty:
        raise ValueError("SSURGO returned no legend rows for requested states.")
    syms = sorted({s for s in df["areasymbol"].astype(str)
                   if len(s) == 5 and s[2:].isdigit()})
    logger.info(f"SSURGO: {len(syms)} areasymbols across states")
    return syms


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def load_ssurgo(
    state_alphas: Iterable[str], state_alpha_to_fips: dict[str, str]
) -> pd.DataFrame:
    """Load aggregated county-level soil features from USDA SSURGO via SDA.

    Args:
        state_alphas: Iterable of state abbreviations (e.g., ["IA", "IL", "NE"]).
        state_alpha_to_fips: Mapping state abbreviation → 2-digit FIPS.

    Returns:
        DataFrame keyed by county_fips with columns:
        soil_organic_matter_pct, soil_ph_mean, soil_awc_mean, pct_well_drained,
        pct_prime_farmland. One row per county.
    """
    state_alphas = list(state_alphas)
    areasymbols = _list_areasymbols(state_alphas)

    prop_frames: list[pd.DataFrame] = []
    horiz_frames: list[pd.DataFrame] = []
    for chunk in _chunked(areasymbols, 75):
        sym_list = ", ".join(f"'{s}'" for s in chunk)
        logger.info(f"SSURGO: querying {len(chunk)} areasymbols (mapunit-level)")
        prop_frames.append(_sda_query(_PROPERTY_QUERY_TEMPLATE.format(areasymbol_list=sym_list)))
        logger.info(f"SSURGO: querying {len(chunk)} areasymbols (horizon-level)")
        horiz_frames.append(_sda_query(_HORIZON_QUERY_TEMPLATE.format(areasymbol_list=sym_list)))

    props = pd.concat(prop_frames, ignore_index=True) if prop_frames else pd.DataFrame()
    horiz = pd.concat(horiz_frames, ignore_index=True) if horiz_frames else pd.DataFrame()
    if props.empty or horiz.empty:
        raise ValueError("SSURGO returned empty aggregation tables.")

    for col in ("total_acres", "acres_well_drained", "acres_prime_farmland"):
        props[col] = pd.to_numeric(props[col], errors="coerce").fillna(0.0)
    for col in ("om_w", "ph_w", "awc_w", "om_wt", "ph_wt", "awc_wt"):
        horiz[col] = pd.to_numeric(horiz[col], errors="coerce").fillna(0.0)

    df = props.merge(horiz, on="areasymbol", how="inner")

    df["soil_organic_matter_pct"] = df["om_w"] / df["om_wt"].replace(0.0, pd.NA)
    df["soil_ph_mean"] = df["ph_w"] / df["ph_wt"].replace(0.0, pd.NA)
    df["soil_awc_mean"] = df["awc_w"] / df["awc_wt"].replace(0.0, pd.NA)
    df["pct_well_drained"] = (df["acres_well_drained"] / df["total_acres"].replace(0.0, pd.NA)) * 100.0
    df["pct_prime_farmland"] = (df["acres_prime_farmland"] / df["total_acres"].replace(0.0, pd.NA)) * 100.0

    def _to_county_fips(areasymbol: str) -> str | None:
        if not isinstance(areasymbol, str) or len(areasymbol) != 5:
            return None
        state_alpha, county_part = areasymbol[:2], areasymbol[2:]
        state_fips = state_alpha_to_fips.get(state_alpha)
        if state_fips is None or not county_part.isdigit():
            return None
        return f"{state_fips}{county_part.zfill(3)}"

    df["county_fips"] = df["areasymbol"].map(_to_county_fips)
    df = df.dropna(subset=["county_fips"])

    out_cols = [
        "county_fips",
        "soil_organic_matter_pct",
        "soil_ph_mean",
        "soil_awc_mean",
        "pct_well_drained",
        "pct_prime_farmland",
    ]
    out = df.loc[:, out_cols].copy()
    out = out.dropna(subset=[c for c in out_cols if c != "county_fips"], how="all")
    out = out.drop_duplicates("county_fips").sort_values("county_fips").reset_index(drop=True)
    logger.info(f"SSURGO: aggregated to {len(out)} counties")
    return out
