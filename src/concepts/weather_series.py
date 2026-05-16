"""
Concept: WeatherSeries
Purpose: Provide weather measurements aggregated to county-month for a given year range.
State: Per (county_fips, year, month) record with tmin_c, tmax_c, prcp_mm.
Actions: fetch(state_alphas, year_min, year_max), to_dataframe().
Operational principle: Fetching IA 2023 returns 12 monthly rows per county with all
    weather variables populated.

Data source: NASA POWER daily-point API (`power.larc.nasa.gov`). For each county
centroid we request daily T2M_MAX, T2M_MIN, PRECTOTCORR, then aggregate to monthly
means (temperature) and totals (precipitation).

Why NASA POWER: free, no auth, no aggressive rate limits, archived data from 1981+,
monthly aggregation is unambiguous (daily → mean for temperatures, sum for precip),
and a single call returns the full 15-year range (~1.2 s / county). It replaced the
NOAA Climate-at-a-Glance source listed in CLAUDE.md §8.2 (returning >100 s/CSV at
build time) and the NASA POWER Historical Archive (free-tier hourly quota proved
too tight for a 294-county pull).

Gotchas applied:
- #1: county_fips kept as 5-char string throughout.
- #2: tenacity retry on transient HTTP errors.
- #22: tqdm progress over the per-county loop.
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

NASA_POWER_DAILY = "https://power.larc.nasa.gov/api/temporal/daily/point"
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2024_Gazetteer/2024_Gaz_counties_national.zip"
)


# ---------------------------------------------------------------------------
# Centroid lookup
# ---------------------------------------------------------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(requests.RequestException),
)
def fetch_county_centroids() -> pd.DataFrame:
    """Download the 2024 Census Gazetteer county centroid table.

    Returns:
        DataFrame with columns ["county_fips" (str, 5-char), "state_alpha", "county_name",
        "lat", "lon"]. One row per US county.
    """
    response = requests.get(GAZETTEER_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            raw = pd.read_csv(f, sep="\t", dtype={"GEOID": str})
    raw.columns = [c.strip() for c in raw.columns]
    out = raw.rename(
        columns={
            "USPS": "state_alpha",
            "GEOID": "county_fips",
            "NAME": "county_name",
            "INTPTLAT": "lat",
            "INTPTLONG": "lon",
        }
    ).loc[:, ["state_alpha", "county_fips", "county_name", "lat", "lon"]].copy()
    out["county_fips"] = out["county_fips"].astype(str).str.zfill(5)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# NASA POWER per-county fetch
# ---------------------------------------------------------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(requests.RequestException),
)
def _fetch_one(lat: float, lon: float, year_min: int, year_max: int) -> dict:
    """Call NASA POWER (daily) for a single county and return parsed JSON."""
    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR",
        "community": "AG",
        "longitude": float(lon),
        "latitude": float(lat),
        "start": f"{year_min}0101",
        "end": f"{year_max}1231",
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_DAILY, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def _aggregate_daily_to_monthly(payload: dict) -> pd.DataFrame:
    """Convert NASA POWER daily JSON to a monthly DataFrame.

    Returns columns: year, month, tmax_c, tmin_c, prcp_mm.

    NASA POWER uses -999.0 as the fill_value for missing days; we drop those before
    aggregating.
    """
    params = payload.get("properties", {}).get("parameter", {})
    if not params or "T2M_MAX" not in params:
        return pd.DataFrame(columns=["year", "month", "tmax_c", "tmin_c", "prcp_mm"])
    fill = float(payload.get("header", {}).get("fill_value", -999.0))
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(list(params["T2M_MAX"].keys()), format="%Y%m%d"),
            "tmax_c": pd.to_numeric(list(params["T2M_MAX"].values()), errors="coerce"),
            "tmin_c": pd.to_numeric(list(params["T2M_MIN"].values()), errors="coerce"),
            "prcp_mm": pd.to_numeric(list(params["PRECTOTCORR"].values()), errors="coerce"),
        }
    )
    for col in ("tmax_c", "tmin_c", "prcp_mm"):
        df.loc[df[col] == fill, col] = pd.NA
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    monthly = (
        df.dropna(subset=["tmax_c", "tmin_c", "prcp_mm"])
        .groupby(["year", "month"], as_index=False)
        .agg(tmax_c=("tmax_c", "mean"), tmin_c=("tmin_c", "mean"), prcp_mm=("prcp_mm", "sum"))
    )
    return monthly


def fetch_from_nclimgrid(
    state_alphas: Iterable[str],
    state_alpha_to_fips: dict[str, str],
    year_min: int,
    year_max: int,
    *,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch monthly weather for all counties in `state_alphas`.

    Despite the historical name (kept stable to match the concept's public action), the
    data source is NASA POWER; see module docstring for rationale.

    Args:
        state_alphas: Iterable of two-letter state codes (e.g., ["IA", "IL", "NE"]).
        state_alpha_to_fips: Unused here (centroid lookup uses state_alpha directly) but
            kept for signature compatibility with the original concept.
        year_min: Inclusive lower year bound.
        year_max: Inclusive upper year bound.
        max_workers: Parallelism for the per-county fetch.

    Returns:
        DataFrame with columns county_fips (str, 5-char), year (int), month (int 1..12),
        tmax_c (float), tmin_c (float), prcp_mm (float).
    """
    del state_alpha_to_fips  # kept for signature compatibility
    states = list(state_alphas)
    logger.info(f"Fetching weather for states={states} years={year_min}-{year_max}")

    centroids = fetch_county_centroids()
    centroids = centroids.loc[centroids["state_alpha"].isin(states)].reset_index(drop=True)
    logger.info(f"NASA POWER: {len(centroids)} counties to fetch")

    def _one_county(row: pd.Series) -> pd.DataFrame:
        payload = _fetch_one(float(row["lat"]), float(row["lon"]), year_min, year_max)
        monthly = _aggregate_daily_to_monthly(payload)
        monthly["county_fips"] = row["county_fips"]
        return monthly.loc[:, ["county_fips", "year", "month", "tmax_c", "tmin_c", "prcp_mm"]]

    results: list[pd.DataFrame] = []
    failures: list[pd.Series] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_one_county, row): row for _, row in centroids.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="NASA POWER", unit="county"):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 — collect failure, retry serially later
                logger.warning(
                    f"NASA POWER failed for {row['state_alpha']}/{row['county_fips']}: {exc}"
                )
                failures.append(row)

    if failures:
        logger.info(f"Retrying {len(failures)} failed counties serially with 5s spacing")
        import time

        for row in tqdm(failures, desc="NASA POWER retry", unit="county"):
            time.sleep(5)
            try:
                results.append(_one_county(row))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"NASA POWER retry failed permanently for "
                    f"{row['state_alpha']}/{row['county_fips']}: {exc}"
                )
                raise

    weather = pd.concat(results, ignore_index=True)
    weather["county_fips"] = weather["county_fips"].astype(str).str.zfill(5)
    weather["year"] = weather["year"].astype(int)
    weather["month"] = weather["month"].astype(int)
    weather = weather.dropna(subset=["tmax_c", "tmin_c", "prcp_mm"]).reset_index(drop=True)
    logger.info(f"NASA POWER: merged {len(weather)} county-month rows")
    return weather
