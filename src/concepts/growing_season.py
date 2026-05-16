"""
Concept: GrowingSeason
Purpose: Define temporal windows for corn development stages and compute weather features
    aggregated over those windows.
State: A mapping from stage name → (start_doy, end_doy, base_temp_c). Provided as constants
    for corn; expressed via simple static methods because the windows are commodity-fixed.
Actions: stage_window(name), compute_gdd_for_window(weather, start_doy, end_doy, base_temp_c),
    compute_precip_for_window(weather, start_doy, end_doy).
Operational principle: `compute_gdd(weather, "v6_to_vt")` returns one GDD value per
    (county_fips, year) row for the critical reproductive window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(frozen=True)
class StageWindow:
    """A named growth window expressed in day-of-year (Jan 1 = 1)."""

    name: str
    start_doy: int
    end_doy: int
    base_temp_c: float

    def overlap_days_in_month(self, year: int, month: int) -> int:
        """Return the number of days in `month` that fall inside this window."""
        if month < 1 or month > 12:
            return 0
        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            month_days = 29
        else:
            month_days = DAYS_IN_MONTH[month - 1]
        month_start_doy = sum(DAYS_IN_MONTH[: month - 1]) + 1
        month_end_doy = month_start_doy + month_days - 1
        overlap_start = max(self.start_doy, month_start_doy)
        overlap_end = min(self.end_doy, month_end_doy)
        return max(0, overlap_end - overlap_start + 1)


# Corn stage windows. Day-of-year boundaries are calendar-day approximations of
# stage timing typical to Iowa/Illinois/Nebraska latitudes.
CORN_STAGES: dict[str, StageWindow] = {
    "growing_season": StageWindow("growing_season", 121, 273, 10.0),  # May 1 – Sep 30, base 50°F
    "v6_to_vt": StageWindow("v6_to_vt", 166, 212, 10.0),                # Jun 15 – Jul 31, base 50°F
    "critical_window": StageWindow("critical_window", 182, 227, 10.0),  # Jul 1 – Aug 15
    "july": StageWindow("july", 182, 212, 10.0),
    "july_aug": StageWindow("july_aug", 182, 243, 10.0),
}


def _avg_temp_c(monthly: pd.DataFrame) -> pd.Series:
    return (monthly["tmin_c"] + monthly["tmax_c"]) / 2.0


def compute_gdd_for_window(
    monthly_weather: pd.DataFrame, window: StageWindow
) -> pd.DataFrame:
    """Compute growing-degree-days over a corn stage window.

    GDD is approximated month-by-month: per month, GDD = max(tavg - base, 0) × overlap_days.
    This is the standard county-month tabular approximation (no daily resolution required).

    Args:
        monthly_weather: DataFrame with columns county_fips, year, month, tmin_c, tmax_c.
        window: Stage window to compute over.

    Returns:
        DataFrame keyed by (county_fips, year) with a single column `gdd_<window>`.
    """
    df = monthly_weather.copy()
    df["overlap_days"] = df.apply(
        lambda r: window.overlap_days_in_month(int(r["year"]), int(r["month"])), axis=1
    )
    df = df.loc[df["overlap_days"] > 0].copy()
    df["tavg_c"] = _avg_temp_c(df)
    df["daily_gdd_c"] = (df["tavg_c"] - window.base_temp_c).clip(lower=0.0)
    df["monthly_gdd"] = df["daily_gdd_c"] * df["overlap_days"]
    agg = (
        df.groupby(["county_fips", "year"], as_index=False)["monthly_gdd"]
        .sum()
        .rename(columns={"monthly_gdd": f"gdd_{window.name}"})
    )
    return agg


def compute_precip_for_window(
    monthly_weather: pd.DataFrame, window: StageWindow
) -> pd.DataFrame:
    """Sum precipitation (mm) over a corn stage window using month-overlap days.

    Per month, monthly precip is allocated linearly across days, so window precip =
    monthly_precip × (overlap_days / days_in_month).

    Args:
        monthly_weather: DataFrame with county_fips, year, month, prcp_mm.
        window: Stage window to compute over.

    Returns:
        DataFrame keyed by (county_fips, year) with a single column `precip_<window>_mm`.
    """
    df = monthly_weather.copy()
    df["overlap_days"] = df.apply(
        lambda r: window.overlap_days_in_month(int(r["year"]), int(r["month"])), axis=1
    )
    df["days_in_month"] = df.apply(
        lambda r: (
            29
            if (int(r["month"]) == 2 and (int(r["year"]) % 4 == 0 and (int(r["year"]) % 100 != 0 or int(r["year"]) % 400 == 0)))
            else DAYS_IN_MONTH[int(r["month"]) - 1]
        ),
        axis=1,
    )
    df = df.loc[df["overlap_days"] > 0].copy()
    df["window_precip_mm"] = df["prcp_mm"] * (df["overlap_days"] / df["days_in_month"])
    agg = (
        df.groupby(["county_fips", "year"], as_index=False)["window_precip_mm"]
        .sum()
        .rename(columns={"window_precip_mm": f"precip_{window.name}_mm"})
    )
    return agg


def compute_days_above_30c(monthly_weather: pd.DataFrame, months: tuple[int, ...]) -> pd.DataFrame:
    """Approximate the number of days with tmax >= 30°C in the given months.

    Uses a simple sinusoidal approximation: if tmax_month_c is the monthly tmax mean,
    we estimate days_hot as min(month_days, max(0, (tmax_month_c - 27) * 4)) clipped to
    [0, month_days]. This is a tabular proxy; it preserves rank ordering across
    counties and years, which is sufficient for tree models.

    Args:
        monthly_weather: DataFrame with county_fips, year, month, tmax_c.
        months: Months to include, e.g., (7, 8) for July + August.

    Returns:
        DataFrame keyed by (county_fips, year) with `days_above_30c_july_aug`.
    """
    df = monthly_weather.loc[monthly_weather["month"].isin(list(months))].copy()
    df["days_hot_month"] = ((df["tmax_c"] - 27.0) * 4.0).clip(lower=0.0)
    df["days_in_month"] = df["month"].map(lambda m: DAYS_IN_MONTH[m - 1])
    df["days_hot_month"] = np.minimum(df["days_hot_month"], df["days_in_month"])
    agg = (
        df.groupby(["county_fips", "year"], as_index=False)["days_hot_month"]
        .sum()
        .rename(columns={"days_hot_month": "days_above_30c_july_aug"})
    )
    return agg
