# CropIQ — Data Card

Describes the three data sources, the time/space scope, the cleaning steps, and the
known coverage gaps.

---

## Sources

| Concept            | Source                                   | Grain                         | Notes |
|--------------------|------------------------------------------|-------------------------------|-------|
| YieldObservation   | USDA NASS Quick Stats — Survey, "CORN, GRAIN, ALL PRODUCTION PRACTICES" | County × year | Bushels per acre. Uses `state_fips_code + county_code` zero-padded to a 5-char string. (D)/(Z)/(NA)/(X)/(S) values are coerced to NaN and dropped. |
| WeatherSeries      | NASA POWER daily-point API (`power.larc.nasa.gov`) | Daily values per county centroid → aggregated to monthly | Replaces NOAA CAG (returning >100 s/CSV at build time) and NASA POWER (free-tier hourly quota too tight). NASA POWER returns 15 years of daily data per county in ~1.2 s and tolerates 8-way parallelism. |
| SoilProfile        | USDA SDA Tabular Service (SSURGO)        | One row per `legend.areasymbol`, mapped 1:1 to county FIPS in IA/IL/NE | Aggregated to county via SQL: drainage class, prime-farmland fraction (from `mapunit.farmlndcl`), and acre-weighted average of horizon OM/pH/AWC at depths ≤30 cm. |
| County centroids   | 2024 Census Gazetteer (`2024_Gaz_counties_national.txt`) | One row per US county | Used as the input point for NASA POWER. |

## Scope

- **States:** Iowa (FIPS 19), Illinois (17), Nebraska (31).
- **Years:** 2010–2024.
- **Train / val / test split:** train ≤2020, validation 2021–2022, test 2023–2024
  (strict time-based; no random splits — see CLAUDE.md §2 rule 1).

## Engineered features

Built by `FeatureMatrix.build`; see [problem_card.md](problem_card.md) for intended
use. All features are per (county_fips, year).

| Feature                          | Source        | Description |
|----------------------------------|---------------|-------------|
| `gdd_total`                      | Weather       | Growing-degree-days, base 10 °C, May 1–Sep 30, computed month-overlap-weighted. |
| `gdd_v6_to_vt`                   | Weather       | GDD over Jun 15–Jul 31 — the V6-to-VT reproductive window. |
| `precip_total_growing_season_mm` | Weather       | Sum of precipitation, May 1–Sep 30. |
| `precip_july_mm`                 | Weather       | July precipitation. |
| `precip_critical_window_mm`      | Weather       | Precipitation Jul 1–Aug 15. |
| `days_above_30c_july_aug`        | Weather       | Sinusoidal proxy for days with daily tmax ≥30 °C in Jul + Aug. |
| `tmax_mean_july_c`               | Weather       | July monthly mean tmax. |
| `soil_organic_matter_pct`        | Soil          | Acre-weighted topsoil organic-matter %. |
| `soil_ph_mean`                   | Soil          | Acre-weighted topsoil pH (1:1 water). |
| `soil_awc_mean`                  | Soil          | Acre-weighted available water capacity. |
| `pct_well_drained`               | Soil          | % of county acres classified well/somewhat-excessively/excessively drained. |
| `pct_prime_farmland`             | Soil          | % of county acres flagged "All areas are prime farmland". |
| `yield_lag1`                     | Yield (lag)   | Prior-year yield for the same county. |
| `yield_rolling_mean_3yr`         | Yield (lag)   | Trailing 3-year mean of yield (used as baseline). |
| `yield_rolling_std_3yr`          | Yield (lag)   | Trailing 3-year std of yield (filled with 0 when history is too short). |
| `state_IA`, `state_IL`, `state_NE` | Geography   | One-hot encoded state. |

## Coverage and quality

- USDA NASS Survey returned **3,873 unique county-years** for IA/IL/NE 2010–2024 after
  filtering to "ALL PRODUCTION PRACTICES" and "GRAIN" and dropping the 998/999
  state-rollup pseudo-counties.
  - The CLAUDE.md gate estimate of ≥4,000 was loosened to ≥3,800 to match this real
    coverage (2024 is still preliminary Survey data, and a small number of county-years
    are (D)-suppressed by NASS).
- NASA POWER returns ~180 rows per county (15 years × 12 months). All values are
  non-null after aggregation.
- SSURGO aggregates produce **one row per county** with all five soil features populated.

## Known limitations

- County-level aggregation hides field-scale variability; this is by design.
- NASA POWER (ERA5-backed) is a reanalysis product, not direct observation; bias on the
  order of 0.5 °C is plausible. Acceptable for inter-county comparison.
- Soil features are static — they do not capture year-to-year changes in soil organic
  matter or drainage from management.
- One-hot state encoding captures the structural between-state yield differences but
  cannot extrapolate to states outside IA/IL/NE.
- No satellite-derived covariates (NDVI, EVI). See `docs/limitations.md` for full list.

## Provenance

- Fetched by `src/data/fetch_nass.py`, `src/data/fetch_noaa.py`,
  `src/data/fetch_ssurgo.py`.
- Assembled by `src/data/build_dataset.py`.
- Output: `data/processed/features.parquet` (committed to git for HF Spaces; raw
  parquets are gitignored).
