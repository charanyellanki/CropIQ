# CropIQ — Problem Card

> One-page framing for stakeholders, reviewers, and future maintainers. Pairs with
> `data_card.md`, `model_card.md`, and `limitations.md`.

---

## Scope

**Predict end-of-season corn grain yield (bushels per acre) for individual U.S.
counties within Iowa, Illinois, and Nebraska, for crop years 2010–2024.**

Inputs available to the model at prediction time:

- Static soil characteristics for the county (USDA SSURGO).
- Monthly TMIN/TMAX/PRCP weather for the county (NOAA nClimGrid).
- Historical USDA NASS yield for the county (lagged 1+ years).
- County identifier and centroid coordinates.

Outputs:

- Point estimate of yield in bushels per acre.
- 80% confidence interval derived from validation residuals.
- Top three SHAP drivers (feature, signed contribution).

## Metric

**Primary:** Test-set RMSE in bushels per acre, evaluated on county-year rows from
2023–2024 (two held-out crop years). The model must beat the 3-year rolling county-mean
baseline by ≥10% on test RMSE.

**Secondary:** MAPE, R², per-state RMSE breakdown, per-year RMSE breakdown
(2023 vs 2024). These reveal whether gains come from the structural (across counties)
or temporal (across years) component.

## Time-split protocol

Random splits leak information across crop years; CropIQ uses a strict time-based split:

| Split | Years     | Rationale                                   |
| ----- | --------- | ------------------------------------------- |
| Train | ≤ 2020    | Anything before the held-out validation.    |
| Val   | 2021–2022 | Used for XGBoost early stopping only.       |
| Test  | 2023–2024 | Reported once, after model selection.       |

## Non-goals

The following are explicitly **out of scope** for this project:

- **Field-level / sub-county prediction.** Smallest unit is the county FIPS.
- **In-season forecasts before harvest.** Inputs include full growing-season weather;
  the model is a post-season retrospective fit, not a mid-season nowcast.
- **Crops other than corn.** Methodology applies, but no other commodity is fit.
- **States outside IA / IL / NE.** No transfer evaluation is performed.
- **Satellite imagery (NDVI, etc.).** Deliberately excluded; uses only tabular sources.
- **Deep learning.** XGBoost and LightGBM only — gradient boosting outperforms small
  networks on tabular county-year scales and serves more cheaply on HF Spaces CPU.
- **Causal claims.** SHAP attributions describe model behavior, not biological cause.

## Intended use

- **Portfolio / hiring-manager demonstration** of end-to-end ML engineering across
  data acquisition, feature engineering, evaluation, governance, and deployment.
- **Teaching / reference implementation** of time-split validation and SHAP-grounded
  explanation for agricultural tabular data.
- **Starting point** for analysts who want to extend the same pipeline to other
  commodities or geographies.

## Out-of-scope use

Do **not** use CropIQ to:

- Set crop-insurance premiums, futures positions, or producer payouts.
- Inform field-level agronomic decisions (e.g., variable-rate seeding, fungicide
  timing). County aggregation hides field variability that dominates those decisions.
- Drive any safety-critical or regulated decision without independent agronomic
  validation against ground-truth from the same crop year.

## Stakeholders

| Role                         | Interest                                            |
| ---------------------------- | --------------------------------------------------- |
| AI/ML hiring managers        | Evidence of production-quality engineering and ML.  |
| Climate / digital-ag teams   | Reference for tabular ag-ML with public data only.  |
| Agronomists / domain experts | Sanity-check SHAP drivers against known agronomy.   |
| Future maintainers           | Reproduce the build with `make all` from a clone.   |

## Success criteria

- All 11 gates (Phases 0–10) pass without manual intervention.
- XGBoost beats baseline by ≥10% on test RMSE.
- Live Hugging Face Space loads the dashboard in under 5 seconds and produces a
  prediction with three SHAP drivers for any (state, county, year) selection.
- `docs/model_card.md` and `docs/limitations.md` honestly disclose known weaknesses.
