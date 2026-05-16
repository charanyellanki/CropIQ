# CropIQ — Limitations

What CropIQ **doesn't** see, can't represent, or cannot transfer to. Read this before
using the model for anything beyond a portfolio demonstration.

---

## Granularity

- **County is the smallest unit.** Sub-county variation in soil, planting date,
  hybrid, irrigation, and pest pressure is invisible to the model.
- **Annual cadence.** No intra-season output; the model does not nowcast yields
  during the growing season.

## Data scope

- **Three states only.** IA, IL, NE. Performance outside this geography is not
  evaluated and should not be assumed.
- **Fifteen years.** 2010–2024. Long-term trend extrapolation is unreliable.
- **No satellite imagery.** NDVI, EVI, SIF, or radar-derived metrics could
  substantially improve sub-county and in-season inference, but are explicitly
  out of scope.

## Inputs the model does not have

- **Hybrid / variety information.** Genetic gain over the period is absorbed into
  the year-linked `yield_lag1` / rolling-mean features rather than captured directly.
- **Management practices.** Planting date, fertilizer N rate, tillage, cover-crop
  status, fungicide passes, irrigation scheduling — all unobserved.
- **Pest / disease pressure.** Tar spot, gray leaf spot, corn rootworm — not modeled.
- **Economic factors.** Commodity prices, crop insurance rules, market signals do not
  feed back into the agronomic features.
- **Field heterogeneity within county.** Soil features are area-weighted averages;
  the variance across map units inside a county is collapsed.

## Coverage gaps in the training data

- **(D)-suppressed USDA NASS rows** are dropped, so counties with confidential
  acreage in a given year are silently absent.
- **2024 is preliminary Survey data**, with fewer counties reporting than 2010–2023.
- **NASA POWER reanalysis bias** of ~0.5 °C is plausible. Acceptable for relative
  comparison; not a substitute for in-situ station data.

## Modeling caveats

- **Time-based train/val/test split** prevents look-ahead leakage, but the model is
  evaluated on only **two test years (2023, 2024)**. Decadal climate trends are not
  exercised.
- **SHAP attributions explain model behavior**, not biology. A negative SHAP
  contribution from July heat is consistent with known corn heat stress, but the
  model has no causal access to physiology.
- **Tree models extrapolate poorly.** Predictions for inputs outside the training
  envelope (e.g., a county-year with weather more extreme than anything in
  2010–2022) will revert toward training-distribution means.

## Deployment caveats

- The Hugging Face Space runs on free CPU. First load can take a few seconds; SHAP
  inference for one row is sub-second after the engine is warm. The model is loaded
  once via `@st.cache_resource` (CLAUDE.md gotcha #10).
- The FastAPI service is **not deployed** to HF Spaces. It exists for local
  development and contract testing.
- The Space uses the in-process `PredictionEngine`, never HTTP, because HF Spaces
  runs a single Streamlit process.

## Appropriate uses

✅ Portfolio / hiring-manager demonstration
✅ Teaching example of time-split CV + SHAP explanations
✅ Starting point for a more ambitious ag-ML pipeline

## Inappropriate uses

❌ Field-level agronomic decisions (variable-rate seeding, fungicide timing)
❌ Setting crop-insurance premiums or producer payouts
❌ Driving futures positions or hedging strategies
❌ Any safety-critical, regulatory, or financial decision without independent
   agronomic validation against ground-truth from the same crop year
