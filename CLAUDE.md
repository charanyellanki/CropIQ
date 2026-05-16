# CLAUDE.md — CropIQ Project Configuration

> **You are building CropIQ end-to-end in auto-build mode.** Read this entire document before starting. Do not deviate from the rules. Self-validate at each gate. Only stop and ask the user if a gate fails after 2 retry attempts.

---

## 1. PROJECT IDENTITY

**Name:** CropIQ
**Purpose:** County-level corn yield forecasting system for Iowa, Illinois, and Nebraska (2010–2024) using USDA NASS yield data, NOAA weather data, and USDA SSURGO soil data. Delivers a Streamlit dashboard (deployed to Hugging Face Spaces), an optional local FastAPI service, and SHAP-grounded explanations.
**Audience:** AI/ML hiring managers at Bayer Climate LLC, Corteva Digital, John Deere (Blue River AI), Syngenta Cropwise.
**Quality bar:** This must look like production engineering work, not a Kaggle notebook. Real domain insight, real evaluation, real serving.
**Deployment target:** Hugging Face Spaces (Streamlit SDK). Public URL must load in <5 seconds.

---

## 2. HARD RULES — NON-NEGOTIABLE

### MUST
1. **Time-based cross-validation only.** Train ≤2020, validation 2021–2022, test 2023–2024. NEVER random split.
2. **Baseline first.** County 3-year rolling mean yield. Every ML model is reported alongside this baseline. Final model must beat baseline test RMSE by ≥10%.
3. **MLflow tracking for every model run.** Params, metrics, artifacts, model signature.
4. **Type hints on every function.** Google-style docstrings on every public function.
5. **County FIPS as 5-character STRING with leading zeros.** Never int. This is the #1 ag data bug.
6. **Reproducibility:** set seeds in numpy, python random, xgboost, lightgbm. Pin all dependencies.
7. **All scripts runnable as `python -m src.module.script`** with absolute imports.
8. **Self-validation gates after each phase** (see §13). Auto-retry up to 2x on gate failure before stopping.
9. **Streamlit dashboard calls `PredictionEngine` in-process, not via HTTP.** The FastAPI service is local-only and exists for portfolio signal + local development.
10. **Repo root must have `app.py`** that imports and runs the Streamlit dashboard — Hugging Face Spaces requires this.

### MUST NOT
1. **Do not generate synthetic data when an API call fails.** Fix the call. Synthetic data invalidates the project.
2. **Do not use deep learning.** XGBoost and LightGBM only. Skip Optuna if it adds >30 min — defaults are fine.
3. **Do not pull satellite imagery.** Out of scope.
4. **Do not commit:** raw data, mlruns/, .env. Use .gitignore.
5. **DO commit `models/best_model.pkl` and `data/processed/features.parquet`** — needed by the deployed Space. Use `.gitignore` exceptions (§14 row 31).
6. **Do not use pandas chained assignment.** Use `.loc[]`.
7. **Do not silently swallow exceptions.** Log and re-raise or fail loudly.
8. **Do not use Pydantic v1 syntax.** This project uses Pydantic v2.
9. **Do not load the model inside the request handler.** Load once at startup (FastAPI lifespan; Streamlit `@st.cache_resource`).
10. **Do not have the Streamlit app make HTTP calls to FastAPI when deployed.** HF Spaces runs one process — Streamlit and FastAPI cannot coexist on a single Space.

---

## 3. TECH STACK — PINNED VERSIONS

These versions are tested compatible and known to build on Hugging Face Spaces. Do not change without explicit user approval.

```
python = "3.11"

# Core
pandas = "2.2.3"
numpy = "1.26.4"          # NOT 2.x — breaks several downstream libs
pyarrow = "17.0.0"
polars = "1.12.0"
pydantic = "2.9.2"
pydantic-settings = "2.6.1"
python-dotenv = "1.0.1"
loguru = "0.7.2"
pyyaml = "6.0.2"

# Modeling
scikit-learn = "1.5.2"
xgboost = "2.1.2"
lightgbm = "4.5.0"
shap = "0.46.0"
mlflow = "2.17.2"

# Data acquisition
requests = "2.32.3"
tenacity = "9.0.0"

# Serving (local-only)
fastapi = "0.115.4"
uvicorn = {extras = ["standard"], version = "0.32.0"}
httpx = "0.27.2"           # used only in tests via FastAPI TestClient

# Dashboard (deployed)
streamlit = "1.40.1"
plotly = "5.24.1"

# Dev
pytest = "8.3.3"
pytest-cov = "5.0.0"
ruff = "0.7.4"
ipykernel = "6.29.5"
```

Use `requirements.txt` — required by Hugging Face Spaces.

---

## 4. METHODOLOGY — AI DEVELOPMENT LIFE CYCLE (AIDLC)

The project follows the eight-phase AIDLC plus two deployment phases. Each phase has explicit inputs, outputs, and exit criteria.

| Phase | Name | Output Artifact | Exit Gate |
|-------|------|-----------------|-----------|
| 0 | Setup | repo scaffold + deps installed | `make install` passes; config loads |
| 1 | Problem Framing | `docs/problem_card.md` | Card exists, covers scope/metric/non-goals |
| 2 | Data Acquisition | `data/raw/*.parquet` | All 3 sources fetched, row counts validated |
| 3 | Data Understanding | `notebooks/01_eda.ipynb`, `docs/data_card.md` | EDA notebook executes top-to-bottom |
| 4 | Data Preparation | `data/processed/features.parquet` | Schema validated, no NaN in required cols |
| 5 | Modeling | `models/best_model.pkl`, `mlruns/` | XGBoost beats baseline by ≥10% on test RMSE |
| 6 | Evaluation | `reports/results.md`, `reports/figures/*.png` | SHAP plots, per-state breakdown, agronomic insight |
| 7 | Local Deployment | `src/inference/`, `src/api/`, `src/dashboard/` | Streamlit runs locally; FastAPI contract tests pass |
| 8 | Governance | `docs/model_card.md`, `docs/limitations.md` | Cards complete |
| 9 | README + Polish | `README.md` with HF frontmatter + screenshots | All sections complete |
| 10 | HF Spaces Deploy | Live public URL | Space builds; dashboard loads; prediction works |

**Gates are automated assertions, not user check-ins.** See §13.

---

## 5. CONCEPT-CENTRIC ARCHITECTURE (Daniel Jackson, adapted for ML)

The codebase is organized around **concepts** — independent, reusable units with a clear purpose, state, and actions. Each concept lives in its own module. Concepts compose via explicit **synchronizations** (§6).

Every module must have a header docstring matching this template:

```python
"""
Concept: <ConceptName>
Purpose: <one sentence — why this concept exists>
State: <what it holds>
Actions: <verbs it supports>
Operational principle: <one concrete example of fulfillment>
"""
```

### 5.1 Domain Concepts

#### Concept: `YieldObservation`
- **Purpose:** Record an observed annual yield for a county-commodity pair.
- **State:** `county_fips: str`, `year: int`, `commodity: str`, `yield_bu_per_acre: float`, `source: str`
- **Actions:** `fetch_from_nass(state, year_range)`, `validate(row)`, `to_dataframe()`
- **Operational principle:** Fetching corn yield for IA 2023 returns ~99 rows (one per Iowa county) with non-null yields.
- **Module:** `src/concepts/yield_observation.py`

#### Concept: `WeatherSeries`
- **Purpose:** Provide weather measurements aggregated to county-month.
- **State:** `county_fips`, `year`, `month`, `tmin_c`, `tmax_c`, `prcp_mm`
- **Actions:** `fetch_from_nclimgrid(state, year_range)`, `aggregate_to_county()`, `to_dataframe()`
- **Operational principle:** Fetching IA 2023 returns 12 monthly rows per county with all weather variables populated.
- **Module:** `src/concepts/weather_series.py`

#### Concept: `SoilProfile`
- **Purpose:** Provide static soil characteristics for a county.
- **State:** `county_fips`, `organic_matter_pct`, `ph_mean`, `awc_mean`, `pct_well_drained`, `pct_prime_farmland`
- **Actions:** `load_ssurgo(state)`, `aggregate_to_county()`, `to_dataframe()`
- **Operational principle:** Loading IA SSURGO produces one row per county with all five soil features populated.
- **Module:** `src/concepts/soil_profile.py`

#### Concept: `GrowingSeason`
- **Purpose:** Define temporal windows for corn development stages.
- **State:** `commodity`, `stage_name`, `start_doy`, `end_doy`, `base_temp_c`
- **Actions:** `get_stage_window(stage)`, `compute_gdd(weather, stage)`, `compute_precip(weather, stage)`
- **Operational principle:** `compute_gdd(weather, "v6_to_vt")` returns a single GDD value for the critical reproductive window.
- **Module:** `src/concepts/growing_season.py`

### 5.2 ML Concepts

#### Concept: `FeatureMatrix`
- **Purpose:** Produce a model-ready feature matrix by joining domain concepts on (county_fips, year).
- **State:** Rows of `(county_fips, year, target, *features)`
- **Actions:** `build(yields, weather, soil, seasons)`, `validate_schema()`, `save(path)`, `load(path)`
- **Operational principle:** Building for IA/IL/NE 2010–2024 produces ~5K rows with 20+ columns and zero NaN in required columns.
- **Module:** `src/concepts/feature_matrix.py`

#### Concept: `YieldPredictor`
- **Purpose:** Produce a yield estimate from a feature vector.
- **State:** `model`, `feature_schema`, `training_metadata` (run_id, training_date, version, residual_std)
- **Actions:** `train(X_train, y_train, X_val, y_val)`, `predict(X)`, `save(path)`, `load(path)`
- **Operational principle:** A trained predictor returns 5K predictions on the test set with RMSE below the baseline.
- **Module:** `src/concepts/yield_predictor.py`
- **Implementations:** `BaselinePredictor`, `XGBoostPredictor`, `LightGBMPredictor`

#### Concept: `Explanation`
- **Purpose:** Attribute a single prediction to feature contributions.
- **State:** `prediction_value`, `base_value`, `feature_contributions: Dict[str, float]`
- **Actions:** `explain_instance(predictor, x)`, `explain_global(predictor, X)`, `to_dict()`, `top_k(k)`
- **Operational principle:** Explaining a Story County 2023 prediction returns the top-3 drivers with signed SHAP contributions.
- **Module:** `src/concepts/explanation.py`

#### Concept: `Experiment`
- **Purpose:** Track a training run with parameters, metrics, artifacts.
- **State:** `run_id`, `params`, `metrics`, `artifacts`
- **Actions:** `start(name)`, `log_param`, `log_metric`, `log_artifact`, `end()`
- **Operational principle:** Wrapping an XGBoost training run records all hyperparameters and metrics to MLflow under a named run.
- **Module:** `src/concepts/experiment.py`
- **Backed by:** MLflow

#### Concept: `Evaluation`
- **Purpose:** Measure predictor quality on held-out data.
- **State:** `predictions`, `actuals`, `metrics: Dict[str, float]`
- **Actions:** `compute(preds, actuals)`, `breakdown_by(group_col)`, `plot_predicted_vs_actual()`, `plot_residuals()`
- **Operational principle:** Evaluating XGBoost test predictions returns RMSE, MAPE, R² overall and per-state breakdown.
- **Module:** `src/concepts/evaluation.py`

### 5.3 Inference & Service Concepts

#### Concept: `PredictionEngine` ⭐ *(shared by FastAPI and Streamlit)*
- **Purpose:** Produce a yield prediction + explanation for a `(state, county_fips, year)` input. Pure logic, no transport layer.
- **State:** Loaded `YieldPredictor`, loaded SHAP `TreeExplainer`, loaded `FeatureMatrix` lookup, residual_std for CI, model metadata
- **Actions:** `predict(state, county_fips, year) -> PredictionResult`, `list_counties(state) -> List[County]`, `get_metadata() -> ModelMetadata`
- **Operational principle:** `engine.predict("IA", "19169", 2024)` returns predicted yield, 80% CI, top-3 SHAP drivers, model version — in <100ms on CPU.
- **Module:** `src/inference/prediction_engine.py`
- **This is the single source of truth for inference.** Both `PredictionService` (HTTP) and `DashboardView` (Streamlit) delegate to it.

#### Concept: `PredictionService` *(local-only HTTP adapter)*
- **Purpose:** Expose `PredictionEngine` over HTTP for local development and API contract testing.
- **State:** Reference to `PredictionEngine` loaded at startup
- **Actions:** `GET /predict/{state}/{county_fips}/{year}`, `GET /health`, `GET /counties/{state}`
- **Operational principle:** GET `/predict/IA/19169/2024` calls `PredictionEngine.predict(...)` and returns JSON.
- **Module:** `src/api/`
- **NOT deployed.** Exists for: (a) local development, (b) demonstrating API contract design, (c) reusability in hypothetical microservices deployment.

#### Concept: `DashboardView`
- **Purpose:** Present predictions visually, calling `PredictionEngine` in-process.
- **State:** Cached `PredictionEngine` via `@st.cache_resource`; selected state/year/county
- **Actions:** `render_map(state, year)`, `render_county_detail(state, county_fips, year)`
- **Operational principle:** Selecting IA/2024 renders a choropleth; clicking Story County opens a detail panel with prediction + drivers.
- **Module:** `src/dashboard/app.py`
- **Deployment:** This is the only inference path live on Hugging Face Spaces.

---

## 6. CONCEPT SYNCHRONIZATIONS

| When | Then | Where |
|------|------|-------|
| `FeatureMatrix.build` is called | `YieldObservation`, `WeatherSeries`, `SoilProfile` queried; `GrowingSeason.compute_gdd` and `compute_precip` applied per row | `src/concepts/feature_matrix.py` |
| `YieldPredictor.train` is called | `Experiment.start` opens an MLflow run; metrics via `log_metric`; model via `log_artifact` | `src/concepts/yield_predictor.py` |
| `PredictionEngine.predict` is called | `YieldPredictor.predict` produces value; `Explanation.explain_instance` computes drivers; both merged into `PredictionResult` | `src/inference/prediction_engine.py` |
| `PredictionService` GET `/predict` | `PredictionEngine.predict` invoked; result serialized via Pydantic v2 | `src/api/main.py` |
| `DashboardView.render_county_detail` | `PredictionEngine.predict` invoked in-process; result unpacked into Streamlit components | `src/dashboard/app.py` |

**No concept may import another's internal state directly. Communicate via public actions only.**

---

## 7. REPOSITORY STRUCTURE

```
cropiq/
├── CLAUDE.md                       # this file
├── README.md                       # with HF Spaces YAML frontmatter (§16)
├── app.py                          # HF Spaces entry point — imports src.dashboard.app
├── requirements.txt
├── Makefile
├── .gitignore                      # with exceptions for model + features parquet
├── .env.example
├── pyproject.toml
├── config/
│   └── config.yaml
├── docs/
│   ├── problem_card.md
│   ├── data_card.md
│   ├── model_card.md
│   ├── limitations.md
│   └── deploy_hf.md                # HF Spaces deployment guide
├── data/
│   ├── raw/                        # gitignored
│   ├── interim/                    # gitignored
│   └── processed/
│       └── features.parquet        # COMMITTED — needed by deployed dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_setup.py
│   ├── paths.py
│   ├── concepts/
│   │   ├── __init__.py
│   │   ├── yield_observation.py
│   │   ├── weather_series.py
│   │   ├── soil_profile.py
│   │   ├── growing_season.py
│   │   ├── feature_matrix.py
│   │   ├── yield_predictor.py
│   │   ├── explanation.py
│   │   ├── experiment.py
│   │   └── evaluation.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_nass.py
│   │   ├── fetch_noaa.py
│   │   ├── fetch_ssurgo.py
│   │   └── build_dataset.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── prediction_engine.py    # ⭐ shared by FastAPI and Streamlit
│   ├── train.py
│   ├── api/                        # local-only
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── dependencies.py         # lifespan loader
│   └── dashboard/
│       └── app.py                  # imports PredictionEngine directly
├── tests/
│   ├── conftest.py
│   ├── test_concepts_*.py
│   ├── test_prediction_engine.py
│   └── test_api.py
├── models/                         # gitignored EXCEPT best_model.pkl
│   └── best_model.pkl              # COMMITTED for HF deploy
├── mlruns/                         # gitignored
├── scripts/
│   ├── run_gate.py
│   └── precompute_predictions.py   # optional: cache test-year predictions
└── reports/
    ├── figures/
    └── results.md
```

---

## 8. DATA SOURCES — EXACT SPECIFICATIONS

### 8.1 USDA NASS Quick Stats (Yield)
- **API base:** `https://quickstats.nass.usda.gov/api/api_GET/`
- **Auth:** API key as `key=` query param. Free signup at `https://quickstats.nass.usda.gov/api`.
- **Query for corn yield:**
  ```
  source_desc=SURVEY
  sector_desc=CROPS
  commodity_desc=CORN
  statisticcat_desc=YIELD
  agg_level_desc=COUNTY
  state_alpha=IA (then IL, then NE)
  year__GE=2010
  year__LE=2024
  unit_desc=BU / ACRE
  ```
- Concatenate `state_fips_code` (2 chars) + `county_code` (3 chars) → 5-char `county_fips`.
- **Expected row count:** ~5,000.
- **Rate limit:** 50,000 cells per query — chunk by state.

### 8.2 NOAA nClimGrid (Weather) — PREFERRED
- **Source:** `https://www.ncei.noaa.gov/access/monitoring/nclimgrid-monthly/` — county-level monthly TMIN/TMAX/PRCP.
- **Why preferred:** No API quota, county-aggregated, simpler than GHCN-Daily.
- **Fallback:** If unavailable, use GHCN-Daily via NOAA CDO API with `tenacity` exponential backoff.

### 8.3 USDA SSURGO (Soil)
- **PREFERRED:** STATSGO2 county-level summaries — `https://www.nrcs.usda.gov/resources/data-and-reports/soil-survey-geographic-database-ssurgo`
- **Fallback:** gSSURGO state-level `muaggatt` table. DO NOT download the full geodatabase.
- **Required fields:** `om_r`, `ph1to1h2o_r`, `awc_r`, `drclassdcd`, `niccdcd`.

---

## 9. FEATURE ENGINEERING SPECIFICATION

### Weather (per county-year)
- `gdd_total` — base 50°F, May 1 – Sep 30
- `gdd_v6_to_vt` — base 50°F, June 15 – July 31
- `precip_total_growing_season_mm`
- `precip_july_mm`
- `precip_critical_window_mm` — July 1 – Aug 15
- `days_above_30c_july_aug`
- `tmax_mean_july_c`

### Soil (per county, static)
- `soil_organic_matter_pct`, `soil_ph_mean`, `soil_awc_mean`, `pct_well_drained`, `pct_prime_farmland`

### Temporal (per county-year)
- `yield_lag1`, `yield_rolling_mean_3yr`, `yield_rolling_std_3yr`

### Geographic
- `state` (one-hot encoded BEFORE training), `lat_centroid`, `lon_centroid`

**Target:** `yield_bu_per_acre`

---

## 10. MODELING SPECIFICATION

### Baseline
`BaselinePredictor`: predicts trailing 3-year county mean. MLflow run: `baseline_rolling_mean`.

### XGBoost (primary)
```
objective="reg:squarederror", n_estimators=1000, learning_rate=0.05,
max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=50, random_state=42
```
Train ≤2020, validate 2021–2022 (early stopping), test 2023–2024.

### LightGBM (comparison)
Equivalent hyperparameters. `num_leaves=31`.

### Model selection
Lowest test RMSE that beats baseline by ≥10%. Save to `models/best_model.pkl` with: model, feature schema, residual_std, training metadata, sklearn/xgboost versions.

---

## 11. EVALUATION SPECIFICATION

`reports/results.md` must include:
1. Summary table: Model | Val RMSE | Test RMSE | Test MAPE | Test R² | vs Baseline
2. Per-state test RMSE breakdown
3. Per-year test RMSE breakdown (2023 vs 2024)
4. Pred vs actual scatter → `reports/figures/pred_vs_actual.png`
5. Residual histogram → `reports/figures/residuals.png`
6. SHAP global summary → `reports/figures/shap_summary.png`
7. Five SHAP waterfalls → `reports/figures/shap_waterfall_*.png`
8. **Agronomic Insight paragraph** — top 3 SHAP drivers connected to known corn agronomy.

---

## 12. SERVICE SPECIFICATIONS

### 12.1 PredictionEngine (the shared core)

`src/inference/prediction_engine.py`:

```python
class PredictionEngine:
    def __init__(self, model_path: Path, features_path: Path):
        # Load YieldPredictor with metadata (model, feature_schema, residual_std)
        # Load features parquet for input lookup
        # Initialize SHAP TreeExplainer ONCE (cache for reuse)

    def predict(self, state: str, county_fips: str, year: int) -> PredictionResult:
        # Look up feature row, validate non-null, run model, run SHAP, return

    def list_counties(self, state: str) -> List[County]: ...
    def get_metadata(self) -> ModelMetadata: ...
```

`PredictionResult` is a Pydantic v2 model:
```python
class PredictionResult(BaseModel):
    state: str
    county_fips: str
    county_name: str
    year: int
    predicted_yield_bu_per_acre: float
    confidence_interval_80pct: tuple[float, float]
    top_drivers: list[Driver]
    model_version: str
```

CI: `prediction ± 1.28 * residual_std` (residual_std baked into model metadata at training time).

### 12.2 FastAPI Service (local-only)

Endpoints, all delegating to `PredictionEngine`:
- `GET /health` → `{"status": "ok", **engine.get_metadata().model_dump()}`
- `GET /counties/{state}` → list of counties
- `GET /predict/{state}/{county_fips}/{year}` → `PredictionResult` as JSON

Use FastAPI `lifespan` (NOT `@app.on_event`) to load `PredictionEngine` once at startup → `app.state.engine`. CORS allowing `http://localhost:*`. Pydantic v2 schemas.

**This service is NOT deployed to HF Spaces.**

### 12.3 Streamlit Dashboard (deployed)

Root `app.py`:
```python
"""HF Spaces entry point."""
from src.dashboard.app import main

if __name__ == "__main__":
    main()
```

`src/dashboard/app.py`:
```python
import streamlit as st
from pathlib import Path
from src.inference.prediction_engine import PredictionEngine

@st.cache_resource
def load_engine() -> PredictionEngine:
    return PredictionEngine(
        model_path=Path("models/best_model.pkl"),
        features_path=Path("data/processed/features.parquet"),
    )

@st.cache_data(ttl=3600)
def cached_predict(state: str, county_fips: str, year: int):
    engine = load_engine()
    return engine.predict(state, county_fips, year).model_dump()

def main():
    st.set_page_config(page_title="CropIQ", page_icon="🌽", layout="wide")
    engine = load_engine()
    # ... UI code ...
```

Page layout:
- Sidebar: state (IA/IL/NE), year (2023/2024), county selector
- Main: Plotly choropleth (filter geojson to IA/IL/NE, cached); county detail card with prediction + CI + top-3 SHAP drivers as horizontal bars; predicted-vs-actual table for selected state
- Use geojson from `https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json`, filtered and cached locally to `data/processed/counties_geo.json`

---

## 13. SELF-VALIDATION GATES

Implement `scripts/run_gate.py <phase_num>`. Each gate runs its phase's assertions. On fail, retry phase once. If still failing, stop and report.

```python
# Gate 0: Setup
from src.config import settings  # must import without error
assert (PROJECT_ROOT / "requirements.txt").exists()
assert (PROJECT_ROOT / "app.py").exists()  # HF Spaces requires this

# Gate 2: Data Acquisition
nass = pd.read_parquet("data/raw/nass_yields.parquet")
assert len(nass) >= 4000
assert nass["county_fips"].dtype == "object"
assert nass["county_fips"].str.len().eq(5).all()
assert nass["yield_bu_per_acre"].notna().mean() > 0.95

# Gate 4: Data Preparation
features = pd.read_parquet("data/processed/features.parquet")
required = ["county_fips", "year", "yield_bu_per_acre", "gdd_total",
            "precip_july_mm", "soil_organic_matter_pct", "yield_lag1"]
for col in required:
    assert col in features.columns
    assert features[col].notna().mean() > 0.95

# Gate 5: Modeling
runs = mlflow.search_runs()
xgb_rmse = runs.query("`tags.mlflow.runName` == 'xgboost_v1'")["metrics.test_rmse"].iloc[0]
base_rmse = runs.query("`tags.mlflow.runName` == 'baseline_rolling_mean'")["metrics.test_rmse"].iloc[0]
assert xgb_rmse < base_rmse * 0.9
assert Path("models/best_model.pkl").exists()

# Gate 6: Evaluation
assert Path("reports/figures/shap_summary.png").exists()
assert Path("reports/figures/pred_vs_actual.png").exists()
assert len(list(Path("reports/figures").glob("shap_waterfall_*.png"))) >= 5

# Gate 7: Local Deployment
from src.inference.prediction_engine import PredictionEngine
engine = PredictionEngine(Path("models/best_model.pkl"),
                          Path("data/processed/features.parquet"))
result = engine.predict("IA", "19169", 2024)
assert 50 < result.predicted_yield_bu_per_acre < 300
assert len(result.top_drivers) == 3

from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)
assert client.get("/health").status_code == 200
assert client.get("/predict/IA/19169/2024").status_code == 200

# Gate 9: README + git tracking
readme = Path("README.md").read_text()
assert readme.startswith("---\ntitle: CropIQ"), "Missing HF Spaces frontmatter"
assert "sdk: streamlit" in readme
assert "app_file: app.py" in readme
# Confirm runtime files are tracked (not gitignored)
import subprocess
def is_tracked(path):
    r = subprocess.run(["git", "ls-files", path], capture_output=True, text=True)
    return r.stdout.strip() != ""
assert is_tracked("models/best_model.pkl"), "Model not committed — HF Space will fail"
assert is_tracked("data/processed/features.parquet"), "Features parquet not committed"

# Gate 10: HF Spaces Deploy (smoke test)
import app as deployed_app  # root app.py must import without error
# User-confirmed: live URL loads dashboard
```

---

## 14. COMMON GOTCHAS — PRE-EMPTIVE FIXES

| # | Gotcha | Pre-emptive fix |
|---|--------|-----------------|
| 1 | County FIPS as int → loses leading zeros | Always `dtype=str`; `.zfill(5)` after numeric ops |
| 2 | NOAA API rate limit returns empty | `tenacity` exponential backoff; assert non-empty |
| 3 | NASS returns `"(D)"` for suppressed values | `pd.to_numeric(..., errors='coerce')` |
| 4 | Pandas chained assignment warning | Use `.loc[mask, col] = value` always |
| 5 | XGBoost wrong column order at inference | Save with model: `joblib.dump({"model": m, "features": list(X.columns), "residual_std": s, "version": v}, path)` |
| 6 | Random split contaminates time-series | Sort by year; split on year boundary; never `train_test_split` |
| 7 | SHAP OOM on full test set | `shap.sample(X_test, 200)` for global; full set only for 5 waterfalls |
| 8 | FastAPI reloads model per request | `lifespan` context manager → `app.state.engine` |
| 9 | Pydantic v1 syntax in v2 | Use `@field_validator`, `model_config = ConfigDict(...)` |
| 10 | Streamlit re-runs whole script | `@st.cache_resource` for engine; `@st.cache_data` for predictions |
| 11 | Plotly choropleth FIPS type mismatch | Cast both sides to 5-char string before merge |
| 12 | Parquet errors with mixed-type object dtype | Cast columns explicitly before `to_parquet` |
| 13 | Relative paths break across dirs | `from src.paths import PROJECT_ROOT` everywhere |
| 14 | `.env` not loaded — dotenv called too late | Load in `src/config.py` at module import |
| 15 | MLflow URI confused with relative paths | `mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")` |
| 16 | XGBoost rejects string `state` column | One-hot encode BEFORE training |
| 17 | Train/test feature mismatch from one-hot | `pd.get_dummies` on FULL dataset before split |
| 18 | Joblib version mismatch on load | Pin joblib; include sklearn/xgboost version in model metadata |
| 19 | County name mismatches | Join on `county_fips`, never name |
| 20 | Year as object dtype | Cast to `int` after load |
| 21 | Streamlit choropleth slow on full US geojson | Filter to IA/IL/NE once, cache to `data/processed/counties_geo.json` |
| 22 | Long fetch with no progress | `tqdm` on loops > 5 iter; `loguru.info` at milestones |
| 23 | Notebook can't import from `src.` | First cell: `import sys; sys.path.insert(0, str(Path.cwd().parent))` |
| 24 | Tests using real API → slow/flaky | `requests-mock` fixtures for all external HTTP |
| 25 | USDA CSV encoding errors | `encoding="latin-1"` |
| 26 | Date columns parsed as object | Pass `parse_dates=[...]` to readers |
| 27 | NumPy 2.x breaks XGBoost/SHAP | Pinned to 1.26.x |
| 28 | geopandas fails on GDAL | Optional; use static centroid lookup CSV |
| 29 | FastAPI TestClient requires httpx | httpx pinned |
| 30 | NaN in prediction inputs → silent zero | Validate input in `PredictionEngine.predict`; raise on NaN |
| **31** | **HF Spaces needs model committed but gitignore blocks it** | `.gitignore`: `models/*` then `!models/best_model.pkl`. Same for `data/processed/*` and `!data/processed/features.parquet` |
| **32** | **HF Spaces missing YAML frontmatter → Space won't start** | First 9 lines of README.md are YAML frontmatter per §16 |
| **33** | **HF Spaces can't find app — needs `app.py` at REPO ROOT** | Root `app.py`: `from src.dashboard.app import main; main()` |
| **34** | **HF Spaces secrets via env vars, not `.env`** | CropIQ uses no runtime secrets — all data is baked in at build. If ever needed, use Space Settings UI; read via `os.environ.get(...)` |
| **35** | **HF Spaces uses port 7860, not 8501** | Don't hardcode port — HF launches Streamlit correctly via SDK |
| **36** | **HF Spaces CPU is weak; SHAP per-request slow** | Initialize `shap.TreeExplainer` once at startup via `@st.cache_resource`; reuse |
| **37** | **HF Spaces build fails on version conflicts** | All §3 versions tested compatible — do not change |
| **38** | **HF Spaces large file via regular git fails** | XGBoost model on this data is <20MB — commit directly. If >10MB, use `git lfs track "*.pkl"` |
| **39** | **`@st.cache_resource` returns stale engine across restarts** | OK — Space restart on new push handles it |
| **40** | **HF Spaces logs limited — debug locally first** | Always run `streamlit run app.py` locally end-to-end before pushing |
| **41** | **HF Spaces Streamlit version drift from local** | Set `sdk_version` in README YAML to match `requirements.txt` exactly |
| **42** | **First load on a quiet Space is slow** | Document this in README — normal, subsequent requests are fast |

---

## 15. EXECUTION PLAN — AUTO-BUILD MODE

Execute phases sequentially. Run gate after each phase. Retry once on failure; stop and report on second failure.

### Phase 0 — Setup
- Create directory structure per §7 (including `app.py` at root, `src/inference/`, `scripts/`)
- Write `requirements.txt`, `Makefile` (targets: `install`, `data`, `train`, `serve`, `dashboard`, `test`, `gate`, `clean`, `all`)
- Write `.gitignore` with **exceptions for `!models/best_model.pkl` and `!data/processed/features.parquet`** (pre-emptive fix #31)
- Write `.env.example`, `pyproject.toml`
- Write `src/config.py`, `src/paths.py`, `src/logging_setup.py`, `config/config.yaml`
- Write stub `app.py` at root (will be filled in Phase 7)
- **Gate 0**

### Phase 1 — Problem Framing
- `docs/problem_card.md`: scope, metric, non-goals, intended use, out-of-scope use, stakeholders
- **Gate 1**

### Phase 2 — Data Acquisition
- Implement `src/concepts/yield_observation.py`, `weather_series.py`, `soil_profile.py`
- Implement `src/data/fetch_*.py` CLI scripts
- Apply fixes #1, #2, #3, #25
- **Gate 2**

### Phase 3 — Data Understanding
- `notebooks/01_eda.ipynb`, `docs/data_card.md`
- **Gate 3**

### Phase 4 — Data Preparation
- Implement `src/concepts/growing_season.py`, `feature_matrix.py`
- Implement `src/data/build_dataset.py`
- Apply fixes #11, #12, #17, #20
- **Gate 4**

### Phase 5 — Modeling
- Implement `src/concepts/yield_predictor.py`, `experiment.py`
- Implement `src/train.py` (baseline → XGBoost → LightGBM, all MLflow-logged)
- Apply fixes #5, #6, #15, #16, #18
- Save `models/best_model.pkl` with full metadata
- **Gate 5**

### Phase 6 — Evaluation
- Implement `src/concepts/evaluation.py`, `explanation.py`
- `notebooks/02_model_analysis.ipynb`, figures per §11, `reports/results.md` with Agronomic Insight
- Apply fix #7
- **Gate 6**

### Phase 7 — Local Deployment
- Implement `src/inference/prediction_engine.py` (shared concept — apply fixes #30, #36)
- Implement `src/api/` (FastAPI thin adapter over PredictionEngine — apply fix #8)
- Implement `src/dashboard/app.py` (Streamlit calling PredictionEngine — apply fixes #10, #21)
- Fill in root `app.py`:
  ```python
  from src.dashboard.app import main
  if __name__ == "__main__":
      main()
  ```
- Apply fixes #9, #39
- Smoke-test locally: `streamlit run app.py` loads dashboard; `make serve` starts FastAPI
- Write `tests/test_prediction_engine.py`, `tests/test_api.py`
- **Gate 7**

### Phase 8 — Governance
- `docs/model_card.md`: intended use, training data, performance, ethical considerations, limitations
- `docs/limitations.md`: county-level granularity, no satellite, no field heterogeneity, no economic factors, geographic scope
- **Gate 8**

### Phase 9 — README + Polish
- Write `README.md` per §16 (HF YAML frontmatter as FIRST 9 LINES)
- Take screenshots: dashboard map, county detail, SHAP waterfall → `reports/figures/`
- Run `make test` — all pass
- End-to-end smoke: `make all` from clean state
- Verify git tracking:
  ```bash
  git check-ignore -v models/best_model.pkl                # should NOT be ignored
  git check-ignore -v data/processed/features.parquet      # should NOT be ignored
  git ls-files models/best_model.pkl                       # should print the path
  git ls-files data/processed/features.parquet             # should print the path
  ```
- **Gate 9**

### Phase 10 — Deploy to Hugging Face Spaces
This phase needs user action. Prepare everything; output exact commands.

Write `docs/deploy_hf.md` with these steps:
1. Create Space at `https://huggingface.co/new-space`:
   - Owner: your HF username
   - Name: `cropiq`
   - License: MIT
   - SDK: **Streamlit**
   - Hardware: CPU basic (free)
   - Visibility: Public
2. Add Space as a git remote (Space provides exact URL):
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/cropiq
   ```
3. Verify runtime files are committed:
   ```bash
   git ls-files models/best_model.pkl
   git ls-files data/processed/features.parquet
   git ls-files app.py
   ```
4. Push to Space:
   ```bash
   git push space main
   ```
5. Watch build at `https://huggingface.co/spaces/<username>/cropiq` (3–5 min)
6. Smoke test: load URL, select IA → 2024 → Story County, verify prediction renders

Troubleshooting section in `deploy_hf.md`:
- **Build fails on requirements:** version conflict; pin exactly per §3
- **App import error:** test `python -c "import app"` locally first
- **Model not found at runtime:** check `git ls-files models/best_model.pkl` shows the file
- **Slow first load after idle:** normal — subsequent requests fast
- **Choropleth blank:** geojson source URL unreachable; use cached `data/processed/counties_geo.json`

**Gate 10 (automated):**
```python
import app  # root app.py imports cleanly
```
Plus user-confirmed: deployed URL loads dashboard and produces a prediction within 5 seconds.

---

## 16. README REQUIREMENTS

The README must start with HF Spaces YAML frontmatter (FIRST 9 LINES, exactly):

```yaml
---
title: CropIQ
emoji: 🌽
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.40.1
app_file: app.py
pinned: false
license: mit
---
```

Then in order:
1. **Title + tagline**
2. **Live demo link** — `https://huggingface.co/spaces/<username>/cropiq`
3. **Hero image** — choropleth screenshot
4. **Key results table** — Model | Val RMSE | Test RMSE | Test MAPE | Test R² | vs Baseline
5. **Architecture diagram** — Mermaid: AIDLC phases + concept layers
6. **Demo screenshots** — dashboard map, county detail, SHAP waterfall
7. **Quickstart (local):**
   ```
   git clone <repo>
   cd cropiq
   cp .env.example .env  # NASS_API_KEY, NOAA_TOKEN
   make install
   make all              # fetch → build → train → eval → gates
   make serve            # FastAPI on :8000 (local-only)
   make dashboard        # Streamlit on :8501
   ```
8. **Deploy to HF Spaces** — one paragraph + link to `docs/deploy_hf.md`
9. **Methodology — AIDLC** — 5–8 bullets
10. **Concept architecture** — brief Daniel Jackson explanation with source link
11. **Agronomic Insight** — paragraph copied from `results.md`
12. **Limitations** — bulleted, honest
13. **Tech stack** — badges
14. **License** — MIT

---

## 17. DEFINITION OF DONE

- [ ] All 11 gates pass (Phases 0–10)
- [ ] `make all` runs end-to-end on clean local checkout
- [ ] `pytest` runs all tests, 100% pass
- [ ] XGBoost test RMSE beats baseline by ≥10%
- [ ] All 12 concepts implemented per §5 with module docstrings matching template
- [ ] All 5 docs in `docs/` complete (problem, data, model, limitations, deploy_hf)
- [ ] README starts with valid HF YAML frontmatter
- [ ] `models/best_model.pkl` and `data/processed/features.parquet` tracked in git
- [ ] Root `app.py` imports successfully (`python -c "import app"`)
- [ ] Public Hugging Face Space loads dashboard in <5 seconds
- [ ] Selecting IA/2024/Story on deployed Space shows prediction + 3 drivers + map
- [ ] Resume-ready: both GitHub link and HF Spaces link functional

---

## 18. WHEN TO STOP AND ASK

Do not stop and ask for:
- Routine implementation decisions covered by this spec
- Minor library version conflicts (resolve by pinning per §3)
- Test failures (fix the underlying bug)

DO stop and ask when:
- A gate fails after one retry
- An API source is unavailable or schema-changed
- The chosen approach genuinely cannot work (explain why before asking)
- The user must do something out-of-band (sign up for API key, create HF Space)

When you stop, report:
1. Phase
2. Gate that failed
3. Exact assertion that failed
4. Proposed fix
5. Whether user action is needed

---

## 19. HUGGING FACE SPACES — DEPLOYMENT REFERENCE

### What HF Spaces provides
- Free CPU basic hardware, always-on (no cold starts)
- Native Streamlit SDK support
- Git-based deployment (push to remote = deploy)
- 50GB storage limit
- Build logs visible in Space UI

### What HF Spaces requires
- `README.md` with YAML frontmatter as first 9 lines (§16)
- `app.py` at repo root
- `requirements.txt` at repo root
- All runtime files committed to git (model + features parquet)

### What HF Spaces does NOT support
- Multiple processes per Space (FastAPI + Streamlit cannot coexist) → Streamlit-only deployment with `PredictionEngine` called in-process
- `.env` files at runtime → use Space Settings UI for secrets (CropIQ needs none)
- Persistent disk writes between builds → all data committed or fetched on first run
- Free-tier GPU → model must be CPU-friendly (XGBoost is fine)

### Build sequence on `git push space main`
1. HF reads README frontmatter → determines SDK
2. Installs `requirements.txt`
3. Launches `streamlit run app.py` (SDK-managed; don't override)
4. Logs at `https://huggingface.co/spaces/<username>/cropiq/logs`

### Resume bullet template once deployed

```
CropIQ | Python, XGBoost, FastAPI, Streamlit, SHAP, MLflow | GitHub | Live Demo
County-level corn yield forecasting (IA/IL/NE, 2010–2024) on USDA NASS yield,
NOAA nClimGrid weather, and USDA SSURGO soil data; XGBoost with
growing-degree-day and growth-stage precipitation features, time-based CV,
SHAP-grounded agronomic drivers, MLflow tracking; deployed Streamlit
dashboard on Hugging Face Spaces with quote-grounded prediction explanations.
```

---

**Start with Phase 0. Run Gate 0. If it passes, proceed to Phase 1 automatically. Keep going until you hit a gate failure or reach Phase 10.**
