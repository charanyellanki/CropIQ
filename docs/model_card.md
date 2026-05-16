# CropIQ — Model Card

Filled in after `make train` completes. Pairs with `docs/problem_card.md`,
`docs/data_card.md`, and `docs/limitations.md`.

---

## Model description

| Field                   | Value |
|-------------------------|-------|
| Task                    | Regression — county-year corn grain yield (bu/acre) |
| Algorithm               | XGBoost gradient-boosted trees (LightGBM trained as comparison) |
| Library                 | `xgboost==2.1.2`, `scikit-learn==1.5.2` |
| Training framework      | `src/train.py` (`make train`) |
| Tracking                | MLflow (`mlruns/` — gitignored) |
| Inference path          | `src/inference/prediction_engine.py` — single source of truth used by both FastAPI and the deployed Streamlit dashboard |

## Inputs

See [`data_card.md`](data_card.md) for the full feature list. At inference time the
engine looks up the pre-built feature row for `(county_fips, year)` and runs the
trained model. The Pydantic v2 `PredictionResult` carries the predicted yield, an
80% confidence interval (computed as ±1.28 × residual_std), and the top-3 SHAP
drivers.

## Training protocol

- **Split:** train ≤ 2020, validation 2021–2022, test 2023–2024. Strict time-based;
  no random splits (CLAUDE.md §2 rule 1).
- **Early stopping:** XGBoost / LightGBM stop on `eval_metric="rmse"` against the
  validation set with `early_stopping_rounds=50`.
- **One-hot state encoding** is applied to the full feature matrix before splitting
  (CLAUDE.md gotcha #17).
- **Seeds:** Python `random`, NumPy `np.random`, XGBoost `random_state=42`,
  LightGBM `random_state=42`.

## Selection rule

The trained model with the **lowest test RMSE** is saved to
`models/best_model.pkl`, provided it beats the 3-year rolling county-mean baseline
by ≥10% on test RMSE (CLAUDE.md §2 rule 2 / §10).

## Metrics

Populated by `notebooks/02_model_analysis.ipynb` and copied into
`reports/results.md`. Headline numbers:

| Model     | Val RMSE | Test RMSE | Test MAPE | Test R² | vs Baseline |
| --------- | -------- | --------- | --------- | ------- | ----------- |
| Baseline  | _filled in by `reports/results.md`_  | | | | — |
| XGBoost   |  | | | | |
| LightGBM  |  | | | | |

Per-state and per-year breakdowns are reported in `reports/results.md` and the
analysis notebook.

## Intended use

See [`problem_card.md`](problem_card.md). In one sentence: portfolio-grade
county-level corn yield forecasting for IA / IL / NE 2010–2024, served via a
Hugging Face Space.

## Ethical considerations

- **Producer impact.** The model is not intended for any decision that affects
  producer income, insurance premiums, or land valuation. Misuse could
  systematically penalize counties with structural data gaps.
- **Geographic equity.** Nebraska's irrigated/non-irrigated split is collapsed to
  the headline NASS "ALL PRODUCTION PRACTICES" row, which under-represents
  irrigation-specific yield dynamics.
- **Disclosure.** All sources are publicly available USDA / Open-Meteo datasets.
  No personal data, no proprietary inputs.

## Known limitations

Read [`limitations.md`](limitations.md) before any use.
