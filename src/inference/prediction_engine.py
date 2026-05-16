"""
Concept: PredictionEngine ⭐ (shared by FastAPI and Streamlit)
Purpose: Produce a yield prediction + explanation for a (state, county_fips, year)
    input. Pure logic, no transport layer.
State: Loaded YieldPredictor (with metadata), SHAP TreeExplainer, FeatureMatrix lookup,
    residual_std for CI, model metadata.
Actions: predict(state, county_fips, year) -> PredictionResult,
    list_counties(state) -> List[County],
    get_metadata() -> ModelMetadata.
Operational principle: engine.predict("IA", "19169", 2024) returns predicted yield,
    80% CI, top-3 SHAP drivers, model version — in <100ms on CPU.

This is the single source of truth for inference. Both PredictionService (HTTP) and
DashboardView (Streamlit) delegate to it.

Gotchas applied:
- #30: validates NaN-free input row at predict time; raises on missing data.
- #36: TreeExplainer is built ONCE in __init__ and reused for every request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.concepts.explanation import explain_instance, make_tree_explainer
from src.concepts.feature_matrix import load as load_features
from src.concepts.yield_predictor import _add_baseline, load_model

# 80% CI z-score = 1.2816 (two-tailed)
_Z80 = 1.2816


class Driver(BaseModel):
    """Single SHAP driver: feature name and signed contribution in target units."""

    model_config = ConfigDict(frozen=True)
    feature: str
    contribution: float
    direction: str = Field(description='"positive" or "negative"')


class County(BaseModel):
    model_config = ConfigDict(frozen=True)
    state: str
    county_fips: str
    county_name: str | None = None


class ModelMetadata(BaseModel):
    """Surfaced via GET /health and engine.get_metadata()."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    training_date: str
    residual_std: float
    feature_count: int
    library_versions: dict[str, str]


class PredictionResult(BaseModel):
    """Output of `PredictionEngine.predict`."""

    model_config = ConfigDict(protected_namespaces=())

    state: str
    county_fips: str
    county_name: str | None
    year: int
    predicted_yield_bu_per_acre: float
    confidence_interval_80pct: tuple[float, float]
    top_drivers: list[Driver]
    model_version: str


def _direction(value: float) -> str:
    return "positive" if value >= 0 else "negative"


class PredictionEngine:
    """In-process predictor used by both FastAPI and Streamlit."""

    def __init__(self, model_path: Path | str, features_path: Path | str) -> None:
        model_path = Path(model_path)
        features_path = Path(features_path)
        logger.info(f"PredictionEngine: loading model={model_path} features={features_path}")
        payload = load_model(model_path)
        self._kind: str = str(payload["kind"])
        self._model: Any = payload["model"]
        self._metadata: dict[str, Any] = dict(payload["metadata"])
        self._feature_schema: list[str] = list(self._metadata["feature_schema"])
        self._residual_std: float = float(self._metadata["residual_std"])
        self._residual_target: bool = bool(self._metadata.get("residual_target", False))

        self._features: pd.DataFrame = load_features(features_path)
        self._explainer: shap.TreeExplainer | None = None
        if self._model is not None:
            self._explainer = make_tree_explainer(self._model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, state: str, county_fips: str, year: int) -> PredictionResult:
        """Return a yield prediction with CI and top-3 SHAP drivers.

        Args:
            state: Two-letter state code, e.g. "IA".
            county_fips: 5-char zero-padded county FIPS, e.g. "19169".
            year: Crop year.

        Raises:
            KeyError if no feature row exists for the inputs.
            ValueError if the feature row has NaN in any required input column.
        """
        county_fips = str(county_fips).zfill(5)
        mask = (
            (self._features["county_fips"] == county_fips)
            & (self._features["year"] == int(year))
        )
        sub = self._features.loc[mask]
        if sub.empty:
            raise KeyError(
                f"No feature row for state={state} county_fips={county_fips} year={year}. "
                "Available year range: "
                f"{int(self._features['year'].min())}-{int(self._features['year'].max())}."
            )

        x = sub.iloc[[0]].loc[:, self._feature_schema].copy()
        # gotcha #30: refuse to silently impute.
        nan_cols = [c for c in x.columns if x[c].isna().any()]
        if nan_cols:
            raise ValueError(
                f"Feature row for {state}/{county_fips}/{year} has NaN in: {nan_cols}"
            )

        raw = np.asarray(self._model.predict(x)).reshape(-1)
        if self._residual_target:
            yhat = float(_add_baseline(x, raw)[0])
        else:
            yhat = float(raw[0])
        ci_low = yhat - _Z80 * self._residual_std
        ci_high = yhat + _Z80 * self._residual_std

        explanation = explain_instance(self._explainer, x)
        top = explanation.top_k(3)
        drivers = [
            Driver(feature=name, contribution=float(val), direction=_direction(val))
            for name, val in top
        ]
        county_name = self._lookup_county_name(county_fips)
        return PredictionResult(
            state=state.upper(),
            county_fips=county_fips,
            county_name=county_name,
            year=int(year),
            predicted_yield_bu_per_acre=yhat,
            confidence_interval_80pct=(float(ci_low), float(ci_high)),
            top_drivers=drivers,
            model_version=str(self._metadata.get("run_id") or self._metadata.get("training_date", "unknown")),
        )

    def list_counties(self, state: str) -> list[County]:
        """Return all counties in the feature matrix for the given state."""
        state_alpha = state.upper()
        if "state" in self._features.columns:
            sub = self._features.loc[self._features["state"] == state_alpha, "county_fips"]
        else:
            state_col = f"state_{state_alpha}"
            if state_col not in self._features.columns:
                raise KeyError(f"Cannot infer state column for {state_alpha!r}")
            sub = self._features.loc[self._features[state_col] == 1.0, "county_fips"]
        unique = sorted(sub.unique())
        return [County(state=state_alpha, county_fips=cf) for cf in unique]

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        return ModelMetadata(
            model_name=str(self._metadata.get("model_name", "unknown")),
            model_version=str(self._metadata.get("run_id") or self._metadata.get("training_date", "unknown")),
            training_date=str(self._metadata.get("training_date", "unknown")),
            residual_std=float(self._residual_std),
            feature_count=len(self._feature_schema),
            library_versions=dict(self._metadata.get("library_versions", {})),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _lookup_county_name(self, county_fips: str) -> str | None:
        # Optional friendly name from feature row if present; not currently populated
        # (NASS county_name was dropped during cleaning). Returns None gracefully.
        return None
