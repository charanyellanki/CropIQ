"""
Concept: YieldPredictor
Purpose: Produce a yield estimate from a feature vector.
State: A trained model (BaselinePredictor / XGBoostPredictor / LightGBMPredictor) along
    with its feature schema and training metadata (run_id, training_date, version,
    residual_std).
Actions: train(X_train, y_train, X_val, y_val), predict(X), save(path), load(path).
Operational principle: A trained predictor returns ~N_test predictions on the test set
    with RMSE below the baseline.

Gotchas applied:
- #5: model is persisted with feature schema + residual_std + versions.
- #6: callers must split by year, not at random.
- #16, #17: callers must one-hot encode `state` BEFORE training.
- #18: sklearn / xgboost / lightgbm versions are recorded with the model.
"""

from __future__ import annotations

import datetime as _dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

import sklearn

TARGET_COL = "yield_bu_per_acre"


@dataclass
class TrainingMetadata:
    """Provenance metadata bundled with every persisted model."""

    model_name: str
    feature_schema: list[str]
    residual_std: float
    training_date: str
    library_versions: dict[str, str]
    run_id: str | None = None
    residual_target: bool = False  # True if model output is `y - rolling_mean`

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_schema": list(self.feature_schema),
            "residual_std": float(self.residual_std),
            "training_date": self.training_date,
            "library_versions": dict(self.library_versions),
            "run_id": self.run_id,
            "residual_target": bool(self.residual_target),
        }


def _library_versions() -> dict[str, str]:
    import platform

    versions = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    try:
        import xgboost as xgb  # noqa: WPS433

        versions["xgboost"] = xgb.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import lightgbm as lgb  # noqa: WPS433

        versions["lightgbm"] = lgb.__version__
    except Exception:  # pragma: no cover
        pass
    return versions


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


class BaselinePredictor:
    """3-year rolling county-mean baseline.

    Uses `yield_rolling_mean_3yr` (already computed in the feature matrix) as the
    prediction. No fit data needed except for residual_std on the validation set.
    """

    def __init__(self) -> None:
        self.metadata: TrainingMetadata | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> "BaselinePredictor":
        # No parameters to fit; compute residual_std on validation predictions.
        val_pred = self.predict(X_val)
        residual_std = float(np.std(np.asarray(y_val) - np.asarray(val_pred)))
        self.metadata = TrainingMetadata(
            model_name="baseline_rolling_mean",
            feature_schema=list(X_train.columns),
            residual_std=residual_std,
            training_date=_dt.datetime.utcnow().isoformat(),
            library_versions=_library_versions(),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "yield_rolling_mean_3yr" not in X.columns:
            raise ValueError("BaselinePredictor requires `yield_rolling_mean_3yr` in X.")
        # Fall back to `yield_lag1` when rolling mean is NaN (e.g., year 2 per county).
        preds = X["yield_rolling_mean_3yr"].copy()
        if "yield_lag1" in X.columns:
            preds = preds.fillna(X["yield_lag1"])
        preds = preds.fillna(preds.mean())
        return preds.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


_BASELINE_FEATURE = "yield_rolling_mean_3yr"


def _to_residual_target(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Return `y - X[_BASELINE_FEATURE]`. Used by `residual_target=True` predictors."""
    base = X[_BASELINE_FEATURE].fillna(X.get("yield_lag1", pd.Series(0.0, index=X.index))).fillna(y.mean())
    return y.astype(float) - base.astype(float)


def _add_baseline(X: pd.DataFrame, residual_pred: np.ndarray) -> np.ndarray:
    """Add the rolling-mean baseline back onto a residual prediction."""
    base = X[_BASELINE_FEATURE].fillna(X.get("yield_lag1", pd.Series(0.0, index=X.index)))
    base = base.fillna(base.mean()).to_numpy(dtype=float)
    return base + np.asarray(residual_pred, dtype=float)


class XGBoostPredictor:
    """XGBoost regression predictor with early stopping on a validation set.

    By default trains on a **residual target** (actual yield minus the 3-year rolling
    county mean). The model therefore learns weather- and soil-driven deviations from
    the autoregressive baseline; calls to `predict` add the baseline back, so the
    output is in the original units (bu/acre).

    Pass `residual_target=False` in the params dict to train on the raw target.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = dict(params)
        self.residual_target: bool = bool(self.params.pop("residual_target", True))
        self.model: Any = None
        self.metadata: TrainingMetadata | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> "XGBoostPredictor":
        from xgboost import XGBRegressor

        params = dict(self.params)
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))
        if self.residual_target:
            y_train_t = _to_residual_target(X_train, y_train)
            y_val_t = _to_residual_target(X_val, y_val)
        else:
            y_train_t = y_train.astype(float)
            y_val_t = y_val.astype(float)
        model = XGBRegressor(
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="rmse",
            **params,
        )
        model.fit(
            X_train,
            y_train_t,
            eval_set=[(X_val, y_val_t)],
            verbose=False,
        )
        self.model = model
        val_pred = self.predict(X_val)
        residual_std = float(np.std(np.asarray(y_val) - np.asarray(val_pred)))
        self.metadata = TrainingMetadata(
            model_name=f"xgboost{'_residual' if self.residual_target else ''}",
            feature_schema=list(X_train.columns),
            residual_std=residual_std,
            training_date=_dt.datetime.utcnow().isoformat(),
            library_versions=_library_versions(),
            residual_target=self.residual_target,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoostPredictor.predict called before fit().")
        raw = self.model.predict(X)
        if self.residual_target:
            return _add_baseline(X, raw)
        return raw


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


class LightGBMPredictor:
    """LightGBM regression predictor (residual target by default — see `XGBoostPredictor`)."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = dict(params)
        self.residual_target: bool = bool(self.params.pop("residual_target", True))
        self.model: Any = None
        self.metadata: TrainingMetadata | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> "LightGBMPredictor":
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation

        params = dict(self.params)
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))
        if self.residual_target:
            y_train_t = _to_residual_target(X_train, y_train)
            y_val_t = _to_residual_target(X_val, y_val)
        else:
            y_train_t = y_train.astype(float)
            y_val_t = y_val.astype(float)
        model = LGBMRegressor(**params, verbose=-1)
        model.fit(
            X_train,
            y_train_t,
            eval_set=[(X_val, y_val_t)],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False), log_evaluation(0)],
        )
        self.model = model
        val_pred = self.predict(X_val)
        residual_std = float(np.std(np.asarray(y_val) - np.asarray(val_pred)))
        self.metadata = TrainingMetadata(
            model_name=f"lightgbm{'_residual' if self.residual_target else ''}",
            feature_schema=list(X_train.columns),
            residual_std=residual_std,
            training_date=_dt.datetime.utcnow().isoformat(),
            library_versions=_library_versions(),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBMPredictor.predict called before fit().")
        raw = self.model.predict(X)
        if self.residual_target:
            return _add_baseline(X, raw)
        return raw


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(predictor: Any, path: Path | str) -> Path:
    """Persist a fitted predictor + metadata to disk via joblib.

    Args:
        predictor: A fitted BaselinePredictor / XGBoostPredictor / LightGBMPredictor.
        path: Destination path.

    Returns:
        The path written.
    """
    if getattr(predictor, "metadata", None) is None:
        raise ValueError("Predictor metadata is missing — fit before saving.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": type(predictor).__name__,
        "model": getattr(predictor, "model", None),
        "metadata": predictor.metadata.to_dict(),
    }
    joblib.dump(payload, path)
    logger.info(f"Saved {payload['kind']} to {path}")
    return path


def load_model(path: Path | str) -> dict[str, Any]:
    """Load a predictor payload back from disk.

    Returns the dict with keys: kind, model, metadata.
    """
    payload = joblib.load(Path(path))
    return payload


def split_by_year(
    features: pd.DataFrame,
    train_year_max: int,
    val_years: list[int],
    test_years: list[int],
    *,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Apply the project-mandated time split (CLAUDE.md §2 rule #1).

    Args:
        features: Output of FeatureMatrix.build (already one-hot encoded).
        train_year_max: Inclusive upper bound for train years.
        val_years: List of validation years.
        test_years: List of test years.
        feature_columns: If provided, restrict X to these columns. Otherwise use all
            columns excluding non-feature identifiers and the target.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    df = features.copy()
    df["year"] = df["year"].astype(int)
    drop_cols = {"county_fips", "year", TARGET_COL, "state"}
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c not in drop_cols]

    train_mask = df["year"] <= train_year_max
    val_mask = df["year"].isin(val_years)
    test_mask = df["year"].isin(test_years)

    X_train = df.loc[train_mask, feature_columns].copy()
    y_train = df.loc[train_mask, TARGET_COL].astype(float).copy()
    X_val = df.loc[val_mask, feature_columns].copy()
    y_val = df.loc[val_mask, TARGET_COL].astype(float).copy()
    X_test = df.loc[test_mask, feature_columns].copy()
    y_test = df.loc[test_mask, TARGET_COL].astype(float).copy()
    return X_train, y_train, X_val, y_val, X_test, y_test
