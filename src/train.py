"""Train baseline, XGBoost, LightGBM. Log everything to MLflow, save best model.

Usage:
    python -m src.train
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.concepts.evaluation import metrics
from src.concepts.experiment import Experiment
from src.concepts.feature_matrix import load as load_features
from src.concepts.yield_predictor import (
    BaselinePredictor,
    LightGBMPredictor,
    XGBoostPredictor,
    save_model,
    split_by_year,
)
from src.config import settings
from src.logging_setup import configure_logging
from src.paths import BEST_MODEL_PATH, FEATURES_PARQUET


def _seed_everything(seed: int) -> None:
    """Set Python, NumPy seeds and (best-effort) xgboost/lightgbm."""
    random.seed(seed)
    np.random.seed(seed)


def _run_and_log(
    predictor: Any,
    run_name: str,
    params: dict[str, Any],
    splits: tuple,
) -> dict[str, float]:
    """Fit `predictor`, evaluate on val + test, log to MLflow. Returns metrics dict."""
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    logger.info(f"Training run: {run_name}")
    with Experiment(run_name) as run:
        run.log_params({k: v for k, v in params.items() if not isinstance(v, (list, dict))})
        predictor.fit(X_train, y_train, X_val, y_val)

        val_metrics = metrics(y_val, predictor.predict(X_val))
        test_metrics = metrics(y_test, predictor.predict(X_test))
        run.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        run.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Attach training metadata + run_id
        if predictor.metadata is not None:
            predictor.metadata.run_id = run.run_id

        logger.info(
            f"{run_name}: val_rmse={val_metrics['rmse']:.2f} "
            f"test_rmse={test_metrics['rmse']:.2f} "
            f"r2={test_metrics['r2']:.3f}"
        )
        return {"val_rmse": val_metrics["rmse"], "test_rmse": test_metrics["rmse"]}


def main() -> Path:
    """Run baseline + XGBoost + LightGBM, save best, return path to best_model.pkl."""
    configure_logging()
    seed = settings.random_seed
    _seed_everything(seed)

    features = load_features(FEATURES_PARQUET)
    logger.info(f"Loaded features: {features.shape}")

    splits = split_by_year(
        features,
        train_year_max=settings.train_year_max,
        val_years=settings.val_years,
        test_years=settings.test_years,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    logger.info(
        f"Split sizes: train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )

    # --- Baseline ---
    baseline = BaselinePredictor()
    base_metrics = _run_and_log(baseline, "baseline_rolling_mean", {"kind": "rolling_mean", "window": 3}, splits)

    # --- XGBoost ---
    xgb_params = dict(settings.get_yaml_section("model").get("xgboost", {}))
    xgb = XGBoostPredictor(xgb_params)
    xgb_metrics = _run_and_log(xgb, "xgboost_v1", xgb_params, splits)

    # --- LightGBM ---
    lgb_params = dict(settings.get_yaml_section("model").get("lightgbm", {}))
    lgb = LightGBMPredictor(lgb_params)
    lgb_metrics = _run_and_log(lgb, "lightgbm_v1", lgb_params, splits)

    # --- Select best (lowest test RMSE that beats baseline by ≥10%) ---
    candidates = [
        ("xgboost", xgb, xgb_metrics["test_rmse"]),
        ("lightgbm", lgb, lgb_metrics["test_rmse"]),
    ]
    candidates.sort(key=lambda t: t[2])
    best_name, best_predictor, best_test_rmse = candidates[0]
    base_rmse = base_metrics["test_rmse"]
    logger.info(f"Best: {best_name} test_rmse={best_test_rmse:.2f} vs baseline={base_rmse:.2f}")
    if best_test_rmse >= 0.9 * base_rmse:
        logger.warning(
            f"Best model did NOT beat baseline by 10%: "
            f"{best_test_rmse:.2f} vs 0.9×{base_rmse:.2f} = {0.9*base_rmse:.2f}"
        )

    save_model(best_predictor, BEST_MODEL_PATH)
    logger.info(f"Saved best model ({best_name}) to {BEST_MODEL_PATH}")
    return BEST_MODEL_PATH


if __name__ == "__main__":  # pragma: no cover
    main()
