"""
Concept: Experiment
Purpose: Track a training run with parameters, metrics, and artifacts.
State: An open MLflow run with run_id, params, metrics, artifacts.
Actions: start(name), log_param, log_params, log_metric, log_metrics, log_artifact,
    log_dict, end().
Operational principle: Wrapping an XGBoost training run records all hyperparameters and
    metrics to MLflow under a named run.

This is a thin context-manager facade over MLflow so callers don't have to import the
library directly — keeps tests cheap and the rest of the codebase decoupled.

Gotchas applied:
- #15: tracking URI is set explicitly via `mlflow.set_tracking_uri(file://...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from src.config import settings


class Experiment:
    """Context manager wrapping a single MLflow run.

    Usage:
        with Experiment("xgboost_v1") as run:
            run.log_params({"max_depth": 6})
            run.log_metric("train_rmse", 5.4)
    """

    def __init__(
        self,
        run_name: str,
        *,
        experiment_name: str = "cropiq",
        tracking_uri: str | None = None,
    ) -> None:
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or settings.mlflow_uri()
        self._active_run: Any = None
        self.run_id: str | None = None

    def __enter__(self) -> "Experiment":
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._active_run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self._active_run.info.run_id
        return self

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: object | None) -> bool:
        status = "FAILED" if exc_type is not None else "FINISHED"
        mlflow.end_run(status=status)
        return False  # don't suppress exceptions

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(name, float(value), step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_dict(self, payload: dict[str, Any], artifact_path: str) -> None:
        mlflow.log_dict(payload, artifact_path)
