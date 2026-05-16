"""
Concept: Evaluation
Purpose: Measure predictor quality on held-out data.
State: predictions, actuals, grouping keys, computed metrics dict.
Actions: compute(preds, actuals), breakdown_by(group), plot_predicted_vs_actual(),
    plot_residuals().
Operational principle: Evaluating XGBoost test predictions returns RMSE, MAPE, R²
    overall and per-state breakdown, and renders the figures listed in CLAUDE.md §11.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error


@dataclass
class Evaluation:
    """Bundles predictions and grouping keys for downstream metrics + plotting."""

    actuals: pd.Series
    predictions: pd.Series
    groups: pd.DataFrame  # arbitrary columns for breakdown (county_fips, year, state...)

    @classmethod
    def from_arrays(
        cls, y_true: Iterable[float], y_pred: Iterable[float], groups: pd.DataFrame | None = None
    ) -> "Evaluation":
        a = pd.Series(np.asarray(y_true, dtype=float), name="actual").reset_index(drop=True)
        p = pd.Series(np.asarray(y_pred, dtype=float), name="prediction").reset_index(drop=True)
        if groups is None:
            g = pd.DataFrame(index=range(len(a)))
        else:
            g = groups.reset_index(drop=True).iloc[: len(a)].copy()
        return cls(actuals=a, predictions=p, groups=g)

    def compute(self) -> dict[str, float]:
        return _metrics(self.actuals, self.predictions)

    def breakdown_by(self, column: str) -> pd.DataFrame:
        """Return per-group RMSE / MAPE / R² for the given grouping column."""
        if column not in self.groups.columns:
            raise KeyError(f"groups DataFrame has no column {column!r}")
        df = self.groups.copy()
        df["__actual"] = self.actuals.to_numpy()
        df["__pred"] = self.predictions.to_numpy()
        out_rows = []
        for key, sub in df.groupby(column):
            m = _metrics(sub["__actual"], sub["__pred"])
            m[column] = key
            m["n"] = len(sub)
            out_rows.append(m)
        out = pd.DataFrame(out_rows)
        return out.loc[:, [column, "n", "rmse", "mape", "r2"]].sort_values(column).reset_index(drop=True)

    def plot_predicted_vs_actual(self, path: Path | str, *, title: str = "Predicted vs Actual (test)") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.actuals, self.predictions, s=10, alpha=0.5)
        lo, hi = min(self.actuals.min(), self.predictions.min()), max(self.actuals.max(), self.predictions.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("Actual yield (bu/acre)")
        ax.set_ylabel("Predicted yield (bu/acre)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_residuals(self, path: Path | str, *, title: str = "Residuals (test)") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        residuals = self.actuals - self.predictions
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residuals, bins=40, alpha=0.8)
        ax.axvline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual (actual − predicted, bu/acre)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


def _metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(root_mean_squared_error(y_t, y_p)),
        "mape": float(mean_absolute_percentage_error(y_t, y_p)),
        "r2": float(r2_score(y_t, y_p)),
    }


def metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Convenience wrapper for `_metrics` used by callers outside the Evaluation class."""
    return _metrics(y_true, y_pred)
