"""
Concept: Explanation
Purpose: Attribute a single prediction to feature contributions.
State: prediction_value, base_value, feature_contributions: Dict[str, float].
Actions: explain_instance(model, x), explain_global(model, X), to_dict(), top_k(k).
Operational principle: Explaining a Story County 2023 prediction returns the top-3
    drivers with signed SHAP contributions.

Gotchas applied:
- #7: global SHAP uses `shap.sample(X, 200)` to avoid OOM; per-row waterfalls run on
  the full row.
- #36: callers should cache a `shap.TreeExplainer` once at engine startup, not per
  request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


@dataclass
class Explanation:
    """Single prediction explanation."""

    prediction_value: float
    base_value: float
    feature_contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_value": float(self.prediction_value),
            "base_value": float(self.base_value),
            "feature_contributions": {k: float(v) for k, v in self.feature_contributions.items()},
        }

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        items = sorted(self.feature_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return items[:k]


def make_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer once for a fitted gradient-boosting model."""
    return shap.TreeExplainer(model)


def explain_instance(
    explainer: shap.TreeExplainer, x: pd.DataFrame, feature_names: list[str] | None = None
) -> Explanation:
    """Return an Explanation for a single-row feature frame.

    Args:
        explainer: A prepared SHAP TreeExplainer wrapping a fitted model.
        x: Single-row DataFrame (or DataFrame; we use the first row).
        feature_names: Optional override of feature names.

    Returns:
        Explanation dataclass with prediction_value, base_value, feature_contributions.
    """
    if len(x) == 0:
        raise ValueError("Cannot explain an empty DataFrame.")
    row = x.iloc[[0]]
    shap_values = explainer.shap_values(row)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    arr = np.asarray(shap_values).reshape(-1)
    names = list(feature_names) if feature_names is not None else list(row.columns)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).reshape(-1)[0])
    contributions = {name: float(val) for name, val in zip(names, arr)}
    prediction_value = float(base_value) + float(arr.sum())
    return Explanation(prediction_value=prediction_value, base_value=float(base_value), feature_contributions=contributions)


def explain_global(model: Any, X: pd.DataFrame, *, sample_size: int = 200, path: Path | str | None = None) -> Path | None:
    """Save a SHAP global-summary figure and return the path.

    Args:
        model: Fitted tree model.
        X: Feature matrix to draw the background sample from.
        sample_size: Number of rows to sample for the global summary (gotcha #7).
        path: Output path. If None, the plot is shown but not saved.

    Returns:
        Path to the figure, or None if not saved.
    """
    sample = shap.sample(X, min(sample_size, len(X)))
    explainer = make_tree_explainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, show=False)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
    plt.close()
    return None


def plot_waterfall(
    explainer: shap.TreeExplainer,
    x: pd.DataFrame,
    path: Path | str,
    *,
    title: str = "SHAP waterfall",
) -> Path:
    """Render a single-row SHAP waterfall plot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = x.iloc[[0]]
    values = explainer(row)
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(values[0], show=False, max_display=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
