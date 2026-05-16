"""Tests for YieldPredictor: time-based split + baseline predict."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.concepts.yield_predictor import BaselinePredictor, split_by_year


def _toy_features() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    rows = []
    for cf in ("19001", "17031"):
        for y in range(2015, 2025):
            rows.append({
                "county_fips": cf,
                "year": y,
                "yield_bu_per_acre": float(170 + rng.standard_normal() * 10),
                "yield_lag1": float(170 + rng.standard_normal() * 10),
                "yield_rolling_mean_3yr": float(170 + rng.standard_normal() * 5),
                "yield_rolling_std_3yr": 5.0,
                "state": cf[:2],
                "feat1": float(rng.standard_normal()),
            })
    df = pd.DataFrame(rows)
    # One-hot states
    df = pd.concat([df, pd.get_dummies(df["state"], prefix="state", dtype=float)], axis=1)
    return df


def test_split_by_year_yields_strict_time_partitions() -> None:
    features = _toy_features()
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_year(
        features,
        train_year_max=2020,
        val_years=[2021, 2022],
        test_years=[2023, 2024],
    )
    # Sizes: 2 counties × 6 train years = 12 train, 4 val, 4 test.
    assert len(X_train) == 12
    assert len(X_val) == 4
    assert len(X_test) == 4
    # Feature columns must NOT include identifiers or the target.
    for c in ("county_fips", "year", "yield_bu_per_acre", "state"):
        assert c not in X_train.columns


def test_baseline_predictor_uses_rolling_mean() -> None:
    features = _toy_features()
    splits = split_by_year(features, 2020, [2021, 2022], [2023, 2024])
    X_train, y_train, X_val, y_val, X_test, _ = splits
    base = BaselinePredictor()
    base.fit(X_train, y_train, X_val, y_val)
    preds = base.predict(X_test)
    assert len(preds) == len(X_test)
    # Baseline predictions should equal the rolling-mean column.
    np.testing.assert_allclose(preds, X_test["yield_rolling_mean_3yr"].to_numpy(), atol=1e-6)
    # Metadata is populated post-fit.
    assert base.metadata is not None
    assert base.metadata.residual_std >= 0.0
    assert "feat1" in base.metadata.feature_schema
