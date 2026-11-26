"""Regression tests for preprocessor outlier clipping and Ridge regression stability."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from cncwear.models import make_preprocessor, with_prep


def test_preprocessor_clips_scaled_outliers_within_bounds():
    """Verify that scaled features are strictly bounded within [-10, 10]."""
    # Training data with normal sensor variance
    train_df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_b": [10.0, 10.5, 9.8, 10.2, 10.1],
    })
    prep = make_preprocessor().fit(train_df)

    # Test data containing extreme DAQ glitch spikes (+1e30, -1e30)
    test_df = pd.DataFrame({
        "feat_a": [1e30, -1e30, 3.0],
        "feat_b": [-1e25, 10.2, 1e28],
    })
    scaled = prep.transform(test_df)

    # Values must remain strictly within [-10.0, 10.0]
    assert np.all(scaled >= -10.0), f"Minimum scaled value was {scaled.min()}, expected >= -10.0"
    assert np.all(scaled <= 10.0), f"Maximum scaled value was {scaled.max()}, expected <= 10.0"


def test_ridge_regression_cv_rmse_is_finite_and_bounded_under_outliers():
    """Verify Ridge regression does not blow up on datasets with extreme sensor spikes.
    
    On unclipped pipelines, a single sensor spike (e.g. 1e30) results in an RMSE of ~1e29.
    With clipping, RMSE must stay finite and comparable to the mean baseline.
    """
    np.random.seed(42)
    # Synthetic dataset with 3 tools, 10 runs each
    cases = np.repeat([1, 2, 3], 10)
    runs = np.tile(np.arange(1, 11), 3)
    ruls = np.tile(10 - np.arange(1, 11), 3).astype(float)
    sensor_feat = np.random.randn(30)
    
    # Tool 2 has an extreme DAQ glitch in one run (analogous to case 2 run 1 in mill.mat)
    sensor_feat[10] = 1e30

    df = pd.DataFrame({"case": cases, "run": runs, "feat": sensor_feat, "RUL": ruls})

    # Train on tool 1 & 3, validate on tool 2
    tr_mask = df["case"].isin([1, 3])
    val_mask = df["case"] == 2

    X_tr = df.loc[tr_mask, ["feat"]]
    y_tr = df.loc[tr_mask, "RUL"]
    X_val = df.loc[val_mask, ["feat"]]
    y_val = df.loc[val_mask, "RUL"]

    # Baseline mean RUL prediction
    baseline_pred = np.full(len(y_val), y_tr.mean())
    baseline_rmse = float(np.sqrt(mean_squared_error(y_val, baseline_pred)))

    # Fit pipeline model with Ridge
    model = with_prep(Ridge(alpha=0.1))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)
    ridge_rmse = float(np.sqrt(mean_squared_error(y_val, pred)))

    # Confirm failure on old unclipped behavior:
    old_prep = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("drop_constant", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
    ])
    old_model = Pipeline([("prep", old_prep), ("model", Ridge(alpha=0.1))])
    old_model.fit(X_tr, y_tr)
    old_rmse = float(np.sqrt(mean_squared_error(y_val, old_model.predict(X_val))))
    assert old_rmse > 1e20, "Old behavior should have blown up with extreme RMSE"

    # New behavior must be finite and not wildly larger than baseline
    assert np.isfinite(ridge_rmse), "Ridge RMSE must be finite"
    assert ridge_rmse < baseline_rmse * 2.0 + 5.0, (
        f"Ridge RMSE ({ridge_rmse:.2f}) was wildly larger than baseline ({baseline_rmse:.2f})"
    )
