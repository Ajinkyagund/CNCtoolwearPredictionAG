"""Stage 8 - explainability with SHAP (model-agnostic permutation explainer).

SHAP values say how much each feature pushed one prediction up or down relative to a
background of typical training runs. For classification we explain the probability of the
"Worn" class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WORN = 2


def _predict_fn(model, feats: list[str], kind: str, class_index: int = WORN):
    pos = list(model.classes_).index(class_index) if kind == "clf" else None

    def fn(arr):
        frame = pd.DataFrame(np.asarray(arr, float), columns=feats)
        if kind == "reg":
            return np.clip(model.predict(frame), 0, None)
        return model.predict_proba(frame)[:, pos]

    return fn


def _explainer(model, background: pd.DataFrame, kind: str, seed: int | None = None):
    import shap

    feats = list(background.columns)
    masker = shap.maskers.Independent(background.to_numpy(float), max_samples=len(background))
    fn = _predict_fn(model, feats, kind)
    try:
        return shap.Explainer(fn, masker, algorithm="permutation", feature_names=feats, seed=seed)
    except TypeError:                       # older shap without the seed argument
        return shap.Explainer(fn, masker, algorithm="permutation", feature_names=feats)


def compute_shap(model, X_train: pd.DataFrame, kind: str, n_background: int, n_explain: int, seed: int):
    """Return (shap_values[n_explain x n_features], explained rows, background rows)."""
    feats = list(X_train.columns)
    bg = X_train.sample(min(n_background, len(X_train)), random_state=seed)
    ex = X_train.sample(min(n_explain, len(X_train)), random_state=seed + 1)
    explainer = _explainer(model, bg, kind, seed)
    values = explainer(ex.to_numpy(float), max_evals=2 * len(feats) + 1).values
    return values, ex, bg


def local_shap(model, background: pd.DataFrame, row: pd.DataFrame, kind: str) -> tuple[pd.Series, float]:
    """SHAP values for one input row (used live by the dashboard)."""
    feats = list(background.columns)
    explainer = _explainer(model, background, kind)
    sv = explainer(row[feats].to_numpy(float), max_evals=2 * len(feats) + 1)
    return pd.Series(sv.values[0], index=feats), float(np.ravel(sv.base_values)[0])


def mean_abs(values: np.ndarray, feats: list[str]) -> pd.Series:
    return pd.Series(np.abs(values).mean(axis=0), index=feats).sort_values(ascending=False)


def beeswarm(values: np.ndarray, rows: pd.DataFrame, title: str, path) -> None:
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(values, rows, show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
