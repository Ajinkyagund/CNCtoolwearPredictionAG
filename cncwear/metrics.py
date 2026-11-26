"""Metrics and small table helpers shared by validation, evaluation and the app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)

REG_METRICS = ["MAE", "RMSE", "R2"]
CLF_METRICS = ["Accuracy", "Precision", "Recall", "F1"]   # Precision / Recall / F1 are macro-averaged
N_CLASSES = 3


def reg_metrics(y, pred) -> dict:
    pred = np.clip(np.asarray(pred, float), 0, None)       # a negative RUL is meaningless
    y = np.asarray(y, float)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": r2_score(y, pred) if len(y) > 1 and np.var(y) > 0 else np.nan,
    }


def clf_metrics(y, pred) -> dict:
    labels = list(range(N_CLASSES))
    kw = dict(labels=labels, average="macro", zero_division=0)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, **kw),
        "Recall": recall_score(y, pred, **kw),
        "F1": f1_score(y, pred, **kw),
    }


def score(kind: str, y, pred) -> dict:
    return reg_metrics(y, pred) if kind == "reg" else clf_metrics(y, pred)


def metric_names(kind: str) -> list[str]:
    return REG_METRICS if kind == "reg" else CLF_METRICS


def summarise(fold_df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Per-model mean and std (over folds / repeats) of every metric."""
    cols = metric_names(kind)
    g = fold_df.groupby("Model", sort=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    ordered = ["Model"] + [f"{m}_{s}" for m in cols for s in ("mean", "std")]
    return out[ordered]


def format_mean_std(summary: pd.DataFrame, kind: str, digits: int = 3) -> pd.DataFrame:
    """Turn *_mean / *_std columns into 'mean ± std' strings for reports and the app."""
    out = pd.DataFrame({"Model": summary["Model"]})
    for m in metric_names(kind):
        if f"{m}_mean" in summary:
            out[m] = [f"{a:.{digits}f} ± {0 if pd.isna(b) else b:.{digits}f}"
                      for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    return out


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [(f"{v:.3f}".rstrip("0").rstrip(".") if not pd.isna(v) else "") if isinstance(v, (float, np.floating))
                 else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
