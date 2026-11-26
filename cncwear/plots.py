"""Matplotlib figures for the report (saved as PNG into outputs/)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

ACCENT, GREY, LIGHT, DARK, RED = "#2563eb", "#9ca3af", "#d1d5db", "#111827", "#dc2626"
CLASS_COLORS = ["#16a34a", "#f59e0b", "#dc2626"]

plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": DARK, "text.color": DARK,
    "xtick.color": DARK, "ytick.color": DARK, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.7,
})


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- stage 1
def plot_eda(df: pd.DataFrame, feats: list[str], cfg: dict, out: Path) -> None:
    lab = cfg["labels"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for _, g in df.groupby("case"):
        ax.plot(g["run"], g["VB"], color=GREY, lw=1, alpha=0.8)
    ax.axhline(lab["healthy_limit"], color=ACCENT, ls="--", lw=1)
    ax.axhline(lab["worn_limit"], color=RED, ls="--", lw=1)
    ax.text(df["run"].max(), lab["healthy_limit"], " healthy limit", va="bottom", ha="right", color=ACCENT, fontsize=8)
    ax.text(df["run"].max(), lab["worn_limit"], " worn limit", va="bottom", ha="right", color=RED, fontsize=8)
    ax.set(xlabel="Run number", ylabel="Flank wear VB (mm)", title="Tool wear grows with use (one line per tool)")
    _save(fig, out / "stage1_wear_curves.png")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    names = lab["class_names"]
    counts = df["wear_class"].value_counts().reindex(range(len(names))).fillna(0)
    ax.bar(names, counts.values, color=CLASS_COLORS[:len(names)])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    ax.set(ylabel="Number of runs", title="Class balance (all data)")
    ax.grid(axis="x", visible=False)
    _save(fig, out / "stage1_class_balance.png")

    corr = df[feats + ["VB"]].corr()["VB"].drop("VB").dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(corr.index, corr.values, color=[ACCENT if v > 0 else GREY for v in corr.values])
    ax.set(xlabel="Correlation with flank wear VB", title="Features most related to wear")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "stage1_feature_correlation.png")


# ----------------------------------------------------------------------------- stages 4-6
def plot_comparison(summary: pd.DataFrame, metric: str, title: str, path: Path, lower_better: bool) -> None:
    """Horizontal bars of mean +- std over folds; best model in blue, baselines light grey."""
    d = summary.sort_values(f"{metric}_mean", ascending=lower_better).reset_index(drop=True)
    best = d["Model"].iloc[0]
    colors = [ACCENT if m == best else (LIGHT if m.startswith("Baseline") else GREY) for m in d["Model"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(d) + 1.6))
    ax.barh(d["Model"], d[f"{metric}_mean"], xerr=d[f"{metric}_std"].fillna(0), color=colors,
            error_kw=dict(ecolor=DARK, lw=1, capsize=3))
    ax.invert_yaxis()
    ax.set(xlabel=f"{metric} (mean ± std over folds) - {'lower' if lower_better else 'higher'} is better",
           title=title)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_ablation(table: pd.DataFrame, title: str, path: Path) -> None:
    d = table[table["Variant"] != "All features (full model)"].copy()
    d["label"] = d["Variant"].str.replace("Without ", "", regex=False)
    d = d.sort_values("Performance drop")
    metric = table["Primary metric"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(d) + 1.6))
    ax.barh(d["label"], d["Performance drop"], color=[ACCENT if v > 0 else GREY for v in d["Performance drop"]])
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set(xlabel=f"Performance drop in {metric} when removed (positive = component helps)", title=title)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


# ----------------------------------------------------------------------------- stage 7
def plot_confusion(y, pred, names: list[str], title: str, path: Path) -> None:
    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.grid(False)
    ax.set_title(title)
    _save(fig, path)


def plot_roc(y, proba: np.ndarray, names: list[str], title: str, path: Path) -> None:
    """One-vs-rest ROC curve per wear class."""
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for i, name in enumerate(names):
        pos = (y == i)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue
        fpr, tpr, _ = roc_curve(pos, proba[:, i])
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_rul_scatter(y, pred, title: str, path: Path) -> None:
    pred = np.clip(pred, 0, None)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y, pred, s=26, color=ACCENT, alpha=0.8)
    lim = [0, max(np.max(y), np.max(pred)) * 1.05 + 0.5]
    ax.plot(lim, lim, color=DARK, ls="--", lw=1)
    ax.set(xlabel="Actual RUL (runs)", ylabel="Predicted RUL (runs)", title=title, xlim=lim, ylim=lim)
    _save(fig, path)


def plot_per_tool(per_tool: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar([f"Tool {t}" for t in per_tool["Tool"]], per_tool["RUL MAE (runs)"], color=ACCENT)
    ax.set(ylabel="RUL MAE (runs)", title="Error analysis: RUL error per test tool")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


# ----------------------------------------------------------------------------- stage 8
def plot_importance(series: pd.Series, title: str, xlabel: str, path: Path, top: int = 15) -> None:
    d = series.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d.index, d.values, color=ACCENT)
    ax.set(xlabel=xlabel, title=title)
    ax.grid(axis="y", visible=False)
    _save(fig, path)
