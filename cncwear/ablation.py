"""Stage 6 - ablation: remove one component at a time and measure the change.

The same CV folds are used for every variant, so the comparison is paired and fair.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from .metrics import score, summarise
from .validation import predict, stratified_group_splits

PRIMARY = {"reg": ("RMSE", True), "clf": ("F1", False)}   # (metric, lower_is_better)


def resolve_groups(feats: list[str], groups_cfg: dict) -> dict[str, list[str]]:
    """Map a group name to the feature columns it covers.

    A pattern matches a feature when the feature equals it, starts with '<pattern>_',
    or (for patterns that begin with '_') ends with it.
    """
    out = {}
    for name, patterns in groups_cfg.items():
        cols = [f for f in feats
                if any(f == p or f.startswith(p + "_") or (p.startswith("_") and f.endswith(p))
                       for p in patterns)]
        if cols:
            out[name] = cols
    return out


def run_ablation(estimator, X, y, strat, groups, feats: list[str], kind: str, cfg: dict) -> pd.DataFrame:
    """CV score of the full model and of the model with each component removed."""
    ab = cfg["ablation"]
    splits = list(stratified_group_splits(strat, groups, ab["n_splits"], ab["n_repeats"], cfg["seed"] + 700))
    metric, lower = PRIMARY[kind]

    def cv_scores(cols: list[str], label: str) -> pd.DataFrame:
        rows = []
        for r, k, tr, te in splits:
            est = clone(estimator).fit(X.iloc[tr][cols], y[tr])
            rows.append({"Model": label, "repeat": r, "fold": k,
                         **score(kind, y[te], predict(est, X.iloc[te][cols], kind))})
        return summarise(pd.DataFrame(rows), kind).assign(**{"Features used": len(cols)})

    full = cv_scores(feats, "All features (full model)")
    results = [full]
    for name, cols in resolve_groups(feats, ab["groups"]).items():
        kept = [f for f in feats if f not in cols]
        if kept:
            results.append(cv_scores(kept, f"Without {name}").assign(**{"Features removed": len(cols)}))
    table = pd.concat(results, ignore_index=True).rename(columns={"Model": "Variant"})
    table["Features removed"] = table["Features removed"].fillna(0).astype(int)
    base = full[f"{metric}_mean"].iloc[0]
    # positive = the removed component was helping (error went up / score went down)
    drop = table[f"{metric}_mean"] - base if lower else base - table[f"{metric}_mean"]
    table["Performance drop"] = drop
    table["Primary metric"] = metric
    return table
