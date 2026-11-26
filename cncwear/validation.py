"""Stages 4-5 - repeated stratified group CV, nested CV and hyper-parameter tuning.

All splitters keep every tool (group) on one side and keep the wear-class mix similar in
every fold (StratifiedGroupKFold). Seeds are derived from the single config seed.
"""
from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .metrics import score

SCORING = {"reg": "neg_root_mean_squared_error", "clf": "f1_macro"}


def stratified_group_splits(strat, groups, n_splits: int, n_repeats: int, seed: int
                            ) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (repeat, fold, train_idx, test_idx); a different shuffle for every repeat."""
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(sgkf.split(np.zeros(len(strat)), strat, groups)):
            yield r, k, tr, te


def predict(est, X, kind: str):
    p = est.predict(X)
    return np.clip(p, 0, None) if kind == "reg" else np.asarray(p).astype(int)


def inner_splits(strat, groups, n_splits: int, seed: int) -> list:
    """Pre-computed inner CV folds (GridSearchCV cannot stratify a continuous RUL target)."""
    n_splits = max(2, min(n_splits, len(np.unique(groups))))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(strat)), strat, groups))


def repeated_cv(specs, X, y, strat, groups, kind: str, n_splits: int, n_repeats: int, seed: int
                ) -> pd.DataFrame:
    """Stage 4: default hyper-parameters, repeated stratified group CV. One row per fold."""
    rows = []
    for spec in specs:
        for r, k, tr, te in stratified_group_splits(strat, groups, n_splits, n_repeats, seed):
            est = clone(spec.estimator).fit(X.iloc[tr], y[tr])
            rows.append({"Model": spec.name, "repeat": r, "fold": k,
                         **score(kind, y[te], predict(est, X.iloc[te], kind))})
    return pd.DataFrame(rows)


def nested_cv(specs, X, y, strat, groups, kind: str, cfg: dict) -> pd.DataFrame:
    """Stage 5: nested CV. The outer folds give the score; an inner grid search tunes on the
    outer-training part only, so the outer test tools never influence the tuning."""
    t = cfg["tuning"]
    seed = cfg["seed"]
    rows = []
    for spec in specs:
        for r, k, tr, te in stratified_group_splits(strat, groups, t["outer_splits"], t["outer_repeats"], seed):
            inner = inner_splits(strat[tr], groups[tr], t["inner_splits"], seed + 100 + r)
            gs = GridSearchCV(clone(spec.estimator), spec.grid, scoring=SCORING[kind], cv=inner,
                              refit=True, n_jobs=1).fit(X.iloc[tr], y[tr])
            rows.append({"Model": spec.name, "repeat": r, "fold": k,
                         **score(kind, y[te], predict(gs.best_estimator_, X.iloc[te], kind)),
                         "best_params": json.dumps({a.replace("model__", ""): _py(b)
                                                    for a, b in gs.best_params_.items()})})
    return pd.DataFrame(rows)


def tune_final(specs, X, y, strat, groups, kind: str, cfg: dict) -> dict:
    """Grid search on the WHOLE training set (never the test set) -> best params per model."""
    cv = inner_splits(strat, groups, cfg["tuning"]["inner_splits"], cfg["seed"] + 500)
    return {s.name: GridSearchCV(clone(s.estimator), s.grid, scoring=SCORING[kind], cv=cv,
                                 refit=True, n_jobs=1).fit(X, y)
            for s in specs}


def out_of_fold_predictions(estimator, X, y, strat, groups, kind: str, n_splits: int, seed: int):
    """Cross-validated predictions for the training tools (used by the dashboard replay)."""
    pred = np.zeros(len(y), dtype=float)
    for _, _, tr, te in stratified_group_splits(strat, groups, n_splits, 1, seed):
        est = clone(estimator).fit(X.iloc[tr], y[tr])
        pred[te] = predict(est, X.iloc[te], kind)
    return pred


def _py(v):
    """JSON-friendly python value."""
    return v.item() if isinstance(v, np.generic) else v
