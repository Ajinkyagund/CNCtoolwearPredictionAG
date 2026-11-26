"""Stage 2 - stratified hold-out of whole TOOLS (the test set is touched once, in Stage 7).

Why group the split by tool: runs of the same tool are highly correlated, so splitting
rows at random would leak information from train to test. StratifiedGroupKFold keeps
every tool on one side while keeping the wear-class proportions as similar as possible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _ok(y: np.ndarray, n_classes: int, min_count: int) -> bool:
    counts = np.bincount(y, minlength=n_classes)
    return bool((counts >= min_count).all())


def make_split(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (train_idx, test_idx, seed_used). Deterministic for a given config seed."""
    seed = cfg["seed"]
    y = df["wear_class"].to_numpy()
    groups = df["case"].to_numpy()
    n_classes = len(cfg["labels"]["class_names"])
    n_splits = int(np.clip(round(1 / cfg["split"]["test_fraction"]), 2, len(np.unique(groups))))
    for offset in range(200):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + offset)
        train, test = next(iter(sgkf.split(np.zeros(len(df)), y, groups)))
        # both sides must contain every wear class (test: at least 2 runs of each)
        if _ok(y[train], n_classes, 1) and _ok(y[test], n_classes, 2):
            return train, test, seed + offset
    raise RuntimeError("Could not find a split with every wear class on both sides. "
                       "Check the wear thresholds in config.yaml.")


def describe_split(df: pd.DataFrame, train: np.ndarray, test: np.ndarray, cfg: dict) -> pd.DataFrame:
    names = cfg["labels"]["class_names"]
    rows = []
    for label, idx in [("Train", train), ("Test", test)]:
        part = df.iloc[idx]
        counts = part["wear_class"].value_counts().reindex(range(len(names))).fillna(0).astype(int)
        row = {"Split": label, "Tools": part["case"].nunique(), "Runs": len(part)}
        row.update({f"{n} (runs)": int(counts[i]) for i, n in enumerate(names)})
        row.update({f"{n} (%)": round(100 * counts[i] / len(part), 1) for i, n in enumerate(names)})
        rows.append(row)
    return pd.DataFrame(rows)


def save_split(path: Path, df: pd.DataFrame, train: np.ndarray, test: np.ndarray, seed_used: int) -> None:
    info = {
        "seed_used": int(seed_used),
        "train_tools": sorted(int(c) for c in df.iloc[train]["case"].unique()),
        "test_tools": sorted(int(c) for c in df.iloc[test]["case"].unique()),
        "n_train_runs": int(len(train)),
        "n_test_runs": int(len(test)),
    }
    path.write_text(json.dumps(info, indent=2))
