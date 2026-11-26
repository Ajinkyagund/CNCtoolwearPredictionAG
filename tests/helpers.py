"""Synthetic dataset with the same columns as the NASA milling features (for tests only)."""
import numpy as np
import pandas as pd

SENSORS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
STATS = ["mean", "std", "rms", "max", "min", "ptp", "skew", "kurt"]


def make_synthetic(n_cases: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for case in range(1, n_cases + 1):
        n = int(rng.integers(6, 14))
        doc, feed, mat = rng.choice([0.75, 1.5]), rng.choice([0.25, 0.5]), rng.choice([1, 2])
        for run in range(1, n + 1):
            vb = 0.02 + 1.0 * (run / n) ** 1.4 * rng.uniform(0.85, 1.15)
            row = dict(case=case, run=run, VB=vb, time=run * 5.0, DOC=doc, feed=feed, material=mat)
            for k, s in enumerate(SENSORS):
                for j, st in enumerate(STATS):
                    row[f"{s}_{st}"] = vb * (0.5 + 0.1 * k) + rng.normal(0, 0.15) + j * 0.01
            rows.append(row)
    return pd.DataFrame(rows)
