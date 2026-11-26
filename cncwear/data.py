"""Stage 1 - data loading, label construction and a short exploratory summary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

SENSOR_CHANNELS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

# Identifiers and targets are never model inputs. `time` is excluded on purpose:
# elapsed cutting time reveals the tool's life directly (it would leak the target).
NON_FEATURES = {"case", "run", "VB", "time", "wear_class", "wear_state", "RUL"}


# ----------------------------------------------------------------------------- loading
def signal_features(x, prefix: str) -> dict:
    """Statistics of one raw sensor signal (one machining run)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {}
    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": x.std(),
        f"{prefix}_rms": np.sqrt(np.mean(x ** 2)),
        f"{prefix}_max": x.max(),
        f"{prefix}_min": x.min(),
        f"{prefix}_ptp": np.ptp(x),
        f"{prefix}_skew": skew(x),
        f"{prefix}_kurt": kurtosis(x),
    }


def load_mat(path) -> pd.DataFrame:
    """Raw NASA milling file (mill.mat): 167 runs, each with 6 signals of 9000 samples."""
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    key = "mill" if "mill" in mat else [k for k in mat if not k.startswith("__")][0]
    rows = []
    for rec in np.atleast_1d(mat[key]):
        row = {
            "case": int(rec.case), "run": int(rec.run), "VB": float(rec.VB),
            "time": float(rec.time), "DOC": float(rec.DOC), "feed": float(rec.feed),
            "material": float(rec.material),
        }
        for ch in SENSOR_CHANNELS:
            row.update(signal_features(getattr(rec, ch), ch))
        rows.append(row)
    return pd.DataFrame(rows)


def load_csv(path) -> pd.DataFrame:
    """CSV with columns case, run, VB (+ any numeric sensor / process columns)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    rename = {lower[s.lower()]: s for s in ["case", "run", "VB", "time", "DOC", "feed", "material"]
              if s.lower() in lower}
    df = df.rename(columns=rename)
    missing = [c for c in ["case", "run", "VB"] if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}. Found: {list(df.columns)}")
    df["VB"] = pd.to_numeric(df["VB"], errors="coerce")
    return df


def find_dataset(path=None) -> Path:
    """Use the given path, else look for data/mill.mat or data/mill.csv."""
    candidates = [path] if path else ["data/mill.mat", "data/mill.csv"]
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    raise SystemExit(
        "Dataset not found. Put the NASA milling file in data/ as mill.mat or mill.csv "
        "(see README.md), or pass --data PATH."
    )


def load_raw(path=None) -> pd.DataFrame:
    p = find_dataset(path)
    print(f"Loading {p}")
    return load_mat(p) if p.suffix.lower() == ".mat" else load_csv(p)


# ----------------------------------------------------------------------------- labels
def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: run number first, then every numeric column that is not an id/target."""
    others = [c for c in df.columns
              if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    return (["run"] if "run" in df.columns else []) + others


def make_labels(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add the two targets.

    * VB gaps are interpolated INSIDE each tool only (never across tools).
    * wear_class: 0 Healthy / 1 Moderate / 2 Worn from the VB thresholds in config.yaml.
    * RUL: runs left until the last recorded run of that tool.
    """
    lab = cfg["labels"]
    df = raw.sort_values(["case", "run"]).reset_index(drop=True).copy()
    df["VB"] = df.groupby("case")["VB"].transform(lambda s: s.interpolate(limit_direction="both"))
    df = df.dropna(subset=["VB"]).reset_index(drop=True)
    df["wear_class"] = np.select([df["VB"] < lab["healthy_limit"], df["VB"] < lab["worn_limit"]],
                                 [0, 1], default=2).astype(int)
    df["wear_state"] = df["wear_class"].map(dict(enumerate(lab["class_names"])))
    df["RUL"] = (df.groupby("case")["run"].transform("max") - df["run"]).astype(float)
    return df


def prepare(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    df = make_labels(raw, cfg)
    return df, feature_columns(df)


# ----------------------------------------------------------------------------- EDA
def eda_summary(raw: pd.DataFrame, df: pd.DataFrame, feats: list[str], cfg: dict) -> pd.DataFrame:
    """One-table summary of the dataset: size, missing values, class balance."""
    names = cfg["labels"]["class_names"]
    counts = df["wear_class"].value_counts().reindex(range(len(names))).fillna(0).astype(int)
    runs = df.groupby("case").size()
    rows = [
        ("Rows (machining runs)", len(df)),
        ("Tools (cases)", df["case"].nunique()),
        ("Runs per tool (min / median / max)", f"{runs.min()} / {runs.median():.0f} / {runs.max()}"),
        ("Input features", len(feats)),
        ("Missing VB values before interpolation", int(raw["VB"].isna().sum())),
        ("Missing feature cells", int(df[feats].isna().sum().sum())),
        ("Constant features (dropped inside the pipeline)", int((df[feats].nunique() <= 1).sum())),
        ("Flank wear VB mean (mm)", round(df["VB"].mean(), 3)),
        ("Flank wear VB min / max (mm)", f"{df['VB'].min():.2f} / {df['VB'].max():.2f}"),
        ("RUL min / max (runs)", f"{df['RUL'].min():.0f} / {df['RUL'].max():.0f}"),
    ]
    for i, n in enumerate(names):
        rows.append((f"Class balance: {n}", f"{counts[i]} ({counts[i] / len(df):.1%})"))
    return pd.DataFrame(rows, columns=["Item", "Value"])
