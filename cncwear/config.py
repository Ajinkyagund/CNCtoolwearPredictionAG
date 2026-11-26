"""Load config.yaml (single source of truth for all settings)."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None, quick: bool = False) -> dict:
    """Read the YAML config. `quick=True` shrinks everything for a fast smoke test."""
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    cfg["quick"] = bool(quick)
    if quick:
        cfg["cv"]["n_repeats"] = 1
        cfg["tuning"]["outer_repeats"] = 1
        cfg["ablation"]["n_repeats"] = 1
        cfg["explain"] = {"n_background": 8, "n_explain": 15}
    return cfg
