import numpy as np
import pandas as pd

from cncwear.data import feature_columns, make_labels, prepare


def test_rul_is_runs_left_until_last_run(cfg, raw_df):
    df = make_labels(raw_df, cfg)
    for _, g in df.groupby("case"):
        assert (g["RUL"] == g["run"].max() - g["run"]).all()
        assert g.loc[g["run"].idxmax(), "RUL"] == 0


def test_wear_class_thresholds(cfg):
    vb = [0.05, 0.19, 0.20, 0.39, 0.40, 0.90]
    raw = pd.DataFrame({"case": 1, "run": range(1, len(vb) + 1), "VB": vb})
    df = make_labels(raw, cfg)
    assert df["wear_class"].tolist() == [0, 0, 1, 1, 2, 2]
    assert df["wear_state"].iloc[0] == "Healthy" and df["wear_state"].iloc[-1] == "Worn"


def test_vb_interpolation_stays_inside_each_tool(cfg):
    raw = pd.DataFrame({
        "case": [1, 1, 1, 2, 2, 2],
        "run": [1, 2, 3, 1, 2, 3],
        "VB": [0.1, np.nan, 0.3, np.nan, 0.5, 0.9],
    })
    df = make_labels(raw, cfg)
    assert df.loc[(df["case"] == 1) & (df["run"] == 2), "VB"].iloc[0] == np.float64(0.2)
    # the leading gap of tool 2 is filled from tool 2 itself, never from tool 1
    assert df.loc[(df["case"] == 2) & (df["run"] == 1), "VB"].iloc[0] == 0.5


def test_features_exclude_targets_and_time_but_start_with_run(cfg, raw_df):
    df, feats = prepare(raw_df, cfg)
    assert feats[0] == "run"
    for bad in ["VB", "RUL", "wear_class", "wear_state", "time", "case"]:
        assert bad not in feats
    assert "DOC" in feats and "vib_table_rms" in feats
