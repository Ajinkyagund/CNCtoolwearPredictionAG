from sklearn.linear_model import Ridge

from cncwear.ablation import resolve_groups, run_ablation
from cncwear.data import prepare
from cncwear.models import with_prep
from cncwear.split import make_split


def test_resolve_groups_maps_patterns_to_columns(cfg, raw_df):
    _, feats = prepare(raw_df, cfg)
    groups = resolve_groups(feats, cfg["ablation"]["groups"])
    motor = groups["Motor current (smcAC, smcDC)"]
    assert "smcAC_mean" in motor and "smcDC_kurt" in motor and "vib_table_mean" not in motor
    assert groups["Run number"] == ["run"]
    assert set(groups["Cutting parameters (DOC, feed, material)"]) == {"DOC", "feed", "material"}
    shape = groups["Shape statistics (skewness, kurtosis)"]
    assert "AE_table_skew" in shape and "AE_table_mean" not in shape


def test_ablation_table_has_full_model_with_zero_drop(cfg, raw_df):
    df, feats = prepare(raw_df, cfg)
    tr, _, _ = make_split(df, cfg)
    d = df.iloc[tr].reset_index(drop=True)
    local = {**cfg, "ablation": {**cfg["ablation"], "n_splits": 3, "n_repeats": 1}}
    table = run_ablation(with_prep(Ridge()), d[feats], d["RUL"].to_numpy(float), d["wear_class"].to_numpy(),
                         d["case"].to_numpy(), feats, "reg", local)
    assert table.loc[0, "Variant"] == "All features (full model)" and table.loc[0, "Performance drop"] == 0
    assert len(table) == 1 + len(resolve_groups(feats, cfg["ablation"]["groups"]))
    assert (table["Primary metric"] == "RMSE").all()
