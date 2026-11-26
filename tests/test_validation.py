import numpy as np
from sklearn.linear_model import Ridge

from cncwear.data import prepare
from cncwear.models import ModelSpec, with_prep
from cncwear.split import make_split
from cncwear.validation import nested_cv, repeated_cv, stratified_group_splits


def _train_data(cfg, raw_df):
    df, feats = prepare(raw_df, cfg)
    tr, _, _ = make_split(df, cfg)
    d = df.iloc[tr].reset_index(drop=True)
    return d[feats], d["RUL"].to_numpy(float), d["wear_class"].to_numpy(), d["case"].to_numpy()


def test_cv_folds_never_split_a_tool(cfg, raw_df):
    _, _, strat, groups = _train_data(cfg, raw_df)
    n = 0
    for _, _, tr, te in stratified_group_splits(strat, groups, 4, 2, seed=42):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        n += 1
    assert n == 4 * 2                                     # n_splits x n_repeats


def test_cv_folds_are_reproducible(cfg, raw_df):
    _, _, strat, groups = _train_data(cfg, raw_df)
    a = [te.tolist() for *_, te in stratified_group_splits(strat, groups, 4, 1, seed=1)]
    b = [te.tolist() for *_, te in stratified_group_splits(strat, groups, 4, 1, seed=1)]
    assert a == b


def test_repeated_cv_reports_every_fold_for_every_model(cfg, raw_df):
    X, y, strat, groups = _train_data(cfg, raw_df)
    specs = [ModelSpec("Ridge", with_prep(Ridge()), {"model__alpha": [1.0]})]
    folds = repeated_cv(specs, X, y, strat, groups, "reg", n_splits=3, n_repeats=2, seed=42)
    assert len(folds) == 3 * 2
    assert {"MAE", "RMSE", "R2"} <= set(folds.columns)


def test_nested_cv_tunes_inside_each_outer_fold(cfg, raw_df):
    X, y, strat, groups = _train_data(cfg, raw_df)
    specs = [ModelSpec("Ridge", with_prep(Ridge()), {"model__alpha": [0.1, 10.0]})]
    local = {**cfg, "tuning": {"outer_splits": 3, "outer_repeats": 1, "inner_splits": 2}}
    folds = nested_cv(specs, X, y, strat, groups, "reg", local)
    assert len(folds) == 3
    assert folds["best_params"].str.contains("alpha").all()
    assert np.isfinite(folds["RMSE"]).all()
