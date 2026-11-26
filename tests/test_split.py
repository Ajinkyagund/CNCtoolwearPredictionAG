import numpy as np

from cncwear.data import prepare
from cncwear.split import make_split


def test_no_tool_appears_on_both_sides(cfg, raw_df):
    df, _ = prepare(raw_df, cfg)
    tr, te, _ = make_split(df, cfg)
    assert set(df.iloc[tr]["case"]).isdisjoint(set(df.iloc[te]["case"]))
    assert len(tr) + len(te) == len(df)


def test_every_wear_class_is_in_both_splits(cfg, raw_df):
    df, _ = prepare(raw_df, cfg)
    tr, te, _ = make_split(df, cfg)
    assert set(df.iloc[tr]["wear_class"]) == {0, 1, 2}
    assert set(df.iloc[te]["wear_class"]) == {0, 1, 2}


def test_split_is_reproducible_for_the_same_seed(cfg, raw_df):
    df, _ = prepare(raw_df, cfg)
    a_tr, a_te, a_seed = make_split(df, cfg)
    b_tr, b_te, b_seed = make_split(df, cfg)
    assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te) and a_seed == b_seed


def test_test_set_is_a_minority_of_tools(cfg, raw_df):
    df, _ = prepare(raw_df, cfg)
    _, te, _ = make_split(df, cfg)
    n_test_tools = df.iloc[te]["case"].nunique()
    assert 2 <= n_test_tools <= df["case"].nunique() // 2
