import numpy as np
import pandas as pd

from cncwear.models import (
    RunRuleClassifier, SafeXGBClassifier, ScheduleRULBaseline, fitted_feature_names, make_preprocessor, with_prep,
)
from sklearn.linear_model import Ridge


def test_schedule_baseline_predicts_median_life_minus_run():
    X = pd.DataFrame({"run": [1, 2, 3, 1, 2, 3, 4]})
    y = np.array([2, 1, 0, 3, 2, 1, 0])        # tool A lasts 3 runs, tool B lasts 4 -> median life 4 (rows)
    model = ScheduleRULBaseline().fit(X, y)
    pred = model.predict(pd.DataFrame({"run": [1, 3, 10]}))
    assert pred[0] == model.life_ - 1
    assert pred[2] == 0                          # never negative


def test_run_rule_classifier_outputs_valid_classes_and_probabilities():
    X = pd.DataFrame({"run": np.arange(1, 13)})
    y = np.array([0] * 4 + [1] * 4 + [2] * 4)
    clf = RunRuleClassifier().fit(X, y)
    proba = clf.predict_proba(X)
    assert set(clf.predict(X)) <= {0, 1, 2}
    assert proba.shape == (12, 3) and np.allclose(proba.sum(axis=1), 1)


def test_preprocessor_learns_medians_from_training_data_only():
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    test = pd.DataFrame({"a": [np.nan, 1000.0], "b": [5.0, 6.0]})
    pre = make_preprocessor().fit(train)
    assert pre.named_steps["impute"].statistics_[0] == np.nanmedian(train["a"])   # not influenced by test
    out = pre.transform(test)
    assert not np.isnan(out).any()


def test_all_nan_and_constant_columns_do_not_break_the_pipeline():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "all_nan": np.nan, "const": 7.0})
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pipe = with_prep(Ridge()).fit(X, y)
    assert fitted_feature_names(pipe, list(X.columns)) == ["a"]
    assert pipe.predict(X).shape == (4,)


def test_safe_xgboost_handles_a_missing_class():
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    y = np.array([0] * 10 + [2] * 10)           # class 1 absent in this training fold
    clf = SafeXGBClassifier(n_estimators=10).fit(X, y)
    assert list(clf.classes_) == [0, 2]
    assert set(clf.predict(X)) <= {0, 2}
