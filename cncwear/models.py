"""Stages 3-4 - the shared preprocessing Pipeline, the candidate models and the baselines."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:  # xgboost is optional
    HAS_XGB = False


# ----------------------------------------------------------------------------- stage 3
def make_preprocessor() -> Pipeline:
    """Imputation + constant-feature removal + scaling + outlier clipping, fitted ONLY on training folds.

    Because these steps live inside the Pipeline, every cross-validation fold learns its
    medians / means / scales from its own training part: no information leaks from the
    validation or test tools. Features are clipped to [-10, 10] after standardization to
    prevent extreme sensor DAQ glitches from causing numerical instability in linear models.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("drop_constant", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -10.0, "a_max": 10.0})),
    ])


def with_prep(model) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def fitted_feature_names(pipeline: Pipeline, feats: list[str]) -> list[str]:
    """Feature names that survive the constant-feature filter of a fitted pipeline."""
    mask = pipeline.named_steps["prep"].named_steps["drop_constant"].get_support()
    return list(np.asarray(feats)[mask])


# ----------------------------------------------------------------------------- baselines
class ScheduleRULBaseline(BaseEstimator, RegressorMixin):
    """Fixed-schedule rule: every tool lasts the median training life; RUL = life - run."""

    def fit(self, X, y):
        self.life_ = float(np.median(np.asarray(X["run"], float) + np.asarray(y, float)))
        return self

    def predict(self, X):
        return np.clip(self.life_ - np.asarray(X["run"], float), 0, None)


class RunRuleClassifier(BaseEstimator, ClassifierMixin):
    """Single-feature rule: a small decision tree that only sees the run number."""

    def __init__(self, max_depth=2, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.tree_ = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self.tree_.fit(np.asarray(X["run"], float).reshape(-1, 1), y)
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X):
        return self.tree_.predict(np.asarray(X["run"], float).reshape(-1, 1))

    def predict_proba(self, X):
        return self.tree_.predict_proba(np.asarray(X["run"], float).reshape(-1, 1))


class SafeXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost needs labels 0..k-1; a CV fold may miss a class, so encode the labels ourselves."""

    def __init__(self, n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.m_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=self.random_state,
            verbosity=0, n_jobs=1,
        ).fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        return self.le_.inverse_transform(self.m_.predict(X))

    def predict_proba(self, X):
        return self.m_.predict_proba(X)


# ----------------------------------------------------------------------------- registries
@dataclass
class ModelSpec:
    name: str
    estimator: object
    grid: dict = field(default_factory=dict)
    is_baseline: bool = False


def _g(grid: dict, quick: bool) -> dict:
    """In quick (smoke-test) mode keep a single value per hyper-parameter."""
    return {k: v[:1] for k, v in grid.items()} if quick else grid


def regression_baselines() -> list[ModelSpec]:
    return [
        ModelSpec("Baseline: mean RUL", DummyRegressor(strategy="mean"), is_baseline=True),
        ModelSpec("Baseline: schedule rule", ScheduleRULBaseline(), is_baseline=True),
    ]


def classification_baselines(seed: int) -> list[ModelSpec]:
    return [
        ModelSpec("Baseline: majority class", DummyClassifier(strategy="most_frequent"), is_baseline=True),
        ModelSpec("Baseline: run-number rule", RunRuleClassifier(random_state=seed), is_baseline=True),
    ]


def regression_models(seed: int, quick: bool = False) -> list[ModelSpec]:
    n = 50 if quick else 200
    boost = {"model__n_estimators": [100, 200], "model__max_depth": [2, 3],
             "model__learning_rate": [0.05, 0.1]}
    specs = [
        ModelSpec("Ridge Regression", with_prep(Ridge()),
                  _g({"model__alpha": [0.1, 1, 10, 100, 1000]}, quick)),
        ModelSpec("SVR", with_prep(SVR()),
                  _g({"model__C": [1, 10, 100], "model__gamma": ["scale", 0.01]}, quick)),
        ModelSpec("Random Forest", with_prep(RandomForestRegressor(n_estimators=n, random_state=seed, n_jobs=-1)),
                  _g({"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3]}, quick)),
        ModelSpec("Gradient Boosting", with_prep(GradientBoostingRegressor(random_state=seed)),
                  _g(boost, quick)),
        ModelSpec("KNN", with_prep(KNeighborsRegressor()),
                  _g({"model__n_neighbors": [3, 5, 9]}, quick)),
    ]
    if HAS_XGB:
        specs.append(ModelSpec(
            "XGBoost", with_prep(XGBRegressor(random_state=seed, verbosity=0, n_jobs=1)),
            _g({"model__n_estimators": [150, 300], "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.05, 0.1]}, quick)))
    return specs


def classification_models(seed: int, quick: bool = False) -> list[ModelSpec]:
    n = 50 if quick else 200
    boost = {"model__n_estimators": [100, 200], "model__max_depth": [2, 3],
             "model__learning_rate": [0.05, 0.1]}
    specs = [
        ModelSpec("Logistic Regression", with_prep(LogisticRegression(max_iter=2000)),
                  _g({"model__C": [0.1, 1, 10]}, quick)),
        ModelSpec("SVM", with_prep(SVC(probability=True, random_state=seed)),
                  _g({"model__C": [1, 10, 100], "model__gamma": ["scale", 0.01]}, quick)),
        ModelSpec("Random Forest", with_prep(RandomForestClassifier(n_estimators=n, random_state=seed, n_jobs=-1)),
                  _g({"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3]}, quick)),
        ModelSpec("Gradient Boosting", with_prep(GradientBoostingClassifier(random_state=seed)),
                  _g(boost, quick)),
        ModelSpec("KNN", with_prep(KNeighborsClassifier()),
                  _g({"model__n_neighbors": [3, 5, 9]}, quick)),
    ]
    if HAS_XGB:
        specs.append(ModelSpec(
            "XGBoost", with_prep(SafeXGBClassifier(random_state=seed)),
            _g({"model__n_estimators": [150, 300], "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.05, 0.1]}, quick)))
    return specs
