"""Runs the whole pipeline in quick mode and checks the outputs and the 'test set used once' rule."""
import json

import joblib
import pandas as pd

import cncwear.pipeline as pipeline
from tests.helpers import make_synthetic


def test_full_pipeline_quick_run(quick_cfg, tmp_path, monkeypatch):
    csv = tmp_path / "mill.csv"
    make_synthetic().to_csv(csv, index=False)
    out, models = tmp_path / "outputs", tmp_path / "models"

    # record how many rows every training-time stage sees (stages 4-6 must see TRAIN rows only)
    seen = {}

    def spy(name):
        original = getattr(pipeline, name)

        def wrapper(*args, **kwargs):
            seen.setdefault(name, []).append(len(args[1]))     # X is the 2nd positional argument
            return original(*args, **kwargs)
        monkeypatch.setattr(pipeline, name, wrapper)

    for fn in ["repeated_cv", "nested_cv", "tune_final"]:
        spy(fn)
    summary = pipeline.run(quick_cfg, str(csv), out, models)

    n_train = summary["train_runs"]
    assert {"repeated_cv", "nested_cv", "tune_final"} <= set(seen)            # the spies really fired
    assert all(n == n_train for calls in seen.values() for n in calls)
    assert summary["train_runs"] + summary["test_runs"] == len(pd.read_csv(csv))   # nothing lost or duplicated

    split = json.loads((out / "stage2_split.json").read_text())
    assert set(split["train_tools"]).isdisjoint(split["test_tools"])

    for name in ["stage1_eda_summary.csv", "stage4_rul_baselines.csv", "stage4_wear_baselines.csv",
                 "stage5_rul_nested.csv", "stage5_wear_best_params.csv", "stage6_rul_ablation.csv",
                 "stage6_wear_ablation.csv", "stage7_rul_test_metrics.csv", "stage7_wear_test_metrics.csv",
                 "stage7_wear_confusion_matrix.png", "stage7_wear_roc.png", "stage8_feature_importance.csv",
                 "processed_features.csv", "results_tables.md", "run_summary.json"]:
        assert (out / name).exists(), name

    baselines = pd.read_csv(out / "stage4_rul_baselines.csv")
    assert baselines["Model"].str.startswith("Baseline").sum() == 2 and len(baselines) >= 5
    test_tbl = pd.read_csv(out / "stage7_wear_test_metrics.csv")
    assert {"Accuracy", "F1", "ROC-AUC"} <= set(test_tbl.columns)

    assert (models / "meta.json").exists()
    assert joblib.load(models / "rul_model.joblib") is not None
    app_df = pd.read_csv(out / "processed_features.csv")
    assert set(app_df["split"]) == {"train", "test"} and app_df["pred_rul"].notna().all()
