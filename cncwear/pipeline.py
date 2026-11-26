"""The 8-stage ML pipeline required by the project rubric.

 1 Data + EDA         5 Tuning (nested CV, fixed seed)
 2 Split (stratified,  6 Ablation (remove one component at a time)
   hold out test)     7 Test evaluation (once)
 3 Preprocess inside   8 Deliver (explainability, saved models, app data)
   a Pipeline
 4 Baselines (repeated stratified group CV, mean +- std)

Stages 1-6 use ONLY the training tools. The test tools are used once, in stage 7.
"""
from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score

from . import ablation, data, explain, plots, split
from .metrics import (
    clf_metrics, format_mean_std, metric_names, reg_metrics, score, summarise, to_markdown,
)
from .models import (
    classification_baselines, classification_models, fitted_feature_names, make_preprocessor,
    regression_baselines, regression_models, with_prep,
)
from .validation import (
    nested_cv, out_of_fold_predictions, predict, repeated_cv, tune_final, _py,
)

warnings.filterwarnings("ignore")

TASKS = {
    "rul": dict(kind="reg", title="RUL regression", target="RUL", primary="RMSE", lower=True),
    "wear": dict(kind="clf", title="Wear-state classification", target="wear_class", primary="F1", lower=False),
}


def _stage(n: int, title: str) -> None:
    print(f"\n[Stage {n}/8] {title}")


def _proba(model, X, n_classes: int = 3):
    """Class probabilities aligned to classes 0..n-1 (None if the model has none)."""
    try:
        raw = model.predict_proba(X)
    except Exception:
        return None
    full = np.zeros((len(X), n_classes))
    for j, c in enumerate(model.classes_):
        full[:, int(c)] = raw[:, j]
    return full


def _auc(y, proba) -> float:
    if proba is None:
        return np.nan
    try:
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except Exception:
        return np.nan


def _show(df: pd.DataFrame, kind: str | None = None) -> None:
    print((format_mean_std(df, kind) if kind else df).to_string(index=False))


def run(cfg: dict, data_path=None, out_dir="outputs", models_dir="models") -> dict:
    out, mdir = Path(out_dir), Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    seed, quick = cfg["seed"], cfg["quick"]
    names = cfg["labels"]["class_names"]
    np.random.seed(seed)
    tables: dict[str, pd.DataFrame] = {}
    if quick:
        print("QUICK MODE: reduced grids and repeats (smoke test only, do not report these numbers).")

    # ------------------------------------------------------------------ 1. data + EDA
    _stage(1, "Data and EDA")
    raw = data.load_raw(data_path)
    df, feats = data.prepare(raw, cfg)
    eda = data.eda_summary(raw, df, feats, cfg)
    eda.to_csv(out / "stage1_eda_summary.csv", index=False)
    plots.plot_eda(df, feats, cfg, out)
    _show(eda)
    tables["Stage 1 - Dataset summary"] = eda

    # ------------------------------------------------------------------ 2. split
    _stage(2, "Stratified group split (test tools held out)")
    tr, te, seed_used = split.make_split(df, cfg)
    split.save_split(out / "stage2_split.json", df, tr, te, seed_used)
    split_tbl = split.describe_split(df, tr, te, cfg)
    split_tbl.to_csv(out / "stage2_split_summary.csv", index=False)
    _show(split_tbl)
    tables["Stage 2 - Train / test split (by tool)"] = split_tbl
    dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    Xtr, Xte = dtr[feats], dte[feats]
    groups, strat = dtr["case"].to_numpy(), dtr["wear_class"].to_numpy()

    # ------------------------------------------------------------------ 3. preprocessing
    _stage(3, "Preprocessing Pipeline (fitted inside every CV fold)")
    pre = make_preprocessor()
    desc = "\n".join(f"{i + 1}. {n}: {type(s).__name__}" for i, (n, s) in enumerate(pre.steps))
    (out / "stage3_pipeline.txt").write_text(
        "Preprocessing Pipeline (identical for every model, fitted only on training data):\n" + desc + "\n")
    print(desc)

    chosen, test_pred, test_proba, oof_pred, final_models = {}, {}, {}, {}, {}
    for key, t in TASKS.items():
        kind, pm, lower = t["kind"], t["primary"], t["lower"]
        dtype = float if kind == "reg" else int
        y_tr, y_te = dtr[t["target"]].to_numpy(dtype), dte[t["target"]].to_numpy(dtype)
        base = regression_baselines() if kind == "reg" else classification_baselines(seed)
        models = (regression_models if kind == "reg" else classification_models)(seed, quick)

        # -------------------------------------------------------------- 4. baselines
        _stage(4, f"{t['title']}: baselines vs models (repeated stratified group CV, mean ± std)")
        s4 = summarise(repeated_cv(base + models, Xtr, y_tr, strat, groups, kind,
                                   cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed), kind)
        s4.to_csv(out / f"stage4_{key}_baselines.csv", index=False)
        plots.plot_comparison(s4, pm, f"{t['title']}: baselines vs models (training CV)",
                              out / f"stage4_{key}_baselines.png", lower)
        _show(s4, kind)
        tables[f"Stage 4 - {t['title']}: baselines vs models (default parameters)"] = format_mean_std(s4, kind)
        base_rows = s4[s4["Model"].str.startswith("Baseline")]
        pick = base_rows[f"{pm}_mean"].idxmin() if lower else base_rows[f"{pm}_mean"].idxmax()
        best_base = (base_rows.loc[pick, "Model"], float(base_rows.loc[pick, f"{pm}_mean"]))

        # -------------------------------------------------------------- 5. tuning
        _stage(5, f"{t['title']}: tuning with nested CV")
        nested = nested_cv(models, Xtr, y_tr, strat, groups, kind, cfg)
        nested.to_csv(out / f"stage5_{key}_nested_folds.csv", index=False)
        s5 = summarise(nested, kind)
        s5.to_csv(out / f"stage5_{key}_nested.csv", index=False)
        plots.plot_comparison(s5, pm, f"{t['title']}: tuned models (nested CV)",
                              out / f"stage5_{key}_nested.png", lower)
        _show(s5, kind)
        tables[f"Stage 5 - {t['title']}: tuned models, nested CV"] = format_mean_std(s5, kind)

        final = tune_final(models, Xtr, y_tr, strat, groups, kind, cfg)
        bp = pd.DataFrame([{
            "Model": n,
            "Best parameters": json.dumps({a.replace("model__", ""): _py(b) for a, b in g.best_params_.items()}),
            "Tuning-CV score": -g.best_score_ if kind == "reg" else g.best_score_,
        } for n, g in final.items()])
        bp.to_csv(out / f"stage5_{key}_best_params.csv", index=False)
        tables[f"Stage 5 - {t['title']}: best parameters"] = bp
        print("\nBest parameters (grid search on the training tools):")
        _show(bp)

        best_row = s5.sort_values(f"{pm}_mean", ascending=lower).iloc[0]
        best_name, best_model = best_row["Model"], final[best_row["Model"]].best_estimator_
        gain = ((best_base[1] - best_row[f"{pm}_mean"]) / best_base[1] if lower
                else (best_row[f"{pm}_mean"] - best_base[1]) / max(best_base[1], 1e-9))
        print(f"\nBest tuned model: {best_name} | {pm} {best_row[f'{pm}_mean']:.3f} vs best baseline "
              f"({best_base[0]}) {best_base[1]:.3f} -> {gain:+.0%}")

        # -------------------------------------------------------------- 6. ablation
        _stage(6, f"{t['title']}: ablation ({best_name}, remove one component at a time)")
        abl = ablation.run_ablation(best_model, Xtr, y_tr, strat, groups, feats, kind, cfg)
        abl.to_csv(out / f"stage6_{key}_ablation.csv", index=False)
        plots.plot_ablation(abl, f"{t['title']}: ablation ({best_name})", out / f"stage6_{key}_ablation.png")
        show = format_mean_std(abl.rename(columns={"Variant": "Model"}), kind).rename(columns={"Model": "Variant"})
        show["Performance drop"] = abl["Performance drop"].round(3).to_numpy()
        _show(show)
        tables[f"Stage 6 - {t['title']}: ablation ({best_name})"] = show

        # -------------------------------------------------------------- 7. test (once)
        _stage(7, f"{t['title']}: evaluation on the held-out test tools (used once)")
        pred = predict(best_model, Xte, kind)
        proba = _proba(best_model, Xte) if kind == "clf" else None
        rows = [{"Model": f"{best_name} (tuned, chosen)", **score(kind, y_te, pred)}]
        if kind == "clf":
            rows[0]["ROC-AUC"] = _auc(y_te, proba)
        for b in base:
            bm = clone(b.estimator).fit(Xtr, y_tr)
            row = {"Model": b.name, **score(kind, y_te, predict(bm, Xte, kind))}
            if kind == "clf":
                row["ROC-AUC"] = _auc(y_te, _proba(bm, Xte))
            rows.append(row)
        test_tbl = pd.DataFrame(rows)
        test_tbl.to_csv(out / f"stage7_{key}_test_metrics.csv", index=False)
        _show(test_tbl.round(3))
        tables[f"Stage 7 - {t['title']}: test-set results"] = test_tbl

        if kind == "clf":
            rep = classification_report(y_te, pred, labels=[0, 1, 2], target_names=names,
                                        output_dict=True, zero_division=0)
            per_class = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "Class"})
            per_class.to_csv(out / "stage7_wear_per_class.csv", index=False)
            tables["Stage 7 - Wear state: per-class test results"] = per_class
            plots.plot_confusion(y_te, pred, names, f"Wear state: confusion matrix\n({best_name}, test tools)",
                                 out / "stage7_wear_confusion_matrix.png")
            if proba is not None:
                plots.plot_roc(y_te, proba, names, f"Wear state: ROC curves ({best_name}, test tools)",
                               out / "stage7_wear_roc.png")
        else:
            plots.plot_rul_scatter(y_te, pred, f"RUL: predicted vs actual\n({best_name}, test tools)",
                                   out / "stage7_rul_pred_vs_actual.png")

        test_pred[key], test_proba[key] = pred, proba
        final_models[key] = best_model
        oof_pred[key] = out_of_fold_predictions(best_model, Xtr, y_tr, strat, groups, kind,
                                                cfg["cv"]["n_splits"], seed + 900)
        joblib.dump(best_model, mdir / f"{key}_model.joblib")
        chosen[key] = {
            "name": best_name,
            "params": {a.replace("model__", ""): _py(b) for a, b in final[best_name].best_params_.items()},
            "nested_cv": {m: {"mean": float(best_row[f"{m}_mean"]), "std": float(best_row[f"{m}_std"])}
                          for m in metric_names(kind)},
            "test": {k: (None if pd.isna(v) else float(v)) for k, v in test_tbl.iloc[0].items() if k != "Model"},
            "best_baseline": {"name": best_base[0], pm: best_base[1]},
            "improvement_vs_best_baseline": float(gain),
        }

    # error analysis per test tool (needs both tasks' predictions)
    per_tool = []
    for case, g in dte.groupby("case"):
        i = g.index.to_numpy()
        per_tool.append({
            "Tool": int(case), "Runs": len(i),
            "RUL MAE (runs)": float(np.mean(np.abs(dte.loc[i, "RUL"].to_numpy() - np.clip(test_pred["rul"][i], 0, None)))),
            "Wear-state accuracy": float(np.mean(dte.loc[i, "wear_class"].to_numpy() == test_pred["wear"][i])),
        })
    per_tool = pd.DataFrame(per_tool)
    per_tool.to_csv(out / "stage7_per_tool_errors.csv", index=False)
    plots.plot_per_tool(per_tool, out / "stage7_per_tool_errors.png")
    tables["Stage 7 - Error analysis per test tool"] = per_tool
    print("\nError analysis per test tool:")
    _show(per_tool.round(3))

    # ------------------------------------------------------------------ 8. deliver
    _stage(8, "Explainability, saved models and app data")
    rf = with_prep(RandomForestRegressor(n_estimators=100 if quick else 300, random_state=seed, n_jobs=-1))
    rf.fit(Xtr, dtr["RUL"])
    importance = pd.Series(rf.named_steps["model"].feature_importances_,
                           index=fitted_feature_names(rf, feats)).sort_values(ascending=False)
    importance.to_csv(out / "stage8_feature_importance.csv", header=["importance"])
    plots.plot_importance(importance, "Which signals drive the RUL prediction? (Random Forest)",
                          "Importance", out / "stage8_feature_importance.png")

    shap_rank = None
    ex = cfg["explain"]
    try:
        for key, t in TASKS.items():
            vals, rows, bg = explain.compute_shap(final_models[key], Xtr, t["kind"], ex["n_background"],
                                                  ex["n_explain"], seed)
            rank = explain.mean_abs(vals, feats)
            rank.to_csv(out / f"stage8_shap_{key}.csv", header=["mean_abs_shap"])
            what = "RUL (runs)" if key == "rul" else "probability of Worn"
            plots.plot_importance(rank, f"SHAP: what drives the {what} prediction", "Mean |SHAP value|",
                                  out / f"stage8_shap_{key}.png")
            try:
                explain.beeswarm(vals, rows, f"SHAP summary: {what}", out / f"stage8_shap_{key}_beeswarm.png")
            except Exception as err:                       # the bar chart is enough if this fails
                print(f"  (beeswarm plot skipped: {err})")
            if key == "rul":
                shap_rank = rank
        print("SHAP explanations saved.")
    except Exception as err:
        print(f"SHAP skipped ({type(err).__name__}: {err}). Is the 'shap' package installed?")
    Xtr.sample(min(ex["n_background"], len(Xtr)), random_state=seed).to_csv(out / "background.csv", index=False)

    rank_src = shap_rank if shap_rank is not None else importance
    top_feats = [f for f in rank_src.index if f != "run"][:8]

    # predictions for the dashboard: test tools = final model (never seen); train tools = CV predictions
    app = df[["case", "run", "VB", "RUL", "wear_class", "wear_state"] + [f for f in feats if f != "run"]].copy()
    app["split"], app["pred_source"] = "train", "cross-validation"
    app.loc[te, "split"], app.loc[te, "pred_source"] = "test", "held-out test"
    app["pred_rul"] = np.nan
    app.loc[tr, "pred_rul"], app.loc[te, "pred_rul"] = oof_pred["rul"], np.clip(test_pred["rul"], 0, None)
    app["pred_wear"] = np.nan
    app.loc[tr, "pred_wear"], app.loc[te, "pred_wear"] = oof_pred["wear"], test_pred["wear"]
    for i, n in enumerate(names):
        app[f"p_{n}"] = np.nan
        if test_proba["wear"] is not None:
            app.loc[te, f"p_{n}"] = test_proba["wear"][:, i]
    app.to_csv(out / "processed_features.csv", index=False)

    summary = {
        "seed": seed, "quick_mode": quick, "split_seed_used": seed_used,
        "train_tools": int(dtr["case"].nunique()), "test_tools": int(dte["case"].nunique()),
        "train_runs": len(dtr), "test_runs": len(dte), "models": chosen,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=_py))
    meta = {
        "features": feats, "top_features": top_feats,
        "defaults": Xtr.median().fillna(0).to_dict(), "min": Xtr.min().fillna(0).to_dict(),
        "max": Xtr.max().fillna(0).to_dict(), "class_names": names,
        "healthy_limit": cfg["labels"]["healthy_limit"], "worn_limit": cfg["labels"]["worn_limit"],
        "summary": summary,
    }
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2, default=_py))

    md = ["# Results tables (generated by train.py)\n",
          "Mean ± std are over cross-validation folds of the TRAINING tools. "
          "Stage 7 numbers are from the held-out test tools, evaluated once.\n"]
    for title, tbl in tables.items():
        md.append(f"\n## {title}\n\n{to_markdown(tbl)}\n")
    (out / "results_tables.md").write_text("\n".join(md))

    print(f"\nDone. Tables for the report: {out / 'results_tables.md'}")
    print("Next: streamlit run app.py")
    return summary
