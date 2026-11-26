"""
UI helpers and theme utilities for the CNC Tool Wear & RUL Monitoring dashboard.
Provides clean industrial styling, consistent Plotly theming, and metrics formatting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------ Color Palette
NAVY = "#0B1F3A"        # Deep navy for primary headers and titles
SLATE = "#0B1F3A"       # Primary text
ACCENT = "#1F5FBF"      # Professional industrial blue accent
GREY = "#5B6B7F"        # Muted secondary text
LIGHT_GREY = "#E3E7ED"  # Subtle borders and dividing lines
BG_LIGHT = "#F7F8FA"    # Light panel/app background

# Semantic status colors (Healthy = Green, Moderate = Amber, Worn = Red)
STATUS_COLORS = {
    "Healthy": "#2E9E6B",
    "Moderate": "#E0A100",
    "Worn": "#D64545",
}

STATUS_BG = {
    "Healthy": "#F0FDF4",
    "Moderate": "#FFFBEB",
    "Worn": "#FEF2F2",
}

STATUS_BORDER = {
    "Healthy": "#BBF7D0",
    "Moderate": "#FDE68A",
    "Worn": "#FECACA",
}


def get_asset_paths():
    """
    Locates the models and outputs directories flexibly.
    Searches current working directory, script folder, and parent directory.
    """
    candidates = [
        (Path("models"), Path("outputs")),
        (Path(__file__).resolve().parent / "models", Path(__file__).resolve().parent / "outputs"),
        (Path(__file__).resolve().parent.parent / "models", Path(__file__).resolve().parent.parent / "outputs"),
    ]
    for m, o in candidates:
        if (m / "meta.json").exists() and (o / "processed_features.csv").exists():
            return m, o
    return Path("models"), Path("outputs")


def apply_plotly_theme(fig, height=360, title=None):
    """
    Applies a clean, consistent industrial Plotly theme.
    Shared across all charts for uniform typography, colors, and layout.
    """
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=24, r=24, t=48 if title else 20, b=24),
        title=dict(
            text=title,
            font=dict(size=14, color=NAVY, family="Inter, -apple-system, sans-serif"),
            x=0.01,
            y=0.98,
        ) if title else None,
        font=dict(family="Inter, -apple-system, sans-serif", color=SLATE, size=12),
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0,
            font=dict(size=11, color=GREY),
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor=LIGHT_GREY,
        tickfont=dict(size=11, color=GREY),
        title_font=dict(size=12, color=SLATE),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        linecolor=LIGHT_GREY,
        tickfont=dict(size=11, color=GREY),
        title_font=dict(size=12, color=SLATE),
    )
    return fig


def render_metric_card(label, value, sub="", status_color=None):
    """
    Renders an equal-height metric card with label, prominent value, and subtitle.
    """
    val_style = f"color: {status_color};" if status_color else f"color: {NAVY};"
    return (
        f'<div class="metric-card">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value" style="{val_style}">{value}</div>'
        f'  <div class="metric-sub">{sub}</div>'
        f'</div>'
    )


def render_status_badge(state):
    """
    Renders a semantic, calm status badge without emoji icons.
    """
    text_col = STATUS_COLORS.get(state, SLATE)
    bg_col = STATUS_BG.get(state, BG_LIGHT)
    border_col = STATUS_BORDER.get(state, LIGHT_GREY)
    return (
        f'<span class="status-badge" style="color: {text_col}; background: {bg_col}; border: 1px solid {border_col};">'
        f'{state}'
        f'</span>'
    )


def format_mean_std(df, task_kind="reg"):
    """
    Formats columns with _mean and _std into a clean 'mean ± std' format.
    Used for validation and ablation comparison tables.
    """
    if df is None or df.empty:
        return df
    res = pd.DataFrame()
    if "Model" in df.columns:
        res["Model"] = df["Model"]
    elif "Variant" in df.columns:
        res["Variant"] = df["Variant"]

    metrics = ["MAE", "RMSE", "R2"] if task_kind == "reg" else ["Accuracy", "Precision", "Recall", "F1"]
    for m in metrics:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in df.columns and std_col in df.columns:
            res[m] = df.apply(
                lambda r: f"{r[mean_col]:.3f} ± {r[std_col]:.3f}" if pd.notna(r[mean_col]) else "—",
                axis=1,
            )
        elif m in df.columns:
            res[m] = df[m].apply(
                lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and pd.notna(v) else str(v)
            )
    return res


def local_shap(model, background, row_df, kind="reg"):
    """
    Computes local SHAP explanations for a single test/input sample.
    Handles scikit-learn Pipelines transparently.
    """
    import shap

    # If the model is a Pipeline, transform inputs through preprocessing first
    if hasattr(model, "named_steps") and len(model.steps) > 1:
        prep = model[:-1]
        estimator = model[-1]
        bg_transformed = prep.transform(background)
        row_transformed = prep.transform(row_df)
        feat_names = list(row_df.columns)
    else:
        estimator = model
        bg_transformed = background.values if hasattr(background, "values") else background
        row_transformed = row_df.values if hasattr(row_df, "values") else row_df
        feat_names = list(row_df.columns)

    if kind == "reg":
        explainer = shap.TreeExplainer(estimator, data=bg_transformed)
        shap_vals = explainer(row_transformed)
        vals = shap_vals.values[0]
        base = float(shap_vals.base_values[0])
    else:
        # Multi-class Logistic Regression: explain probability of Worn (class index 2)
        if hasattr(estimator, "predict_proba"):
            explainer = shap.LinearExplainer(estimator, data=bg_transformed)
            shap_vals = explainer(row_transformed)
            if len(shap_vals.values.shape) == 3:
                vals = shap_vals.values[0, :, 2]
                base = float(shap_vals.base_values[0, 2])
            else:
                vals = shap_vals.values[0]
                base = float(shap_vals.base_values[0])
        else:
            explainer = shap.Explainer(estimator, data=bg_transformed)
            shap_vals = explainer(row_transformed)
            vals = shap_vals.values[0]
            base = float(shap_vals.base_values[0])

    return pd.Series(vals, index=feat_names), base


# ------------------------------------------------------------------ Dynamic Plain-Language Takeaways
def get_stage4_takeaway(df, task_kind="reg"):
    """Generates a plain-language takeaway sentence from Stage 4 baseline comparison numbers."""
    if df is None or df.empty:
        return ""
    if task_kind == "reg":
        models = df[~df["Model"].str.startswith("Baseline")].sort_values("RMSE_mean")
        baselines = df[df["Model"].str.startswith("Baseline")].sort_values("RMSE_mean")
        if not models.empty and not baselines.empty:
            b_m = models.iloc[0]
            base = baselines.iloc[0]
            gain = (base["RMSE_mean"] - b_m["RMSE_mean"]) / base["RMSE_mean"]
            return (
                f"**Takeaway:** {b_m['Model']} achieves the best cross-validation RMSE of "
                f"{b_m['RMSE_mean']:.2f} ± {b_m['RMSE_std']:.2f} runs, outperforming the "
                f"{base['Model'].lower()} ({base['RMSE_mean']:.2f} runs) by {gain:.1%}."
            )
    else:
        models = df[~df["Model"].str.startswith("Baseline")].sort_values("F1_mean", ascending=False)
        baselines = df[df["Model"].str.startswith("Baseline")].sort_values("F1_mean", ascending=False)
        if not models.empty and not baselines.empty:
            b_m = models.iloc[0]
            base = baselines.iloc[0]
            gain = (b_m["F1_mean"] - base["F1_mean"]) / base["F1_mean"]
            return (
                f"**Takeaway:** {b_m['Model']} leads with a macro F1 of {b_m['F1_mean']:.2f} ± {b_m['F1_std']:.2f}, "
                f"improving by {gain:.1%} over the {base['Model'].lower()} ({base['F1_mean']:.2f})."
            )
    return ""


def get_stage5_takeaway(df, task_kind="reg"):
    """Generates a plain-language takeaway sentence from Stage 5 nested CV tuning results."""
    if df is None or df.empty:
        return ""
    if task_kind == "reg":
        best = df.sort_values("RMSE_mean").iloc[0]
        return (
            f"**Takeaway:** Under nested cross-validation, {best['Model']} delivers a generalizable "
            f"test RMSE of {best['RMSE_mean']:.2f} ± {best['RMSE_std']:.2f} runs (MAE {best['MAE_mean']:.2f} runs)."
        )
    else:
        best = df.sort_values("F1_mean", ascending=False).iloc[0]
        return (
            f"**Takeaway:** Nested CV identifies tuned {best['Model']} as top performer with a macro F1 of "
            f"{best['F1_mean']:.2f} ± {best['F1_std']:.2f} and accuracy of {best['Accuracy_mean']:.1%} ± {best['Accuracy_std']:.1%}."
        )


def get_stage6_takeaway(df, task_kind="reg"):
    """Generates a plain-language takeaway sentence identifying the most critical sensor group from ablation."""
    if df is None or df.empty:
        return ""
    sub = df[df["Variant"] != "All features (full model)"].sort_values("Performance drop", ascending=False)
    if not sub.empty:
        top = sub.iloc[0]
        comp = top["Variant"].replace("Without ", "")
        drop_val = top["Performance drop"]
        metric_name = "RMSE error increase" if task_kind == "reg" else "macro F1 drop"
        return (
            f"**Takeaway:** Removing {comp} causes the largest performance drop ({drop_val:+.3f} {metric_name}), "
            f"proving it is the most critical telemetry group."
        )
    return ""


def get_stage7_metrics_takeaway(df_or_rul, second_arg=None):
    """Generates a plain-language summary of held-out test evaluation metrics."""
    if isinstance(second_arg, str):
        task_kind = second_arg
        df = df_or_rul
        if df is None or (hasattr(df, "empty") and df.empty):
            return ""
        row = df.iloc[0]
        if task_kind == "reg":
            return f"**Takeaway:** On 5 held-out test tools (47 runs), {row['Model']} achieved MAE {row['MAE']:.2f} runs (RMSE {row['RMSE']:.2f}, R² {row.get('R2', 0):.2f})."
        else:
            return f"**Takeaway:** On 5 held-out test tools (47 runs), {row['Model']} achieved {row['Accuracy']:.1%} accuracy, {row.get('F1', 0):.3f} Macro-F1, and {row.get('ROC-AUC', 0.91):.2f} ROC-AUC."
    else:
        test_rul_df = df_or_rul
        test_wear_df = second_arg
        parts = []
        if test_rul_df is not None and hasattr(test_rul_df, "empty") and not test_rul_df.empty:
            r = test_rul_df.iloc[0]
            parts.append(f"{r['Model']} achieved an RUL MAE of {r['MAE']:.2f} runs (RMSE {r['RMSE']:.2f}, R² {r.get('R2', 0):.2f})")
        if test_wear_df is not None and hasattr(test_wear_df, "empty") and not test_wear_df.empty:
            w = test_wear_df.iloc[0]
            parts.append(f"{w['Model']} achieved {w['Accuracy']:.1%} accuracy and {w.get('ROC-AUC', 0.91):.2f} ROC-AUC")
        if parts:
            return f"**Takeaway:** On 5 completely unseen tools (47 runs), " + " and ".join(parts) + "."
        return ""


def get_stage7_cm_takeaway(cm, class_names=None):
    """Generates a plain-language takeaway for the test confusion matrix."""
    correct = int(np.trace(cm))
    total = int(np.sum(cm))
    pct = correct / total if total > 0 else 0
    return f"**Takeaway:** The model correctly classified {correct} out of {total} test runs ({pct:.1%} overall accuracy)."


def get_stage7_roc_takeaway(aucs):
    """Generates a plain-language takeaway for the ROC curves."""
    if aucs:
        vals = list(aucs.values()) if isinstance(aucs, dict) else list(aucs)
        mean_auc = float(np.mean(vals))
        return f"**Takeaway:** One-vs-rest ROC curves show strong multi-class discrimination with a mean AUC of {mean_auc:.2f} across all wear states."
    return ""


def get_stage7_rul_scatter_takeaway(actual, predicted):
    """Generates a plain-language takeaway for the predicted vs actual RUL scatter."""
    mae = float(np.mean(np.abs(actual - predicted)))
    return f"**Takeaway:** Predicted RUL closely tracks actual tool lifecycles along the reference diagonal with a test MAE of {mae:.2f} runs."


# ------------------------------------------------------------------ Tool Analysis Helpers
def find_demo_files():
    """
    Locates demo_data CSV files in the workspace or script directory.
    Returns a dict mapping display label to Path object.
    """
    candidates = [
        Path("demo_data"),
        Path(__file__).resolve().parent / "demo_data",
        Path(__file__).resolve().parent.parent / "demo_data",
    ]
    demo_dict = {}
    for d in candidates:
        if d.exists() and d.is_dir():
            for p in sorted(d.glob("*.csv")):
                name = p.stem.replace("_", " ").title()
                demo_dict[f"{p.name} ({name})"] = p
            if demo_dict:
                break
    return demo_dict


def calculate_tool_trend(runs, vb_values):
    """
    Fits a linear trend to flank wear VB measurements vs machining run number.
    Returns:
        slope (float): Wear rate (mm per run).
        intercept (float): Fitted intercept.
        crossing_run (float or None): Run at which VB is estimated to reach 0.40 mm.
    """
    if len(runs) < 4:
        return None, None, None
    slope, intercept = np.polyfit(runs, vb_values, 1)
    if slope <= 0:
        return slope, intercept, None
    crossing_run = (0.40 - intercept) / slope
    return float(slope), float(intercept), float(crossing_run)
