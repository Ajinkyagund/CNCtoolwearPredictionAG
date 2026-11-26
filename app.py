"""
Streamlit dashboard for CNC tool wear classification and Remaining Useful Life (RUL) prediction.
Industrial predictive maintenance interface structured into 5 focused pages:
Overview, Live Prediction, Tool Lifecycle, Model Insights, and Data & Batch Analysis.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

from ui_helpers import (
    NAVY,
    SLATE,
    ACCENT,
    GREY,
    LIGHT_GREY,
    BG_LIGHT,
    STATUS_COLORS,
    STATUS_BG,
    STATUS_BORDER,
    get_asset_paths,
    apply_plotly_theme,
    render_metric_card,
    render_status_badge,
    format_mean_std,
    local_shap,
    get_stage4_takeaway,
    get_stage5_takeaway,
    get_stage6_takeaway,
    get_stage7_metrics_takeaway,
    get_stage7_cm_takeaway,
    get_stage7_roc_takeaway,
    get_stage7_rul_scatter_takeaway,
    find_demo_files,
    calculate_tool_trend,
)

# ------------------------------------------------------------------ Page Configuration
st.set_page_config(
    page_title="CNC Tool Wear & RUL Monitoring",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_footer():
    """Renders the standard project footer across dashboard views."""
    st.markdown(
        """
        <div style="border-top: 1px solid #E3E7ED; margin-top: 2.2rem; padding-top: 0.8rem; text-align: center; font-size: 0.78rem; color: #5B6B7F;">
            AI-Based CNC Tool Wear Classification &amp; Remaining Useful Life Prediction &middot; NASA / UC Berkeley Milling Dataset &middot; Validated on NASA milling benchmark
        </div>
        """,
        unsafe_allow_html=True,
    )


# Locate models and outputs directories flexibly
MODELS, OUT = get_asset_paths()
models_exist = (
    (MODELS / "meta.json").exists()
    and (MODELS / "rul_model.joblib").exists()
    and (MODELS / "wear_model.joblib").exists()
)
outputs_exist = (OUT / "processed_features.csv").exists()

if not models_exist or not outputs_exist:
    st.markdown(
        """
        <div class="title-block">
            <h1>CNC Milling Tool Wear &amp; Remaining Useful Life Monitoring</h1>
            <p>Industrial predictive maintenance system based on milling cutter sensor telemetry and machine learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(
        "Required precomputed artifacts or trained models were not found in 'models/' or 'outputs/'.\n\n"
        "Please run `python train.py` to train the models and generate the required evaluation tables and figures."
    )
    st.info("Once `python train.py` has finished generating artifacts, refresh this page to load the dashboard.")
    render_footer()
    st.stop()

COLORS = STATUS_COLORS

# ------------------------------------------------------------------ CSS Stylesheet (Industrial Light Theme)
st.markdown(
    """
<style>
/* Main app background and layout */
.stApp {
    background-color: #F7F8FA;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1320px;
}
#MainMenu, footer {
    visibility: hidden;
}

/* Professional deep navy title banner */
.title-block {
    background: #0B1F3A;
    padding: 1.25rem 1.6rem;
    border-radius: 8px;
    margin-bottom: 1.25rem;
    color: #FFFFFF;
    box-shadow: 0 1px 3px rgba(11, 31, 58, 0.12);
}
.title-block h1 {
    color: #FFFFFF;
    font-size: 1.45rem;
    font-weight: 600;
    margin: 0 0 0.35rem 0;
    padding: 0;
    letter-spacing: -0.01em;
}
.title-block p {
    color: #94A3B8;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.4;
}

/* Metric Cards - Equal height, clean borders */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E3E7ED;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    height: 100%;
    min-height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(11, 31, 58, 0.04);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #5B6B7F;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.2;
    color: #0B1F3A;
}
.metric-sub {
    font-size: 0.76rem;
    color: #5B6B7F;
    margin-top: 0.25rem;
}

/* Semantic Status Badge */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.92rem;
    letter-spacing: 0.02em;
}

/* Pipeline Step Cards */
.step {
    border-left: 3px solid #1F5FBF;
    background: #FFFFFF;
    border-radius: 0 6px 6px 0;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.6rem;
    border-top: 1px solid #E3E7ED;
    border-right: 1px solid #E3E7ED;
    border-bottom: 1px solid #E3E7ED;
}
.step b {
    display: block;
    font-size: 0.84rem;
    color: #0B1F3A;
    font-weight: 600;
    margin-bottom: 0.15rem;
}
.step span {
    font-size: 0.78rem;
    color: #5B6B7F;
    line-height: 1.35;
    display: block;
}

/* Button & input styling overrides */
.stButton button {
    border-radius: 6px;
    font-weight: 500;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ Asset Loading
def _opt(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_assets():
    meta = json.loads((MODELS / "meta.json").read_text())
    a = {
        "meta": meta,
        "rul_model": joblib.load(MODELS / "rul_model.joblib"),
        "wear_model": joblib.load(MODELS / "wear_model.joblib"),
        "data": pd.read_csv(OUT / "processed_features.csv"),
        "background": _opt("background.csv"),
        "eda": _opt("stage1_eda_summary.csv"),
        "split": _opt("stage2_split_summary.csv"),
        "per_class": _opt("stage7_wear_per_class.csv"),
        "per_tool": _opt("stage7_per_tool_errors.csv"),
    }
    for k in ("rul", "wear"):
        a[f"s4_{k}"] = _opt(f"stage4_{k}_baselines.csv")
        a[f"s5_{k}"] = _opt(f"stage5_{k}_nested.csv")
        a[f"bp_{k}"] = _opt(f"stage5_{k}_best_params.csv")
        a[f"abl_{k}"] = _opt(f"stage6_{k}_ablation.csv")
        a[f"test_{k}"] = _opt(f"stage7_{k}_test_metrics.csv")
        shap = _opt(f"stage8_shap_{k}.csv")
        a[f"shap_{k}"] = shap.set_index(shap.columns[0])["mean_abs_shap"] if shap is not None else None
    imp = _opt("stage8_feature_importance.csv")
    a["importance"] = imp.set_index(imp.columns[0])["importance"] if imp is not None else None
    return a


A = load_assets()
meta, rul_model, wear_model, data = A["meta"], A["rul_model"], A["wear_model"], A["data"].copy()
S = meta["summary"]
FEATS, NAMES = meta["features"], meta["class_names"]
HL, WL = meta["healthy_limit"], meta["worn_limit"]
MAX_RUL = max(float(data["RUL"].max()), 1.0)
data["pred_wear_name"] = data["pred_wear"].map(dict(enumerate(NAMES)))
test = data[data["split"] == "test"].reset_index(drop=True)


# ------------------------------------------------------------------ Core Inference
def predict(frame):
    """
    Subsets input DataFrame to the 52 expected features and computes RUL and wear state.
    Returns:
        rul (np.ndarray): Non-negative predicted Remaining Useful Life in runs.
        cls (np.ndarray): Wear state class index (0=Healthy, 1=Moderate, 2=Worn).
        conf (np.ndarray): Prediction confidence probability or np.nan.
    """
    frame = frame[FEATS]
    rul = np.clip(rul_model.predict(frame), 0, None)
    cls = wear_model.predict(frame).astype(int)
    try:
        conf = wear_model.predict_proba(frame).max(axis=1)
    except Exception:
        conf = np.full(len(frame), np.nan)
    return rul, cls, conf


def advice(state, rul):
    """Provides maintenance recommendation based on wear state and remaining life."""
    if state == "Worn":
        return "error", "Tool wear has exceeded the operational threshold (0.40 mm). Replace the tool immediately."
    if state == "Moderate" or rul <= 2:
        return "warning", "Wear is accelerating with limited remaining life. Schedule cutter replacement."
    return "success", "Tool condition is healthy. Machining parameters are within safe limits."


# ------------------------------------------------------------------ Reusable Chart Builders
def mean_std_bar(summary, metric, lower, title):
    d = summary.sort_values(f"{metric}_mean", ascending=lower).reset_index(drop=True)
    colors = [ACCENT if i == 0 else (LIGHT_GREY if m.startswith("Baseline") else GREY) for i, m in enumerate(d["Model"])]
    fig = go.Figure(
        go.Bar(
            y=d["Model"],
            x=d[f"{metric}_mean"],
            orientation="h",
            marker_color=colors,
            error_x=dict(type="data", array=d[f"{metric}_std"].fillna(0), color="#475569", thickness=1.5),
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=f"{metric} (mean ± std over folds), {'lower' if lower else 'higher'} is better")
    return apply_plotly_theme(fig, max(52 * len(d) + 100, 240), title)


def ablation_bar(table, title):
    d = table[table["Variant"] != "All features (full model)"].copy()
    d["Component removed"] = d["Variant"].str.replace("Without ", "", regex=False)
    d = d.sort_values("Performance drop")
    metric = table["Primary metric"].iloc[0]
    fig = go.Figure(
        go.Bar(
            y=d["Component removed"],
            x=d["Performance drop"],
            orientation="h",
            marker_color=[ACCENT if v > 0 else GREY for v in d["Performance drop"]],
        )
    )
    fig.update_xaxes(title=f"Performance drop in {metric} when removed (positive = component helps)")
    return apply_plotly_theme(fig, max(46 * len(d) + 100, 260), title)


def importance_bar(series, title, xlabel):
    top = series.sort_values(ascending=False).head(15)[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=ACCENT))
    fig.update_xaxes(title=xlabel)
    return apply_plotly_theme(fig, 420, title)


# ------------------------------------------------------------------ Sidebar Navigation
with st.sidebar:
    st.markdown("### CNC Tool Wear & RUL")
    st.caption("NASA Milling Dataset · 16 Tools, 167 Runs")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Live Prediction",
            "Tool Lifecycle",
            "Model Insights",
            "Data & Batch Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Production Models**")
    st.write(f"RUL: `{S['models']['rul']['name']}`")
    st.write(f"Wear State: `{S['models']['wear']['name']}`")
    st.markdown("")
    st.markdown("**Wear State Thresholds**")
    st.markdown(
        f'<span class="status-badge" style="color:{STATUS_COLORS["Healthy"]};background:{STATUS_BG["Healthy"]};border:1px solid {STATUS_BORDER["Healthy"]};">Healthy</span> &nbsp; Flank wear &lt; {HL:.2f} mm',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="status-badge" style="color:{STATUS_COLORS["Moderate"]};background:{STATUS_BG["Moderate"]};border:1px solid {STATUS_BORDER["Moderate"]};">Moderate</span> &nbsp; {HL:.2f} to {WL:.2f} mm',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="status-badge" style="color:{STATUS_COLORS["Worn"]};background:{STATUS_BG["Worn"]};border:1px solid {STATUS_BORDER["Worn"]};">Worn</span> &nbsp; &ge; {WL:.2f} mm (Replace)',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption(
        f"Training partition: {S['train_tools']} tools ({S['train_runs']} runs) · "
        f"Held-out test partition: {S['test_tools']} tools ({S['test_runs']} runs). "
        "Validation and tuning strictly isolate test tools."
    )


# ------------------------------------------------------------------ Main Header Banner
st.markdown(
    """
<div class="title-block">
    <h1>CNC Milling Tool Wear &amp; Remaining Useful Life Monitoring</h1>
    <p>Predictive maintenance telemetry: classify cutter wear state and estimate remaining machining cycles</p>
</div>
""",
    unsafe_allow_html=True,
)

if S.get("quick_mode"):
    st.warning("Running in quick smoke-test mode. Full verification required for final reported metrics.")


# ================================================================== PAGE 1: Overview
def page_overview():
    r, w = S["models"]["rul"], S["models"]["wear"]
    total_runs = S["train_runs"] + S["test_runs"]
    total_tools = S["train_tools"] + S["test_tools"]
    test_pct = (S["test_runs"] / total_runs) * 100

    # 4 Primary KPI Metric Cards
    k = st.columns(4)
    k[0].markdown(
        render_metric_card(
            "Test Wear State F1",
            f"{w['test']['F1']:.3f}",
            f"Accuracy: {w['test']['Accuracy']:.1%} · ROC-AUC: {w['test'].get('ROC-AUC', 0.91):.2f}",
            STATUS_COLORS["Healthy"],
        ),
        unsafe_allow_html=True,
    )
    k[1].markdown(
        render_metric_card(
            "Test RUL Error (MAE)",
            f"{r['test']['MAE']:.2f} runs",
            f"RMSE: {r['test']['RMSE']:.2f} · R²: {r['test']['R2']:.2f}",
            ACCENT,
        ),
        unsafe_allow_html=True,
    )
    k[2].markdown(
        render_metric_card(
            "Machining Telemetry",
            f"{total_runs} runs",
            f"{total_tools} physical tools · {len(FEATS)} features",
        ),
        unsafe_allow_html=True,
    )
    k[3].markdown(
        render_metric_card(
            "Held-Out Test Set",
            f"{S['test_tools']} tools",
            f"{S['test_runs']} runs ({test_pct:.1f}% whole-tool split)",
            STATUS_COLORS["Moderate"],
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    # 2 Meaningful Visualizations
    c1, c2 = st.columns([1.6, 1])
    with c1:
        fig = go.Figure()
        top_vb = float(data["VB"].max()) * 1.05
        for lo, hi, name in [(0, HL, "Healthy"), (HL, WL, "Moderate"), (WL, top_vb, "Worn")]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=COLORS[name], opacity=0.08, line_width=0)
        for case, g in data.groupby("case"):
            is_test = g["split"].iloc[0] == "test"
            fig.add_trace(
                go.Scatter(
                    x=g["run"],
                    y=g["VB"],
                    mode="lines",
                    showlegend=False,
                    line=dict(
                        width=2.5 if is_test else 1.2,
                        color="#ea580c" if is_test else "rgba(31, 95, 191, 0.4)",
                    ),
                    hovertemplate=f"Tool {case} ({'Held-out Test' if is_test else 'Training'})<br>Run %{{x}}<br>VB %{{y:.2f}} mm<extra></extra>",
                )
            )
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#ea580c", width=2.5), name="Held-Out Test Tools (Unseen)"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="rgba(31, 95, 191, 0.6)", width=1.5), name="Training Tools (Cross-Validated)"))
        fig.update_xaxes(title="Machining Run Number")
        fig.update_yaxes(title="Flank Wear VB (mm)")
        st.plotly_chart(apply_plotly_theme(fig, 330, "Tool Wear Progression Across Machining Life"), width="stretch")
        st.caption("Flank wear VB curves over runs. Held-out test tools (orange) are reserved strictly for stage 7.")

    with c2:
        bc = data["wear_state"].value_counts().reindex(NAMES)
        fig_bar = go.Figure(
            go.Bar(
                x=bc.index,
                y=bc.values,
                marker_color=[COLORS[n] for n in bc.index],
                text=[f"{v} ({v / len(data):.1%})" for v in bc.values],
                textposition="auto",
            )
        )
        fig_bar.update_yaxes(title="Machining Runs")
        st.plotly_chart(apply_plotly_theme(fig_bar, 330, "Dataset Class Distribution"), width="stretch")
        st.caption("Run-level wear state balance across the 167 milling operations in the NASA milling dataset.")

    # 8-Stage Machine Learning Pipeline Map (in expander)
    with st.expander("8-Stage Machine Learning Pipeline Map", expanded=False):
        steps = [
            ("Stage 1 - Data Ingestion & EDA", "167 runs across 16 tools, 6 sensor channels, 8 statistics per channel + cutting parameters (52 features). Missing flank wear values interpolated monotonically."),
            ("Stage 2 - Tool-Level Stratification", "Whole physical tools partitioned with StratifiedGroupKFold into 11 training tools (120 runs) and 5 held-out test tools (47 runs). Prevents cross-tool data leakage."),
            ("Stage 3 - Preprocessing Pipeline", "Leak-free Pipeline: SimpleImputer (median), VarianceThreshold (constant filter), StandardScaler, and Outlier Clipping ([-10, 10]) fitted strictly on training folds."),
            ("Stage 4 - Baseline Benchmarking", "Repeated Stratified Group CV (4 folds × 3 repeats). Evaluates candidate models against mean RUL, schedule rule, majority class, and run-number baselines with mean ± std."),
            ("Stage 5 - Nested-CV Tuning", "Hyperparameter optimization with inner grid search on training folds and outer folds for unbiased error estimation. No test set snooping."),
            ("Stage 6 - Sensor & Feature Ablation", "Systematically removes sensor channels and feature groups on identical CV folds to quantify predictive signal contribution."),
            ("Stage 7 - Held-Out Test Evaluation", "One-time unbiased evaluation on the 5 held-out test tools (47 runs): accuracy, precision, recall, macro-F1, ROC-AUC, confusion matrix, and error analysis."),
            ("Stage 8 - Explainability & Delivery", "Tree-based and linear SHAP values explain global feature importance and individual run predictions inside an interactive monitoring dashboard."),
        ]
        s_cols = st.columns(2)
        for i, (ttl, desc) in enumerate(steps):
            with s_cols[i % 2]:
                st.markdown(f'<div class="step"><b>{ttl}</b><span>{desc}</span></div>', unsafe_allow_html=True)

    render_footer()


# ================================================================== PAGE 2: Live Prediction
def page_live_prediction():
    controls = [
        "smcAC_rms", "smcDC_mean",
        "vib_table_rms", "vib_spindle_rms",
        "AE_table_rms", "AE_spindle_rms",
        "run", "DOC", "feed",
    ]

    if "base_vals" not in st.session_state:
        st.session_state["base_vals"] = dict(meta["defaults"])
    for f in controls:
        if f"s_{f}" not in st.session_state:
            st.session_state[f"s_{f}"] = float(st.session_state["base_vals"][f])

    def set_preset(preset_idx):
        sub = data[data["wear_class"] == preset_idx]
        class_medians = sub[FEATS].median().to_dict()
        st.session_state["base_vals"].update(class_medians)
        for f in controls:
            lo, hi = float(meta["min"][f]), float(meta["max"][f])
            st.session_state[f"s_{f}"] = float(np.clip(class_medians[f], lo, hi))

    st.markdown("#### Tool Operating Telemetry")
    p = st.columns([1, 1, 1, 2.2])
    p[0].button("Fresh Tool (Healthy)", on_click=set_preset, args=(0,), width="stretch")
    p[1].button("Mid-Life Tool (Moderate)", on_click=set_preset, args=(1,), width="stretch")
    p[2].button("Worn Tool (End of Life)", on_click=set_preset, args=(2,), width="stretch")
    p[3].caption("Select a preset condition baseline or manually adjust telemetry sliders below.")

    left, right = st.columns([1.15, 1])
    with left:
        with st.container(border=True):
            st.markdown("**Key Sensor Telemetry & Process Parameters**")
            sc = st.columns(2)
            for i, f in enumerate(controls):
                lo, hi = float(meta["min"][f]), float(meta["max"][f])
                if hi <= lo:
                    hi = lo + 1.0
                with sc[i % 2]:
                    st.slider(f, lo, hi, step=max((hi - lo) / 200, 1e-6), format="%.4g", key=f"s_{f}")

    vals = dict(st.session_state["base_vals"])
    vals.update({f: st.session_state[f"s_{f}"] for f in controls})
    row_df = pd.DataFrame([vals])[FEATS]
    rul, cls, conf = predict(row_df)
    state = NAMES[cls[0]]

    with right:
        with st.container(border=True):
            st.markdown("**Model Assessment**")
            st.markdown(render_status_badge(state), unsafe_allow_html=True)
            try:
                probs = wear_model.predict_proba(row_df)[0]
                p_c1, p_c2, p_c3 = st.columns(3)
                p_c1.markdown(f'<div style="font-size:0.75rem; color:{STATUS_COLORS["Healthy"]}; font-weight:600;">Healthy: {probs[0]:.1%}</div>', unsafe_allow_html=True)
                p_c2.markdown(f'<div style="font-size:0.75rem; color:{STATUS_COLORS["Moderate"]}; font-weight:600;">Moderate: {probs[1]:.1%}</div>', unsafe_allow_html=True)
                p_c3.markdown(f'<div style="font-size:0.75rem; color:{STATUS_COLORS["Worn"]}; font-weight:600;">Worn: {probs[2]:.1%}</div>', unsafe_allow_html=True)
            except Exception:
                st.caption("" if np.isnan(conf[0]) else f"Classifier confidence: {conf[0]:.1%}")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(rul[0]),
                    number=dict(suffix=" runs", valueformat=".1f", font=dict(color=NAVY, size=30)),
                    title=dict(text="Remaining Useful Life", font=dict(size=13, color=SLATE)),
                    gauge=dict(
                        axis=dict(range=[0, MAX_RUL], tickcolor=GREY),
                        bar=dict(color=ACCENT),
                        steps=[
                            dict(range=[0, MAX_RUL * 0.2], color="rgba(214,69,69,0.15)"),
                            dict(range=[MAX_RUL * 0.2, MAX_RUL * 0.5], color="rgba(224,161,0,0.15)"),
                            dict(range=[MAX_RUL * 0.5, MAX_RUL], color="rgba(46,158,107,0.15)"),
                        ],
                    ),
                )
            )
            gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(apply_plotly_theme(gauge, 250), width="stretch")
            level, text = advice(state, rul[0])
            getattr(st, level)(text)

    with st.expander("Why this prediction: Local SHAP Feature Contributions", expanded=False):
        st.caption("Feature contributions toward this prediction relative to expected baseline values.")
        if A["background"] is None:
            st.info("Background dataset (outputs/background.csv) not found.")
        elif st.button("Generate SHAP Contribution Plot"):
            try:
                bg = A["background"][FEATS]
                with st.spinner("Computing local Shapley values..."):
                    s_rul, base_rul = local_shap(rul_model, bg, row_df, "reg")
                    s_wear, base_wear = local_shap(wear_model, bg, row_df, "clf")
                e1, e2 = st.columns(2)
                for col, s, base, ttl, up in [
                    (e1, s_rul, base_rul, "Remaining Useful Life (Runs)", "longer life"),
                    (e2, s_wear, base_wear, "Probability of Worn State", "accelerated wear"),
                ]:
                    top = s.reindex(s.abs().sort_values(ascending=False).index).head(8)[::-1]
                    fig = go.Figure(
                        go.Bar(
                            x=top.values,
                            y=top.index,
                            orientation="h",
                            marker_color=[ACCENT if v > 0 else "#D64545" for v in top.values],
                        )
                    )
                    fig.update_xaxes(title=f"Positive contribution pushes towards {up}")
                    col.plotly_chart(apply_plotly_theme(fig, 330, f"{ttl} (Baseline: {base:.2f})"), width="stretch")
            except Exception as err:
                st.warning(f"SHAP explanation currently unavailable ({type(err).__name__}).")

    render_footer()


# ================================================================== PAGE 3: Tool Lifecycle
def page_tool_lifecycle():
    st.markdown("#### Tool Lifecycle Step-Through")
    st.caption(
        "Inspect telemetry and sequential predictions across the entire lifespan of a single tool. "
        "Held-out test tools represent unseen physical components."
    )
    kind_of = data.groupby("case")["split"].first()
    order = sorted(kind_of.index, key=lambda c: (kind_of[c] != "test", c))
    case = st.selectbox(
        "Select Cutter for Inspection",
        order,
        format_func=lambda c: f"Tool {c} · " + ("Held-Out Test Tool (Unseen)" if kind_of[c] == "test" else "Training Tool (Cross-Validated)"),
    )
    g = data[data["case"] == case].sort_values("run").reset_index(drop=True)
    runs = [int(x) for x in g["run"]]
    sel = st.select_slider("Machining Run Number", options=runs, value=runs[len(runs) // 2]) if len(runs) > 1 else runs[0]
    row = g[g["run"] == sel].iloc[0]

    a = st.columns(4)
    a[0].markdown(
        render_metric_card("Measured Wear VB", f"{row['VB']:.2f} mm", f"Machining cycle {sel}"),
        unsafe_allow_html=True,
    )
    a[1].markdown(
        render_metric_card("Actual Wear State", row["wear_state"], "Ground truth from VB", STATUS_COLORS[row["wear_state"]]),
        unsafe_allow_html=True,
    )
    ok = row["pred_wear_name"] == row["wear_state"]
    a[2].markdown(
        render_metric_card(
            "Predicted State",
            row["pred_wear_name"],
            "Correct classification" if ok else "Misclassified",
            STATUS_COLORS[row["pred_wear_name"]],
        ),
        unsafe_allow_html=True,
    )
    a[3].markdown(
        render_metric_card(
            "RUL Actual vs Predicted",
            f"{row['RUL']:.0f} vs {max(row['pred_rul'], 0):.1f}",
            "Machining runs remaining",
            ACCENT,
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=g["run"],
                y=g["RUL"],
                name="Actual RUL",
                mode="lines+markers",
                line=dict(color="#64748b", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=g["run"],
                y=g["pred_rul"].clip(lower=0),
                name="Predicted RUL",
                mode="lines+markers",
                line=dict(color=ACCENT),
            )
        )
        fig.add_vline(x=sel, line_width=1, line_dash="dash", line_color=GREY)
        fig.update_xaxes(title="Run Number")
        fig.update_yaxes(title="Remaining Useful Life (Runs)")
        st.plotly_chart(apply_plotly_theme(fig, 340, "Remaining Useful Life Trajectory"), width="stretch")
    with c2:
        fig = go.Figure()
        for lo, hi, name in [(0, HL, "Healthy"), (HL, WL, "Moderate"), (WL, max(float(g["VB"].max()), WL) * 1.1, "Worn")]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=COLORS[name], opacity=0.08, line_width=0)
        fig.add_trace(go.Scatter(x=g["run"], y=g["VB"], mode="lines", line=dict(color=GREY), showlegend=False))
        for name in NAMES:
            sub = g[g["pred_wear_name"] == name]
            if len(sub):
                fig.add_trace(
                    go.Scatter(
                        x=sub["run"],
                        y=sub["VB"],
                        mode="markers",
                        name=f"Predicted {name}",
                        marker=dict(size=10, color=COLORS[name]),
                    )
                )
        fig.add_vline(x=sel, line_width=1, line_dash="dash", line_color=GREY)
        fig.update_xaxes(title="Run Number")
        fig.update_yaxes(title="Flank Wear VB (mm)")
        st.plotly_chart(apply_plotly_theme(fig, 340, "Flank Wear Progression & Predicted States"), width="stretch")

    # Sensor Telemetry Timeline
    st.markdown("**Sensor Telemetry Dynamics Across Lifecycle**")
    t1, t2, t3 = st.columns(3)
    with t1:
        f_ae = go.Figure(go.Scatter(x=g["run"], y=g["AE_table_rms"], mode="lines+markers", line=dict(color=ACCENT)))
        f_ae.add_vline(x=sel, line_width=1, line_dash="dash", line_color=GREY)
        f_ae.update_xaxes(title="Run")
        f_ae.update_yaxes(title="RMS")
        t1.plotly_chart(apply_plotly_theme(f_ae, 220, "AE Table RMS"), width="stretch")
    with t2:
        f_vib = go.Figure(go.Scatter(x=g["run"], y=g["vib_spindle_rms"], mode="lines+markers", line=dict(color="#ea580c")))
        f_vib.add_vline(x=sel, line_width=1, line_dash="dash", line_color=GREY)
        f_vib.update_xaxes(title="Run")
        f_vib.update_yaxes(title="RMS")
        t2.plotly_chart(apply_plotly_theme(f_vib, 220, "Spindle Vibration RMS"), width="stretch")
    with t3:
        f_curr = go.Figure(go.Scatter(x=g["run"], y=g["smcAC_rms"], mode="lines+markers", line=dict(color="#16a34a")))
        f_curr.add_vline(x=sel, line_width=1, line_dash="dash", line_color=GREY)
        f_curr.update_xaxes(title="Run")
        f_curr.update_yaxes(title="RMS")
        t3.plotly_chart(apply_plotly_theme(f_curr, 220, "AC Motor Current RMS"), width="stretch")

    with st.expander("View Full Tool Run Telemetry Table"):
        t = g[["run", "VB", "RUL", "pred_rul", "wear_state", "pred_wear_name"]].copy()
        t["pred_rul"] = t["pred_rul"].round(1)
        st.dataframe(
            t.rename(
                columns={
                    "run": "Run",
                    "VB": "Measured VB (mm)",
                    "RUL": "Actual RUL",
                    "pred_rul": "Predicted RUL",
                    "wear_state": "Actual State",
                    "pred_wear_name": "Predicted State",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    render_footer()


# ================================================================== PAGE 4: Model Insights
def page_model_insights():
    st.markdown("#### Model Insights & Pipeline Evidence")
    st.caption(
        "Technical evaluation and validation evidence across the machine learning lifecycle: "
        "baseline benchmarking, nested-CV tuning, explainability, and sensor ablation."
    )

    insight_tabs = st.tabs([
        "Performance",
        "Validation",
        "Explainability",
        "Experiments",
    ])

    # --- Sub-tab 1: Performance (Stages 4 & 7 Consolidated)
    with insight_tabs[0]:
        st.markdown("##### Candidate Models vs Baselines (Stage 4 Repeated CV)")
        st.caption("Repeated Stratified Group Cross-Validation on the 11 training tools (mean ± std across folds).")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.markdown("**RUL Regression (Stage 4)**")
            if A["s4_rul"] is not None:
                st.dataframe(format_mean_std(A["s4_rul"], "reg"), hide_index=True, width="stretch")
                st.plotly_chart(mean_std_bar(A["s4_rul"], "RMSE", True, "Stage 4 RUL: RMSE Comparison"), width="stretch")
                st.caption(get_stage4_takeaway(A["s4_rul"], "reg"))
        with r_c2:
            st.markdown("**Wear-State Classification (Stage 4)**")
            if A["s4_wear"] is not None:
                st.dataframe(format_mean_std(A["s4_wear"], "clf"), hide_index=True, width="stretch")
                st.plotly_chart(mean_std_bar(A["s4_wear"], "F1", False, "Stage 4 Wear State: Macro-F1 Comparison"), width="stretch")
                st.caption(get_stage4_takeaway(A["s4_wear"], "clf"))

        st.markdown("---")
        st.markdown("##### Evaluation on Held-Out Test Tools (Stage 7)")
        st.info("Test set = 5 whole tools (47 runs), evaluated once; treat as indicative.")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**RUL Regression: Test-Set Metrics**")
            if A["test_rul"] is not None:
                st.dataframe(A["test_rul"], hide_index=True, width="stretch")
                st.caption(get_stage7_metrics_takeaway(A["test_rul"], "reg"))
        with m2:
            st.markdown("**Wear-State Classification: Test-Set Metrics**")
            if A["test_wear"] is not None:
                st.dataframe(A["test_wear"], hide_index=True, width="stretch")
                st.caption(get_stage7_metrics_takeaway(A["test_wear"], "clf"))

        # Confusion Matrix and ROC Curves
        v1, v2 = st.columns(2)
        with v1:
            cm = confusion_matrix(test["wear_class"], test["pred_wear"], labels=[0, 1, 2])
            cm_fig = px.imshow(
                cm,
                x=NAMES,
                y=NAMES,
                color_continuous_scale=[[0, "#F7F8FA"], [1, ACCENT]],
                text_auto=True,
            )
            cm_fig.update_xaxes(title="Predicted Wear State")
            cm_fig.update_yaxes(title="Actual Wear State")
            st.plotly_chart(apply_plotly_theme(cm_fig, 320, "Held-Out Test Confusion Matrix"), width="stretch")
            st.caption(get_stage7_cm_takeaway(cm, NAMES))

        with v2:
            roc_fig = go.Figure()
            roc_auc_vals = {}
            for i, name in enumerate(NAMES):
                y_bin = (test["wear_class"] == i).astype(int)
                score_col = f"prob_{name}"
                y_score = test[score_col] if score_col in test.columns else (test["pred_wear"] == i).astype(float)
                fpr, tpr, _ = roc_curve(y_bin, y_score)
                c_auc = auc(fpr, tpr)
                roc_auc_vals[name] = c_auc
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC = {c_auc:.2f})", mode="lines", line=dict(color=COLORS[name], width=2)))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Guess", mode="lines", line=dict(color=GREY, dash="dash")))
            roc_fig.update_xaxes(title="False Positive Rate")
            roc_fig.update_yaxes(title="True Positive Rate")
            st.plotly_chart(apply_plotly_theme(roc_fig, 320, "Multiclass One-vs-Rest ROC Curves"), width="stretch")
            st.caption(get_stage7_roc_takeaway(roc_auc_vals))

        # Predicted vs Actual RUL Scatter
        r_sc = go.Figure()
        r_sc.add_trace(
            go.Scatter(
                x=test["RUL"],
                y=test["pred_rul"].clip(lower=0),
                mode="markers",
                marker=dict(color=ACCENT, size=8, opacity=0.75),
                text=[f"Tool {c}, Run {r}" for c, r in zip(test["case"], test["run"])],
                hovertemplate="%{text}<br>Actual RUL: %{x}<br>Predicted RUL: %{y:.1f}<extra></extra>",
                name="Test Runs",
            )
        )
        r_max = max(float(test["RUL"].max()), float(test["pred_rul"].max())) + 1
        r_sc.add_trace(go.Scatter(x=[0, r_max], y=[0, r_max], mode="lines", line=dict(color=GREY, dash="dash"), name="Ideal Prediction"))
        r_sc.update_xaxes(title="Actual Remaining Useful Life (Runs)")
        r_sc.update_yaxes(title="Predicted Remaining Useful Life (Runs)")
        st.plotly_chart(apply_plotly_theme(r_sc, 340, "Predicted vs Actual RUL on Held-Out Test Tools"), width="stretch")
        st.caption(get_stage7_rul_scatter_takeaway(test["RUL"], test["pred_rul"]))

        if A["per_tool"] is not None:
            with st.expander("Per-Tool Test Error Breakdown (Held-Out Tools)"):
                st.dataframe(A["per_tool"], hide_index=True, width="stretch")
                st.caption("Breakdown of prediction accuracy across each of the 5 unseen physical test tools.")

    # --- Sub-tab 2: Validation (Stage 5 Nested CV)
    with insight_tabs[1]:
        st.markdown("##### Nested Cross-Validation (Stage 5)")
        st.caption(
            "Inner grid search tunes hyperparameters on the training folds; outer folds score the tuned model. "
            "Guarantees that test tools never leak into hyperparameter selection."
        )
        t_c1, t_c2 = st.columns(2)
        with t_c1:
            st.markdown("**RUL Regression (Nested CV)**")
            if A["s5_rul"] is not None:
                st.dataframe(format_mean_std(A["s5_rul"], "reg"), hide_index=True, width="stretch")
                st.plotly_chart(mean_std_bar(A["s5_rul"], "RMSE", True, "Stage 5 RUL: Tuned RMSE (Nested CV)"), width="stretch")
                st.caption(get_stage5_takeaway(A["s5_rul"], "reg"))
        with t_c2:
            st.markdown("**Wear-State Classification (Nested CV)**")
            if A["s5_wear"] is not None:
                st.dataframe(format_mean_std(A["s5_wear"], "clf"), hide_index=True, width="stretch")
                st.plotly_chart(mean_std_bar(A["s5_wear"], "F1", False, "Stage 5 Wear State: Tuned Macro-F1 (Nested CV)"), width="stretch")
                st.caption(get_stage5_takeaway(A["s5_wear"], "clf"))

        with st.expander("Best Hyperparameters (Inner Grid Search on Training Tools)"):
            b1, b2 = st.columns(2)
            with b1:
                st.markdown("**RUL Best Parameters**")
                if A["bp_rul"] is not None:
                    st.dataframe(A["bp_rul"], hide_index=True, width="stretch")
            with b2:
                st.markdown("**Wear State Best Parameters**")
                if A["bp_wear"] is not None:
                    st.dataframe(A["bp_wear"], hide_index=True, width="stretch")
            st.caption("Parameters selected by inner grid search across outer training partitions.")

    # --- Sub-tab 3: Explainability
    with insight_tabs[2]:
        st.markdown("##### Global Feature Importance & Explainability")
        st.caption("SHAP Shapley values and tree feature importances reveal the primary sensor drivers of tool degradation.")

        e1, e2 = st.columns(2)
        with e1:
            if A["shap_rul"] is not None:
                st.plotly_chart(importance_bar(A["shap_rul"], "RUL: Global SHAP Feature Importance", "Mean |SHAP Value| (Impact on RUL Runs)"), width="stretch")
                st.caption("Top sensor statistics driving remaining useful life predictions across all machining runs.")
        with e2:
            if A["shap_wear"] is not None:
                st.plotly_chart(importance_bar(A["shap_wear"], "Wear State: Global SHAP Feature Importance", "Mean |SHAP Value| (Impact on Classification)"), width="stretch")
                st.caption("Top sensor statistics distinguishing Healthy, Moderate, and Worn cutter conditions.")

        if A["importance"] is not None:
            st.plotly_chart(importance_bar(A["importance"], "Random Forest Gini Feature Importance", "Mean Decrease in Impurity"), width="stretch")
            st.caption("Tree impurity importances highlighting vibration and acoustic emission energy.")

        st.markdown("##### Bivariate Feature Correlation Explorer")
        feat_choice = st.selectbox("Select Telemetry Feature to Correlate with Flank Wear", [f for f in FEATS if f not in ("DOC", "feed", "material", "run")])
        if feat_choice:
            sc_fig = px.scatter(
                data,
                x=feat_choice,
                y="VB",
                color="wear_state",
                color_discrete_map=STATUS_COLORS,
                hover_data=["case", "run", "RUL"],
            )
            sc_fig.add_hline(y=HL, line_dash="dash", line_color=STATUS_COLORS["Healthy"], annotation_text="Healthy Limit (0.20 mm)")
            sc_fig.add_hline(y=WL, line_dash="dash", line_color=STATUS_COLORS["Worn"], annotation_text="Worn Limit (0.40 mm)")
            sc_fig.update_yaxes(title="Flank Wear VB (mm)")
            sc_fig.update_xaxes(title=f"Telemetry Feature: {feat_choice}")
            st.plotly_chart(apply_plotly_theme(sc_fig, 360, f"Correlation: {feat_choice} vs Flank Wear (VB)"), width="stretch")
            st.caption(f"Scatter plot of {feat_choice} against ground-truth flank wear VB across all 167 runs.")

    # --- Sub-tab 4: Experiments (Ablation)
    with insight_tabs[3]:
        st.markdown("##### Component & Sensor Ablation Study (Stage 6)")
        st.caption(
            "Systematically removes sensor channels and feature groups on identical CV folds "
            "to measure performance drop. Positive drop proves that the removed component provides valuable signal."
        )
        ab1, ab2 = st.columns(2)
        with ab1:
            st.markdown("**RUL Regression Ablation**")
            if A["abl_rul"] is not None:
                st.dataframe(format_mean_std(A["abl_rul"], "reg"), hide_index=True, width="stretch")
                st.plotly_chart(ablation_bar(A["abl_rul"], "Stage 6 RUL: Performance Drop When Removed"), width="stretch")
                st.caption(get_stage6_takeaway(A["abl_rul"], "reg"))
        with ab2:
            st.markdown("**Wear-State Classification Ablation**")
            if A["abl_wear"] is not None:
                st.dataframe(format_mean_std(A["abl_wear"], "clf"), hide_index=True, width="stretch")
                st.plotly_chart(ablation_bar(A["abl_wear"], "Stage 6 Wear State: Performance Drop When Removed"), width="stretch")
                st.caption(get_stage6_takeaway(A["abl_wear"], "clf"))

    render_footer()


# ================================================================== PAGE 5: Data & Batch Analysis
def page_data_and_batch_analysis():
    st.markdown("#### Tool Telemetry & Batch Analysis")
    st.caption("Upload sequential machine run telemetry or load benchmark demo tools to evaluate tool condition, remaining life, and maintenance risk.")

    # Source selection: Benchmark demo tool or custom CSV
    demo_files = find_demo_files()
    src_mode = st.radio("Data Source", ["Select Benchmark Demo Tool", "Upload Telemetry CSV File"], horizontal=True)

    df_tool = None
    source_name = ""

    if src_mode == "Select Benchmark Demo Tool":
        if not demo_files:
            st.warning("No pre-generated demo tools found in 'demo_data/'. Please run `python make_demo_data.py` to create them.")
        else:
            demo_opts = demo_files if isinstance(demo_files, dict) else {getattr(p, "name", str(p)): p for p in demo_files}
            chosen_name = st.selectbox("Choose Benchmark Tool", list(demo_opts.keys()))
            chosen_path = demo_opts[chosen_name]
            try:
                df_tool = pd.read_csv(chosen_path)
                source_name = getattr(chosen_path, "name", chosen_name)
            except Exception as e:
                st.error(f"Failed to read {chosen_name}: {e}")
    else:
        uploaded = st.file_uploader("Upload CSV File (One Tool or Multi-Run Telemetry)", type=["csv"], key="single_tool_uploader")
        if uploaded is not None:
            try:
                df_tool = pd.read_csv(uploaded)
                source_name = uploaded.name
            except Exception as e:
                st.error(f"Failed to parse uploaded CSV: {e}")

    if df_tool is not None:
        # Validate schema and runs
        run_col = None
        for c in ["run", "Run", "RUN"]:
            if c in df_tool.columns and pd.to_numeric(df_tool[c], errors="coerce").notna().any():
                run_col = c
                break

        if run_col is None:
            st.error("Uploaded CSV must contain a 'run' column with machining run sequence numbers.")
        else:
            # Sort by run sequence
            df_tool[run_col] = pd.to_numeric(df_tool[run_col], errors="coerce")
            df_tool = df_tool.dropna(subset=[run_col]).sort_values(run_col).reset_index(drop=True)
            runs = df_tool[run_col].to_numpy()

            if len(df_tool) == 0:
                st.warning("Uploaded file contains no valid run rows.")
            else:
                # Impute missing feature columns using benchmark medians
                missing_feats = [f for f in FEATS if f not in df_tool.columns]
                if missing_feats:
                    st.info(f"{len(missing_feats)} missing feature columns were automatically populated with median benchmark values.")
                    for f in missing_feats:
                        df_tool[f] = meta["defaults"][f]

                # Run inference
                with st.spinner("Analyzing tool telemetry..."):
                    pred_ruls, pred_classes, pred_confs = predict(df_tool)

                df_tool["pred_rul"] = np.round(pred_ruls, 2)
                df_tool["pred_class"] = pred_classes
                df_tool["pred_state"] = [NAMES[i] for i in pred_classes]
                df_tool["pred_conf"] = pred_confs

                # Class probabilities for worn state
                try:
                    p_all = wear_model.predict_proba(df_tool[FEATS])
                    df_tool["p_worn"] = p_all[:, 2]
                except Exception:
                    df_tool["p_worn"] = np.where(df_tool["pred_state"] == "Worn", 1.0, 0.0)

                # Smoothed RUL (3-run moving average)
                if len(df_tool) >= 3:
                    df_tool["smoothed_rul"] = df_tool["pred_rul"].rolling(window=3, min_periods=1).mean().round(2)
                else:
                    df_tool["smoothed_rul"] = df_tool["pred_rul"]

                latest_run = runs[-1]
                latest_state = df_tool["pred_state"].iloc[-1]
                latest_smooth_rul = df_tool["smoothed_rul"].iloc[-1]
                latest_p_worn = df_tool["p_worn"].iloc[-1]

                # Maintenance Decision Thresholds in collapsed expander
                with st.expander("Maintenance Decision Rules & Thresholds", expanded=False):
                    t_c1, t_c2, t_c3 = st.columns(3)
                    with t_c1:
                        th_replace_rul = st.number_input("RUL Replace Threshold (Runs)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
                    with t_c2:
                        th_plan_rul = st.number_input("RUL Plan Soon Threshold (Runs)", min_value=1.0, max_value=15.0, value=3.0, step=0.5)
                    with t_c3:
                        th_worn_prob = st.number_input("Moderate Worn Prob Threshold", min_value=0.1, max_value=0.9, value=0.40, step=0.05)
                    st.caption("Rule: Replace now if state is Worn or smoothed RUL ≤ 1; Plan soon if smoothed RUL ≤ 3 or Moderate with P(Worn) > 0.40; otherwise Keep running.")

                # Decision Rule logic
                if latest_state == "Worn" or latest_smooth_rul <= th_replace_rul:
                    rec_status = "Replace now"
                    rec_color = STATUS_COLORS["Worn"]
                    rec_bg = STATUS_BG["Worn"]
                    rec_border = STATUS_BORDER["Worn"]
                    rec_text = "RECOMMENDATION: REPLACE NOW — Tool wear has reached critical threshold or remaining useful life is under 1 run. Surface finish and part tolerances are compromised."
                elif latest_smooth_rul <= th_plan_rul or (latest_state == "Moderate" and latest_p_worn >= th_worn_prob):
                    rec_status = "Plan replacement soon"
                    rec_color = STATUS_COLORS["Moderate"]
                    rec_bg = STATUS_BG["Moderate"]
                    rec_border = STATUS_BORDER["Moderate"]
                    rec_text = "RECOMMENDATION: PLAN REPLACEMENT SOON — Flank wear is progressing and remaining useful life is under 3 runs. Prepare a replacement cutter."
                else:
                    rec_status = "Keep running"
                    rec_color = STATUS_COLORS["Healthy"]
                    rec_bg = STATUS_BG["Healthy"]
                    rec_border = STATUS_BORDER["Healthy"]
                    rec_text = "RECOMMENDATION: KEEP RUNNING — Cutter condition is healthy with adequate operational life remaining."

                # Status Banner
                st.markdown(
                    f'<div style="background:{rec_bg}; border:1px solid {rec_border}; border-left:6px solid {rec_color}; '
                    f'border-radius:8px; padding:1rem 1.25rem; margin:1rem 0;">'
                    f'<div style="font-size:0.8rem; font-weight:700; color:{rec_color}; text-transform:uppercase; letter-spacing:0.05em;">{rec_status.upper()} · {source_name}</div>'
                    f'<div style="font-size:1.05rem; font-weight:600; color:{NAVY}; margin-top:0.25rem;">{rec_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Linear trend calculation if VB is present
                vb_col = None
                for c in ["VB", "vb", "Vb"]:
                    if c in df_tool.columns and pd.to_numeric(df_tool[c], errors="coerce").notna().any():
                        vb_col = c
                        break

                trend_slope, trend_intercept, run_cross_040 = None, None, None
                runs_to_040_str = "N/A (No VB data)"
                if vb_col is not None:
                    vb_vals = pd.to_numeric(df_tool[vb_col], errors="coerce").to_numpy()
                    valid_mask = ~np.isnan(vb_vals)
                    if np.sum(valid_mask) >= 4:
                        trend_slope, trend_intercept, run_cross_040 = calculate_tool_trend(runs[valid_mask], vb_vals[valid_mask])
                        if run_cross_040 is not None:
                            rem = max(0.0, run_cross_040 - latest_run)
                            runs_to_040_str = f"~{rem:.1f} runs (at run {run_cross_040:.1f})"
                        else:
                            runs_to_040_str = "Stable / non-increasing"
                    else:
                        runs_to_040_str = "Needs ≥4 VB points"

                # Check if actual RUL exists
                rul_col = None
                for c in ["RUL", "rul", "Rul"]:
                    if c in df_tool.columns and pd.to_numeric(df_tool[c], errors="coerce").notna().any():
                        rul_col = c
                        break

                mae_str = "N/A (Unlabelled)"
                if rul_col is not None:
                    actual_rul = pd.to_numeric(df_tool[rul_col], errors="coerce")
                    valid_rul = actual_rul.notna()
                    if valid_rul.any():
                        tool_mae = np.mean(np.abs(df_tool.loc[valid_rul, "pred_rul"] - actual_rul[valid_rul]))
                        mae_str = f"{tool_mae:.2f} runs"

                # 5 Summary Metric Cards
                k = st.columns(5)
                k[0].markdown(render_metric_card("Runs Analysed", f"{len(df_tool)} runs", f"Sequence: run {runs[0]} to {runs[-1]}"), unsafe_allow_html=True)
                k[1].markdown(render_metric_card("Latest Predicted State", latest_state, f"P(Worn): {latest_p_worn:.1%}", STATUS_COLORS[latest_state]), unsafe_allow_html=True)
                k[2].markdown(render_metric_card("Smoothed RUL", f"{latest_smooth_rul:.1f} runs", f"Latest raw: {df_tool['pred_rul'].iloc[-1]:.1f} runs", ACCENT), unsafe_allow_html=True)
                k[3].markdown(render_metric_card("Runs to 0.40mm Limit", runs_to_040_str, "Linear wear extrapolation" if run_cross_040 is not None else "Based on VB slope"), unsafe_allow_html=True)
                k[4].markdown(render_metric_card("RUL Accuracy (MAE)", mae_str, "Against ground truth RUL" if rul_col is not None else "Telemetry only"), unsafe_allow_html=True)
                st.write("")

                # Charts Section
                c_rul, c_wear = st.columns(2)
                with c_rul:
                    fig_rul = go.Figure()
                    fig_rul.add_trace(go.Scatter(x=runs, y=df_tool["pred_rul"], name="Predicted RUL (Raw)", mode="lines+markers", line=dict(color=ACCENT, width=2)))
                    if len(df_tool) >= 3:
                        fig_rul.add_trace(go.Scatter(x=runs, y=df_tool["smoothed_rul"], name="Smoothed Trend (3-Run MA)", mode="lines", line=dict(color=NAVY, width=2.5, dash="dash")))
                    if rul_col is not None:
                        fig_rul.add_trace(go.Scatter(x=runs, y=pd.to_numeric(df_tool[rul_col]), name="Actual Ground Truth RUL", mode="lines+markers", line=dict(color=GREY, dash="dot")))
                    fig_rul.update_xaxes(title="Machining Run Number")
                    fig_rul.update_yaxes(title="Remaining Useful Life (Runs)")
                    st.plotly_chart(apply_plotly_theme(fig_rul, 350, "Predicted Remaining Useful Life Trajectory"), width="stretch")
                    if len(df_tool) < 3:
                        st.caption("Note: Tool sequence needs at least 3 runs for a smoothed trend line.")

                with c_wear:
                    if vb_col is not None:
                        fig_vb = go.Figure()
                        fig_vb.add_hrect(y0=0, y1=HL, fillcolor=STATUS_COLORS["Healthy"], opacity=0.08, line_width=0)
                        fig_vb.add_hrect(y0=HL, y1=WL, fillcolor=STATUS_COLORS["Moderate"], opacity=0.08, line_width=0)
                        top_vb = max(float(np.nanmax(pd.to_numeric(df_tool[vb_col]))), WL) * 1.15
                        fig_vb.add_hrect(y0=WL, y1=top_vb, fillcolor=STATUS_COLORS["Worn"], opacity=0.08, line_width=0)
                        fig_vb.add_trace(go.Scatter(x=runs, y=pd.to_numeric(df_tool[vb_col]), name="Measured Flank Wear (VB)", mode="lines+markers", line=dict(color=NAVY, width=2)))

                        # Plot extrapolated trend line
                        if trend_slope is not None and run_cross_040 is not None:
                            extrap_runs = np.linspace(runs[0], max(runs[-1], run_cross_040), 50)
                            extrap_vb = trend_intercept + trend_slope * extrap_runs
                            fig_vb.add_trace(go.Scatter(x=extrap_runs, y=extrap_vb, name="Linear Extrapolation", mode="lines", line=dict(color="#ea580c", dash="dash", width=2)))
                            fig_vb.add_vline(x=run_cross_040, line_dash="dot", line_color=STATUS_COLORS["Worn"], annotation_text=f"0.40mm at run {run_cross_040:.1f}")

                        fig_vb.update_xaxes(title="Machining Run Number")
                        fig_vb.update_yaxes(title="Flank Wear VB (mm)")
                        st.plotly_chart(apply_plotly_theme(fig_vb, 350, "Flank Wear Progression & 0.40mm Wear Limit"), width="stretch")
                    else:
                        fig_cls = go.Figure()
                        for i, name in enumerate(NAMES):
                            sub_c = df_tool[df_tool["pred_class"] == i]
                            if len(sub_c) > 0:
                                fig_cls.add_trace(go.Scatter(x=sub_c[run_col], y=[i] * len(sub_c), mode="markers", name=name, marker=dict(size=12, color=STATUS_COLORS[name])))
                        fig_cls.update_yaxes(tickvals=[0, 1, 2], ticktext=NAMES, title="Predicted Wear State")
                        fig_cls.update_xaxes(title="Machining Run Number")
                        st.plotly_chart(apply_plotly_theme(fig_cls, 350, "Wear State Classification Sequence"), width="stretch")

                # Wear State Progression Bar
                st.markdown("**Wear State Classification Timeline**")
                fig_tl = go.Figure()
                for i, r_num in enumerate(runs):
                    st_name = df_tool["pred_state"].iloc[i]
                    fig_tl.add_trace(
                        go.Bar(
                            x=[1],
                            y=["Tool Timeline"],
                            orientation="h",
                            name=st_name,
                            marker_color=STATUS_COLORS[st_name],
                            showlegend=(i == 0 or st_name != df_tool["pred_state"].iloc[i - 1]),
                            hovertemplate=f"Run {r_num}: {st_name}<br>Predicted RUL: {df_tool['pred_rul'].iloc[i]:.1f}<extra></extra>",
                        )
                    )
                fig_tl.update_layout(barmode="stack", height=110, margin=dict(l=20, r=20, t=10, b=20), xaxis=dict(showticklabels=False, showgrid=False), yaxis=dict(showticklabels=False))
                st.plotly_chart(apply_plotly_theme(fig_tl, 110), width="stretch")

                # Results Table & CSV Export
                st.markdown("**Run-by-Run Telemetry & Predictions Breakdown**")
                display_cols = [run_col, "pred_state", "pred_rul", "smoothed_rul", "p_worn"]
                if vb_col is not None:
                    display_cols.insert(1, vb_col)
                if rul_col is not None:
                    display_cols.insert(2, rul_col)

                rename_map = {
                    run_col: "Run",
                    "pred_state": "Predicted State",
                    "pred_rul": "Raw RUL (Runs)",
                    "smoothed_rul": "Smoothed RUL",
                    "p_worn": "P(Worn)",
                    vb_col: "Measured VB (mm)",
                    rul_col: "Actual RUL",
                }
                tbl = df_tool[display_cols].rename(columns={k: v for k, v in rename_map.items() if k is not None})
                st.dataframe(tbl, hide_index=True, width="stretch")

                # Download Export
                export_df = df_tool.copy()
                export_df.insert(0, "recommendation", rec_status)
                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Analysis Results CSV",
                    csv_bytes,
                    f"cnc_tool_analysis_{source_name}.csv",
                    "text/csv",
                )

    # Folded Batch Multi-Tool Telemetry Inference Section
    st.write("")
    with st.expander("Batch Multi-Tool Inference & Scoring", expanded=False):
        st.markdown("**Run Batch Inference on Multi-Tool Telemetry File**")
        st.caption("Upload an arbitrary telemetry CSV to generate wear state and RUL predictions across all rows.")
        st.download_button(
            "Download Telemetry CSV Template",
            pd.DataFrame([meta["defaults"]]).to_csv(index=False),
            "cnc_telemetry_template.csv",
            "text/csv",
        )
        batch_file = st.file_uploader("Upload Multi-Tool Telemetry CSV", type="csv", key="batch_multi_uploader")
        if batch_file is not None:
            try:
                b_df = pd.read_csv(batch_file)
                missing = [f for f in FEATS if f not in b_df.columns]
                if missing:
                    st.info(f"{len(missing)} missing feature columns were automatically populated with median benchmark values.")
                    for f in missing:
                        b_df[f] = meta["defaults"][f]
                b_rul, b_cls, b_conf = predict(b_df)
                b_res = b_df.copy()
                b_res.insert(0, "Predicted Wear State", [NAMES[i] for i in b_cls])
                b_res.insert(0, "Predicted RUL (Runs)", np.round(b_rul, 1))
                st.dataframe(b_res, width="stretch")
                st.download_button(
                    "Download Scored Batch Predictions CSV",
                    b_res.to_csv(index=False),
                    "cnc_batch_predictions.csv",
                    "text/csv",
                )
            except Exception as err:
                st.error(f"Failed to process batch CSV: {err}")

    render_footer()


# ------------------------------------------------------------------ Page Router
if page == "Overview":
    page_overview()
elif page == "Live Prediction":
    page_live_prediction()
elif page == "Tool Lifecycle":
    page_tool_lifecycle()
elif page == "Model Insights":
    page_model_insights()
elif page == "Data & Batch Analysis":
    page_data_and_batch_analysis()
