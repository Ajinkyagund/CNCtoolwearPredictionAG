# AI-Based CNC Tool Wear Classification and Remaining Useful Life (RUL) Prediction

Predictive-maintenance mini project (Practical No. 07). A milling cutter wears out with use. Replace it too
early and you waste tool life; too late and you get scrap parts and machine downtime. From sensor signals this
project

1. **classifies the wear state** of the tool (Healthy / Moderate / Worn), and
2. **predicts the Remaining Useful Life (RUL)** in runs,

and serves both through a Streamlit dashboard.

The code follows the course's **8-stage ML pipeline**. Stages 1 to 6 use only the training tools; the held-out
test tools are used **once**, in stage 7.

---

## Rubric to project map

| Stage | What the rubric asks | What this project does | Where |
|---|---|---|---|
| 1 Data and EDA | Missing values, class balance, correlations | Dataset summary table, wear curves, class-balance and correlation plots | `cncwear/data.py` → `outputs/stage1_*` |
| 2 Split | Stratified split, test set untouched, fixed seed | Whole **tools** are held out (about 25%), stratified by wear class with `StratifiedGroupKFold`, seed 42 | `cncwear/split.py` → `outputs/stage2_*` |
| 3 Preprocess | Imputation, scaling inside a Pipeline | `SimpleImputer` + constant-feature filter + `StandardScaler` inside every model's `Pipeline`, so each CV fold fits them on its own training part | `cncwear/models.py` → `outputs/stage3_pipeline.txt` |
| 4 Baselines | 3+ models plus a simple baseline, repeated stratified CV, mean ± std | 6 models per task plus 2 baselines, 4 folds x 3 repeats, every metric as mean ± std | `cncwear/validation.py` → `outputs/stage4_*` |
| 5 Tuning | GridSearch/Optuna, nested CV, fixed seed | Grid search inside **nested** CV (inner folds tune, outer folds score), then best parameters on the full training set | `cncwear/validation.py` → `outputs/stage5_*` |
| 6 Ablation | Remove one component at a time | Drops one sensor group / feature type at a time on identical folds, reports the performance drop | `cncwear/ablation.py` → `outputs/stage6_*` |
| 7 Test | Evaluate once: metrics, confusion matrix, ROC | One evaluation on the test tools: accuracy / precision / recall / F1 / ROC-AUC, confusion matrix, ROC curves, per-class and per-tool error analysis | `cncwear/pipeline.py` → `outputs/stage7_*` |
| 8 Deliver | Explainability, repo, README, demo | SHAP explanations, this repo with tests and CI, a 7-tab dashboard | `cncwear/explain.py`, `app.py` |

The marking scheme's "Code quality, reproducibility, demo" row is covered by: modular `cncwear/` package,
`config.yaml` (all settings in one place), `requirements.txt` + `requirements-lock.txt`, a pytest suite
(`tests/`), a GitHub Actions workflow and the dashboard.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

1. Download the **NASA / UC Berkeley Milling Wear** dataset (search "NASA milling data set" on the NASA
   Prognostics Data Repository, or "Milling Tool Wear" on Kaggle) and put it in `data/` as `mill.mat`
   (recommended) or `mill.csv` (columns `case`, `run`, `VB` plus numeric sensor columns).
2. Train and evaluate everything: `python train.py` (a few minutes). Read the console tables and
   `outputs/results_tables.md`.
3. Open the dashboard: `streamlit run app.py`.
4. Check the code: `pip install -r requirements-dev.txt` then `pytest`.

`python train.py --quick` is a fast smoke test with reduced settings. **Never report numbers from quick mode.**

**Google Colab:** upload the project zip and use `colab_run.ipynb` (one cell per step, including the app).

## About the data

16 tools ("cases") were run until worn, 167 runs in total. Each run has the cutting parameters (depth of cut
`DOC`, `feed`, `material`), six sensor signals (spindle motor current `smcAC` / `smcDC`, table and spindle
vibration, table and spindle acoustic emission) and the measured flank wear `VB` (mm). Each raw signal (9000
samples) is summarised into 8 statistics (mean, std, RMS, max, min, peak-to-peak, skewness, kurtosis).

- **Wear state:** VB below 0.20 mm = Healthy, 0.20 to 0.40 = Moderate, 0.40 or more = Worn.
- **RUL:** the last recorded run of a tool minus the current run.
- Both definitions, the seed, the split size, CV settings and ablation groups are in `config.yaml`.
- The elapsed-time column `time` is **not** used as a feature because it reveals the tool's life directly.

## Why the split is by tool (important for the viva)

Runs of the same tool are highly correlated. A random row split would put near-copies of the same tool in
train and test and report fake-high scores. So the test set, every CV fold and every tuning fold keep each tool
on one side only, while `StratifiedGroupKFold` keeps the wear-class mix similar across folds. With only 16 tools,
the test set holds out exactly 5 whole tools (tools 3, 7, 10, 14, 16; 47 runs), so the test numbers are indicative;
the repeated CV mean ± std on the 11 training tools shows how stable they are.

## Dashboard pages

The dashboard is structured into 5 clean, focused pages:

| Page | What it shows |
|---|---|
| Overview | Executive summary with 4 KPI cards (test Macro-F1, test RUL MAE/RMSE, runs/tools count, 5 held-out test tools), flank wear progression curves, class balance distribution, and collapsible 8-stage ML pipeline map |
| Live Prediction | Telemetry presets (Fresh, Mid-life, Worn), interactive sensor/cutting parameter controls, gauge chart for RUL, predicted wear state with class probabilities, actionable maintenance advice, and local SHAP explanation |
| Tool Lifecycle | Interactive step-through of any tool's lifecycle comparing actual vs predicted RUL and wear state, with test tools clearly highlighted |
| Model Insights | Consolidated deep dive across 4 tabs: **Performance** (baselines vs candidate models, held-out test metrics, confusion matrix, ROC curve), **Validation** (nested-CV tuning, best hyperparameters, fold stability), **Explainability** (global SHAP summary, Random Forest feature importance, bivariate feature correlation explorer), and **Experiments** (sensor group ablation and component impact) |
| Data & Batch Analysis | Upload single-tool or multi-run telemetry CSVs or select pre-generated benchmark demo tools; displays maintenance risk banner, 5 KPI metric cards, smoothed RUL trajectory, linear flank wear extrapolation to 0.40 mm, and CSV export |

### How to use the Data & Batch Analysis page
1. Navigate to **Data & Batch Analysis** from the sidebar navigation.
2. Select a benchmark tool from the **Load Demo Tool** dropdown (e.g. `tool_03.csv`), or upload a telemetry CSV containing sequential runs with a `run` column (and optional `VB` flank wear / `RUL` columns).
3. The dashboard automatically validates the schema, imputes missing features with benchmark median defaults, and computes:
   - **Maintenance status banner**: Evaluates operational risk ("Replace now", "Plan replacement soon", "Keep running") according to transparent, adjustable decision rules.
   - **5 Summary KPI cards**: Displays runs analysed, latest predicted wear state, smoothed RUL (3-run moving average), estimated runs remaining until the 0.40 mm wear limit, and MAE error (if actual RUL is provided).
   - **Telemetry trajectory charts**: RUL trend with moving average smoothing, measured flank wear with linear regression extrapolation to 0.40 mm, and wear-state progression timeline.
   - **Results table & export**: Full run breakdown with a **Download Analysis Results CSV** export button.

### How to regenerate demo data
To regenerate the benchmark demo CSV files from `outputs/processed_features.csv`:
```bash
python make_demo_data.py
```
This generates individual CSV files for each held-out test tool (`tool_03.csv`, `tool_07.csv`, `tool_10.csv`, `tool_14.csv`, `tool_16.csv`) and a 5-run partial tool (`tool_XX_partial.csv`) in `demo_data/`.

## Project structure

```
cnc-wear-rul/
├── config.yaml            # thresholds, seed, split, CV, ablation groups
├── train.py               # entry point: runs the 8 stages
├── app.py                 # Streamlit dashboard
├── cncwear/
│   ├── data.py            # loading, labels, EDA          (stage 1)
│   ├── split.py           # stratified group hold-out     (stage 2)
│   ├── models.py          # Pipeline, models, baselines   (stages 3-4)
│   ├── validation.py      # repeated + nested CV, tuning  (stages 4-5)
│   ├── ablation.py        #                               (stage 6)
│   ├── metrics.py, plots.py
│   ├── explain.py         # SHAP                          (stage 8)
│   └── pipeline.py        # orchestrates all stages
├── tests/                 # pytest suite
├── docs/REPORT_GUIDE.md   # how to turn the outputs into the IEEE paper
├── colab_run.ipynb        # step-by-step Colab runner
├── data/                  # put mill.mat here (not committed)
├── models/                # created by train.py
└── outputs/               # created by train.py: tables, figures, results_tables.md
```

## Results (fill in after `python train.py`)

Copy the final numbers from `outputs/results_tables.md` (it already contains every table in markdown).

| Task | Chosen model | Nested-CV (mean ± std) | Held-out test |
|---|---|---|---|
| RUL regression | | RMSE = | MAE = , RMSE = , R² = |
| Wear-state classification | | Macro-F1 = | Accuracy = , Macro-F1 = , ROC-AUC = |

## Limitations (say these yourself in the viva)

- Only 16 tools and one machine setup, so results may not transfer to other machines, materials or tools.
- The test set is 5 whole tools (47 runs); scores on it have wide uncertainty and should be treated as indicative.
- Features are summary statistics; frequency-domain features (FFT) or a deep model (LSTM / 1D-CNN) could help.
- RUL is the number of runs until the end of the recorded experiment, not until a physical failure.
- The dashboard works on entered or uploaded readings, not on a live sensor stream.

## Ideas to extend

FFT / wavelet features, an LSTM or 1D-CNN on raw signals, prediction intervals for RUL, a cost-based
replace-now vs keep-running recommendation, validation on a second dataset (PHM 2010).

## Suggested Git history (the rubric looks at it)

Commit in small steps so the history tells the story, for example:
`Add data loading and EDA` → `Add stratified group split` → `Add preprocessing pipeline and baselines` →
`Add nested CV tuning` → `Add ablation study` → `Add test evaluation and ROC` → `Add SHAP explanations` →
`Add Streamlit dashboard` → `Add tests and CI` → `Final results and README`.
