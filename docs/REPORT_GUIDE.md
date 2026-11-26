# Report guide: turning the outputs into the IEEE paper (10 marks)

The report is a 3 to 6 page paper in IEEE conference format. Marks are for the quality of the paper. The
tables and figures below are produced by `python train.py`; the **writing, the literature review and the
references must be yours**, in your own words (the rubric checks plagiarism below 15% and penalises unedited
AI-generated text).

## Section by section

| Rubric item (marks) | What to write | Use these outputs |
|---|---|---|
| Abstract and problem statement (1) | 150 to 250 words: the problem (tool wear causes scrap and downtime), objective, method, main result with numbers. 4 to 6 keywords. | `outputs/run_summary.json` (final numbers) |
| Literature review and research gap (2) | Review **10 to 15 papers from the last 5 years** on tool wear / RUL prediction with machine learning. Put them in a table (reference, data, method, result, limitation). State the gap and your contribution in 2 to 3 sentences. | Your own search (see below) |
| Methodology and experimental setup (2) | Dataset, labels (VB thresholds, RUL definition), features, split by tool, preprocessing pipeline, the 6 models and 2 baselines, nested-CV tuning, ablation. Detailed enough that someone could repeat it. Add a block diagram of the 8 stages. | `config.yaml`, `outputs/stage1_*`, `stage2_*`, `stage3_pipeline.txt` |
| Results and discussion (2) | Comparison table (mean ± std), ablation, test-set results, confusion matrix, ROC, error analysis, SHAP. Be honest about limits (small number of tools, small test set). | `outputs/results_tables.md` and the PNG figures |
| IEEE format and references (2) | IEEE template, numbered figures and tables, **15 to 25** IEEE-style references, each cited in the text. | |
| Originality and language (1) | Own wording, clear grammar, run a plagiarism check. | |

## Figures worth including (all in `outputs/`)

1. `stage1_wear_curves.png`: the problem in one picture.
2. `stage4_rul_baselines.png` and `stage4_wear_baselines.png`: models vs baselines.
3. `stage6_*_ablation.png`: which sensors matter.
4. `stage7_wear_confusion_matrix.png`, `stage7_wear_roc.png`, `stage7_rul_pred_vs_actual.png`.
5. `stage8_shap_rul.png`: explainability.

## Finding the literature (real papers only)

- Search IEEE Xplore, Google Scholar, ScienceDirect, MDPI. Useful phrases: "tool wear prediction milling",
  "remaining useful life cutting tool", "NASA milling dataset tool wear", "machine learning tool condition
  monitoring", "SHAP predictive maintenance".
- Keep papers from the last 5 years, read the abstract, method and limitations, and write each summary yourself.
- Copy the citation details from the publisher page (or use "Cite" -> IEEE) so the references are exact.
  Never cite a paper you have not opened.
- Cite the dataset itself (NASA Prognostics Data Repository, milling data from UC Berkeley) and the tools you
  used (scikit-learn, XGBoost, SHAP).

## Pre-submission checklist

- [ ] Abstract is 150 to 250 words, with keywords
- [ ] 10 to 15 recent papers reviewed, gap and contribution stated
- [ ] Methodology is repeatable and has a block diagram
- [ ] Every number in the paper matches `outputs/results_tables.md` from a NON-quick run
- [ ] Results table shows mean ± std; ablation and error analysis included; limitations stated honestly
- [ ] IEEE template, numbered figures and tables, 15 to 25 references cited in the text
- [ ] Written in your own words; plagiarism report below 15%
