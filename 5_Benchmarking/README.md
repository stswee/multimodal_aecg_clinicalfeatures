# MUSIC four-year structured clinical baselines

This directory contains the final leakage-controlled structured-variable benchmark for predicting four-year sudden cardiac death (SCD) and pump-failure death (PFD) in the MUSIC cohort.

The corrected analysis is complete. It includes anticoagulant/antithrombotic use in both prompt-matched feature sets and writes results to `tabular_nested_4year_v2`.

## Files

| File | Purpose |
|---|---|
| `train_tabular_nested_cv.py` | Runs cohort and feature audits, nested tuning, outer-fold fitting, fold-specific Platt calibration, threshold selection, bootstrap confidence intervals, and paired comparisons. |
| `run_tabular_nested_cv.sh` | Starts the complete nested-CV analysis in a detached tmux session. |
| `tabular_posthoc_plots.py` | Generates ROC, precision-recall, calibration, and exploratory decision-curve figures from saved outer-test predictions. It does not retrain or recalibrate models. |
| `missing_sentinels.example.json` | Optional template for verified dataset-specific missing-value codes. |
| `README.md` | Documents the analysis, execution procedure, outputs, and principal results. |

## Final analysis status

- Patients: 730.
- No cardiac death by four years: 577.
- SCD by four years: 71.
- PFD by four years: 82.
- Outer folds: 5.
- Inner folds per outer-training cohort: 4.
- Random seed: 42.
- Bootstrap replicates: 5,000.
- Completed final models: 30 = 2 feature sets × 3 model arms × 5 outer folds.
- Outer-test outcomes used for preprocessing, tuning, calibration, or threshold selection: no.

For the SCD analysis, PFD patients are masked, leaving 648 applicable patients and 71 SCD events. For the PFD analysis, SCD patients are masked, leaving 659 applicable patients and 82 PFD events.

## Feature sets

### Primary: `prompt_matched_no_ecg`

The primary feature set contains the baseline structured variables supplied to the corrected no-ECG language-model prompt:

- Demographics.
- NYHA class and blood pressure.
- Heart-failure etiology and medical history.
- Laboratory measurements.
- Left ventricular ejection fraction.
- Baseline medications.

It excludes:

- Holter ECG impressions.
- Cause of death.
- Four-year outcome labels.
- Follow-up duration.
- Study-exit and transplantation information.
- Post-baseline events.

### Sensitivity analysis: `prompt_matched_with_ecg`

This feature set contains all primary variables plus five Holter-derived clinical impressions:

- Ventricular extrasystole.
- Ventricular tachycardia.
- Non-sustained ventricular tachycardia.
- Paroxysmal supraventricular tachyarrhythmia.
- Bradycardia.

This evaluates whether the Holter impressions add predictive information and provides an ECG-overlap sensitivity analysis.

## Corrected anticoagulant field

The exact MUSIC source column is:

```text
Anticoagulants/antitrombotics  (yes=1)
```

There are two spaces before `(yes=1)`. The corrected script includes this variable in both feature sets and stops before model fitting unless the locked cohort contains:

```text
Yes:     610
No:      120
Missing:   0
```

The audit is saved to:

```text
analysis_setup/anticoagulant_feature_audit.csv
```

## Validation design

The analysis uses the same patient-level nested cross-validation assignments as the ECG and text analyses:

- Five stratified outer folds provide final performance evaluation.
- Four inner folds within each outer-training cohort perform all model development.
- Each patient appears in exactly one outer test fold.

The corresponding outer test fold is not used for:

- Missing-value imputation.
- Missingness-indicator construction.
- Categorical encoding.
- Continuous-variable scaling.
- Class-weight calculation.
- Hyperparameter selection.
- Model-family selection.
- Platt calibration.
- Classification-threshold selection.

## Outcome handling

SCD and PFD are fitted as separate task-specific classifiers using the same patient folds.

For SCD:

- Patients without cardiac death are controls.
- Patients with SCD are cases.
- Patients with PFD are masked.

For PFD:

- Patients without cardiac death are controls.
- Patients with PFD are cases.
- Patients with SCD are masked.

Thus, a competing cardiac-death endpoint is not treated as a negative label for the other task.

## Missing-data handling

No patient is excluded because an individual predictor is missing.

Within each training split:

- Continuous variables use median imputation.
- Continuous-variable missingness indicators are added.
- Continuous variables are standardized.
- Categorical missing values use an explicit missing category.
- Categorical variables are one-hot encoded.
- Previously unseen validation or test categories are ignored safely.

Every fitted transformation is applied unchanged to its validation or test patients.

The script does not assume that values such as `999`, `-9`, or `-1` are missing. Verified dataset-specific sentinel values can be supplied with:

```bash
--sentinel_json /path/to/missing_sentinels.json
```

Example:

```json
{
  "Example feature": [999, -9]
}
```

Do not define sentinel values without confirmation from the data documentation.

## Candidate models

### L2-penalized logistic regression

The inverse regularization strength is selected from:

```text
C ∈ {0.001, 0.01, 0.1, 1, 10, 100}
```

This produces six logistic-regression candidates.

### Histogram gradient boosting

The grid crosses:

```text
Learning rate:          {0.03, 0.10}
Maximum leaf nodes:     {7, 15}
L2 regularization:      {0, 1}
Minimum samples/leaf:   20
Maximum iterations:     300
```

This produces eight gradient-boosting candidates.

### Total grid

```text
6 logistic candidates + 8 gradient-boosting candidates = 14 candidates
```

## Hyperparameter selection

Each candidate is evaluated using pooled inner out-of-fold predictions. The primary selection score is:

```text
0.5 × (SCD ROC-AUC + PFD ROC-AUC)
```

Within each outer-training cohort:

1. Identify the maximum pooled inner mean ROC-AUC.
2. Retain candidates within 0.005 of the maximum.
3. Select the candidate with the lowest prespecified complexity.
4. Break remaining ties using the highest pooled mean PR-AUC.
5. Use configuration name as the final deterministic tie-breaker.

Three model arms are retained:

- `penalized_logistic`.
- `gradient_boosting`.
- `selected_tabular`, selected across both model families.

In the final corrected run, L2-penalized logistic regression with `C=0.001` was selected in all five outer folds for both feature sets. Therefore, `selected_tabular` and `penalized_logistic` have identical final predictions in this run.

## Class imbalance

For each task and training split, positive examples receive weight:

```text
number of negative patients / number of positive patients
```

Weights are recalculated within every training split and supplied to both model families as sample weights.

## Calibration

Separate Platt mappings are fitted for SCD and PFD within each outer fold:

1. Obtain cross-fitted inner out-of-fold probabilities from the outer-training cohort.
2. Fit the Platt mapping using those probabilities and outcomes.
3. Fit the selected predictive model using the full outer-training cohort.
4. Apply the saved mapping unchanged to the corresponding outer test fold.

Calibrated probabilities are used for:

- Brier score.
- Calibration intercept.
- Calibration slope.
- Calibration-in-the-large.
- Threshold-specific classification.
- Exploratory decision-curve analysis.

## Classification thresholds

One threshold is selected per outer fold and outcome by maximizing Youden's J on calibrated inner out-of-fold predictions:

```text
Youden's J = sensitivity + specificity − 1
```

Ties are resolved by higher sensitivity and then the lower threshold. The selected threshold is applied unchanged to the corresponding outer-test patients.

These thresholds are statistical operating points, not validated clinical intervention thresholds.

## Confidence intervals and paired comparisons

The analysis uses 5,000 patient-level bootstrap replicates. The same sampled patients are used for both models in every paired comparison.

Reported quantities include:

- ROC-AUC and PR-AUC.
- Calibrated Brier score.
- Calibration intercept and slope.
- Calibration-in-the-large.
- Sensitivity, specificity, PPV, NPV, accuracy, F1, and balanced accuracy.
- Confusion-matrix counts.
- Paired differences with 95% confidence intervals and p-values.
- Holm-adjusted p-values across prespecified tabular comparisons.

ROC-AUC and PR-AUC comparisons use uncalibrated outer-test scores. Brier-score comparisons use calibrated probabilities.

## Prerequisites

The scripts require:

- Python 3.
- pandas.
- NumPy.
- SciPy.
- scikit-learn.
- joblib.
- matplotlib.
- tmux for detached execution.

The launcher activates:

```text
shdb-af-analysis
```

GPUs are not required.

## Directory structure

```text
5_Benchmarking/
├── train_tabular_nested_cv.py
├── run_tabular_nested_cv.sh
├── tabular_posthoc_plots.py
├── missing_sentinels.example.json
└── README.md

/home/sswee/music/
├── subject-info.csv
├── ecg_nested_4year_three_wave/
│   └── analysis_setup/
│       └── nested_patient_folds.csv
└── tabular_nested_4year_v2/
```

## Run the nested-CV benchmark

```bash
cd /home/sswee/multimodal_aecg_clinicalfeatures/4_LLM_Modeling

chmod +x train_tabular_nested_cv.py run_tabular_nested_cv.sh

./run_tabular_nested_cv.sh
```

The launcher creates:

```text
tmux session: tabular_nested_4year_v2
output root:  /home/sswee/music/tabular_nested_4year_v2
```

Monitor with:

```bash
tmux attach -t tabular_nested_4year_v2
```

or:

```bash
tail -f \
  /home/sswee/music/tabular_nested_4year_v2/launcher_logs/tabular_nested_cv.log
```

## Verify completion

```bash
TAB_ROOT="/home/sswee/music/tabular_nested_4year_v2"

find "${TAB_ROOT}/final_models" \
  -name run_complete.json | wc -l

python -m json.tool \
  "${TAB_ROOT}/evaluation/evaluation_manifest.json"

cat \
  "${TAB_ROOT}/analysis_setup/anticoagulant_feature_audit.csv"
```

Expected model count:

```text
30
```

## Generate ROC/PR, calibration, and decision-curve figures

The post hoc script reads only the completed pooled outer-test predictions. It does not retrain models, recalibrate predictions, or select new thresholds.

```bash
cd /home/sswee/multimodal_aecg_clinicalfeatures/4_LLM_Modeling

chmod +x tabular_posthoc_plots.py

python tabular_posthoc_plots.py \
  --tabular_root /home/sswee/music/tabular_nested_4year_v2 \
  --bootstrap_replicates 5000 \
  --seed 42
```

The script uses:

- Uncalibrated outer-test scores for ROC and precision-recall curves.
- Fold-specific Platt-calibrated outer-test probabilities for calibration plots and decision curves.
- Five equal-sized risk groups with Wilson 95% intervals for calibration plots.
- Treat-all and treat-none strategies for decision curves.
- A 2%–25% exploratory threshold range with 93 grid points.
- Paired pointwise patient-bootstrap confidence intervals for net-benefit differences.

Decision-curve intervals are pointwise, not simultaneous, and are not adjusted for examining multiple thresholds. No clinical action was prespecified; therefore, the decision curves are exploratory and do not establish clinical utility.

To replace an existing post hoc output directory intentionally, add:

```bash
--overwrite
```

## Main nested-CV outputs

| Location under `tabular_nested_4year_v2` | Contents |
|---|---|
| `analysis_setup/analysis_manifest.json` | Input hashes, final feature definitions, cohort counts, software version, and leakage-control metadata. |
| `analysis_setup/anticoagulant_feature_audit.csv` | Corrected anticoagulant-field counts. |
| `analysis_setup/feature_missingness.csv` | Missing counts and percentages for each predictor. |
| `tuning/` | Candidate-by-outer-fold inner-loop results. |
| `selection/` | Fold-specific logistic, boosting, and overall selections. |
| `final_models/` | Model bundles, inner OOF predictions, and untouched outer-test predictions. |
| `evaluation/fold_specific_platt_parameters.csv` | Fold-specific Platt parameters fitted from inner OOF predictions. |
| `evaluation/fold_specific_selected_thresholds.csv` | Fold-specific thresholds selected from calibrated inner OOF predictions. |
| `evaluation/pooled_performance_calibration_with_95ci.csv` | Discrimination, Brier, and calibration estimates with 95% CIs. |
| `evaluation/pooled_threshold_metrics_with_95ci.csv` | Threshold metrics with 95% CIs. |
| `evaluation/pooled_confusion_matrix_counts.csv` | Pooled TN, FP, FN, and TP counts. |
| `evaluation/paired_tabular_model_comparisons_with_95ci.csv` | Paired differences and Holm-adjusted p-values. |
| `evaluation/models/` | Patient-level pooled outer-test predictions for each feature set and model arm. |
| `evaluation/evaluation_manifest.json` | Completion and leakage-control metadata. |

## Plotting outputs

The plotting script writes to:

```text
/home/sswee/music/tabular_nested_4year_v2/tabular_posthoc_figures/
```

Principal figures:

```text
principal_tabular_roc_pr_curves.png
principal_tabular_roc_pr_curves.pdf
principal_tabular_calibration_plots.png
principal_tabular_calibration_plots.pdf
principal_tabular_decision_curves.png
principal_tabular_decision_curves.pdf
```

Additional outputs:

```text
prompt_matched_no_ecg_roc_pr_curves.*
prompt_matched_with_ecg_roc_pr_curves.*
decision_curves_by_model/
discrimination_point_estimates.csv
calibration_plot_points.csv
decision_curve_summary.csv
net_benefit_ranges.json
tabular_posthoc_manifest.json
```

## Principal corrected results

| Feature set and model | SCD ROC-AUC | SCD PR-AUC | PFD ROC-AUC | PFD PR-AUC |
|---|---:|---:|---:|---:|
| No ECG, selected/logistic | 0.696 (0.623–0.762) | 0.244 (0.174–0.352) | 0.787 (0.732–0.839) | 0.360 (0.272–0.470) |
| No ECG, gradient boosting | 0.657 (0.582–0.729) | 0.197 (0.143–0.278) | 0.678 (0.611–0.744) | 0.280 (0.203–0.391) |
| With ECG, selected/logistic | 0.710 (0.643–0.774) | 0.252 (0.182–0.359) | 0.792 (0.736–0.842) | 0.364 (0.273–0.476) |
| With ECG, gradient boosting | 0.704 (0.636–0.768) | 0.241 (0.175–0.345) | 0.724 (0.669–0.779) | 0.303 (0.223–0.415) |

For the primary no-ECG feature set, gradient boosting minus logistic regression had a PFD ROC-AUC difference of −0.109 (95% CI, −0.172 to −0.052; Holm-adjusted p=0.022), favoring logistic regression. The SCD ROC-AUC difference was −0.039 (95% CI, −0.104 to 0.025; Holm-adjusted p=1.000).

Adding the five ECG impressions did not significantly improve the selected model. The with-ECG minus no-ECG ROC-AUC difference was 0.015 (95% CI, −0.004 to 0.035; Holm-adjusted p=1.000) for SCD and 0.004 (95% CI, −0.007 to 0.016; Holm-adjusted p=1.000) for PFD.

## Exploratory decision-curve summary

For the primary selected no-ECG model, paired pointwise 95% bootstrap intervals supported greater net benefit than both treat-all and treat-none at:

- SCD: 8.25%–17.75% and 18.25%–20.0%.
- PFD: 5.0%–5.75% and 6.5%–25.0%.

These intervals were not adjusted across thresholds, and the 2%–25% range was not linked to a prespecified clinical intervention. They should be described as exploratory rather than evidence of established clinical utility.

## Reproducibility files

```bash
conda env export --no-builds > environment.yml

python --version

python -c "
import matplotlib
import numpy
import pandas
import scipy
import sklearn

print('matplotlib:', matplotlib.__version__)
print('NumPy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('SciPy:', scipy.__version__)
print('scikit-learn:', sklearn.__version__)
"

sha256sum \
  train_tabular_nested_cv.py \
  run_tabular_nested_cv.sh \
  tabular_posthoc_plots.py \
  README.md \
  > tabular_analysis_code_sha256.txt
```

## Reviewer-facing contributions

This analysis provides:

- Conventional L2-logistic and gradient-boosting benchmarks.
- Patient-level nested cross-validation.
- Training-only preprocessing, tuning, calibration, and threshold selection.
- One untouched outer-test prediction per applicable patient and model.
- ROC-AUC and PR-AUC with patient-bootstrap 95% CIs.
- Calibration measures and plots.
- Brier scores.
- Threshold metrics and confusion matrices.
- Paired comparisons with Holm adjustment.
- Exploratory decision curves with uncertainty.
- A corrected prompt-matched anticoagulant variable.
- An ECG-overlap sensitivity analysis.

The saved outer-test predictions can subsequently be used for paired comparisons with deterministic-template text, LLM-generated text, ECG-only, and multimodal models.

## Recommended release contents

Release, subject to study-governance approval:

- Analysis and plotting scripts.
- This README.
- Environment specification.
- Feature definitions and missingness summaries.
- Candidate and fold-selection tables.
- Aggregate performance, calibration, threshold, and paired-comparison tables.
- Aggregate ROC/PR, calibration, and decision-curve figures.
- Input and code hashes.

Do not publicly release patient-level structured data, patient-level predictions, or serialized model artifacts if these could expose protected or identifiable information without explicit governance approval.