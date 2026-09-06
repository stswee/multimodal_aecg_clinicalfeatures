# MUSIC Four-Year ECG Analysis

This directory contains the reproducible ECG-only analysis for predicting sudden cardiac death (SCD) and pump failure death (PFD) within four years of the baseline Holter recording.

The workflow uses patient-level nested cross-validation, training-fold-only preprocessing and model selection, fold-specific calibration and threshold selection, and pooled out-of-fold evaluation.

## Files

| File                                                  | Purpose                                                                                                                                                                   |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py` | Performs ECG auditing, inner-fold construction, MIL-TCN tuning, final outer-fold training, and export of predictions, weights, attention values, and ECG representations. |
| `wave1_capacity.sh`                                   | Main launcher. Runs all three tuning waves, final outer-fold refits, and aggregation using eight GPUs. Despite its filename, it runs the complete ECG pipeline.           |
| `calibrate_ecg_platt.py`                              | Fits fold-specific Platt calibrators using inner out-of-fold predictions and applies them to untouched outer-test predictions.                                            |
| `select_ecg_thresholds.py`                            | Selects one threshold per outer fold and outcome from calibrated inner out-of-fold predictions and applies it unchanged to the outer-test fold.                           |
| `ecg_decision_curve_analysis.py`                      | Performs exploratory decision-curve analysis using calibrated pooled outer-test predictions.                                                                              |
| `ECG_Analysis.ipynb`                                  | Reviews hyperparameter selection and calculates pooled uncalibrated performance with patient-bootstrap 95% confidence intervals.                                          |

## Obsolete Backup Launcher

`wave1_capacity.sh.before_resume_update` is an obsolete backup. Remove it from the GitHub or release folder after confirming that the current launcher contains the `--resume-after-wave1` option:

```bash
grep -n -- "--resume-after-wave1" wave1_capacity.sh
```

Git history or a tagged release is a clearer way to retain the earlier version. Publishing both launchers could cause users to run the obsolete version accidentally.

## Analysis Design

* Prediction index: baseline 24-hour Holter recording.
* Prediction horizon: four years, defined as 1,460 days after the baseline Holter.
* Final cohort: 730 patients.
* No cardiac death by four years: 577 patients.
* SCD by four years: 71 patients.
* PFD by four years: 82 patients.
* Non-cardiac death and cardiac transplantation were excluded during cohort construction.
* Five prespecified stratified outer folds were generated with `random_state=42`.
* Four inner folds are constructed independently inside each outer-training cohort using seeds `42 + outer_fold`.
* All ECG segments from a patient remain in the same fold.
* A control contributes a negative label to both outcome heads.
* An SCD patient contributes only to the SCD loss; the PFD loss is masked.
* A PFD patient contributes only to the PFD loss; the SCD loss is masked.
* The competing cardiac-death endpoint is therefore not treated as a negative outcome.
* Missing ECG features are retained.
* Nonfinite feature values are mean-imputed using the applicable training fold.
* Continuous features are standardized using the applicable training-fold mean and standard deviation.
* One missingness indicator is added for every ECG feature.
* Positive-class weights are calculated using only the applicable training patients.
* Early stopping and hyperparameter selection occur exclusively inside the inner loop.
* Outer-test patients are used only for final evaluation.

## Prerequisites

The scripts require:

* Python 3
* PyTorch with CUDA support
* pandas
* NumPy
* scikit-learn
* SciPy
* Matplotlib
* tqdm
* tmux
* Eight visible NVIDIA GPUs

The launcher currently activates the Conda environment:

```bash
shdb-af-analysis
```

Record the exact environment used for the manuscript:

```bash
conda env export --no-builds > environment.yml
nvidia-smi > gpu_environment.txt
python --version
```

## Expected Directory Structure

The launcher assumes the following directory structure:

```text
analysis_directory/
├── ECG_Analysis.ipynb
├── wave1_capacity.sh
├── train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py
├── calibrate_ecg_platt.py
├── select_ecg_thresholds.py
└── ecg_decision_curve_analysis.py

../../music/
├── music_patient_folds_5cv.csv
├── preprocessed_segments_HRV_complete/
└── ecg_nested_4year_three_wave/
```

Review the following variables near the top of `wave1_capacity.sh` before running:

```bash
CONDA_ENV="shdb-af-analysis"
FOLDS_CSV="${MUSIC_DIR}/music_patient_folds_5cv.csv"
FEATURES_DIR="${MUSIC_DIR}/preprocessed_segments_HRV_complete"
OUTPUT_ROOT="${MUSIC_DIR}/ecg_nested_4year_three_wave"
```

The fold-generation notebook may have saved the fold file as:

```text
music_patient_4year_outer_folds_5cv.csv
```

If so, either update `FOLDS_CSV` in `wave1_capacity.sh` or copy the file under the expected name.

Before training, confirm that the fold file contains:

* 730 unique patients.
* `Patient ID`.
* `SCD_4year_label`.
* `PFD_4year_label`.
* `outer_fold`.

Do not accidentally use an older fold file based on variable follow-up.

# Execution Order

## Step 1: Run the Nested ECG Training Pipeline

Make the launcher executable:

```bash
chmod +x wave1_capacity.sh
```

Start the complete pipeline:

```bash
bash wave1_capacity.sh
```

The launcher creates a detached tmux session named:

```text
ecg_nested_4year
```

It then performs the following stages.

### Prepare

The preparation stage:

* Verifies the expected cohort and outcome counts.
* Audits every patient’s ECG feature file.
* Verifies the minimum number of ECG segments.
* Locks the ECG feature schema.
* Constructs four inner folds inside each outer-training cohort.
* Records input-file hashes and fold assignments.

### Wave 1: Model Capacity

Wave 1 evaluates eight combinations of:

* Feature embedding dimension: 128 or 256.
* TCN hidden dimension: 128 or 256.
* TCN layers: 3 or 4.

Each configuration is evaluated across:

```text
5 outer-training cohorts × 4 inner folds
```

### Wave 2: Dropout

Wave 2 inherits the Wave 1 winner separately for each outer fold and evaluates eight combinations of:

* Feature-encoder dropout: 0.0 or 0.2.
* TCN dropout: 0.1 or 0.3.
* Task-branch dropout: 0.0 or 0.2.

### Wave 3: Optimizer

Wave 3 inherits the Wave 2 winner separately for each outer fold and evaluates eight combinations of:

* Learning rate: `3e-5`, `1e-4`, `3e-4`, or `1e-3`.
* Weight decay: 0 or `1e-5`.

### Final Outer-Fold Training

For each outer fold, the selected Wave 3 configuration is fitted using the complete outer-training cohort.

The number of final training epochs is the rounded median of the four selected inner-fold best epochs.

The trained model is then applied once to the untouched outer-test patients.

### Aggregation

The final stage concatenates predictions from the five outer-test folds and verifies that every patient has exactly one out-of-fold prediction.

## Hyperparameter Selection Rule

Each candidate is evaluated using pooled inner out-of-fold predictions.

The primary selection score is:

```text
0.5 × (SCD ROC-AUC + PFD ROC-AUC)
```

Selection proceeds as follows:

1. Identify the maximum pooled inner mean ROC-AUC.
2. Retain candidates within 0.005 of the maximum.
3. Select the candidate with the fewest trainable parameters.
4. If still tied, select the candidate with the highest mean PR-AUC.
5. If still tied, use the configuration name as a deterministic tie-breaker.

Different outer folds may correctly select different hyperparameters. Outer-test performance must not be used afterward to choose a single global configuration.

## Training Configuration

The default training settings are:

```text
Maximum epochs: 30
Minimum epochs: 5
Early-stopping patience: 7
Random seed: 42
Number of outer folds: 5
Number of inner folds: 4
Number of GPUs: 8
```

The complete analysis includes:

```text
Wave 1: 8 × 5 × 4 = 160 fits
Wave 2: 8 × 5 × 4 = 160 fits
Wave 3: 8 × 5 × 4 = 160 fits
Final outer-fold models: 5 fits
Total: 485 model fits
```

## Resume After a Completed Wave 1

Resume mode should only be used when all 40 Wave 1 configuration-by-outer-fold tuning summaries exist.

Check the number of completed summaries:

```bash
find ../../music/ecg_nested_4year_three_wave/tuning \
  -path "*/outer_fold_*/tuning_summary.json" \
  | wc -l
```

If Wave 1 is complete, continue with:

```bash
bash wave1_capacity.sh --resume-after-wave1
```

Resume mode:

* Preserves completed Wave 1 outputs.
* Recreates the Wave 1 summary.
* Continues with Wave 2.
* Continues with Wave 3.
* Fits the final outer-fold models.
* Aggregates outer-test predictions.

It is not a general partial-run recovery switch.

## Monitoring Training

Attach to the tmux session:

```bash
tmux attach -t ecg_nested_4year
```

Detach without stopping the analysis by pressing:

```text
Ctrl-b
d
```

Monitor GPU utilization:

```bash
watch -n 5 nvidia-smi
```

Monitor an individual configuration:

```bash
tail -f ../../music/ecg_nested_4year_three_wave/launcher_logs/tune_wave2__enc0p0_tcn0p1_br0p0.log
```

Count completed model fits:

```bash
find ../../music/ecg_nested_4year_three_wave \
  -name run_complete.json \
  | wc -l
```

The expected final count is:

```text
485
```

If the tmux session exits unexpectedly, inspect the most recent logs:

```bash
ls -lt ../../music/ecg_nested_4year_three_wave/launcher_logs | head
```

Then inspect the end of the relevant log:

```bash
tail -n 100 ../../music/ecg_nested_4year_three_wave/launcher_logs/*.log
```

## Principal Training Outputs

| Location under `OUTPUT_ROOT`                                         | Contents                                                                                                         |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `analysis_setup/`                                                    | ECG availability audit, feature schema, nested fold assignments, fold counts, and preparation manifest.          |
| `tuning/<configuration>/outer_fold_<k>/`                             | Inner-fold checkpoints, epoch histories, validation predictions, preprocessing statistics, and tuning summaries. |
| `wave_results/`                                                      | Candidate tables and selected hyperparameters for Waves 1–3.                                                     |
| `final_models/outer_fold_<k>/`                                       | Selected configuration, final checkpoint, training history, predictions, embeddings, and attention values.       |
| `pooled_outer_test/pooled_outer_test_predictions.csv`                | One untouched outer-test prediction per patient.                                                                 |
| `pooled_outer_test/final_selected_hyperparameters_by_outer_fold.csv` | Final configuration selected separately in each outer fold.                                                      |
| `pooled_outer_test/final_hyperparameter_selection_frequency.csv`     | Frequency with which each configuration was selected.                                                            |
| `pooled_outer_test/pooled_outer_test_embeddings_evaluation_only.npz` | Pooled outer-test ECG representations.                                                                           |

The pooled ECG embeddings come from five different fold-specific encoders and are evaluation-only.

For downstream fusion-model training, use the corresponding fold-specific training, validation, and test representations. Do not train a downstream model using the pooled evaluation-only embeddings.

# Step 2: Review Hyperparameters and Uncalibrated Performance

Open the analysis notebook:

```bash
jupyter lab ECG_Analysis.ipynb
```

Set `root` to the actual output directory:

```python
from pathlib import Path

root = Path(
    "/home/sswee/music/ecg_nested_4year_three_wave"
)
```

The notebook displays:

* Selected configurations from each tuning wave.
* Final configuration selected for each outer fold.
* Hyperparameter selection frequencies.
* Outer-fold performance.
* Pooled out-of-fold performance.
* Patient-bootstrap 95% confidence intervals.

Before the confidence-interval cell, load the pooled predictions:

```python
predictions = pd.read_csv(
    root
    / "pooled_outer_test"
    / "pooled_outer_test_predictions.csv",
    dtype={"Patient ID": "string"},
)
```

The attached notebook currently uses `predictions` without defining it, so this statement must be added before calculating the confidence intervals.

The notebook saves:

```text
pooled_outer_test/ecg_pooled_performance_with_95ci.csv
```

## Observed Uncalibrated Pooled Performance

| Outcome |      Metric | Estimate |      95% CI |   N | Events |
| ------- | ----------: | -------: | ----------: | --: | -----: |
| SCD     |     ROC-AUC |    0.628 | 0.550–0.707 | 648 |     71 |
| SCD     |      PR-AUC |    0.212 | 0.150–0.315 | 648 |     71 |
| SCD     | Brier score |    0.205 | 0.195–0.216 | 648 |     71 |
| PFD     |     ROC-AUC |    0.628 | 0.567–0.686 | 659 |     82 |
| PFD     |      PR-AUC |    0.182 | 0.137–0.259 | 659 |     82 |
| PFD     | Brier score |    0.242 | 0.231–0.254 | 659 |     82 |

The pooled out-of-fold estimates are the primary ECG-only performance estimates.

Fold-level results should be reported only as descriptive information about variation across folds.

# Step 3: Fit Fold-Specific Platt Calibration

Run calibration after all final outer-fold predictions and selected inner-fold predictions have been exported:

```bash
python calibrate_ecg_platt.py \
  --output_root /home/sswee/music/ecg_nested_4year_three_wave \
  --bootstrap_replicates 5000 \
  --seed 42
```

For each outer fold and outcome, the script fits:

```text
logit(p_calibrated)
    = intercept + slope × logit(p_uncalibrated)
```

The calibrator is fitted using only selected cross-fitted inner predictions from the corresponding outer-training cohort.

The fitted mapping is then applied unchanged to the outer-test probabilities. Outer-test outcomes are not used to fit the calibrator.

## Calibration Outputs

Outputs are stored in:

```text
ecg_nested_4year_three_wave/calibration/
```

Principal files include:

```text
fold_specific_platt_parameters.csv
pooled_outer_test_predictions_calibrated.csv
pooled_calibration_metrics_with_95ci.csv
calibration_bin_summary.csv
ecg_calibration_plot.png
calibration_manifest.json
```

## Observed Calibrated Performance

| Outcome | ROC-AUC | PR-AUC | Brier score | Calibration intercept | Calibration slope | Calibration-in-the-large |
| ------- | ------: | -----: | ----------: | --------------------: | ----------------: | -----------------------: |
| SCD     |   0.617 |  0.198 |      0.0968 |                −0.838 |             0.553 |                    0.112 |
| PFD     |   0.636 |  0.183 |      0.1070 |                −0.529 |             0.742 |                   −0.067 |

Exact 95% confidence intervals are stored in:

```text
calibration/pooled_calibration_metrics_with_95ci.csv
```

Calibration substantially reduced Brier error and corrected the mean-risk overprediction produced by the class-weighted raw model outputs.

Recommended reporting:

* Use the original pooled outer-test probabilities for the primary discrimination results.
* Use calibrated probabilities for Brier scores, calibration analyses, classification thresholds, and decision-curve analysis.
* Calibrated discrimination may be reported as a supplementary result.

If calibration outputs already exist, the script stops to prevent accidental replacement. Use `--overwrite` only when intentionally regenerating the analysis.

# Step 4: Select Thresholds Using Training-Only Predictions

Run:

```bash
python select_ecg_thresholds.py \
  --output_root /home/sswee/music/ecg_nested_4year_three_wave \
  --bootstrap_replicates 5000 \
  --seed 42
```

For each outer fold and outcome, the script:

1. Applies the fold-specific Platt mapping to the selected inner out-of-fold predictions.
2. Maximizes Youden’s J using those calibrated inner predictions.
3. Resolves ties using the highest sensitivity and then the lowest threshold.
4. Applies the selected threshold unchanged to the corresponding calibrated outer-test predictions.
5. Pools the outer-test classifications.
6. Calculates patient-bootstrap 95% confidence intervals.

Outer-test outcomes are not used to select thresholds.

## Threshold Outputs

Outputs are stored in:

```text
ecg_nested_4year_three_wave/threshold_analysis/
```

Principal files include:

```text
fold_specific_selected_thresholds.csv
pooled_outer_test_classifications.csv
pooled_threshold_metrics_with_95ci.csv
pooled_confusion_matrix_counts.csv
pooled_confusion_matrices.png
threshold_manifest.json
```

## Observed Threshold-Specific Performance

| Outcome | Sensitivity | Specificity |   PPV |   NPV |    F1 | Balanced accuracy |
| ------- | ----------: | ----------: | ----: | ----: | ----: | ----------------: |
| SCD     |       0.408 |       0.820 | 0.218 | 0.918 | 0.284 |             0.614 |
| PFD     |       0.707 |       0.506 | 0.169 | 0.924 | 0.273 |             0.607 |

The pooled confusion-matrix counts were:

| Outcome |  TN |  FP | FN | TP |
| ------- | --: | --: | -: | -: |
| SCD     | 473 | 104 | 42 | 29 |
| PFD     | 292 | 285 | 24 | 58 |

Exact 95% confidence intervals are stored in:

```text
threshold_analysis/pooled_threshold_metrics_with_95ci.csv
```

These thresholds are statistical operating points selected using Youden’s J. They are not clinically validated intervention thresholds and should not be described as clinical cutoffs.

# Step 5: Run Exploratory Decision-Curve Analysis

Run:

```bash
python ecg_decision_curve_analysis.py \
  --output_root /home/sswee/music/ecg_nested_4year_three_wave \
  --threshold_min 0.02 \
  --threshold_max 0.25 \
  --threshold_points 93 \
  --bootstrap_replicates 5000 \
  --seed 42
```

The script:

* Uses calibrated pooled outer-test probabilities.
* Includes the model, treat-all, and treat-none strategies.
* Uses patient-level bootstrap sampling.
* Calculates paired net-benefit differences.
* Reports ranges where the model has greater estimated net benefit than both default strategies.

## Decision-Curve Outputs

Outputs are stored in:

```text
ecg_nested_4year_three_wave/decision_curve_analysis/
```

Principal files include:

```text
decision_curve_summary.csv
net_benefit_ranges.json
ecg_decision_curves.png
decision_curve_manifest.json
```

## Observed Exploratory Net-Benefit Ranges

The paired-bootstrap 95% confidence intervals supported greater net benefit than both treat-all and treat-none over the following continuous grid ranges:

| Outcome | Exploratory threshold range |
| ------- | --------------------------: |
| SCD     |               10.75%–14.00% |
| PFD     |               10.00%–13.00% |

Isolated single-grid-point findings at 2% should not be emphasized.

Because no specific intervention or independently justified clinical action threshold was supplied, the decision-curve analysis remains exploratory. It does not establish clinical utility.

The decision-curve action threshold must not be conflated with the Youden classification threshold.

# Re-running Existing Outputs

The scripts intentionally refuse to overwrite completed outputs that do not match the requested run.

For a genuinely new analysis, prefer using a new `OUTPUT_ROOT`.

Use `--overwrite` only after verifying the exact output directory and preserving any results that must remain reproducible.

# How This Addresses Reviewer Comment 2

This ECG workflow provides:

* Strict patient-level nested cross-validation.
* Training-fold-only preprocessing.
* Training-fold-only feature imputation and scaling.
* Training-fold-only class weighting.
* Inner-loop early stopping.
* Inner-loop hyperparameter selection.
* Complete candidate hyperparameter tables.
* Fold-specific selected hyperparameter tables.
* One untouched outer-test prediction per patient.
* Pooled out-of-fold ROC-AUC.
* Pooled out-of-fold PR-AUC.
* Pooled out-of-fold Brier score.
* Patient-bootstrap 95% confidence intervals.
* Fold-specific model checkpoints.
* Fold-specific ECG representations.
* Training-only calibration.
* Training-only classification-threshold selection.
* Calibration plots.
* Confusion matrices.
* Exploratory decision-curve analysis.
* Saved manifests and source hashes.

Paired confidence intervals and statistical tests for differences between the ECG model and another model cannot be calculated from the ECG-only outputs alone.

After comparator models generate predictions for the same outer-test patients and folds:

1. Align predictions by `Patient ID`.
2. Use the same patient bootstrap sample for both models.
3. Calculate paired differences in ROC-AUC, PR-AUC, Brier score, and other prespecified metrics.
4. Report the paired difference estimate, 95% confidence interval, and paired two-sided p-value.
5. Optionally report a paired DeLong test for ROC-AUC.

# Recommended GitHub or Release Contents

Publish:

* `README.md`.
* `ECG_Analysis.ipynb`.
  `.
* `wave1_capacity.sh`.
* `train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py`.
* `calibrate_ecg_platt.py`.
* `select_ecg_thresholds.py`.
* `ecg_decision_curve_analysis.py`.
* `environment.yml`.
* Hardware and software metadata.
* Aggregate result tables.
* Calibration, confusion-matrix, and decision-curve figures.
* Complete Wave 1, Wave 2, and Wave 3 candidate tables.
* Selected hyperparameters for every outer fold.
* Analysis manifests.
* Cryptographic hashes.
* Model checkpoints and preprocessing statistics, if permitted.

Do not publish identifiable patient information, raw patient-level predictions, patient-level embeddings, or ECG recordings unless explicitly permitted by the study’s governance and consent framework.

If approved checkpoints or other artifacts are too large for Git, attach them to a versioned GitHub Release and record their SHA-256 hashes in the repository.
