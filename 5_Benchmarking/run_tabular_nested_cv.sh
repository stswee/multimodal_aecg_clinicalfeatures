#!/usr/bin/env bash
# ============================================================================
# MUSIC FOUR-YEAR STRUCTURED CLINICAL BASELINES
# ============================================================================
#
# PURPOSE
# -------
# Fit conventional structured-variable baselines for four-year SCD and PFD:
#   1. L2-penalized logistic regression.
#   2. Histogram gradient boosting.
#
# The primary feature set exactly matches the variables supplied to the LLM
# prompt without Holter ECG impressions. A sensitivity analysis adds the five
# categorical Holter impressions used in the with-ECG prompt.
#
# VALIDATION
# ----------
# - Five patient-level outer folds provide untouched test predictions.
# - Four inner folds inside each outer-training cohort perform all tuning.
# - Imputation, missingness indicators, one-hot encoding, scaling, class
#   weighting, hyperparameter selection, Platt calibration, and threshold
#   selection use training data only.
# - SCD patients contribute only to the SCD task; PFD patients contribute only
#   to the PFD task; controls contribute negative labels to both tasks.
# - The primary selection criterion is the equally weighted mean of pooled
#   inner OOF SCD and PFD ROC-AUC. Candidates within 0.005 of the maximum are
#   resolved by prespecified model complexity, then PR-AUC, then name.
# - Final metrics use pooled outer-test predictions and 5,000 patient-bootstrap
#   replicates. Paired comparisons use identical resampled patients and Holm
#   multiplicity correction.
#
# OUTPUTS
# -------
# Model bundles, preprocessing objects, inner OOF predictions, untouched outer
# test predictions, fold-specific selections, Platt parameters, thresholds,
# ROC-AUC, PR-AUC, Brier scores, calibration estimates, confusion matrices,
# 95% confidence intervals, and paired comparisons are saved under OUTPUT_ROOT.
#
# IMPORTANT
# ---------
# Review the source CSV and any dataset-specific sentinel missing-value codes
# before the definitive run. No numeric sentinel (for example, 999) is treated
# as missing unless supplied explicitly through --sentinel_json.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="shdb-af-analysis"
MUSIC_DIR="${SCRIPT_DIR}/../../music"

TABULAR_CSV="${MUSIC_DIR}/subject-info.csv"
FOLDS_CSV="${MUSIC_DIR}/ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv"
OUTPUT_ROOT="${MUSIC_DIR}/tabular_nested_4year_v2"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"
LOG_FILE="${LOG_DIR}/tabular_nested_cv.log"
SESSION_NAME="tabular_nested_4year_v2"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${TABULAR_CSV}" ]]; then
    echo "Missing tabular CSV: ${TABULAR_CSV}" >&2
    exit 1
fi

if [[ ! -f "${FOLDS_CSV}" ]]; then
    echo "Missing nested folds CSV: ${FOLDS_CSV}" >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

COMMAND=$(cat <<EOF
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python train_tabular_nested_cv.py --tabular_csv "${TABULAR_CSV}" --folds_csv "${FOLDS_CSV}" --output_root "${OUTPUT_ROOT}" --feature_set prompt_matched_no_ecg --feature_set prompt_matched_with_ecg --outer_splits 5 --inner_splits 4 --seed 42 --auc_tolerance 0.005 --bootstrap_replicates 5000 --expected_anticoagulant_yes 610 --expected_anticoagulant_no 120 2>&1 | tee "${LOG_FILE}"
status=\${PIPESTATUS[0]}
echo "Analysis exit status: \${status}"
echo "Finished at: \$(date --iso-8601=seconds)"
exec bash
EOF
)

tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${COMMAND}")"

echo "Started structured-baseline analysis."
echo "tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Follow log: tail -f ${LOG_FILE}"
echo "Output root: ${OUTPUT_ROOT}"