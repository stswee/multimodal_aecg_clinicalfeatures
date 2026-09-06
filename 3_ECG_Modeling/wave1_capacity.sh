#!/usr/bin/env bash

# ==============================================================================
# MUSIC FOUR-YEAR ECG ANALYSIS: AUTOMATED THREE-WAVE NESTED CROSS-VALIDATION
# ==============================================================================
#
# PURPOSE
# -------
# Train and evaluate the MIL-TCN ECG model for four-year SCD and PFD outcomes
# using patient-level nested cross-validation. All preprocessing, early stopping,
# and hyperparameter selection occur within the outer-training cohorts. Each
# outer-test fold remains untouched until final evaluation.
#
# COHORT AND LABELS
# -----------------
# Expected cohort:
#   - Total patients:             730
#   - No cardiac death by 4 y:    577
#   - SCD by 4 y:                  71
#   - PFD by 4 y:                  82
#
# Multitask loss masking:
#   - Control: contributes a negative label to both SCD and PFD heads.
#   - SCD:     contributes only to the SCD head; PFD loss is masked.
#   - PFD:     contributes only to the PFD head; SCD loss is masked.
#
# Non-cardiac death and transplantation have already been excluded during
# cohort construction.
#
# CROSS-VALIDATION
# ----------------
# Outer evaluation:
#   - Five prespecified stratified outer folds.
#   - Each patient appears in exactly one outer-test fold.
#
# Inner tuning:
#   - Four inner folds are constructed separately within each outer-training
#     cohort.
#   - Imputation, scaling, class weights, early stopping, and hyperparameter
#     selection use only the corresponding inner-training/validation patients.
#
# Candidate performance is calculated from pooled inner out-of-fold predictions:
#
#     selection score = 0.5 * (SCD ROC-AUC + PFD ROC-AUC)
#
# Selection rule:
#   1. Find the maximum pooled inner mean ROC-AUC.
#   2. Retain candidates within 0.005 of the maximum.
#   3. Select the candidate with the fewest parameters.
#   4. If still tied, select the candidate with the highest mean PR-AUC.
#   5. If still tied, use the configuration name as a deterministic tie-breaker.
#
# HYPERPARAMETER WAVES
# --------------------
# Wave 1 -- model capacity:
#   - Feature embedding dimension: 128, 256
#   - TCN hidden dimension:        128, 256
#   - TCN layers:                  3, 4
#   - Total configurations:        8
#
# Wave 2 -- dropout:
#   - Inherits the Wave 1 winner separately for each outer fold.
#   - Feature-encoder dropout:     0.0, 0.2
#   - TCN dropout:                 0.1, 0.3
#   - Task-branch dropout:         0.0, 0.2
#   - Total configurations:        8
#
# Wave 3 -- optimizer:
#   - Inherits the Wave 2 winner separately for each outer fold.
#   - Learning rate:               3e-5, 1e-4, 3e-4, 1e-3
#   - Weight decay:                0, 1e-5
#   - Total configurations:        8
#
# Weighted binary cross-entropy is fixed across all three waves. Positive-class
# weights are recalculated using only the applicable training patients.
#
# TRAINING
# --------
#   - Maximum epochs:              30
#   - Minimum epochs:               5
#   - Early-stopping patience:      7
#   - Final outer-fold epochs: rounded median of the four inner-fold best epochs
#   - Random seed:                 42
#   - GPUs:                        8, one candidate configuration per GPU
#
# Total model fits:
#   - Wave 1: 8 configurations x 5 outer cohorts x 4 inner folds = 160
#   - Wave 2: 8 configurations x 5 outer cohorts x 4 inner folds = 160
#   - Wave 3: 8 configurations x 5 outer cohorts x 4 inner folds = 160
#   - Final outer-fold refits: 5
#   - Total: 485 model fits
#
# ECG PREPROCESSING
# -----------------
#   - Segment-level feature list is locked before training.
#   - Nonfinite values are imputed using training-fold feature means.
#   - Continuous values are standardized using training-fold mean and SD.
#   - One missingness indicator is added for every ECG feature.
#   - Partially missing segments are retained.
#   - An unavailable/empty ECG file or fewer than than three segments stops the run
#     rather than silently changing the analysis cohort.
#
# SAVED OUTPUTS
# -------------
# The pipeline stores:
#   - Complete Wave 1, Wave 2, and Wave 3 AUC/PR-AUC tables.
#   - Selected hyperparameters for each outer fold and wave.
#   - Inner-fold and final model checkpoints.
#   - Training-fold preprocessing statistics and class weights.
#   - Inner-validation predictions and ECG representations.
#   - Selected inner-fold train/validation representations.
#   - Final outer-training and untouched outer-test representations.
#   - Outer-test SCD/PFD probabilities and attention weights.
#   - One pooled outer-test prediction per patient.
#
# IMPORTANT INTERPRETATION
# ------------------------
# Each outer fold may select different hyperparameters. Outer-test performance
# is never used to select one global configuration. The pooled outer-test
# predictions are used later for confidence intervals, calibration, threshold
# analysis, paired model comparisons, and decision-curve analysis.
#
# Pooled ECG embeddings come from different fold-specific encoders and are
# evaluation-only. Downstream model training must use the corresponding
# fold-specific train/test or selected-inner-fold artifacts.
# ==============================================================================

# Automatic three-wave nested four-year ECG analysis for the reviewer response.
#
# Stage 1 (CPU): audit all 730 selected patients and create four inner folds
# independently inside each of the five outer-training cohorts.
#
# Wave 1 (GPUs 0-7): assign one prespecified capacity configuration to each
# GPU. Every configuration is evaluated in 5 outer-training cohorts x 4 inner
# folds. Outer-test patients are never used for epoch or architecture selection.
#
# Wave 2 automatically inherits each outer fold's Wave 1 winner and evaluates
# the 2x2x2 encoder/TCN/branch dropout grid. Wave 3 automatically inherits each
# outer fold's Wave 2 winner and evaluates the 4x2 learning-rate/weight-decay
# grid. Each wave writes its complete AUC table and selected hyperparameters.
#
# Final refitting uses each outer fold's Wave 3 winner. The last CPU stage
# concatenates the five untouched outer-test prediction files,
# verify exactly one prediction per patient, and produce pooled uncalibrated
# metrics. Calibration, thresholds, confidence intervals, and paired model
# comparisons are intentionally performed later using these saved predictions.

set -euo pipefail

SESSION_NAME="ecg_nested_4year"
CONDA_ENV="shdb-af-analysis"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py"
MUSIC_DIR="${SCRIPT_DIR}/../../music"
FOLDS_CSV="${MUSIC_DIR}/music_patient_folds_5cv.csv"
FEATURES_DIR="${MUSIC_DIR}/preprocessed_segments_HRV_complete"
OUTPUT_ROOT="${MUSIC_DIR}/ecg_nested_4year_three_wave"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"

EPOCHS=30
MIN_EPOCHS=5
PATIENCE=7
LEARNING_RATE="1e-4"
WEIGHT_DECAY="1e-5"
SEED=42
REPRESENTATION_DIM=128

CONFIG_NAMES=(
    "wave1__emb128_hid128_l3"
    "wave1__emb128_hid128_l4"
    "wave1__emb128_hid256_l3"
    "wave1__emb128_hid256_l4"
    "wave1__emb256_hid128_l3"
    "wave1__emb256_hid128_l4"
    "wave1__emb256_hid256_l3"
    "wave1__emb256_hid256_l4"
)
EMBEDDING_DIMS=(128 128 128 128 256 256 256 256)
HIDDEN_DIMS=(128 128 256 256 128 128 256 256)
TCN_LAYERS=(3 4 3 4 3 4 3 4)

WAVE2_CONFIG_NAMES=(
    "wave2__enc0p0_tcn0p1_br0p0"
    "wave2__enc0p0_tcn0p1_br0p2"
    "wave2__enc0p0_tcn0p3_br0p0"
    "wave2__enc0p0_tcn0p3_br0p2"
    "wave2__enc0p2_tcn0p1_br0p0"
    "wave2__enc0p2_tcn0p1_br0p2"
    "wave2__enc0p2_tcn0p3_br0p0"
    "wave2__enc0p2_tcn0p3_br0p2"
)
WAVE2_ENCODER_DROPOUTS=(0.0 0.0 0.0 0.0 0.2 0.2 0.2 0.2)
WAVE2_TCN_DROPOUTS=(0.1 0.1 0.3 0.3 0.1 0.1 0.3 0.3)
WAVE2_BRANCH_DROPOUTS=(0.0 0.2 0.0 0.2 0.0 0.2 0.0 0.2)

FINAL_CONFIG_NAMES=(
    "wave3__lr3e5_wd0"
    "wave3__lr3e5_wd1e5"
    "wave3__lr1e4_wd0"
    "wave3__lr1e4_wd1e5"
    "wave3__lr3e4_wd0"
    "wave3__lr3e4_wd1e5"
    "wave3__lr1e3_wd0"
    "wave3__lr1e3_wd1e5"
)
WAVE3_LEARNING_RATES=(3e-5 3e-5 1e-4 1e-4 3e-4 3e-4 1e-3 1e-3)
WAVE3_WEIGHT_DECAYS=(0 1e-5 0 1e-5 0 1e-5 0 1e-5)

COMMON_ARGS=(
    --folds_csv "${FOLDS_CSV}"
    --features_dir "${FEATURES_DIR}"
    --output_root "${OUTPUT_ROOT}"
    --patient_id_col "Patient ID"
    --scd_label_col SCD_4year_label
    --pfd_label_col PFD_4year_label
    --outer_fold_col outer_fold
    --sort_by window_idx
    --min_segments 3
    --outer_splits 5
    --inner_splits 4
    --seed "${SEED}"
    --num_workers 0
)

MODEL_FIXED_ARGS=(
    --tcn_kernel_size 3
    --tcn_dropout 0.1
    --attn_dim 128
    --enc_hidden 128
    --enc_dropout 0.2
    --branch_dropout 0.0
    --representation_dim "${REPRESENTATION_DIM}"
    --epochs "${EPOCHS}"
    --min_epochs "${MIN_EPOCHS}"
    --patience "${PATIENCE}"
    --lr "${LEARNING_RATE}"
    --weight_decay "${WEIGHT_DECAY}"
)

activate_environment() {
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc" 2>/dev/null || true
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found after sourcing ${HOME}/.bashrc" >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
}

check_inputs() {
    [[ -f "${PYTHON_SCRIPT}" ]] || { echo "Missing ${PYTHON_SCRIPT}" >&2; exit 1; }
    [[ -f "${FOLDS_CSV}" ]] || { echo "Missing ${FOLDS_CSV}" >&2; exit 1; }
    [[ -d "${FEATURES_DIR}" ]] || { echo "Missing ${FEATURES_DIR}" >&2; exit 1; }
}

wave1_selected_config_values() {
    local outer_fold="$1"
    local selection_file="${OUTPUT_ROOT}/wave_results/wave1_selected_by_outer_fold.json"
    python - "${selection_file}" "${outer_fold}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    selected = json.load(handle)["selected_by_outer_fold"][sys.argv[2]]["config"]

keys = [
    "embedding_dim", "tcn_hidden_dim", "tcn_layers", "tcn_kernel_size",
    "attn_dim", "enc_hidden", "branch_hidden", "representation_dim",
    "lr", "weight_decay",
]
print("\t".join(str(selected[key]) for key in keys))
PY
}

wave2_selected_config_values() {
    local outer_fold="$1"
    local selection_file="${OUTPUT_ROOT}/wave_results/wave2_selected_by_outer_fold.json"
    python - "${selection_file}" "${outer_fold}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    selected = json.load(handle)["selected_by_outer_fold"][sys.argv[2]]["config"]

keys = [
    "embedding_dim", "tcn_hidden_dim", "tcn_layers", "tcn_kernel_size",
    "tcn_dropout", "attn_dim", "enc_hidden", "enc_dropout", "branch_hidden",
    "branch_dropout", "representation_dim",
]
print("\t".join(str(selected[key]) for key in keys))
PY
}

run_wave2() {
    local worker_pids=()
    local failed=0
    for index in "${!WAVE2_CONFIG_NAMES[@]}"; do
        local config_name="${WAVE2_CONFIG_NAMES[$index]}"
        (
            export CUDA_VISIBLE_DEVICES="${index}"
            export PYTHONUNBUFFERED=1
            export OMP_NUM_THREADS=4
            export MKL_NUM_THREADS=4
            for outer_fold in 0 1 2 3 4; do
                IFS=$'\t' read -r embedding_dim hidden_dim layers kernel attention_dim \
                    encoder_hidden branch_hidden representation_dim learning_rate weight_decay \
                    <<<"$(wave1_selected_config_values "${outer_fold}")"
                python "${PYTHON_SCRIPT}" \
                    --stage tune \
                    "${COMMON_ARGS[@]}" \
                    --device cuda \
                    --outer_fold "${outer_fold}" \
                    --config_name "${config_name}" \
                    --embedding_dim "${embedding_dim}" \
                    --tcn_hidden_dim "${hidden_dim}" \
                    --tcn_layers "${layers}" \
                    --tcn_kernel_size "${kernel}" \
                    --attn_dim "${attention_dim}" \
                    --enc_hidden "${encoder_hidden}" \
                    --branch_hidden "${branch_hidden}" \
                    --representation_dim "${representation_dim}" \
                    --enc_dropout "${WAVE2_ENCODER_DROPOUTS[$index]}" \
                    --tcn_dropout "${WAVE2_TCN_DROPOUTS[$index]}" \
                    --branch_dropout "${WAVE2_BRANCH_DROPOUTS[$index]}" \
                    --lr "${learning_rate}" \
                    --weight_decay "${weight_decay}" \
                    --epochs "${EPOCHS}" \
                    --min_epochs "${MIN_EPOCHS}" \
                    --patience "${PATIENCE}"
            done
        ) >"${LOG_DIR}/tune_${config_name}.log" 2>&1 &
        local worker_pid="$!"
        worker_pids+=("${worker_pid}")
        echo "  Wave 2 GPU ${index}: ${config_name} (PID ${worker_pid})"
    done

    for pid in "${worker_pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "Wave 2 failed. Inspect ${LOG_DIR}/tune_wave2__*.log" >&2
        exit 1
    fi

    local candidate_args=()
    for config_name in "${WAVE2_CONFIG_NAMES[@]}"; do
        candidate_args+=(--candidate_config "${config_name}")
    done
    CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
        --stage summarize \
        "${COMMON_ARGS[@]}" \
        "${candidate_args[@]}" \
        --wave_name wave2 \
        --auc_tolerance 0.005 \
        --device cpu \
        2>&1 | tee "${LOG_DIR}/wave2_summary.log"
}

run_wave3() {
    local worker_pids=()
    local failed=0
    for index in "${!FINAL_CONFIG_NAMES[@]}"; do
        local config_name="${FINAL_CONFIG_NAMES[$index]}"
        (
            export CUDA_VISIBLE_DEVICES="${index}"
            export PYTHONUNBUFFERED=1
            export OMP_NUM_THREADS=4
            export MKL_NUM_THREADS=4
            for outer_fold in 0 1 2 3 4; do
                IFS=$'\t' read -r embedding_dim hidden_dim layers kernel tcn_dropout \
                    attention_dim encoder_hidden encoder_dropout branch_hidden branch_dropout \
                    representation_dim <<<"$(wave2_selected_config_values "${outer_fold}")"
                python "${PYTHON_SCRIPT}" \
                    --stage tune \
                    "${COMMON_ARGS[@]}" \
                    --device cuda \
                    --outer_fold "${outer_fold}" \
                    --config_name "${config_name}" \
                    --embedding_dim "${embedding_dim}" \
                    --tcn_hidden_dim "${hidden_dim}" \
                    --tcn_layers "${layers}" \
                    --tcn_kernel_size "${kernel}" \
                    --tcn_dropout "${tcn_dropout}" \
                    --attn_dim "${attention_dim}" \
                    --enc_hidden "${encoder_hidden}" \
                    --enc_dropout "${encoder_dropout}" \
                    --branch_hidden "${branch_hidden}" \
                    --branch_dropout "${branch_dropout}" \
                    --representation_dim "${representation_dim}" \
                    --lr "${WAVE3_LEARNING_RATES[$index]}" \
                    --weight_decay "${WAVE3_WEIGHT_DECAYS[$index]}" \
                    --epochs "${EPOCHS}" \
                    --min_epochs "${MIN_EPOCHS}" \
                    --patience "${PATIENCE}"
            done
        ) >"${LOG_DIR}/tune_${config_name}.log" 2>&1 &
        local worker_pid="$!"
        worker_pids+=("${worker_pid}")
        echo "  Wave 3 GPU ${index}: ${config_name} (PID ${worker_pid})"
    done

    for pid in "${worker_pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "Wave 3 failed. Inspect ${LOG_DIR}/tune_wave3__*.log" >&2
        exit 1
    fi

    local candidate_args=()
    for config_name in "${FINAL_CONFIG_NAMES[@]}"; do
        candidate_args+=(--candidate_config "${config_name}")
    done
    CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
        --stage summarize \
        "${COMMON_ARGS[@]}" \
        "${candidate_args[@]}" \
        --wave_name wave3 \
        --auc_tolerance 0.005 \
        --device cpu \
        2>&1 | tee "${LOG_DIR}/wave3_summary.log"
}

run_inside_tmux() {
    activate_environment
    check_inputs
    mkdir -p "${LOG_DIR}"

    if [[ "${RESUME_AFTER_WAVE1:-0}" != "1" ]]; then
        echo "[$(date)] Setup: preparing folds and auditing ECG availability"
        python "${PYTHON_SCRIPT}" \
            --stage prepare \
            "${COMMON_ARGS[@]}" \
            --expected_patients 730 \
            --expected_controls 577 \
            --expected_scd 71 \
            --expected_pfd 82 \
            2>&1 | tee "${LOG_DIR}/prepare.log"

        echo "[$(date)] Wave 1/3: capacity tuning on eight GPUs"
        tuning_pids=()
        for index in "${!CONFIG_NAMES[@]}"; do
            config_name="${CONFIG_NAMES[$index]}"
            gpu_id="${index}"
            (
                export CUDA_VISIBLE_DEVICES="${gpu_id}"
                export PYTHONUNBUFFERED=1
                export OMP_NUM_THREADS=4
                export MKL_NUM_THREADS=4
                for outer_fold in 0 1 2 3 4; do
                    python "${PYTHON_SCRIPT}" \
                        --stage tune \
                        "${COMMON_ARGS[@]}" \
                        "${MODEL_FIXED_ARGS[@]}" \
                        --device cuda \
                        --outer_fold "${outer_fold}" \
                        --config_name "${config_name}" \
                        --embedding_dim "${EMBEDDING_DIMS[$index]}" \
                        --tcn_hidden_dim "${HIDDEN_DIMS[$index]}" \
                        --tcn_layers "${TCN_LAYERS[$index]}" \
                        --branch_hidden "${HIDDEN_DIMS[$index]}"
                done
            ) >"${LOG_DIR}/tune_${config_name}.log" 2>&1 &
            worker_pid="$!"
            tuning_pids+=("${worker_pid}")
            echo "  GPU ${gpu_id}: ${config_name} (PID ${worker_pid})"
        done

        tuning_failed=0
        for pid in "${tuning_pids[@]}"; do
            if ! wait "${pid}"; then
                tuning_failed=1
            fi
        done
        if [[ "${tuning_failed}" -ne 0 ]]; then
            echo "At least one tuning worker failed. Inspect ${LOG_DIR}/tune_*.log" >&2
            exit 1
        fi
    else
        echo "[$(date)] Resume mode: preserving completed Wave 1 tuning outputs"
        echo "[$(date)] Resume mode: starting with Wave 1 summarization"
    fi

    candidate_args=()
    for config_name in "${CONFIG_NAMES[@]}"; do
        candidate_args+=(--candidate_config "${config_name}")
    done

    CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
        --stage summarize \
        "${COMMON_ARGS[@]}" \
        "${candidate_args[@]}" \
        --wave_name wave1 \
        --auc_tolerance 0.005 \
        --device cpu \
        2>&1 | tee "${LOG_DIR}/wave1_summary.log"

    echo "[$(date)] Wave 2/3: dropout tuning using outer-specific Wave 1 winners"
    run_wave2

    echo "[$(date)] Wave 3/3: optimizer tuning using outer-specific Wave 2 winners"
    run_wave3

    final_candidate_args=()
    for config_name in "${FINAL_CONFIG_NAMES[@]}"; do
        final_candidate_args+=(--candidate_config "${config_name}")
    done

    echo "[$(date)] Final refitting and fold-specific embedding export"
    final_pids=()
    for outer_fold in 0 1 2 3 4; do
        (
            export CUDA_VISIBLE_DEVICES="${outer_fold}"
            export PYTHONUNBUFFERED=1
            export OMP_NUM_THREADS=4
            export MKL_NUM_THREADS=4
            python "${PYTHON_SCRIPT}" \
                --stage final \
                "${COMMON_ARGS[@]}" \
                "${MODEL_FIXED_ARGS[@]}" \
                "${final_candidate_args[@]}" \
                --device cuda \
                --outer_fold "${outer_fold}" \
                --auc_tolerance 0.005 \
                --export_selected_inner_embeddings
        ) >"${LOG_DIR}/final_outer_fold_${outer_fold}.log" 2>&1 &
        worker_pid="$!"
        final_pids+=("${worker_pid}")
        echo "  GPU ${outer_fold}: final outer fold ${outer_fold} (PID ${worker_pid})"
    done

    final_failed=0
    for pid in "${final_pids[@]}"; do
        if ! wait "${pid}"; then
            final_failed=1
        fi
    done
    if [[ "${final_failed}" -ne 0 ]]; then
        echo "At least one final worker failed. Inspect ${LOG_DIR}/final_outer_fold_*.log" >&2
        exit 1
    fi

    echo "[$(date)] Pooling untouched outer-test outputs"
    CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
        --stage aggregate \
        "${COMMON_ARGS[@]}" \
        --device cpu \
        2>&1 | tee "${LOG_DIR}/aggregate.log"

    echo "[$(date)] Nested ECG analysis complete"
    echo "Pooled predictions: ${OUTPUT_ROOT}/pooled_outer_test/pooled_outer_test_predictions.csv"
    echo "Fold-specific embeddings: ${OUTPUT_ROOT}/final_models/outer_fold_*/"
    echo "Wave AUC tables: ${OUTPUT_ROOT}/wave_results/wave*_auc_results.csv"
}

if [[ "${1:-}" == "__inside_tmux" ]]; then
    run_inside_tmux
    exit 0
fi

if [[ "${1:-}" == "__inside_tmux_resume_after_wave1" ]]; then
    RESUME_AFTER_WAVE1=1 run_inside_tmux
    exit 0
fi

check_inputs
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session '${SESSION_NAME}' already exists." >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

inside_mode="__inside_tmux"
if [[ "${1:-}" == "--resume-after-wave1" ]]; then
    inside_mode="__inside_tmux_resume_after_wave1"
elif [[ -n "${1:-}" ]]; then
    echo "Usage: bash $(basename "${SCRIPT_PATH}") [--resume-after-wave1]" >&2
    exit 2
fi

printf -v tmux_command '%q %q %q' bash "${SCRIPT_PATH}" "${inside_mode}"
tmux new-session -d -s "${SESSION_NAME}" -n coordinator "${tmux_command}"

echo "Started nested ECG analysis in tmux session: ${SESSION_NAME}"
echo "Attach:  tmux attach -t ${SESSION_NAME}"
echo "Logs:    tail -f ${LOG_DIR}/prepare.log"
echo "GPU use: watch -n 5 nvidia-smi"
echo "Status:  find ${OUTPUT_ROOT} -name run_complete.json | wc -l  # 485 when complete"