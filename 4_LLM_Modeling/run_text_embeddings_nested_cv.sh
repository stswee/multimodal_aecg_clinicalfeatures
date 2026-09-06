#!/usr/bin/env bash

# ==============================================================================
# MUSIC FOUR-YEAR TEXT ANALYSIS: NESTED CROSS-VALIDATION AND ABLATIONS
# ==============================================================================
#
# Primary selection condition
# ---------------------------
# Detailed full LLM risk response generated without Holter ECG impressions.
#
# Candidate selection
# -------------------
# Inside each of five outer-training cohorts, four inner folds select among:
#   - LLaMA-3.1-8B and LLaMA-3.2-3B responses;
#   - frozen BioBERT and ClinicalBERT CLS embeddings; and
#   - eight prespecified linear/MLP classifier configurations.
#
# SCD and PFD are selected and fitted independently. The selection score is
# pooled inner ROC-AUC for the corresponding endpoint-specific binary task.
#
# Candidates within 0.005 of the maximum remain eligible. The fewest-parameter
# candidate is chosen, followed by higher mean PR-AUC and configuration name.
# Outer-test outcomes are never used for selection.
#
# Fixed comparison arms
# ---------------------
# After primary selection, the selected LLaMA checkpoint, frozen encoder, and
# classifier hyperparameters are held fixed within that outer fold while the
# following final representations are trained/evaluated for each endpoint:
#   1. endpoint-specific full risk response without ECG impressions;
#   2. endpoint-specific risk label only without ECG impressions;
#   3. endpoint-specific rationale only without ECG impressions;
#   4. joint SCD+PFD full response without ECG impressions;
#   5. neutral non-reasoning summary without ECG impressions;
#   6. deterministic non-LLM template without ECG impressions;
#   7. endpoint-specific full risk response with ECG impressions;
#   8. endpoint-specific risk label only with ECG impressions; and
#   9. endpoint-specific rationale only with ECG impressions.
#
# Leakage controls
# ----------------
#   - Reuses the exact nested fold file exported by the ECG analysis.
#   - Frozen embeddings are outcome-independent and aligned by Patient ID.
#   - Detailed responses use mean-pooled overlapping 512-token chunks; the
#     preflight fails if any response was silently truncated.
#   - Embedding standardization, class weights, early stopping, classifier
#     tuning, Platt calibration, and Youden thresholds use training data only.
#   - The competing cardiac-death endpoint is removed before scaling/training.
#   - Independent binary classifiers are selected for SCD and PFD.
#   - One untouched outer-test prediction is produced per eligible patient/arm.
#
# Evaluation
# ----------
# Patient-bootstrap 95% confidence intervals are generated for ROC-AUC,
# PR-AUC, Brier score, calibration intercept/slope, and threshold metrics.
# Prespecified paired comparisons use the same bootstrap patient samples and
# Holm adjustment. Exploratory decision-curve analysis is run afterward from
# the saved calibrated outer-test predictions without retraining.
# ============================================================================== 

set -euo pipefail

SESSION_NAME="text_nested_4year_v4_detailed"
CONDA_ENV="shdb-af-analysis"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_text_embeddings_nested_cv.py"

MUSIC_DIR="${SCRIPT_DIR}/../../music"
EMBEDDING_ROOT="${MUSIC_DIR}/text_embeddings_4year_v4_detailed"

# Reuse the exact outer and inner folds already used for the ECG analysis.
FOLDS_CSV="${MUSIC_DIR}/ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv"
OUTPUT_ROOT="${MUSIC_DIR}/text_nested_4year_v4_detailed"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"

SEED=42
EPOCHS=100
MIN_EPOCHS=10
PATIENCE=15
BOOTSTRAP_REPLICATES=5000
AUC_TOLERANCE=0.005
EXPECTED_POOLING="cls"
EXPECTED_MAX_LENGTH=512
EXPECTED_LONG_TEXT_STRATEGY="mean_chunks"

SOURCES=("LLaMA3.1-8B" "LLaMA3.2-3B")
ENCODERS=("BioBERT" "ClinicalBERT")
TASKS=("scd" "pfd")

# Eight prespecified classifier configurations. Each GPU processes all four
# LLaMA/encoder combinations for one classifier configuration.
CLASSIFIER_TAGS=(
    "linear_lr1e3_wd0"
    "linear_lr3e4_wd1e5"
    "mlp_h128_l1_d0p1_lr1e3_wd1e5"
    "mlp_h128_l2_d0p2_lr3e4_wd1e5"
    "mlp_h256_l1_d0p2_lr1e3_wd1e5"
    "mlp_h256_l2_d0p2_lr3e4_wd1e5"
    "mlp_h256_l2_d0p5_lr1e4_wd1e4"
    "mlp_h512_l2_d0p2_lr1e4_wd1e5"
)

CLASSIFIERS=("linear" "linear" "mlp" "mlp" "mlp" "mlp" "mlp" "mlp")
HIDDEN_DIMS=(128 128 128 128 256 256 256 512)
LAYERS=(1 1 1 2 1 2 2 2)
DROPOUTS=(0.0 0.0 0.1 0.2 0.2 0.2 0.5 0.2)
LEARNING_RATES=(1e-3 3e-4 1e-3 3e-4 1e-3 3e-4 1e-4 1e-4)
WEIGHT_DECAYS=(0 1e-5 1e-5 1e-5 1e-5 1e-5 1e-4 1e-5)

COMMON_ARGS=(
    --folds_csv "${FOLDS_CSV}"
    --embedding_root "${EMBEDDING_ROOT}"
    --patient_id_col "Patient ID"
    --scd_label_col SCD_4year_label
    --pfd_label_col PFD_4year_label
    --outer_fold_col outer_fold
    --outer_splits 5
    --inner_splits 4
    --seed "${SEED}"
    --expected_patients 730
    --expected_controls 577
    --expected_scd 71
    --expected_pfd 82
    --expected_pooling "${EXPECTED_POOLING}"
    --expected_max_length "${EXPECTED_MAX_LENGTH}"
    --expected_long_text_strategy "${EXPECTED_LONG_TEXT_STRATEGY}"
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
    [[ -d "${EMBEDDING_ROOT}" ]] || { echo "Missing ${EMBEDDING_ROOT}" >&2; exit 1; }
}

candidate_name() {
    local source="$1"
    local encoder="$2"
    local classifier_tag="$3"
    local source_slug="${source//./p}"
    echo "${source_slug}__${encoder}__${classifier_tag}"
}

build_candidate_names() {
    CANDIDATE_NAMES=()
    for source in "${SOURCES[@]}"; do
        for encoder in "${ENCODERS[@]}"; do
            for tag in "${CLASSIFIER_TAGS[@]}"; do
                CANDIDATE_NAMES+=("$(candidate_name "${source}" "${encoder}" "${tag}")")
            done
        done
    done
}

run_tuning_worker() {
    local gpu_id="$1"
    local classifier_index="$2"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4

    for task in "${TASKS[@]}"; do
        for source in "${SOURCES[@]}"; do
            for encoder in "${ENCODERS[@]}"; do
                local tag="${CLASSIFIER_TAGS[$classifier_index]}"
                local name
                name="$(candidate_name "${source}" "${encoder}" "${tag}")"
                for outer_fold in 0 1 2 3 4; do
                    python "${PYTHON_SCRIPT}" \
                        --stage tune \
                        "${COMMON_ARGS[@]}" \
                        --task "${task}" \
                        --output_root "${OUTPUT_ROOT}/tasks/${task}" \
                        --device cuda \
                        --outer_fold "${outer_fold}" \
                        --config_name "${name}" \
                        --source_name "${source}" \
                        --encoder_name "${encoder}" \
                        --classifier "${CLASSIFIERS[$classifier_index]}" \
                        --hidden_dim "${HIDDEN_DIMS[$classifier_index]}" \
                        --layers "${LAYERS[$classifier_index]}" \
                        --dropout "${DROPOUTS[$classifier_index]}" \
                        --lr "${LEARNING_RATES[$classifier_index]}" \
                        --weight_decay "${WEIGHT_DECAYS[$classifier_index]}" \
                        --epochs "${EPOCHS}" \
                        --min_epochs "${MIN_EPOCHS}" \
                        --patience "${PATIENCE}"
                done
            done
        done
    done
}

run_inside_tmux() {
    activate_environment
    check_inputs
    mkdir -p "${LOG_DIR}"
    build_candidate_names

    echo "[$(date)] Auditing endpoint-specific cohorts, folds, and embeddings"
    for task in "${TASKS[@]}"; do
        CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
            --stage prepare \
            "${COMMON_ARGS[@]}" \
            --task "${task}" \
            --output_root "${OUTPUT_ROOT}/tasks/${task}" \
            --device cpu \
            2>&1 | tee "${LOG_DIR}/prepare_${task}.log"
    done

    echo "[$(date)] Tuning 32 primary full-response/no-ECG candidates"
    worker_pids=()
    for gpu_id in 0 1 2 3 4 5 6 7; do
        (
            run_tuning_worker "${gpu_id}" "${gpu_id}"
        ) >"${LOG_DIR}/tuning_gpu_${gpu_id}.log" 2>&1 &
        worker_pid="$!"
        worker_pids+=("${worker_pid}")
        echo "  GPU ${gpu_id}: ${CLASSIFIER_TAGS[$gpu_id]} (PID ${worker_pid})"
    done

    failed=0
    for pid in "${worker_pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "At least one tuning worker failed. Inspect ${LOG_DIR}/tuning_gpu_*.log" >&2
        exit 1
    fi

    candidate_args=()
    for name in "${CANDIDATE_NAMES[@]}"; do
        candidate_args+=(--candidate_config "${name}")
    done

    echo "[$(date)] Selecting one primary configuration per task and outer fold"
    for task in "${TASKS[@]}"; do
        CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
            --stage select \
            "${COMMON_ARGS[@]}" \
            --task "${task}" \
            --output_root "${OUTPUT_ROOT}/tasks/${task}" \
            "${candidate_args[@]}" \
            --auc_tolerance "${AUC_TOLERANCE}" \
            --device cpu \
            2>&1 | tee "${LOG_DIR}/selection_${task}.log"
    done

    echo "[$(date)] Fitting all nine fixed arms for both binary endpoints"
    final_pids=()
    final_labels=()
    for gpu_id in 0 1 2 3 4 5 6 7; do
        (
            export CUDA_VISIBLE_DEVICES="${gpu_id}"
            export PYTHONUNBUFFERED=1
            export OMP_NUM_THREADS=4
            export MKL_NUM_THREADS=4
            for ((work_index=gpu_id; work_index<10; work_index+=8)); do
                task_index=$((work_index / 5))
                outer_fold=$((work_index % 5))
                task="${TASKS[$task_index]}"
                python "${PYTHON_SCRIPT}" \
                    --stage final \
                    "${COMMON_ARGS[@]}" \
                    --task "${task}" \
                    --output_root "${OUTPUT_ROOT}/tasks/${task}" \
                    --outer_fold "${outer_fold}" \
                    --device cuda \
                    >"${LOG_DIR}/final_${task}_outer_fold_${outer_fold}.log" 2>&1
            done
        ) &
        worker_pid="$!"
        final_pids+=("${worker_pid}")
        final_labels+=("GPU ${gpu_id}")
        echo "  GPU ${gpu_id}: endpoint/fold worker (PID ${worker_pid})"
    done

    failed=0
    for pid in "${final_pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "At least one final worker failed. Inspect ${LOG_DIR}/final_*_outer_fold_*.log" >&2
        exit 1
    fi

    for task in "${TASKS[@]}"; do
        echo "[$(date)] Pooling untouched ${task^^} outer-test predictions"
        CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
            --stage aggregate \
            "${COMMON_ARGS[@]}" \
            --task "${task}" \
            --output_root "${OUTPUT_ROOT}/tasks/${task}" \
            --device cpu \
            2>&1 | tee "${LOG_DIR}/aggregate_${task}.log"

        echo "[$(date)] Evaluating ${task^^} arms"
        CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
            --stage evaluate \
            "${COMMON_ARGS[@]}" \
            --task "${task}" \
            --output_root "${OUTPUT_ROOT}/tasks/${task}" \
            --device cpu \
            --bootstrap_replicates "${BOOTSTRAP_REPLICATES}" \
            2>&1 | tee "${LOG_DIR}/evaluate_${task}.log"
    done

    echo "[$(date)] Combining endpoint results and applying cross-endpoint Holm correction"
    CUDA_VISIBLE_DEVICES="" python "${PYTHON_SCRIPT}" \
        --stage combine \
        "${COMMON_ARGS[@]}" \
        --output_root "${OUTPUT_ROOT}" \
        --device cpu \
        2>&1 | tee "${LOG_DIR}/combine.log"

    echo "[$(date)] Text embedding analysis complete"
    echo "SCD results: ${OUTPUT_ROOT}/tasks/scd/"
    echo "PFD results: ${OUTPUT_ROOT}/tasks/pfd/"
    echo "Combined results: ${OUTPUT_ROOT}/combined_evaluation/"
}

if [[ "${1:-}" == "__inside_tmux" ]]; then
    run_inside_tmux
    exit 0
fi

if [[ -n "${1:-}" ]]; then
    echo "Usage: bash $(basename "${SCRIPT_PATH}")" >&2
    exit 2
fi

check_inputs

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session '${SESSION_NAME}' already exists." >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

printf -v tmux_command '%q %q %q' bash "${SCRIPT_PATH}" "__inside_tmux"
tmux new-session -d -s "${SESSION_NAME}" -n coordinator "${tmux_command}"

echo "Started nested text analysis in tmux session: ${SESSION_NAME}"
echo "Attach:  tmux attach -t ${SESSION_NAME}"
echo "Logs:    ${LOG_DIR}"
echo "GPU use: watch -n 5 nvidia-smi"
echo "Status:  find ${OUTPUT_ROOT}/tasks -name tuning_summary.json | wc -l  # 320 when tuning is complete"
