#!/usr/bin/env bash
# ==============================================================================
# MUSIC FOUR-YEAR ECG + LITERATURE-REDUCED CLINICAL FUSION
# ==============================================================================
#
# Runs ECG plus the externally motivated reduced clinical variable set
# independently for SCD and PFD. This is a sensitivity experiment; it does not
# replace the prespecified full tabular benchmark.
#
# Every pair retains five fusion arms:
#   - direct concatenation;
#   - projected concatenation;
#   - patient-specific scalar gating;
#   - patient-specific vector gating; and
#   - global weighted sum.
#
# Two prespecified capacity profiles are evaluated per method. Fusion method,
# capacity, and training duration are selected only inside each outer-training
# cohort. The competing endpoint is excluded from each binary task. The
# launcher distributes task-by-pair-by-outer-fold tuning and final fitting
# tasks across eight GPUs. A CPU orchestrator runs prepare, selection,
# aggregation, and bootstrap evaluation stages.
# ==============================================================================

set -euo pipefail

SESSION_NAME="multimodal_literature_reduced_continuous_lvef_4year_v1"
CONDA_ENV="shdb-af-analysis"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_multimodal_literature_reduced_nested_cv.py"

MUSIC_DIR="${SCRIPT_DIR}/../../../../music"
FOLDS_CSV="${MUSIC_DIR}/ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv"
ECG_ROOT="${MUSIC_DIR}/ecg_nested_4year_three_wave"
# The completed revised text analysis records this embedding root. If your
# server used a versioned directory instead, change only this line.
TEXT_EMBEDDING_ROOT="${MUSIC_DIR}/text_embeddings_4year_v3_endpoint_specific"
TEXT_RESULTS_ROOT="${MUSIC_DIR}/text_nested_4year_v3_endpoint_specific"
TABULAR_CSV="${MUSIC_DIR}/subject-info.csv"
OUTPUT_ROOT="${MUSIC_DIR}/multimodal_literature_reduced_continuous_lvef_4year_v1"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"
STATUS_DIR="${OUTPUT_ROOT}/launcher_status"

SEED=42
EPOCHS=100
MIN_EPOCHS=10
PATIENCE=15
BATCH_SIZE=128
AUC_TOLERANCE=0.005
BOOTSTRAP_REPLICATES=5000
GPU_COUNT=8
LITERATURE_REFERENCE="${LITERATURE_REFERENCE:-ADD_COMPLETE_CITATION_OR_DOI_BEFORE_MANUSCRIPT_USE}"

MODALITY_PAIRS=(
    "ecg_literature_reduced_tabular"
)

TASKS=(
    "scd"
    "pfd"
)

COMMON_ARGS=(
    --folds_csv "${FOLDS_CSV}"
    --ecg_root "${ECG_ROOT}"
    --text_embedding_root "${TEXT_EMBEDDING_ROOT}"
    --text_results_root "${TEXT_RESULTS_ROOT}"
    --tabular_csv "${TABULAR_CSV}"
    --output_root "${OUTPUT_ROOT}"
    --outer_splits 5
    --inner_splits 4
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --min_epochs "${MIN_EPOCHS}"
    --patience "${PATIENCE}"
    --batch_size "${BATCH_SIZE}"
    --auc_tolerance "${AUC_TOLERANCE}"
    --bootstrap_replicates "${BOOTSTRAP_REPLICATES}"
    --expected_patients 730
    --expected_controls 577
    --expected_scd 71
    --expected_pfd 82
    --literature_reference "${LITERATURE_REFERENCE}"
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
    [[ -d "${ECG_ROOT}" ]] || { echo "Missing ${ECG_ROOT}" >&2; exit 1; }
    [[ -d "${TEXT_EMBEDDING_ROOT}" ]] || { echo "Missing ${TEXT_EMBEDDING_ROOT}" >&2; exit 1; }
    [[ -d "${TEXT_RESULTS_ROOT}" ]] || { echo "Missing ${TEXT_RESULTS_ROOT}" >&2; exit 1; }
    [[ -f "${TABULAR_CSV}" ]] || { echo "Missing ${TABULAR_CSV}" >&2; exit 1; }
}

reset_status_markers() {
    mkdir -p "${STATUS_DIR}"
    local worker
    for worker in 0 1 2 3 4 5 6 7; do
        rm -f \
            "${STATUS_DIR}/tune_worker_${worker}.done" \
            "${STATUS_DIR}/tune_worker_${worker}.failed" \
            "${STATUS_DIR}/final_worker_${worker}.done" \
            "${STATUS_DIR}/final_worker_${worker}.failed"
    done
    rm -f \
        "${STATUS_DIR}/selection.done" \
        "${STATUS_DIR}/pipeline.failed" \
        "${STATUS_DIR}/pipeline.done"
}

run_python() {
    python "${PYTHON_SCRIPT}" "$@" "${COMMON_ARGS[@]}"
}

run_worker_tasks() {
    local phase="$1"
    local worker_id="$2"
    local task_index=0
    local task
    local pair
    local outer_fold

    export CUDA_VISIBLE_DEVICES="${worker_id}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2

    for task in "${TASKS[@]}"; do
        for pair in "${MODALITY_PAIRS[@]}"; do
            for outer_fold in 0 1 2 3 4; do
                if (( task_index % GPU_COUNT == worker_id )); then
                    echo "[$(date --iso-8601=seconds)] ${phase}: ${task}, ${pair}, outer fold ${outer_fold}"
                    run_python \
                        --stage "${phase}" \
                        --task "${task}" \
                        --device cuda \
                        --modality_pair "${pair}" \
                        --outer_fold "${outer_fold}" || return $?
                fi
                task_index=$((task_index + 1))
            done
        done
    done
}

wait_for_selection() {
    while [[ ! -f "${STATUS_DIR}/selection.done" ]]; do
        if [[ -f "${STATUS_DIR}/pipeline.failed" ]]; then
            echo "Pipeline failed before selection completed." >&2
            return 1
        fi
        sleep 10
    done
}

worker_main() {
    local worker_id="$1"
    local log_file="${LOG_DIR}/gpu_worker_${worker_id}.log"
    activate_environment
    cd "${SCRIPT_DIR}"

    if ! run_worker_tasks tune "${worker_id}" 2>&1 | tee "${log_file}"; then
        touch "${STATUS_DIR}/tune_worker_${worker_id}.failed"
        touch "${STATUS_DIR}/pipeline.failed"
        echo "GPU worker ${worker_id} tuning failed."
        return 1
    fi
    touch "${STATUS_DIR}/tune_worker_${worker_id}.done"
    echo "GPU worker ${worker_id} completed tuning tasks."

    wait_for_selection

    if ! run_worker_tasks final "${worker_id}" 2>&1 | tee -a "${log_file}"; then
        touch "${STATUS_DIR}/final_worker_${worker_id}.failed"
        touch "${STATUS_DIR}/pipeline.failed"
        echo "GPU worker ${worker_id} final fitting failed."
        return 1
    fi
    touch "${STATUS_DIR}/final_worker_${worker_id}.done"
    echo "GPU worker ${worker_id} completed final tasks."
}

wait_for_workers() {
    local phase="$1"
    local worker
    while true; do
        if [[ -f "${STATUS_DIR}/pipeline.failed" ]]; then
            echo "At least one GPU worker failed during ${phase}." >&2
            return 1
        fi
        local complete=0
        for worker in 0 1 2 3 4 5 6 7; do
            if [[ -f "${STATUS_DIR}/${phase}_worker_${worker}.done" ]]; then
                complete=$((complete + 1))
            fi
        done
        echo "[$(date --iso-8601=seconds)] ${phase} workers complete: ${complete}/${GPU_COUNT}"
        if (( complete == GPU_COUNT )); then
            return 0
        fi
        sleep 15
    done
}

orchestrator_main() {
    local log_file="${LOG_DIR}/orchestrator.log"
    activate_environment
    cd "${SCRIPT_DIR}"
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8

    {
        echo "Waiting for all tuning workers."
        wait_for_workers tune || return $?

        echo "Selecting method-specific and overall configurations."
        run_python --stage select --device cpu || return $?
        touch "${STATUS_DIR}/selection.done"

        echo "Waiting for all final-model workers."
        wait_for_workers final || return $?

        echo "Aggregating untouched outer-test predictions."
        run_python --stage aggregate --device cpu || return $?

        echo "Evaluating multimodal arms with bootstrap confidence intervals."
        run_python --stage evaluate --device cpu || return $?

        touch "${STATUS_DIR}/pipeline.done"
        echo "Multimodal pipeline completed successfully."
        echo "Finished at: $(date --iso-8601=seconds)"
    } 2>&1 | tee "${log_file}"
}

if [[ "${1:-}" == "_worker" ]]; then
    worker_main "$2"
    exit $?
fi

if [[ "${1:-}" == "_orchestrator" ]]; then
    if ! orchestrator_main; then
        touch "${STATUS_DIR}/pipeline.failed"
        exit 1
    fi
    exit 0
fi

check_inputs
activate_environment
cd "${SCRIPT_DIR}"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

echo "Running multimodal preflight before launching GPU work."
run_python --stage prepare --device cpu
reset_status_markers

tmux new-session -d -s "${SESSION_NAME}" -n orchestrator \
    "bash '${SCRIPT_PATH}' _orchestrator"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on

for worker in 0 1 2 3 4 5 6 7; do
    tmux new-window -t "${SESSION_NAME}" -n "gpu${worker}" \
        "bash '${SCRIPT_PATH}' _worker '${worker}'"
done

tmux select-window -t "${SESSION_NAME}:orchestrator"

echo "Started nested multimodal fusion."
echo "tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Switch windows: Ctrl-b n (next) or Ctrl-b p (previous)"
echo "Orchestrator log: ${LOG_DIR}/orchestrator.log"
echo "GPU logs: ${LOG_DIR}/gpu_worker_0.log through gpu_worker_7.log"
echo "Output root: ${OUTPUT_ROOT}"
