#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MUSIC MULTIMODAL ECG/TEXT EXPLANATION ARTIFACTS
# ==============================================================================
# Starts one detached tmux session. Within that session, folds are processed
# sequentially from the completed detailed-response v4 checkpoints.
# This launcher does not train or select models.
# ==============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_multimodal_ecg_text_explanations.py"
MUSIC_DIR="${SCRIPT_DIR}/../../music"

ECG_ROOT="${MUSIC_DIR}/ecg_nested_4year_three_wave"
TEXT_EMBEDDING_ROOT="${MUSIC_DIR}/text_embeddings_4year_v4_detailed"
TEXT_RESULTS_ROOT="${MUSIC_DIR}/text_nested_4year_v4_detailed"
MULTIMODAL_ROOT="${MUSIC_DIR}/multimodal_nested_4year_v4_detailed"
LLAMA8B_CSV="${MUSIC_DIR}/llm_responses_4year_v3_detailed/LLaMA3.1-8B-4year-responses.csv"
LLAMA3B_CSV="${MUSIC_DIR}/llm_responses_4year_v3_detailed/LLaMA3.2-3B-4year-responses-postprocessed.csv"
MULTIMODAL_TRAINING_SCRIPT="${SCRIPT_DIR}/../6_Multimodal_Modeling/train_multimodal_nested_cv.py"

SESSION_NAME="${SESSION_NAME:-multimodal_ecg_text_explanations_v4_detailed}"
CONDA_ENV="${CONDA_ENV:-shdb-af-analysis}"
FOLDS="${FOLDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cuda}"
TOP_K_CASES="${TOP_K_CASES:-15}"
TOP_TEXT_TOKENS="${TOP_TEXT_TOKENS:-30}"
PATIENT_IDS="${PATIENT_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${MULTIMODAL_ROOT}/explain_multimodal_ecg_text}"
LOG_DIR="${OUTPUT_DIR}/launcher_logs"
LOG_FILE="${LOG_DIR}/generate_multimodal_ecg_text_explanations.log"
STATUS_FILE="${LOG_DIR}/generate_multimodal_ecg_text_explanations.exit_status"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-music-explain}"

TARGETS=(
    "SCD:selected_fusion"
    "PFD:selected_fusion"
)

activate_environment() {
    local conda_init="${HOME}/miniconda3/etc/profile.d/conda.sh"
    if [[ -f "${conda_init}" ]]; then
        # shellcheck disable=SC1090
        source "${conda_init}"
        conda activate "${CONDA_ENV}"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${CONDA_ENV}"
    else
        echo "ERROR: conda is unavailable." >&2
        exit 1
    fi
}

check_inputs() {
    [[ -f "${PYTHON_SCRIPT}" ]] || { echo "ERROR: Missing ${PYTHON_SCRIPT}" >&2; exit 1; }
    [[ -d "${ECG_ROOT}" ]] || { echo "ERROR: Missing ECG root: ${ECG_ROOT}" >&2; exit 1; }
    [[ -d "${TEXT_EMBEDDING_ROOT}" ]] || { echo "ERROR: Missing detailed text embeddings root: ${TEXT_EMBEDDING_ROOT}" >&2; exit 1; }
    [[ -d "${TEXT_RESULTS_ROOT}" ]] || { echo "ERROR: Missing detailed text modeling root: ${TEXT_RESULTS_ROOT}" >&2; exit 1; }
    [[ -d "${MULTIMODAL_ROOT}" ]] || { echo "ERROR: Missing detailed multimodal root: ${MULTIMODAL_ROOT}" >&2; exit 1; }
    [[ -f "${LLAMA8B_CSV}" ]] || { echo "ERROR: Missing detailed LLaMA3.1-8B responses: ${LLAMA8B_CSV}" >&2; exit 1; }
    [[ -f "${LLAMA3B_CSV}" ]] || { echo "ERROR: Missing postprocessed detailed LLaMA3.2-3B responses: ${LLAMA3B_CSV}" >&2; exit 1; }
    [[ -f "${MULTIMODAL_TRAINING_SCRIPT}" ]] || { echo "ERROR: Missing multimodal training script: ${MULTIMODAL_TRAINING_SCRIPT}" >&2; exit 1; }
}

run_inside_tmux() {
    activate_environment
    check_inputs
    cd "${SCRIPT_DIR}"
    mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}"
    export MPLCONFIGDIR PYTHONUNBUFFERED=1

    local patient_args=()
    if [[ -n "${PATIENT_IDS}" ]]; then
        # shellcheck disable=SC2206
        patient_args=(--patient_ids ${PATIENT_IDS})
    fi

    echo "Started multimodal explanation generation at:"
    date --iso-8601=seconds
    echo "Folds: ${FOLDS}"
    echo "Targets: ${TARGETS[*]}"
    echo "Output: ${OUTPUT_DIR}"

    local fold
    for fold in ${FOLDS}; do
        if [[ ! "${fold}" =~ ^[0-4]$ ]]; then
            echo "ERROR: Invalid fold: ${fold}" >&2
            exit 2
        fi

        echo
        echo "=========================================="
        echo "Generating multimodal explanations for fold ${fold}"
        echo "=========================================="

        python -u "${PYTHON_SCRIPT}" \
            --val_fold "${fold}" \
            --ecg_embedding_dir "${ECG_ROOT}" \
            --ecg_encoder_root "${ECG_ROOT}" \
            --text_embedding_dir "${TEXT_EMBEDDING_ROOT}" \
            --text_model_root "${TEXT_RESULTS_ROOT}" \
            --multimodal_root "${MULTIMODAL_ROOT}" \
            --out_dir "${OUTPUT_DIR}" \
            --top_k_cases "${TOP_K_CASES}" \
            --top_text_tokens "${TOP_TEXT_TOKENS}" \
            --max_text_length 512 \
            --chunk_stride 64 \
            --device "${DEVICE}" \
            --targets "${TARGETS[@]}" \
            "${patient_args[@]}"
    done

    echo
    echo "Multimodal explanation generation completed successfully."
    echo "Results: ${OUTPUT_DIR}"
    echo "Finished at:"
    date --iso-8601=seconds
}

launch_tmux() {
    command -v tmux >/dev/null 2>&1 || {
        echo "ERROR: tmux is not installed or is not on PATH." >&2
        exit 1
    }

    check_inputs
    mkdir -p "${LOG_DIR}"
    : > "${STATUS_FILE}"

    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "ERROR: tmux session '${SESSION_NAME}' already exists." >&2
        echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
        exit 1
    fi

    local command
    local wrapped_command
    printf -v command \
        "env CONDA_ENV=%q FOLDS=%q DEVICE=%q TOP_K_CASES=%q TOP_TEXT_TOKENS=%q PATIENT_IDS=%q OUTPUT_DIR=%q MPLCONFIGDIR=%q bash %q _inside 2>&1 | tee %q; status=\${PIPESTATUS[0]}; echo \${status} > %q; echo; echo 'Explanation launcher exit status:' \${status}; exec bash" \
        "${CONDA_ENV}" "${FOLDS}" "${DEVICE}" "${TOP_K_CASES}" \
        "${TOP_TEXT_TOKENS}" "${PATIENT_IDS}" "${OUTPUT_DIR}" \
        "${MPLCONFIGDIR}" "${SCRIPT_PATH}" "${LOG_FILE}" "${STATUS_FILE}"

    printf -v wrapped_command "bash -lc %q" "${command}"
    tmux new-session -d -s "${SESSION_NAME}" -n explanations "${wrapped_command}"
    tmux set-option -t "${SESSION_NAME}" remain-on-exit on

    echo "Started multimodal explanation generation."
    echo "tmux session: ${SESSION_NAME}"
    echo "Attach: tmux attach -t ${SESSION_NAME}"
    echo "Log: ${LOG_FILE}"
    echo "Exit status: ${STATUS_FILE}"
    echo "Results: ${OUTPUT_DIR}"
}

case "${1:-}" in
    _inside)
        run_inside_tmux
        ;;
    "")
        launch_tmux
        ;;
    *)
        echo "Usage: $0" >&2
        exit 2
        ;;
esac
