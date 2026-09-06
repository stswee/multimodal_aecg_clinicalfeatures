#!/usr/bin/env bash
# Run the complete endpoint-specific comparative analysis from one launcher.
#
# Default behavior is resumable and never overwrites completed stages.
# To intentionally regenerate comparative outputs, use:
#   OVERWRITE=1 ./run_endpoint_specific_comparative_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_endpoint_specific_comparative_analysis.py"
CONDA_INIT="/home/sswee/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-shdb-af-analysis}"
MUSIC_ROOT="/home/sswee/music"

FOLDS_CSV="${MUSIC_ROOT}/ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv"
ECG_PREDICTIONS="${MUSIC_ROOT}/ecg_nested_4year_three_wave/threshold_analysis/pooled_outer_test_classifications.csv"
TABULAR_PREDICTIONS="${MUSIC_ROOT}/tabular_nested_4year_v2/evaluation/models/prompt_matched_no_ecg/selected_tabular/pooled_predictions_calibrated_and_classified.csv"
TEXT_PREDICTIONS="${MUSIC_ROOT}/text_nested_4year_v4_detailed/combined_evaluation/all_arms_pooled_predictions_calibrated_and_classified.csv"
MULTIMODAL_PREDICTIONS="${MUSIC_ROOT}/multimodal_nested_4year_v4_detailed/combined_evaluation/all_multimodal_pooled_outer_test_predictions.csv"
SUBJECT_INFO_CSV="${MUSIC_ROOT}/subject-info.csv"
PROMPT_CSV="${MUSIC_ROOT}/subject-info-cleaned-4year-with-prompts.csv"
ECG_ROOT="${MUSIC_ROOT}/ecg_nested_4year_three_wave"
TEXT_EMBEDDING_ROOT="${MUSIC_ROOT}/text_embeddings_4year_v4_detailed"
TEXT_RESULTS_ROOT="${MUSIC_ROOT}/text_nested_4year_v4_detailed"
MULTIMODAL_ROOT="${MUSIC_ROOT}/multimodal_nested_4year_v4_detailed"
LLAMA8B_CSV="${MUSIC_ROOT}/llm_responses_4year_v3_detailed/LLaMA3.1-8B-4year-responses.csv"
LLAMA3B_CSV="${MUSIC_ROOT}/llm_responses_4year_v3_detailed/LLaMA3.2-3B-4year-responses-postprocessed.csv"
MULTIMODAL_TRAINING_SCRIPT="${SCRIPT_DIR}/../6_Multimodal_Modeling/train_multimodal_nested_cv.py"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/sswee/music/comparative_analysis_4year_v4_detailed_endpoint_specific}"
SESSION="${SESSION:-comparative_4year_v4_detailed_endpoint_specific}"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"
STATUS_DIR="${OUTPUT_ROOT}/launcher_status"
OVERWRITE="${OVERWRITE:-0}"
COMPARATIVE_MULTIMODAL_ARM="${COMPARATIVE_MULTIMODAL_ARM:-selected_fusion}"
ATTRIBUTION_MULTIMODAL_ARM="${ATTRIBUTION_MULTIMODAL_ARM:-selected_fusion}"

activate_environment() {
    if [[ ! -f "${CONDA_INIT}" ]]; then
        echo "ERROR: Conda initialization file is missing: ${CONDA_INIT}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${CONDA_INIT}"
    conda activate "${CONDA_ENV}"
    cd "${SCRIPT_DIR}"
}

python_args() {
    PYTHON_ARGS=(
        --folds_csv "${FOLDS_CSV}"
        --ecg_predictions "${ECG_PREDICTIONS}"
        --tabular_predictions "${TABULAR_PREDICTIONS}"
        --text_predictions "${TEXT_PREDICTIONS}"
        --multimodal_predictions "${MULTIMODAL_PREDICTIONS}"
        --subject_info_csv "${SUBJECT_INFO_CSV}"
        --prompt_csv "${PROMPT_CSV}"
        --ecg_root "${ECG_ROOT}"
        --text_embedding_root "${TEXT_EMBEDDING_ROOT}"
        --text_results_root "${TEXT_RESULTS_ROOT}"
        --multimodal_root "${MULTIMODAL_ROOT}"
        --llama8b_csv "${LLAMA8B_CSV}"
        --llama3b_csv "${LLAMA3B_CSV}"
        --multimodal_training_script "${MULTIMODAL_TRAINING_SCRIPT}"
        --output_root "${OUTPUT_ROOT}"
        --bootstrap_replicates 5000
        --attribution_bootstrap_replicates 2000
        --seed 42
        --expected_patients 730
        --expected_controls 577
        --expected_scd 71
        --expected_pfd 82
        --threshold_min 0.02
        --threshold_max 0.25
        --threshold_step 0.0025
        --multimodal_arm "${MULTIMODAL_ARM:-${COMPARATIVE_MULTIMODAL_ARM}}"
        --calibration_groups 5
        --samples_per_class 6
        --ig_steps 24
        --refit_seeds 101 202 303
        --mask_fractions 0.05 0.10 0.20 0.30
        --random_control_repeats 10
        --chunk_stride 64
        --device cuda
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
        PYTHON_ARGS+=(--overwrite)
    fi
}

write_status() {
    local name="$1"
    local status="$2"
    mkdir -p "${STATUS_DIR}"
    printf '%s\n' "${status}" > "${STATUS_DIR}/${name}.status"
}

run_compare_worker() {
    activate_environment
    python_args
    mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
    set +e
    set -o pipefail
    python -u "${PYTHON_SCRIPT}" --stage compare "${PYTHON_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/comparative_statistics.log"
    status=${PIPESTATUS[0]}
    set -e
    write_status compare "${status}"
    echo "Comparative statistics exit status: ${status}"
    exit "${status}"
}

run_attribution_worker() {
    local fold="$1"
    MULTIMODAL_ARM="${ATTRIBUTION_MULTIMODAL_ARM}"
    activate_environment
    python_args
    mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
    export CUDA_VISIBLE_DEVICES="${fold}"
    set +e
    set -o pipefail
    python -u "${PYTHON_SCRIPT}" --stage attribute-fold --outer_fold "${fold}" \
        "${PYTHON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/attribution_outer_fold_${fold}.log"
    status=${PIPESTATUS[0]}
    set -e
    write_status "attribution_outer_fold_${fold}" "${status}"
    echo "Attribution outer fold ${fold} exit status: ${status}"
    exit "${status}"
}

run_finalizer() {
    MULTIMODAL_ARM="${ATTRIBUTION_MULTIMODAL_ARM}"
    activate_environment
    python_args
    mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
    echo "Waiting for comparative statistics and five attribution folds."
    while true; do
        complete=1
        [[ -s "${STATUS_DIR}/compare.status" ]] || complete=0
        for fold in 0 1 2 3 4; do
            [[ -s "${STATUS_DIR}/attribution_outer_fold_${fold}.status" ]] || complete=0
        done
        [[ "${complete}" == "1" ]] && break
        sleep 15
    done

    failed=0
    for file in "${STATUS_DIR}/compare.status" \
        "${STATUS_DIR}"/attribution_outer_fold_*.status; do
        status="$(<"${file}")"
        printf '%s: exit status %s\n' "$(basename "${file}" .status)" "${status}"
        [[ "${status}" == "0" ]] || failed=1
    done
    if [[ "${failed}" == "1" ]]; then
        echo "ERROR: At least one prerequisite stage failed; attribution aggregation was not run." >&2
        exit 1
    fi

    set +e
    set -o pipefail
    python -u "${PYTHON_SCRIPT}" --stage attribute-aggregate "${PYTHON_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/attribution_aggregate.log"
    status=${PIPESTATUS[0]}
    set -e
    write_status attribution_aggregate "${status}"
    echo "Attribution aggregation exit status: ${status}"
    if [[ "${status}" == "0" ]]; then
        echo "All comparative analyses completed successfully: ${OUTPUT_ROOT}"
    fi
    exit "${status}"
}

case "${1:-}" in
    _compare)
        run_compare_worker
        ;;
    _attribute)
        [[ "${2:-}" =~ ^[0-4]$ ]] || { echo "ERROR: invalid outer fold" >&2; exit 2; }
        run_attribution_worker "$2"
        ;;
    _finalize)
        run_finalizer
        ;;
esac

activate_environment
python_args

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Missing ${PYTHON_SCRIPT}" >&2
    exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed or not on PATH." >&2
    exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session ${SESSION} already exists." >&2
    echo "Attach with: tmux attach -t ${SESSION}"
    exit 1
fi

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
if [[ "${OVERWRITE}" == "1" ]]; then
    echo "OVERWRITE=1: completed outputs may be regenerated after preflight."
else
    echo "Resume-safe mode: completed stages will be reused."
fi

echo "Running endpoint-specific comparative preflight."
python -u "${PYTHON_SCRIPT}" --stage preflight "${PYTHON_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/preflight.log"

# Status files are stage-local coordination markers. Remove only those files;
# scientific output directories are preserved unless OVERWRITE=1 was requested.
find "${STATUS_DIR}" -maxdepth 1 -type f -name '*.status' -delete

tmux new-session -d -s "${SESSION}" -n compare \
    "env OVERWRITE='${OVERWRITE}' OUTPUT_ROOT='${OUTPUT_ROOT}' CONDA_ENV='${CONDA_ENV}' COMPARATIVE_MULTIMODAL_ARM='${COMPARATIVE_MULTIMODAL_ARM}' bash '${BASH_SOURCE[0]}' _compare; exec bash"
for fold in 0 1 2 3 4; do
    tmux new-window -t "${SESSION}" -n "gpu${fold}" \
        "env OVERWRITE='${OVERWRITE}' OUTPUT_ROOT='${OUTPUT_ROOT}' CONDA_ENV='${CONDA_ENV}' ATTRIBUTION_MULTIMODAL_ARM='${ATTRIBUTION_MULTIMODAL_ARM}' bash '${BASH_SOURCE[0]}' _attribute '${fold}'; exec bash"
done
tmux new-window -t "${SESSION}" -n finalizer \
    "env OVERWRITE='${OVERWRITE}' OUTPUT_ROOT='${OUTPUT_ROOT}' CONDA_ENV='${CONDA_ENV}' ATTRIBUTION_MULTIMODAL_ARM='${ATTRIBUTION_MULTIMODAL_ARM}' bash '${BASH_SOURCE[0]}' _finalize; exec bash"
tmux select-window -t "${SESSION}:compare"

echo
echo "Started final endpoint-specific comparative analysis."
echo "tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Switch windows: Ctrl-b n / Ctrl-b p"
echo "Logs: ${LOG_DIR}"
echo "Results: ${OUTPUT_ROOT}"
