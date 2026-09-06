#!/usr/bin/env bash
# Run multimodal post-analysis, figures, and the checkpoint-based text-shuffling
# experiment in separate tmux windows. Existing result directories are never
# overwritten unless the corresponding Python command is deliberately rerun
# with --overwrite.

set -euo pipefail

SESSION_NAME="multimodal_posthoc_4year_v4_detailed"
CONDA_ENV="shdb-af-analysis"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MUSIC_DIR="${SCRIPT_DIR}/../../music"

MULTIMODAL_ROOT="${MUSIC_DIR}/multimodal_nested_4year_v4_detailed"
FOLDS_CSV="${MUSIC_DIR}/ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv"
ECG_ROOT="${MUSIC_DIR}/ecg_nested_4year_three_wave"
TEXT_EMBEDDING_ROOT="${MUSIC_DIR}/text_embeddings_4year_v4_detailed"
TEXT_RESULTS_ROOT="${MUSIC_DIR}/text_nested_4year_v4_detailed"
TABULAR_CSV="${MUSIC_DIR}/subject-info.csv"
LOG_DIR="${MULTIMODAL_ROOT}/launcher_logs"

BOOTSTRAP_REPLICATES=5000
PERMUTATION_REPLICATES=1000
SEED=42

# Optional: set this environment variable to the v4 standardized comparative
# prediction CSV to generate the five-model manuscript decision curve.
COMPARATIVE_PREDICTIONS="${COMPARATIVE_PREDICTIONS:-}"

activate_environment() {
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc" 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    cd "${SCRIPT_DIR}"
    export PYTHONUNBUFFERED=1
}

run_posthoc() {
    activate_environment
    python "${SCRIPT_DIR}/multimodal_posthoc_evaluation.py" \
        --multimodal_root "${MULTIMODAL_ROOT}" \
        --bootstrap_replicates "${BOOTSTRAP_REPLICATES}" \
        --seed "${SEED}" \
        2>&1 | tee "${LOG_DIR}/multimodal_posthoc_evaluation.log"
    status=${PIPESTATUS[0]}
    echo "Posthoc evaluation exit status: ${status}"
    return "${status}"
}

run_plots() {
    activate_environment
    export CUDA_VISIBLE_DEVICES=""
    plot_extra_args=()
    if [[ -n "${COMPARATIVE_PREDICTIONS}" ]]; then
        plot_extra_args+=(--comparative_predictions "${COMPARATIVE_PREDICTIONS}")
    fi
    python "${SCRIPT_DIR}/plot_multimodal_evaluation.py" \
        --multimodal_root "${MULTIMODAL_ROOT}" \
        --bootstrap_replicates "${BOOTSTRAP_REPLICATES}" \
        --seed "${SEED}" \
        --threshold_min 0.02 \
        --threshold_max 0.25 \
        --threshold_points 93 \
        "${plot_extra_args[@]}" \
        2>&1 | tee "${LOG_DIR}/plot_multimodal_evaluation.log"
    status=${PIPESTATUS[0]}
    echo "Plotting exit status: ${status}"
    return "${status}"
}

run_shuffling() {
    activate_environment
    export CUDA_VISIBLE_DEVICES=0
    python "${SCRIPT_DIR}/multimodal_text_shuffling_analysis.py" \
        --folds_csv "${FOLDS_CSV}" \
        --ecg_root "${ECG_ROOT}" \
        --text_embedding_root "${TEXT_EMBEDDING_ROOT}" \
        --text_results_root "${TEXT_RESULTS_ROOT}" \
        --tabular_csv "${TABULAR_CSV}" \
        --multimodal_root "${MULTIMODAL_ROOT}" \
        --permutation_replicates "${PERMUTATION_REPLICATES}" \
        --seed "${SEED}" \
        --device cuda \
        2>&1 | tee "${LOG_DIR}/multimodal_text_shuffling.log"
    status=${PIPESTATUS[0]}
    echo "Text-shuffling exit status: ${status}"
    return "${status}"
}

case "${1:-launcher}" in
    _posthoc) run_posthoc; exit $? ;;
    _plots) run_plots; exit $? ;;
    _shuffle) run_shuffling; exit $? ;;
esac

mkdir -p "${LOG_DIR}"
for required in \
    "${SCRIPT_DIR}/multimodal_posthoc_evaluation.py" \
    "${SCRIPT_DIR}/plot_multimodal_evaluation.py" \
    "${SCRIPT_DIR}/multimodal_text_shuffling_analysis.py" \
    "${MULTIMODAL_ROOT}/analysis_setup/analysis_manifest.json" \
    "${MULTIMODAL_ROOT}/combined_evaluation/all_multimodal_pooled_outer_test_predictions.csv" \
    "${TEXT_RESULTS_ROOT}/combined_evaluation/combined_evaluation_manifest.json"
do
    [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 1; }
done

if [[ -n "${COMPARATIVE_PREDICTIONS}" && ! -s "${COMPARATIVE_PREDICTIONS}" ]]; then
    echo "Missing comparative prediction file: ${COMPARATIVE_PREDICTIONS}" >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n posthoc \
    "bash '${BASH_SOURCE[0]}' _posthoc; status=\$?; echo 'Window exit status:' \$status; exec bash"
tmux new-window -t "${SESSION_NAME}" -n plots \
    "bash '${BASH_SOURCE[0]}' _plots; status=\$?; echo 'Window exit status:' \$status; exec bash"
checkpoint_count=$(find "${MULTIMODAL_ROOT}/tasks" -path '*/arms/*/checkpoint.pt' -type f 2>/dev/null | wc -l)
if (( checkpoint_count == 150 )); then
    tmux new-window -t "${SESSION_NAME}" -n shuffle_gpu0 \
        "bash '${BASH_SOURCE[0]}' _shuffle; status=\$?; echo 'Window exit status:' \$status; exec bash"
else
    echo "Warning: found ${checkpoint_count}/150 expected endpoint-specific fusion checkpoints."
    echo "Posthoc evaluation and plots will run, but the shuffling window was not started."
fi
tmux select-window -t "${SESSION_NAME}:posthoc"

echo "Started multimodal post-analysis."
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Switch windows: Ctrl-b n or Ctrl-b p"
echo "Results root: ${MULTIMODAL_ROOT}"
