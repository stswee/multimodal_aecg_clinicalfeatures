#!/usr/bin/env bash

# ==============================================================================
# DETAILED-RESPONSE ENDPOINT-SPECIFIC TEXT POSTHOC EVALUATION
# ==============================================================================
#
# Generates:
#   1. pooled ROC and precision-recall curves;
#   2. calibration plots; and
#   3. exploratory decision curves with bootstrap uncertainty.
#
# This script reads saved outer-test predictions from the detailed-response
# nested analysis. It does not retrain classifiers, refit calibration models,
# regenerate embeddings, or reselect classification thresholds.
# ==============================================================================

set -euo pipefail

SESSION_NAME="text_posthoc_4year_v4_detailed"
CONDA_ENV="shdb-af-analysis"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

PLOT_SCRIPT="${SCRIPT_DIR}/plot_text_evaluation.py"
DCA_SCRIPT="${SCRIPT_DIR}/text_decision_curve_analysis.py"

MUSIC_DIR="${SCRIPT_DIR}/../../music"
OUTPUT_ROOT="${MUSIC_DIR}/text_nested_4year_v4_detailed"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"

BOOTSTRAP_REPLICATES=5000
SEED=42
THRESHOLD_MIN=0.02
THRESHOLD_MAX=0.25
THRESHOLD_POINTS=93

activate_environment() {
    # shellcheck disable=SC1091
    source /home/sswee/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"
}

check_inputs() {
    local required

    for required in \
        "${PLOT_SCRIPT}" \
        "${DCA_SCRIPT}" \
        "${OUTPUT_ROOT}/combined_evaluation/combined_evaluation_manifest.json"
    do
        if [[ ! -s "${required}" ]]; then
            echo "ERROR: Required file is missing or empty:" >&2
            echo "${required}" >&2
            exit 1
        fi
    done

    local task
    local arm

    for task in scd pfd; do
        required="${OUTPUT_ROOT}/tasks/${task}/analysis_setup/prepare_manifest.json"
        if [[ ! -s "${required}" ]]; then
            echo "ERROR: Missing analysis-preparation manifest:" >&2
            echo "${required}" >&2
            exit 1
        fi

        for arm in \
            full_risk_no_ecg \
            label_only_no_ecg \
            rationale_only_no_ecg \
            joint_full_risk_no_ecg \
            neutral_summary_no_ecg \
            deterministic_template_no_ecg \
            full_risk_with_ecg \
            label_only_with_ecg \
            rationale_only_with_ecg
        do
            required="${OUTPUT_ROOT}/tasks/${task}/evaluation/arms/${arm}/pooled_predictions_calibrated_and_classified.csv"

            if [[ ! -s "${required}" ]]; then
                echo "ERROR: Missing calibrated prediction file:" >&2
                echo "${required}" >&2
                exit 1
            fi
        done
    done
}

run_inside_tmux() {
    activate_environment
    check_inputs

    mkdir -p "${LOG_DIR}"

    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export CUDA_VISIBLE_DEVICES=""

    echo "Started detailed-response text posthoc evaluation at:"
    date --iso-8601=seconds

    echo
    echo "Generating ROC, precision-recall, and calibration figures."

    set +e
    python -u "${PLOT_SCRIPT}" \
        --output_root "${OUTPUT_ROOT}" \
        2>&1 | tee "${LOG_DIR}/plot_text_evaluation_v4_detailed.log"
    plot_status="${PIPESTATUS[0]}"
    set -e

    echo "Plotting exit status: ${plot_status}"

    if [[ "${plot_status}" -ne 0 ]]; then
        echo "ERROR: Plot generation failed. Decision-curve analysis was not run." >&2
        exit "${plot_status}"
    fi

    echo
    echo "Running exploratory decision-curve analysis."

    set +e
    python -u "${DCA_SCRIPT}" \
        --output_root "${OUTPUT_ROOT}" \
        --threshold_min "${THRESHOLD_MIN}" \
        --threshold_max "${THRESHOLD_MAX}" \
        --threshold_points "${THRESHOLD_POINTS}" \
        --bootstrap_replicates "${BOOTSTRAP_REPLICATES}" \
        --seed "${SEED}" \
        2>&1 | tee "${LOG_DIR}/text_decision_curve_v4_detailed.log"
    dca_status="${PIPESTATUS[0]}"
    set -e

    echo "Decision-curve exit status: ${dca_status}"

    if [[ "${dca_status}" -ne 0 ]]; then
        exit "${dca_status}"
    fi

    echo
    echo "Detailed-response text posthoc evaluation completed successfully."
    echo "Figures:"
    echo "${OUTPUT_ROOT}/text_evaluation_figures_endpoint_specific"
    echo "Decision curves:"
    echo "${OUTPUT_ROOT}/text_decision_curve_analysis_endpoint_specific"
    echo "Finished at:"
    date --iso-8601=seconds
}

launch_tmux() {
    command -v tmux >/dev/null 2>&1 || {
        echo "ERROR: tmux is not installed or is not on PATH." >&2
        exit 1
    }

    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "ERROR: A tmux session named ${SESSION_NAME} already exists." >&2
        echo "Attach with:" >&2
        echo "tmux attach -t ${SESSION_NAME}" >&2
        exit 1
    fi

    mkdir -p "${LOG_DIR}"

    printf -v tmux_command \
        '%q %q %q' \
        bash \
        "${SCRIPT_PATH}" \
        _inside

    tmux new-session \
        -d \
        -s "${SESSION_NAME}" \
        -n posthoc \
        "${tmux_command}; status=\$?; echo; echo 'Posthoc launcher exit status:' \${status}; exec bash"

    echo "Started detailed-response text posthoc evaluation."
    echo "tmux session: ${SESSION_NAME}"
    echo "Attach with:"
    echo "tmux attach -t ${SESSION_NAME}"
    echo
    echo "Plotting log:"
    echo "${LOG_DIR}/plot_text_evaluation_v4_detailed.log"
    echo
    echo "Decision-curve log:"
    echo "${LOG_DIR}/text_decision_curve_v4_detailed.log"
}

case "${1:-}" in
    _inside)
        run_inside_tmux
        ;;
    "")
        launch_tmux
        ;;
    *)
        echo "Usage: bash $(basename -- "${SCRIPT_PATH}")" >&2
        exit 2
        ;;
esac
