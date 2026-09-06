#!/usr/bin/env bash
# Architecture-matched tabular MLP sensitivity analysis: five outer folds on five GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_INIT="/home/sswee/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="shdb-af-analysis"
MUSIC_DIR="${SCRIPT_DIR}/../../music"
TABULAR_CSV="${MUSIC_DIR}/subject-info.csv"
FOLDS_CSV="${MUSIC_DIR}/ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv"
OUTPUT_ROOT="${MUSIC_DIR}/tabular_mlp_matched_4year_v2"
TABULAR_REFERENCE="${MUSIC_DIR}/tabular_nested_4year_v2/evaluation/models/prompt_matched_no_ecg/selected_tabular/pooled_predictions_calibrated_and_classified.csv"
TEXT_REFERENCE="${MUSIC_DIR}/text_nested_4year_v2/evaluation/all_arms_pooled_predictions_calibrated_and_classified.csv"
SESSION_NAME="tabular_mlp_matched_4year_v2"
LOG_DIR="${OUTPUT_ROOT}/launcher_logs"
STATUS_DIR="${LOG_DIR}/worker_status"

COMMON_ARGS=(
  --tabular_csv "${TABULAR_CSV}"
  --folds_csv "${FOLDS_CSV}"
  --output_root "${OUTPUT_ROOT}"
  --tabular_reference "${TABULAR_REFERENCE}"
  --text_reference "${TEXT_REFERENCE}"
  --outer_splits 5 --inner_splits 4
  --epochs 100 --min_epochs 10 --patience 15
  --auc_tolerance 0.005 --bootstrap_replicates 5000 --seed 42
  --expected_patients 730 --expected_controls 577 --expected_scd 71 --expected_pfd 82
  --expected_anticoagulant_yes 610 --expected_anticoagulant_no 120
)

activate_environment() {
  if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "Missing conda initialization file: ${CONDA_INIT}" >&2
    return 1
  fi
  # shellcheck source=/dev/null
  source "${CONDA_INIT}"
  conda activate "${CONDA_ENV}"
  cd "${SCRIPT_DIR}"
  export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
}

if [[ "${1:-}" == "_worker" ]]; then
  gpu="${2:?GPU index required}"
  fold="${3:?outer fold required}"
  overwrite="${4:-0}"
  mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
  activate_environment
  extra=()
  [[ "${overwrite}" == "1" ]] && extra+=(--overwrite)
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" python train_tabular_mlp_matched_nested_cv.py \
    --stage fit-fold --outer_fold "${fold}" --device cuda \
    "${COMMON_ARGS[@]}" "${extra[@]}" 2>&1 | tee "${LOG_DIR}/outer_fold_${fold}_gpu_${gpu}.log"
  status="${PIPESTATUS[0]}"
  set -e
  temporary="${STATUS_DIR}/outer_fold_${fold}.status.tmp"
  echo "${status}" > "${temporary}"
  mv "${temporary}" "${STATUS_DIR}/outer_fold_${fold}.status"
  echo "Outer fold ${fold} exit status: ${status}"
  exit "${status}"
fi

if [[ "${1:-}" == "_orchestrator" ]]; then
  overwrite="${2:-0}"
  activate_environment
  echo "Waiting for five outer-fold workers."
  while :; do
    status_count="$(find "${STATUS_DIR}" -maxdepth 1 -name 'outer_fold_*.status' -type f 2>/dev/null | wc -l)"
    [[ "${status_count}" -eq 5 ]] && break
    echo "$(date --iso-8601=seconds): ${status_count}/5 workers have reported status."
    sleep 15
  done
  failed=0
  for fold in 0 1 2 3 4; do
    status="$(tr -d '[:space:]' < "${STATUS_DIR}/outer_fold_${fold}.status")"
    echo "Outer fold ${fold}: exit status ${status}"
    [[ "${status}" == "0" ]] || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one worker failed; aggregation was not started." >&2
    exit 1
  fi
  extra=()
  [[ "${overwrite}" == "1" ]] && extra+=(--overwrite)
  python train_tabular_mlp_matched_nested_cv.py --stage aggregate \
    "${COMMON_ARGS[@]}" "${extra[@]}" 2>&1 | tee "${LOG_DIR}/aggregate.log"
  status="${PIPESTATUS[0]}"
  echo "Aggregation exit status: ${status}"
  exit "${status}"
fi

overwrite=0
if [[ "${1:-}" == "--overwrite" ]]; then
  overwrite=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--overwrite]" >&2
  exit 2
fi

for required in "${TABULAR_CSV}" "${FOLDS_CSV}" "${TABULAR_REFERENCE}" "${TEXT_REFERENCE}" "${SCRIPT_DIR}/train_tabular_mlp_matched_nested_cv.py"; do
  [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
command -v tmux >/dev/null || { echo "tmux is not installed or not on PATH." >&2; exit 1; }
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
find "${STATUS_DIR}" -maxdepth 1 -name 'outer_fold_*.status*' -type f -delete
activate_environment
prepare_extra=()
[[ "${overwrite}" -eq 1 ]] && prepare_extra+=(--overwrite)
python train_tabular_mlp_matched_nested_cv.py --stage prepare "${COMMON_ARGS[@]}" "${prepare_extra[@]}"

tmux new-session -d -s "${SESSION_NAME}" -n orchestrator \
  "bash '${SCRIPT_DIR}/run_tabular_mlp_matched_nested_cv.sh' _orchestrator '${overwrite}'; status=\$?; echo 'Orchestrator exit status:' \${status}; exec bash"
for fold in 0 1 2 3 4; do
  tmux new-window -t "${SESSION_NAME}" -n "fold${fold}" \
    "bash '${SCRIPT_DIR}/run_tabular_mlp_matched_nested_cv.sh' _worker '${fold}' '${fold}' '${overwrite}'; status=\$?; echo 'Worker exit status:' \${status}; exec bash"
done
tmux select-window -t "${SESSION_NAME}:orchestrator"

echo "Started architecture-matched tabular MLP sensitivity analysis."
echo "tmux session: ${SESSION_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Switch windows: Ctrl-b n or Ctrl-b p"
echo "Logs: ${LOG_DIR}"
echo "Output root: ${OUTPUT_ROOT}"