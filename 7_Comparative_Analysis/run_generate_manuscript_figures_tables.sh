#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${SESSION_NAME:-music_manuscript_reporting_v4_detailed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/manuscript_outputs_v4_detailed}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/generate_manuscript_figures_tables.log"
STATUS_FILE="${LOG_DIR}/generate_manuscript_figures_tables.exit_status"

mkdir -p "${LOG_DIR}"
rm -f "${STATUS_FILE}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the detached reporting workflow." >&2
  exit 127
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Existing tmux session ${SESSION_NAME} is already running." >&2
  exit 1
fi

COMMAND="cd '${SCRIPT_DIR}' && if [[ -f '/home/sswee/miniconda3/etc/profile.d/conda.sh' ]]; then source '/home/sswee/miniconda3/etc/profile.d/conda.sh' && conda activate shdb-af-analysis; fi; python3 generate_manuscript_figures_tables.py --output-root '${OUTPUT_ROOT}' > '${LOG_FILE}' 2>&1; status=\$?; echo \${status} > '${STATUS_FILE}'; exit \${status}"
tmux new-session -d -s "${SESSION_NAME}" "${COMMAND}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
echo "Status file: ${STATUS_FILE}"
echo "Monitor with: tmux attach -t ${SESSION_NAME}"

while tmux has-session -t "${SESSION_NAME}" 2>/dev/null; do
  if [[ -f "${LOG_FILE}" ]]; then
    tail -n 5 "${LOG_FILE}" || true
  fi
  sleep 10
done

if [[ -f "${STATUS_FILE}" ]]; then
  status="$(cat "${STATUS_FILE}")"
else
  status=1
fi

echo "Final exit status: ${status}"
exit "${status}"
