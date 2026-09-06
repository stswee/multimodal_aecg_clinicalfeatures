#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/../../../../music/tabular_literature_reduced_continuous_lvef_4year_v1"
LOG_DIR="${ROOT}/launcher_logs"
SESSION="tabular_literature_reduced_continuous_lvef_posthoc_v1"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 1
fi

COMMAND=$(cat <<EOF
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shdb-af-analysis
cd "${SCRIPT_DIR}"
python -u tabular_literature_reduced_posthoc_plots.py --tabular_root "${ROOT}" --bootstrap_replicates 5000 --seed 42 --overwrite 2>&1 | tee "${LOG_DIR}/posthoc.log"
status=\${PIPESTATUS[0]}
echo "Posthoc exit status: \${status}"
exec bash
EOF
)

tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${COMMAND}")"
echo "Started reduced-clinical tabular posthoc analysis."
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${LOG_DIR}/posthoc.log"
