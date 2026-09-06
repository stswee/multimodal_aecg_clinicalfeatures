#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/../../../../music/multimodal_literature_reduced_continuous_lvef_4year_v1"
LOG_DIR="${ROOT}/launcher_logs"
SESSION="multimodal_literature_reduced_continuous_lvef_posthoc_v1"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 1
fi

COMMAND=$(cat <<EOF
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shdb-af-analysis
cd "${SCRIPT_DIR}"
python -u multimodal_literature_reduced_posthoc.py --multimodal_root "${ROOT}" --bootstrap_replicates 5000 --seed 42 --expected_scd_patients 648 --expected_pfd_patients 659 --overwrite 2>&1 | tee "${LOG_DIR}/posthoc.log"
s1=\${PIPESTATUS[0]}
if (( s1 != 0 )); then echo "Posthoc exit status: \${s1}"; exec bash; fi
python -u plot_multimodal_literature_reduced.py --multimodal_root "${ROOT}" --bootstrap_replicates 5000 --seed 42 --calibration_bins 5 --threshold_min 0.02 --threshold_max 0.25 --threshold_points 93 --expected_scd_patients 648 --expected_pfd_patients 659 --overwrite 2>&1 | tee "${LOG_DIR}/plots.log"
s2=\${PIPESTATUS[0]}
echo "Plotting exit status: \${s2}"
exec bash
EOF
)

tmux new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${COMMAND}")"
echo "Started reduced-clinical multimodal posthoc analysis."
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs: ${LOG_DIR}/posthoc.log and ${LOG_DIR}/plots.log"
