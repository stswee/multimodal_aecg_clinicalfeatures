#!/usr/bin/env bash
#
# run_mil_tmux_survival_validation.sh
#
# tmux launcher for MIL ECG SURVIVAL VALIDATION
# (C-index, time-dependent AUC, KM curves, calibration, HR)
#

set -e

# ======================================================
# USER-EDITABLE SECTION
# ======================================================

# tmux session name
SESSION_NAME="mil_survival_validation_time_SCD"

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/validate_mil_ecg_survival.py"

# Paths
SEGMENTS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments"
CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv_survival_cardiac_time.csv"
MODELS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_outputs_survival_emb256_time_SCD"
OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_validation_survival_time_SCD"

# Model hyperparameters (must match training)
EMBEDDING_DIM=256
ATTENTION_DIM=128
DEVICE="cuda"

# GPU selection
CUDA_DEVICE=0

# ======================================================
# BUILD COMMAND
# ======================================================

CMD="export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
python ${PYTHON_SCRIPT} \
  --csv_path ${CSV_PATH} \
  --segments_dir ${SEGMENTS_DIR} \
  --models_dir ${MODELS_DIR} \
  --embedding_dim ${EMBEDDING_DIM} \
  --attention_dim ${ATTENTION_DIM} \
  --device ${DEVICE} \
  --out_dir ${OUTPUT_DIR}
"

# ======================================================
# LAUNCH TMUX
# ======================================================

tmux new-session -d -s "${SESSION_NAME}"

tmux send-keys -t "${SESSION_NAME}" "cd ${SCRIPT_DIR}" C-m
tmux send-keys -t "${SESSION_NAME}" "${CMD}" C-m

echo "Started tmux session: ${SESSION_NAME}"
echo
echo "Command:"
echo "${CMD}"
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
