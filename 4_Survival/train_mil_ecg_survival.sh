#!/usr/bin/env bash
#
# run_mil_tmux_survival.sh
#
# tmux launcher for MIL ECG SURVIVAL training (Cox / C-index)
#

set -e

# ======================================================
# USER-EDITABLE SECTION
# ======================================================

# tmux session name
SESSION_NAME="mil_survival_fold4_emb256_SCD"

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_mil_ecg_survival.py"

# Experiment arguments
VAL_FOLD=4

SEGMENTS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments"
CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv_survival_SCD.csv"
OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_outputs_survival_emb256_SCD"

# Training hyperparameters
EPOCHS=30
LR="1e-4"
ATTENTION_DIM=128
EMBEDDING_DIM=256
PATIENT_BATCH_SIZE=2
DEVICE="cuda"

# GPU selection
CUDA_DEVICE=4

# Extra args (optional / commonly changed)
EXTRA_ARGS="--seed 42 --weight_decay 1e-5 --num_workers 4"

# ======================================================
# BUILD COMMAND
# ======================================================

CMD="export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
python ${PYTHON_SCRIPT} \
  --val_fold ${VAL_FOLD} \
  --segments_dir ${SEGMENTS_DIR} \
  --csv_path ${CSV_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --embedding_dim ${EMBEDDING_DIM} \
  --attention_dim ${ATTENTION_DIM} \
  --patient_batch_size ${PATIENT_BATCH_SIZE} \
  --device ${DEVICE} \
  ${EXTRA_ARGS}
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
