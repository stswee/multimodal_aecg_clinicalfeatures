#!/usr/bin/env bash
#
# run_mil_tmux_survival.sh
#
# tmux launcher for TEMPORAL MIL ECG SURVIVAL training
# (SCD / PFD cause-specific Cox, temporal order preserved)
#

set -e

# ======================================================
# USER-EDITABLE SECTION
# ======================================================

# tmux session name
SESSION_NAME="mil_temporal_survival_fold0_emb256_SCDPFD"

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_mil_ecg_survival.py"

# Experiment arguments
VAL_FOLD=0

SEGMENTS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments"
CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_survival_5cv_cause_specific.csv"
OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_outputs_temporal_survival_emb256"

# ============================
# Model hyperparameters
# ============================

EMBEDDING_DIM=256

# Temporal TCN parameters
TCN_HIDDEN=256
TCN_LEVELS=7          # 7 → receptive field spans long Holter context
TCN_KERNEL=3
TCN_DROPOUT=0.10

# Attention
ATTN_DIM=128

# Encoding efficiency
ENCODE_CHUNK_SIZE=128   # segments encoded per forward chunk

# ============================
# Training hyperparameters
# ============================

EPOCHS=30
LR="1e-4"
PATIENT_BATCH_SIZE=2
DEVICE="cuda"

# GPU selection
CUDA_DEVICE=0

# Extra args (safe defaults)
EXTRA_ARGS="--seed 42 --weight_decay 1e-5 --num_workers 4"

# Optional explainability output
# Uncomment if you want per-epoch top-k segment explanations
EXTRA_ARGS="${EXTRA_ARGS} --save_val_topk --topk 20"

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
  --tcn_hidden ${TCN_HIDDEN} \
  --tcn_levels ${TCN_LEVELS} \
  --tcn_kernel ${TCN_KERNEL} \
  --tcn_dropout ${TCN_DROPOUT} \
  --attn_dim ${ATTN_DIM} \
  --encode_chunk_size ${ENCODE_CHUNK_SIZE} \
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
