#!/usr/bin/env bash
# ============================================================
# tmux launcher for HRV / ECG TCN + Attention-MIL training
# Class-weighted loss + validation threshold tuning
# ============================================================

# ---------------- USER CONFIG ----------------
SESSION_NAME="tcn_attn_mil_multiclass_fold4"
GPU_ID=1

VAL_FOLD=4

CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv"

# Use ONE of the following depending on experiment:
SEGMENTS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments"
# FEATURES_DIR=".../preprocessed_segments_HRV"   # (if HRV version)

OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/tcn_attn_mil_outputs_multiclass"

# Training
EPOCHS=30
LR=1e-4
WEIGHT_DECAY=1e-5
SEED=42

# Encoder
EMBEDDING_DIM=256

# TCN
TCN_HIDDEN_DIM=256
TCN_LAYERS=4
TCN_KERNEL_SIZE=3
TCN_DROPOUT=0.2

# Attention MIL
ATTN_DIM=128

# Optional flags
NO_THRESHOLD_TUNING=""   # set to "--no_threshold_tuning" to disable

# Environment
CONDA_ENV="shdb-af-analysis"
PYTHON_SCRIPT="train_tcn_mil_hrv_multiclass.py"

LOG_DIR="tmux_logs"
mkdir -p ${LOG_DIR}
# ---------------- END CONFIG ----------------

echo "Starting tmux session: ${SESSION_NAME}"

tmux new-session -d -s ${SESSION_NAME}

tmux send-keys -t ${SESSION_NAME} "
echo 'Activating environment...'
source ~/.bashrc
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONUNBUFFERED=1

echo 'Running TCN + Attention MIL training'

python ${PYTHON_SCRIPT} \
  --val_fold ${VAL_FOLD} \
  --segments_dir ${SEGMENTS_DIR} \
  --csv_path ${CSV_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --seed ${SEED} \
  --embedding_dim ${EMBEDDING_DIM} \
  --tcn_hidden_dim ${TCN_HIDDEN_DIM} \
  --tcn_layers ${TCN_LAYERS} \
  --tcn_kernel_size ${TCN_KERNEL_SIZE} \
  --tcn_dropout ${TCN_DROPOUT} \
  --attn_dim ${ATTN_DIM} \
  --device cuda \
  2>&1 | tee ${LOG_DIR}/${SESSION_NAME}.log

echo 'Training finished.'
" C-m

echo "tmux session '${SESSION_NAME}' started."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
