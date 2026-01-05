#!/usr/bin/env bash
# ============================================================
# tmux launcher for HRV TCN + MIL training (feature-based)
# WITH Top-K MIL pooling + class-weighted loss
# ============================================================

# ---------------- USER CONFIG ----------------
SESSION_NAME="hrv_tcn_mil_fold0"
GPU_ID=0

VAL_FOLD=1

CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv"
FEATURES_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments_HRV"
OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/hrv_tcn_mil_outputs"

# Training
EPOCHS=30
LR=1e-3
WEIGHT_DECAY=1e-5
SEED=42

# Model
EMBEDDING_DIM=64

# TCN
TCN_HIDDEN_DIM=128
TCN_LAYERS=5
TCN_KERNEL_SIZE=3
TCN_DROPOUT=0.2

# MIL
TOPK=50

# Environment
CONDA_ENV="shdb-af-analysis"
PYTHON_SCRIPT="train_tcn_mil_hrv.py"

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

echo 'Running HRV TCN + MIL training (Top-K pooling)'

python ${PYTHON_SCRIPT} \
  --val_fold ${VAL_FOLD} \
  --features_dir ${FEATURES_DIR} \
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
  --topk ${TOPK} \
  --device cuda \
  2>&1 | tee ${LOG_DIR}/${SESSION_NAME}.log

echo 'Training finished.'
" C-m

echo "tmux session '${SESSION_NAME}' started."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
