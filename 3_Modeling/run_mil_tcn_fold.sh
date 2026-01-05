#!/usr/bin/env bash
# ============================================
# tmux launcher for MIL TCN ECG training
# ============================================

# ---- USER CONFIG ----
SESSION_NAME="mil_tcn_fold4"
GPU_ID=1

VAL_FOLD=4
CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv"
SEGMENTS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments"
OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/tcn_binary"

EPOCHS=10
LR=1e-4
WEIGHT_DECAY=1e-5
SEED=42

EMBEDDING_DIM=256
TCN_HIDDEN_DIM=256
TCN_LAYERS=4
TCN_KERNEL_SIZE=3
TCN_DROPOUT=0.2

CONDA_ENV="shdb-af-analysis"
PYTHON_SCRIPT="train_mil_ecg_tcn.py"

LOG_DIR="tmux_logs"
mkdir -p ${LOG_DIR}

# ---- END CONFIG ----

echo "Starting tmux session: ${SESSION_NAME}"

tmux new-session -d -s ${SESSION_NAME}

tmux send-keys -t ${SESSION_NAME} "
echo 'Activating environment...'
source ~/.bashrc
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONUNBUFFERED=1

echo 'Running MIL TCN training'
python ${PYTHON_SCRIPT} \
  --val_fold ${VAL_FOLD} \
  --csv_path ${CSV_PATH} \
  --segments_dir ${SEGMENTS_DIR} \
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
  --device cuda \
  2>&1 | tee ${LOG_DIR}/${SESSION_NAME}.log

echo 'Training finished.'
" C-m

echo "tmux session '${SESSION_NAME}' started."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
