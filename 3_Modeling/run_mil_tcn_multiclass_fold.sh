#!/usr/bin/env bash
# ==========================================================
# tmux launcher: Multiclass MIL TCN + Temporal Attention
# (Healthy / SCD / PFD)
# ==========================================================

# ---------------- USER CONFIG ----------------
SESSION_NAME="mil_tcn_attention_multiclass_fold1"
GPU_ID=1

VAL_FOLD=1
CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv"
SEGMENTS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments"
OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/tcn_multiclass"

# Training hyperparameters
EPOCHS=10
LR=1e-4
SEED=42

# Model hyperparameters (must match script)
EMB_DIM=256
HID_DIM=256
TCN_LAYERS=4
TCN_KERNEL=3
TCN_DROPOUT=0.2

# Environment
CONDA_ENV="shdb-af-analysis"   # change if needed
PYTHON_SCRIPT="train_mil_ecg_tcn_multiclass.py"

LOG_DIR="tmux_logs"
mkdir -p ${LOG_DIR}
# ----------------------------------------------------------

echo "Starting tmux session: ${SESSION_NAME}"

tmux new-session -d -s ${SESSION_NAME}

tmux send-keys -t ${SESSION_NAME} "
echo 'Activating environment'
source ~/.bashrc
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONUNBUFFERED=1

echo 'Running multiclass MIL TCN + Temporal Attention training'
python ${PYTHON_SCRIPT} \
  --val_fold ${VAL_FOLD} \
  --csv_path ${CSV_PATH} \
  --segments_dir ${SEGMENTS_DIR} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --seed ${SEED} \
  --emb_dim ${EMB_DIM} \
  --hid_dim ${HID_DIM} \
  --layers ${TCN_LAYERS} \
  --kernel ${TCN_KERNEL} \
  --dropout ${TCN_DROPOUT} \
  --device cuda \
  2>&1 | tee ${LOG_DIR}/${SESSION_NAME}.log

echo 'Training finished'
" C-m

echo "tmux session '${SESSION_NAME}' started"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
