#!/usr/bin/env bash
# ============================================================
# tmux launcher for HRV / FEATURE-based TCN + Attention-MIL
# Uses per-window HRV / RR / PVC features (CSV-based)
# ============================================================

# ---------------- USER CONFIG ----------------
SESSION_NAME="tcn_attn_mil_features_csv_SCDPFD_fold2"
GPU_ID=4

VAL_FOLD=2

CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv"

# Feature directory (each <pid>/<pid>_segment_features.csv)
FEATURES_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments_HRV"

OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/tcn_attn_mil_feature_outputs_SCDPFD_csv"

# Training
EPOCHS=30
LR=1e-4
WEIGHT_DECAY=1e-5
SEED=42

# Feature encoder / embedding
EMBEDDING_DIM=128
ENC_HIDDEN=128
ENC_DROPOUT=0.1

# TCN
TCN_HIDDEN_DIM=128
TCN_LAYERS=4
TCN_KERNEL_SIZE=3
TCN_DROPOUT=0.2

# Attention MIL
ATTN_DIM=128

# Data handling
MIN_SEGMENTS=3
SORT_BY="window_idx"     # or "start_idx"
DROP_NA_ROWS="--drop_na_rows"
# NO_ZSCORE="--no_zscore"   # uncomment to disable z-score normalization

# Environment
CONDA_ENV="shdb-af-analysis"
PYTHON_SCRIPT="train_tcn_mil_hrv_csv_multiclass_SCDPFD.py"

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

echo 'Running FEATURE-based TCN + Attention MIL training'

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
  --enc_hidden ${ENC_HIDDEN} \
  --enc_dropout ${ENC_DROPOUT} \
  --tcn_hidden_dim ${TCN_HIDDEN_DIM} \
  --tcn_layers ${TCN_LAYERS} \
  --tcn_kernel_size ${TCN_KERNEL_SIZE} \
  --tcn_dropout ${TCN_DROPOUT} \
  --attn_dim ${ATTN_DIM} \
  --min_segments ${MIN_SEGMENTS} \
  --sort_by ${SORT_BY} \
  ${DROP_NA_ROWS} \
  --device cuda \
  2>&1 | tee ${LOG_DIR}/${SESSION_NAME}.log

echo 'Training finished.'
" C-m

echo "tmux session '${SESSION_NAME}' started."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
