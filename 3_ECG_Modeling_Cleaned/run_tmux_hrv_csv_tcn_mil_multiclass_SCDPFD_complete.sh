#!/usr/bin/env bash
# ============================================================
# tmux launcher for FEATURE-based MIL + TCN (MULTIBRANCH)
# Runs ALL 5 folds sequentially in ONE tmux session
# ============================================================

# ---------------- USER CONFIG ----------------
SESSION_NAME="tcn_mil_features_multibranch_5fold_emb256_drop0_3"

GPU_ID=5

# CSV with Patient ID / label / fold
CSV_PATH="../../music/music_patient_folds_5cv.csv"

# Feature directory (each <pid>/<pid>_segment_features.csv)
FEATURES_DIR="../../music/preprocessed_segments_HRV_complete"

# Output root (script will create val_fold_0 ... val_fold_4)
OUTPUT_ROOT="../../music/results"
OUTPUT_DIR="${OUTPUT_ROOT}/${SESSION_NAME}"

# Training
EPOCHS=30
LR=1e-4
WEIGHT_DECAY=1e-5
SEED=42

# Feature encoder / embedding
EMBEDDING_DIM=256
ENC_HIDDEN=128
ENC_DROPOUT=0.1

# TCN
TCN_HIDDEN_DIM=128
TCN_LAYERS=4
TCN_KERNEL_SIZE=3
TCN_DROPOUT=0.2

# Attention MIL
ATTN_DIM=128

# Task-specific branches
BRANCH_HIDDEN=128
BRANCH_DROPOUT=0.2

# Losses
LOSS="bce"          # bce | focal
FOCAL_ALPHA=0.75
FOCAL_GAMMA=2.0
NORMALIZE_LOSSES="--normalize_losses"

# Data handling
MIN_SEGMENTS=3
SORT_BY="window_idx"     # or "start_idx"
DROP_NA_ROWS="--drop_na_rows"
# NO_ZSCORE="--no_zscore"   # uncomment to disable z-score normalization

# Environment
CONDA_ENV="shdb-af-analysis"
PYTHON_SCRIPT="train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py"

LOG_DIR="tmux_logs"
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}
# ---------------- END CONFIG ----------------

echo "Starting tmux session: ${SESSION_NAME}"

tmux new-session -d -s ${SESSION_NAME}

tmux send-keys -t ${SESSION_NAME} "
echo '========================================'
echo 'Activating environment'
echo '========================================'
source ~/.bashrc
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo 'GPU(s): ' \$CUDA_VISIBLE_DEVICES
echo '========================================'

for VAL_FOLD in 0 1 2 3 4
do
  echo ''
  echo '========================================'
  echo 'Starting fold:' \$VAL_FOLD
  echo 'Start time:' \$(date)
  echo '========================================'

  python ${PYTHON_SCRIPT} \
    --val_fold \$VAL_FOLD \
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
    --branch_hidden ${BRANCH_HIDDEN} \
    --branch_dropout ${BRANCH_DROPOUT} \
    --loss_type ${LOSS} \
    --focal_alpha ${FOCAL_ALPHA} \
    --focal_gamma ${FOCAL_GAMMA} \
    ${NORMALIZE_LOSSES} \
    --min_segments ${MIN_SEGMENTS} \
    --sort_by ${SORT_BY} \
    ${DROP_NA_ROWS} \
    --device cuda \
    2>&1 | tee ${LOG_DIR}/${SESSION_NAME}_fold\${VAL_FOLD}.log

  echo 'Finished fold:' \$VAL_FOLD
  echo 'End time:' \$(date)
  echo '========================================'
done

echo ''
echo '========================================'
echo 'ALL FOLDS COMPLETE'
echo 'Finished at:' \$(date)
echo '========================================'
" C-m

echo ""
echo "tmux session '${SESSION_NAME}' started."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Detach with: Ctrl+b then d"
