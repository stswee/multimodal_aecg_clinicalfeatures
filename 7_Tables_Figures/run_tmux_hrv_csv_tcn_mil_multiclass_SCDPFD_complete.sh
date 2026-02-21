#!/usr/bin/env bash
# ============================================================
# Launch 5 folds in PARALLEL
# Each fold gets its own tmux session + GPU
# ============================================================

BASE_NAME="tcn_ecg_embeddings"

# -------- GPU ASSIGNMENT ----------
GPUS=(0 1 2 3 4)

# -------- Data ----------
CSV_PATH="../../music/music_patient_folds_5cv.csv"
FEATURES_DIR="../../music/preprocessed_segments_HRV_complete"

# -------- Output ----------
OUTPUT_ROOT="../../music/best_results"
OUTPUT_DIR="${OUTPUT_ROOT}/${BASE_NAME}"

# -------- ECG Embeddings ----------
ECG_EMBED_DIR="../../music/best_ecg_embeddings"

# -------- Training ----------
EPOCHS=30
LR=1e-4
WEIGHT_DECAY=1e-5
SEED=42

# -------- Model ----------
EMBEDDING_DIM=256
ENC_HIDDEN=128
ENC_DROPOUT=0

TCN_HIDDEN_DIM=128
TCN_LAYERS=4
TCN_KERNEL_SIZE=3
TCN_DROPOUT=0.3

ATTN_DIM=128
BRANCH_HIDDEN=128
BRANCH_DROPOUT=0

LOSS="bce"
FOCAL_ALPHA=0.75
FOCAL_GAMMA=2.0

MIN_SEGMENTS=3
SORT_BY="window_idx"
DROP_NA_ROWS="--drop_na_rows"

CONDA_ENV="shdb-af-analysis"
PYTHON_SCRIPT="train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py"

LOG_DIR="tmux_logs"
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${ECG_EMBED_DIR}

# ============================================================
# Launch folds
# ============================================================

for VAL_FOLD in 0 1 2 3 4
do
    GPU_ID=${GPUS[$VAL_FOLD]}
    SESSION_NAME="${BASE_NAME}_fold_${VAL_FOLD}"

    echo "----------------------------------------"
    echo "Starting session: ${SESSION_NAME}"
    echo "GPU: ${GPU_ID}"
    echo "----------------------------------------"

    # Kill session if it already exists (optional but safe)
    tmux kill-session -t ${SESSION_NAME} 2>/dev/null

    tmux new-session -d -s ${SESSION_NAME}

    tmux send-keys -t ${SESSION_NAME} "
    echo '========================================'
    echo 'Fold ${VAL_FOLD} running on GPU ${GPU_ID}'
    echo 'Start time:' \$(date)
    echo '========================================'

    source ~/.bashrc
    conda activate ${CONDA_ENV}

    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4

    echo 'Using GPU:' \$CUDA_VISIBLE_DEVICES

    python ${PYTHON_SCRIPT} \
        --val_fold ${VAL_FOLD} \
        --features_dir ${FEATURES_DIR} \
        --csv_path ${CSV_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --ecg_embedding_dir ${ECG_EMBED_DIR} \
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
        --min_segments ${MIN_SEGMENTS} \
        --sort_by ${SORT_BY} \
        ${DROP_NA_ROWS} \
        --device cuda \
        2>&1 | tee ${LOG_DIR}/${SESSION_NAME}.log

    echo '========================================'
    echo 'Finished fold ${VAL_FOLD}'
    echo 'End time:' \$(date)
    echo '========================================'
    " C-m

done

echo ""
echo "========================================"
echo "All fold sessions launched."
echo "Attach to a fold with:"
echo "tmux attach -t ${BASE_NAME}_fold_0"
echo "========================================"
