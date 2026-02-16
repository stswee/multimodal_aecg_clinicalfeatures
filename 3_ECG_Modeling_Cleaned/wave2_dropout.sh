#!/usr/bin/env bash
# ============================================================
# Wave 2: Dropout Factorial Sweep (8 configs, 8 GPUs)
# Architecture: emb128 / hid256 / 3 TCN layers
# ============================================================

CSV_PATH="../../music/music_patient_folds_5cv.csv"
FEATURES_DIR="../../music/preprocessed_segments_HRV_complete"
OUTPUT_ROOT="../../music/results"

CONDA_ENV="shdb-af-analysis"
PYTHON_SCRIPT="train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py"

EPOCHS=30
LR=1e-4
WEIGHT_DECAY=1e-5
SEED=42

LOG_DIR="tmux_logs"
mkdir -p ${LOG_DIR}

# ============================================================
# ----------- ARCHITECTURE (4 LAYERS) ------------------------
# ============================================================

EMB_DIM=128
TCN_HID=256
TCN_LAY=4

ARCH_NAME="emb128_hid256_l${TCN_LAY}"

# ============================================================
# 2×2×2 Dropout Grid
# enc ∈ {0.0, 0.2}
# tcn ∈ {0.1, 0.3}
# branch ∈ {0.0, 0.2}
# ============================================================

CONFIGS=(
"enc0p0_tcn0p1_br0p0"
"enc0p0_tcn0p1_br0p2"
"enc0p0_tcn0p3_br0p0"
"enc0p0_tcn0p3_br0p2"
"enc0p2_tcn0p1_br0p0"
"enc0p2_tcn0p1_br0p2"
"enc0p2_tcn0p3_br0p0"
"enc0p2_tcn0p3_br0p2"
)

ENC=(0.0 0.0 0.0 0.0 0.2 0.2 0.2 0.2)
TCN=(0.1 0.1 0.3 0.3 0.1 0.1 0.3 0.3)
BRN=(0.0 0.2 0.0 0.2 0.0 0.2 0.0 0.2)

# ============================================================
# LAUNCH
# ============================================================

for i in {0..7}
do
  SESSION_NAME="wave2_${ARCH_NAME}_${CONFIGS[$i]}"
  GPU_ID=$i

  ENC_DROPOUT=${ENC[$i]}
  TCN_DROPOUT=${TCN[$i]}
  BRANCH_DROPOUT=${BRN[$i]}

  OUTPUT_DIR="${OUTPUT_ROOT}/${SESSION_NAME}"

  echo "Launching ${SESSION_NAME} on GPU ${GPU_ID}"

  tmux new-session -d -s "${SESSION_NAME}"

  tmux send-keys -t "${SESSION_NAME}" "
  source ~/.bashrc
  conda activate ${CONDA_ENV}

  export CUDA_VISIBLE_DEVICES=${GPU_ID}
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  mkdir -p ${OUTPUT_DIR}

  for VAL_FOLD in 0 1 2 3 4
  do
    python ${PYTHON_SCRIPT} \
      --val_fold \$VAL_FOLD \
      --features_dir ${FEATURES_DIR} \
      --csv_path ${CSV_PATH} \
      --output_dir ${OUTPUT_DIR} \
      --epochs ${EPOCHS} \
      --lr ${LR} \
      --weight_decay ${WEIGHT_DECAY} \
      --seed ${SEED} \
      --embedding_dim ${EMB_DIM} \
      --enc_hidden 128 \
      --enc_dropout ${ENC_DROPOUT} \
      --tcn_hidden_dim ${TCN_HID} \
      --tcn_layers ${TCN_LAY} \
      --tcn_kernel_size 3 \
      --tcn_dropout ${TCN_DROPOUT} \
      --attn_dim 128 \
      --branch_hidden ${TCN_HID} \
      --branch_dropout ${BRANCH_DROPOUT} \
      --loss_type bce \
      --min_segments 3 \
      --sort_by window_idx \
      --drop_na_rows \
      --device cuda \
      2>&1 | tee ${LOG_DIR}/${SESSION_NAME}_fold\${VAL_FOLD}.log
  done
  " C-m

done

echo "============================================"
echo "Wave 2 launched for ${ARCH_NAME}"
echo "Check with: tmux ls"
echo "============================================"
