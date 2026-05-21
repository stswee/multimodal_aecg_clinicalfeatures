#!/usr/bin/env bash
# ============================================================
# Wave 1: Capacity Sweep
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

ENC_DROPOUT=0.2
TCN_DROPOUT=0.1
BRANCH_DROPOUT=0

LOG_DIR="tmux_logs"
mkdir -p ${LOG_DIR}

CONFIGS=(
"emb128_hid128_l3"
"emb128_hid128_l4"
"emb128_hid256_l3"
"emb128_hid256_l4"
"emb256_hid128_l3"
"emb256_hid128_l4"
"emb256_hid256_l3"
"emb256_hid256_l4"
)

EMB=(128 128 128 128 256 256 256 256)
HID=(128 128 256 256 128 128 256 256)
LAY=(3   4   3   4   3   4   3   4)

for i in {0..7}
do
  SESSION_NAME="wave1_${CONFIGS[$i]}"
  GPU_ID=$i

  EMB_DIM=${EMB[$i]}
  TCN_HID=${HID[$i]}
  TCN_LAY=${LAY[$i]}

  OUTPUT_DIR="${OUTPUT_ROOT}/${SESSION_NAME}"

  echo "Launching ${SESSION_NAME} on GPU ${GPU_ID}"

  tmux new-session -d -s ${SESSION_NAME}

  tmux send-keys -t ${SESSION_NAME} "
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
echo "Wave 1 launched (8 sessions)"
echo "Check with: tmux ls"
echo "============================================"
