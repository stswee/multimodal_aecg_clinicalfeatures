#!/usr/bin/env bash
# ============================================================
# TABULAR CLASSIFIER LAUNCHER
# Runs linear and MLP tabular models in separate tmux sessions
# ============================================================

TABULAR_CSV="../../music/subject-info.csv"
FOLD_CSV="../../music/music_patient_folds_5cv.csv"
BASE_OUTPUT_DIR="../../music/tabular_results"

EPOCHS=100
LR=1e-3
WEIGHT_DECAY=1e-5

CONDA_ENV="shdb-af-analysis"
GPUS=(0 1)
MODELS=("linear" "mlp")

for idx in "${!MODELS[@]}"; do
  MODEL="${MODELS[$idx]}"
  GPU="${GPUS[$idx]}"

  SESSION_NAME="tabular_${MODEL}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL}"
  LOG_FILE="${OUTPUT_DIR}/train.log"

  echo "Launching ${SESSION_NAME} on GPU ${GPU}"

  tmux kill-session -t "${SESSION_NAME}" 2>/dev/null
  tmux new-session -d -s "${SESSION_NAME}"

  tmux send-keys -t "${SESSION_NAME}" "
    source ~/miniconda3/etc/profile.d/conda.sh &&
    conda activate ${CONDA_ENV} &&
    export CUDA_VISIBLE_DEVICES=${GPU} &&
    mkdir -p ${OUTPUT_DIR} &&
    python train_tabular_classifier.py \
      --tabular_csv ${TABULAR_CSV} \
      --fold_csv ${FOLD_CSV} \
      --output_dir ${OUTPUT_DIR} \
      --epochs ${EPOCHS} \
      --lr ${LR} \
      --weight_decay ${WEIGHT_DECAY} \
      --classifier ${MODEL} \
      > ${LOG_FILE} 2>&1
  " C-m
done

echo "=========================================="
echo "Tabular classifier runs launched."
echo "Sessions: ${MODELS[*]}"
echo "Use: tmux ls"
echo "Logs:"
echo "  ${BASE_OUTPUT_DIR}/linear/train.log"
echo "  ${BASE_OUTPUT_DIR}/mlp/train.log"
echo "=========================================="
