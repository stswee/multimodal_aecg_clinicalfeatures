#!/usr/bin/env bash
# ============================================================
# TEXT CLASSIFIER WAVE 2
# Capacity Sweep (Layers x Hidden)
# LLM=LLaMA8B, LM=BioBERT
# 2 processes per GPU allowed
# ============================================================

# ---------------- USER CONFIG ----------------

EMBEDDING_ROOT="../../music/lm_embeddings"
FOLD_CSV="../../music/music_patient_folds_5cv.csv"

BASE_OUTPUT_DIR="../../music/text_wave2_results"
EPOCHS=100
LR=1e-3
DROPOUT=0.2
WD=0

LLM="LLaMA8B"
LM="BioBERT"

# GPUs available
GPUS=(0 1 2 3 4 5 6 7)

# MLP capacity grid
LAYERS=(1 2 3)
HIDDENS=(64 128 256 512)

# ============================================================
# Build job list
# ============================================================

JOBS=()

# ---- Linear baseline (layers=0) ----
JOBS+=("linear|0|0")

# ---- MLP configs ----
for L in "${LAYERS[@]}"; do
  for H in "${HIDDENS[@]}"; do
    JOBS+=("mlp|${L}|${H}")
  done
done

echo "Total Wave 2 jobs: ${#JOBS[@]}"

mkdir -p "${BASE_OUTPUT_DIR}"

# ============================================================
# Launch jobs
# 2 processes per GPU
# ============================================================

JOB_ID=0

for JOB in "${JOBS[@]}"; do

  IFS='|' read -r CLS L H <<< "${JOB}"

  # GPU index allows 2 jobs per GPU
  GPU_INDEX=$(( JOB_ID % (${#GPUS[@]} * 2) ))
  GPU=${GPUS[$(( GPU_INDEX / 2 ))]}

  if [[ "${CLS}" == "linear" ]]; then
    SESSION_NAME="text_w2_linear"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/linear"

    CMD="python train_text_classifier.py \
      --embedding_dir ${EMBEDDING_ROOT} \
      --fold_csv ${FOLD_CSV} \
      --llm_model_name ${LLM} \
      --lm_model_name ${LM} \
      --output_dir ${OUTPUT_DIR} \
      --epochs ${EPOCHS} \
      --lr ${LR} \
      --classifier linear \
      --seed 42"

  else
    SESSION_NAME="text_w2_L${L}_H${H}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/L${L}_H${H}"

    CMD="python train_text_classifier.py \
      --embedding_dir ${EMBEDDING_ROOT} \
      --fold_csv ${FOLD_CSV} \
      --llm_model_name ${LLM} \
      --lm_model_name ${LM} \
      --output_dir ${OUTPUT_DIR} \
      --epochs ${EPOCHS} \
      --lr ${LR} \
      --classifier mlp \
      --mlp_layers ${L} \
      --mlp_hidden ${H} \
      --mlp_dropout ${DROPOUT} \
      --seed 42"
  fi

  echo "Launching ${SESSION_NAME} on GPU ${GPU}"

  tmux new-session -d -s "${SESSION_NAME}"

  tmux send-keys -t "${SESSION_NAME}" "
    export CUDA_VISIBLE_DEVICES=${GPU}
    mkdir -p ${OUTPUT_DIR}
    ${CMD}
  " C-m

  JOB_ID=$((JOB_ID + 1))

done

echo "=========================================="
echo "Wave 2 (Capacity) launched."
echo "2 processes per GPU enabled."
echo "Use: tmux ls"
echo "Outputs: ${BASE_OUTPUT_DIR}"
echo "=========================================="
