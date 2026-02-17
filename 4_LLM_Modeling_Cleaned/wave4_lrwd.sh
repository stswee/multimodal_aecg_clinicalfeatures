#!/usr/bin/env bash
# ============================================================
# TEXT CLASSIFIER WAVE 4
# Optimization Sweep (lr x weight_decay)
# Fixed architecture:
#   layers=2
#   hidden=64
#   dropout=0.0
# ============================================================

EMBEDDING_ROOT="../../music/lm_embeddings"
FOLD_CSV="../../music/music_patient_folds_5cv.csv"

BASE_OUTPUT_DIR="../../music/text_wave4_results"
EPOCHS=100

LLM="LLaMA8B"
LM="BioBERT"

# Fixed architecture
LAYERS=2
HIDDEN=64
DROPOUT=0.0

# Sweep grid
LRS=(5e-3 1e-3 5e-4 1e-4)
WDS=(0 1e-6 1e-5 1e-4 1e-3)

# GPUs available
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p "${BASE_OUTPUT_DIR}"

JOB_ID=0

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do

    # Make tmux-safe strings
    LR_SAFE=$(echo "${LR}" | sed 's/-/m/g' | sed 's/\./p/g')
    WD_SAFE=$(echo "${WD}" | sed 's/-/m/g' | sed 's/\./p/g')

    GPU_INDEX=$(( JOB_ID % (${#GPUS[@]} * 2) ))
    GPU=${GPUS[$(( GPU_INDEX / 2 ))]}

    SESSION_NAME="text_w4_lr${LR_SAFE}_wd${WD_SAFE}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/lr${LR_SAFE}_wd${WD_SAFE}"

    CMD="python train_text_classifier.py \
      --embedding_dir ${EMBEDDING_ROOT} \
      --fold_csv ${FOLD_CSV} \
      --llm_model_name ${LLM} \
      --lm_model_name ${LM} \
      --output_dir ${OUTPUT_DIR} \
      --epochs ${EPOCHS} \
      --lr ${LR} \
      --weight_decay ${WD} \
      --classifier mlp \
      --mlp_layers ${LAYERS} \
      --mlp_hidden ${HIDDEN} \
      --mlp_dropout ${DROPOUT} \
      --seed 42"

    echo "Launching ${SESSION_NAME} on GPU ${GPU}"

    tmux new-session -d -s "${SESSION_NAME}"
    tmux send-keys -t "${SESSION_NAME}" "
      export CUDA_VISIBLE_DEVICES=${GPU}
      mkdir -p ${OUTPUT_DIR}
      ${CMD}
    " C-m

    JOB_ID=$((JOB_ID + 1))

  done
done

echo "=========================================="
echo "Wave 4 (Optimization Sweep) launched."
echo "2 processes per GPU enabled."
echo "Use: tmux ls"
echo "Outputs: ${BASE_OUTPUT_DIR}"
echo "=========================================="
