#!/usr/bin/env bash
# ============================================================
# TEXT CLASSIFIER WAVE 3
# Dropout Sweep (Fixed tmux-safe session names)
# ============================================================

EMBEDDING_ROOT="../../music/lm_embeddings"
FOLD_CSV="../../music/music_patient_folds_5cv.csv"

BASE_OUTPUT_DIR="../../music/text_wave3_results"
EPOCHS=100
LR=1e-3

LLM="LLaMA8B"
LM="BioBERT"

GPUS=(0 1 2 3 4 5 6 7)

ARCHS=(
  "2|64"
  "3|64"
)

DROPOUTS=(0.0 0.1 0.2 0.3 0.4 0.5)

mkdir -p "${BASE_OUTPUT_DIR}"

JOB_ID=0

for ARCH in "${ARCHS[@]}"; do
  IFS='|' read -r L H <<< "${ARCH}"

  for D in "${DROPOUTS[@]}"; do

    # Make tmux-safe dropout string
    D_SAFE=$(echo "${D}" | sed 's/\./p/')

    GPU_INDEX=$(( JOB_ID % (${#GPUS[@]} * 2) ))
    GPU=${GPUS[$(( GPU_INDEX / 2 ))]}

    SESSION_NAME="text_w3_L${L}_H${H}_D${D_SAFE}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/L${L}_H${H}_D${D_SAFE}"

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
      --mlp_dropout ${D} \
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
echo "Wave 3 launched (tmux-safe names)."
echo "Use: tmux ls"
echo "=========================================="
