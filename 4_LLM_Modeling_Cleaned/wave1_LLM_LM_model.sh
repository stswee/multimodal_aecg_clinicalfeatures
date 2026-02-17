#!/usr/bin/env bash
# ============================================================
# TEXT CLASSIFIER WAVE 1 SWEEP
# LLM x LM x MODEL
# 8 experiments (one per GPU)
# ============================================================

# ---------------- USER CONFIG ----------------

# IMPORTANT: this is the ROOT that contains:
#   lm_embeddings/LLaMA8B/BioBERT/*.npy
EMBEDDING_ROOT="../../music/lm_embeddings"

FOLD_CSV="../../music/music_patient_folds_5cv.csv"

BASE_OUTPUT_DIR="../../music/text_wave1_results"
EPOCHS=100
LR=1e-3

# GPUs available
GPUS=(0 1 2 3 4 5 6 7)

# LLM options (must exactly match embedding folder names)
LLMS=("LLaMA3B" "LLaMA8B")

# LM options (must exactly match embedding folder names)
LMS=("BioBERT" "ClinicalBERT")

# Classifiers
MODELS=("linear" "mlp")

# ============================================================
# Generate runs
# ============================================================

RUN_ID=0

for LLM in "${LLMS[@]}"; do
  for LM in "${LMS[@]}"; do
    for MODEL in "${MODELS[@]}"; do

      GPU=${GPUS[$RUN_ID]}

      SESSION_NAME="text_w1_${LLM}_${LM}_${MODEL}"
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/${LLM}_${LM}_${MODEL}"

      echo "Launching ${SESSION_NAME} on GPU ${GPU}"

      tmux new-session -d -s ${SESSION_NAME}

      tmux send-keys -t ${SESSION_NAME} "
        export CUDA_VISIBLE_DEVICES=${GPU}
        mkdir -p ${OUTPUT_DIR}
        python train_text_classifier.py \
          --embedding_dir ${EMBEDDING_ROOT} \
          --fold_csv ${FOLD_CSV} \
          --llm_model_name ${LLM} \
          --lm_model_name ${LM} \
          --output_dir ${OUTPUT_DIR} \
          --epochs ${EPOCHS} \
          --lr ${LR} \
          --classifier ${MODEL}
      " C-m

      RUN_ID=$((RUN_ID + 1))

    done
  done
done

echo "=========================================="
echo "Wave 1 sweep launched."
echo "Use: tmux ls"
echo "=========================================="
