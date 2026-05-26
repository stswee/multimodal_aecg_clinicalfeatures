#!/usr/bin/env bash
# ============================================================
# MULTIMODAL FUSION PARAMETER SWEEP (Sequential, 1 GPU)
# ============================================================

set -e  # stop on error

CUDA_VISIBLE_DEVICES=0
ECG_EMB_DIR="../../music/best_ecg_embeddings"
TEXT_EMB_DIR="../../music/best_text_embeddings_LLaMA8B_BioBERT"

BASE_OUTPUT_DIR="../../music/multimodal_wave1_capacity"

EPOCHS=100
LR=1e-3
WEIGHT_DECAY=1e-5
DROPOUT=0.2

# ------------------------------------------------------------

mkdir -p "${BASE_OUTPUT_DIR}"

HIDDEN_DIMS=(128 256)
PROJ_DIMS=(128 256)
LAYERS=(1 2)

for H in "${HIDDEN_DIMS[@]}"
do
  for L in "${LAYERS[@]}"
  do
    OUT_DIR="${BASE_OUTPUT_DIR}/concat_hid${H}_L${L}"
    mkdir -p "${OUT_DIR}"

    echo "===================================================="
    echo "DIRECT CONCAT | hidden=${H} | layers=${L}"
    echo "===================================================="

    for FOLD in 0 1 2 3 4
    do
      python train_multimodal_concat.py \
        --val_fold ${FOLD} \
        --ecg_embedding_dir ${ECG_EMB_DIR} \
        --text_embedding_dir ${TEXT_EMB_DIR} \
        --output_dir ${OUT_DIR} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --hidden_dim ${H} \
        --dropout ${DROPOUT} \
        --num_layers ${L} \
        --seed 42
    done
  done
done

METHOD_NAMES=(
  projected_concat
  scalar_gating
  vector_gating
  weighted_sum
)

SCRIPTS=(
  train_multimodal_projectconcat.py
  train_multimodal_scalar_gating.py
  train_multimodal_vector_gating.py
  train_multimodal_weighted_sum.py
)

for i in "${!METHOD_NAMES[@]}"
do
  METHOD_NAME="${METHOD_NAMES[$i]}"
  SCRIPT="${SCRIPTS[$i]}"

  for P in "${PROJ_DIMS[@]}"
  do
    for H in "${HIDDEN_DIMS[@]}"
    do
      for L in "${LAYERS[@]}"
      do
        OUT_DIR="${BASE_OUTPUT_DIR}/${METHOD_NAME}_proj${P}_hid${H}_L${L}"
        mkdir -p "${OUT_DIR}"

        echo "===================================================="
        echo "${METHOD_NAME} | proj=${P} | hidden=${H} | layers=${L}"
        echo "===================================================="

        for FOLD in 0 1 2 3 4
        do
          python ${SCRIPT} \
            --val_fold ${FOLD} \
            --ecg_embedding_dir ${ECG_EMB_DIR} \
            --text_embedding_dir ${TEXT_EMB_DIR} \
            --output_dir ${OUT_DIR} \
            --epochs ${EPOCHS} \
            --lr ${LR} \
            --weight_decay ${WEIGHT_DECAY} \
            --proj_dim ${P} \
            --hidden_dim ${H} \
            --dropout ${DROPOUT} \
            --num_layers ${L} \
            --seed 42
        done

      done
    done
  done
done

echo "===================================================="
echo "ALL MULTIMODAL SWEEPS COMPLETE"
echo "===================================================="
