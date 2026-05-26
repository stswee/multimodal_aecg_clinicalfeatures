#!/usr/bin/env bash
# ============================================================
# MULTIMODAL WAVE 2
# Sequential, 1 GPU
# ============================================================

set -e
export CUDA_VISIBLE_DEVICES=5

ECG_EMB_DIR="../../music/best_ecg_embeddings"
TEXT_EMB_DIR="../../music/best_text_embeddings_LLaMA8B_BioBERT"

BASE_OUTPUT_DIR="../../music/multimodal_wave2_dropout"

EPOCHS=100
LR=1e-3
WEIGHT_DECAY=1e-5

# Dropout grid for Wave 2
DROPOUTS=(0.0 0.2 0.4)

mkdir -p "${BASE_OUTPUT_DIR}"

FOLDS=(0 1 2 3 4)

DC_CONFIGS=(
  "128 1"
  "256 2"
)

for CFG in "${DC_CONFIGS[@]}"
do
  read H L <<< "${CFG}"

  for D in "${DROPOUTS[@]}"
  do
    OUT_DIR="${BASE_OUTPUT_DIR}/DC_hid${H}_L${L}_drop${D}"
    mkdir -p "${OUT_DIR}"

    echo "===================================================="
    echo "DC | hidden=${H} | layers=${L} | dropout=${D}"
    echo "===================================================="

    for FOLD in "${FOLDS[@]}"
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
        --dropout ${D} \
        --num_layers ${L} \
        --seed 42
    done
  done
done

run_projected_method () {
  METHOD_NAME=$1
  SCRIPT=$2
  PROJ=$3
  HIDDEN=$4
  LAYERS=$5

  for D in "${DROPOUTS[@]}"
  do
    OUT_DIR="${BASE_OUTPUT_DIR}/${METHOD_NAME}_proj${PROJ}_hid${HIDDEN}_L${LAYERS}_drop${D}"
    mkdir -p "${OUT_DIR}"

    echo "===================================================="
    echo "${METHOD_NAME} | proj=${PROJ} | hidden=${HIDDEN} | layers=${LAYERS} | dropout=${D}"
    echo "===================================================="

    for FOLD in "${FOLDS[@]}"
    do
      python ${SCRIPT} \
        --val_fold ${FOLD} \
        --ecg_embedding_dir ${ECG_EMB_DIR} \
        --text_embedding_dir ${TEXT_EMB_DIR} \
        --output_dir ${OUT_DIR} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --proj_dim ${PROJ} \
        --hidden_dim ${HIDDEN} \
        --dropout ${D} \
        --num_layers ${LAYERS} \
        --seed 42
    done
  done
}

# ---- Projected Concat ----
run_projected_method projected_concat train_multimodal_projectconcat.py 256 128 2
run_projected_method projected_concat train_multimodal_projectconcat.py 256 256 2

# ---- Scalar Gating ----
run_projected_method scalar_gating train_multimodal_scalar_gating.py 256 128 2
run_projected_method scalar_gating train_multimodal_scalar_gating.py 256 256 1

# ---- Vector Gating ----
run_projected_method vector_gating train_multimodal_vector_gating.py 128 128 1
run_projected_method vector_gating train_multimodal_vector_gating.py 256 256 2

# ---- Weighted Sum ----
run_projected_method weighted_sum train_multimodal_weighted_sum.py 256 128 2
run_projected_method weighted_sum train_multimodal_weighted_sum.py 256 256 1

echo "===================================================="
echo "WAVE 2 DROPOUT SWEEP COMPLETE"
echo "===================================================="
