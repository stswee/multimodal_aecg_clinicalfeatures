#!/usr/bin/env bash
# ============================================================
# MULTIMODAL WAVE 3 (LR × WD SWEEP)
# Sequential, 1 GPU
# ============================================================

set -e
export CUDA_VISIBLE_DEVICES=5

ECG_EMB_DIR="../../music/best_ecg_embeddings"
TEXT_EMB_DIR="../../music/best_text_embeddings_LLaMA8B_BioBERT"

BASE_OUTPUT_DIR="../../music/multimodal_wave3_lr_wd"

EPOCHS=100

# Wave 3 grid
LRS=(1e-3 3e-4 1e-4)
WDS=(0 1e-5)

mkdir -p "${BASE_OUTPUT_DIR}"

FOLDS=(0 1 2 3 4)

for LR in "${LRS[@]}"
do
  for WD in "${WDS[@]}"
  do
    OUT_DIR="${BASE_OUTPUT_DIR}/DC_hid256_L2_drop0_lr${LR}_wd${WD}"
    mkdir -p "${OUT_DIR}"

    echo "===================================================="
    echo "DC | lr=${LR} | wd=${WD}"
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
        --weight_decay ${WD} \
        --hidden_dim 256 \
        --dropout 0.2 \
        --num_layers 2 \
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
  DROPOUT=$6

  for LR in "${LRS[@]}"
  do
    for WD in "${WDS[@]}"
    do
      OUT_DIR="${BASE_OUTPUT_DIR}/${METHOD_NAME}_proj${PROJ}_hid${HIDDEN}_L${LAYERS}_drop${DROPOUT}_lr${LR}_wd${WD}"
      mkdir -p "${OUT_DIR}"

      echo "===================================================="
      echo "${METHOD_NAME} | lr=${LR} | wd=${WD}"
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
          --weight_decay ${WD} \
          --proj_dim ${PROJ} \
          --hidden_dim ${HIDDEN} \
          --dropout ${DROPOUT} \
          --num_layers ${LAYERS} \
          --seed 42
      done
    done
  done
}

# ---- Projected Concat ----
# Best: proj=256, hidden=256, layers=1, dropout=0.2
run_projected_method projected_concat train_multimodal_projectconcat.py 256 128 2 0.4

# ---- Scalar Gating ----
# Best: proj=128, hidden=128, layers=1, dropout=0.2
run_projected_method scalar_gating train_multimodal_scalar_gating.py 256 128 2 0

# ---- Vector Gating ----
# Best: proj=128, hidden=128, layers=2, dropout=0.0
run_projected_method vector_gating train_multimodal_vector_gating.py 128 128 1 0.4

# ---- Weighted Sum ----
# Best: proj=128, hidden=128, layers=1, dropout=0.2
run_projected_method weighted_sum train_multimodal_weighted_sum.py 256 128 2 0


echo "===================================================="
echo "WAVE 3 LR × WD SWEEP COMPLETE"
echo "===================================================="
