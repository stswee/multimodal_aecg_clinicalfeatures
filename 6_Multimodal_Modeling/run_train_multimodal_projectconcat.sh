#!/usr/bin/env bash
# ============================================================
# MULTIMODAL PROJECTED CONCATENATION (ECG + TEXT)
# No parameter sweep
#
# Projection:
#   h_ECG  = W_e * z_ECG   -> 128
#   h_Text = W_t * z_Text  -> 128
#
# Fusion:
#   z = [h_ECG ; h_Text]   -> 256
#
# Classifier:
#   hidden_dim=128
#   dropout=0.2
# ============================================================

# ---------------- USER CONFIG ----------------

ECG_EMB_DIR="../../music/ecg_embeddings"
TEXT_EMB_DIR="../../music/text_shared_embeddings"

OUTPUT_DIR="../../music/multimodal_projected_concat_results"

EPOCHS=100
LR=1e-3
WEIGHT_DECAY=1e-5

PROJ_DIM=128
HIDDEN_DIM=128
DROPOUT=0.2

GPU=3
SESSION_NAME="multimodal_projected_concat"

# ------------------------------------------------

mkdir -p "${OUTPUT_DIR}"

echo "Launching ${SESSION_NAME} on GPU ${GPU}"

tmux new-session -d -s "${SESSION_NAME}"

tmux send-keys -t "${SESSION_NAME}" "
export CUDA_VISIBLE_DEVICES=${GPU}

for FOLD in 0 1 2 3 4
do
  echo '=========================================='
  echo \"Running fold \${FOLD}\"
  echo '=========================================='

  python train_multimodal_projectconcat.py \
    --val_fold \${FOLD} \
    --ecg_embedding_dir ${ECG_EMB_DIR} \
    --text_embedding_dir ${TEXT_EMB_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --proj_dim ${PROJ_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --dropout ${DROPOUT} \
    --seed 42

done
" C-m

echo "=========================================="
echo "Multimodal projected concatenation launched."
echo "Session: ${SESSION_NAME}"
echo "GPU: ${GPU}"
echo "Use: tmux attach -t ${SESSION_NAME}"
echo "Results will be saved to:"
echo "${OUTPUT_DIR}"
echo "=========================================="
