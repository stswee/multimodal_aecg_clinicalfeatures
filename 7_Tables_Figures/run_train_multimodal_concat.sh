#!/usr/bin/env bash
# ============================================================
# MULTIMODAL DIRECT CONCATENATION (ECG + TEXT)
# No parameter sweep
#
# Fusion:
#   z = [z_ECG ; z_Text]
#
# Classifier:
#   hidden_dim=128
#   dropout=0.2
# ============================================================

# ---------------- USER CONFIG ----------------

ECG_EMB_DIR="../../music/best_ecg_embeddings"
TEXT_EMB_DIR="../../music/best_text_embeddings_LLaMA8B_BioBERT"

OUTPUT_DIR="../../music/best_results/directconcat_embeddings_LLaMA8B_BioBERT"

EPOCHS=100
LR=1e-3
WEIGHT_DECAY=1e-5

HIDDEN_DIM=256
DROPOUT=0.2
LAYERS=2


GPU=5
SESSION_NAME="multimodal_concat"

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

  python train_multimodal_concat.py \
    --val_fold \${FOLD} \
    --ecg_embedding_dir ${ECG_EMB_DIR} \
    --text_embedding_dir ${TEXT_EMB_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --hidden_dim ${HIDDEN_DIM} \
    --dropout ${DROPOUT} \
    --num_layers ${LAYERS} \
    --seed 42

done
" C-m

echo "=========================================="
echo "Multimodal direct concatenation launched."
echo "Session: ${SESSION_NAME}"
echo "GPU: ${GPU}"
echo "Use: tmux attach -t ${SESSION_NAME}"
echo "Results will be saved to:"
echo "${OUTPUT_DIR}"
echo "=========================================="
