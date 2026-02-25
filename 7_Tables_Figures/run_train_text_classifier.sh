#!/usr/bin/env bash
# ============================================================
# TEXT CLASSIFIER (SHARED EMBEDDING EXTRACTION)
# No parameter sweep
# Architecture:
#   layers=2
#   hidden=64
#   dropout=0.0
# ============================================================

# ---------------- USER CONFIG ----------------

EMBEDDING_ROOT="../../music/lm_embeddings"
FOLD_CSV="../../music/music_patient_folds_5cv.csv"

OUTPUT_DIR="../../music/best_results/text_embeddings_LLaMA8B_BioBERT"
TEXT_EMB_DIR="../../music/best_text_embeddings_LLaMA8B_BioBERT"

EPOCHS=100
LR=5e-3
WEIGHT_DECAY=1e-4

LLM="LLaMA8B"
LM="BioBERT"


LAYERS=2
HIDDEN=64
DROPOUT=0

GPU=5
SESSION_NAME="text_shared_embeddings"

# ------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TEXT_EMB_DIR}"

CMD="python train_text_classifier.py \
  --embedding_dir ${EMBEDDING_ROOT} \
  --fold_csv ${FOLD_CSV} \
  --llm_model_name ${LLM} \
  --lm_model_name ${LM} \
  --output_dir ${OUTPUT_DIR} \
  --text_embedding_dir ${TEXT_EMB_DIR} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --classifier mlp \
  --mlp_layers ${LAYERS} \
  --mlp_hidden ${HIDDEN} \
  --mlp_dropout ${DROPOUT} \
  --seed 42"

echo "Launching ${SESSION_NAME} on GPU ${GPU}"

tmux new-session -d -s "${SESSION_NAME}"
tmux send-keys -t "${SESSION_NAME}" "
  export CUDA_VISIBLE_DEVICES=${GPU}
  ${CMD}
" C-m

echo "=========================================="
echo "Text training + shared embedding extraction launched."
echo "Session: ${SESSION_NAME}"
echo "GPU: ${GPU}"
echo "Use: tmux attach -t ${SESSION_NAME}"
echo "Embeddings will be saved to:"
echo "${TEXT_EMB_DIR}"
echo "=========================================="
