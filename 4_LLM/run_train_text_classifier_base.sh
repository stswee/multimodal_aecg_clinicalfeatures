#!/bin/bash

# ===============================
# Configuration
# ===============================

SESSION_NAME="text_classifier"
ENV_NAME="shdb-af-analysis"

SCRIPT_PATH="train_text_classifier.py"

EMBED_DIR="../../music/lm_embeddings/LLaMA8B/BioBERT"
FOLD_CSV="../../music/music_patient_folds_5cv.csv"

LLM_MODEL_NAME="LLaMA8B"
LM_MODEL_NAME="BioBERT"

GPU_ID=0,1,2,3   # <-- change if needed

LOG_FILE="train_text_$(date +%Y%m%d_%H%M%S).log"

# ===============================
# Start tmux session
# ===============================

tmux kill-session -t $SESSION_NAME 2>/dev/null
tmux new-session -d -s $SESSION_NAME

tmux send-keys -t $SESSION_NAME "
source ~/miniconda3/etc/profile.d/conda.sh &&
conda activate $ENV_NAME &&

export CUDA_VISIBLE_DEVICES=$GPU_ID &&

echo 'Using GPU:' &&
nvidia-smi &&

echo 'Embedding directory:' &&
echo $EMBED_DIR &&

echo 'Starting 5-fold training...' &&

python $SCRIPT_PATH \
    --embedding_dir $EMBED_DIR \
    --fold_csv $FOLD_CSV \
    --llm_model_name $LLM_MODEL_NAME \
    --lm_model_name $LM_MODEL_NAME \
    > $LOG_FILE 2>&1
" C-m

echo "Session '$SESSION_NAME' started."
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Monitor logs with: tail -f $LOG_FILE"
