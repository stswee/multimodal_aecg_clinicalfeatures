#!/bin/bash

SESSION_NAME="embed_text"
ENV_NAME="shdb-af-analysis"   # change if needed
SCRIPT_PATH="embed_llm_risks.py"    # <-- update if filename differs

CSV_PATH="../../music/llm_responses/LLaMA3B_responses.csv"
PATIENT_ID_COL="Patient ID"
TEXT_COL="reasoning_text"

OUTPUT_ROOT="../../music/lm_embeddings"
LLM_MODEL_NAME="LLaMA3B"

BATCH_SIZE=32
MAX_LENGTH=256

LOG_FILE="embed_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new detached session
tmux new-session -d -s $SESSION_NAME

# Send commands to tmux session
tmux send-keys -t $SESSION_NAME "
source ~/miniconda3/etc/profile.d/conda.sh &&
conda activate $ENV_NAME &&
echo 'Using GPU:' &&
nvidia-smi &&
echo 'Starting embedding job...' &&
python $SCRIPT_PATH \
    --csv_path $CSV_PATH \
    --patient_id_col \"$PATIENT_ID_COL\" \
    --text_col \"$TEXT_COL\" \
    --output_root $OUTPUT_ROOT \
    --llm_model_name $LLM_MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --device cuda \
    > $LOG_FILE 2>&1
" C-m

echo "Session '$SESSION_NAME' started."
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Monitor logs with: tail -f $LOG_FILE"
