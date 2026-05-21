#!/usr/bin/env bash
# ============================================================
# tmux launcher for LLaMA-based SCD / PFD risk generation
# Sequential execution: LLaMA 8B -> LLaMA 3B
# ============================================================

# ---------------- USER CONFIG ----------------
SESSION_NAME="llama_scd_pfd_pipeline"
GPU_ID=4,5,6,7

# Python environment
CONDA_ENV="shdb-af-analysis"

# Script
SCRIPT_PATH="generate_llm_risks.py"

# Input CSV
CSV_PATH="../../music/subject-info-cleaned-with-prompts.csv"

# Outputs
OUTPUT_8B="../../music/llm_responses/LLaMA8B_responses.csv"
OUTPUT_3B="../../music/llm_responses/LLaMA3B_responses.csv"

# Models
MODEL_8B="meta-llama/Llama-3.1-8B-Instruct"
MODEL_3B="meta-llama/Llama-3.2-3B-Instruct"

MAX_RETRIES=3

# ---------------- END CONFIG ----------------

echo "Launching tmux session: ${SESSION_NAME}"

tmux new-session -d -s ${SESSION_NAME}

tmux send-keys -t ${SESSION_NAME} "
source ~/.bashrc
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TOKENIZERS_PARALLELISM=false

echo '=================================================='
echo 'Running LLaMA 8B...'
echo '=================================================='

python ${SCRIPT_PATH} \
  --model ${MODEL_8B} \
  --input_csv ${CSV_PATH} \
  --output_csv ${OUTPUT_8B} \
  --max_retries ${MAX_RETRIES}

echo 'Finished LLaMA 8B at:'
date

echo '=================================================='
echo 'Running LLaMA 3B...'
echo '=================================================='

python ${SCRIPT_PATH} \
  --model ${MODEL_3B} \
  --input_csv ${CSV_PATH} \
  --output_csv ${OUTPUT_3B} \
  --max_retries ${MAX_RETRIES}

echo 'Finished LLaMA 3B at:'
date

echo '=================================================='
echo 'All models complete.'
echo '=================================================='
" C-m

tmux attach -t ${SESSION_NAME}
