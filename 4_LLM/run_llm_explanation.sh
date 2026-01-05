#!/usr/bin/env bash
# ============================================================
# tmux launcher for LLaMA-based clinical explanation generation
# Joint SCD / PFD explanations (one per patient)
# ============================================================

# ---------------- USER CONFIG ----------------
SESSION_NAME="llama8B_scd_pfd_explanations"
GPU_ID=4

# Python environment
CONDA_ENV="shdb-af-analysis"   # change if needed

# Paths
SCRIPT_PATH="LLM_Explanation.py"

CSV_PATH="../../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/subject-info-cleaned-with-prompts.csv"
PROMPT_DIR="prompts"

OUTPUT_JSONL="../../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/llm_explanations_llama8B.jsonl"

# Model config
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS=1000

# ---------------- END CONFIG ----------------

echo "Launching tmux session: ${SESSION_NAME}"

tmux new-session -d -s ${SESSION_NAME}

tmux send-keys -t ${SESSION_NAME} "
source ~/.bashrc
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TOKENIZERS_PARALLELISM=false

python ${SCRIPT_PATH} \
  --csv_path ${CSV_PATH} \
  --model_name ${MODEL_NAME} \
  --prompt_dir ${PROMPT_DIR} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --output_jsonl ${OUTPUT_JSONL}

echo 'LLM explanation generation finished.'
" C-m

tmux attach -t ${SESSION_NAME}
