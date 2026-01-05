#!/usr/bin/env bash
# ============================================================
# tmux launcher for HRV + ECG feature extraction
# One CSV per patient, one row per 30s segment
# ============================================================

# ---------------- USER CONFIG ----------------
SESSION_NAME="hrv_feature_extraction_music"
CONDA_ENV="shdb-af-analysis"          # <-- change if needed

# Input: pre-generated ECG segments
SEGMENTS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments_HRV"

# Output: per-patient CSVs
OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/segment_features_HRV"

FS=200

SCRIPT_PATH="calculate_hrv_features.py"

# Optional logging
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/hrv_feature_extraction_$(date +%Y%m%d_%H%M%S).log"

# ---------------- TMUX SETUP ----------------
tmux new-session -d -s ${SESSION_NAME}

tmux send-keys -t ${SESSION_NAME} "
echo 'Starting HRV feature extraction...'
echo 'Session: ${SESSION_NAME}'
echo 'Logging to: ${LOG_FILE}'
date

source ~/.bashrc
conda activate ${CONDA_ENV}

python ${SCRIPT_PATH} \
  --segments_dir ${SEGMENTS_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --fs ${FS} \
  2>&1 | tee ${LOG_FILE}

echo 'Finished HRV feature extraction.'
date
" C-m

tmux attach-session -t ${SESSION_NAME}
