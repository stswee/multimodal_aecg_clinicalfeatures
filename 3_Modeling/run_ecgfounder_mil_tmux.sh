#!/usr/bin/env bash
# ============================================
# TMUX launcher for ECGFounder Hierarchical MIL
# ============================================

SESSION_NAME="ecgfounder_mil_fold0"
GPU_ID=2

# ---- Paths (EDIT THESE) ----
SCRIPT_PATH="train_mil_ecgfounder_hierarchical.py"

SEGMENTS_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments"
CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv"
ECGFOUNDER_WEIGHTS="../../1_lead_ECGFounder.pth"

OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/ecgfounder_mil_outputs"

VAL_FOLD=0

# ---- Training hyperparameters ----
EPOCHS=30
LR=1e-4
ENCODER_LR=1e-5
ACC_STEPS=4
ENCODE_BATCH_SIZE=128
NUM_WORKERS=4

# ---- Environment ----
CONDA_ENV="shdb-af-analysis"

# ============================================
# Create tmux session
# ============================================
tmux new-session -d -s ${SESSION_NAME}

# Set GPU
tmux send-keys -t ${SESSION_NAME} "export CUDA_VISIBLE_DEVICES=${GPU_ID}" C-m

# Activate conda
tmux send-keys -t ${SESSION_NAME} "source ~/.bashrc" C-m
tmux send-keys -t ${SESSION_NAME} "conda activate ${CONDA_ENV}" C-m

# Go to project directory (optional)
tmux send-keys -t ${SESSION_NAME} "cd $(dirname ${SCRIPT_PATH})" C-m

# Run training
tmux send-keys -t ${SESSION_NAME} "
python ${SCRIPT_PATH} \
  --val_fold ${VAL_FOLD} \
  --segments_dir ${SEGMENTS_DIR} \
  --csv_path ${CSV_PATH} \
  --ecgfounder_weights ${ECGFOUNDER_WEIGHTS} \
  --output_dir ${OUTPUT_DIR} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --encoder_lr ${ENCODER_LR} \
  --accumulation_steps ${ACC_STEPS} \
  --encode_batch_size ${ENCODE_BATCH_SIZE} \
  --num_workers ${NUM_WORKERS}
" C-m

echo "🚀 Launched tmux session: ${SESSION_NAME}"
echo "👉 Attach with: tmux attach -t ${SESSION_NAME}"
