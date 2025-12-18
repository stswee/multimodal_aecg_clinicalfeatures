#!/usr/bin/env bash
#
# run_mil_tmux_fusion.sh
#
# Late-fusion MIL (time + frequency domain ECG)
#

set -e

# ======================================================
# USER-EDITABLE SECTION
# ======================================================

# tmux session name
SESSION_NAME="mil_train_fold1_binary_testing_late_fusion_emb256"

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_mil_ecg_attention_binary_testing_fusion.py"

# Experiment arguments
VAL_FOLD=1

# Segment directories (TIME + FREQ)
SEGMENTS_TIME_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments"
SEGMENTS_FREQ_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments_fft"

CSV_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv"

OUTPUT_DIR="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_outputs_binary_testing_late_fusion_emb256"

# Optional / frequently changed hyperparameters
EPOCHS=10
LR="1e-5"
ATTENTION_DIM=128
DEVICE="cuda"

# Any extra args you want to experiment with
EXTRA_ARGS="--seed 42 --embedding_dim 256"

# ======================================================
# BUILD COMMAND
# ======================================================

CMD="export CUDA_VISIBLE_DEVICES=1 && \
python ${PYTHON_SCRIPT} \
  --val_fold ${VAL_FOLD} \
  --segments_time_dir ${SEGMENTS_TIME_DIR} \
  --segments_freq_dir ${SEGMENTS_FREQ_DIR} \
  --csv_path ${CSV_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --attention_dim ${ATTENTION_DIM} \
  --device ${DEVICE} \
  ${EXTRA_ARGS}
"

# ======================================================
# LAUNCH TMUX
# ======================================================

tmux new-session -d -s "${SESSION_NAME}"

tmux send-keys -t "${SESSION_NAME}" "cd ${SCRIPT_DIR}" C-m
tmux send-keys -t "${SESSION_NAME}" "${CMD}" C-m

echo "Started tmux session: ${SESSION_NAME}"
echo
echo "Command:"
echo "${CMD}"
echo
echo "Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
