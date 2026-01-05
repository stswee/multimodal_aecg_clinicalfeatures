#!/bin/bash
# ================================================================
# run_preprocess_all_ecgs.sh
# ------------------------------------------------
# Launch batch ECG preprocessing inside a dedicated tmux session.
#
# Modify the ARGUMENTS section to point to your dataset and settings.
# ================================================================

SESSION_NAME="preprocess_ecgs_HRV_complete"

# ---------------------------
# ARGUMENTS TO MODIFY
# ---------------------------
BASE_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/Holter_ECG"               # e.g., /local3/sswee/music_download/.../Holter_ECG
VERSION="."                                # "." or e.g. "1.0.1"
OUTPUT_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_HRV_complete"              # e.g., /local3/sswee/MUSIC/preprocessed
SKIP_SECONDS=30                             # e.g., 30 for MUSIC, 0 for SHDB-AF
CSV_NAME="preprocessing_metadata_HRV_complete.csv"       # choose output metadata CSV name
LIMIT=""                                    # e.g., 50 (or leave blank for all records)

# Python script
SCRIPT="preprocess_all_ecgs_HRV_complete.py"

# ---------------------------
# Build command
# ---------------------------
CMD="python $SCRIPT \
    --base_path $BASE_PATH \
    --version $VERSION \
    --output_path $OUTPUT_PATH \
    --skip_seconds $SKIP_SECONDS \
    --csv_name $CSV_NAME"

# Optionally add limit argument
if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# ---------------------------
# Start tmux session
# ---------------------------
echo "[INFO] Starting tmux session: $SESSION_NAME"
echo "[INFO] Running command:"
echo "$CMD"
echo

# Create session only if it doesn't already exist
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "[INFO] Session $SESSION_NAME already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

# Create new tmux session and run
tmux new-session -d -s $SESSION_NAME "$CMD"

echo "[INFO] Tmux session '$SESSION_NAME' started."
echo "Run:  tmux attach -t $SESSION_NAME"
echo "To detach: press CTRL+b then d"
