#!/bin/bash
# ================================================================
# run_preprocess_all_ecgs_HRV_complete.sh
# ------------------------------------------------
# Launch batch ECG preprocessing inside a dedicated tmux session.
#
# Modify the ARGUMENTS section to point to your dataset and settings.
# ================================================================

SESSION_NAME="preprocess_ecgs_HRV_complete"
BASE_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/Holter_ECG"               
VERSION="."                               
OUTPUT_PATH="../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_HRV_complete"              
SKIP_SECONDS=30                             
CSV_NAME="preprocessing_metadata_HRV_complete.csv"       
LIMIT=""                                    

SCRIPT="preprocess_all_ecgs_HRV_complete.py"
CMD="python $SCRIPT \
    --base_path $BASE_PATH \
    --version $VERSION \
    --output_path $OUTPUT_PATH \
    --skip_seconds $SKIP_SECONDS \
    --csv_name $CSV_NAME"
if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "[INFO] Starting tmux session: $SESSION_NAME"
echo "[INFO] Running command:"
echo "$CMD"
echo

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "[INFO] Session $SESSION_NAME already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

tmux new-session -d -s $SESSION_NAME "$CMD"

echo "[INFO] Tmux session '$SESSION_NAME' started."
echo "Run:  tmux attach -t $SESSION_NAME"
echo "To detach: press CTRL+b then d"
