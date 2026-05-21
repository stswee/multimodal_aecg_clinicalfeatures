#!/bin/bash
# ================================================================
# run_segment_HRV_tmux.sh
# ------------------------------------------------
# Launch ECG segmentation + HRV feature extraction
# (16-worker multiprocessing, shared-server safe)
# ================================================================

SESSION_NAME="segment_hrv_complete"

CMD="
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

nice -n 5 python segment_all_patients_by_start_indices_HRV_complete_parallel.py \
  --base_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/ \
  --preprocessed_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_HRV/ \
  --segments_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments_HRV_complete \
  --csv_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/window_index_metadata_HRV.csv \
  --fs 200 \
  --window_sec 30
"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[INFO] tmux session '$SESSION_NAME' already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

tmux new-session -d -s "$SESSION_NAME"

tmux rename-window -t "$SESSION_NAME":0 "HRV-segmentation"

tmux send-keys -t "$SESSION_NAME":0 "$CMD" C-m

echo "============================================================"
echo " HRV segmentation job launched in tmux session: $SESSION_NAME"
echo " Attach with:"
echo "   tmux attach -t $SESSION_NAME"
echo "============================================================"
