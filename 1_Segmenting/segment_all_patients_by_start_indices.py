#!/usr/bin/env python3
"""
segment_all_patients_by_start_indices.py

Goal:
-----
Segment all preprocessed ECGs into fixed-length windows
using the start indices provided in window_index_metadata.csv.

Usage:
------
This script should be run after create_window_index_metadata.py. Make sure that the arguments match what was used in create_window_index_metadata.py (fs and window_sec)

python segment_all_patients_by_start_indices.py --base_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/ --preprocessed_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed/ --segments_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments --csv_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/window_index_metadata.csv --fs 200 --window_sec 30

Notes:
-----
For 936 patients in MUSIC study, segmenting the ECGs into 30s windows takes ~10min
"""

import os
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_npz_for_patient(preprocessed_dir, patient_id):
    """
    Find the .npz file corresponding to a given patient_id.

    Matches filenames that contain the numeric ID anywhere in the filename.

    Example:
        patient_id = "0001"
        Matches:
          - "0001_preprocessed.npz"
          - "P0001_data.npz"
          - "record_0001_signal.npz"
    """
    candidates = [
        f for f in os.listdir(preprocessed_dir)
        if f.endswith(".npz") and patient_id in f
    ]

    if len(candidates) == 0:
        return None  # handled upstream with warning

    if len(candidates) > 1:
        print(f"[!] Multiple matches for patient {patient_id}. Using the first:")
        for c in candidates:
            print("   -", c)

    return os.path.join(preprocessed_dir, candidates[0])


def main():

    parser = argparse.ArgumentParser(description="Segment ECGs using precomputed start indices.")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Base directory of the dataset")
    parser.add_argument("--preprocessed_dir", type=str, required=True,
                        help="Directory containing preprocessed NPZ files")
    parser.add_argument("--segments_dir", type=str, required=True,
                        help="Directory to save output segments")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to window_index_metadata.csv")
    parser.add_argument("--fs", type=int, default=200,
                        help="Sampling frequency (default: 200 Hz)")
    parser.add_argument("--window_sec", type=int, default=30,
                        help="Window length in seconds (default: 30)")
    args = parser.parse_args()

    FS = args.fs
    WINDOW_SEC = args.window_sec
    WINDOW_LEN = FS * WINDOW_SEC

    # --- Load metadata (force patient_id as string) ---
    meta = pd.read_csv(args.csv_path, dtype={"patient_id": str})
    meta["patient_id"] = meta["patient_id"].str.strip()
    print(f"Loaded metadata with {len(meta)} rows.")

    # --- Process each patient ---
    for _, row in meta.iterrows():
        patient_id = row["patient_id"]  # e.g., "0001"
        num_windows = int(row["n_windows"])
        start_indices_raw = row["start_indices"]

        # --- Parse start indices list ---
        try:
            start_indices = ast.literal_eval(start_indices_raw)
        except Exception as e:
            print(f"[!] Failed to parse start_indices for patient {patient_id}: {e}")
            continue

        if not isinstance(start_indices, (list, tuple)):
            print(f"[!] start_indices for {patient_id} is not a list, skipping.")
            continue

        print(f"\n=== Processing patient {patient_id} ({len(start_indices)} windows) ===")

        # --- Locate ECG file robustly ---
        npz_path = find_npz_for_patient(args.preprocessed_dir, patient_id)
        if npz_path is None:
            print(f"[!] No ECG file found for patient {patient_id}, skipping.")
            continue

        try:
            data = np.load(npz_path)
        except Exception as e:
            print(f"[!] Could not load ECG file for patient {patient_id}: {e}")
            continue

        key = list(data.keys())[0]
        ecg = data[key]
        total_len = len(ecg)

        print(f"  ECG length: {total_len} samples")

        # --- Create output directory ---
        patient_out_dir = os.path.join(args.segments_dir, patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)

        # --- Segment and Save ---
        for i, start_idx in tqdm(
            enumerate(start_indices),
            total=len(start_indices),
            desc=f"Segmenting {patient_id}",
            ncols=80
        ):
            start_idx = int(start_idx)
            end_idx = start_idx + WINDOW_LEN

            if end_idx <= total_len:
                segment = ecg[start_idx:end_idx].astype(np.float32)
                out_path = os.path.join(patient_out_dir, f"{patient_id}_window{i:04d}.npy")
                np.save(out_path, segment)
            else:
                print(f"  Skipping window {i}: end_idx {end_idx} > {total_len}")

        num_saved = len([f for f in os.listdir(patient_out_dir) if f.endswith(".npy")])
        print(f"Done: saved {num_saved}/{num_windows} windows for patient {patient_id}")

    print("\nAll patients processed successfully!")


if __name__ == "__main__":
    main()
