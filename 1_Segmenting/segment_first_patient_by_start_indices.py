#!/usr/bin/env python3
"""
segment_first_patient_by_start_indices.py

Goal:
-----
Segment a single preprocessed ECG into fixed-length windows
using the start indices provided in window_index_metadata.csv.

Arguments:
----------
--base_path             Base dataset directory
--preprocessed_dir      Directory containing preprocessed NPZ files
--segments_dir          Output directory for segmented windows
--csv_path              Metadata CSV containing start indices
--patient_id            Patient ID to process (e.g., "0001")
--fs                    Sampling frequency (default: 200)
--window_sec            Window length in seconds (default: 30)

Usage:
-----
This script can be used to test segmentation of a single patient. It is optional to run.
Make sure that the arguments match what was used in create_window_index_metadata.py (fs and window_sec)

python segment_first_patient_by_start_indices.py --base_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/ --preprocessed_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed/ --segments_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments --csv_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/window_index_metadata.csv --patient_id 0001 --fs 200 --window_sec 30


Notes:
-----
For segmenting a single 24h reading, the segmenting should be nearly instantaneous.
"""

import os
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_npz_for_patient(preprocessed_dir, patient_id):
    """
    Find the appropriate .npz file for a patient based on numeric patient_id.

    Matches filenames that contain the patient_id substring.
    Examples (patient_id = "0001"):
        "0001_preprocessed.npz"
        "P0001_preprocessed.npz"
        "record_0001_data.npz"
    """
    candidates = []
    for fname in os.listdir(preprocessed_dir):
        if fname.endswith(".npz") and patient_id in fname:
            candidates.append(fname)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No .npz file found in {preprocessed_dir} containing patient_id '{patient_id}'."
        )

    if len(candidates) > 1:
        print(f"[!] Multiple npz files match patient {patient_id}. Using the first:")
        for c in candidates:
            print("   -", c)

    return os.path.join(preprocessed_dir, candidates[0])


def main():

    # ---------------------------------------------------------
    # Argument Parser (inside main)
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Segment a single ECG using precomputed start indices.")

    parser.add_argument("--base_path", type=str, required=True,
                        help="Base directory of the dataset")

    parser.add_argument("--preprocessed_dir", type=str, required=True,
                        help="Directory containing preprocessed NPZ files")

    parser.add_argument("--segments_dir", type=str, required=True,
                        help="Directory to save segmented windows")

    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to window_index_metadata.csv")

    parser.add_argument("--patient_id", type=str, required=True,
                        help="Patient ID as stored in metadata (e.g., 0001)")

    parser.add_argument("--fs", type=int, default=200,
                        help="Sampling frequency (default: 200 Hz)")

    parser.add_argument("--window_sec", type=int, default=30,
                        help="Window length in seconds (default: 30)")

    args = parser.parse_args()
    # ---------------------------------------------------------

    FS = args.fs
    WINDOW_SEC = args.window_sec
    WINDOW_LEN = FS * WINDOW_SEC
    PATIENT_ID = str(args.patient_id)

    # --- Load metadata, forcing patient_id as string ---
    meta = pd.read_csv(args.csv_path, dtype={"patient_id": str})
    meta["patient_id"] = meta["patient_id"].str.strip()
    print(f"Loaded metadata with {len(meta)} rows: {list(meta.columns)}")

    # --- Filter for specific patient (string match first) ---
    row = meta.loc[meta["patient_id"] == PATIENT_ID]

    # Fallback: try numeric match (handles '1' vs '0001' cases)
    if row.empty:
        try:
            target_int = int(PATIENT_ID)
            meta_pid_int = (
                meta["patient_id"]
                .str.lstrip("0")          # "0001" -> "1"
                .replace("", "0")         # handle all-zero case
                .astype(int)
            )
            row = meta.loc[meta_pid_int == target_int]
        except Exception as e:
            print(f"[!] Numeric fallback for patient_id matching failed: {e}")

    if row.empty:
        raise ValueError(f"No metadata found for patient {PATIENT_ID}")

    row = row.iloc[0]
    num_windows = int(row["n_windows"])
    start_indices_raw = row["start_indices"]

    # Parse start indices
    try:
        start_indices = ast.literal_eval(start_indices_raw)
    except Exception as e:
        raise ValueError(f"Failed to parse start_indices for {PATIENT_ID}: {e}")

    if not isinstance(start_indices, (list, tuple)):
        raise ValueError(f"start_indices for {PATIENT_ID} is not a list")

    print(f"\n=== Patient {PATIENT_ID} ===")
    print(f"  Expected windows: {num_windows}")
    print(f"  Parsed start indices: {len(start_indices)}")

    # --- Create output directory ---
    patient_out_dir = os.path.join(args.segments_dir, PATIENT_ID)
    os.makedirs(patient_out_dir, exist_ok=True)

    # --- Locate ECG file robustly ---
    npz_path = find_npz_for_patient(args.preprocessed_dir, PATIENT_ID)
    print(f"\nLoading ECG from: {npz_path}")

    data = np.load(npz_path)
    key = list(data.keys())[0]
    ecg = data[key]
    total_len = len(ecg)

    print(f"  ECG shape: {ecg.shape}, dtype: {ecg.dtype}, length: {total_len}")

    # --- Segment and Save ---
    for i, start_idx in tqdm(
        enumerate(start_indices),
        total=len(start_indices),
        desc=f"Segmenting patient {PATIENT_ID}",
        ncols=80
    ):
        start_idx = int(start_idx)
        end_idx = start_idx + WINDOW_LEN

        if end_idx <= total_len:
            segment = ecg[start_idx:end_idx].astype(np.float32)
            out_path = os.path.join(patient_out_dir, f"{PATIENT_ID}_window{i:04d}.npy")
            np.save(out_path, segment)
        else:
            print(f"  Skipping window {i}: end_idx {end_idx} > signal length {total_len}")

    num_saved = len([f for f in os.listdir(patient_out_dir) if f.endswith(".npy")])

    print(f"\nDone! Saved {num_saved}/{num_windows} windows to {patient_out_dir}\n")


if __name__ == "__main__":
    main()
