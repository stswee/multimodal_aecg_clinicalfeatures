#!/usr/bin/env python3
"""
create_window_index_metadata.py

Goal:
-----
Generate window_index_metadata.csv for on-the-fly ECG segmentation.

Each row corresponds to one preprocessed ECG (.npz) file and includes:
    patient_id, fs, window_sec, signal_length, n_windows, start_indices

Usage:
------
This script should be run before segment_all_patients_by_start_indices.py

python create_window_index_metadata.py --input_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed/ --output_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/ --fs 200 --window_sec 30

Notes:
-----
For 936 patients in MUSIC study, generating metadata for 30s windows takes ~5min
"""

import os
import csv
import argparse
import re
import numpy as np
from tqdm import tqdm


def extract_numeric_patient_id(pid_raw):
    """
    Extract trailing numeric digits from a patient id.
    Examples:
        "P0001" -> "0001"
        "patient045" -> "045"
        "REC_102" -> "102"
        "003" -> "003"
    If no digits found, return the original string.
    """
    pid_raw = str(pid_raw)

    # Find trailing digits (robust across datasets)
    match = re.search(r"(\d+)$", pid_raw)
    if match:
        return match.group(1).zfill(3)  # ensure consistent padding (3 digits or more)
    else:
        return pid_raw  # fallback: return original if no numeric pattern found


def generate_window_indices(input_dir, output_dir, fs=200, window_sec=30):
    """Scan preprocessed ECGs and compute window start indices."""
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "window_index_metadata_HRV.csv")

    win_len = fs * window_sec
    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "fs", "window_sec", "signal_length", "n_windows", "start_indices"])

        for fname in tqdm(npz_files, desc="Processing ECGs", unit="file"):
            fpath = os.path.join(input_dir, fname)
            try:
                data = np.load(fpath)

                sig = data["signal"]
                if sig.ndim > 1:
                    sig = sig[0]  # (C, N) → (N,)
                T = len(sig)

                n_windows = T // win_len
                if n_windows == 0:
                    print(f"[!] Skipping {fname}: too short for one {window_sec}s window.")
                    continue

                starts = list(range(0, n_windows * win_len, win_len))

                # --- Extract patient ID ---
                raw_pid = data.get("record_name", os.path.basename(fname).split("_")[0])
                if isinstance(raw_pid, np.ndarray):
                    raw_pid = raw_pid.item()

                clean_pid = extract_numeric_patient_id(raw_pid)

                writer.writerow([clean_pid, fs, window_sec, T, n_windows, str(starts)])

            except Exception as e:
                print(f"[!] Error on {fname}: {e}")

    print(f"\nMetadata saved to: {output_csv}")
    print(f"Total processed ECGs: {len(npz_files)}")


def main():
    parser = argparse.ArgumentParser(description="Generate window index metadata for ECGs.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to preprocessed ECG folder.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save window_index_metadata.csv.")
    parser.add_argument("--fs", type=int, default=200,
                        help="Sampling frequency (Hz).")
    parser.add_argument("--window_sec", type=int, default=30,
                        help="Window duration (seconds).")
    args = parser.parse_args()

    generate_window_indices(args.input_dir, args.output_dir,
                            fs=args.fs, window_sec=args.window_sec)


if __name__ == "__main__":
    main()