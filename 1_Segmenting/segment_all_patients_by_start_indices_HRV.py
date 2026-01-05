#!/usr/bin/env python3
"""
segment_all_patients_by_start_indices_HRV.py

Goal:
-----
Segment all preprocessed ECGs into fixed-length windows
using the start indices provided in window_index_metadata_HRV.csv.

Usage:
------
This script should be run after create_window_index_metadata.py. Make sure that the arguments match what was used in create_window_index_metadata.py (fs and window_sec)

Uses precomputed R-peaks stored in preprocessing step.
Computes HRV / RR / PVC features per segment efficiently.

python segment_all_patients_by_start_indices_HRV.py --base_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/ --preprocessed_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_HRV/ --segments_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments_HRV --csv_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/window_index_metadata_HRV.csv --fs 200 --window_sec 30

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
from scipy.stats import skew, kurtosis


# -------------------------------------------------------------
# Locate NPZ for patient
# -------------------------------------------------------------
def find_npz_for_patient(preprocessed_dir, patient_id):
    candidates = [
        f for f in os.listdir(preprocessed_dir)
        if f.endswith(".npz") and patient_id in f
    ]
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        print(f"[!] Multiple matches for patient {patient_id}. Using first.")
    return os.path.join(preprocessed_dir, candidates[0])


# -------------------------------------------------------------
# FAST feature extraction using precomputed R-peaks
# -------------------------------------------------------------
def extract_segment_features_from_rpeaks(ecg, rpeaks_seg, fs):
    if len(rpeaks_seg) < 3:
        return None

    feats = {}

    # RR intervals (ms)
    rr = np.diff(rpeaks_seg) / fs * 1000.0
    rr_median = np.median(rr)

    # --- RR / HRV features ---
    feats.update({
        "rr_min": float(np.min(rr)),
        "rr_mean": float(np.mean(rr)),
        "rr_max": float(np.max(rr)),
        "rr_range": float(np.max(rr) - np.min(rr)),
        "longest_rr_pause": float(np.max(rr)),
        "SDNN": float(np.std(rr)),
        "RMSSD": float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        "pNN50": float(np.mean(np.abs(np.diff(rr)) > 50) * 100),
    })

    # --- PVC heuristics (RR-based) ---
    pvc_mask = rr < 0.8 * rr_median
    feats.update({
        "pvc_count": int(np.sum(pvc_mask)),
        "pvc_burden_pct": float(100 * np.sum(pvc_mask) / len(rr)),
        "pvc_couplets": int(np.sum(pvc_mask[:-1] & pvc_mask[1:])),
    })

    # --- Heart rate ---
    duration_sec = len(ecg) / fs
    feats.update({
        "n_rpeaks": int(len(rpeaks_seg)),
        "beats_per_min": float(len(rpeaks_seg) * (60.0 / duration_sec)),
    })

    # --- Signal morphology ---
    feats.update({
        "signal_mean": float(np.mean(ecg)),
        "signal_std": float(np.std(ecg)),
        "signal_var": float(np.var(ecg)),
        "signal_rms": float(np.sqrt(np.mean(ecg ** 2))),
        "signal_range": float(np.ptp(ecg)),
        "signal_skew": float(skew(ecg)),
        "signal_kurtosis": float(kurtosis(ecg)),
        "zero_crossings": int(np.sum(np.diff(np.sign(ecg)) != 0)),
    })

    return feats


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FAST segment ECGs and compute HRV using precomputed R-peaks."
    )
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--preprocessed_dir", type=str, required=True)
    parser.add_argument("--segments_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--window_sec", type=int, default=30)
    args = parser.parse_args()

    FS = args.fs
    WINDOW_SEC = args.window_sec
    WINDOW_LEN = FS * WINDOW_SEC

    meta = pd.read_csv(args.csv_path, dtype={"patient_id": str})
    meta["patient_id"] = meta["patient_id"].str.strip()
    print(f"Loaded metadata with {len(meta)} rows.")

    for _, row in tqdm(
        meta.iterrows(),
        total=len(meta),
        desc="Processing patients",
        unit="patient",
        ncols=100,
    ):
        patient_id = row["patient_id"]
        start_indices = ast.literal_eval(row["start_indices"])

        npz_path = find_npz_for_patient(args.preprocessed_dir, patient_id)
        if npz_path is None:
            print(f"[!] Missing ECG for {patient_id}")
            continue

        data = np.load(npz_path)
        ecg = data["signal"]
        rpeaks_all = data["rpeaks"]
        total_len = len(ecg)

        patient_out_dir = os.path.join(args.segments_dir, patient_id)
        segments_out_dir = os.path.join(patient_out_dir, "segments")
        os.makedirs(segments_out_dir, exist_ok=True)

        feature_rows = []

        for i, start_idx in tqdm(
            enumerate(start_indices),
            total=len(start_indices),
            desc=f"Segmenting {patient_id}",
            unit="segment",
            leave=False,
            ncols=80,
        ):
            start_idx = int(start_idx)
            end_idx = start_idx + WINDOW_LEN
            if end_idx > total_len:
                continue

            segment = ecg[start_idx:end_idx].astype(np.float32)

            # Save segment
            np.save(
                os.path.join(segments_out_dir, f"{patient_id}_window{i:04d}.npy"),
                segment,
            )

            # Slice R-peaks for this segment (local indices)
            mask = (rpeaks_all >= start_idx) & (rpeaks_all < end_idx)
            rpeaks_seg = rpeaks_all[mask] - start_idx

            feats = extract_segment_features_from_rpeaks(segment, rpeaks_seg, FS)
            if feats is not None:
                feats.update({
                    "patient_id": patient_id,
                    "window_idx": i,
                    "start_idx": start_idx,
                    "duration_sec": WINDOW_SEC,
                })
                feature_rows.append(feats)

        # --- Save ONE CSV per patient ---
        if feature_rows:
            df_feat = pd.DataFrame(feature_rows)
            csv_out = os.path.join(
                patient_out_dir,
                f"{patient_id}_segment_features.csv",
            )
            df_feat.to_csv(csv_out, index=False)

    print("\nAll patients processed successfully!")


if __name__ == "__main__":
    main()
