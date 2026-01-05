#!/usr/bin/env python3
"""
calculate_hrv_features.py

Goal
----
For each patient:
- Load pre-generated 30s ECG segments (.npy)
- Compute ECG + HRV + morphology features
- Save ONE CSV per patient (one row per segment)

Author: You
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
from scipy.stats import skew, kurtosis


# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def safe_diff(arr):
    return np.diff(arr) if len(arr) > 1 else np.array([])


def compute_durations(onsets, offsets, fs):
    """
    Compute durations (ms) given onset and offset indices.
    Handles NeuroKit list outputs safely.
    """
    if onsets is None or offsets is None:
        return np.array([])

    if len(onsets) == 0 or len(offsets) == 0:
        return np.array([])

    onsets = np.asarray(onsets, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)

    n = min(len(onsets), len(offsets))
    durations = (offsets[:n] - onsets[:n]) / fs * 1000.0

    # Remove negative or absurd durations (NeuroKit occasional glitches)
    durations = durations[durations > 0]

    return durations



# ---------------------------------------------------------------------
# Core feature extractor (one 30s segment)
# ---------------------------------------------------------------------
def extract_segment_features(ecg, fs):
    feats = {}

    try:
        signals, info = nk.ecg_process(ecg, sampling_rate=fs, method="neurokit")
    except Exception:
        return None

    rpeaks = info.get("ECG_R_Peaks", [])
    if len(rpeaks) < 3:
        return None

    # --------------------------------------------------
    # RR & HR
    # --------------------------------------------------
    rr = safe_diff(rpeaks) / fs * 1000.0
    hr = nk.ecg_rate(rpeaks, sampling_rate=fs)

    feats.update({
        "n_rpeaks": int(len(rpeaks)),
        "beats_per_min": float(len(rpeaks) * 60 / (len(ecg) / fs)),
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
        "rr_mean": float(np.mean(rr)),
        "rr_median": float(np.median(rr)),
        "rr_min": float(np.min(rr)),
        "rr_max": float(np.max(rr)),
        "rr_range": float(np.ptp(rr)),
        "SDNN": float(np.std(rr)),
        "RMSSD": float(np.sqrt(np.mean(safe_diff(rr) ** 2))),
        "pNN50": float(np.mean(np.abs(safe_diff(rr)) > 50) * 100),
        "longest_rr_pause": float(np.max(rr)),
    })

    # --------------------------------------------------
    # HRV (NeuroKit2 time-domain)
    # --------------------------------------------------
    try:
        hrv = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
        for col in hrv.columns:
            feats[f"hrv_{col}"] = float(hrv.iloc[0][col])
    except Exception:
        pass

    # --------------------------------------------------
    # Fiducial points
    # --------------------------------------------------
    waves = info

    # ---------- P-wave ----------
    p_on = waves.get("ECG_P_Onsets", [])
    p_off = waves.get("ECG_P_Offsets", [])
    p_dur = compute_durations(p_on, p_off, fs)

    feats.update({
        "p_present_fraction": float(len(p_dur) / len(rpeaks)),
        "p_duration_mean": float(np.mean(p_dur)) if len(p_dur) else np.nan,
        "p_duration_std": float(np.std(p_dur)) if len(p_dur) else np.nan,
    })

    # ---------- QRS ----------
    r_on = waves.get("ECG_R_Onsets", [])
    r_off = waves.get("ECG_R_Offsets", [])
    qrs_dur = compute_durations(r_on, r_off, fs)

    feats.update({
        "qrs_duration_mean": float(np.mean(qrs_dur)) if len(qrs_dur) else np.nan,
        "qrs_duration_std": float(np.std(qrs_dur)) if len(qrs_dur) else np.nan,
        "wide_qrs_fraction": float(np.mean(qrs_dur > 120)) if len(qrs_dur) else np.nan,
    })

    # ---------- QT ----------
    t_on = waves.get("ECG_T_Onsets", [])
    t_off = waves.get("ECG_T_Offsets", [])
    qt_dur = compute_durations(r_on, t_off, fs)

    feats.update({
        "qt_interval_mean": float(np.mean(qt_dur)) if len(qt_dur) else np.nan,
        "qt_interval_std": float(np.std(qt_dur)) if len(qt_dur) else np.nan,
        "qt_interval_min": float(np.min(qt_dur)) if len(qt_dur) else np.nan,
        "qt_interval_max": float(np.max(qt_dur)) if len(qt_dur) else np.nan,
    })

    # --------------------------------------------------
    # Signal morphology
    # --------------------------------------------------
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fs", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    patient_ids = sorted(os.listdir(args.segments_dir))

    for pid in tqdm(patient_ids, desc="Patients", ncols=100):
        seg_dir = os.path.join(args.segments_dir, pid, "segments")
        if not os.path.isdir(seg_dir):
            continue

        rows = []

        files = sorted(f for f in os.listdir(seg_dir) if f.endswith(".npy"))

        for f in tqdm(files, leave=False, desc=pid, ncols=80):
            m = re.search(r"window(\d+)", f)
            window_idx = int(m.group(1)) if m else None

            ecg = np.load(os.path.join(seg_dir, f)).astype(np.float32)

            feats = extract_segment_features(ecg, args.fs)
            if feats is None:
                continue

            feats.update({
                "patient_id": pid,
                "window_idx": window_idx,
                "duration_sec": len(ecg) / args.fs,
            })

            rows.append(feats)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(
                os.path.join(args.output_dir, f"{pid}_segment_features.csv"),
                index=False,
            )

    print("\nAll patient CSVs generated successfully.")


if __name__ == "__main__":
    main()
