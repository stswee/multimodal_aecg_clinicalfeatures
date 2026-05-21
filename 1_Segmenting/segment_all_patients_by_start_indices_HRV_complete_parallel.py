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

Extended version:
-----------------
Adds NeuroKit-derived TABULAR features only (no vectors saved).

Uses:
- Precomputed ECG
- Precomputed R-peaks
- NeuroKit2 for HRV, morphology, phase summaries

python segment_all_patients_by_start_indices_HRV_complete.py --base_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/ --preprocessed_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_HRV/ --segments_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments_HRV_complete --csv_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/window_index_metadata_HRV_complete.csv --fs 200 --window_sec 30

Notes:
-----
For 936 patients in MUSIC study, segmenting the ECGs into 30s windows takes ~10min
"""

import os
import ast
import argparse
import numpy as np
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from multiprocessing import get_context
import warnings

warnings.filterwarnings(
    "ignore",
    message="DFA_alpha2 related indices will not be calculated"
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide"
)

N_WORKERS = 16 


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
# FAST RR / PVC / signal features
# -------------------------------------------------------------
def extract_segment_features_from_rpeaks(ecg, rpeaks_seg, fs):
    if len(rpeaks_seg) < 3:
        return None

    feats = {}

    rr = np.diff(rpeaks_seg) / fs * 1000.0
    rr_median = np.median(rr)

    feats.update({
        "rr_min": float(np.min(rr)),
        "rr_mean": float(np.mean(rr)),
        "rr_max": float(np.max(rr)),
        "rr_range": float(np.ptp(rr)),
        "longest_rr_pause": float(np.max(rr)),
        "SDNN": float(np.std(rr)),
        "RMSSD": float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        "pNN50": float(np.mean(np.abs(np.diff(rr)) > 50) * 100),
    })

    pvc_mask = rr < 0.8 * rr_median
    feats.update({
        "pvc_count": int(np.sum(pvc_mask)),
        "pvc_burden_pct": float(100 * np.sum(pvc_mask) / len(rr)),
        "pvc_couplets": int(np.sum(pvc_mask[:-1] & pvc_mask[1:])),
    })

    duration_sec = len(ecg) / fs
    feats.update({
        "n_rpeaks": int(len(rpeaks_seg)),
        "beats_per_min": float(len(rpeaks_seg) * (60.0 / duration_sec)),
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
# NeuroKit TABULAR features
# -------------------------------------------------------------
def extract_neurokit_tabular_features(ecg, rpeaks_seg, fs):
    feats = {}

    if len(rpeaks_seg) < 5:
        return feats

    try:
        signals, _ = nk.ecg_process(ecg, sampling_rate=fs)
    except Exception:
        return feats

    hr = signals["ECG_Rate"].values
    feats.update({
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
        "hr_range": float(np.ptp(hr)),
        "hr_slope": float(np.polyfit(np.arange(len(hr)), hr, 1)[0]),
    })

    q = signals["ECG_Quality"].values
    feats.update({
        "ecg_quality_mean": float(np.mean(q)),
        "ecg_quality_std": float(np.std(q)),
        "ecg_quality_low_frac": float(np.mean(q < 0.8)),
    })

    try:
        hrv = pd.concat([
            nk.hrv_time(rpeaks_seg, sampling_rate=fs, show=False),
            nk.hrv_frequency(rpeaks_seg, sampling_rate=fs, show=False),
            nk.hrv_nonlinear(rpeaks_seg, sampling_rate=fs, show=False),
        ], axis=1)
        feats.update(hrv.iloc[0].to_dict())
    except Exception:
        pass

    return feats


# -------------------------------------------------------------
# Patient-level worker (NEW, minimal)
# -------------------------------------------------------------
def process_patient(args_tuple):
    row, args, FS, WINDOW_LEN = args_tuple

    patient_id = row["patient_id"]
    start_indices = ast.literal_eval(row["start_indices"])

    out_dir = os.path.join(args.segments_dir, patient_id)
    out_csv = os.path.join(out_dir, f"{patient_id}_segment_features.csv")

    # Resume support
    if os.path.exists(out_csv):
        return

    npz_path = find_npz_for_patient(args.preprocessed_dir, patient_id)
    if npz_path is None:
        return

    data = np.load(npz_path)
    ecg = data["signal"]
    rpeaks_all = data["rpeaks"]

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for i, start_idx in enumerate(start_indices):
        end_idx = start_idx + WINDOW_LEN
        if end_idx > len(ecg):
            continue

        segment = ecg[start_idx:end_idx].astype(np.float32)
        rpeaks_seg = rpeaks_all[
            (rpeaks_all >= start_idx) & (rpeaks_all < end_idx)
        ] - start_idx

        feats = extract_segment_features_from_rpeaks(segment, rpeaks_seg, FS)
        if feats is None:
            continue

        feats.update(extract_neurokit_tabular_features(segment, rpeaks_seg, FS))
        feats.update({
            "patient_id": patient_id,
            "window_idx": i,
            "start_idx": start_idx,
            "duration_sec": args.window_sec,
        })

        rows.append(feats)

    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)


# -------------------------------------------------------------
# Main (MINIMALLY MODIFIED)
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--preprocessed_dir", type=str, required=True)
    parser.add_argument("--segments_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--window_sec", type=int, default=30)
    args = parser.parse_args()

    FS = args.fs
    WINDOW_LEN = FS * args.window_sec

    meta = pd.read_csv(args.csv_path, dtype={"patient_id": str})
    meta["patient_id"] = meta["patient_id"].str.strip()

    tasks = [(row, args, FS, WINDOW_LEN) for _, row in meta.iterrows()]

    ctx = get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        list(tqdm(
            pool.imap_unordered(process_patient, tasks),
            total=len(tasks),
            desc="Processing patients"
        ))

    print("All patients processed successfully.")


if __name__ == "__main__":
    main()
