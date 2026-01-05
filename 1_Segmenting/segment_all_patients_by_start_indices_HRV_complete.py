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
import warnings
warnings.filterwarnings(
    "ignore",
    message="DFA_alpha2 related indices will not be calculated"
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide"
)

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
# FAST RR / PVC / signal features (existing)
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
    })

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
# NeuroKit TABULAR features only
# -------------------------------------------------------------
def extract_neurokit_tabular_features(ecg, rpeaks_seg, fs):
    feats = {}

    if len(rpeaks_seg) < 5:
        return feats

    # --- ECG processing ---
    signals, info = nk.ecg_process(ecg, sampling_rate=fs)

    # ---------------- HR dynamics ----------------
    hr = signals["ECG_Rate"].values
    feats.update({
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
        "hr_range": float(np.ptp(hr)),
        "hr_slope": float(np.polyfit(np.arange(len(hr)), hr, 1)[0]),
    })

    # ---------------- Signal quality ----------------
    q = signals["ECG_Quality"].values
    feats.update({
        "ecg_quality_mean": float(np.mean(q)),
        "ecg_quality_std": float(np.std(q)),
        "ecg_quality_low_frac": float(np.mean(q < 0.8)),
    })

    # ---------------- HRV (time / freq / nonlinear) ----------------
    try:
        hrv = pd.concat([
            nk.hrv_time(rpeaks_seg, sampling_rate=fs, show=False),
            nk.hrv_frequency(rpeaks_seg, sampling_rate=fs, show=False),
            nk.hrv_nonlinear(rpeaks_seg, sampling_rate=fs, show=False),
        ], axis=1)
        feats.update(hrv.iloc[0].to_dict())
    except Exception:
        pass

    # ---------------- Morphology & phase ----------------
    try:
        _, delineate = nk.ecg_delineate(
            ecg,
            rpeaks=rpeaks_seg,
            sampling_rate=fs,
            method="dwt"
        )

        # Peak counts
        feats["n_p_peaks"] = int(np.sum(signals["ECG_P_Peaks"]))
        feats["n_t_peaks"] = int(np.sum(signals["ECG_T_Peaks"]))

        # Intervals (ms)
        if "ECG_Q_Onsets" in delineate and "ECG_S_Offsets" in delineate:
            qrs = (delineate["ECG_S_Offsets"] - delineate["ECG_Q_Onsets"]) / fs * 1000
            qrs = qrs[qrs > 0]
            if len(qrs):
                feats["qrs_mean"] = float(np.mean(qrs))
                feats["qrs_std"] = float(np.std(qrs))

        if "ECG_Q_Onsets" in delineate and "ECG_T_Offsets" in delineate:
            qt = (delineate["ECG_T_Offsets"] - delineate["ECG_Q_Onsets"]) / fs * 1000
            qt = qt[qt > 0]
            if len(qt):
                feats["qt_mean"] = float(np.mean(qt))
                feats["qt_std"] = float(np.std(qt))

        # Phase mechanics
        vent = signals["ECG_Phase_Ventricular"].values
        atr = signals["ECG_Phase_Atrial"].values
        vent_p = signals["ECG_Phase_Completion_Ventricular"].values

        feats.update({
            "ventricular_systole_frac": float(np.mean(vent)),
            "atrial_systole_frac": float(np.mean(atr)),
            "vent_phase_completion_mean": float(np.mean(vent_p)),
            "vent_phase_completion_std": float(np.std(vent_p)),
            "vent_phase_entropy": float(-np.mean(vent_p * np.log(vent_p + 1e-6))),
        })

    except Exception:
        pass

    return feats


# -------------------------------------------------------------
# Main
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

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing patients"):
        patient_id = row["patient_id"]
        start_indices = ast.literal_eval(row["start_indices"])

        npz_path = find_npz_for_patient(args.preprocessed_dir, patient_id)
        if npz_path is None:
            continue

        data = np.load(npz_path)
        ecg = data["signal"]
        rpeaks_all = data["rpeaks"]

        patient_out_dir = os.path.join(args.segments_dir, patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)

        rows = []

        for i, start_idx in tqdm(
            enumerate(start_indices),
            total=len(start_indices),
            desc=f"  Segments {patient_id}",
            leave=False,
            ncols=80
        ):
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
            pd.DataFrame(rows).to_csv(
                os.path.join(patient_out_dir, f"{patient_id}_segment_features.csv"),
                index=False
            )

    print("All patients processed successfully.")


if __name__ == "__main__":
    main()
