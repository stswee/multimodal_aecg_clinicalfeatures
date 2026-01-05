#!/usr/bin/env python3
"""
preprocess_single_ecg_HRV_complete.py

Purpose
-------
Preprocess one ambulatory ECG recording. This script performs:

  1. Load a single Holter ECG record
  2. Optional trimming of initial seconds
  3. Bandpass filtering (0.5–40 Hz)
  4. Baseline correction
  5. Single-pass R-peak detection
  6. Single-pass NeuroKit processing (FULL ECG)
  7. Save sliceable outputs for fast segment-level HRV extraction
"""

import wfdb
import numpy as np
import scipy.signal as sp
import argparse
import time
import os
import neurokit2 as nk


# -------------------------------------------------------------
# Normalize record ID
# -------------------------------------------------------------
def normalize_record_id(record_str):
    record_str = record_str.strip()

    if record_str.startswith("P") and len(record_str) == 5:
        return record_str, int(record_str[1:])

    try:
        num = int(record_str)
        return f"P{num:04d}", num
    except ValueError:
        raise ValueError(f"Invalid record ID format: {record_str}")


# -------------------------------------------------------------
# Bandpass filter
# -------------------------------------------------------------
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    nyquist = 0.5 * fs
    b, a = sp.butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return sp.filtfilt(b, a, signal)


# -------------------------------------------------------------
# Baseline correction
# -------------------------------------------------------------
def baseline_correction(signal, fs, window_sec=0.8):
    kernel = int(window_sec * fs // 2 * 2 + 1)
    baseline = sp.medfilt(signal, kernel_size=kernel)
    return signal - baseline


# -------------------------------------------------------------
# Preprocess one ECG (FULL, SINGLE-PASS)
# -------------------------------------------------------------
def preprocess_record(record_number_raw, base_path, output_path, skip_seconds):

    meta = {
        "record_input": record_number_raw,
        "status": "success",
        "error_msg": "",
        "skip_seconds": skip_seconds,
    }

    # ---------------------------------------------------------
    # Normalize record ID
    # ---------------------------------------------------------
    try:
        record_id, record_num_int = normalize_record_id(record_number_raw)
        meta["record_id"] = record_id
    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = str(e)
        return meta

    hea_path = os.path.join(base_path, record_id + ".hea")
    dat_path = os.path.join(base_path, record_id + ".dat")
    if not (os.path.exists(hea_path) and os.path.exists(dat_path)):
        meta["status"] = "error"
        meta["error_msg"] = "Missing .hea or .dat file"
        return meta

    # ---------------------------------------------------------
    # Load WFDB
    # ---------------------------------------------------------
    try:
        t0 = time.time()
        record = wfdb.rdrecord(os.path.join(base_path, record_id))
        signal = record.p_signal[:, 1]
        fs = record.fs
        meta["sampling_rate_hz"] = fs
        meta["orig_n_samples"] = len(signal)
        meta["load_time_s"] = round(time.time() - t0, 3)
    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"Loading failure: {e}"
        return meta

    # ---------------------------------------------------------
    # Trim
    # ---------------------------------------------------------
    skip_samples = int(skip_seconds * fs)
    if skip_samples > 0:
        signal = signal[skip_samples:]
    meta["post_trim_samples"] = len(signal)

    # ---------------------------------------------------------
    # Bandpass filter
    # ---------------------------------------------------------
    try:
        t1 = time.time()
        filtered = bandpass_filter(signal, fs)
        meta["filter_time_s"] = round(time.time() - t1, 3)
    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"Filtering failure: {e}"
        return meta

    # ---------------------------------------------------------
    # Baseline correction
    # ---------------------------------------------------------
    try:
        t2 = time.time()
        cleaned = baseline_correction(filtered, fs)
        meta["baseline_time_s"] = round(time.time() - t2, 3)
    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"Baseline correction failure: {e}"
        return meta

    cleaned = cleaned.astype(np.float32)

    # ---------------------------------------------------------
    # R-peak detection (ONCE)
    # ---------------------------------------------------------
    try:
        t3 = time.time()
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = rpeaks["ECG_R_Peaks"].astype(np.int32)
        meta["rpeak_time_s"] = round(time.time() - t3, 3)
        meta["n_rpeaks"] = int(len(rpeaks))
    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"R-peak detection failure: {e}"
        return meta

    # ---------------------------------------------------------
    # NeuroKit full ECG processing (ONCE)
    # ---------------------------------------------------------
    try:
        t4 = time.time()
        signals, _ = nk.ecg_process(cleaned, sampling_rate=fs)
        meta["neurokit_time_s"] = round(time.time() - t4, 3)
    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"NeuroKit processing failure: {e}"
        return meta

    # ---------------------------------------------------------
    # Save sliceable outputs
    # ---------------------------------------------------------
    try:
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"{record_num_int:04d}_preprocessed.npz")

        t5 = time.time()
        np.savez_compressed(
            save_path,
            signal=cleaned,
            rpeaks=rpeaks,
            fs=fs,

            # ---- Sliceable NeuroKit outputs ----
            ECG_Rate=signals["ECG_Rate"].values.astype(np.float32),
            ECG_Quality=signals["ECG_Quality"].values.astype(np.float32),
            ECG_Phase_Ventricular=signals["ECG_Phase_Ventricular"].values.astype(np.float32),
            ECG_Phase_Atrial=signals["ECG_Phase_Atrial"].values.astype(np.float32),
            ECG_Phase_Completion_Ventricular=signals[
                "ECG_Phase_Completion_Ventricular"
            ].values.astype(np.float32),

            record_name=record_id,
            n_samples=len(cleaned),
        )

        meta["save_time_s"] = round(time.time() - t5, 3)
        meta["output_file"] = save_path
        meta["output_size_mb"] = round(os.path.getsize(save_path) / 1e6, 3)

        meta["total_time_s"] = round(
            meta["load_time_s"]
            + meta["filter_time_s"]
            + meta["baseline_time_s"]
            + meta["rpeak_time_s"]
            + meta["neurokit_time_s"]
            + meta["save_time_s"],
            3,
        )

    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"Saving failure: {e}"
        return meta

    return meta


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ECG with single-pass R-peak + NeuroKit processing"
    )
    parser.add_argument("--record", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--skip_seconds", type=float, default=30.0)
    args = parser.parse_args()

    meta = preprocess_record(
        record_number_raw=args.record,
        base_path=args.base_path,
        output_path=args.output_path,
        skip_seconds=args.skip_seconds,
    )

    print("\nMetadata summary:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
