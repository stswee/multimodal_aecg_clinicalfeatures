#!/usr/bin/env python3
"""
preprocess_single_ecg.py

Purpose
-------
Preprocess one ambulatory ECG recording. This script performs:

  1. Load a single Holter ECG record:
        - Accepts record inputs like: "1", "001", "P0001"
        - Converts them to valid WFDB record IDs ("P0001")
        - Extracts a single ECG lead (index 1)
        - Retrieves sampling rate

  2. (Optional) Remove the first N seconds:
        - Controlled by --skip_seconds
        - MUSIC often uses 30 seconds
        - SHDB-AF often uses 0 seconds

  3. Bandpass filtering (0.5–40 Hz)
  4. Baseline correction (median filter)
  5. Z-score normalization
  6. Save compressed .npz file
  7. Return full metadata, including processing times

This version is robust to missing .hea/.dat files.
"""

import wfdb
import numpy as np
import scipy.signal as sp
import argparse
import time
import os


# -------------------------------------------------------------
# Normalize record ID (accepts "1", "001", "P0001")
# -------------------------------------------------------------
def normalize_record_id(record_str):
    """Convert flexible record string into valid ID 'PXXXX'."""
    record_str = record_str.strip()

    # Already correct: P0001
    if record_str.startswith("P") and len(record_str) == 5:
        try:
            num = int(record_str[1:])
            return record_str, num
        except ValueError:
            pass

    # Numeric-only input
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
# Preprocess one ECG (robust version)
# -------------------------------------------------------------
def preprocess_record(record_number_raw, base_path, output_path, skip_seconds):

    # Metadata container (always returned)
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
        meta["error_msg"] = f"Record normalization failed: {e}"
        return meta

    # ---------------------------------------------------------
    # Check for existence of .hea and .dat (robust skipping)
    # ---------------------------------------------------------
    hea_path = os.path.join(base_path, record_id + ".hea")
    dat_path = os.path.join(base_path, record_id + ".dat")

    if not (os.path.exists(hea_path) and os.path.exists(dat_path)):
        meta["status"] = "error"
        meta["error_msg"] = "Missing .hea or .dat file"
        return meta

    # ---------------------------------------------------------
    # Load WFDB record
    # ---------------------------------------------------------
    try:
        t0 = time.time()
        record = wfdb.rdrecord(os.path.join(base_path, record_id))
        signal = record.p_signal[:, 1]   # Use single lead
        fs = record.fs
        load_time = time.time() - t0

        meta["sampling_rate_hz"] = fs
        meta["orig_n_samples"] = len(signal)
        meta["load_time_s"] = round(load_time, 3)

    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"Loading failure: {e}"
        return meta

    # ---------------------------------------------------------
    # Optional trimming
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
        baseline_corrected = baseline_correction(filtered, fs)
        meta["baseline_time_s"] = round(time.time() - t2, 3)
    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"Baseline correction failure: {e}"
        return meta

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------
    try:
        t3 = time.time()
        normalized = (baseline_corrected - np.mean(baseline_corrected)) / np.std(baseline_corrected)
        normalized32 = normalized.astype(np.float32)
        meta["normalize_time_s"] = round(time.time() - t3, 3)
    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"Normalization failure: {e}"
        return meta

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    try:
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"{record_num_int:04d}_preprocessed.npz")

        t4 = time.time()
        np.savez_compressed(
            save_path,
            signal=normalized32,
            fs=fs,
            record_name=record_id,
            n_samples=len(normalized32),
        )
        save_time = time.time() - t4

        meta["save_time_s"] = round(save_time, 3)
        meta["output_file"] = save_path
        meta["output_size_mb"] = round(os.path.getsize(save_path) / 1e6, 3)

        # total
        total = (
            meta["load_time_s"]
            + meta["filter_time_s"]
            + meta["baseline_time_s"]
            + meta["normalize_time_s"]
            + meta["save_time_s"]
        )
        meta["total_time_s"] = round(total, 3)

    except Exception as e:
        meta["status"] = "error"
        meta["error_msg"] = f"Saving failure: {e}"
        return meta

    return meta


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a single Holter ECG recording.")
    parser.add_argument("--record", type=str, required=True, help="Record number: '1', '001', or 'P0001'")
    parser.add_argument("--base_path", type=str, required=True, help="Directory containing WFDB files")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save .npz files")
    parser.add_argument("--skip_seconds", type=float, default=30.0,
                        help="Seconds to trim from start (default: 30)")
    args = parser.parse_args()

    meta = preprocess_record(
        record_number_raw=args.record,
        base_path=args.base_path,
        output_path=args.output_path,
        skip_seconds=args.skip_seconds,
    )

    print("\nMetadata summary:", meta)
