#!/usr/bin/env python3
"""
preprocess_all_ecgs_HRV_complete.py

Purpose
-------
Batch-preprocess ambulatory ECGs using
preprocess_single_ecg_HRV_complete.py.

This script:
  - Discovers WFDB ECG records (*.hea)
  - Normalizes record names
  - Runs full single-pass ECG preprocessing
  - Logs detailed metadata to CSV

Designed for MUSIC / SHDB-AF scale processing.
"""

import os
import csv
import time
import argparse
from tqdm import tqdm
from datetime import datetime

# UPDATED IMPORT (matches modified single-ECG script)
from preprocess_single_ecg_HRV_complete import (
    preprocess_record,
    normalize_record_id,
)

# -------------------------------------------------------------
# Discover WFDB records
# -------------------------------------------------------------
def discover_records(folder):
    hea_files = [f for f in os.listdir(folder) if f.endswith(".hea")]

    records = []
    for f in hea_files:
        base = f[:-4]
        try:
            norm, _ = normalize_record_id(base)
            records.append(norm)
        except Exception:
            continue

    return sorted(records)


# -------------------------------------------------------------
# Initialize CSV metadata log
# -------------------------------------------------------------
def init_csv(csv_path):
    headers = [
        "timestamp",
        "record_id",
        "orig_n_samples",
        "post_trim_samples",
        "sampling_rate_hz",
        "skip_seconds",
        "n_rpeaks",
        "output_file",
        "output_size_mb",
        "load_time_s",
        "filter_time_s",
        "baseline_time_s",
        "rpeak_time_s",
        "neurokit_time_s",
        "save_time_s",
        "total_time_s",
        "status",
        "error_msg",
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


# -------------------------------------------------------------
# Append one record's metadata
# -------------------------------------------------------------
def log_metadata(csv_path, meta):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            meta.get("record_id", ""),
            meta.get("orig_n_samples", ""),
            meta.get("post_trim_samples", ""),
            meta.get("sampling_rate_hz", ""),
            meta.get("skip_seconds", ""),
            meta.get("n_rpeaks", ""),
            meta.get("output_file", ""),
            meta.get("output_size_mb", ""),
            meta.get("load_time_s", ""),
            meta.get("filter_time_s", ""),
            meta.get("baseline_time_s", ""),
            meta.get("rpeak_time_s", ""),
            meta.get("neurokit_time_s", ""),
            meta.get("save_time_s", ""),
            meta.get("total_time_s", ""),
            meta.get("status", ""),
            meta.get("error_msg", ""),
        ])


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch preprocess ECG Holter recordings (HRV-complete)."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base directory containing WFDB dataset."
    )
    parser.add_argument(
        "--version",
        type=str,
        default=".",
        help="Version subfolder (e.g., '1.0.1') or '.'"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save preprocessed ECGs."
    )
    parser.add_argument(
        "--skip_seconds",
        type=float,
        default=30.0,
        help="Seconds to trim from start of ECG."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records."
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="preprocessing_metadata_HRV_complete.csv",
        help="CSV filename for metadata logging."
    )

    args = parser.parse_args()

    data_dir = os.path.join(args.base_path, args.version)
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    os.makedirs(args.output_path, exist_ok=True)
    csv_path = os.path.join(args.output_path, args.csv_name)
    init_csv(csv_path)

    records = discover_records(data_dir)
    if args.limit:
        records = records[: args.limit]

    print(f"Found {len(records)} WFDB records in: {data_dir}")

    start_time = time.time()
    processed = 0
    errors = 0

    for rec in tqdm(records, desc="Preprocessing ECGs", unit="record"):
        try:
            meta = preprocess_record(
                record_number_raw=rec,
                base_path=data_dir,
                output_path=args.output_path,
                skip_seconds=args.skip_seconds,
            )
            processed += 1

        except Exception as e:
            errors += 1
            meta = {
                "record_id": rec,
                "status": "error",
                "error_msg": str(e),
            }
            print(f"[ERROR] Record {rec}: {e}")

        finally:
            log_metadata(csv_path, meta)

    total_minutes = (time.time() - start_time) / 60.0

    print("\n==============================================")
    print("Batch preprocessing complete.")
    print(f"Processed: {processed}, Errors: {errors}")
    print(f"Total time: {total_minutes:.2f} minutes")
    print(f"Metadata saved to: {csv_path}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
