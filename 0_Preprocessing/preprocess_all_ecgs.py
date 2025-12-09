#!/usr/bin/env python3
"""
preprocess_all_ecgs.py

Purpose
-------
Batch-preprocess ambulatory ECGs from any dataset where records are stored
in WFDB format (.hea/.dat). Works with MUSIC, SHDB-AF, or any dataset
containing PXXXX.hea files.

This script:
  - Discovers ECG records automatically by scanning for *.hea files
  - Normalizes record names (e.g., '1', '001', 'P0001')
  - Calls preprocess_record() from preprocess_single_ecg.py
  - Logs metadata for each processed record into a CSV
  - Supports dataset versioning (e.g., "1.0.1")
  - Supports limiting number of processed records (for debugging)

Metadata includes all processing times, output file size, duration, status, etc.
"""

import os
import re
import csv
import time
import argparse
from tqdm import tqdm
from datetime import datetime

# Import your updated single-record processor
from preprocess_single_ecg import preprocess_record, normalize_record_id


# -------------------------------------------------------------
# Discover WFDB records in a folder
# -------------------------------------------------------------
def discover_records(folder):
    """
    Return sorted list of WFDB record base names (e.g., P0001, P0123). 
    Works for both:
       P0001.hea
       0001.hea
       1.hea
    """
    hea_files = [f for f in os.listdir(folder) if f.endswith(".hea")]

    records = []
    for f in hea_files:
        base = f[:-4]  # drop .hea
        try:
            norm, _ = normalize_record_id(base)
            records.append(norm)
        except Exception:
            # ignore non-conforming files
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
        "output_file",
        "output_size_mb",
        "load_time_s",
        "filter_time_s",
        "baseline_time_s",
        "normalize_time_s",
        "save_time_s",
        "total_time_s",
        "status",
        "error_msg",
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


# -------------------------------------------------------------
# Append one record's metadata into CSV
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
            meta.get("output_file", ""),
            meta.get("output_size_mb", ""),
            meta.get("load_time_s", ""),
            meta.get("filter_time_s", ""),
            meta.get("baseline_time_s", ""),
            meta.get("normalize_time_s", ""),
            meta.get("save_time_s", ""),
            meta.get("total_time_s", ""),
            meta.get("status", ""),
            meta.get("error_msg", ""),
        ])


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch preprocess ECG Holter recordings.")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base directory containing WFDB dataset (folder containing version subfolder)."
    )
    parser.add_argument(
        "--version",
        type=str,
        default=".",
        help="Version subfolder name (e.g., '1.0.1'). Use '.' if files are directly in base_path."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Folder where preprocessed files should be saved."
    )
    parser.add_argument(
        "--skip_seconds",
        type=float,
        default=30.0,
        help="Seconds to trim from the beginning of every ECG."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit the number of records to process."
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="preprocessing_metadata.csv",
        help="CSV filename where metadata is logged."
    )

    args = parser.parse_args()

    # Discover directory containing WFDB files
    data_dir = os.path.join(args.base_path, args.version)
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # CSV path
    csv_path = os.path.join(args.output_path, args.csv_name)
    os.makedirs(args.output_path, exist_ok=True)
    init_csv(csv_path)

    # Discover records
    records = discover_records(data_dir)
    if args.limit:
        records = records[: args.limit]

    print(f"Found {len(records)} WFDB records in: {data_dir}")

    # Process
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
    print(f"Batch preprocessing complete.")
    print(f"Processed: {processed}, Errors: {errors}")
    print(f"Total time: {total_minutes:.2f} minutes")
    print(f"Metadata saved to: {csv_path}")
    print("==============================================\n")


if __name__ == "__main__":
    main()
