#!/usr/bin/env python3
"""
extract_interpretability_panels.py

Given:
- patient_id
- block_start (segment index)
- block_end   (segment index)

This script:
1) Extracts hour-level HRV summary (Panel C)
2) Selects a representative 30s ECG window within the block using arrhythmia-oriented features (if available)
3) Saves the 30s ECG segment plot WITHOUT any annotations (Panel B)
4) Saves outputs for figure assembly

Example:

python extract_interpretability_panels.py \
  --patient_id 0779 \
  --block_start 1080 \
  --block_end 1200 \
  --task scd \
  --segments_dir ../../music/preprocessed_segments_HRV_complete \
  --preprocessed_dir ../../music/preprocessed_HRV \
  --fs 200 \
  --window_sec 30 \
  --out_dir interpretability_0779

python extract_interpretability_panels.py \
  --patient_id 0480 \
  --block_start 1440 \
  --block_end 1560 \
  --task pfd \
  --segments_dir ../../music/preprocessed_segments_HRV_complete \
  --preprocessed_dir ../../music/preprocessed_HRV \
  --fs 200 \
  --window_sec 30 \
  --out_dir interpretability_0480
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ECG plotting (NO annotations)
# ============================================================

def save_ecg_plot(ecg_snippet: np.ndarray, fs: int, out_path: str, title: str = "Representative 30-sec ECG Segment"):
    t = np.arange(len(ecg_snippet)) / fs
    plt.figure(figsize=(12, 3))
    plt.plot(t, ecg_snippet, linewidth=0.8)
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude (a.u.)")
    plt.title(title)
    plt.xlim([0, len(ecg_snippet) / fs])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ============================================================
# Segment selection (arrhythmia-oriented heuristics)
# ============================================================

def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median/MAD. Safe for heavy-tailed features like PVC burden."""
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        # fallback to std
        std = np.nanstd(x)
        if not np.isfinite(std) or std < 1e-12:
            return np.zeros_like(x, dtype=np.float64)
        return (x - med) / std
    return 0.6745 * (x - med) / mad


def select_representative_window(block_df: pd.DataFrame, task: str) -> int:
    """
    Select representative window differently for SCD vs PFD.

    SCD emphasizes:
        - PVC burden
        - pauses
        - ectopy
        - high short-term variability

    PFD emphasizes:
        - rate instability
        - diffuse variability
        - pauses
        - quality-controlled instability
    """

    score = np.zeros(len(block_df), dtype=np.float64)

    def add_feature(col, weight=1.0, transform="z"):
        nonlocal score
        if col not in block_df.columns:
            return
        vals = pd.to_numeric(block_df[col], errors="coerce").to_numpy()
        if transform == "z":
            score += weight * robust_z(vals)
        elif transform == "absdev":
            med = np.nanmedian(vals)
            score += weight * robust_z(np.abs(vals - med))

    # =============================
    # SCD scoring
    # =============================
    if task == "scd":

        add_feature("pvc_burden_pct", weight=3.0)
        add_feature("pvc_count", weight=2.0)
        add_feature("pvc_couplets", weight=2.0)
        add_feature("longest_rr_pause", weight=2.5)

        add_feature("HRV_RMSSD", weight=1.5)
        add_feature("RMSSD", weight=1.5)
        add_feature("HRV_SDNN", weight=1.0)
        add_feature("SDNN", weight=1.0)

        add_feature("rr_range", weight=1.0)

    # =============================
    # PFD scoring
    # =============================
    elif task == "pfd":

        add_feature("hr_mean", weight=2.0, transform="absdev")
        add_feature("beats_per_min", weight=2.0, transform="absdev")

        add_feature("HRV_SDNN", weight=2.0)
        add_feature("SDNN", weight=2.0)
        add_feature("HRV_RMSSD", weight=1.5)
        add_feature("RMSSD", weight=1.5)

        add_feature("longest_rr_pause", weight=1.5)
        add_feature("pvc_burden_pct", weight=1.0)

    # =============================
    # Quality penalty (for both)
    # =============================
    if "ecg_quality_low_frac" in block_df.columns:
        qlow = pd.to_numeric(block_df["ecg_quality_low_frac"], errors="coerce").to_numpy()
        score -= 2.0 * robust_z(qlow)

    if "ecg_quality_mean" in block_df.columns:
        qmean = pd.to_numeric(block_df["ecg_quality_mean"], errors="coerce").to_numpy()
        score += 1.0 * robust_z(qmean)

    if not np.any(np.isfinite(score)) or np.nanmax(np.abs(score)) < 1e-9:
        return len(block_df) // 2

    return int(np.nanargmax(score))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--patient_id", type=str, required=True)
    parser.add_argument("--block_start", type=int, required=True)
    parser.add_argument("--block_end", type=int, required=True)
    parser.add_argument("--segments_dir", type=str, required=True)
    parser.add_argument("--preprocessed_dir", type=str, required=True)
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--window_sec", type=int, default=30)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["scd", "pfd"])

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pid = str(args.patient_id).zfill(4)

    # ---------------------------------------------------------
    # Load segment CSV
    # ---------------------------------------------------------
    seg_csv = os.path.join(args.segments_dir, pid, f"{pid}_segment_features.csv")
    df = pd.read_csv(seg_csv)
    df = df.sort_values("window_idx").reset_index(drop=True)

    block_df = df.iloc[args.block_start:args.block_end].copy()
    if len(block_df) == 0:
        raise ValueError("Selected block has no segments.")

    # ---------------------------------------------------------
    # Hour-level feature summary (Panel C)
    # ---------------------------------------------------------
    numeric_cols = block_df.select_dtypes(include=[np.number]).columns
    hour_summary = block_df[numeric_cols].mean(numeric_only=True)
    hour_summary.to_csv(os.path.join(args.out_dir, "hour_summary.csv"), header=["mean_value"])

    # ---------------------------------------------------------
    # Choose a representative window within block (arrhythmia-oriented)
    # ---------------------------------------------------------
    best_local = select_representative_window(block_df, args.task)
    segment_row = block_df.iloc[best_local]

    if "start_idx" not in segment_row.index:
        raise KeyError("segment CSV is missing required column: start_idx")

    segment_start_sample = int(segment_row["start_idx"])
    segment_end_sample = segment_start_sample + args.window_sec * args.fs

    # Save which window we picked (useful for reproducibility)
    pick_info = {
        "patient_id": pid,
        "block_start": int(args.block_start),
        "block_end": int(args.block_end),
        "selected_block_local_index": int(best_local),
        "selected_window_idx": int(segment_row["window_idx"]) if "window_idx" in segment_row.index else None,
        "selected_start_idx": int(segment_start_sample),
        "selected_duration_sec": int(args.window_sec),
    }
    with open(os.path.join(args.out_dir, "selection.json"), "w") as f:
        import json
        json.dump(pick_info, f, indent=2)

    # ---------------------------------------------------------
    # Load raw ECG (use mmap to avoid full load stalls on slow FS)
    # ---------------------------------------------------------
    npz_candidates = [f for f in os.listdir(args.preprocessed_dir) if f.endswith(".npz") and pid in f]
    if len(npz_candidates) == 0:
        raise FileNotFoundError(f"Raw ECG NPZ not found for patient_id={pid} in {args.preprocessed_dir}")

    npz_path = os.path.join(args.preprocessed_dir, npz_candidates[0])
    data = np.load(npz_path, mmap_mode="r")
    ecg = data["signal"]

    if segment_end_sample > len(ecg):
        raise ValueError(
            f"Selected segment exceeds ECG length: end={segment_end_sample}, len(ecg)={len(ecg)}"
        )

    ecg_snippet = np.asarray(ecg[segment_start_sample:segment_end_sample], dtype=np.float32)

    # Save waveform
    np.save(os.path.join(args.out_dir, "ecg_snippet.npy"), ecg_snippet)

    # Plot waveform ONLY (no R-peaks, no extra annotations)
    save_ecg_plot(
        ecg_snippet,
        args.fs,
        os.path.join(args.out_dir, "ecg_snippet.png"),
        title="Representative 30-sec ECG Segment"
    )

    # ---------------------------------------------------------
    # Save selected segment-level features (for Panel C callout or debugging)
    # ---------------------------------------------------------
    segment_summary = segment_row[numeric_cols]
    segment_summary.to_csv(os.path.join(args.out_dir, "segment_summary.csv"), header=["value"])

    print("Interpretability panels extracted successfully.")
    print(f"Selected window_idx={pick_info['selected_window_idx']} (local_in_block={best_local})")


if __name__ == "__main__":
    main()