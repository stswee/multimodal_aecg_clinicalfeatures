#!/usr/bin/env python3
"""
fft_all_ecg_segments.py

Goal:
-----
Convert time-domain ECG segments (.npy) into frequency-domain
representations using the Fourier transform.

Input:
------
segments_dir/
    ├── 0001/
    │   ├── 0001_window0000.npy
    │   ├── 0001_window0001.npy
    │   └── ...
    ├── 0002/
    │   └── ...

Output:
-------
fft_segments_dir/
    ├── 0001/
    │   ├── 0001_window0000_fft.npy
    │   ├── 0001_window0001_fft.npy
    │   └── ...
    ├── 0002/
    │   └── ...

Usage:
------
python fft_all_ecg_segments.py \
    --segments_dir path/to/preprocessed_segments \
    --output_dir path/to/fft_segments \
    --fs 200 \
    --log_scale

Notes:
------
- Uses rFFT (real FFT): output length = N/2 + 1
- Saves magnitude spectrum |FFT|
- Log scaling is recommended for CNN training
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def compute_fft(segment, fs, log_scale=False):
    """
    Compute magnitude spectrum using real FFT.

    Parameters
    ----------
    segment : np.ndarray
        Time-domain ECG segment (1D)
    fs : int
        Sampling frequency (Hz)
    log_scale : bool
        Apply log(1 + x) scaling

    Returns
    -------
    mag : np.ndarray
        Magnitude spectrum
    """
    # Remove DC offset (important for ECG FFTs)
    segment = segment - np.mean(segment)

    # Real FFT
    fft_vals = np.fft.rfft(segment)

    # Magnitude
    mag = np.abs(fft_vals)

    if log_scale:
        mag = np.log1p(mag)

    return mag.astype(np.float32)


def main():

    parser = argparse.ArgumentParser(description="Fourier transform ECG segments.")
    parser.add_argument("--segments_dir", type=str, required=True,
                        help="Directory with time-domain ECG segments")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save FFT segments")
    parser.add_argument("--fs", type=int, default=200,
                        help="Sampling frequency (Hz)")
    parser.add_argument("--log_scale", action="store_true",
                        help="Apply log(1 + magnitude)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    patient_ids = sorted([
        d for d in os.listdir(args.segments_dir)
        if os.path.isdir(os.path.join(args.segments_dir, d))
    ])

    print(f"Found {len(patient_ids)} patients.")

    for pid in patient_ids:
        in_dir = os.path.join(args.segments_dir, pid)
        out_dir = os.path.join(args.output_dir, pid)
        os.makedirs(out_dir, exist_ok=True)

        segment_files = sorted([
            f for f in os.listdir(in_dir)
            if f.endswith(".npy")
        ])

        print(f"\n=== Processing patient {pid} ({len(segment_files)} segments) ===")

        for fname in tqdm(segment_files, desc=f"FFT {pid}", ncols=80):
            in_path = os.path.join(in_dir, fname)
            out_name = fname.replace(".npy", "_fft.npy")
            out_path = os.path.join(out_dir, out_name)

            segment = np.load(in_path)

            fft_mag = compute_fft(
                segment,
                fs=args.fs,
                log_scale=args.log_scale
            )

            np.save(out_path, fft_mag)

        print(f"Saved {len(segment_files)} FFT segments for patient {pid}")

    print("\nAll FFT segments generated successfully!")


if __name__ == "__main__":
    main()
