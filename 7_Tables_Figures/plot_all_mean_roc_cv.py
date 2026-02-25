#!/usr/bin/env python3
"""
plot_roc_comparison.py

Combines precomputed mean ROC curves from multiple models
into a single comparison plot.

Expected per-model files:
    roc_scd_mean_<model>.npz
    roc_pfd_mean_<model>.npz

Example:
python plot_all_mean_roc_cv.py --model_dirs ../../music/best_results/tcn_ecg_embeddings/ ../../music/best_results/text_embeddings_LLaMA3B_BioBERT/ ../../music/best_results/text_embeddings_LLaMA8B_BioBERT/ ../../music/best_results/text_embeddings_LLaMA3B_ClinicalBERT/ ../../music/best_results/text_embeddings_LLaMA8B_ClinicalBERT/ ../../music/best_results/directconcat_embeddings_LLaMA8B_BioBERT/ ../../music/best_results/projectconcat_embeddings_LLaMA8B_BioBERT/ ../../music/best_results/scalargating_embeddings_LLaMA8B_BioBERT/ ../../music/best_results/vectorgating_embeddings_LLaMA8B_BioBERT/ ../../music/best_results/weightedsum_embeddings_LLaMA8B_BioBERT/ --model_names ECG LLaMA3B-BioBERT LLaMA8B-BioBERT LLaMA3B-ClinicalBERT LLaMA8B-ClinicalBERT DC PC SG VG WSG --save_dir ../../music/best_results/comparison

python plot_all_mean_roc_cv.py --model_dirs ../../music/best_results/tcn_ecg_embeddings/  ../../music/best_results/text_embeddings_LLaMA8B_BioBERT/ ../../music/best_results/text_embeddings_LLaMA3B_ClinicalBERT/  ../../music/best_results/projectconcat_embeddings_LLaMA8B_BioBERT/  ../../music/best_results/vectorgating_embeddings_LLaMA8B_BioBERT/  --model_names ECG LLaMA8B-BioBERT LLaMA3B-ClinicalBERT PC VG --save_dir ../../music/best_results/comparison_top
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Color Mapping (Structured by Modality)
# =========================================================

def get_color(model_name):

    if model_name == "ECG":
        return "#1f77b4"  # blue

    elif model_name == "PC":
        return "#d62728"  # red

    elif model_name == "VG":
        return "#9467bd"  # purple

    elif "BioBERT" in model_name:
        return "#ff7f0e"  # orange

    elif "ClinicalBERT" in model_name:
        return "#2ca02c"  # green

    return "black"


# =========================================================
# Plot One Panel
# =========================================================

def plot_panel(ax, model_dirs, model_names, task):

    for model_dir, model_name in zip(model_dirs, model_names):

        roc_file = model_dir / f"roc_{task}_mean_{model_name}.npz"

        if not roc_file.exists():
            print(f"Missing: {roc_file}")
            continue

        data = np.load(roc_file)

        mean_fpr = data["mean_fpr"]
        mean_tpr = data["mean_tpr"]
        mean_auc = float(data["mean_auc"])

        color = get_color(model_name)

        # Slight emphasis for multimodal
        lw = 2.8 if model_name in ["PC", "VG"] else 2.2

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            linewidth=lw,
            label=f"{model_name} (AUC = {mean_auc:.3f})",
        )

    # Random baseline
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
        alpha=0.4,
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(task.upper(), fontsize=14)
    ax.grid(alpha=0.2)

    # Legend INSIDE panel
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=10,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        borderpad=0.8,
    )
    
    legend.get_frame().set_linewidth(0.8)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dirs",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--model_names",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    model_dirs = [Path(d) for d in args.model_dirs]
    model_names = args.model_names

    assert len(model_dirs) == len(model_names)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    # 1x2 Panel
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    plot_panel(axes[0], model_dirs, model_names, task="scd")
    plot_panel(axes[1], model_dirs, model_names, task="pfd")

    plt.tight_layout()

    plt.savefig(
        args.save_dir / "roc_comparison_panel.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\n1x2 ROC comparison panel saved.")


if __name__ == "__main__":
    main()