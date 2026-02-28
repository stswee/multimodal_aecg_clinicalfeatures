#!/usr/bin/env python3
"""
plot_decision_curve_from_cv.py

Computes and plots Decision Curve Analysis (DCA)
directly from cross-fold validation predictions.

Expected per-model structure:

model_dir/
    val_fold_0/val_predictions.npz
    ...
    val_fold_4/val_predictions.npz

Example: python plot_all_mean_dca.py --model_dirs ../../music/best_results/tcn_ecg_embeddings/  ../../music/best_results/text_embeddings_LLaMA8B_BioBERT/ ../../music/best_results/text_embeddings_LLaMA3B_ClinicalBERT/  ../../music/best_results/projectconcat_embeddings_LLaMA8B_BioBERT/  ../../music/best_results/vectorgating_embeddings_LLaMA8B_BioBERT/  --model_names ECG LLaMA8B-BioBERT LLaMA3B-ClinicalBERT PC VG --save_dir ../../music/best_results/comparison_top
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Color Mapping (same logic as ROC script)
# =========================================================

def get_color(model_name):

    if model_name == "ECG":
        return "#1f77b4"

    elif model_name == "PC":
        return "#d62728"

    elif model_name == "VG":
        return "#9467bd"

    elif "BioBERT" in model_name:
        return "#ff7f0e"

    elif "ClinicalBERT" in model_name:
        return "#2ca02c"

    return "black"


# =========================================================
# Decision Curve
# =========================================================

def decision_curve(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []

    for pt in thresholds:
        preds = (y_prob >= pt).astype(int)

        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))

        nb = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(nb)

    return np.array(net_benefits)


def treat_all_net_benefit(y_true, thresholds):
    prevalence = np.mean(y_true)

    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))


# =========================================================
# Load CV Predictions
# =========================================================

def load_cv_predictions(model_dir, task):

    y_true_all = []
    y_prob_all = []

    fold_dirs = sorted(Path(model_dir).glob("val_fold_*"))

    if len(fold_dirs) == 0:
        raise RuntimeError(f"No val_fold_* found in {model_dir}")

    for fold_dir in fold_dirs:
        f = fold_dir / "val_predictions.npz"
        if not f.exists():
            continue

        data = np.load(f)

        y_true_all.append(data[f"y_true_{task}"])
        y_prob_all.append(data[f"y_prob_{task}"])

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_prob_all),
    )


# =========================================================
# Plot Panel
# =========================================================

def plot_panel(ax, model_dirs, model_names, task):

    thresholds = np.linspace(0.01, 0.30, 200)

    treat_all_plotted = False
    treat_none_plotted = False

    for model_dir, model_name in zip(model_dirs, model_names):

        y_true, y_prob = load_cv_predictions(model_dir, task)

        nb = decision_curve(y_true, y_prob, thresholds)
        nb_all = treat_all_net_benefit(y_true, thresholds)
        nb_none = np.zeros_like(thresholds)

        color = get_color(model_name)
        lw = 2.8 if model_name in ["PC", "VG"] else 2.2

        ax.plot(
            thresholds,
            nb,
            color=color,
            linewidth=lw,
            label=model_name,
        )

        if not treat_all_plotted:
            ax.plot(
                thresholds,
                nb_all,
                linestyle="--",
                color="gray",
                linewidth=1.5,
                label="Treat All",
            )
            treat_all_plotted = True

        if not treat_none_plotted:
            ax.plot(
                thresholds,
                nb_none,
                linestyle=":",
                color="black",
                linewidth=1.5,
                label="Treat None",
            )
            treat_none_plotted = True

    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(task.upper(), fontsize=14)
    ax.grid(alpha=0.2)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
    )

    legend.get_frame().set_linewidth(0.8)


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--save_dir", type=Path, required=True)

    args = parser.parse_args()

    model_dirs = [Path(d) for d in args.model_dirs]
    model_names = args.model_names

    assert len(model_dirs) == len(model_names)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    plot_panel(axes[0], model_dirs, model_names, "scd")
    plot_panel(axes[1], model_dirs, model_names, "pfd")

    plt.tight_layout()
    plt.savefig(
        args.save_dir / "decision_curve_comparison_panel.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nDecision curve panel saved.")


if __name__ == "__main__":
    main()