#!/usr/bin/env python3
"""
plot_mean_roc_cv.py

Constructs mean ROC curve with ±1 std band
from 5-fold cross-validation predictions.

Expected structure:

output_dir/
    val_fold_0/val_predictions.npz
    val_fold_1/val_predictions.npz
    ...
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# =========================================================
# Core ROC Aggregation
# =========================================================

def compute_mean_roc(y_true_list, y_prob_list, n_points=200):
    """
    Computes mean ROC and std band across folds.
    Returns:
        mean_fpr
        mean_tpr
        std_tpr
        mean_auc
        std_auc
    """

    mean_fpr = np.linspace(0, 1, n_points)
    tprs = []
    aucs = []

    for y_true, y_prob in zip(y_true_list, y_prob_list):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    tprs = np.array(tprs)

    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc


# =========================================================
# Plotting
# =========================================================

def plot_mean_roc(mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc,
                  title, save_path):

    plt.figure(figsize=(6, 6))

    # Mean ROC
    plt.plot(
        mean_fpr,
        mean_tpr,
        linewidth=2,
        label=f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}",
    )

    # ±1 SD band
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        alpha=0.3,
        label="±1 SD",
    )

    # Random baseline
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root directory containing val_fold_* folders",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ECG",
        help="Used for figure titles",
    )

    args = parser.parse_args()

    # ==========================================
    # Collect fold predictions
    # ==========================================
    y_true_scd_list = []
    y_prob_scd_list = []
    y_true_pfd_list = []
    y_prob_pfd_list = []

    for fold in range(args.n_folds):
        pred_path = args.output_dir / f"val_fold_{fold}" / "val_predictions.npz"
        if not pred_path.exists():
            print(f"Warning: missing fold {fold}")
            continue

        data = np.load(pred_path)

        y_true_scd_list.append(data["y_true_scd"])
        y_prob_scd_list.append(data["y_prob_scd"])

        y_true_pfd_list.append(data["y_true_pfd"])
        y_prob_pfd_list.append(data["y_prob_pfd"])

    if len(y_true_scd_list) == 0:
        raise RuntimeError("No folds found.")

    # ==========================================
    # Compute Mean ROC - SCD
    # ==========================================
    mean_fpr_s, mean_tpr_s, std_tpr_s, mean_auc_s, std_auc_s = compute_mean_roc(
        y_true_scd_list, y_prob_scd_list
    )

    plot_mean_roc(
        mean_fpr_s,
        mean_tpr_s,
        std_tpr_s,
        mean_auc_s,
        std_auc_s,
        title=f"{args.model_name} - Mean ROC (SCD)",
        save_path=args.output_dir / f"roc_scd_mean_{args.model_name}.png",
    )

    np.savez_compressed(
        args.output_dir / f"roc_scd_mean_{args.model_name}.npz",
        mean_fpr=mean_fpr_s,
        mean_tpr=mean_tpr_s,
        std_tpr=std_tpr_s,
        mean_auc=mean_auc_s,
        std_auc=std_auc_s,
    )

    # ==========================================
    # Compute Mean ROC - PFD
    # ==========================================
    mean_fpr_p, mean_tpr_p, std_tpr_p, mean_auc_p, std_auc_p = compute_mean_roc(
        y_true_pfd_list, y_prob_pfd_list
    )

    plot_mean_roc(
        mean_fpr_p,
        mean_tpr_p,
        std_tpr_p,
        mean_auc_p,
        std_auc_p,
        title=f"{args.model_name} - Mean ROC (PFD)",
        save_path=args.output_dir / f"roc_pfd_mean_{args.model_name}.png",
    )

    np.savez_compressed(
        args.output_dir / f"roc_pfd_mean_{args.model_name}.npz",
        mean_fpr=mean_fpr_p,
        mean_tpr=mean_tpr_p,
        std_tpr=std_tpr_p,
        mean_auc=mean_auc_p,
        std_auc=std_auc_p,
    )

    print("\nMean ROC curves saved successfully.")


if __name__ == "__main__":
    main()