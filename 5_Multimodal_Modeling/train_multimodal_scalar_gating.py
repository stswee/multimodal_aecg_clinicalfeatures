#!/usr/bin/env python3
"""
train_multimodal_scalar_gating.py

Scalar Gating Multimodal Fusion

ECG embedding: 128-d
Text embedding: auto-inferred

Projection:
    h_ECG  = W_e * z_ECG   -> proj_dim
    h_Text = W_t * z_Text  -> proj_dim

Scalar Gate:
    g = sigmoid(alpha)   (learnable scalar)

Fusion:
    z = g*h_ECG + (1-g)*h_Text

Two binary heads:
- SCD vs Survivor
- PFD vs Survivor

Tracks:
- Per-epoch metrics
- Best checkpoint by mean AUC
- Reports learned gate value g
- Global CV summary CSV
"""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# =========================================================
# Logging / Seed
# =========================================================

def setup_logging(out_dir: Path):
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_multimodal_scalar_gating_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_path}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================================================
# Model
# =========================================================

class ScalarGatingMultiHead(nn.Module):
    """
    Scalar gating multimodal classifier.
    """

    def __init__(
        self,
        ecg_dim: int,
        text_dim: int,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_layers: int = 1,
    ):
        super().__init__()

        self.ecg_proj = nn.Sequential(
            nn.LayerNorm(ecg_dim),
            nn.Linear(ecg_dim, proj_dim),
            nn.ReLU(),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
        )

        # Learnable scalar parameter (unconstrained)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # Configurable fusion trunk
        layers = []
        in_dim = proj_dim

        layers.append(nn.LayerNorm(proj_dim))

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        self.head_scd = nn.Linear(hidden_dim, 1)
        self.head_pfd = nn.Linear(hidden_dim, 1)

    def forward(self, ecg, text):

        h_ecg = self.ecg_proj(ecg)
        h_text = self.text_proj(text)

        g = torch.sigmoid(self.alpha)

        z_fused = g * h_ecg + (1.0 - g) * h_text

        z = self.trunk(z_fused)

        return (
            self.head_scd(z).squeeze(-1),
            self.head_pfd(z).squeeze(-1),
            z,
            g,
        )

# =========================================================
# Evaluation
# =========================================================

def eval_head_np(y_true, y_prob):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return float(acc), float(prec), float(rec), float(f1), float(auc)

# =========================================================
# Embedding Loader
# =========================================================

def load_npz(path):
    data = np.load(path)
    return (
        data["pids"],
        data["z"],
        data["y_scd"],
        data["y_pfd"],
    )

# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--ecg_embedding_dir", type=Path, required=True)
    parser.add_argument("--text_embedding_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("multimodal_outputs"))

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=1,
                    help="Number of MLP layers in fusion trunk")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = args.output_dir / f"val_fold_{args.val_fold}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)

    # -----------------------------------------------------
    # Load embeddings
    # -----------------------------------------------------

    ecg_fold_dir = args.ecg_embedding_dir / f"val_fold_{args.val_fold}"
    text_fold_dir = args.text_embedding_dir / f"val_fold_{args.val_fold}"

    ecg_train = load_npz(ecg_fold_dir / "train_embeddings.npz")
    ecg_val = load_npz(ecg_fold_dir / "val_embeddings.npz")

    text_train = load_npz(text_fold_dir / "train_embeddings.npz")
    text_val = load_npz(text_fold_dir / "val_embeddings.npz")

    X_ecg_train = ecg_train[1]
    X_ecg_val = ecg_val[1]

    X_text_train = text_train[1]
    X_text_val = text_val[1]

    y_train = np.stack([ecg_train[2], ecg_train[3]], axis=1)
    y_val = np.stack([ecg_val[2], ecg_val[3]], axis=1)

    ecg_dim = X_ecg_train.shape[1]
    text_dim = X_text_train.shape[1]

    logging.info(f"ECG dim: {ecg_dim}")
    logging.info(f"Text dim: {text_dim}")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = ScalarGatingMultiHead(
        ecg_dim=ecg_dim,
        text_dim=text_dim,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Class imbalance
    n_pos_scd = y_train[:, 0].sum()
    n_pos_pfd = y_train[:, 1].sum()
    n_neg = len(y_train) - ((y_train.sum(axis=1) > 0).sum())

    w_scd = torch.tensor([n_neg / max(n_pos_scd, 1)], device=device)
    w_pfd = torch.tensor([n_neg / max(n_pos_pfd, 1)], device=device)

    crit_scd = nn.BCEWithLogitsLoss(pos_weight=w_scd)
    crit_pfd = nn.BCEWithLogitsLoss(pos_weight=w_pfd)

    X_ecg_train_t = torch.tensor(X_ecg_train, dtype=torch.float32).to(device)
    X_text_train_t = torch.tensor(X_text_train, dtype=torch.float32).to(device)
    X_ecg_val_t = torch.tensor(X_ecg_val, dtype=torch.float32).to(device)
    X_text_val_t = torch.tensor(X_text_val, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    best_mean_auc = -np.inf
    best_epoch = -1
    best_scd_auc = np.nan
    best_pfd_auc = np.nan
    best_gate = np.nan

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()

        z_scd, z_pfd, _, g = model(X_ecg_train_t, X_text_train_t)

        loss = (
            crit_scd(z_scd, y_train_t[:, 0]) +
            crit_pfd(z_pfd, y_train_t[:, 1])
        )

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            z_scd, z_pfd, _, g_val = model(X_ecg_val_t, X_text_val_t)

            p_scd = torch.sigmoid(z_scd).cpu().numpy()
            p_pfd = torch.sigmoid(z_pfd).cpu().numpy()

        acc_s, prec_s, rec_s, f1_s, auc_s = eval_head_np(y_val[:,0], p_scd)
        acc_p, prec_p, rec_p, f1_p, auc_p = eval_head_np(y_val[:,1], p_pfd)

        mean_auc = np.nanmean([auc_s, auc_p])
        gate_value = float(g_val.item())

        logging.info(
            f"Epoch {epoch+1} | "
            f"SCD AUC={auc_s:.3f} | "
            f"PFD AUC={auc_p:.3f} | "
            f"MEAN AUC={mean_auc:.3f} | "
            f"Gate g={gate_value:.4f}"
        )

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_epoch = epoch + 1
            best_scd_auc = auc_s
            best_pfd_auc = auc_p
            best_gate = gate_value

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "gate_value": best_gate,
                },
                out / "best_model.pt",
            )

    # -----------------------------------------------------
    # Save fold summary
    # -----------------------------------------------------

    fold_summary = {
        "val_fold": args.val_fold,
        "best_epoch": best_epoch,
        "best_mean_auc": best_mean_auc,
        "best_scd_auc": best_scd_auc,
        "best_pfd_auc": best_pfd_auc,
        "best_gate": best_gate,
    }

    for k, v in vars(args).items():
        fold_summary[f"param_{k}"] = v

    global_csv = args.output_dir / "cv_results_all_folds.csv"
    df_row = pd.DataFrame([fold_summary])

    if global_csv.exists():
        df_row.to_csv(global_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(global_csv, index=False)

    logging.info("========================================")
    logging.info(f"Best mean AUC: {best_mean_auc:.4f}")
    logging.info(f"Best gate g  : {best_gate:.4f}")
    logging.info("========================================")

if __name__ == "__main__":
    main()
