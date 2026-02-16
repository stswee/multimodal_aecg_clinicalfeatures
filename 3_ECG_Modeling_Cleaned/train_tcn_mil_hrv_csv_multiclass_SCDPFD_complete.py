#!/usr/bin/env python3
"""
train_mil_features_tcn_multibranch.py

MIL + TCN training using per-segment engineered features (HRV/RR/PVC/morphology),
with a shared temporal encoder and task-specific branches for SCD and PFD.

Classes:
0 = Survivor
3 = Sudden Cardiac Death (SCD)
6 = Pump Failure Death (PFD)

Two binary heads:
- SCD vs Survivor (labels {0,3})
- PFD vs Survivor (labels {0,6})

Input per patient:
- One CSV per patient: <features_dir>/<pid>/<pid>_segment_features.csv
  containing one row per segment/window.

Sequence formed by sorting rows by window_idx (or start_idx).

Key behavior in this version:
- Drops known-invalid long-window HRV columns (e.g., SDANN*, ULF/VLF/LF/LFHF/LFn)
- Robust to NaN/Inf:
  - if --drop_na_rows: drop any segment rows with NaN/Inf
  - else: impute NaN/Inf with --impute_nan_with
- Train-fold z-score normalization (optional; disabled with --no_zscore)
- Validation uses the exact same cleaning + normalization as training
- Logs per-epoch: SCD AUC/F1, PFD AUC/F1, mean AUC
- Saves best checkpoint by mean AUC, last checkpoint, and metrics CSV
"""

import argparse
import json
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
)

# =========================================================
# Logging / Seed
# =========================================================
def setup_logging(out_dir: Path):
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_mil_features_tcn_multibranch_{ts}.log"

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
# Utilities: feature columns + normalization stats
# =========================================================
NON_FEATURE_COLS_DEFAULT = {"patient_id", "window_idx", "start_idx", "duration_sec"}

HRV_ALWAYS_INVALID = {
    "HRV_SDANN1",
    "HRV_SDNNI1",
    "HRV_SDANN2",
    "HRV_SDNNI2",
    "HRV_SDANN5",
    "HRV_SDNNI5",
    "HRV_ULF",
    "HRV_VLF",
    "HRV_LF",
    "HRV_LFHF",
    "HRV_LFn",
}

# =========================================================
# Allowed feature subset (defensible 30s HRV set)
# =========================================================
ALLOWED_FEATURES = {
    # --- Time-domain ---
    "HRV_MeanNN",
    "HRV_MedianNN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_SDSD",
    "HRV_CVNN",
    "HRV_CVSD",
    "HRV_IQRNN",
    "HRV_MadNN",
    "HRV_pNN20",
    "HRV_pNN50",
    "HRV_MinNN",
    "HRV_MaxNN",

    # --- Poincaré ---
    "HRV_SD1",
    "HRV_SD2",
    "HRV_SD1SD2",
    "HRV_S",

    # --- Frequency (30s defensible subset) ---
    "HRV_HF",
    "HRV_LnHF",
    "HRV_TP",

    # --- Nonlinear (short-window safe subset) ---
    "HRV_SampEn",

    # --- Rhythm / quality context ---
    "n_rpeaks",
    "pvc_burden_pct",
    "longest_rr_pause",
    "hr_mean",
    "hr_std",
    "ecg_quality_mean",
    "ecg_quality_low_frac",
}


def infer_feature_cols(
    df: pd.DataFrame,
    non_feature_cols=NON_FEATURE_COLS_DEFAULT,
    min_non_nan_frac: float = 0.05,
):
    """
    Select only allowed HRV features that:
    - are numeric
    - exist in the CSV
    - pass NaN fraction filter
    - are not explicitly invalid
    """

    feature_cols = []

    for c in df.columns:
        if c in non_feature_cols:
            continue
        if c in HRV_ALWAYS_INVALID:
            continue
        if c not in ALLOWED_FEATURES:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().mean() < float(min_non_nan_frac):
            continue

        feature_cols.append(c)

    return sorted(feature_cols)



@torch.no_grad()
def compute_train_feature_stats(
    train_df: pd.DataFrame,
    features_dir: Path,
    feature_cols: list[str],
    sort_by: str = "window_idx",
    pid_col: str = "Patient ID",
    drop_na_rows: bool = True,
    impute_nan_with: float = 0.0,
):
    """
    Compute mean/std over ALL segments from TRAIN patients only.
    Robust to NaN/Inf:
      - if drop_na_rows: drops rows with any NaN/Inf before computing stats
      - else: imputes NaN/Inf with impute_nan_with before computing stats
    """
    sums = np.zeros(len(feature_cols), dtype=np.float64)
    sumsqs = np.zeros(len(feature_cols), dtype=np.float64)
    counts = np.zeros(len(feature_cols), dtype=np.int64)

    missing_files = 0
    total_patients = len(train_df)

    for _, row in tqdm(
        train_df.iterrows(),
        total=total_patients,
        desc="Compute train feature stats",
        ncols=100,
    ):
        pid = str(row[pid_col]).zfill(4)
        csv_path = features_dir / pid / f"{pid}_segment_features.csv"
        if not csv_path.exists():
            missing_files += 1
            continue

        df_feat = pd.read_csv(csv_path)
        if df_feat.empty:
            continue

        if sort_by in df_feat.columns:
            df_feat = df_feat.sort_values(sort_by, ascending=True)

        # ensure columns exist
        if any(c not in df_feat.columns for c in feature_cols):
            continue

        X = df_feat[feature_cols].to_numpy(dtype=np.float64, copy=False)

        if drop_na_rows:
            # keep only fully finite rows
            good = np.isfinite(X).all(axis=1)
            X = X[good]
            if X.shape[0] == 0:
                continue
        else:
            X = np.where(np.isfinite(X), X, float(impute_nan_with)).astype(np.float64)

        mask = np.isfinite(X)
        sums += np.nansum(X, axis=0)
        sumsqs += np.nansum(X * X, axis=0)
        counts += mask.sum(axis=0)

    if missing_files > 0:
        logging.info(f"[Stats] Missing feature CSV for {missing_files}/{total_patients} train patients.")

    mean = np.zeros_like(sums)
    std = np.ones_like(sums)

    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    var = np.zeros_like(sums)
    var[valid] = (sumsqs[valid] / counts[valid]) - (mean[valid] ** 2)
    var[var < 1e-12] = 1e-12
    std[valid] = np.sqrt(var[valid])

    dead = [feature_cols[i] for i in range(len(feature_cols)) if counts[i] == 0]
    if dead:
        logging.warning(
            f"[Stats] {len(dead)} features have zero valid samples in TRAIN after cleaning. "
            f"Setting mean=0,std=1 for them. Examples: {dead[:10]}"
        )

    return mean.astype(np.float32), std.astype(np.float32)


# =========================================================
# Dataset
# =========================================================
class FeatureMILDataset(Dataset):
    """
    Returns:
      X: (N_segments, F) float32
      y: (2,) float32  [y_scd, y_pfd]
      pid: str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        feature_cols: list[str],
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        sort_by: str = "window_idx",
        pid_col: str = "Patient ID",
        label_col: str = "label",
        drop_na_rows: bool = True,
        impute_nan_with: float = 0.0,
        min_segments: int = 3,
    ):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir
        self.feature_cols = feature_cols
        self.mean = mean
        self.std = std
        self.sort_by = sort_by
        self.pid_col = pid_col
        self.label_col = label_col
        self.drop_na_rows = bool(drop_na_rows)
        self.impute_nan_with = float(impute_nan_with)
        self.min_segments = int(min_segments)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row[self.pid_col]).zfill(4)
        label = int(row[self.label_col])

        y_scd = 1 if label == 3 else 0
        y_pfd = 1 if label == 6 else 0
        y = torch.tensor([y_scd, y_pfd], dtype=torch.float32)

        csv_path = self.features_dir / pid / f"{pid}_segment_features.csv"
        if not csv_path.exists():
            X = torch.zeros((0, len(self.feature_cols)), dtype=torch.float32)
            return X, y, pid

        df_feat = pd.read_csv(csv_path)
        if df_feat.empty:
            X = torch.zeros((0, len(self.feature_cols)), dtype=torch.float32)
            return X, y, pid

        if self.sort_by in df_feat.columns:
            df_feat = df_feat.sort_values(self.sort_by, ascending=True)

        missing_cols = [c for c in self.feature_cols if c not in df_feat.columns]
        if missing_cols:
            X = torch.zeros((0, len(self.feature_cols)), dtype=torch.float32)
            return X, y, pid

        X = df_feat[self.feature_cols].to_numpy(dtype=np.float32, copy=False)

        if self.drop_na_rows:
            # Drop any segment rows with NaN OR ±Inf
            good = np.isfinite(X).all(axis=1)
            X = X[good]
        else:
            # Impute NaN/Inf with a constant
            X = np.where(np.isfinite(X), X, self.impute_nan_with).astype(np.float32)

        if X.shape[0] < self.min_segments:
            X = np.zeros((0, len(self.feature_cols)), dtype=np.float32)
            return torch.tensor(X, dtype=torch.float32), y, pid

        # Z-score (train-fold stats), then final safety clamp
        if self.mean is not None and self.std is not None:
            X = (X - self.mean) / self.std
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return torch.tensor(X, dtype=torch.float32), y, pid


def mil_collate_fn(batch):
    Xs, ys, pids = zip(*batch)
    return Xs, torch.stack(ys), pids


# =========================================================
# Losses
# =========================================================
class FocalLoss(nn.Module):
    """
    Binary focal loss on logits.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


# =========================================================
# Model
# =========================================================
class FeatureEncoder(nn.Module):
    """
    Per-segment feature encoder: (F) -> (emb)
    """
    def __init__(self, in_dim: int, emb_dim: int, hidden: int = 128, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return self.net(x)  # (N, emb)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, drop):
        super().__init__()
        pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.drop(self.relu(self.conv(x)))[:, :, :x.size(2)]
        return y + (x if self.down is None else self.down(x))


class TCN(nn.Module):
    def __init__(self, in_dim, hid, layers, k, drop):
        super().__init__()
        self.net = nn.Sequential(
            *[
                TemporalBlock(in_dim if i == 0 else hid, hid, k, 2**i, drop)
                for i in range(layers)
            ]
        )

    def forward(self, x):
        return self.net(x)


class AttentionMIL(nn.Module):
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, T):
        # T: (1, C, N)
        X = T.squeeze(0).transpose(0, 1)  # (N, C)
        a = torch.softmax(self.attn(X).squeeze(1), dim=0)  # (N,)
        z = (X * a.unsqueeze(1)).sum(dim=0)  # (C,)
        return z, a


class MILTCNFeatureMultiBranch(nn.Module):
    """
    Shared temporal feature encoder + MIL pooling, then task-specific branches.

    Forward returns:
      z_scd_logit, z_pfd_logit, z_shared, attn_weights
    """
    def __init__(
        self,
        in_feat_dim: int,
        emb: int,
        hid: int,
        layers: int,
        k: int,
        drop: float,
        attn: int,
        branch_hidden: int | None = None,
        branch_drop: float | None = None,
        enc_hidden: int = 128,
        enc_drop: float = 0.1,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(in_feat_dim, emb, hidden=enc_hidden, drop=enc_drop)
        self.tcn = TCN(emb, hid, layers, k, drop)
        self.pool = AttentionMIL(hid, attn)

        bh = hid if branch_hidden is None else int(branch_hidden)
        bd = drop if branch_drop is None else float(branch_drop)

        self.scd_branch = nn.Sequential(
            nn.Linear(hid, bh),
            nn.ReLU(),
            nn.Dropout(bd),
            nn.Linear(bh, hid),
            nn.ReLU(),
        )
        self.pfd_branch = nn.Sequential(
            nn.Linear(hid, bh),
            nn.ReLU(),
            nn.Dropout(bd),
            nn.Linear(bh, hid),
            nn.ReLU(),
        )

        self.head_scd = nn.Linear(hid, 1)
        self.head_pfd = nn.Linear(hid, 1)

    def forward(self, X: torch.Tensor):
        """
        X: (N_segments, F)
        """
        H = self.encoder(X)                    # (N, emb)
        H = H.transpose(0, 1).unsqueeze(0)     # (1, emb, N)
        T = self.tcn(H)                        # (1, hid, N)
        z, a = self.pool(T)                    # z: (hid,)

        z_scd = self.scd_branch(z)
        z_pfd = self.pfd_branch(z)

        return self.head_scd(z_scd), self.head_pfd(z_pfd), z, a


# =========================================================
# Evaluation
# =========================================================
# def tune_threshold_np(y_true: np.ndarray, y_prob: np.ndarray):
#     ts = np.linspace(0.05, 0.95, 181)
#     f1s = [f1_score(y_true, y_prob >= t, zero_division=0) for t in ts]
#     i = int(np.argmax(f1s))
#     return float(ts[i]), float(f1s[i])


def eval_head_np(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Returns: acc, prec, rec, f1, auc
    Uses fixed threshold = 0.5
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    mask = np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    # Fixed threshold
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1b, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return float(acc), float(prec), float(rec), float(f1b), float(auc)



# =========================================================
# Main
# =========================================================
def main():
    p = argparse.ArgumentParser()

    # --- Required args (keep as requested) ---
    p.add_argument("--val_fold", type=int, required=True)
    p.add_argument("--features_dir", type=Path, required=True,
                   help="Root dir containing <pid>/<pid>_segment_features.csv")
    p.add_argument("--csv_path", type=Path, required=True,
                   help="Patient-level CSV with columns: Patient ID, label, fold")

    p.add_argument("--output_dir", type=Path, default=Path("mil_feature_outputs"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # model (shared)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--tcn_hidden_dim", type=int, default=128)
    p.add_argument("--tcn_layers", type=int, default=4)
    p.add_argument("--tcn_kernel_size", type=int, default=3)
    p.add_argument("--tcn_dropout", type=float, default=0.2)
    p.add_argument("--attn_dim", type=int, default=128)

    # feature encoder
    p.add_argument("--enc_hidden", type=int, default=128)
    p.add_argument("--enc_dropout", type=float, default=0.1)

    # task branches
    p.add_argument("--branch_hidden", type=int, default=None,
                   help="Hidden width inside task branches. Default: use tcn_hidden_dim.")
    p.add_argument("--branch_dropout", type=float, default=None,
                   help="Dropout inside task branches. Default: use tcn_dropout.")

    # losses (shared for both heads)
    p.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal"])
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # data handling
    p.add_argument("--sort_by", type=str, default="window_idx", choices=["window_idx", "start_idx"])
    p.add_argument("--drop_na_rows", action="store_true",
                   help="If set, drop any segment rows that contain NaN/Inf.")
    p.add_argument("--impute_nan_with", type=float, default=0.0,
                   help="Used only if --drop_na_rows is NOT set.")
    p.add_argument("--min_segments", type=int, default=3,
                   help="Skip patients with fewer than this many segments after filtering.")
    p.add_argument("--no_zscore", action="store_true",
                   help="Disable train-fold z-score normalization of features.")

    args = p.parse_args()

    # ---- seed + device ----
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- split folds ----
    df = pd.read_csv(args.csv_path)
    train_df = df[df.fold != args.val_fold].copy()
    val_df = df[df.fold == args.val_fold].copy()

    out = args.output_dir / f"val_fold_{args.val_fold}"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)

    logging.info(
        f"Data cleaning: drop_na_rows={args.drop_na_rows}, "
        f"impute_nan_with={args.impute_nan_with}, "
        f"min_segments={args.min_segments}"
    )

    # ---- infer feature columns from first available TRAIN patient ----
    feature_cols = None
    for _, row in train_df.iterrows():
        pid = str(row["Patient ID"]).zfill(4)
        csvp = args.features_dir / pid / f"{pid}_segment_features.csv"
        if csvp.exists():
            tmp = pd.read_csv(csvp, nrows=20)
            if not tmp.empty:
                feature_cols = infer_feature_cols(tmp)
                break

    if feature_cols is None or len(feature_cols) == 0:
        raise RuntimeError("Could not infer any feature columns from TRAIN CSVs.")

    logging.info(f"Using {len(feature_cols)} feature columns.")
    (out / "feature_cols.txt").write_text("\n".join(feature_cols) + "\n")

    # ---- compute train-fold normalization stats (optional) ----
    mean = std = None
    if not args.no_zscore:
        mean, std = compute_train_feature_stats(
            train_df=train_df,
            features_dir=args.features_dir,
            feature_cols=feature_cols,
            sort_by=args.sort_by,
            pid_col="Patient ID",
            drop_na_rows=args.drop_na_rows,
            impute_nan_with=args.impute_nan_with,
        )
        np.save(out / "feature_mean.npy", mean)
        np.save(out / "feature_std.npy", std)
        logging.info("Saved train-fold feature mean/std to output dir.")
    else:
        logging.info("Z-score disabled (--no_zscore).")

    # ---- dataloaders ----
    train_loader = DataLoader(
        FeatureMILDataset(
            train_df,
            args.features_dir,
            feature_cols,
            mean=mean,
            std=std,
            sort_by=args.sort_by,
            pid_col="Patient ID",
            label_col="label",
            drop_na_rows=args.drop_na_rows,
            impute_nan_with=args.impute_nan_with,
            min_segments=args.min_segments,
        ),
        batch_size=1,
        shuffle=True,
        collate_fn=mil_collate_fn,
    )
    val_loader = DataLoader(
        FeatureMILDataset(
            val_df,
            args.features_dir,
            feature_cols,
            mean=mean,
            std=std,
            sort_by=args.sort_by,
            pid_col="Patient ID",
            label_col="label",
            drop_na_rows=args.drop_na_rows,
            impute_nan_with=args.impute_nan_with,
            min_segments=args.min_segments,
        ),
        batch_size=1,
        shuffle=False,
        collate_fn=mil_collate_fn,
    )

    # ---- model ----
    model = MILTCNFeatureMultiBranch(
        in_feat_dim=len(feature_cols),
        emb=args.embedding_dim,
        hid=args.tcn_hidden_dim,
        layers=args.tcn_layers,
        k=args.tcn_kernel_size,
        drop=args.tcn_dropout,
        attn=args.attn_dim,
        branch_hidden=args.branch_hidden,
        branch_drop=args.branch_dropout,
        enc_hidden=args.enc_hidden,
        enc_drop=args.enc_dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- class imbalance weights (pos_weight) from TRAIN ----
    y_scd_pos = int((train_df.label == 3).sum())
    y_pfd_pos = int((train_df.label == 6).sum())
    y_neg = int((train_df.label == 0).sum())

    w_scd = torch.tensor([y_neg / max(y_scd_pos, 1)], device=device, dtype=torch.float32)
    w_pfd = torch.tensor([y_neg / max(y_pfd_pos, 1)], device=device, dtype=torch.float32)

    if args.loss_type == "focal":
        crit_scd = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
        crit_pfd = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma).to(device)
        logging.info(
            f"Using FOCAL loss for BOTH heads "
            f"(alpha={args.focal_alpha}, gamma={args.focal_gamma})."
        )
    else:
        crit_scd = nn.BCEWithLogitsLoss(pos_weight=w_scd).to(device)
        crit_pfd = nn.BCEWithLogitsLoss(pos_weight=w_pfd).to(device)
        logging.info(
            f"Using BCEWithLogitsLoss for BOTH heads "
            f"(pos_weight_scd={float(w_scd.item()):.3f}, "
            f"pos_weight_pfd={float(w_pfd.item()):.3f})."
        )

    # ---- training loop ----
    best_mean_auc = -np.inf
    best_epoch = -1
    best_scd_auc = np.nan
    best_pfd_auc = np.nan

    metrics_rows = []

    for e in range(args.epochs):
        logging.info(f"========== Epoch {e+1}/{args.epochs} ==========")
        model.train()

        train_pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {e+1}",
            leave=False,
            total=len(train_loader),
            ncols=110,
        )

        running_loss = 0.0
        n_steps = 0
        skipped_train = 0

        for Xs, labels, _ in train_pbar:
            X = Xs[0].to(device)         # (N, F)
            y = labels[0].to(device)     # (2,)

            if X.numel() == 0 or X.size(0) < args.min_segments:
                skipped_train += 1
                continue

            z_scd, z_pfd, _, _ = model(X)

            loss_scd = crit_scd(z_scd.view(-1), y[0:1])
            loss_pfd = crit_pfd(z_pfd.view(-1), y[1:2])
            loss = loss_scd + loss_pfd

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            n_steps += 1
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", nseg=int(X.size(0)))

        mean_train_loss = running_loss / max(n_steps, 1)
        if skipped_train > 0:
            logging.info(f"[Train] Skipped {skipped_train} patients (<min_segments or missing/invalid features).")

        # ---- validation ----
        model.eval()
        yts, yps = [], []   # scd true/prob
        ytp, ypp = [], []   # pfd true/prob
        skipped_val = 0
        nanprob_val = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Val Epoch {e+1}",
            leave=False,
            total=len(val_loader),
            ncols=110,
        )

        with torch.no_grad():
            for Xs, labels, _ in val_pbar:
                X = Xs[0].to(device)
                if X.numel() == 0 or X.size(0) < args.min_segments:
                    skipped_val += 1
                    continue

                z_scd, z_pfd, _, _ = model(X)
                ps = torch.sigmoid(z_scd).item()
                pp = torch.sigmoid(z_pfd).item()

                if not (np.isfinite(ps) and np.isfinite(pp)):
                    nanprob_val += 1
                    continue

                yts.append(labels[0, 0].item())
                ytp.append(labels[0, 1].item())
                yps.append(ps)
                ypp.append(pp)

        if skipped_val > 0:
            logging.info(f"[Val] Skipped {skipped_val} patients (<min_segments or missing/invalid features).")
        if nanprob_val > 0:
            logging.warning(f"[Val] Dropped {nanprob_val} patients due to non-finite model probabilities.")

        if len(yts) == 0:
            logging.info("[Val] No valid patients after filtering; skipping metrics this epoch.")
            continue

        acc_s, p_s, r_s, f1_s, auc_s = eval_head_np(np.array(yts), np.array(yps))
        acc_p, p_p, r_p, f1_p, auc_p = eval_head_np(np.array(ytp), np.array(ypp))
        mean_auc = float(np.nanmean([auc_s, auc_p]))

        logging.info(
            f"Epoch {e+1} | "
            f"TrainLoss={mean_train_loss:.4f} | "
            f"SCD AUC={auc_s:.3f} F1={f1_s:.3f} | "
            f"PFD AUC={auc_p:.3f} F1={f1_p:.3f} | "
            f"MEAN AUC={mean_auc:.3f}"
        )

        metrics_rows.append(
            {
                "epoch": e + 1,
                "train_loss": mean_train_loss,
                "scd_acc": acc_s,
                "scd_prec": p_s,
                "scd_rec": r_s,
                "scd_f1": f1_s,
                "scd_auc": auc_s,
                "pfd_acc": acc_p,
                "pfd_prec": p_p,
                "pfd_rec": r_p,
                "pfd_f1": f1_p,
                "pfd_auc": auc_p,
                "mean_auc": mean_auc,
                "skipped_train": skipped_train,
                "skipped_val": skipped_val,
                "nanprob_val": nanprob_val,
            }
        )

        # ---- Save best by mean AUC ----
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_epoch = e + 1
            best_scd_auc = auc_s
            best_pfd_auc = auc_p

            # ckpt = {
            #     "model_state_dict": model.state_dict(),
            #     "val_fold": args.val_fold,
            #     "feature_cols": feature_cols,
            #     "feature_mean": mean,
            #     "feature_std": std,
            #     "args": vars(args),
            #     "best_epoch": best_epoch,
            #     "best_mean_auc": float(best_mean_auc),
            #     "best_scd_auc": float(best_scd_auc) if np.isfinite(best_scd_auc) else None,
            #     "best_pfd_auc": float(best_pfd_auc) if np.isfinite(best_pfd_auc) else None,
            #     "pos_weight_scd": float(w_scd.item()),
            #     "pos_weight_pfd": float(w_pfd.item()),
            # }
            # torch.save(ckpt, out / "mil_tcn_features_multibranch_best.pt")
            logging.info(f"New BEST model saved (mean AUC={mean_auc:.3f})")

    # ---- Save last checkpoint + metrics ----
    # ckpt_last = {
    #     "model_state_dict": model.state_dict(),
    #     "val_fold": args.val_fold,
    #     "feature_cols": feature_cols,
    #     "feature_mean": mean,
    #     "feature_std": std,
    #     "args": vars(args),
    #     "best_epoch": best_epoch,
    #     "best_mean_auc": float(best_mean_auc) if best_epoch != -1 else float("nan"),
    #     "best_scd_auc": float(best_scd_auc) if best_epoch != -1 and np.isfinite(best_scd_auc) else float("nan"),
    #     "best_pfd_auc": float(best_pfd_auc) if best_epoch != -1 and np.isfinite(best_pfd_auc) else float("nan"),
    #     "pos_weight_scd": float(w_scd.item()),
    #     "pos_weight_pfd": float(w_pfd.item()),
    # }
    # torch.save(ckpt_last, out / "mil_tcn_features_multibranch_last.pt")

    # =========================================================
    # Save per-fold summary (append to global CSV)
    # =========================================================
    
    fold_summary = {
        "val_fold": args.val_fold,
        "best_epoch": best_epoch,
        "best_mean_auc": float(best_mean_auc),
        "best_scd_auc": float(best_scd_auc) if np.isfinite(best_scd_auc) else np.nan,
        "best_pfd_auc": float(best_pfd_auc) if np.isfinite(best_pfd_auc) else np.nan,
    }
    
    # Add ALL hyperparameters used
    for k, v in vars(args).items():
        fold_summary[f"param_{k}"] = v
    
    global_csv = args.output_dir / "cv_results_all_folds.csv"
    
    df_row = pd.DataFrame([fold_summary])
    
    if global_csv.exists():
        df_row.to_csv(global_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(global_csv, index=False)
    
    logging.info("========================================")
    logging.info("FOLD COMPLETE")
    logging.info(f"Fold         : {args.val_fold}")
    logging.info(f"Best epoch   : {best_epoch}")
    logging.info(f"Best mean AUC: {best_mean_auc:.4f}")
    logging.info("========================================")

if __name__ == "__main__":
    main()
