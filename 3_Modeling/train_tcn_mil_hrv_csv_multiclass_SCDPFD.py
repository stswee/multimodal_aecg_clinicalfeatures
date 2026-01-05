#!/usr/bin/env python3
"""
train_mil_features_tcn.py

MIL + TCN training using per-segment engineered features (HRV/RR/PVC/morphology),
instead of raw ECG signal segments.

Classes:
0 = Survivor
3 = Sudden Cardiac Death (SCD)
6 = Pump Failure Death (PFD)

Two binary heads:
- SCD vs Survivor
- PFD vs Survivor

Input per patient:
- One CSV per patient: <features_dir>/<pid>/<pid>_segment_features.csv
  containing one row per segment/window.

Sequence formed by sorting rows by window_idx (or start_idx).
"""

import argparse
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
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
def setup_logging():
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_mil_features_tcn_{ts}.log"

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

def infer_feature_cols(df: pd.DataFrame, non_feature_cols=NON_FEATURE_COLS_DEFAULT):
    cols = [c for c in df.columns if c not in non_feature_cols]
    return cols

@torch.no_grad()
def compute_train_feature_stats(
    train_df: pd.DataFrame,
    features_dir: Path,
    feature_cols: list[str],
    pid_col: str = "Patient ID",
):
    """
    Compute mean/std over ALL segments from TRAIN patients only, without loading everything into memory.
    Uses sum/sumsq/count. Ignores NaNs per-feature.
    """
    sums = np.zeros(len(feature_cols), dtype=np.float64)
    sumsqs = np.zeros(len(feature_cols), dtype=np.float64)
    counts = np.zeros(len(feature_cols), dtype=np.int64)

    missing = 0
    total_patients = len(train_df)
    for _, row in tqdm(train_df.iterrows(), total=total_patients, desc="Compute train feature stats", ncols=100):
        pid = str(row[pid_col]).zfill(4)
        csv_path = features_dir / pid / f"{pid}_segment_features.csv"
        if not csv_path.exists():
            missing += 1
            continue

        df_feat = pd.read_csv(csv_path)
        if df_feat.empty:
            continue

        # Ensure the expected feature columns exist (skip if not)
        if any(c not in df_feat.columns for c in feature_cols):
            continue

        X = df_feat[feature_cols].to_numpy(dtype=np.float64, copy=False)

        # handle NaNs per feature
        mask = np.isfinite(X)
        # sum only finite entries
        sums += np.nansum(X, axis=0)
        sumsqs += np.nansum(X * X, axis=0)
        counts += mask.sum(axis=0)

    # avoid divide-by-zero
    counts_safe = np.maximum(counts, 1)
    mean = sums / counts_safe
    var = (sumsqs / counts_safe) - (mean * mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    if missing > 0:
        logging.info(f"[Stats] Missing feature CSV for {missing}/{total_patients} train patients.")

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
        sort_by: str = "window_idx",   # or "start_idx"
        pid_col: str = "Patient ID",
        label_col: str = "label",
        drop_na_rows: bool = True,
        impute_nan_with: float = 0.0,
        min_segments: int = 1,
    ):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir
        self.feature_cols = feature_cols
        self.mean = mean
        self.std = std
        self.sort_by = sort_by
        self.pid_col = pid_col
        self.label_col = label_col
        self.drop_na_rows = drop_na_rows
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

        csv_path = self.features_dir / pid / f"{pid}_segment_features.csv"
        if not csv_path.exists():
            # Return an "empty" bag — caller can skip by checking size
            X = torch.zeros((0, len(self.feature_cols)), dtype=torch.float32)
            y = torch.tensor([y_scd, y_pfd], dtype=torch.float32)
            return X, y, pid

        df_feat = pd.read_csv(csv_path)
        if df_feat.empty:
            X = torch.zeros((0, len(self.feature_cols)), dtype=torch.float32)
            y = torch.tensor([y_scd, y_pfd], dtype=torch.float32)
            return X, y, pid

        # Sort in temporal order
        if self.sort_by in df_feat.columns:
            df_feat = df_feat.sort_values(self.sort_by, ascending=True)

        # Keep only feature columns
        missing_cols = [c for c in self.feature_cols if c not in df_feat.columns]
        if missing_cols:
            # If schema mismatch, treat as empty
            X = torch.zeros((0, len(self.feature_cols)), dtype=torch.float32)
            y = torch.tensor([y_scd, y_pfd], dtype=torch.float32)
            return X, y, pid

        X = df_feat[self.feature_cols].to_numpy(dtype=np.float32, copy=False)

        # Optionally drop rows with any NaN/Inf
        if self.drop_na_rows:
            good = np.isfinite(X).all(axis=1)
            X = X[good]
        else:
            # Impute non-finite
            X = np.where(np.isfinite(X), X, self.impute_nan_with).astype(np.float32)

        # Enforce minimum segments
        if X.shape[0] < self.min_segments:
            X = np.zeros((0, len(self.feature_cols)), dtype=np.float32)

        # Normalize (train-fold stats)
        if self.mean is not None and self.std is not None and X.shape[0] > 0:
            X = (X - self.mean) / self.std

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor([y_scd, y_pfd], dtype=torch.float32)
        return X, y, pid


def mil_collate_fn(batch):
    Xs, ys, pids = zip(*batch)
    return Xs, torch.stack(ys), pids


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
        # x: (N, F)
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
        self.net = nn.Sequential(*[
            TemporalBlock(in_dim if i == 0 else hid, hid, k, 2**i, drop)
            for i in range(layers)
        ])

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
        X = T.squeeze(0).transpose(0, 1)     # (N, C)
        a = torch.softmax(self.attn(X).squeeze(1), dim=0)  # (N,)
        z = (X * a.unsqueeze(1)).sum(dim=0)  # (C,)
        return z, a


class MILTCNFeatureMultiHead(nn.Module):
    def __init__(self, in_feat_dim, emb, hid, layers, k, drop, attn, enc_hidden=128, enc_drop=0.1):
        super().__init__()
        self.encoder = FeatureEncoder(in_feat_dim, emb, hidden=enc_hidden, drop=enc_drop)
        self.tcn = TCN(emb, hid, layers, k, drop)
        self.pool = AttentionMIL(hid, attn)
        self.head_scd = nn.Linear(hid, 1)
        self.head_pfd = nn.Linear(hid, 1)

    def forward(self, X):
        """
        X: (N_segments, F)
        """
        H = self.encoder(X)                    # (N, emb)
        H = H.transpose(0, 1).unsqueeze(0)     # (1, emb, N)
        T = self.tcn(H)                        # (1, hid, N)
        z, a = self.pool(T)                    # z: (hid,)
        return self.head_scd(z), self.head_pfd(z), z, a


# =========================================================
# Evaluation
# =========================================================
def tune_threshold(y_true, y_prob):
    ts = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, y_prob >= t, zero_division=0) for t in ts]
    i = int(np.argmax(f1s))
    return ts[i], f1s[i]


def eval_head(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    t, f1 = tune_threshold(y_true, y_prob)
    y_pred = (y_prob >= t).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1b, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return acc, prec, rec, f1b, auc, t


# =========================================================
# Main
# =========================================================
def main():
    setup_logging()
    p = argparse.ArgumentParser()

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

    # model
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--tcn_hidden_dim", type=int, default=128)
    p.add_argument("--tcn_layers", type=int, default=4)
    p.add_argument("--tcn_kernel_size", type=int, default=3)
    p.add_argument("--tcn_dropout", type=float, default=0.2)
    p.add_argument("--attn_dim", type=int, default=128)

    # feature encoder
    p.add_argument("--enc_hidden", type=int, default=128)
    p.add_argument("--enc_dropout", type=float, default=0.1)

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

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    train_df = df[df.fold != args.val_fold].copy()
    val_df = df[df.fold == args.val_fold].copy()

    out = args.output_dir / f"val_fold_{args.val_fold}"
    out.mkdir(parents=True, exist_ok=True)

    # --- Infer feature columns from first available patient CSV in TRAIN ---
    feature_cols = None
    for _, row in train_df.iterrows():
        pid = str(row["Patient ID"]).zfill(4)
        csvp = args.features_dir / pid / f"{pid}_segment_features.csv"
        if csvp.exists():
            tmp = pd.read_csv(csvp, nrows=5)
            if not tmp.empty:
                feature_cols = infer_feature_cols(tmp)
                break
    if feature_cols is None:
        raise RuntimeError("Could not find any train patient feature CSVs to infer feature columns.")

    logging.info(f"Using {len(feature_cols)} feature columns: {feature_cols}")

    # --- Compute train-fold normalization stats (optional) ---
    mean = std = None
    if not args.no_zscore:
        mean, std = compute_train_feature_stats(train_df, args.features_dir, feature_cols)
        np.save(out / "feature_mean.npy", mean)
        np.save(out / "feature_std.npy", std)
        logging.info("Saved train-fold feature mean/std to output dir.")

    train_loader = DataLoader(
        FeatureMILDataset(
            train_df,
            args.features_dir,
            feature_cols,
            mean=mean, std=std,
            sort_by=args.sort_by,
            drop_na_rows=args.drop_na_rows,
            impute_nan_with=args.impute_nan_with,
            min_segments=args.min_segments,
        ),
        batch_size=1, shuffle=True, collate_fn=mil_collate_fn
    )
    val_loader = DataLoader(
        FeatureMILDataset(
            val_df,
            args.features_dir,
            feature_cols,
            mean=mean, std=std,
            sort_by=args.sort_by,
            drop_na_rows=args.drop_na_rows,
            impute_nan_with=args.impute_nan_with,
            min_segments=args.min_segments,
        ),
        batch_size=1, shuffle=False, collate_fn=mil_collate_fn
    )

    model = MILTCNFeatureMultiHead(
        in_feat_dim=len(feature_cols),
        emb=args.embedding_dim,
        hid=args.tcn_hidden_dim,
        layers=args.tcn_layers,
        k=args.tcn_kernel_size,
        drop=args.tcn_dropout,
        attn=args.attn_dim,
        enc_hidden=args.enc_hidden,
        enc_drop=args.enc_dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # pos_weight per head
    y_scd = int((train_df.label == 3).sum())
    y_pfd = int((train_df.label == 6).sum())
    y_neg = int((train_df.label == 0).sum())

    w_scd = torch.tensor([y_neg / max(y_scd, 1)], device=device)
    w_pfd = torch.tensor([y_neg / max(y_pfd, 1)], device=device)

    crit_scd = nn.BCEWithLogitsLoss(pos_weight=w_scd)
    crit_pfd = nn.BCEWithLogitsLoss(pos_weight=w_pfd)

    best_auc = -np.inf

    for e in range(args.epochs):
        logging.info(f"========== Epoch {e+1}/{args.epochs} ==========")
        model.train()

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {e+1}", leave=False, total=len(train_loader), ncols=100)
        for Xs, labels, _ in train_pbar:
            X = Xs[0].to(device)         # (N, F)
            y = labels[0].to(device)     # (2,)

            # skip empty bags
            if X.numel() == 0 or X.size(0) < args.min_segments:
                continue

            z_scd, z_pfd, _, _ = model(X)
            loss = crit_scd(z_scd, y[0:1]) + crit_pfd(z_pfd, y[1:2])

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_pbar.set_postfix(loss=f"{loss.item():.4f}", nseg=int(X.size(0)))

        # ---- validation ----
        model.eval()
        yts, yps, ytp, ypp = [], [], [], []
        skipped = 0

        val_pbar = tqdm(val_loader, desc=f"Val Epoch {e+1}", leave=False, total=len(val_loader), ncols=100)
        with torch.no_grad():
            for Xs, labels, _ in val_pbar:
                X = Xs[0].to(device)
                if X.numel() == 0 or X.size(0) < args.min_segments:
                    skipped += 1
                    continue

                z_scd, z_pfd, _, _ = model(X)

                yts.append(labels[0, 0].item())
                ytp.append(labels[0, 1].item())
                yps.append(torch.sigmoid(z_scd).item())
                ypp.append(torch.sigmoid(z_pfd).item())

        if skipped > 0:
            logging.info(f"[Val] Skipped {skipped} patients due to < min_segments or missing features.")

        acc_s, p_s, r_s, f1_s, auc_s, t_s = eval_head(np.array(yts), np.array(yps))
        acc_p, p_p, r_p, f1_p, auc_p, t_p = eval_head(np.array(ytp), np.array(ypp))
        mean_auc = np.nanmean([auc_s, auc_p])

        logging.info(
            f"Epoch {e+1} | "
            f"SCD AUC={auc_s:.3f} F1={f1_s:.3f} Thr={t_s:.3f} | "
            f"PFD AUC={auc_p:.3f} F1={f1_p:.3f} Thr={t_p:.3f} | "
            f"MEAN AUC={mean_auc:.3f}"
        )

        # Save best
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), out / "mil_tcn_features_best.pt")
            # Save feature schema for reproducibility
            (out / "feature_cols.txt").write_text("\n".join(feature_cols) + "\n")
            logging.info(f"New BEST model saved (mean AUC={mean_auc:.3f})")

    logging.info("Training complete.")


if __name__ == "__main__":
    main()
