#!/usr/bin/env python3
"""
train_mil_ecg_tcn.py

TCN-based MIL training for segmented ECGs with fold-aware validation.

TCN-based MIL training for segmented ECGs (MULTICLASS).

Classes:
0 = Survivor
3 = Sudden Cardiac Death (SCD)
6 = Pump Failure Death (PFD)

Single MIL-TCN backbone with two binary heads:
- SCD vs Survivor
- PFD vs Survivor

Workflow:
1. Preprocessed Holter ECGs are segmented into 30s windows
2. Each segment is encoded with a shared 1D AlexNet encoder
3. Segment embeddings are stacked in temporal order (N_segments × D)
4. A Temporal Convolutional Network (TCN) models long-range dependencies
5. Attention MIL pooling produces a patient-level embedding
6. A classifier predicts pathological vs healthy outcome
7. Save:
   - BEST model checkpoint (by validation AUC)
   - Patient embeddings (training only)
   - Per-epoch validation metrics (logs, incl. tuned threshold)
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
# Logging
# =========================================================
def setup_logging():
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_mil_ecg_tcn_multilabel_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================================================
# Dataset
# =========================================================
class ECGMILDataset(Dataset):
    def __init__(self, df, segments_dir):
        self.df = df.reset_index(drop=True)
        self.segments_dir = segments_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["Patient ID"]).zfill(4)
        label = int(row["label"])

        y_scd = 1 if label == 3 else 0
        y_pfd = 1 if label == 6 else 0

        seg_dir = self.segments_dir / pid
        files = sorted(seg_dir.glob("*.npy"))
        segments = torch.tensor(
            np.stack([np.load(f) for f in files]),
            dtype=torch.float32,
        )
        return segments, torch.tensor([y_scd, y_pfd], dtype=torch.float32), pid


def mil_collate_fn(batch):
    segs, labels, pids = zip(*batch)
    return segs, torch.stack(labels), pids

# =========================================================
# Model
# =========================================================
class AlexNet1DEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 11, stride=4, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Conv1d(64, 192, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Conv1d(192, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.net(x.unsqueeze(1)).squeeze(-1)
        return self.fc(x)


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
        X = T.squeeze(0).transpose(0, 1)
        a = torch.softmax(self.attn(X).squeeze(1), dim=0)
        z = (X * a.unsqueeze(1)).sum(dim=0)
        return z, a


class MILTCNMultiHead(nn.Module):
    def __init__(self, emb, hid, layers, k, drop, attn):
        super().__init__()
        self.encoder = AlexNet1DEncoder(emb)
        self.tcn = TCN(emb, hid, layers, k, drop)
        self.pool = AttentionMIL(hid, attn)
        self.head_scd = nn.Linear(hid, 1)
        self.head_pfd = nn.Linear(hid, 1)

    def forward(self, segments):
        H = self.encoder(segments)
        H = H.transpose(0, 1).unsqueeze(0)
        T = self.tcn(H)
        z, a = self.pool(T)
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
    p.add_argument("--segments_dir", type=Path, required=True)
    p.add_argument("--csv_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("mil_outputs"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--tcn_hidden_dim", type=int, default=256)
    p.add_argument("--tcn_layers", type=int, default=4)
    p.add_argument("--tcn_kernel_size", type=int, default=3)
    p.add_argument("--tcn_dropout", type=float, default=0.2)
    p.add_argument("--attn_dim", type=int, default=128)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    train_df = df[df.fold != args.val_fold]
    val_df = df[df.fold == args.val_fold]

    out = args.output_dir / f"val_fold_{args.val_fold}"
    out.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        ECGMILDataset(train_df, args.segments_dir),
        batch_size=1, shuffle=True, collate_fn=mil_collate_fn
    )
    val_loader = DataLoader(
        ECGMILDataset(val_df, args.segments_dir),
        batch_size=1, shuffle=False, collate_fn=mil_collate_fn
    )

    model = MILTCNMultiHead(
        args.embedding_dim, args.tcn_hidden_dim, args.tcn_layers,
        args.tcn_kernel_size, args.tcn_dropout, args.attn_dim
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # pos_weight per head
    y_scd = (train_df.label == 3).sum()
    y_pfd = (train_df.label == 6).sum()
    y_neg = (train_df.label == 0).sum()

    w_scd = torch.tensor([y_neg / max(y_scd, 1)], device=device)
    w_pfd = torch.tensor([y_neg / max(y_pfd, 1)], device=device)

    crit_scd = nn.BCEWithLogitsLoss(pos_weight=w_scd)
    crit_pfd = nn.BCEWithLogitsLoss(pos_weight=w_pfd)

    best_auc = -np.inf
    for e in range(args.epochs):
        logging.info(f"========== Epoch {e+1}/{args.epochs} ==========")
        model.train()
        train_pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {e+1}",
            leave=False,
            total=len(train_loader),
        )
        
        for segs, labels, _ in train_pbar:
            segs = segs[0].to(device)
            y = labels[0].to(device)
        
            z_scd, z_pfd, _, _ = model(segs)
            loss = crit_scd(z_scd, y[0:1]) + crit_pfd(z_pfd, y[1:2])
        
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ---- validation ----
        model.eval()
        yts, yps, ytp, ypp = [], [], [], []
        
        val_pbar = tqdm(
            val_loader,
            desc=f"Val Epoch {e+1}",
            leave=False,
            total=len(val_loader),
        )
        
        with torch.no_grad():
            for segs, labels, _ in val_pbar:
                segs = segs[0].to(device)
                z_scd, z_pfd, _, _ = model(segs)
        
                yts.append(labels[0, 0].item())
                ytp.append(labels[0, 1].item())
                yps.append(torch.sigmoid(z_scd).item())
                ypp.append(torch.sigmoid(z_pfd).item())

        acc_s, p_s, r_s, f1_s, auc_s, t_s = eval_head(np.array(yts), np.array(yps))
        acc_p, p_p, r_p, f1_p, auc_p, t_p = eval_head(np.array(ytp), np.array(ypp))
        mean_auc = np.nanmean([auc_s, auc_p])

        logging.info(
            f"Epoch {e+1} | "
            f"SCD AUC={auc_s:.3f} F1={f1_s:.3f} Thr={t_s:.3f} | "
            f"PFD AUC={auc_p:.3f} F1={f1_p:.3f} Thr={t_p:.3f}"
        )

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), out / "mil_tcn_multilabel_best.pt")
            logging.info(f"New BEST model saved (mean AUC={mean_auc:.3f})")

    logging.info("Training complete.")

if __name__ == "__main__":
    main()