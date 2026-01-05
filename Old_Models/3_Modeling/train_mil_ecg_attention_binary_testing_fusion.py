#!/usr/bin/env python3
"""
train_mil_ecg_attention_binary_late_fusion.py

Late-fusion MIL for ECG:
- Time-domain segments
- Frequency-domain segments

Fusion happens AFTER MIL pooling:
    z = concat(z_time, z_freq)
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
    roc_curve,
    confusion_matrix,
)

# =========================================================
# Label mapping (binary)
# =========================================================
LABEL_MAP = {0: 0, 3: 1, 6: 1}

# =========================================================
# Logging / reproducibility
# =========================================================
def setup_logging():
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"train_late_fusion_{datetime.now():%Y%m%d_%H%M%S}.log"

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
# Dataset (TIME + FREQ)
# =========================================================
class ECGMILLateFusionDataset(Dataset):
    def __init__(self, df, segments_time_dir, segments_freq_dir):
        self.df = df.reset_index(drop=True)
        self.time_dir = segments_time_dir
        self.freq_dir = segments_freq_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["Patient ID"]).zfill(4)
        label = LABEL_MAP[int(row["label"])]

        time_files = sorted((self.time_dir / pid).glob("*.npy"))
        freq_files = sorted((self.freq_dir / pid).glob("*.npy"))

        if len(time_files) != len(freq_files):
            raise RuntimeError(f"Segment mismatch for patient {pid}")

        time_segments = torch.tensor(
            np.stack([np.load(f) for f in time_files]), dtype=torch.float32
        )
        freq_segments = torch.tensor(
            np.stack([np.load(f) for f in freq_files]), dtype=torch.float32
        )

        return time_segments, freq_segments, label, pid


def mil_collate_fn(batch):
    t, f, y, pid = zip(*batch)
    return t, f, torch.tensor(y), pid

# =========================================================
# Models
# =========================================================
class AlexNet1DEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.features = nn.Sequential(
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
        x = x.unsqueeze(1)
        x = self.features(x).squeeze(-1)
        return self.fc(x)


class GatedAttention(nn.Module):
    def __init__(self, D, A):
        super().__init__()
        self.V = nn.Linear(D, A)
        self.U = nn.Linear(D, A)
        self.w = nn.Linear(A, 1)

    def forward(self, H):
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))
        return torch.softmax(self.w(A), dim=0)


class LateFusionMILAttentionModel(nn.Module):
    def __init__(self, embedding_dim, attention_dim, num_classes):
        super().__init__()

        self.encoder_t = AlexNet1DEncoder(embedding_dim)
        self.attn_t = GatedAttention(embedding_dim, attention_dim)

        self.encoder_f = AlexNet1DEncoder(embedding_dim)
        self.attn_f = GatedAttention(embedding_dim, attention_dim)

        self.classifier = nn.Linear(2 * embedding_dim, num_classes)

    def forward(self, seg_t, seg_f):
        Ht = self.encoder_t(seg_t)
        At = self.attn_t(Ht)
        zt = torch.sum(At * Ht, dim=0)

        Hf = self.encoder_f(seg_f)
        Af = self.attn_f(Hf)
        zf = torch.sum(Af * Hf, dim=0)

        z = torch.cat([zt, zf], dim=0)
        logits = self.classifier(z)

        return logits, z, At.squeeze(), Af.squeeze()

# =========================================================
# Loss
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

# =========================================================
# Validation utilities
# =========================================================
def select_threshold_youden(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.median(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]

def evaluate_metrics(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for seg_t, seg_f, y, _ in loader:
            seg_t = seg_t[0].to(device)
            seg_f = seg_f[0].to(device)
            y = y.item()

            logits, _, _, _ = model(seg_t, seg_f)
            prob = torch.softmax(logits.unsqueeze(0), dim=1)[0, 1].item()

            y_true.append(y)
            y_prob.append(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    threshold = select_threshold_youden(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "pct_positive": 100.0 * y_pred.mean(),
        "prob_min": y_prob.min(),
        "prob_med": np.median(y_prob),
        "prob_max": y_prob.max(),
    }

# =========================================================
# Main
# =========================================================
def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--segments_time_dir", type=Path, required=True)
    parser.add_argument("--segments_freq_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("mil_late_fusion"))
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir / f"val_fold_{args.val_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    train_df = df[df.fold != args.val_fold]
    val_df = df[df.fold == args.val_fold]

    train_loader = DataLoader(
        ECGMILLateFusionDataset(train_df, args.segments_time_dir, args.segments_freq_dir),
        batch_size=1, shuffle=True, collate_fn=mil_collate_fn
    )

    val_loader = DataLoader(
        ECGMILLateFusionDataset(val_df, args.segments_time_dir, args.segments_freq_dir),
        batch_size=1, shuffle=False, collate_fn=mil_collate_fn
    )

    model = LateFusionMILAttentionModel(
        args.embedding_dim, args.attention_dim, num_classes=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss()

    best_auc = -np.inf
    best_f1 = -np.inf

    for epoch in range(args.epochs):
        model.train()
        for seg_t, seg_f, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            seg_t = seg_t[0].to(device)
            seg_f = seg_f[0].to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _, _, _ = model(seg_t, seg_f)
            loss = criterion(logits.unsqueeze(0), y)
            loss.backward()
            optimizer.step()

        metrics = evaluate_metrics(model, val_loader, device)

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"VAL Acc={metrics['acc']:.3f} | "
            f"Prec={metrics['prec']:.3f} | "
            f"Rec={metrics['rec']:.3f} | "
            f"F1={metrics['f1']:.3f} | "
            f"AUC={metrics['auc']:.3f}"
        )
        logging.info(
            f"Threshold={metrics['threshold']:.4f} | "
            f"% Predicted Positive={metrics['pct_positive']:.2f}%"
        )
        logging.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        logging.info(
            f"Prob range: min={metrics['prob_min']:.4f}, "
            f"median={metrics['prob_med']:.4f}, "
            f"max={metrics['prob_max']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), output_dir / "mil_model_best_auc.pt")
            logging.info(
                f"New BEST-AUC model saved at epoch {epoch+1} "
                f"(AUC={best_auc:.3f})"
            )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), output_dir / "mil_model_best_f1.pt")
            logging.info(
                f"New BEST-F1 model saved at epoch {epoch+1} "
                f"(F1={best_f1:.3f}, "
                f"Prec={metrics['prec']:.3f}, "
                f"Rec={metrics['rec']:.3f})"
            )

    logging.info(
        f"Training complete. Best AUC={best_auc:.3f}, Best F1={best_f1:.3f}"
    )

if __name__ == "__main__":
    main()
