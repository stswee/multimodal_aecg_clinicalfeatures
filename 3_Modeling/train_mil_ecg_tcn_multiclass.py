#!/usr/bin/env python3
"""
train_mil_ecg_tcn_attention_multiclass.py

TCN + Temporal Attention MIL training for segmented ECGs.

Workflow:
1. Holter ECG → 30s segments
2. Segment encoder (AlexNet1D)
3. Temporal modeling (TCN)
4. Temporal attention pooling
5. Patient-level classification:
     0 = Healthy
     1 = Sudden Cardiac Death (SCD)
     2 = Pump Failure Death (PFD)
6. Class-weighted loss
7. Save:
   - Best model (by multiclass AUC)
   - Patient embeddings (train only)
   - Logs
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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# =========================================================
# Labels
# =========================================================
LABEL_MAP = {0: 0, 3: 1, 6: 2}
NUM_CLASSES = 3

# =========================================================
# Logging
# =========================================================
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_tcn_attention_multiclass_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_path}")

# =========================================================
# Reproducibility
# =========================================================
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
        label = LABEL_MAP[int(row["label"])]

        seg_files = sorted((self.segments_dir / pid).glob("*.npy"))
        if len(seg_files) == 0:
            raise RuntimeError(f"No segments for patient {pid}")

        segments = torch.tensor(
            np.stack([np.load(f) for f in seg_files]),
            dtype=torch.float32,
        )
        return segments, label, pid

def mil_collate_fn(batch):
    segments, labels, pids = zip(*batch)
    return segments, torch.tensor(labels), pids

# =========================================================
# Models
# =========================================================
class AlexNet1DEncoder(nn.Module):
    def __init__(self, emb_dim):
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
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x).squeeze(-1)
        return self.fc(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, drop):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k,
                              padding=(k - 1) * d,
                              dilation=d)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.conv(x)[:, :, :x.size(2)]
        y = self.drop(self.relu(y))
        res = x if self.down is None else self.down(x)
        return y + res

class TCN(nn.Module):
    def __init__(self, in_dim, hid_dim, layers, k, drop):
        super().__init__()
        blocks = []
        for i in range(layers):
            blocks.append(
                TemporalBlock(
                    in_dim if i == 0 else hid_dim,
                    hid_dim,
                    k,
                    2 ** i,
                    drop,
                )
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, T):
        # T: (H, N)
        scores = self.attn(T.transpose(0, 1))  # (N,1)
        alpha = torch.softmax(scores.squeeze(-1), dim=0)  # (N,)
        z = torch.sum(T * alpha.unsqueeze(0), dim=1)
        return z, alpha

class MILTCNAttentionModel(nn.Module):
    def __init__(self, emb_dim, hid_dim, layers, k, drop, num_classes):
        super().__init__()
        self.encoder = AlexNet1DEncoder(emb_dim)
        self.tcn = TCN(emb_dim, hid_dim, layers, k, drop)
        self.attn = TemporalAttention(hid_dim)
        self.cls = nn.Linear(hid_dim, num_classes)

    def forward(self, segments):
        H = self.encoder(segments)           # (N,D)
        H = H.transpose(0, 1).unsqueeze(0)   # (1,D,N)
        T = self.tcn(H).squeeze(0)           # (Hdim,N)
        z, alpha = self.attn(T)
        logits = self.cls(z)
        return logits, z, alpha

# =========================================================
# Metrics
# =========================================================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for segs, labels, _ in loader:
            segs = segs[0].to(device)
            logits, _, _ = model(segs)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            y_true.append(labels[0].item())
            y_prob.append(probs)

    y_true = np.array(y_true)
    y_prob = np.vstack(y_prob)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    auc = roc_auc_score(pd.get_dummies(y_true), y_prob, multi_class="ovr")
    return acc, prec, rec, f1, auc

# =========================================================
# Main
# =========================================================
def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--segments_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    train_df = df[df["fold"] != args.val_fold]
    val_df = df[df["fold"] == args.val_fold]

    # ----- class weights -----
    y_train = train_df["label"].map(LABEL_MAP).values
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    weights = len(y_train) / (NUM_CLASSES * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logging.info(f"Class weights: {weights}")

    train_loader = DataLoader(
        ECGMILDataset(train_df, args.segments_dir),
        batch_size=1, shuffle=True, collate_fn=mil_collate_fn
    )
    val_loader = DataLoader(
        ECGMILDataset(val_df, args.segments_dir),
        batch_size=1, shuffle=False, collate_fn=mil_collate_fn
    )

    model = MILTCNAttentionModel(
        args.emb_dim, args.hid_dim,
        args.layers, args.kernel,
        args.dropout, NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_auc = -np.inf
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for segs, labels, _ in pbar:
            segs = segs[0].to(device)
            label = labels[0].to(device)

            optimizer.zero_grad()
            logits, _, _ = model(segs)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss))

        acc, prec, rec, f1, auc = evaluate(model, val_loader, device)
        logging.info(
            f"Epoch {epoch+1} | Acc={acc:.3f} | "
            f"Prec={prec:.3f} | Rec={rec:.3f} | "
            f"F1={f1:.3f} | AUC={auc:.3f}"
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "best_tcn_attention_multiclass.pt")
            logging.info("New BEST model saved")

    logging.info(f"Training complete. Best AUC={best_auc:.3f}")

if __name__ == "__main__":
    main()
