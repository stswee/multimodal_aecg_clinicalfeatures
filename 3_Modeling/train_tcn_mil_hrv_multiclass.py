#!/usr/bin/env python3
"""
train_mil_ecg_tcn.py

TCN-based MIL training for segmented ECGs with fold-aware validation.

TCN-based MIL training for segmented ECGs (MULTICLASS).

Classes:
0 = Survivor
3 = Sudden Cardiac Death (SCD)
6 = Pump Failure Death (PFD)

UPDATED:
- Replace mean pooling with Attention MIL pooling (smooth, stable episode selection)
- Use class-weighted CrossEntropyLoss (computed from training fold)
- Tune decision threshold on validation to maximize F1 (and log tuned metrics)

Differences vs binary version:
- Multiclass CrossEntropyLoss (3 classes)
- No decision-threshold tuning
- Metrics: macro-F1, per-class PRF, multiclass AUC (OvR)

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
)

# =========================================================
# Label mapping (multiclass)
# =========================================================
LABEL_MAP = {0: 0, 3: 1, 6: 2}
INV_LABEL_MAP = {0: "Survivor", 1: "SCD", 2: "PFD"}
NUM_CLASSES = 3

# =========================================================
# Logging
# =========================================================
def setup_logging():
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_mil_ecg_tcn_multiclass_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_path}")
    return log_path


def save_args_to_txt(args, output_dir: Path):
    with open(output_dir / "args_and_config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================================================
# Dataset
# =========================================================
class ECGMILDataset(Dataset):
    """One item = one patient (bag of ECG segments)"""

    def __init__(self, csv_df: pd.DataFrame, segments_dir: Path):
        self.df = csv_df.reset_index(drop=True)
        self.segments_dir = segments_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["Patient ID"]).zfill(4)
        label = LABEL_MAP[int(row["label"])]

        patient_dir = self.segments_dir / patient_id
        segment_files = sorted(patient_dir.glob("*.npy"))
        if len(segment_files) == 0:
            raise RuntimeError(f"No segments for patient {patient_id}")

        segments = torch.tensor(
            np.stack([np.load(f) for f in segment_files]),
            dtype=torch.float32,
        )

        return segments, label, patient_id


def mil_collate_fn(batch):
    segments, labels, patient_ids = zip(*batch)
    return segments, torch.tensor(labels), patient_ids

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


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        )

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                TemporalBlock(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    2 ** i,
                    dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AttentionMIL(nn.Module):
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, T_seq):
        X = T_seq.squeeze(0).transpose(0, 1)   # (N, H)
        a = torch.softmax(self.attn(X).squeeze(1), dim=0)
        z = torch.sum(X * a.unsqueeze(1), dim=0)
        return z, a


class MILTCNModel(nn.Module):
    def __init__(
        self, embedding_dim, tcn_hidden_dim, tcn_layers,
        tcn_kernel_size, tcn_dropout, num_classes, attn_dim
    ):
        super().__init__()
        self.encoder = AlexNet1DEncoder(embedding_dim)
        self.tcn = TCN(
            embedding_dim, tcn_hidden_dim,
            tcn_layers, tcn_kernel_size, tcn_dropout
        )
        self.pool = AttentionMIL(tcn_hidden_dim, attn_dim)
        self.classifier = nn.Linear(tcn_hidden_dim, num_classes)

    def forward(self, segments):
        H = self.encoder(segments)
        H = H.transpose(0, 1).unsqueeze(0)
        T_seq = self.tcn(H)
        z, a = self.pool(T_seq)
        logits = self.classifier(z)
        return logits, z, a

# =========================================================
# Training / Evaluation
# =========================================================
def train_epoch(model, loader, optimizer, criterion, device, emb_dir, epoch, total):
    model.train()
    losses = []

    for segments, labels, patient_ids in tqdm(loader, leave=False):
        optimizer.zero_grad()
        segments = segments[0].to(device)
        label = labels[0].to(device)

        logits, z, _ = model(segments)
        loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        np.save(emb_dir / f"{patient_ids[0]}.npy", z.detach().cpu().numpy())

    logging.info(f"Epoch {epoch+1}/{total} | Train loss={np.mean(losses):.4f}")


def evaluate_metrics(model, loader, device):
    model.eval()
    y_true, y_prob, y_pred = [], [], []

    with torch.no_grad():
        for segments, labels, _ in loader:
            segments = segments[0].to(device)
            logits, _, _ = model(segments)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            pred = int(np.argmax(probs))

            y_true.append(labels[0].item())
            y_pred.append(pred)
            y_prob.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1)

    auc = roc_auc_score(
        y_true, y_prob, multi_class="ovr", average="macro"
    ) if len(np.unique(y_true)) > 1 else float("nan")

    return acc, macro_f1, auc, prec, rec, f1

# =========================================================
# Main
# =========================================================
def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--segments_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("mil_outputs"))

    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--tcn_hidden_dim", type=int, default=256)
    parser.add_argument("--tcn_layers", type=int, default=4)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.2)
    parser.add_argument("--attn_dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    train_df = df[df["fold"] != args.val_fold]
    val_df = df[df["fold"] == args.val_fold]

    out_dir = args.output_dir / f"val_fold_{args.val_fold}"
    emb_dir = out_dir / "train_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(exist_ok=True)
    save_args_to_txt(args, out_dir)

    train_loader = DataLoader(
        ECGMILDataset(train_df, args.segments_dir),
        batch_size=1, shuffle=True, collate_fn=mil_collate_fn
    )
    val_loader = DataLoader(
        ECGMILDataset(val_df, args.segments_dir),
        batch_size=1, shuffle=False, collate_fn=mil_collate_fn
    )

    model = MILTCNModel(
        args.embedding_dim, args.tcn_hidden_dim, args.tcn_layers,
        args.tcn_kernel_size, args.tcn_dropout, NUM_CLASSES, args.attn_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_labels = train_df["label"].map(LABEL_MAP).values
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    logging.info(f"Train class counts: {counts.tolist()}")
    logging.info(f"CE weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_auc = -np.inf
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, criterion,
                    device, emb_dir, epoch, args.epochs)

        acc, macro_f1, auc, prec, rec, f1 = evaluate_metrics(
            model, val_loader, device
        )

        logging.info(
            f"Epoch {epoch+1} | Acc={acc:.3f} | MacroF1={macro_f1:.3f} | AUC={auc:.3f}"
        )
        for i, name in INV_LABEL_MAP.items():
            logging.info(
                f"  {name}: Prec={prec[i]:.3f} | Rec={rec[i]:.3f} | F1={f1[i]:.3f}"
            )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), out_dir / "mil_tcn_best_auc.pt")
            logging.info(f"New BEST model saved (AUC={auc:.3f})")

    logging.info(f"Training complete. Best AUC={best_auc:.3f}")


if __name__ == "__main__":
    main()