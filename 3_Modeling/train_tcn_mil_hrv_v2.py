#!/usr/bin/env python3
"""
train_mil_hrv_tcn.py

TCN-based MIL training for per-window HRV feature CSVs
(with class-weighted loss + Top-K MIL pooling).

Workflow:
1. Each patient has a CSV of per-window HRV features (30s windows)
2. Each window is encoded with a shared MLP encoder
3. Window embeddings are stacked in temporal order (T × D)
4. A Temporal Convolutional Network (TCN) models long-range dependencies
5. Top-K MIL pooling produces a patient-level embedding
6. A classifier predicts pathological vs healthy outcome
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
# Label mapping (binary)
# =========================================================
LABEL_MAP = {0: 0, 3: 1, 6: 1}

# =========================================================
# Logging
# =========================================================
def setup_logging():
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_mil_hrv_tcn_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logging.info(f"Logging to {log_path}")
    return log_path


def save_args_to_txt(args, output_dir: Path):
    out_path = output_dir / "args_and_config.txt"
    with open(out_path, "w") as f:
        f.write("=== Experiment configuration ===\n\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    logging.info(f"Saved arguments to {out_path}")

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
class HRVMILDataset(Dataset):
    DROP_COLS = {"patient_id", "window_idx", "start_idx", "duration_sec"}

    def __init__(self, csv_df: pd.DataFrame, features_dir: Path):
        self.df = csv_df.reset_index(drop=True)
        self.features_dir = features_dir
        self.feature_cols = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["Patient ID"]).zfill(4)
        label = LABEL_MAP[int(row["label"])]

        csv_path = (
            self.features_dir
            / patient_id
            / f"{patient_id}_segment_features.csv"
        )
        if not csv_path.exists():
            raise RuntimeError(f"Missing HRV CSV for patient {patient_id}")

        df_feat = pd.read_csv(csv_path)

        if "window_idx" in df_feat.columns:
            df_feat = df_feat.sort_values("window_idx")

        if self.feature_cols is None:
            self.feature_cols = [
                c for c in df_feat.columns if c not in self.DROP_COLS
            ]
            logging.info(f"Using {len(self.feature_cols)} HRV features")

        X = torch.tensor(
            df_feat[self.feature_cols].values,
            dtype=torch.float32,
        )

        return X, label, patient_id


def mil_collate_fn(batch):
    Xs, labels, patient_ids = zip(*batch)
    return Xs, torch.tensor(labels), patient_ids

# =========================================================
# Models
# =========================================================
class WindowFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)

# ---------------- TCN ----------------
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
    def __init__(self, input_dim, hidden_dim, num_layers,
                 kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(
                TemporalBlock(
                    in_ch, hidden_dim,
                    kernel_size, dilation, dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MILTCNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        tcn_hidden_dim,
        tcn_layers,
        tcn_kernel_size,
        tcn_dropout,
        num_classes,
        topk,
    ):
        super().__init__()
        self.topk = topk
        self.encoder = WindowFeatureEncoder(input_dim, embedding_dim)
        self.tcn = TCN(
            embedding_dim,
            tcn_hidden_dim,
            tcn_layers,
            tcn_kernel_size,
            tcn_dropout,
        )
        self.classifier = nn.Linear(tcn_hidden_dim, num_classes)

    def forward(self, X):
        H = self.encoder(X)                   # (T, D)
        H = H.transpose(0, 1).unsqueeze(0)    # (1, D, T)
        T = self.tcn(H)                       # (1, Hdim, T)

        k = min(self.topk, T.shape[2])
        z = torch.topk(T, k=k, dim=2).values.mean(dim=2).squeeze(0)

        logits = self.classifier(z)
        return logits, z

# =========================================================
# Training / Evaluation
# =========================================================
def train_epoch(model, loader, optimizer, criterion, device,
                save_embeddings_dir, epoch, total_epochs):
    model.train()
    losses = []

    for Xs, labels, patient_ids in tqdm(
        loader, desc=f"Train {epoch+1}/{total_epochs}", leave=False
    ):
        optimizer.zero_grad()
        X = Xs[0].to(device)
        y = labels[0].to(device)

        logits, z = model(X)
        loss = criterion(logits.unsqueeze(0), y.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        np.save(save_embeddings_dir / f"{patient_ids[0]}.npy",
                z.detach().cpu().numpy())

    logging.info(f"Epoch {epoch+1} | Train loss={np.mean(losses):.4f}")


def evaluate_metrics(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for Xs, labels, _ in loader:
            X = Xs[0].to(device)
            logits, _ = model(X)
            prob = torch.softmax(logits, dim=0)[1].item()
            y_true.append(labels[0].item())
            y_prob.append(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    return acc, prec, rec, f1, auc

# =========================================================
# Main
# =========================================================
def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--tcn_hidden_dim", type=int, default=128)
    parser.add_argument("--tcn_layers", type=int, default=5)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=50)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    train_df = df[df["fold"] != args.val_fold]
    val_df = df[df["fold"] == args.val_fold]

    output_dir = args.output_dir / f"val_fold_{args.val_fold}"
    embeddings_dir = output_dir / "train_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(exist_ok=True)
    save_args_to_txt(args, output_dir)

    train_loader = DataLoader(
        HRVMILDataset(train_df, args.features_dir),
        batch_size=1, shuffle=True, collate_fn=mil_collate_fn
    )
    val_loader = DataLoader(
        HRVMILDataset(val_df, args.features_dir),
        batch_size=1, shuffle=False, collate_fn=mil_collate_fn
    )

    # Infer input feature dimension
    sample_X, _, _ = HRVMILDataset(train_df, args.features_dir)[0]
    input_dim = sample_X.shape[1]

    model = MILTCNModel(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        tcn_hidden_dim=args.tcn_hidden_dim,
        tcn_layers=args.tcn_layers,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dropout=args.tcn_dropout,
        num_classes=2,
        topk=args.topk,
    ).to(device)

    # -------- CLASS-WEIGHTED LOSS --------
    train_labels = train_df["label"].map(LABEL_MAP).values
    counts = np.bincount(train_labels)
    weights = counts.sum() / counts
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_auc = -np.inf
    for epoch in range(args.epochs):
        train_epoch(
            model, train_loader, optimizer, criterion,
            device, embeddings_dir, epoch, args.epochs
        )
        acc, prec, rec, f1, auc = evaluate_metrics(
            model, val_loader, device
        )

        logging.info(
            f"Epoch {epoch+1} | Acc={acc:.3f} | "
            f"Prec={prec:.3f} | Rec={rec:.3f} | "
            f"F1={f1:.3f} | AUC={auc:.3f}"
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(
                model.state_dict(),
                output_dir / "mil_hrv_tcn_model_best_auc.pt"
            )
            logging.info(f"New BEST model saved (AUC={auc:.3f})")

    logging.info(f"Training complete. Best AUC={best_auc:.3f}")


if __name__ == "__main__":
    main()
