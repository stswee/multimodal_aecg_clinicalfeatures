#!/usr/bin/env python3
"""
train_mil_ecg_attention_binary_testing.py

Attention-based MIL training for segmented ECGs with fold-aware validation.

Workflow:
1. Encode ECG segments with a shared 1D CNN encoder
2. Stack segment embeddings per patient (N x D)
3. Attention pooling -> patient embedding z
4. Classify patient outcome
5. Save:
   - BEST model checkpoint only (by validation AUC)
   - Patient embeddings (training only)
   - Per-epoch validation metrics (logs)
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
# 0 = survivor
# 1 = pathological (SCD + pump failure)
# =========================================================
LABEL_MAP = {0: 0, 3: 1, 6: 1}
INV_LABEL_MAP = {0: 0, 1: "pathological"}

# =========================================================
# Logging
# =========================================================
def setup_logging():
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_mil_ecg_attention_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
    )

    logging.info(f"Logging to {log_path}")
    return log_path

def save_args_to_txt(args, output_dir: Path):
    """
    Save all arguments and relevant config to a text file
    for reproducibility.
    """
    out_path = output_dir / "args_and_config.txt"

    with open(out_path, "w") as f:
        f.write("=== Experiment configuration ===\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Script: {Path(__file__).name}\n\n")

        f.write("---- Command-line arguments ----\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- Label mapping ----\n")
        for k, v in LABEL_MAP.items():
            f.write(f"{k} -> {v}\n")

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
class ECGMILDataset(Dataset):
    """One item = one patient (bag of ECG segments)"""

    def __init__(self, csv_df: pd.DataFrame, segments_dir: Path):
        self.df = csv_df.reset_index(drop=True)
        self.segments_dir = segments_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Zero-pad patient ID (e.g., 0693)
        patient_id = str(row["Patient ID"]).zfill(4)

        # Remap labels {0,3,6} -> {0,1,2}
        raw_label = int(row["label"])
        label = LABEL_MAP[raw_label]

        patient_dir = self.segments_dir / patient_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        segment_files = sorted(patient_dir.glob("*.npy"))
        if len(segment_files) == 0:
            raise RuntimeError(f"No segments found for patient {patient_id}")

        segments = torch.tensor(
            np.stack([np.load(f) for f in segment_files]),
            dtype=torch.float32
        )

        return segments, label, patient_id


def mil_collate_fn(batch):
    segments, labels, patient_ids = zip(*batch)
    return segments, torch.tensor(labels), patient_ids


# =========================================================
# Models
# =========================================================
class AlexNet1DEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),

            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),

            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = x.unsqueeze(1)      # (N, 1, L)
        x = self.features(x)
        x = x.squeeze(-1)       # (N, 256)
        return self.fc(x)       # (N, D)


class GatedAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        self.V = nn.Linear(embedding_dim, attention_dim)
        self.U = nn.Linear(embedding_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)

    def forward(self, H):
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))
        A = self.w(A)
        return torch.softmax(A, dim=0)


class MILAttentionModel(nn.Module):
    def __init__(self, embedding_dim, attention_dim, num_classes):
        super().__init__()
        self.encoder = AlexNet1DEncoder(embedding_dim)
        self.attention = GatedAttention(embedding_dim, attention_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, segments):
        H = self.encoder(segments)   # (N, D)
        A = self.attention(H)        # (N, 1)
        z = torch.sum(A * H, dim=0)  # (D,)
        logits = self.classifier(z)
        return logits, z, A.squeeze()


# =========================================================
# Training / Evaluation
# =========================================================
def train_epoch(model, loader, optimizer, criterion, device,
                save_embeddings_dir, epoch, total_epochs):
    model.train()
    losses = []

    for segments, labels, patient_ids in tqdm(
        loader,
        desc=f"Train | Epoch {epoch+1}/{total_epochs}",
        total=len(loader),
        ncols=100,
        leave=False,
    ):
        optimizer.zero_grad()
        segments = segments[0].to(device)
        label = labels[0].to(device)

        logits, z, _ = model(segments)
        loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Save training embedding
        np.save(
            save_embeddings_dir / f"{patient_ids[0]}.npy",
            z.detach().cpu().numpy()
        )

    mean_loss = float(np.mean(losses))
    logging.info(f"Epoch {epoch+1}/{total_epochs} | Train loss: {mean_loss:.4f}")
    return mean_loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)  # prob of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
        
def select_threshold_youden(y_true, y_prob):
    """
    Select threshold that maximizes Youden's J statistic (TPR - FPR).
    Falls back to median probability if ROC is degenerate.
    """
    # Need both classes present
    if len(np.unique(y_true)) < 2:
        return np.median(y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return np.median(y_prob)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

def evaluate_metrics(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for segments, labels, _ in loader:
            segments = segments[0].to(device)
            label = labels[0].item()

            logits, _, _ = model(segments)

            # safer softmax handling
            probs = torch.softmax(
                logits.unsqueeze(0), dim=1
            ).squeeze(0).cpu().numpy()

            y_true.append(label)
            y_prob.append(probs[1])  # P(pathological)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # -------- Threshold selection --------
    threshold = select_threshold_youden(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # -------- Metrics --------
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)

    # -------- Diagnostics --------
    cm = confusion_matrix(y_true, y_pred)
    pct_positive = 100.0 * y_pred.mean()

    prob_min = float(y_prob.min())
    prob_med = float(np.median(y_prob))
    prob_max = float(y_prob.max())

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "pct_positive": pct_positive,
        "confusion_matrix": cm,
        "prob_min": prob_min,
        "prob_med": prob_med,
        "prob_max": prob_max,
    }

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
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    logging.info(f"Arguments: {vars(args)}")

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
        ECGMILDataset(train_df, args.segments_dir),
        batch_size=1, shuffle=True, collate_fn=mil_collate_fn
    )
    val_loader = DataLoader(
        ECGMILDataset(val_df, args.segments_dir),
        batch_size=1, shuffle=False, collate_fn=mil_collate_fn
    )

    model = MILAttentionModel(
        args.embedding_dim, args.attention_dim, args.num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # Track best validation AUC
    best_auc = -np.inf
    best_f1 = -np.inf

    for epoch in tqdm(range(args.epochs), desc="Epochs", ncols=80):
        train_epoch(
            model, train_loader, optimizer, criterion,
            device, embeddings_dir, epoch, args.epochs
        )

        metrics = evaluate_metrics(model, val_loader, device)
        acc = metrics["acc"]
        prec = metrics["prec"]
        rec = metrics["rec"]
        f1 = metrics["f1"]
        auc = metrics["auc"]
        threshold = metrics["threshold"]
        pct_positive = metrics["pct_positive"]
        cm = metrics["confusion_matrix"]

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"VAL Acc={acc:.3f} | "
            f"Prec={prec:.3f} | "
            f"Rec={rec:.3f} | "
            f"F1={f1:.3f} | "
            f"AUC={auc:.3f}"
        )
        
        logging.info(
            f"Threshold={threshold:.4f} | "
            f"% Predicted Positive={pct_positive:.2f}%"
        )
        
        logging.info(
            f"Confusion Matrix:\n{cm}"
        )

        logging.info(
            "Binary label counts (0=survivor, 1=pathological):\n"
            f"{train_df['label'].map(LABEL_MAP).value_counts()}"
        )

        logging.info(
            f"Prob range: min={metrics['prob_min']:.4f}, "
            f"median={metrics['prob_med']:.4f}, "
            f"max={metrics['prob_max']:.4f}"
        )

        # -------- Save best-by-AUC --------
        if auc > best_auc:
            best_auc = auc
            torch.save(
                model.state_dict(),
                output_dir / "mil_model_best_auc.pt"
            )
            logging.info(
                f"New BEST-AUC model saved at epoch {epoch+1} "
                f"(AUC={auc:.3f})"
            )
        
        # -------- Save best-by-F1 --------
        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                model.state_dict(),
                output_dir / "mil_model_best_f1.pt"
            )
            logging.info(
                f"New BEST-F1 model saved at epoch {epoch+1} "
                f"(F1={f1:.3f}, Prec={prec:.3f}, Rec={rec:.3f})"
            )

    logging.info(
        f"Training complete. "
        f"Best AUC = {best_auc:.3f} (mil_model_best_auc.pt), "
        f"Best F1 = {best_f1:.3f} (mil_model_best_f1.pt)"
    )

if __name__ == "__main__":
    main()
