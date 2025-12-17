#!/usr/bin/env python3
"""
infer_mil_ecg_attention_binary.py

Validation / inference script for binary MIL ECG attention model.

- Loads best checkpoint
- Runs inference on validation fold
- Uses data-driven decision threshold (Youden's J)
- Saves per-patient predictions
- Optionally saves attention weights

Usage:
python infer_mil_ecg_attention_binary_testing.py   --val_fold 0   --segments_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments   --csv_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv   --model_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_outputs_binary_emb128_thresh/val_fold_0/mil_model_best_auc.pt   --output_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_outputs_binary_emb128_thresh/val_fold_0_inference   --save_attention --embedding_dim 128
"""

import argparse
from pathlib import Path
import logging

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
# Label mapping (MUST match training)
# =========================================================
LABEL_MAP = {0: 0, 3: 1, 6: 1}

# =========================================================
# Dataset
# =========================================================
class ECGMILDataset(Dataset):
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
            dtype=torch.float32
        )
        return segments, label, patient_id


def mil_collate_fn(batch):
    segments, labels, patient_ids = zip(*batch)
    return segments, torch.tensor(labels), patient_ids


# =========================================================
# Models (EXACT match to training)
# =========================================================
class AlexNet1DEncoder(nn.Module):
    def __init__(self, embedding_dim):
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
        x = x.unsqueeze(1)
        x = self.features(x).squeeze(-1)
        return self.fc(x)


class GatedAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
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
        H = self.encoder(segments)
        A = self.attention(H)
        z = torch.sum(A * H, dim=0)
        logits = self.classifier(z)
        return logits, z, A.squeeze()


# =========================================================
# Threshold selection
# =========================================================
def select_threshold_youden(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.median(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--segments_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_attention", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load validation split
    df = pd.read_csv(args.csv_path)
    val_df = df[df["fold"] == args.val_fold]

    loader = DataLoader(
        ECGMILDataset(val_df, args.segments_dir),
        batch_size=1,
        shuffle=False,
        collate_fn=mil_collate_fn
    )

    # Load model
    model = MILAttentionModel(
        args.embedding_dim, args.attention_dim, args.num_classes
    ).to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_attention:
        attn_dir = args.output_dir / "attention_maps"
        attn_dir.mkdir(exist_ok=True)

    # ---------------- Inference ----------------
    y_true, y_prob, rows = [], [], []

    with torch.no_grad():
        for segments, labels, patient_ids in tqdm(loader, desc="Inference"):
            segments = segments[0].to(device)
            label = labels[0].item()
            pid = patient_ids[0]

            logits, _, A = model(segments)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

            y_true.append(label)
            y_prob.append(probs[1])

            rows.append({
                "patient_id": pid,
                "true_label": label,
                "prob_survivor": probs[0],
                "prob_pathological": probs[1],
            })

            if args.save_attention:
                np.save(attn_dir / f"{pid}_attention.npy", A.cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # ---------------- Thresholding ----------------
    threshold = select_threshold_youden(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # ---------------- Metrics ----------------
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\nValidation metrics (binary):")
    print(f"AUC      : {auc:.3f}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1       : {f1:.3f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"% Predicted Positive: {100*y_pred.mean():.2f}%")
    print("\nConfusion matrix:")
    print(cm)

    print(
        f"\nProb range: "
        f"min={y_prob.min():.4f}, "
        f"median={np.median(y_prob):.4f}, "
        f"max={y_prob.max():.4f}"
    )

    # Save CSV
    for i, r in enumerate(rows):
        r["pred_label"] = int(y_pred[i])

    out_csv = args.output_dir / "validation_predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved predictions to: {out_csv}")


if __name__ == "__main__":
    main()
