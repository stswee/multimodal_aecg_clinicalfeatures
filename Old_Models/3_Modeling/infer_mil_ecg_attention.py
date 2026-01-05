#!/usr/bin/env python3
"""
infer_mil_ecg_attention.py

Inference-only script for MIL ECG attention model.

Runs the trained model on the validation fold and:
- Outputs per-patient predictions
- Saves attention weights per patient
- Computes validation metrics

Assumes model was trained with train_mil_ecg_attention.py
and best weights are saved as mil_model_best_auc.pt

Example usage:
python infer_mil_ecg_attention.py  --val_fold 0 --segments_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/preprocessed_segments --csv_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/music_patient_folds_5cv.csv --model_path ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_outputs/val_fold_0/mil_model_best_auc.pt --output_dir ../../../../local3/sswee/music_download/physionet.org/files/music-sudden-cardiac-death/1.0.1/mil_outputs/val_fold_0_inference --save_attention
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
    confusion_matrix,
)

# =========================================================
# Label mapping (must match training)
# =========================================================
LABEL_MAP = {0: 0, 3: 1, 6: 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

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
        raw_label = int(row["label"])
        label = LABEL_MAP[raw_label]

        patient_dir = self.segments_dir / patient_id
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
# Models (must match training exactly)
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
        x = self.features(x)
        x = x.squeeze(-1)
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
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_attention", action="store_true")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------
    # Load validation split
    # -----------------------------------------------------
    df = pd.read_csv(args.csv_path)
    val_df = df[df["fold"] == args.val_fold]

    dataset = ECGMILDataset(val_df, args.segments_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mil_collate_fn
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    model = MILAttentionModel(
        args.embedding_dim,
        args.attention_dim,
        args.num_classes
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # -----------------------------------------------------
    # Output folders
    # -----------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_attention:
        attn_dir = args.output_dir / "attention_maps"
        attn_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------
    y_true, y_pred, y_prob = [], [], []
    rows = []

    with torch.no_grad():
        for segments, labels, patient_ids in tqdm(loader, desc="Inference"):
            segments = segments[0].to(device)
            label = labels[0].item()
            pid = patient_ids[0]

            logits, _, A = model(segments)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            pred = np.argmax(probs)

            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(probs)

            rows.append({
                "patient_id": pid,
                "true_label": INV_LABEL_MAP[label],
                "pred_label": INV_LABEL_MAP[pred],
                "prob_survivor": probs[0],
                "prob_scd": probs[1],
                "prob_pump_failure": probs[2],
            })

            if args.save_attention:
                np.save(attn_dir / f"{pid}_attention.npy", A.cpu().numpy())

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    y_true_clin = [INV_LABEL_MAP[y] for y in y_true]
    y_pred_clin = [INV_LABEL_MAP[y] for y in y_pred]

    acc = accuracy_score(y_true_clin, y_pred_clin)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_clin, y_pred_clin, average="macro"
    )
    auc = roc_auc_score(
        pd.get_dummies(y_true_clin),
        np.vstack(y_prob),
        multi_class="ovr"
    )

    print("\nValidation metrics:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro Precision: {prec:.3f}")
    print(f"Macro Recall: {rec:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(f"AUC (OvR): {auc:.3f}")

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------
    out_csv = args.output_dir / "validation_predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\nSaved predictions to: {out_csv}")


if __name__ == "__main__":
    main()
