#!/usr/bin/env python3
"""
Hierarchical MIL training for Holter ECGs using ECGFounder backbone.
"""

import argparse
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# =========================================================
# Constants
# =========================================================
LABEL_MAP = {0: 0, 3: 1, 6: 1}
ECGFOUNDER_INPUT_LENGTH = 5000  # 10 sec @ 500 Hz

# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_ecgfounder_mil_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info(f"Logging to {log_path}")

# =========================================================
# Dataset
# =========================================================
class ECGMILDataset(Dataset):
    def __init__(self, df: pd.DataFrame, segments_dir: Path):
        self.df = df.reset_index(drop=True)
        self.segments_dir = segments_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row["Patient ID"]).zfill(4)
        label = LABEL_MAP[int(row["label"])]

        files = sorted((self.segments_dir / pid).glob("*.npy"))
        if len(files) == 0:
            raise RuntimeError(f"No segments for patient {pid}")

        segments = torch.tensor(
            np.stack([np.load(f) for f in files]), dtype=torch.float32
        )
        return segments, label, pid


def mil_collate_fn(batch):
    segments, labels, pids = zip(*batch)
    return segments, torch.tensor(labels), pids

# =========================================================
# Net1D (ECGFounder backbone)
# =========================================================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MyConv1dPadSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=0, groups=groups, bias=False
        )

    def forward(self, x):
        L = x.shape[-1]
        out_L = (L + self.stride - 1) // self.stride
        pad = max(0, (out_L - 1) * self.stride + self.kernel_size - L)
        x = F.pad(x, (pad // 2, pad - pad // 2))
        return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r),
            nn.ReLU(),
            nn.Linear(ch // r, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        s = x.mean(-1)
        w = self.fc(s).view(b, c, 1)
        return x * w


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, groups, downsample):
        super().__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act = Swish()
        self.conv1 = MyConv1dPadSame(in_ch, out_ch, kernel_size, stride, groups)

        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = MyConv1dPadSame(out_ch, out_ch, kernel_size, 1, groups)

        self.se = SEBlock(out_ch)

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(x))
        out = self.conv1(out)
        out = self.act(self.bn2(out))
        out = self.conv2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return out + identity


class Net1D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_filters=64,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
    ):
        super().__init__()

        self.first = nn.Sequential(
            MyConv1dPadSame(in_channels, base_filters, kernel_size, 2),
            nn.BatchNorm1d(base_filters),
            Swish(),
        )

        self.stages = nn.ModuleList()
        in_ch = base_filters

        for out_ch, n_blocks in zip(filter_list, m_blocks_list):
            groups = out_ch // groups_width if out_ch % groups_width == 0 else 1
            stage = []
            curr_in = in_ch

            for i in range(n_blocks):
                s = stride if i == 0 else 1
                down = None
                if i == 0 and (s > 1 or curr_in != out_ch):
                    down = nn.Sequential(
                        MyConv1dPadSame(curr_in, out_ch, 1, s),
                        nn.BatchNorm1d(out_ch),
                    )

                stage.append(
                    BasicBlock1D(curr_in, out_ch, kernel_size, s, groups, down)
                )
                curr_in = out_ch

            self.stages.append(nn.Sequential(*stage))
            in_ch = out_ch

        self.final = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            Swish(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding_dim = in_ch

    def get_embedding(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.first(x)
        for s in self.stages:
            out = s(out)
        out = self.final(out).squeeze(-1)
        return out

# =========================================================
# ECGFounder Encoder
# =========================================================
class ECGFounderEncoder(nn.Module):
    def __init__(self, weights_path: Optional[Path], freeze=True):
        super().__init__()
        self.net = Net1D()
        self.embedding_dim = self.net.embedding_dim
        self.freeze = freeze

        if weights_path is not None:
            sd = torch.load(weights_path, map_location="cpu")
            sd = sd.get("model", sd)
            self.net.load_state_dict(sd, strict=False)

        if freeze:
            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()

    def forward(self, x):
        return self.net.get_embedding(x)

# =========================================================
# MIL
# =========================================================
class GatedAttention(nn.Module):
    def __init__(self, d, a=256):
        super().__init__()
        self.V = nn.Linear(d, a)
        self.U = nn.Linear(d, a)
        self.w = nn.Linear(a, 1)

    def forward(self, H):
        A = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H)))
        A = torch.softmax(A, dim=0)
        z = (A * H).sum(0)
        return z, A.squeeze()


class HierarchicalMILModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(encoder.embedding_dim, 512)
        self.attn = GatedAttention(512)
        self.cls = nn.Linear(512, 2)

    def preprocess(self, segs):
        device = segs.device
        segs = segs.squeeze(1).cpu().numpy()
        windows = []
        for s in segs:
            r = resample_poly(s, 5, 2)
            for i in range(3):
                w = r[i*5000:(i+1)*5000]
                w = (w - w.mean()) / (w.std() + 1e-6)
                windows.append(w)
        return torch.tensor(windows, device=device)

    def forward(self, segs, encode_batch_size=128):
        windows = self.preprocess(segs)

        embs = []
        for i in range(0, len(windows), encode_batch_size):
            batch = windows[i:i+encode_batch_size]
            with torch.no_grad() if self.encoder.freeze else torch.enable_grad():
                embs.append(self.encoder(batch))
        embs = torch.cat(embs, dim=0)

        chunks = embs.view(-1, 50, embs.size(-1)).mean(1)
        chunks = self.proj(chunks)
        z, A = self.attn(chunks)
        return self.cls(z), z, A

# =========================================================
# Loss
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--segments_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--ecgfounder_weights", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--encode_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir / f"val_fold_{args.val_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    df = pd.read_csv(args.csv_path)
    train_df = df[df.fold != args.val_fold]
    val_df = df[df.fold == args.val_fold]

    train_loader = DataLoader(
        ECGMILDataset(train_df, args.segments_dir),
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mil_collate_fn,
    )

    encoder = ECGFounderEncoder(args.ecgfounder_weights, freeze=True)
    model = HierarchicalMILModel(encoder).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )
    criterion = FocalLoss()
    scaler = GradScaler()

    best_auc = -1.0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        for i, (segs, y, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            segs = segs[0].to(device)
            y = y.to(device)

            with autocast():
                logits, _, _ = model(segs, args.encode_batch_size)
                loss = criterion(logits.unsqueeze(0), y) / args.accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        logging.info(f"Epoch {epoch+1} complete")

    torch.save(model.state_dict(), output_dir / "mil_model_final.pt")


if __name__ == "__main__":
    main()
