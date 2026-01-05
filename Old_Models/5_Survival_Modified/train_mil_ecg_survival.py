#!/usr/bin/env python3
"""
train_temporal_mil_ecg_survival_scd_pfd.py

Temporal (ORDER-AWARE) MIL training for 24h Holter ECG using 30s segments (~2700/patient),
with EXPLAINABILITY via temporal attention (highlights important segments).

Upgrades vs your current script:
- Preserves temporal order (segments treated as a sequence, not a set)
- Temporal modeling over segment embeddings (TCN)
- Cause-specific explainability + prediction:
    * SCD cause-specific Cox head
    * PFD cause-specific Cox head
  (shared encoder + shared TCN, separate attention+heads per cause)
- Segment importance output:
    * attention weights per segment (per cause)
    * top-k segment indices saved for interpretability

Training objective:
- Sum of 2 cause-specific Cox partial likelihood losses:
    L = L_scd + L_pfd
  Competing events are treated as censored (cause-specific hazards).

Expected CSV columns (minimum):
- Patient ID
- fold
- time_to_event_days
And either:
A) Cause of death  (preferred; values include 0, 3=SCD, 6=PFD), OR
B) event_scd, event_pfd (binary 0/1 columns)

Segments directory structure:
segments_dir/
  0001/*.npy
  0002/*.npy
  ...
Each *.npy is a 30s 1D signal (shape: (L,)). Files must be sortable in time order.

Example usage:
python train_temporal_mil_ecg_survival_scd_pfd.py \
  --val_fold 0 \
  --segments_dir /path/to/preprocessed_segments \
  --csv_path /path/to/labels_folds.csv \
  --output_dir /path/to/outputs \
  --epochs 20 \
  --patient_batch_size 2 \
  --encode_chunk_size 128 \
  --embedding_dim 256 \
  --tcn_hidden 256 \
  --tcn_levels 7 \
  --tcn_kernel 3 \
  --lr 1e-4

Notes:
- With ~2700 segments, set patient_batch_size small (1–4) and use encode_chunk_size for speed/memory.
- This script keeps per-patient forward passes (simple, robust).
  If you later want maximum throughput, we can implement flatten/split batching across patients.
"""

import argparse
import random
import logging
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional for log-rank cutpoint (same as your script)
from lifelines.statistics import logrank_test


# =========================================================
# Logging
# =========================================================
def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_temporal_mil_survival_{timestamp}.log"

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
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Script: {Path(__file__).name}\n\n")

        f.write("---- Command-line arguments ----\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- Label logic ----\n")
        f.write("Cause-specific hazards:\n")
        f.write("  event_scd = 1 if Cause of death == 3 else 0\n")
        f.write("  event_pfd = 1 if Cause of death == 6 else 0\n")
        f.write("  Competing events treated as censored at time_to_event_days.\n")

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
class ECGTemporalDatasetSurvival(Dataset):
    """
    One item = one patient:
      - ordered segments tensor: (T, L)
      - time: scalar
      - event_scd: scalar {0,1}
      - event_pfd: scalar {0,1}
      - patient_id: str
    """

    def __init__(self, csv_df: pd.DataFrame, segments_dir: Path):
        self.df = csv_df.reset_index(drop=True)
        self.segments_dir = segments_dir

        required = {"Patient ID", "time_to_event_days"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        # We will derive events either from Cause of death OR from event_scd/event_pfd.
        has_cause = "Cause of death" in self.df.columns
        has_events = ("event_scd" in self.df.columns) and ("event_pfd" in self.df.columns)
        if not (has_cause or has_events):
            raise ValueError(
                "CSV must contain either 'Cause of death' or both 'event_scd' and 'event_pfd'."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["Patient ID"]).zfill(4)

        patient_dir = self.segments_dir / patient_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        # IMPORTANT: assumes filename sort corresponds to temporal order
        segment_files = sorted(patient_dir.glob("*.npy"))
        if len(segment_files) == 0:
            raise RuntimeError(f"No segments found for patient {patient_id}")

        segments_np = np.stack([np.load(f) for f in segment_files])  # (T, L)
        segments = torch.tensor(segments_np, dtype=torch.float32)

        t = float(row["time_to_event_days"])
        time = torch.tensor(t, dtype=torch.float32)

        if "Cause of death" in row.index:
            cause = int(row["Cause of death"])
            event_scd = 1.0 if cause == 3 else 0.0
            event_pfd = 1.0 if cause == 6 else 0.0
        else:
            event_scd = float(row["event_scd"])
            event_pfd = float(row["event_pfd"])

        event_scd = torch.tensor(event_scd, dtype=torch.float32)
        event_pfd = torch.tensor(event_pfd, dtype=torch.float32)

        return segments, time, event_scd, event_pfd, patient_id


def temporal_collate_fn(batch):
    segments_list, times, es, ep, pids = zip(*batch)
    return list(segments_list), torch.stack(times), torch.stack(es), torch.stack(ep), list(pids)


# =========================================================
# Models
# =========================================================
class AlexNet1DEncoder(nn.Module):
    """
    Same encoder idea as your script.
    Input:  (Nseg, L)
    Output: (Nseg, D)
    """

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
        # x: (Nseg, L)
        x = x.unsqueeze(1)      # (Nseg, 1, L)
        x = self.features(x)    # (Nseg, 256, 1)
        x = x.squeeze(-1)       # (Nseg, 256)
        return self.fc(x)       # (Nseg, D)


class TCNResidualBlock(nn.Module):
    """
    Temporal Conv residual block over sequence embeddings.
    Input/Output: (1, C, T) per patient (we keep it simple)
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding (left pad)
        self.pad = pad
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (1, C, T)
        # Causal padding on the left
        x_in = x
        x = torch.nn.functional.pad(x, (self.pad, 0))
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)

        x = torch.nn.functional.pad(x, (self.pad, 0))
        x = self.conv2(x)
        x = self.act(x)
        x = self.drop(x)

        return x + x_in


class TemporalTCN(nn.Module):
    """
    Simple TCN stack.
    Input:  (T, D)
    Output: (T, H)
    """
    def __init__(self, in_dim: int, hidden_dim: int, levels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        blocks = []
        for i in range(levels):
            dilation = 2 ** i
            blocks.append(TCNResidualBlock(hidden_dim, kernel_size, dilation, dropout))
        self.blocks = nn.ModuleList(blocks)

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, E):
        # E: (T, D)
        H = self.in_proj(E)       # (T, H)
        H = H.transpose(0, 1).unsqueeze(0)  # (1, H, T)

        for blk in self.blocks:
            H = blk(H)  # (1, H, T)

        H = H.squeeze(0).transpose(0, 1)  # (T, H)
        H = self.out_norm(H)
        return H


class TemporalAttention(nn.Module):
    """
    Produces attention weights over time steps.
    Input:  H (T, Hdim)
    Output: alpha (T,) and patient embedding z (Hdim,)
    """
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.v = nn.Linear(hidden_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, H):
        # H: (T, Hdim)
        s = torch.tanh(self.v(H))         # (T, attn_dim)
        s = self.w(s).squeeze(-1)         # (T,)
        alpha = torch.softmax(s, dim=0)   # (T,)
        z = torch.sum(alpha.unsqueeze(-1) * H, dim=0)  # (Hdim,)
        return alpha, z


class TemporalMILSurvivalModel(nn.Module):
    """
    Shared:
      - segment encoder (AlexNet1D)
      - temporal TCN
    Cause-specific:
      - attention + risk head for SCD
      - attention + risk head for PFD
    """

    def __init__(
        self,
        embedding_dim: int,
        tcn_hidden: int,
        tcn_levels: int,
        tcn_kernel: int,
        tcn_dropout: float,
        attn_dim: int,
        encode_chunk_size: int,
    ):
        super().__init__()
        self.encoder = AlexNet1DEncoder(embedding_dim)
        self.tcn = TemporalTCN(
            in_dim=embedding_dim,
            hidden_dim=tcn_hidden,
            levels=tcn_levels,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
        )

        self.attn_scd = TemporalAttention(hidden_dim=tcn_hidden, attn_dim=attn_dim)
        self.attn_pfd = TemporalAttention(hidden_dim=tcn_hidden, attn_dim=attn_dim)

        self.head_scd = nn.Linear(tcn_hidden, 1)
        self.head_pfd = nn.Linear(tcn_hidden, 1)

        self.encode_chunk_size = encode_chunk_size

    def encode_segments(self, segments):
        """
        segments: (T, L) float32
        Returns: E (T, D)
        Uses chunking to keep memory stable and improve throughput.
        """
        T = segments.shape[0]
        chunks = []
        for x in torch.split(segments, self.encode_chunk_size, dim=0):
            chunks.append(self.encoder(x))  # (chunk, D)
        return torch.cat(chunks, dim=0)     # (T, D)

    def forward(self, segments):
        """
        segments: (T, L)
        Returns:
          risk_scd, risk_pfd (scalars)
          alpha_scd, alpha_pfd (T,)
        """
        E = self.encode_segments(segments)  # (T, D)
        H = self.tcn(E)                     # (T, Hdim)

        alpha_scd, z_scd = self.attn_scd(H)
        alpha_pfd, z_pfd = self.attn_pfd(H)

        risk_scd = self.head_scd(z_scd).squeeze()
        risk_pfd = self.head_pfd(z_pfd).squeeze()
        return risk_scd, risk_pfd, alpha_scd, alpha_pfd


# =========================================================
# Survival loss and metrics
# =========================================================
class CoxPHLoss(nn.Module):
    """
    Negative Cox partial log-likelihood (mini-batch approximation).
    """
    def forward(self, risks, times, events):
        # risks: (B,), times: (B,), events: (B,) in {0,1}
        order = torch.argsort(times, descending=True)
        risks = risks[order]
        events = events[order]

        log_cumsum = torch.logcumsumexp(risks, dim=0)
        per_event = (risks - log_cumsum) * events
        denom = events.sum().clamp_min(1.0)
        return -per_event.sum() / denom


def harrell_c_index(risks, times, events):
    """
    Harrell's C-index. Higher risk => shorter time.
    O(n^2) OK for ~1k.
    """
    risks = np.asarray(risks, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)

    concordant = 0.0
    permissible = 0.0
    n = len(times)

    for i in range(n):
        if events[i] != 1:
            continue
        for j in range(n):
            if times[i] < times[j]:
                permissible += 1
                if risks[i] > risks[j]:
                    concordant += 1
                elif risks[i] == risks[j]:
                    concordant += 0.5

    return (concordant / permissible) if permissible > 0 else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, topk_save_dir: Path | None = None, topk: int = 20):
    """
    Computes cause-specific C-index for SCD and PFD.
    Optionally saves top-k segment indices by attention per patient (SCD & PFD).
    """
    model.eval()
    scd_risks, pfd_risks = [], []
    times_all, es_all, ep_all = [], [], []

    if topk_save_dir is not None:
        topk_save_dir.mkdir(parents=True, exist_ok=True)
        out_rows = []

    for segments_list, times, es, ep, pids in loader:
        for i in range(len(segments_list)):
            seg = segments_list[i].to(device)
            risk_scd, risk_pfd, a_scd, a_pfd = model(seg)

            scd_risks.append(float(risk_scd.item()))
            pfd_risks.append(float(risk_pfd.item()))
            times_all.append(float(times[i].item()))
            es_all.append(float(es[i].item()))
            ep_all.append(float(ep[i].item()))

            if topk_save_dir is not None:
                # top-k indices by attention
                a_scd_np = a_scd.detach().cpu().numpy()
                a_pfd_np = a_pfd.detach().cpu().numpy()

                k = min(topk, len(a_scd_np))
                top_scd = np.argsort(-a_scd_np)[:k].tolist()
                top_pfd = np.argsort(-a_pfd_np)[:k].tolist()

                out_rows.append({
                    "patient_id": pids[i],
                    "T": int(len(a_scd_np)),
                    "topk": int(k),
                    "top_scd_idx": json.dumps(top_scd),
                    "top_scd_w": json.dumps([float(a_scd_np[j]) for j in top_scd]),
                    "top_pfd_idx": json.dumps(top_pfd),
                    "top_pfd_w": json.dumps([float(a_pfd_np[j]) for j in top_pfd]),
                })

    c_scd = harrell_c_index(scd_risks, times_all, es_all)
    c_pfd = harrell_c_index(pfd_risks, times_all, ep_all)

    if topk_save_dir is not None:
        pd.DataFrame(out_rows).to_csv(topk_save_dir / "topk_attention_segments.csv", index=False)

    return {
        "val_cindex_scd": float(c_scd),
        "val_cindex_pfd": float(c_pfd),
        "val_cindex_mean": float(np.nanmean([c_scd, c_pfd])),
        "val_n": int(len(times_all)),
        "val_events_scd": int(np.sum(np.array(es_all) == 1.0)),
        "val_events_pfd": int(np.sum(np.array(ep_all) == 1.0)),
    }


def truncate_at_horizon(times, events, horizon):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    t = np.minimum(times, horizon)
    e = events & (times <= horizon)
    return t, e


def find_best_cutpoint_logrank(risks, times, events, horizon=1460, min_frac=0.10, grid_size=200):
    times, events = truncate_at_horizon(times, events, horizon)
    n = len(risks)
    qs = np.linspace(min_frac, 1.0 - min_frac, grid_size)
    cuts = np.unique(np.quantile(risks, qs))

    best = {"cutoff": None, "chi2": -np.inf, "p_value": None, "n_low": None, "n_high": None}

    for c in cuts:
        high = risks >= c
        n_high = int(high.sum())
        n_low = int((~high).sum())
        if n_high < min_frac * n or n_low < min_frac * n:
            continue

        res = logrank_test(
            times[high], times[~high],
            event_observed_A=events[high],
            event_observed_B=events[~high],
        )
        if res.test_statistic > best["chi2"]:
            best.update({
                "cutoff": float(c),
                "chi2": float(res.test_statistic),
                "p_value": float(res.p_value),
                "n_low": n_low,
                "n_high": n_high,
            })
    return best


# =========================================================
# Training
# =========================================================
def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    losses = []

    for segments_list, times, es, ep, _ in tqdm(
        loader,
        desc=f"Train | Epoch {epoch+1}/{total_epochs}",
        total=len(loader),
        ncols=100,
        leave=False,
    ):
        optimizer.zero_grad()

        batch_risk_scd, batch_risk_pfd = [], []
        for seg in segments_list:
            seg = seg.to(device)
            r_scd, r_pfd, _, _ = model(seg)
            batch_risk_scd.append(r_scd)
            batch_risk_pfd.append(r_pfd)

        batch_risk_scd = torch.stack(batch_risk_scd)          # (B,)
        batch_risk_pfd = torch.stack(batch_risk_pfd)          # (B,)
        times = times.to(device)
        es = es.to(device)
        ep = ep.to(device)

        loss_scd = criterion(batch_risk_scd, times, es)
        loss_pfd = criterion(batch_risk_pfd, times, ep)
        loss = loss_scd + loss_pfd

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    logging.info(f"Epoch {epoch+1}/{total_epochs} | Train loss (SCD+PFD Cox): {mean_loss:.4f}")
    return mean_loss


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--segments_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("temporal_mil_survival_outputs"))

    # Model
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--tcn_hidden", type=int, default=256)
    parser.add_argument("--tcn_levels", type=int, default=7, help="Number of dilation levels. 7 => receptive field covers large context.")
    parser.add_argument("--tcn_kernel", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.10)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--encode_chunk_size", type=int, default=128, help="How many segments to encode at once (memory/throughput tradeoff).")

    # Train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patient_batch_size", type=int, default=2, help="Patients per batch. Use small values for 24h Holters.")
    parser.add_argument("--num_workers", type=int, default=0)

    # Explainability saving
    parser.add_argument("--save_val_topk", action="store_true", help="If set, saves per-patient top-k attention segments on validation set each epoch.")
    parser.add_argument("--topk", type=int, default=20)

    # Post-training cutpoint (optional)
    parser.add_argument("--posthoc_cutpoint", action="store_true", help="If set, computes log-rank cutpoint on TRAIN risks (mean of SCD/PFD risks).")
    parser.add_argument("--cut_horizon_days", type=int, default=1460)

    args = parser.parse_args()

    fold_out = args.output_dir / f"val_fold_{args.val_fold}"
    fold_out.mkdir(parents=True, exist_ok=True)
    setup_logging(fold_out)

    logging.info(f"Arguments: {vars(args)}")
    save_args_to_txt(args, fold_out)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    df = pd.read_csv(args.csv_path)

    df["Patient ID"] = (
        df["Patient ID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)  # remove trailing .0
        .str.zfill(4)
    )

    needed = {"Patient ID", "fold", "time_to_event_days"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    train_df = df[df["fold"] != args.val_fold].copy()
    val_df = df[df["fold"] == args.val_fold].copy()

    train_loader = DataLoader(
        ECGTemporalDatasetSurvival(train_df, args.segments_dir),
        batch_size=args.patient_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=temporal_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        ECGTemporalDatasetSurvival(val_df, args.segments_dir),
        batch_size=args.patient_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=temporal_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = TemporalMILSurvivalModel(
        embedding_dim=args.embedding_dim,
        tcn_hidden=args.tcn_hidden,
        tcn_levels=args.tcn_levels,
        tcn_kernel=args.tcn_kernel,
        tcn_dropout=args.tcn_dropout,
        attn_dim=args.attn_dim,
        encode_chunk_size=args.encode_chunk_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CoxPHLoss().to(device)

    metrics_csv = fold_out / "epoch_metrics.csv"
    metrics_rows = []
    best_score = -np.inf

    logging.info("Starting training (temporal TCN + attention, cause-specific Cox)...")
    for epoch in tqdm(range(args.epochs), desc="Epochs", ncols=80):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)

        topk_dir = None
        if args.save_val_topk:
            topk_dir = fold_out / "val_topk" / f"epoch_{epoch+1:03d}"

        val_metrics = evaluate(model, val_loader, device, topk_save_dir=topk_dir, topk=args.topk)

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"VAL C-index SCD={val_metrics['val_cindex_scd']:.4f} | "
            f"PFD={val_metrics['val_cindex_pfd']:.4f} | "
            f"MEAN={val_metrics['val_cindex_mean']:.4f} | "
            f"Events SCD={val_metrics['val_events_scd']}/{val_metrics['val_n']} | "
            f"PFD={val_metrics['val_events_pfd']}/{val_metrics['val_n']}"
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        }
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

        # Save best by mean C-index
        score = val_metrics["val_cindex_mean"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            torch.save(model.state_dict(), fold_out / "model_best_mean_cindex.pt")
            logging.info(f"New BEST model saved at epoch {epoch+1} (mean C-index={score:.4f})")

    logging.info(
        f"Training complete. Best mean validation C-index = {best_score:.4f} "
        f"(model_best_mean_cindex.pt). Metrics log: {metrics_csv}"
    )

    # =========================================================
    # Post-training: log-rank cutpoint on TRAIN (optional)
    # =========================================================
    if args.posthoc_cutpoint:
        logging.info("Posthoc: estimating data-driven cutoff on training set (mean risk = (SCD+PFD)/2)...")

        ckpt = fold_out / "model_best_mean_cindex.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        train_risks_mean, train_times, train_events_any = [], [], []

        with torch.no_grad():
            for segments_list, times, es, ep, _ in train_loader:
                for i in range(len(segments_list)):
                    seg = segments_list[i].to(device)
                    r_scd, r_pfd, _, _ = model(seg)
                    r = 0.5 * (float(r_scd.item()) + float(r_pfd.item()))
                    train_risks_mean.append(r)
                    train_times.append(float(times[i].item()))
                    # Any cardiac event (SCD or PFD) as "event" for logrank cutpoint
                    train_events_any.append(bool((es[i].item() == 1.0) or (ep[i].item() == 1.0)))

        train_risks_mean = np.asarray(train_risks_mean, dtype=float)
        train_times = np.asarray(train_times, dtype=float)
        train_events_any = np.asarray(train_events_any, dtype=bool)

        # Ensure direction higher = higher risk (flip if negative correlation w/ event indicator)
        corr = np.corrcoef(train_risks_mean, train_events_any.astype(int))[0, 1]
        if np.isfinite(corr) and corr < 0:
            logging.info("Flipping mean risk sign so higher = higher risk")
            train_risks_mean = -train_risks_mean

        cut_info = find_best_cutpoint_logrank(
            risks=train_risks_mean,
            times=train_times,
            events=train_events_any,
            horizon=args.cut_horizon_days,
            min_frac=0.10,
        )

        cut_path = fold_out / "risk_cutoff_meanrisk.json"
        with open(cut_path, "w") as f:
            json.dump(cut_info, f, indent=2)

        logging.info(f"Saved cutoff to {cut_path}")
        logging.info(f"Cutoff summary: {cut_info}")


if __name__ == "__main__":
    main()
