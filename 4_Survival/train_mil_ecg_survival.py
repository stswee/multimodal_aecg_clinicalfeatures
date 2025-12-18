#!/usr/bin/env python3
"""
train_mil_ecg_attention_survival.py

Attention-based MIL training for segmented ECGs with fold-aware validation,
adapted for SURVIVAL ANALYSIS (cardiac death).

Key ideas:
- Each patient = a bag of ECG segments
- Encoder -> segment embeddings
- Gated attention pooling -> patient embedding z
- Risk head -> scalar risk score (higher = higher hazard)
- Loss: Cox partial log-likelihood (mini-batch approximation)
- Metrics per epoch:
    * Train loss
    * Val Cox loss (full validation set)
    * Val Harrell's C-index (primary)
- Saves:
    * BEST model checkpoint only (by validation C-index)
    * Optional: Patient embeddings (training only)
    * Per-epoch validation metrics (CSV log)

Expected CSV columns:
- Patient ID
- fold
- time_to_event_days
- event_cardiac   (1 = cardiac death; 0 = censored (alive or non-cardiac death))
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


# =========================================================
# Logging
# =========================================================
def setup_logging():
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_mil_ecg_attention_survival_{timestamp}.log"

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

        f.write("\n---- Survival labels ----\n")
        f.write("time_to_event_days: min(true_followup_days, days_4years)\n")
        f.write("event_cardiac: 1 if Cause of death in {3,6}; else 0\n")

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
class ECGMILDatasetSurvival(Dataset):
    """One item = one patient (bag of ECG segments) + survival labels."""

    def __init__(self, csv_df: pd.DataFrame, segments_dir: Path):
        self.df = csv_df.reset_index(drop=True)
        self.segments_dir = segments_dir

        required = {"Patient ID", "time_to_event_days", "event_cardiac"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patient_id = str(row["Patient ID"]).zfill(4)

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

        t = float(row["time_to_event_days"])
        e = float(row["event_cardiac"])

        time = torch.tensor(t, dtype=torch.float32)
        event = torch.tensor(e, dtype=torch.float32)

        return segments, time, event, patient_id


def mil_collate_fn(batch):
    # segments is variable-length per patient (bag size can differ)
    segments_list, times, events, patient_ids = zip(*batch)
    return list(segments_list), torch.stack(times), torch.stack(events), list(patient_ids)


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
        # x: (Nseg, L)
        x = x.unsqueeze(1)      # (Nseg, 1, L)
        x = self.features(x)
        x = x.squeeze(-1)       # (Nseg, 256)
        return self.fc(x)       # (Nseg, D)


class GatedAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        self.V = nn.Linear(embedding_dim, attention_dim)
        self.U = nn.Linear(embedding_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)

    def forward(self, H):
        # H: (Nseg, D)
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))
        A = self.w(A)                 # (Nseg, 1)
        return torch.softmax(A, dim=0)


class MILAttentionSurvivalModel(nn.Module):
    """
    MIL attention model that outputs a scalar risk score (log-risk / log-hazard).
    Higher risk => earlier events.
    """
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        self.encoder = AlexNet1DEncoder(embedding_dim)
        self.attention = GatedAttention(embedding_dim, attention_dim)
        self.risk_head = nn.Linear(embedding_dim, 1)

    def forward(self, segments):
        # segments: (Nseg, L)
        H = self.encoder(segments)       # (Nseg, D)
        A = self.attention(H)            # (Nseg, 1)
        z = torch.sum(A * H, dim=0)      # (D,)
        risk = self.risk_head(z).squeeze()  # scalar
        return risk, z, A.squeeze()


# =========================================================
# Survival loss and metrics
# =========================================================
class CoxPHLoss(nn.Module):
    """
    Negative Cox partial log-likelihood.
    Works on a batch of patients (mini-batch approximation).
    """
    def forward(self, risks, times, events):
        # risks: (B,)   times: (B,)   events: (B,) in {0,1}
        # Sort by time descending (so risk set = [0..i])
        order = torch.argsort(times, descending=True)
        risks = risks[order]
        events = events[order]

        log_cumsum = torch.logcumsumexp(risks, dim=0)
        per_event = (risks - log_cumsum) * events

        denom = events.sum().clamp_min(1.0)
        return -per_event.sum() / denom


def harrell_c_index(risks, times, events):
    """
    Harrell's C-index.
    Interprets higher risk => shorter time.
    O(n^2), but fine for ~1k patients.
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
            # i must be an observed event earlier than j's time
            if times[i] < times[j]:
                permissible += 1
                if risks[i] > risks[j]:
                    concordant += 1
                elif risks[i] == risks[j]:
                    concordant += 0.5

    return (concordant / permissible) if permissible > 0 else float("nan")


@torch.no_grad()
def evaluate_survival(model, loader, device):
    model.eval()
    all_risks, all_times, all_events = [], [], []

    for segments_list, times, events, _ in loader:
        # segments_list is a list of length B
        for i in range(len(segments_list)):
            seg = segments_list[i].to(device)
            risk, _, _ = model(seg)
            all_risks.append(float(risk.item()))
            all_times.append(float(times[i].item()))
            all_events.append(float(events[i].item()))

    cindex = harrell_c_index(all_risks, all_times, all_events)

    # Also compute full-set Cox loss (not used for selection, but informative)
    risks_t = torch.tensor(all_risks, dtype=torch.float32, device=device)
    times_t = torch.tensor(all_times, dtype=torch.float32, device=device)
    events_t = torch.tensor(all_events, dtype=torch.float32, device=device)
    cox_loss = float(CoxPHLoss().to(device)(risks_t, times_t, events_t).item())

    n_events = int(np.sum(np.array(all_events) == 1.0))
    return {
        "val_cindex": float(cindex),
        "val_cox_loss": float(cox_loss),
        "val_n": int(len(all_times)),
        "val_events": n_events,
    }


def train_epoch_survival(
    model,
    loader,
    optimizer,
    criterion,
    device,
    save_embeddings_dir: Path | None,
    epoch: int,
    total_epochs: int,
):
    model.train()
    losses = []

    for segments_list, times, events, patient_ids in tqdm(
        loader,
        desc=f"Train | Epoch {epoch+1}/{total_epochs}",
        total=len(loader),
        ncols=100,
        leave=False,
    ):
        optimizer.zero_grad()

        # Compute per-patient risks within this batch
        batch_risks = []
        for seg in segments_list:
            seg = seg.to(device)
            risk, z, _ = model(seg)
            batch_risks.append(risk)

        batch_risks = torch.stack(batch_risks)          # (B,)
        batch_times = times.to(device)                  # (B,)
        batch_events = events.to(device)                # (B,)

        loss = criterion(batch_risks, batch_times, batch_events)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        # Optionally save training embeddings per patient (z) – recompute once with no grad
        if save_embeddings_dir is not None:
            with torch.no_grad():
                for pid, seg in zip(patient_ids, segments_list):
                    seg = seg.to(device)
                    _, z, _ = model(seg)
                    np.save(save_embeddings_dir / f"{pid}.npy", z.detach().cpu().numpy())

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    logging.info(f"Epoch {epoch+1}/{total_epochs} | Train Cox loss: {mean_loss:.4f}")
    return mean_loss


# =========================================================
# Main
# =========================================================
def main():
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--val_fold", type=int, required=True)
    parser.add_argument("--segments_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("mil_survival_outputs"))

    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--attention_dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Survival training detail
    parser.add_argument("--patient_batch_size", type=int, default=8,
                        help="Number of patients per batch for Cox loss (mini-batch approximation).")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_train_embeddings", action="store_true",
                        help="If set, saves per-patient train embeddings each epoch (can be slow/large).")

    args = parser.parse_args()
    logging.info(f"Arguments: {vars(args)}")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    df = pd.read_csv(args.csv_path)

    # Basic checks
    needed = {"Patient ID", "fold", "time_to_event_days", "event_cardiac"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    train_df = df[df["fold"] != args.val_fold].copy()
    val_df = df[df["fold"] == args.val_fold].copy()

    output_dir = args.output_dir / f"val_fold_{args.val_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = None
    if args.save_train_embeddings:
        embeddings_dir = output_dir / "train_embeddings"
        embeddings_dir.mkdir(exist_ok=True)

    save_args_to_txt(args, output_dir)

    train_loader = DataLoader(
        ECGMILDatasetSurvival(train_df, args.segments_dir),
        batch_size=args.patient_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mil_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        ECGMILDatasetSurvival(val_df, args.segments_dir),
        batch_size=args.patient_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mil_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = MILAttentionSurvivalModel(
        embedding_dim=args.embedding_dim,
        attention_dim=args.attention_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = CoxPHLoss().to(device)

    # Logs CSV
    metrics_csv = output_dir / "epoch_metrics.csv"
    metrics_rows = []

    best_cindex = -np.inf

    logging.info("Starting training (survival / Cox)...")
    for epoch in tqdm(range(args.epochs), desc="Epochs", ncols=80):
        train_loss = train_epoch_survival(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            save_embeddings_dir=embeddings_dir,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        val_metrics = evaluate_survival(model, val_loader, device)
        val_cindex = val_metrics["val_cindex"]
        val_cox_loss = val_metrics["val_cox_loss"]

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"VAL C-index={val_cindex:.4f} | "
            f"VAL Cox loss={val_cox_loss:.4f} | "
            f"VAL events={val_metrics['val_events']}/{val_metrics['val_n']}"
        )

        row = {
            "epoch": epoch + 1,
            "train_cox_loss": train_loss,
            "val_cindex": val_cindex,
            "val_cox_loss": val_cox_loss,
            "val_n": val_metrics["val_n"],
            "val_events": val_metrics["val_events"],
        }
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

        # Save best by C-index
        if np.isfinite(val_cindex) and val_cindex > best_cindex:
            best_cindex = val_cindex
            torch.save(model.state_dict(), output_dir / "mil_model_best_cindex.pt")
            logging.info(
                f"New BEST model saved at epoch {epoch+1} (C-index={val_cindex:.4f})"
            )

    logging.info(
        f"Training complete. Best validation C-index = {best_cindex:.4f} "
        f"(mil_model_best_cindex.pt). Metrics log: {metrics_csv}"
    )


if __name__ == "__main__":
    main()
