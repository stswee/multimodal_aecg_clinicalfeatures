#!/usr/bin/env python3
"""
validate_mil_ecg_survival.py

Post-hoc validation for MIL ECG survival models.

Computes:
- Mean ± SD Harrell C-index
- Time-dependent AUC (1, 2, 3, 4 years) using TRUE follow-up time for IPCW
- Kaplan–Meier curves (risk tertiles)
- AUC vs time plot
- Calibration metrics + plot
- Hazard ratio per SD of risk score

Key change vs earlier:
- Use TRUE follow-up time (Follow-up period from enrollment (days)) for IPCW/AUC
  so AUC@4y can be computed when subjects are observed beyond 4 years.
- Still supports using your trained models (no retraining needed).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv


# -------------------------------
# Model (must match training)
# -------------------------------
class AlexNet1DEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, 11, 4, 5), nn.ReLU(), nn.MaxPool1d(3, 2),
            nn.Conv1d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool1d(3, 2),
            nn.Conv1d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv1d(384, 256, 3, padding=1), nn.ReLU(),
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
        return torch.softmax(self.w(A), dim=0)


class MILAttentionSurvivalModel(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super().__init__()
        self.encoder = AlexNet1DEncoder(embedding_dim)
        self.attention = GatedAttention(embedding_dim, attention_dim)
        self.risk_head = nn.Linear(embedding_dim, 1)

    def forward(self, segments):
        H = self.encoder(segments)
        A = self.attention(H)
        z = torch.sum(A * H, dim=0)
        return self.risk_head(z).squeeze()


# -------------------------------
# Utilities
# -------------------------------
@torch.no_grad()
def compute_risks(model, df, segments_dir, device):
    model.eval()
    risks = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Computing risks",
        leave=False
    ):
        pid = str(row["Patient ID"]).zfill(4)
        patient_dir = segments_dir / pid
        segs = sorted(patient_dir.glob("*.npy"))
        if len(segs) == 0:
            raise RuntimeError(f"No segments found for patient {pid} in {patient_dir}")

        x = torch.tensor(
            np.stack([np.load(f) for f in segs]),
            dtype=torch.float32,
            device=device
        )

        risk = float(model(x).item())
        risks.append(risk)

    return np.array(risks, dtype=float)


def safe_time_dependent_auc(y_train, y_test, risks, horizons_days):
    """
    Compute IPCW time-dependent AUC at each horizon.
    Returns (aucs_full, valid_mask) where aucs_full is length len(horizons_days),
    filled with np.nan where AUC cannot be computed.
    """
    times_test = y_test["time"].astype(float)
    max_time = float(np.max(times_test))

    aucs_full = np.full(len(horizons_days), np.nan, dtype=float)

    for i, t in enumerate(horizons_days):
        # sksurv requires evaluation times strictly within follow-up of test data
        if t >= max_time:
            continue

        # Must have at least one subject observed beyond t
        if np.sum(times_test > t) == 0:
            continue

        try:
            auc_t, _ = cumulative_dynamic_auc(
                y_train, y_test, risks, np.array([t], dtype=float)
            )
            aucs_full[i] = float(auc_t[0])
        except ValueError:
            # e.g., censoring survival function is zero at one or more time points
            continue

    return aucs_full


# -------------------------------
# Main validation
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--segments_dir", type=Path, required=True)
    parser.add_argument("--models_dir", type=Path, required=True)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=Path, default=Path("survival_validation"))

    # Column to use for TRUE follow-up time in IPCW AUC / KM / HR
    parser.add_argument(
        "--true_followup_col",
        type=str,
        default="Follow-up period from enrollment (days)",
        help="Column name for TRUE follow-up duration (can exceed 1460)."
    )

    # Optional: for numeric stability at the boundary, evaluate '4y' at 1459 days
    parser.add_argument(
        "--use_1459_for_4y",
        action="store_true",
        help="If set, uses 1459 days instead of 1460 for the 4-year horizon."
    )

    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv_path)

    needed = {"Patient ID", "fold", "event_cardiac", "time_to_event_days", args.true_followup_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Horizons for AUC
    horizons_years = np.array([1, 2, 3, 4], dtype=int)
    horizons_days = horizons_years * 365
    if args.use_1459_for_4y:
        horizons_days = np.array([365, 730, 1095, 1459], dtype=float)
    else:
        horizons_days = horizons_days.astype(float)

    fold_cindices = []
    fold_aucs = []

    all_risks, all_times, all_events = [], [], []

    folds = sorted(df["fold"].unique())

    for fold in tqdm(folds, desc="Validation folds"):
        val_df = df[df["fold"] == fold].copy()

        model = MILAttentionSurvivalModel(
            args.embedding_dim, args.attention_dim
        ).to(device)

        ckpt = args.models_dir / f"val_fold_{fold}" / "mil_model_best_cindex.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        model.load_state_dict(torch.load(ckpt, map_location=device))

        # Predicted risks from ECG
        risks = compute_risks(model, val_df, args.segments_dir, device)

        # --- Use TRUE follow-up time for evaluation ---
        # This enables AUC@4y if true follow-up extends beyond 1460.
        times_true = val_df[args.true_followup_col].astype(float).values
        events = val_df["event_cardiac"].astype(int).values.astype(bool)

        # C-index (use TRUE follow-up time)
        cidx = concordance_index(times_true, -risks, events)
        fold_cindices.append(float(cidx))

        # For IPCW AUC, use y_train/y_test; with CV, we can use the full dataset
        # as "training" for censoring weights, and fold as test.
        # This is standard for cross-validated reporting of time-dependent AUC.
        y_all = Surv.from_arrays(
            df["event_cardiac"].astype(int).values.astype(bool),
            df[args.true_followup_col].astype(float).values
        )
        y_val = Surv.from_arrays(events, times_true)

        aucs_full = safe_time_dependent_auc(y_all, y_val, risks, horizons_days)

        fold_aucs.append(aucs_full)
        all_risks.append(risks)
        all_times.append(times_true)
        all_events.append(events)

        # Optional debug prints per fold (uncomment if needed)
        # print(f"Fold {fold}: max_true_time={times_true.max():.1f}, n_gt_1460={(times_true>1460).sum()}")

    fold_aucs = np.vstack(fold_aucs)

    # -------------------------------
    # Aggregate metrics
    # -------------------------------
    print("\n=== C-index (TRUE follow-up time) ===")
    print(f"Mean ± SD: {np.mean(fold_cindices):.3f} ± {np.std(fold_cindices):.3f}")

    print("\n=== Time-dependent AUC (IPCW; TRUE follow-up time) ===")
    for i, y in enumerate(horizons_years):
        mean_auc = np.nanmean(fold_aucs[:, i])
        sd_auc = np.nanstd(fold_aucs[:, i])
        tlabel = f"{y}y"
        if (y == 4) and args.use_1459_for_4y:
            tlabel = "4y(~1459d)"
        print(f"AUC@{tlabel}: {mean_auc:.3f} ± {sd_auc:.3f}  (valid folds={np.sum(np.isfinite(fold_aucs[:, i]))}/{len(folds)})")

    # -------------------------------
    # Pooled arrays
    # -------------------------------
    all_risks = np.concatenate(all_risks).astype(float)
    all_times = np.concatenate(all_times).astype(float)
    all_events = np.concatenate(all_events).astype(bool)

    # -------------------------------
    # KM curves (tertiles) using TRUE follow-up time
    # -------------------------------
    tertiles = np.quantile(all_risks, [0.33, 0.67])
    groups = np.digitize(all_risks, tertiles)

    kmf = KaplanMeierFitter()
    plt.figure()
    for g in [0, 1, 2]:
        mask = groups == g
        kmf.fit(all_times[mask], all_events[mask], label=f"Risk tertile {g+1}")
        kmf.plot_survival_function()

    plt.title("Kaplan–Meier by predicted risk tertiles")
    plt.xlabel("Days (TRUE follow-up)")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(args.out_dir / "km_risk_tertiles.png", dpi=200)

    # -------------------------------
    # AUC vs time plot (mean across folds, ignoring NaNs)
    # -------------------------------
    plt.figure()
    plt.plot(horizons_years, np.nanmean(fold_aucs, axis=0), marker="o")
    plt.xlabel("Years")
    plt.ylabel("Time-dependent AUC")
    plt.title("AUC vs time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out_dir / "auc_vs_time.png", dpi=200)

    # -------------------------------
    # Calibration (4 years) using TRUE follow-up time
    # NOTE: For Cox models, "risk" is not a calibrated probability.
    # Here we do a simple grouped observed incidence at the 4y horizon.
    # -------------------------------
    horizon = 4 * 365
    if args.use_1459_for_4y:
        horizon = 1459

    bins = pd.qcut(all_risks, q=5, duplicates="drop")

    calib = []
    for b in bins.unique():
        mask = bins == b
        kmf.fit(all_times[mask], all_events[mask])
        surv_prob = float(kmf.predict(horizon))
        calib.append((float(np.mean(all_risks[mask])), 1.0 - surv_prob))

    calib = np.array(calib, dtype=float)

    plt.figure()
    plt.plot(calib[:, 0], calib[:, 1], "o-")
    # Reference diagonal on (risk score scale) isn't truly "perfect calibration" for Cox,
    # but it's a visual guide.
    plt.plot(
        [calib[:, 0].min(), calib[:, 0].max()],
        [calib[:, 1].min(), calib[:, 1].max()],
        "--", color="gray"
    )
    plt.xlabel("Mean predicted risk score")
    plt.ylabel(f"Observed event probability (~{horizon}d)")
    plt.title("Calibration (grouped) at ~4 years")
    plt.tight_layout()
    plt.savefig(args.out_dir / "calibration_4y.png", dpi=200)

    # -------------------------------
    # Hazard ratio per SD using TRUE follow-up time
    # -------------------------------
    df_hr = pd.DataFrame({
        "time": all_times,
        "event": all_events.astype(int),
        "risk": (all_risks - all_risks.mean()) / (all_risks.std() + 1e-12),
    })

    cph = CoxPHFitter()
    cph.fit(df_hr, duration_col="time", event_col="event")

    print("\n=== Hazard ratio per SD of risk (TRUE follow-up time) ===")
    print(cph.summary)

    print(f"\nSaved plots to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
