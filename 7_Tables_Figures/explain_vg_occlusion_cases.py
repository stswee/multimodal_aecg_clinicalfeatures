#!/usr/bin/env python3
"""
explain_fold2_vg_occlusion.py

Occlusion-over-windows explainability for the best multimodal Vector-Gating (VG) model,
using:
  - Frozen ECG encoder (MIL + TCN over window-level engineered features)
  - Frozen text embeddings
  - Trained VG fusion model (takes ECG embedding + text embedding)

For every *positive* SCD patient and every *positive* PFD patient in a chosen fold (e.g., fold 2),
this script generates one per-patient figure:
  1) Occlusion saliency strip over window index (Δp per window)
  2) Raw ECG waveform of the most salient window (30s)
  3) Baseline risk + gate stats + modality contribution summary

Outputs:
  - PNG per patient under: <out_dir>/SCD/ and <out_dir>/PFD/
  - NPZ per patient (optional) storing deltas/gate/baseline under: <out_dir>/npz/

Usage example (fold 2):
python explain_vg_occlusion_cases.py \
  --val_fold 2 \
  --ecg_encoder_fold_dir ../../music/best_results/tcn_ecg_embeddings/val_fold_2 \
  --vg_fold_dir ../../music/best_results/vectorgating_embeddings_LLaMA8B_BioBERT/val_fold_2 \
  --text_val_embeddings_npz ../../music/best_text_embeddings_LLaMA8B_BioBERT/val_fold_2/val_embeddings.npz \
  --segments_dir ../../music/preprocessed_segments_HRV_complete \
  --preprocessed_dir ../../music/preprocessed_HRV \
  --window_index_csv ../../music/window_index_metadata_HRV.csv \
  --fs 200 --window_sec 30 \
  --out_dir explain_fold2_vg \
  --drop_na_rows

Notes:
- Occlusion is applied in the z-scored feature space (setting a window row to 0), which corresponds
  to a neutral/average window under normalization.
- This recomputes ECG embeddings by re-running the frozen ECG encoder for each occlusion.
"""

import argparse
import ast
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ----------------------------
# Small utilities
# ----------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_zfill_pid(pid) -> str:
    s = str(pid).strip()
    # Keep existing zero-padding if present; otherwise pad to 4 like your training
    if s.isdigit():
        return s.zfill(4)
    # If not purely digits, try trailing digits
    m = re.search(r"(\d+)$", s)
    return (m.group(1).zfill(4) if m else s)

def find_npz_for_patient(preprocessed_dir: Path, patient_id: str) -> Optional[Path]:
    """
    Matches your segmentation script behavior: find first .npz containing patient_id in filename.
    """
    patient_id = str(patient_id)
    if not preprocessed_dir.exists():
        return None
    cands = [p for p in preprocessed_dir.iterdir() if p.suffix == ".npz" and patient_id in p.name]
    if len(cands) == 0:
        return None
    return cands[0]

def load_feature_cols(feature_cols_txt: Path) -> List[str]:
    cols = [ln.strip() for ln in feature_cols_txt.read_text().splitlines() if ln.strip()]
    if len(cols) == 0:
        raise RuntimeError(f"Empty feature_cols at: {feature_cols_txt}")
    return cols

def load_window_starts(window_index_csv: Path) -> Dict[str, List[int]]:
    """
    Returns dict: pid -> list(start_idx)
    window_index_csv has columns like: patient_id, start_indices (string repr of list)
    """
    meta = pd.read_csv(window_index_csv, dtype={"patient_id": str})
    meta["patient_id"] = meta["patient_id"].astype(str).str.strip()
    starts_by_pid = {}
    for _, row in meta.iterrows():
        pid = safe_zfill_pid(row["patient_id"])
        starts = ast.literal_eval(row["start_indices"])
        starts_by_pid[pid] = list(starts)
    return starts_by_pid

def load_text_embeddings(text_val_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(text_val_npz, allow_pickle=True)
    pids = data["pids"].astype(str)
    z = data["z"]
    y_scd = data["y_scd"]
    y_pfd = data["y_pfd"]
    # normalize pid formatting to match your folder naming
    pids = np.array([safe_zfill_pid(p) for p in pids])
    return pids, z, y_scd, y_pfd

def read_segment_feature_matrix(
    segments_dir: Path,
    pid: str,
    feature_cols: List[str],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    drop_na_rows: bool,
    impute_nan_with: float,
    min_segments: int,
    sort_by: str = "window_idx",
) -> Optional[np.ndarray]:
    """
    Loads <segments_dir>/<pid>/<pid>_segment_features.csv
    and applies cleaning + z-score like your training pipeline.
    Returns X: (N, F) float32 or None if insufficient.
    """
    pid = safe_zfill_pid(pid)
    csv_path = segments_dir / pid / f"{pid}_segment_features.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return None

    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)

    if drop_na_rows:
        good = np.isfinite(X).all(axis=1)
        X = X[good]
    else:
        X = np.where(np.isfinite(X), X, float(impute_nan_with)).astype(np.float32)

    if X.shape[0] < int(min_segments):
        return None

    if mean is not None and std is not None:
        X = (X - mean) / std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return X


# ----------------------------
# ECG encoder: recreate model (minimal copy of your classes)
#   We infer dims from the checkpoint state_dict so you don't have to pass architecture args.
# ----------------------------

class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, hidden: int = 128, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return self.net(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, drop):
        super().__init__()
        pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.drop(self.relu(self.conv(x)))[:, :, :x.size(2)]
        return y + (x if self.down is None else self.down(x))

class TCN(nn.Module):
    def __init__(self, in_dim, hid, layers, k, drop):
        super().__init__()
        self.net = nn.Sequential(
            *[
                TemporalBlock(in_dim if i == 0 else hid, hid, k, 2**i, drop)
                for i in range(layers)
            ]
        )

    def forward(self, x):
        return self.net(x)

class AttentionMIL(nn.Module):
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, T):
        X = T.squeeze(0).transpose(0, 1)  # (N, C)
        a = torch.softmax(self.attn(X).squeeze(1), dim=0)  # (N,)
        z = (X * a.unsqueeze(1)).sum(dim=0)  # (C,)
        return z, a

class MILTCNFeatureMultiBranch(nn.Module):
    def __init__(
        self,
        in_feat_dim: int,
        emb: int,
        hid: int,
        layers: int,
        k: int,
        drop: float,
        attn: int,
        branch_hidden: Optional[int] = None,
        branch_drop: Optional[float] = None,
        enc_hidden: int = 128,
        enc_drop: float = 0.1,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(in_feat_dim, emb, hidden=enc_hidden, drop=enc_drop)
        self.tcn = TCN(emb, hid, layers, k, drop)
        self.pool = AttentionMIL(hid, attn)

        bh = hid if branch_hidden is None else int(branch_hidden)
        bd = drop if branch_drop is None else float(branch_drop)

        self.scd_branch = nn.Sequential(
            nn.Linear(hid, bh),
            nn.ReLU(),
            nn.Dropout(bd),
            nn.Linear(bh, hid),
            nn.ReLU(),
        )
        self.pfd_branch = nn.Sequential(
            nn.Linear(hid, bh),
            nn.ReLU(),
            nn.Dropout(bd),
            nn.Linear(bh, hid),
            nn.ReLU(),
        )

        self.head_scd = nn.Linear(hid, 1)
        self.head_pfd = nn.Linear(hid, 1)

    def forward(self, X: torch.Tensor):
        H = self.encoder(X)                    # (N, emb)
        H = H.transpose(0, 1).unsqueeze(0)     # (1, emb, N)
        T = self.tcn(H)                        # (1, hid, N)
        z, a = self.pool(T)                    # z: (hid,)
        z_scd = self.scd_branch(z)
        z_pfd = self.pfd_branch(z)
        return self.head_scd(z_scd), self.head_pfd(z_pfd), z, a

def infer_ecg_encoder_arch(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Infer ECG encoder dims from checkpoint state_dict.
    """
    # FeatureEncoder dims
    enc0_w = state["encoder.net.0.weight"]      # (enc_hidden, in_feat_dim)
    enc3_w = state["encoder.net.3.weight"]      # (emb_dim, enc_hidden)
    in_feat_dim = enc0_w.shape[1]
    enc_hidden = enc0_w.shape[0]
    emb_dim = enc3_w.shape[0]

    # TCN dims
    # Find how many TemporalBlocks
    tcn_block_idxs = []
    for k in state.keys():
        m = re.match(r"tcn\.net\.(\d+)\.conv\.weight", k)
        if m:
            tcn_block_idxs.append(int(m.group(1)))
    if not tcn_block_idxs:
        raise RuntimeError("Could not infer TCN layers from state_dict.")
    layers = max(tcn_block_idxs) + 1
    conv0_w = state[f"tcn.net.0.conv.weight"]   # (hid, in_ch, kernel)
    hid = conv0_w.shape[0]
    kernel = conv0_w.shape[2]

    # Attention dim
    attn0_w = state["pool.attn.0.weight"]       # (attn_dim, hid)
    attn_dim = attn0_w.shape[0]

    # Branch hidden
    bh = state["scd_branch.0.weight"].shape[0]

    return {
        "in_feat_dim": int(in_feat_dim),
        "emb_dim": int(emb_dim),
        "hid": int(hid),
        "layers": int(layers),
        "kernel": int(kernel),
        "attn_dim": int(attn_dim),
        "enc_hidden": int(enc_hidden),
        "branch_hidden": int(bh),
    }

def load_ecg_encoder(ecg_encoder_fold_dir: Path, device: torch.device) -> MILTCNFeatureMultiBranch:
    ckpt_path = ecg_encoder_fold_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]

    arch = infer_ecg_encoder_arch(state)
    # Dropouts don't affect eval and have no weights; set to 0.0 safely
    model = MILTCNFeatureMultiBranch(
        in_feat_dim=arch["in_feat_dim"],
        emb=arch["emb_dim"],
        hid=arch["hid"],
        layers=arch["layers"],
        k=arch["kernel"],
        drop=0.0,
        attn=arch["attn_dim"],
        branch_hidden=arch["branch_hidden"],
        branch_drop=0.0,
        enc_hidden=arch["enc_hidden"],
        enc_drop=0.0,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------
# VG model: recreate and infer dims from checkpoint state_dict
# ----------------------------

class VectorGatingMultiHead(nn.Module):
    def __init__(
        self,
        ecg_dim: int,
        text_dim: int,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_layers: int = 1,
    ):
        super().__init__()

        self.ecg_proj = nn.Sequential(
            nn.LayerNorm(ecg_dim),
            nn.Linear(ecg_dim, proj_dim),
            nn.ReLU(),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, proj_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
        )

        layers = []
        in_dim = proj_dim
        layers.append(nn.LayerNorm(proj_dim))
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.head_scd = nn.Linear(hidden_dim, 1)
        self.head_pfd = nn.Linear(hidden_dim, 1)

    def forward(self, ecg, text):
        h_ecg = self.ecg_proj(ecg)
        h_text = self.text_proj(text)
        gate_in = torch.cat([h_ecg, h_text], dim=1)
        g = torch.sigmoid(self.gate(gate_in))
        z_fused = g * h_ecg + (1.0 - g) * h_text
        z = self.trunk(z_fused)
        return (
            self.head_scd(z).squeeze(-1),
            self.head_pfd(z).squeeze(-1),
            z,
            g,
        )

@torch.no_grad()
def gate_stats(g: torch.Tensor) -> Dict[str, float]:
    g = g.detach()
    return {
        "g_mean": float(g.mean().item()),
        "g_std": float(g.std(unbiased=False).item()),
        "g_min": float(g.min().item()),
        "g_max": float(g.max().item()),
    }

def infer_vg_arch(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    # Linear weights are reliable
    proj_w = state["ecg_proj.1.weight"]     # (proj_dim, ecg_dim)
    proj_dim = proj_w.shape[0]
    ecg_dim = proj_w.shape[1]

    text_w = state["text_proj.1.weight"]    # (proj_dim, text_dim)
    text_dim = text_w.shape[1]

    # hidden dim from head
    hidden_dim = state["head_scd.weight"].shape[1]

    # infer num_layers from trunk linear weights
    # trunk has: LayerNorm at index 0, then repeating Linear/ReLU/Dropout -> Linear modules at indices 1,4,7,...
    trunk_linear_idxs = []
    for k in state.keys():
        m = re.match(r"trunk\.(\d+)\.weight", k)
        if m:
            idx = int(m.group(1))
            # exclude LayerNorm weights (1D)
            w = state[k]
            if w.ndim == 2:
                trunk_linear_idxs.append(idx)
    num_layers = len(trunk_linear_idxs)

    return {
        "ecg_dim": int(ecg_dim),
        "text_dim": int(text_dim),
        "proj_dim": int(proj_dim),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
    }

def load_vg_model(vg_fold_dir: Path, device: torch.device) -> VectorGatingMultiHead:
    ckpt_path = vg_fold_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]

    arch = infer_vg_arch(state)
    # Dropout doesn't matter for eval; set to 0
    model = VectorGatingMultiHead(
        ecg_dim=arch["ecg_dim"],
        text_dim=arch["text_dim"],
        proj_dim=arch["proj_dim"],
        hidden_dim=arch["hidden_dim"],
        dropout=0.0,
        num_layers=arch["num_layers"],
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------
# Core occlusion computation
# ----------------------------

@torch.no_grad()
def multimodal_probs_and_gate(
    ecg_encoder: MILTCNFeatureMultiBranch,
    vg: VectorGatingMultiHead,
    X_windows: torch.Tensor,           # (N, F)
    z_text: torch.Tensor,              # (1, D_text)
) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      p_scd, p_pfd, g (proj_dim,), h_ecg (proj_dim,), h_text (proj_dim,)
    """
    _, _, z_ecg, _ = ecg_encoder(X_windows)       # z_ecg: (hid=ecg_emb_dim,)
    z_ecg = z_ecg.unsqueeze(0)                    # (1, ecg_dim)

    z_scd, z_pfd, _, g = vg(z_ecg, z_text)       # logits (1,), gate (1,proj_dim)
    p_scd = torch.sigmoid(z_scd).item()
    p_pfd = torch.sigmoid(z_pfd).item()

    # For modality contribution ratio we also want projected vectors
    h_ecg = vg.ecg_proj(z_ecg).squeeze(0)        # (proj_dim,)
    h_text = vg.text_proj(z_text).squeeze(0)     # (proj_dim,)
    g_vec = g.squeeze(0)                         # (proj_dim,)

    return p_scd, p_pfd, g_vec, h_ecg, h_text

@torch.no_grad()
def compute_occlusion_deltas(
    ecg_encoder: MILTCNFeatureMultiBranch,
    vg: VectorGatingMultiHead,
    X_windows_np: np.ndarray,         # (N, F) float32
    z_text_np: np.ndarray,            # (D_text,) float32
    task: str,                        # "scd" or "pfd"
    device: torch.device,
) -> Tuple[float, np.ndarray, torch.Tensor, Dict[str, float], Dict[str, float]]:
    """
    Returns:
      baseline_prob, deltas (N,), gate_vec (proj_dim,),
      gate_summary, modality_summary
    """
    X = torch.tensor(X_windows_np, dtype=torch.float32, device=device)
    z_text = torch.tensor(z_text_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, D_text)

    p_scd_full, p_pfd_full, g_full, h_ecg_full, h_text_full = multimodal_probs_and_gate(ecg_encoder, vg, X, z_text)
    p_full = p_scd_full if task == "scd" else p_pfd_full

    # Gate summaries
    gs = gate_stats(g_full.unsqueeze(0))

    # Modality contribution summaries (dimension dominance + L2 ratio)
    frac_ecg_dom = float((g_full > 0.7).float().mean().item())
    frac_txt_dom = float((g_full < 0.3).float().mean().item())
    z_ecg_part = g_full * h_ecg_full
    z_txt_part = (1.0 - g_full) * h_text_full
    ecg_norm = float(torch.norm(z_ecg_part).item())
    txt_norm = float(torch.norm(z_txt_part).item())
    denom = max(ecg_norm + txt_norm, 1e-12)
    modality = {
        "frac_ecg_dom": frac_ecg_dom,
        "frac_text_dom": frac_txt_dom,
        "ecg_norm_ratio": float(ecg_norm / denom),
        "text_norm_ratio": float(txt_norm / denom),
    }

    N = X.shape[0]
    deltas = np.zeros(N, dtype=np.float32)

    for i in range(N):
        X_mask = X.clone()
        X_mask[i] = 0.0  # occlude window i in z-scored space

        p_scd_i, p_pfd_i, _, _, _ = multimodal_probs_and_gate(ecg_encoder, vg, X_mask, z_text)
        p_i = p_scd_i if task == "scd" else p_pfd_i

        deltas[i] = float(p_full - p_i)

    return float(p_full), deltas, g_full.detach().cpu(), gs, modality


# ----------------------------
# Plotting
# ----------------------------

def plot_patient_explainability(
    out_png: Path,
    pid: str,
    task: str,
    baseline_prob: float,
    deltas: np.ndarray,
    ecg_segment: np.ndarray,
    fs: int,
    gate_summary: Dict[str, float],
    modality_summary: Dict[str, float],
):
    """
    Creates a single per-patient figure:
      - Saliency strip
      - Raw ECG waveform of most salient window
      - Text box summary
    """
    pid = safe_zfill_pid(pid)
    task_u = task.upper()

    top_idx = int(np.argmax(deltas)) if len(deltas) > 0 else -1
    top_delta = float(deltas[top_idx]) if top_idx >= 0 else float("nan")

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 2.0, 1.4], width_ratios=[2.0, 1.2], hspace=0.6, wspace=0.3)

    # --- Saliency strip ---
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(deltas[None, :], aspect="auto", cmap="hot")
    ax0.set_yticks([])
    ax0.set_xlabel("Window index (30s windows)")
    ax0.set_title(f"{task_u} occlusion saliency | pid={pid} | baseline p={baseline_prob:.3f}")

    # --- Raw ECG waveform for top window ---
    ax1 = fig.add_subplot(gs[1, 0])
    if ecg_segment is not None and len(ecg_segment) > 0:
        t = np.arange(len(ecg_segment)) / float(fs)
        ax1.plot(t, ecg_segment)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"Most salient window: idx={top_idx} | Δp={top_delta:.3f}")
    else:
        ax1.text(0.5, 0.5, "Raw ECG segment unavailable", ha="center", va="center")
        ax1.set_axis_off()

    # --- Summary text box ---
    ax2 = fig.add_subplot(gs[1:, 1])
    ax2.axis("off")

    lines = []
    lines.append(f"Patient: {pid}")
    lines.append(f"Task: {task_u}")
    lines.append(f"Baseline probability: {baseline_prob:.3f}")
    lines.append("")
    lines.append("Gate summary (VG):")
    lines.append(f"  mean={gate_summary['g_mean']:.3f}")
    lines.append(f"  std ={gate_summary['g_std']:.3f}")
    lines.append(f"  min ={gate_summary['g_min']:.3f}")
    lines.append(f"  max ={gate_summary['g_max']:.3f}")
    lines.append("")
    lines.append("Modality contribution (patient-level):")
    lines.append(f"  ECG-dominant dims (g>0.7):  {modality_summary['frac_ecg_dom']*100:.1f}%")
    lines.append(f"  Text-dominant dims (g<0.3): {modality_summary['frac_text_dom']*100:.1f}%")
    lines.append(f"  ECG norm ratio:  {modality_summary['ecg_norm_ratio']*100:.1f}%")
    lines.append(f"  Text norm ratio: {modality_summary['text_norm_ratio']*100:.1f}%")
    lines.append("")
    lines.append("Saliency summary:")
    lines.append(f"  max Δp: {float(np.max(deltas)):.3f}")
    lines.append(f"  meanΔp: {float(np.mean(deltas)):.3f}")
    lines.append(f"  top window: {top_idx}")

    ax2.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    # --- (Optional) bottom-left blank / could add extra snippet later ---
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis("off")
    ax3.text(
        0.0, 0.8,
        "Tip: Add your LLM rationale here manually (or programmatically) for the final paper figure.\n"
        "This script focuses on occlusion + waveform + modality stats.",
        fontsize=9
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--val_fold", type=int, required=True)

    p.add_argument("--ecg_encoder_fold_dir", type=Path, required=True,
                   help="e.g., mil_feature_outputs/val_fold_2 (must contain best_model.pt, feature_cols.txt, feature_mean.npy, feature_std.npy)")
    p.add_argument("--vg_fold_dir", type=Path, required=True,
                   help="e.g., multimodal_outputs/val_fold_2 (must contain best_model.pt)")
    p.add_argument("--text_val_embeddings_npz", type=Path, required=True,
                   help="e.g., <text_embedding_dir>/val_fold_2/val_embeddings.npz")

    p.add_argument("--segments_dir", type=Path, required=True,
                   help="Root containing <pid>/<pid>_segment_features.csv")
    p.add_argument("--preprocessed_dir", type=Path, required=True,
                   help="Folder containing preprocessed ECG .npz files with 'signal' and 'rpeaks'")

    p.add_argument("--window_index_csv", type=Path, required=True,
                   help="CSV with patient_id and start_indices for 30s windows")

    p.add_argument("--fs", type=int, default=200)
    p.add_argument("--window_sec", type=int, default=30)

    p.add_argument("--out_dir", type=Path, default=Path("explain_vg_occlusion"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    # Cleaning/normalization flags (must match ECG encoder training fold)
    p.add_argument("--drop_na_rows", action="store_true")
    p.add_argument("--impute_nan_with", type=float, default=0.0)
    p.add_argument("--min_segments", type=int, default=3)
    p.add_argument("--sort_by", type=str, default="window_idx", choices=["window_idx", "start_idx"])

    # Controls
    p.add_argument("--max_patients_per_task", type=int, default=-1,
                   help="If >0, limit number of patients processed for each task (SCD and PFD).")

    args = p.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Load starts
    starts_by_pid = load_window_starts(args.window_index_csv)
    win_len = int(args.fs * args.window_sec)

    # Load fold feature artifacts
    feature_cols = load_feature_cols(args.ecg_encoder_fold_dir / "feature_cols.txt")

    mean_path = args.ecg_encoder_fold_dir / "feature_mean.npy"
    std_path = args.ecg_encoder_fold_dir / "feature_std.npy"
    mean = np.load(mean_path).astype(np.float32) if mean_path.exists() else None
    std = np.load(std_path).astype(np.float32) if std_path.exists() else None

    # Load models
    ecg_encoder = load_ecg_encoder(args.ecg_encoder_fold_dir, device)
    vg_model = load_vg_model(args.vg_fold_dir, device)

    # Load text embeddings (and labels) for fold val set
    pids, z_text_all, y_scd, y_pfd = load_text_embeddings(args.text_val_embeddings_npz)
    text_by_pid = {pids[i]: z_text_all[i].astype(np.float32) for i in range(len(pids))}

    # Identify positive patients
    scd_pids = [pids[i] for i in range(len(pids)) if int(y_scd[i]) == 1]
    pfd_pids = [pids[i] for i in range(len(pids)) if int(y_pfd[i]) == 1]

    if args.max_patients_per_task and args.max_patients_per_task > 0:
        scd_pids = scd_pids[:args.max_patients_per_task]
        pfd_pids = pfd_pids[:args.max_patients_per_task]

    print(f"[Fold {args.val_fold}] SCD positives: {len(scd_pids)} | PFD positives: {len(pfd_pids)}")
    print(f"Output dir: {args.out_dir.resolve()}")

    # Output folders
    out_scd = args.out_dir / f"fold_{args.val_fold}" / "SCD"
    out_pfd = args.out_dir / f"fold_{args.val_fold}" / "PFD"
    out_npz = args.out_dir / f"fold_{args.val_fold}" / "npz"
    out_scd.mkdir(parents=True, exist_ok=True)
    out_pfd.mkdir(parents=True, exist_ok=True)
    out_npz.mkdir(parents=True, exist_ok=True)

    def process_pid(pid: str, task: str):
        pid = safe_zfill_pid(pid)

        # Load feature matrix
        X = read_segment_feature_matrix(
            segments_dir=args.segments_dir,
            pid=pid,
            feature_cols=feature_cols,
            mean=mean,
            std=std,
            drop_na_rows=args.drop_na_rows,
            impute_nan_with=args.impute_nan_with,
            min_segments=args.min_segments,
            sort_by=args.sort_by,
        )
        if X is None:
            print(f"[Skip] pid={pid} | task={task} | No valid segment features.")
            return

        # Load text embedding
        if pid not in text_by_pid:
            print(f"[Skip] pid={pid} | task={task} | No text embedding.")
            return
        z_text = text_by_pid[pid]

        # Compute occlusion deltas
        baseline_prob, deltas, g_vec, gsum, msum = compute_occlusion_deltas(
            ecg_encoder=ecg_encoder,
            vg=vg_model,
            X_windows_np=X,
            z_text_np=z_text,
            task=task,
            device=device,
        )

        # Load raw ECG and extract most salient window waveform
        starts = starts_by_pid.get(pid, None)
        raw_seg = None
        if starts is not None and len(starts) == X.shape[0]:
            npz_path = find_npz_for_patient(args.preprocessed_dir, pid)
            if npz_path is not None and npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)
                sig = data["signal"]
                if sig.ndim > 1:
                    sig = sig[0]
                top_idx = int(np.argmax(deltas)) if len(deltas) > 0 else 0
                s = int(starts[top_idx])
                e = s + win_len
                if e <= len(sig):
                    raw_seg = sig[s:e].astype(np.float32)
        else:
            # It is okay if starts aren't present or lengths mismatch (e.g., dropped rows)
            # We can still save saliency; waveform panel may be blank.
            pass

        # Save NPZ (so you can later pick representative cases without rerunning)
        np.savez_compressed(
            out_npz / f"{task.upper()}_{pid}.npz",
            pid=pid,
            task=task,
            baseline_prob=np.array([baseline_prob], dtype=np.float32),
            deltas=deltas.astype(np.float32),
            gate=g_vec.numpy().astype(np.float32),
            gate_mean=np.array([gsum["g_mean"]], dtype=np.float32),
            gate_std=np.array([gsum["g_std"]], dtype=np.float32),
            gate_min=np.array([gsum["g_min"]], dtype=np.float32),
            gate_max=np.array([gsum["g_max"]], dtype=np.float32),
            frac_ecg_dom=np.array([msum["frac_ecg_dom"]], dtype=np.float32),
            frac_text_dom=np.array([msum["frac_text_dom"]], dtype=np.float32),
            ecg_norm_ratio=np.array([msum["ecg_norm_ratio"]], dtype=np.float32),
            text_norm_ratio=np.array([msum["text_norm_ratio"]], dtype=np.float32),
        )

        # Plot
        out_png = (out_scd / f"{pid}.png") if task == "scd" else (out_pfd / f"{pid}.png")
        plot_patient_explainability(
            out_png=out_png,
            pid=pid,
            task=task,
            baseline_prob=baseline_prob,
            deltas=deltas,
            ecg_segment=raw_seg,
            fs=args.fs,
            gate_summary=gsum,
            modality_summary=msum,
        )
        print(f"[OK] pid={pid} task={task} -> {out_png}")

    # Process all SCD positives
    for pid in scd_pids:
        process_pid(pid, task="scd")

    # Process all PFD positives
    for pid in pfd_pids:
        process_pid(pid, task="pfd")

    print("Done.")


if __name__ == "__main__":
    main()