#!/usr/bin/env python3

"""
Full-pipeline block occlusion for MIL-TCN + Vector Gating model.
Performs ECG-level occlusion BEFORE attention pooling.
Create perturbation-based saliency maps.

Example:
python compute_occlusion_and_rank_blocks.py \
  --val_fold 2 \
  --mil_fold_dir ../../music/best_results/tcn_ecg_embeddings/val_fold_2 \
  --vg_fold_dir ../../music/best_results/vectorgating_embeddings_LLaMA8B_BioBERT/val_fold_2 \
  --text_val_npz ../../music/best_text_embeddings_LLaMA8B_BioBERT/val_fold_2/val_embeddings.npz \
  --features_dir ../../music/preprocessed_segments_HRV_complete \
  --block_size 120 \
  --out_dir occlusion_fold2

"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# IMPORT YOUR MODELS (copy exact definitions from training)
# ============================================================

# --- You must paste:
# 1. FeatureEncoder
# 2. TCN
# 3. AttentionMIL
# 4. MILTCNFeatureMultiBranch
# 5. VectorGatingMultiHead
#
# (Use EXACT same class definitions from your training scripts)
#
# For brevity here I assume you pasted them.

class FeatureEncoder(nn.Module):
    """
    Per-segment feature encoder: (F) -> (emb)
    """
    def __init__(self, in_dim: int, emb_dim: int, hidden: int = 128, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return self.net(x)  # (N, emb)


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
        # T: (1, C, N)
        X = T.squeeze(0).transpose(0, 1)  # (N, C)
        a = torch.softmax(self.attn(X).squeeze(1), dim=0)  # (N,)
        z = (X * a.unsqueeze(1)).sum(dim=0)  # (C,)
        return z, a


class MILTCNFeatureMultiBranch(nn.Module):
    """
    Shared temporal feature encoder + MIL pooling, then task-specific branches.

    Forward returns:
      z_scd_logit, z_pfd_logit, z_shared, attn_weights
    """
    def __init__(
        self,
        in_feat_dim: int,
        emb: int,
        hid: int,
        layers: int,
        k: int,
        drop: float,
        attn: int,
        branch_hidden: int | None = None,
        branch_drop: float | None = None,
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
        """
        X: (N_segments, F)
        """
        H = self.encoder(X)                    # (N, emb)
        H = H.transpose(0, 1).unsqueeze(0)     # (1, emb, N)
        T = self.tcn(H)                        # (1, hid, N)
        z, a = self.pool(T)                    # z: (hid,)

        z_scd = self.scd_branch(z)
        z_pfd = self.pfd_branch(z)

        return self.head_scd(z_scd), self.head_pfd(z_pfd), z, a

class VectorGatingMultiHead(nn.Module):
    """
    Vector gating multimodal classifier.

    Gate is computed per sample:
        g = sigmoid( W_g [h_ecg; h_text] + b_g )  in R^{proj_dim}
    Fusion:
        z_fused = g ⊙ h_ecg + (1-g) ⊙ h_text
    """

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

        # ---------------------------------------
        # Projections
        # ---------------------------------------

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

        # ---------------------------------------
        # Vector gate
        # ---------------------------------------

        self.gate = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
        )

        # ---------------------------------------
        # Configurable fusion trunk
        # ---------------------------------------

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
        h_ecg = self.ecg_proj(ecg)    # (B, proj_dim)
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


# ============================================================
# Utilities
# ============================================================

def load_feature_cols(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_text_embeddings(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["pids"].astype(str),
        data["z"],
        data["y_scd"],
        data["y_pfd"],
    )


def normalize_features(X, mean, std):
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)

def infer_mil_hparams_from_state_dict(sd: dict):
    """
    Infer MIL model hyperparameters from checkpoint shapes.
    Works even if checkpoint didn't save args.
    """
    # encoder.net.0: Linear(in_feat_dim -> enc_hidden)
    # encoder.net.3: Linear(enc_hidden -> emb)
    w0 = sd["encoder.net.0.weight"]  # (enc_hidden, in_feat_dim)
    w3 = sd["encoder.net.3.weight"]  # (emb, enc_hidden)
    enc_hidden = int(w0.shape[0])
    in_feat_dim = int(w0.shape[1])
    emb = int(w3.shape[0])

    # first temporal block conv weight: (hid, emb, k)
    wconv0 = sd["tcn.net.0.conv.weight"]
    hid = int(wconv0.shape[0])
    k = int(wconv0.shape[2])

    # number of layers = count TemporalBlocks
    # keys look like tcn.net.{i}.conv.weight
    layers = 0
    while f"tcn.net.{layers}.conv.weight" in sd:
        layers += 1

    # attention hidden dim
    # attn.0: Linear(hid -> attn_dim)
    attn_dim = int(sd["pool.attn.0.weight"].shape[0])

    # dropout / enc_drop are not in state_dict; must be provided or defaulted
    return {
        "in_feat_dim": in_feat_dim,
        "enc_hidden": enc_hidden,
        "emb": emb,
        "hid": hid,
        "layers": layers,
        "k": k,
        "attn": attn_dim,
    }
    
# ============================================================
# Occlusion
# ============================================================

@torch.no_grad()
def compute_occlusion_for_patient(
    mil_model,
    vg_model,
    segment_features,
    text_embedding,
    task,
    block_size,
    device,
):
    mil_model.eval()
    vg_model.eval()

    X = torch.tensor(segment_features, dtype=torch.float32).to(device)

    # Baseline ECG embedding
    z_scd_logit, z_pfd_logit, z_ecg, _ = mil_model(X)

    # Expand text
    text_t = torch.tensor(text_embedding,
                          dtype=torch.float32).to(device).unsqueeze(0)

    z_scd_mm, z_pfd_mm, _, _ = vg_model(z_ecg.unsqueeze(0), text_t)

    if task == "scd":
        baseline_p = torch.sigmoid(z_scd_mm).item()
    else:
        baseline_p = torch.sigmoid(z_pfd_mm).item()

    n_segments = X.size(0)
    results = []

    for start in range(0, n_segments, block_size):
        end = min(start + block_size, n_segments)

        X_occ = X.clone()
        X_occ[start:end] = 0.0

        z_scd_occ, z_pfd_occ, z_ecg_occ, _ = mil_model(X_occ)
        z_scd_mm_occ, z_pfd_mm_occ, _, _ = vg_model(z_ecg_occ.unsqueeze(0), text_t)

        if task == "scd":
            p_occ = torch.sigmoid(z_scd_mm_occ).item()
        else:
            p_occ = torch.sigmoid(z_pfd_mm_occ).item()

        delta = baseline_p - p_occ

        results.append({
            "start_segment": int(start),
            "end_segment": int(end),
            "delta_p": float(delta),
        })

    # ranked blocks (largest effect first)
    ranked = sorted(results, key=lambda x: abs(x["delta_p"]), reverse=True)

    # time-ordered strip (in the same order blocks were occluded)
    strip = np.array([r["delta_p"] for r in results], dtype=np.float32)

    return baseline_p, ranked, strip

def save_occlusion_strip_png(strip: np.ndarray, out_png: str, block_size: int):
    """
    strip: (n_blocks,) signed delta_p = baseline_p - p_occ
    block_size: number of 30-sec segments per block (e.g., 120 = 1 hour)

    X-axis will display integer hour indices.
    """

    strip = np.asarray(strip).astype(np.float32)
    n_blocks = len(strip)

    # Convert block index -> hour index
    # If block_size=120 and each segment=30s:
    # 120 segments * 30s = 3600s = 1 hour
    # So each block corresponds to 1 hour.
    hour_indices = np.arange(n_blocks)

    # Create 1xN image
    img = strip.reshape(1, -1)

    # Stronger symmetric contrast scaling
    vmax = float(np.percentile(np.abs(strip), 98))  # robust scaling
    if vmax < 1e-12:
        vmax = 1e-12

    plt.figure(figsize=(14, 2))

    im = plt.imshow(
        img,
        aspect="auto",
        cmap="seismic",   # high contrast red-blue
        vmin=-vmax,
        vmax=vmax
    )

    plt.yticks([])

    # ---- Integer hour ticks only ----
    plt.xticks(
        ticks=np.arange(n_blocks),
        labels=hour_indices,
        rotation=0
    )

    plt.xlabel("Hour")

    # ---- Clean colorbar ----
    cbar = plt.colorbar(im, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("", rotation=0)

    # Remove title
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--val_fold", type=int, required=True)

    parser.add_argument("--mil_fold_dir", type=str, required=True)
    parser.add_argument("--vg_fold_dir", type=str, required=True)
    parser.add_argument("--text_val_npz", type=str, required=True)

    parser.add_argument("--features_dir", type=str, required=True)

    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Load MIL artifacts
    # ---------------------------------------------------------

    mil_ckpt = torch.load(
        os.path.join(args.mil_fold_dir, "best_model.pt"),
        map_location=device
    )

    mean = np.load(os.path.join(args.mil_fold_dir, "feature_mean.npy"))
    std = np.load(os.path.join(args.mil_fold_dir, "feature_std.npy"))
    feature_cols = load_feature_cols(
        os.path.join(args.mil_fold_dir, "feature_cols.txt")
    )

    sd_mil = mil_ckpt["model_state_dict"]
    
    hp = infer_mil_hparams_from_state_dict(sd_mil)
    
    # IMPORTANT:
    # - in_feat_dim must match feature_cols length (sanity check)
    # - drop/enc_drop are not in state_dict; use your training defaults
    mil_model = MILTCNFeatureMultiBranch(
        in_feat_dim=len(feature_cols),
        emb=hp["emb"],
        hid=hp["hid"],
        layers=hp["layers"],
        k=hp["k"],
        drop=0.2,         # set to whatever you used in training (tcn_dropout)
        attn=hp["attn"],
        enc_hidden=hp["enc_hidden"],
        enc_drop=0.1,     # set to whatever you used in training (enc_dropout)
    ).to(device)
    
    mil_model.load_state_dict(sd_mil, strict=True)

    # ---------------------------------------------------------
    # Load VG model
    # ---------------------------------------------------------

    vg_ckpt = torch.load(
        os.path.join(args.vg_fold_dir, "best_model.pt"),
        map_location=device
    )

    sd_vg = vg_ckpt["model_state_dict"]
    
    text_dim = int(sd_vg["text_proj.1.weight"].shape[1])
    proj_dim = int(sd_vg["ecg_proj.1.weight"].shape[0])     # out dim of ecg proj
    ecg_dim = int(sd_vg["ecg_proj.1.weight"].shape[1])      # in dim expected by VG (should match MIL emb)
    hidden_dim = int(sd_vg["head_scd.weight"].shape[1])     # trunk output dim
    
    # Infer num_layers from trunk Linear layers:
    # trunk = [LayerNorm, (Linear/ReLU/Dropout)*num_layers]
    # Count how many Linear modules exist inside trunk.
    num_trunk_linears = sum(1 for k in sd_vg.keys() if k.startswith("trunk.") and k.endswith(".weight"))
    # trunk has 1 LayerNorm weight + N Linear weights (but LayerNorm has weight too)
    # safer: num_layers = number of "trunk.{i}.weight" with 2D shape.
    num_layers = 0
    for k, v in sd_vg.items():
        if k.startswith("trunk.") and k.endswith(".weight") and v.ndim == 2:
            num_layers += 1
    
    vg_model = VectorGatingMultiHead(
        ecg_dim=ecg_dim,
        text_dim=text_dim,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        dropout=0.2,          # if you used different dropout, set it here
        num_layers=num_layers
    ).to(device)
    
    # vg_model.load_state_dict(sd_vg, strict=True)

    vg_model.load_state_dict(vg_ckpt["model_state_dict"])

    # ---------------------------------------------------------
    # Load text embeddings
    # ---------------------------------------------------------

    pids, z_text_all, y_scd, y_pfd = load_text_embeddings(args.text_val_npz)

    # ---------------------------------------------------------
    # Identify positives
    # ---------------------------------------------------------

    scd_indices = np.where(y_scd == 1)[0]
    pfd_indices = np.where(y_pfd == 1)[0]

    print(f"SCD patients: {len(scd_indices)}")
    print(f"PFD patients: {len(pfd_indices)}")

    # ---------------------------------------------------------
    # Loop patients
    # ---------------------------------------------------------

    for idx in list(scd_indices) + list(pfd_indices):

        pid = str(pids[idx]).zfill(4)
        task = "scd" if y_scd[idx] == 1 else "pfd"

        print(f"\nProcessing {task.upper()} patient {pid}")

        csv_path = Path(args.features_dir) / pid / f"{pid}_segment_features.csv"
        df = pd.read_csv(csv_path)
        df = df.sort_values("window_idx")

        X = df[feature_cols].to_numpy(dtype=np.float32)
        X = normalize_features(X, mean, std)

        baseline_p, ranked, strip = compute_occlusion_for_patient(
            mil_model,
            vg_model,
            X,
            z_text_all[idx],
            task,
            args.block_size,
            device
        )

        patient_dir = os.path.join(args.out_dir, f"{task.upper()}_{pid}")
        os.makedirs(patient_dir, exist_ok=True)

        with open(os.path.join(patient_dir, "occlusion.json"), "w") as f:
            json.dump({
                "baseline_probability": baseline_p,
                "ranked_blocks": ranked
            }, f, indent=4)

        # Save raw strip for later plotting/aggregation
        np.save(os.path.join(patient_dir, "occlusion_strip.npy"), strip)

        # Save strip image
        save_occlusion_strip_png(
            strip,
            os.path.join(patient_dir, "occlusion_strip.png"),
            block_size=args.block_size
        )

    print("\nAll occlusion analyses complete.")


if __name__ == "__main__":
    main()