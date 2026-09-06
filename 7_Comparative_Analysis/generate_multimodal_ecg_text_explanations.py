#!/usr/bin/env python3
"""Generate patient-level multimodal ECG/text explanation artifacts.

This script is adapted from the historical plotting workflow in
``/home/sswee/predicting_cardiac_death/7_Tables_Figures`` for plotting logic
only. Its default inputs point to result roots produced by scripts in this
repository, not to the historical source repository.

The default current-analysis roots are:

* ``/home/sswee/music/ecg_nested_4year_three_wave`` from
  ``3_ECG_Modeling``;
* ``/home/sswee/music/text_embeddings_4year_v4_detailed`` and
  ``/home/sswee/music/text_nested_4year_v4_detailed`` from
  ``4_LLM_Modeling``; and
* ``/home/sswee/music/multimodal_nested_4year_v4_detailed`` from
  ``6_Multimodal_Modeling``.

It generates, for selected validation folds and endpoint/fusion-method pairs:

* ECG occlusion saliency strips: ``ecg_saliency.{png,svg}``;
* ECG encoder attention strips: ``ecg_attention.{png,svg}``;
* local ECG/HRV context panels:
  ``top_ecg_segments*.{csv,png,svg}``; and
* text saliency HTML/CSV from the LLM response through the frozen language
  model, text classifier, and multimodal fusion checkpoint.

The default targets use direct concatenation for both outcomes. Other saved
fusion arms may be requested explicitly.
"""

import argparse
import html
import importlib.util
import json
import re
import sys
from types import SimpleNamespace
from typing import Any
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MUSIC_ROOT = PROJECT_ROOT.parent / "music"
CURRENT_ECG_ROOT = MUSIC_ROOT / "ecg_nested_4year_three_wave"
CURRENT_TEXT_EMBEDDING_ROOT = MUSIC_ROOT / "text_embeddings_4year_v4_detailed"
CURRENT_TEXT_RESULTS_ROOT = MUSIC_ROOT / "text_nested_4year_v4_detailed"
CURRENT_MULTIMODAL_ROOT = MUSIC_ROOT / "multimodal_nested_4year_v4_detailed"
CURRENT_EXPLANATION_ROOT = CURRENT_MULTIMODAL_ROOT / "explain_multimodal_ecg_text"

METHOD_CHOICES = ("concat", "projectconcat", "vector_gating", "scalar_gating")
METHOD_TO_CURRENT_ARM = {
    "concat": "concatenation",
    "projectconcat": "projected_concatenation",
    "vector_gating": "vector_gating",
    "scalar_gating": "scalar_gating",
}
ENDPOINT_CHOICES = ("SCD", "PFD")
DEFAULT_TARGETS = (
    ("SCD", "concat"),
    ("PFD", "concat"),
)
LM_CHECKPOINTS = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
}
LLM_RESPONSE_FILES = {
    "LLaMA3.1-8B": MUSIC_ROOT / "llm_responses_4year_v3_detailed" / "LLaMA3.1-8B-4year-responses.csv",
    "LLaMA3.2-3B": MUSIC_ROOT / "llm_responses_4year_v3_detailed" / "LLaMA3.2-3B-4year-responses-postprocessed.csv",
}
ECG_FEATURES_BY_ENDPOINT = {
    "SCD": [
        "hr_mean",
        "HRV_SDNN",
        "HRV_RMSSD",
        "HRV_pNN50",
        "longest_rr_pause",
        "pvc_burden_pct",
    ],
    "PFD": [
        "hr_mean",
        "HRV_SDNN",
        "HRV_RMSSD",
        "HRV_pNN50",
        "longest_rr_pause",
        "pvc_burden_pct",
    ],
}
FEATURE_LABELS = {
    "hr_mean": "Mean Heart Rate",
    "HRV_SDNN": "SDNN",
    "HRV_RMSSD": "RMSSD",
    "HRV_pNN50": "pNN50",
    "longest_rr_pause": "Longest RR Pause",
    "pvc_burden_pct": "PVC Burden (%)",
}


def safe_zfill_pid(pid) -> str:
    s = str(pid).strip()
    if s.isdigit():
        return s.zfill(4)
    match = re.search(r"(\d+)$", s)
    return match.group(1).zfill(4) if match else s


def load_embedding_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    pids = np.array([safe_zfill_pid(pid) for pid in data["pids"].astype(str)])
    z = data["z"].astype(np.float32)
    y_scd = data["y_scd"].astype(np.int64)
    y_pfd = data["y_pfd"].astype(np.int64)
    return pids, z, y_scd, y_pfd


def align_modalities_by_pid(
    pids_ecg: np.ndarray,
    X_ecg: np.ndarray,
    y_scd_ecg: np.ndarray,
    y_pfd_ecg: np.ndarray,
    pids_text: np.ndarray,
    X_text: np.ndarray,
    y_scd_text: np.ndarray,
    y_pfd_text: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ecg_index = {pid: i for i, pid in enumerate(pids_ecg)}
    text_index = {pid: i for i, pid in enumerate(pids_text)}
    shared_pids = [pid for pid in pids_ecg if pid in text_index]

    if not shared_pids:
        raise RuntimeError("No shared patient IDs were found between ECG and text validation embeddings.")

    ecg_rows = np.array([ecg_index[pid] for pid in shared_pids], dtype=np.int64)
    text_rows = np.array([text_index[pid] for pid in shared_pids], dtype=np.int64)
    y_scd = y_scd_ecg[ecg_rows]
    y_pfd = y_pfd_ecg[ecg_rows]

    if not np.array_equal(y_scd, y_scd_text[text_rows]) or not np.array_equal(y_pfd, y_pfd_text[text_rows]):
        raise RuntimeError("ECG/text labels disagree after aligning by patient ID.")

    return np.array(shared_pids), X_ecg[ecg_rows], X_text[text_rows], y_scd, y_pfd


def parse_targets(tokens: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    if not tokens:
        return list(DEFAULT_TARGETS)
    parsed = []
    for token in tokens:
        endpoint, method = token.split(":", 1)
        endpoint = endpoint.strip().upper()
        method = method.strip().lower()
        if endpoint not in ENDPOINT_CHOICES:
            raise ValueError(f"Invalid endpoint in target '{token}'")
        if method not in METHOD_CHOICES:
            raise ValueError(f"Invalid method in target '{token}'")
        parsed.append((endpoint, method))
    return parsed


def infer_num_layers_from_trunk(state_dict: Dict[str, torch.Tensor]) -> int:
    linear_ids = []
    for key, tensor in state_dict.items():
        if key.startswith("trunk.") and key.endswith(".weight") and tensor.ndim == 2:
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                linear_ids.append(int(parts[1]))
    return len(linear_ids)


class ConcatMultiHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head_scd = nn.Linear(hidden_dim, 1)
        self.head_pfd = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        return self.head_scd(z).squeeze(-1), self.head_pfd(z).squeeze(-1), z


class ProjectedConcatMultiHead(nn.Module):
    def __init__(self, ecg_dim: int, text_dim: int, proj_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.ecg_proj = nn.Sequential(nn.LayerNorm(ecg_dim), nn.Linear(ecg_dim, proj_dim), nn.ReLU())
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, proj_dim), nn.ReLU())
        layers: List[nn.Module] = [nn.LayerNorm(proj_dim * 2)]
        in_dim = proj_dim * 2
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head_scd = nn.Linear(hidden_dim, 1)
        self.head_pfd = nn.Linear(hidden_dim, 1)

    def forward(self, ecg: torch.Tensor, text: torch.Tensor):
        h_ecg = self.ecg_proj(ecg)
        h_text = self.text_proj(text)
        z = self.trunk(torch.cat([h_ecg, h_text], dim=1))
        return self.head_scd(z).squeeze(-1), self.head_pfd(z).squeeze(-1), z


class VectorGatingMultiHead(nn.Module):
    def __init__(self, ecg_dim: int, text_dim: int, proj_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.ecg_proj = nn.Sequential(nn.LayerNorm(ecg_dim), nn.Linear(ecg_dim, proj_dim), nn.ReLU())
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, proj_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.LayerNorm(proj_dim * 2), nn.Linear(proj_dim * 2, proj_dim))
        layers: List[nn.Module] = [nn.LayerNorm(proj_dim)]
        in_dim = proj_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head_scd = nn.Linear(hidden_dim, 1)
        self.head_pfd = nn.Linear(hidden_dim, 1)

    def forward(self, ecg: torch.Tensor, text: torch.Tensor):
        h_ecg = self.ecg_proj(ecg)
        h_text = self.text_proj(text)
        g = torch.sigmoid(self.gate(torch.cat([h_ecg, h_text], dim=1)))
        z = self.trunk(g * h_ecg + (1.0 - g) * h_text)
        return self.head_scd(z).squeeze(-1), self.head_pfd(z).squeeze(-1), z, g


class ScalarGatingMultiHead(nn.Module):
    def __init__(self, ecg_dim: int, text_dim: int, proj_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.ecg_proj = nn.Sequential(nn.LayerNorm(ecg_dim), nn.Linear(ecg_dim, proj_dim), nn.ReLU())
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, proj_dim), nn.ReLU())
        self.alpha = nn.Parameter(torch.tensor(0.0))
        layers: List[nn.Module] = [nn.LayerNorm(proj_dim)]
        in_dim = proj_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head_scd = nn.Linear(hidden_dim, 1)
        self.head_pfd = nn.Linear(hidden_dim, 1)

    def forward(self, ecg: torch.Tensor, text: torch.Tensor):
        h_ecg = self.ecg_proj(ecg)
        h_text = self.text_proj(text)
        g = torch.sigmoid(self.alpha)
        z = self.trunk(g * h_ecg + (1.0 - g) * h_text)
        return self.head_scd(z).squeeze(-1), self.head_pfd(z).squeeze(-1), z, g


def build_multimodal_model(method: str, checkpoint_path: Path, ecg_dim: int, text_dim: int) -> nn.Module:
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload

    if method == "concat":
        input_dim = int(state["trunk.0.weight"].shape[0])
        hidden_dim = int(state["head_scd.weight"].shape[1])
        model = ConcatMultiHead(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=infer_num_layers_from_trunk(state))
    elif method == "projectconcat":
        model = ProjectedConcatMultiHead(
            ecg_dim=ecg_dim,
            text_dim=text_dim,
            proj_dim=int(state["ecg_proj.1.weight"].shape[0]),
            hidden_dim=int(state["head_scd.weight"].shape[1]),
            num_layers=infer_num_layers_from_trunk(state),
        )
    elif method == "vector_gating":
        model = VectorGatingMultiHead(
            ecg_dim=ecg_dim,
            text_dim=text_dim,
            proj_dim=int(state["ecg_proj.1.weight"].shape[0]),
            hidden_dim=int(state["head_scd.weight"].shape[1]),
            num_layers=infer_num_layers_from_trunk(state),
        )
    elif method == "scalar_gating":
        model = ScalarGatingMultiHead(
            ecg_dim=ecg_dim,
            text_dim=text_dim,
            proj_dim=int(state["ecg_proj.1.weight"].shape[0]),
            hidden_dim=int(state["head_scd.weight"].shape[1]),
            num_layers=infer_num_layers_from_trunk(state),
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    model.load_state_dict(state)
    model.eval()
    return model


class LinearMultiHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head_scd = nn.Linear(input_dim, 1)
        self.head_pfd = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        if return_embedding:
            return x
        return self.head_scd(x).squeeze(-1), self.head_pfd(x).squeeze(-1)


class MLPMultiHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float = 0.0):
        super().__init__()
        modules: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(layers):
            modules.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*modules)
        self.head_scd = nn.Linear(hidden_dim, 1)
        self.head_pfd = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.trunk(x)
        if return_embedding:
            return z
        return self.head_scd(z).squeeze(-1), self.head_pfd(z).squeeze(-1)


def build_text_model(checkpoint_path: Path) -> nn.Module:
    payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload

    if any(k.startswith("trunk.") for k in state):
        first_linear = state["trunk.0.weight"]
        input_dim = int(first_linear.shape[1])
        hidden_dim = int(first_linear.shape[0])
        layers = sum(1 for k, v in state.items() if k.startswith("trunk.") and k.endswith(".weight") and v.ndim == 2)
        model = MLPMultiHead(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers)
    else:
        input_dim = int(state["head_scd.weight"].shape[1])
        model = LinearMultiHead(input_dim=input_dim)

    model.load_state_dict(state)
    model.eval()
    return model


class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, hidden: int = 128, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, d: int, drop: float):
        super().__init__()
        pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor):
        y = self.drop(self.relu(self.conv(x)))[:, :, :x.size(2)]
        return y + (x if self.down is None else self.down(x))


class TCN(nn.Module):
    def __init__(self, in_dim: int, hid: int, layers: int, k: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(*[TemporalBlock(in_dim if i == 0 else hid, hid, k, 2 ** i, drop) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.net(x)


class AttentionMIL(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))

    def forward(self, T: torch.Tensor):
        X = T.squeeze(0).transpose(0, 1)
        a = torch.softmax(self.attn(X).squeeze(1), dim=0)
        z = (X * a.unsqueeze(1)).sum(dim=0)
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
        self.scd_branch = nn.Sequential(nn.Linear(hid, bh), nn.ReLU(), nn.Dropout(bd), nn.Linear(bh, hid), nn.ReLU())
        self.pfd_branch = nn.Sequential(nn.Linear(hid, bh), nn.ReLU(), nn.Dropout(bd), nn.Linear(bh, hid), nn.ReLU())
        self.head_scd = nn.Linear(hid, 1)
        self.head_pfd = nn.Linear(hid, 1)

    def forward(self, X: torch.Tensor):
        H = self.encoder(X)
        H = H.transpose(0, 1).unsqueeze(0)
        T = self.tcn(H)
        z, a = self.pool(T)
        z_scd = self.scd_branch(z)
        z_pfd = self.pfd_branch(z)
        return self.head_scd(z_scd), self.head_pfd(z_pfd), z, a


def infer_mil_hparams_from_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    w0 = state["encoder.net.0.weight"]
    w3 = state["encoder.net.3.weight"]
    wconv0 = state["tcn.net.0.conv.weight"]
    layers = 0
    while f"tcn.net.{layers}.conv.weight" in state:
        layers += 1
    return {
        "in_feat_dim": int(w0.shape[1]),
        "enc_hidden": int(w0.shape[0]),
        "emb": int(w3.shape[0]),
        "hid": int(wconv0.shape[0]),
        "layers": int(layers),
        "k": int(wconv0.shape[2]),
        "attn": int(state["pool.attn.0.weight"].shape[0]),
        "branch_hidden": int(state["scd_branch.0.weight"].shape[0]),
    }


def build_mil_model(ecg_encoder_fold_dir: Path, device: torch.device) -> Tuple[nn.Module, np.ndarray, np.ndarray, List[str]]:
    checkpoint = load_trusted_checkpoint(
        ecg_encoder_fold_dir / "best_model.pt", map_location=device
    )
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    hp = infer_mil_hparams_from_state_dict(state)

    mean = np.load(ecg_encoder_fold_dir / "feature_mean.npy")
    std = np.load(ecg_encoder_fold_dir / "feature_std.npy")
    feature_cols = [line.strip() for line in (ecg_encoder_fold_dir / "feature_cols.txt").read_text().splitlines() if line.strip()]

    model = MILTCNFeatureMultiBranch(
        in_feat_dim=len(feature_cols),
        emb=hp["emb"],
        hid=hp["hid"],
        layers=hp["layers"],
        k=hp["k"],
        drop=0.3,
        attn=hp["attn"],
        branch_hidden=hp["branch_hidden"],
        enc_hidden=hp["enc_hidden"],
        enc_drop=0.0,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, mean.astype(np.float32), std.astype(np.float32), feature_cols


def predict_endpoint_probs(model: nn.Module, method: str, endpoint: str, X_ecg: np.ndarray, X_text: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        ecg_t = torch.tensor(X_ecg, dtype=torch.float32, device=device)
        text_t = torch.tensor(X_text, dtype=torch.float32, device=device)
        outputs = model(torch.cat([ecg_t, text_t], dim=1)) if method == "concat" else model(ecg_t, text_t)
        logits = outputs[0] if endpoint == "SCD" else outputs[1]
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def select_case_indices(y_true: np.ndarray, y_prob: np.ndarray, top_k: int) -> Tuple[np.ndarray, str]:
    pos_idx = np.flatnonzero(y_true == 1)
    if pos_idx.size > 0:
        ranked = pos_idx[np.argsort(-y_prob[pos_idx])]
        return (ranked[:top_k] if top_k > 0 else ranked), "positive"
    ranked = np.argsort(-y_prob)
    return (ranked[:top_k] if top_k > 0 else ranked), "top_pred"


def find_raw_npz(preprocessed_dir: Path, pid: str) -> Path:
    candidates = sorted(p for p in preprocessed_dir.iterdir() if p.suffix == ".npz" and pid in p.name)
    if not candidates:
        raise FileNotFoundError(f"Raw ECG NPZ not found for patient {pid} in {preprocessed_dir}")
    return candidates[0]


def normalize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xn = (X - mean) / std
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn.astype(np.float32)


def compute_multimodal_prob(model: nn.Module, method: str, endpoint: str, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    outputs = model(torch.cat([ecg_embedding, text_embedding], dim=1)) if method == "concat" else model(ecg_embedding, text_embedding)
    logit = outputs[0] if endpoint == "SCD" else outputs[1]
    return torch.sigmoid(logit)


@torch.no_grad()
def compute_ecg_segment_occlusion(
    mil_model: nn.Module,
    multimodal_model: nn.Module,
    method: str,
    endpoint: str,
    segment_features: np.ndarray,
    text_embedding: np.ndarray,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    X = torch.tensor(segment_features, dtype=torch.float32, device=device)
    text_t = torch.tensor(text_embedding[None, :], dtype=torch.float32, device=device)

    _, _, z_ecg, attn = mil_model(X)
    baseline_p = float(compute_multimodal_prob(multimodal_model, method, endpoint, z_ecg.unsqueeze(0), text_t).item())

    deltas = np.zeros(X.size(0), dtype=np.float32)
    for idx in range(X.size(0)):
        X_occ = X.clone()
        X_occ[idx] = 0.0
        _, _, z_ecg_occ, _ = mil_model(X_occ)
        p_occ = float(compute_multimodal_prob(multimodal_model, method, endpoint, z_ecg_occ.unsqueeze(0), text_t).item())
        deltas[idx] = baseline_p - p_occ

    return baseline_p, deltas, attn.detach().cpu().numpy().astype(np.float32)


def plot_ecg_saliency_strip(save_path: Path, deltas: np.ndarray, endpoint: str, method: str, pid: str):
    vmax = max(float(np.percentile(np.abs(deltas), 98)), 1e-8)
    fig, ax = plt.subplots(figsize=(14, 2.4))
    im = ax.imshow(deltas.reshape(1, -1), aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_yticks([])
    hour_ticks = np.arange(0, len(deltas), 120)
    if len(deltas) > 0 and (len(hour_ticks) == 0 or hour_ticks[-1] != len(deltas) - 1):
        hour_ticks = np.unique(np.append(hour_ticks, len(deltas) - 1))
    hour_labels = [f"{tick / 120.0:.1f}" for tick in hour_ticks]
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels)
    ax.set_xlabel("Time (hours)")
    ax.set_title(f"{pid} | {endpoint} | {method} | ECG occlusion saliency")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Delta predicted risk\n(baseline - occluded)", rotation=90)
    fig.tight_layout()
    save_figure_dual(fig, save_path)
    plt.close(fig)


def plot_ecg_attention_strip(save_path: Path, attention: np.ndarray, endpoint: str, method: str, pid: str):
    vmax = max(float(np.percentile(np.abs(attention), 98)), 1e-8)
    fig, ax = plt.subplots(figsize=(14, 2.4))
    im = ax.imshow(attention.reshape(1, -1), aspect="auto", cmap="Reds", vmin=0.0, vmax=vmax)
    ax.set_yticks([])
    hour_ticks = np.arange(0, len(attention), 120)
    if len(attention) > 0 and (len(hour_ticks) == 0 or hour_ticks[-1] != len(attention) - 1):
        hour_ticks = np.unique(np.append(hour_ticks, len(attention) - 1))
    hour_labels = [f"{tick / 120.0:.1f}" for tick in hour_ticks]
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_labels)
    ax.set_xlabel("Time (hours)")
    ax.set_title(f"{pid} | {endpoint} | {method} | ECG attention")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Attention Weight", rotation=90)
    fig.tight_layout()
    save_figure_dual(fig, save_path)
    plt.close(fig)


def format_feature_summary(row: pd.Series, endpoint: str) -> str:
    parts = []
    for col in ECG_FEATURES_BY_ENDPOINT[endpoint]:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                label = FEATURE_LABELS.get(col, col)
                parts.append(f"{label}: {float(val):.3f}")
    return "\n".join(parts)


def saliency_norm(deltas: np.ndarray):
    deltas = np.asarray(deltas, dtype=np.float32)
    dmin = float(np.min(deltas))
    dmax = float(np.max(deltas))
    if dmin < 0.0 and dmax > 0.0:
        return TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax)
    if dmax <= 0.0:
        return Normalize(vmin=dmin, vmax=0.0 if dmin < 0.0 else 1.0)
    return Normalize(vmin=0.0, vmax=dmax if dmax > 0.0 else 1.0)


def score_norm(scores: np.ndarray):
    scores = np.asarray(scores, dtype=np.float32)
    smin = float(np.min(scores))
    smax = float(np.max(scores))
    if smin < 0.0 and smax > 0.0:
        return TwoSlopeNorm(vmin=smin, vcenter=0.0, vmax=smax)
    if smax <= 0.0:
        return Normalize(vmin=smin, vmax=0.0 if smin < 0.0 else 1.0)
    return Normalize(vmin=0.0, vmax=smax if smax > 0.0 else 1.0)


def plot_focal_segment_trace(
    ax,
    raw_signal: np.ndarray,
    context_df: pd.DataFrame,
    fs: int,
    window_sec: int,
    score_col: str,
    score_label: str,
):
    cmap = plt.get_cmap("seismic")
    norm = score_norm(context_df[score_col].to_numpy(dtype=np.float32))

    focal_row = context_df.loc[context_df["is_focal"] == 1].iloc[0]
    start_idx = int(focal_row["start_idx"])
    end_idx = start_idx + fs * window_sec
    snippet = np.asarray(raw_signal[start_idx:end_idx], dtype=np.float32)
    t = np.arange(snippet.size, dtype=np.float32) / fs
    points = np.column_stack([t, snippet]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    color = cmap(norm(float(focal_row[score_col])))
    lc = LineCollection(segments, colors=[color], linewidths=0.8)
    ax.add_collection(lc)

    ax.set_xlim(0, window_sec)
    ymin = float(np.nanmin(snippet))
    ymax = float(np.nanmax(snippet))
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
        pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (sec)")
    ax.set_title(f"Most important 30-second ECG segment colored by {score_label.lower()}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.022, pad=0.015)
    cbar.set_label(score_label)


def plot_feature_trends(ax_list, context_df: pd.DataFrame, endpoint: str, window_sec: int, score_col: str):
    cmap = plt.get_cmap("seismic")
    norm = score_norm(context_df[score_col].to_numpy(dtype=np.float32))
    focal_idx = int(context_df.index[context_df["is_focal"] == 1][0])
    x_minutes = (np.arange(len(context_df), dtype=np.float32) - focal_idx) * (window_sec / 60.0)

    for ax, feature in zip(ax_list, ECG_FEATURES_BY_ENDPOINT[endpoint]):
        vals = context_df[feature].to_numpy(dtype=np.float32)
        ax.plot(x_minutes, vals, color="0.65", linewidth=1.2, zorder=1)
        ax.scatter(
            x_minutes,
            vals,
            c=context_df[score_col].to_numpy(dtype=np.float32),
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.25,
            s=30,
            zorder=2,
        )
        ax.set_ylabel(FEATURE_LABELS.get(feature, feature))
        ax.grid(alpha=0.18, linewidth=0.6)

    return x_minutes


def plot_attention_trend(ax, context_df: pd.DataFrame, x_minutes: np.ndarray, score_col: str):
    cmap = plt.get_cmap("seismic")
    norm = score_norm(context_df[score_col].to_numpy(dtype=np.float32))
    vals = context_df["attention_weight"].to_numpy(dtype=np.float32)
    ax.plot(x_minutes, vals, color="0.65", linewidth=1.2, zorder=1)
    ax.scatter(
        x_minutes,
        vals,
        c=context_df[score_col].to_numpy(dtype=np.float32),
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.25,
        s=30,
        zorder=2,
    )
    ax.set_ylabel("Attention Weight")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_xlabel("Time relative to focal segment (minutes)")


def plot_local_context_window(
    save_path: Path,
    pid: str,
    endpoint: str,
    method: str,
    context_df: pd.DataFrame,
    raw_signal: np.ndarray,
    fs: int,
    window_sec: int,
    score_col: str,
    score_label: str,
    include_attention_panel: bool = False,
):
    n_rows = 8 if include_attention_panel else 7
    height_ratios = [2.4, 1, 1, 1, 1, 1, 1, 1] if include_attention_panel else [2.4, 1, 1, 1, 1, 1, 1]
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(15, 18 if include_attention_panel else 16),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.35},
    )
    plot_focal_segment_trace(axes[0], raw_signal, context_df, fs, window_sec, score_col, score_label)
    feature_axes = axes[1:7]
    x_minutes = plot_feature_trends(feature_axes, context_df, endpoint, window_sec, score_col)
    if include_attention_panel:
        plot_attention_trend(axes[7], context_df, x_minutes, score_col)
    else:
        feature_axes[-1].set_xlabel("Time relative to focal segment (minutes)")

    focal_rows = context_df.loc[context_df["is_focal"] == 1]
    focal_desc = ""
    if not focal_rows.empty:
        focal_row = focal_rows.iloc[0]
        focal_desc = (
            f" | focal window_idx={int(focal_row['window_idx'])}"
            f" | focal {score_col}={float(focal_row[score_col]):.4f}"
        )

    fig.suptitle(f"{pid} | {endpoint} | {method} | 11-segment local ECG/HRV context by {score_label.lower()}{focal_desc}", y=0.995)
    fig.tight_layout()
    save_figure_dual(fig, save_path)
    plt.close(fig)


def save_figure_dual(fig, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    fig.savefig(save_path.with_suffix(".svg"))


def load_reasoning_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["pid"] = df["Patient ID"].astype(str).map(safe_zfill_pid)
    return df


def build_reasoning_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {row["pid"]: str(row["reasoning_text"]) for _, row in df.iterrows()}


def get_lm_checkpoint(lm_model_name: str, override: Optional[str]) -> str:
    if override:
        return override
    if lm_model_name not in LM_CHECKPOINTS:
        raise ValueError(f"Unsupported lm_model_name '{lm_model_name}'. Pass --lm_checkpoint explicitly.")
    return LM_CHECKPOINTS[lm_model_name]


def load_tokenizer_and_model(lm_checkpoint: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(lm_checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(lm_checkpoint, local_files_only=True).to(device)
    model.eval()
    return tokenizer, model


def token_saliency_for_reasoning(
    reasoning_text: str,
    endpoint: str,
    method: str,
    multimodal_model: nn.Module,
    text_model: nn.Module,
    text_embedding_dim: int,
    ecg_embedding: np.ndarray,
    tokenizer,
    lm_model,
    max_length: int,
    device: torch.device,
):
    encoded = tokenizer(
        reasoning_text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoded.pop("offset_mapping")[0].cpu().numpy()
    encoded = {k: v.to(device) for k, v in encoded.items()}

    input_embeds = lm_model.get_input_embeddings()(encoded["input_ids"]).detach().clone()
    input_embeds.requires_grad_(True)

    outputs = lm_model(
        inputs_embeds=input_embeds,
        attention_mask=encoded.get("attention_mask"),
        token_type_ids=encoded.get("token_type_ids"),
    )
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    shared_text = text_model(cls_embedding, return_embedding=True)
    if shared_text.shape[1] != text_embedding_dim:
        raise RuntimeError(f"Text embedding dim mismatch: expected {text_embedding_dim}, got {shared_text.shape[1]}")

    ecg_t = torch.tensor(ecg_embedding[None, :], dtype=torch.float32, device=device)
    prob = compute_multimodal_prob(multimodal_model, method, endpoint, ecg_t, shared_text)
    multimodal_model.zero_grad(set_to_none=True)
    text_model.zero_grad(set_to_none=True)
    lm_model.zero_grad(set_to_none=True)
    prob.backward(torch.ones_like(prob))

    grads = input_embeds.grad.detach()[0]
    embeds = input_embeds.detach()[0]
    token_scores = torch.norm(grads * embeds, dim=1).cpu().numpy().astype(np.float32)

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].detach().cpu().tolist())
    return {
        "prob": float(prob.item()),
        "tokens": tokens,
        "offsets": offset_mapping,
        "scores": token_scores,
        "input_ids": encoded["input_ids"][0].detach().cpu().numpy(),
    }


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros_like(scores)
    vmax = float(scores[finite].max())
    return scores / vmax if vmax > 0 else np.zeros_like(scores)


def merge_salient_spans(reasoning_text: str, offsets: np.ndarray, scores: np.ndarray, top_tokens: int = 30) -> List[Dict[str, float]]:
    valid = []
    for idx, ((start, end), score) in enumerate(zip(offsets, scores)):
        if start == end or score <= 0:
            continue
        valid.append((idx, int(start), int(end), float(score)))

    valid.sort(key=lambda x: x[3], reverse=True)
    chosen = sorted(valid[:top_tokens], key=lambda x: x[1])

    spans: List[Dict[str, float]] = []
    for _, start, end, score in chosen:
        if not spans:
            spans.append({"start": start, "end": end, "score": score})
            continue
        prev = spans[-1]
        gap_text = reasoning_text[prev["end"]:start]
        if start <= prev["end"] + 1 or gap_text.isspace():
            prev["end"] = max(prev["end"], end)
            prev["score"] = max(prev["score"], score)
        else:
            spans.append({"start": start, "end": end, "score": score})
    return spans


def render_highlighted_html(reasoning_text: str, spans: List[Dict[str, float]], title: str) -> str:
    pieces = [
        "<html><head><meta charset='utf-8'><title>",
        html.escape(title),
        "</title></head><body style='font-family: sans-serif; line-height: 1.6; max-width: 1100px; margin: 24px auto;'>",
        f"<h2>{html.escape(title)}</h2>",
        "<p><strong>Whole reasoning text.</strong> Highlight intensity shows endpoint-specific gradient attribution and is not a clinical explanation.</p>",
        "<div style='font-size: 16px; white-space: pre-wrap;'>",
    ]
    cursor = 0
    max_score = max((span["score"] for span in spans), default=1.0)
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        pieces.append(html.escape(reasoning_text[cursor:start]))
        alpha = 0.18 + 0.52 * (span["score"] / max_score if max_score > 0 else 0.0)
        pieces.append(
            f"<mark style='background: rgba(255, 214, 10, {alpha:.3f}); padding: 0 1px;'>"
            f"{html.escape(reasoning_text[start:end])}</mark>"
        )
        cursor = end
    pieces.append(html.escape(reasoning_text[cursor:]))
    pieces.append("</div></body></html>")
    return "".join(pieces)


def save_token_scores_csv(save_path: Path, reasoning_text: str, tokens: List[str], offsets: np.ndarray, scores: np.ndarray):
    rows = []
    for token, (start, end), score in zip(tokens, offsets, scores):
        rows.append(
            {
                "token": token,
                "char_start": int(start),
                "char_end": int(end),
                "text_span": reasoning_text[int(start):int(end)],
                "saliency": float(score),
            }
        )
    pd.DataFrame(rows).to_csv(save_path, index=False)


def load_repo_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_trusted_checkpoint(path: Path, map_location):
    """Load a checkpoint generated by this project, across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def current_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        stage="evaluate",
        folds_csv=args.ecg_encoder_root / "analysis_setup" / "nested_patient_folds.csv",
        ecg_root=args.ecg_encoder_root,
        text_embedding_root=args.text_embedding_dir,
        text_results_root=args.text_model_root,
        tabular_csv=MUSIC_ROOT / "subject-info.csv",
        output_root=args.multimodal_root,
        patient_id_col="Patient ID",
        scd_label_col="SCD_4year_label",
        pfd_label_col="PFD_4year_label",
        outer_fold_col="outer_fold",
        outer_splits=5,
        inner_splits=4,
        expected_patients=730,
        expected_controls=577,
        expected_scd=71,
        expected_pfd=82,
        expected_anticoagulant_yes=610,
        expected_anticoagulant_no=120,
        expected_text_pooling="cls",
        expected_text_max_length=args.max_text_length,
        expected_text_long_strategy="mean_chunks",
        expected_text_long_text_strategy="mean_chunks",
        expected_text_truncated_patient_count=0,
        expected_text_embedding_policy={
            "pooling": "cls",
            "max_length": args.max_text_length,
            "long_text_strategy": "mean_chunks",
            "truncated_patient_count": 0,
        },
    )


def load_current_ecg_model(ecg_module, ecg_root: Path, outer_fold: int, device: torch.device):
    fold_dir = ecg_root / "final_models" / f"outer_fold_{outer_fold}"
    checkpoint_path = fold_dir / "final_checkpoint.pt"
    preprocessing_path = fold_dir / "preprocessing.npz"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not preprocessing_path.exists():
        raise FileNotFoundError(preprocessing_path)
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    feature_cols = [str(value) for value in checkpoint["feature_columns"]]
    model = ecg_module.MILTCNFeatureMultiBranch(len(feature_cols) * 2, checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    preprocessing = np.load(preprocessing_path, allow_pickle=False)
    stored_cols = preprocessing["feature_columns"].astype(str).tolist()
    if stored_cols != feature_cols:
        raise RuntimeError(f"ECG feature columns disagree between {checkpoint_path} and {preprocessing_path}")
    return (
        model,
        preprocessing["means"].astype(np.float32),
        preprocessing["standard_deviations"].astype(np.float32),
        feature_cols,
    )


def current_ecg_features_from_segments(
    frame: pd.DataFrame,
    feature_cols: list[str],
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> np.ndarray:
    values = frame.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    missing = ~np.isfinite(values)
    imputed = np.where(missing, means, values)
    standardized = (imputed - means) / standard_deviations
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([standardized, missing.astype(np.float32)], axis=1)


def selected_text_config(text_results_root: Path, task: str, outer_fold: int) -> dict[str, Any]:
    path = (
        text_results_root
        / "tasks"
        / task
        / "selection"
        / "selected_primary_configuration_by_outer_fold.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data["selected_by_outer_fold"][str(outer_fold)]
    return selected.get("config", selected)


def current_text_for_patient(config: dict[str, Any], task: str, patient_id: str) -> str:
    source = config["source_name"]
    response_path = LLM_RESPONSE_FILES[source]
    if not response_path.exists():
        raise FileNotFoundError(response_path)
    frame = pd.read_csv(response_path, dtype={"Patient ID": "string"})
    frame["pid"] = frame["Patient ID"].map(safe_zfill_pid)
    rows = frame.loc[frame["pid"].eq(patient_id)]
    if rows.empty:
        raise KeyError(f"Patient {patient_id} not found in {response_path}")
    row = rows.iloc[0]
    prefix = "full_risk_no_ecg"
    if task == "scd":
        return f"SCD_RISK: {str(row[f'{prefix}_scd_risk']).title()}\nSCD_RATIONALE: {str(row[f'{prefix}_scd_rationale']).strip()}"
    return f"PFD_RISK: {str(row[f'{prefix}_pfd_risk']).title()}\nPFD_RATIONALE: {str(row[f'{prefix}_pfd_rationale']).strip()}"


def build_current_fusion_model(mm_module, checkpoint_path: Path, device: torch.device):
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    model = mm_module.FusionNetwork(
        int(checkpoint["ecg_dim"]),
        int(checkpoint["second_dim"]),
        checkpoint["config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def current_checkpoint_path(multimodal_root: Path, task: str, outer_fold: int, method: str) -> Path:
    arm = METHOD_TO_CURRENT_ARM[method]
    return (
        multimodal_root
        / "tasks"
        / task
        / "final_models"
        / "ecg_full_text"
        / f"outer_fold_{outer_fold}"
        / "arms"
        / arm
        / "checkpoint.pt"
    )


def current_outer_predictions_path(multimodal_root: Path, task: str, outer_fold: int, method: str) -> Path:
    arm = METHOD_TO_CURRENT_ARM[method]
    return (
        multimodal_root
        / "tasks"
        / task
        / "final_models"
        / "ecg_full_text"
        / f"outer_fold_{outer_fold}"
        / "arms"
        / arm
        / "outer_test_predictions.csv"
    )


def load_current_pair_data(args: argparse.Namespace, mm_module, task: str, outer_fold: int):
    ns = current_args(args)
    folds = mm_module.read_folds(ns, setup_copy=False)
    return mm_module.load_pair_split(ns, folds, "ecg_full_text", task, outer_fold, None, None), folds


def compute_current_multimodal_prob(model: nn.Module, ecg_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    logit, _ = model(ecg_embedding, text_embedding)
    return torch.sigmoid(logit)


@torch.no_grad()
def compute_current_ecg_segment_occlusion(
    ecg_model: nn.Module,
    multimodal_model: nn.Module,
    segment_features: np.ndarray,
    text_embedding: np.ndarray,
    ecg_mean: np.ndarray,
    ecg_sd: np.ndarray,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    X = torch.tensor(segment_features, dtype=torch.float32, device=device)
    text_t = torch.tensor(text_embedding[None, :], dtype=torch.float32, device=device)
    _, _, z_ecg, _, _, attn = ecg_model(X)
    z_std = torch.tensor(((z_ecg.detach().cpu().numpy() - ecg_mean) / ecg_sd)[None, :], dtype=torch.float32, device=device)
    baseline_p = float(compute_current_multimodal_prob(multimodal_model, z_std, text_t).item())
    deltas = np.zeros(X.size(0), dtype=np.float32)
    for idx in range(X.size(0)):
        X_occ = X.clone()
        X_occ[idx] = 0.0
        _, _, z_occ, _, _, _ = ecg_model(X_occ)
        z_occ_std = torch.tensor(((z_occ.detach().cpu().numpy() - ecg_mean) / ecg_sd)[None, :], dtype=torch.float32, device=device)
        p_occ = float(compute_current_multimodal_prob(multimodal_model, z_occ_std, text_t).item())
        deltas[idx] = baseline_p - p_occ
    return baseline_p, deltas, attn.detach().cpu().numpy().astype(np.float32)


def token_saliency_for_current_reasoning(
    reasoning_text: str,
    multimodal_model: nn.Module,
    ecg_embedding: np.ndarray,
    tokenizer,
    lm_model,
    text_mean: np.ndarray,
    text_sd: np.ndarray,
    max_length: int,
    chunk_stride: int,
    device: torch.device,
):
    encoded = tokenizer(
        reasoning_text,
        truncation=True,
        max_length=max_length,
        stride=chunk_stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = encoded.pop("offset_mapping").cpu().numpy().reshape(-1, 2)
    encoded.pop("overflow_to_sample_mapping", None)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_embeds = lm_model.get_input_embeddings()(encoded["input_ids"]).detach().clone()
    input_embeds.requires_grad_(True)
    outputs = lm_model(
        inputs_embeds=input_embeds,
        attention_mask=encoded.get("attention_mask"),
        token_type_ids=encoded.get("token_type_ids"),
    )
    cls_embedding = outputs.last_hidden_state[:, 0, :].mean(dim=0, keepdim=True)
    mean = torch.tensor(text_mean[None, :], dtype=torch.float32, device=device)
    sd = torch.tensor(text_sd[None, :], dtype=torch.float32, device=device)
    text_embedding = (cls_embedding - mean) / sd
    ecg_t = torch.tensor(ecg_embedding[None, :], dtype=torch.float32, device=device)
    prob = compute_current_multimodal_prob(multimodal_model, ecg_t, text_embedding)
    multimodal_model.zero_grad(set_to_none=True)
    lm_model.zero_grad(set_to_none=True)
    prob.backward(torch.ones_like(prob))
    grads = input_embeds.grad.detach()
    embeds = input_embeds.detach()
    token_scores = torch.norm(grads * embeds, dim=2).cpu().numpy().astype(np.float32).reshape(-1)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"].detach().cpu().reshape(-1).tolist())
    return {
        "prob": float(prob.item()),
        "tokens": tokens,
        "offsets": offset_mapping,
        "scores": token_scores,
        "input_ids": encoded["input_ids"].detach().cpu().numpy(),
        "chunk_count": int(encoded["input_ids"].shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate patient-level ECG saliency, ECG attention, local ECG/HRV "
            "context panels, and LLM text saliency for locked multimodal folds."
        )
    )
    parser.add_argument("--val_fold", type=int, required=True, choices=range(5), metavar="{0,1,2,3,4}")
    parser.add_argument("--ecg_embedding_dir", type=Path, default=CURRENT_ECG_ROOT)
    parser.add_argument("--text_embedding_dir", type=Path, default=CURRENT_TEXT_EMBEDDING_ROOT)
    parser.add_argument("--ecg_encoder_root", type=Path, default=CURRENT_ECG_ROOT)
    parser.add_argument("--features_dir", type=Path, default=MUSIC_ROOT / "preprocessed_segments_HRV_complete")
    parser.add_argument("--preprocessed_dir", type=Path, default=MUSIC_ROOT / "preprocessed_HRV")
    parser.add_argument("--text_model_root", type=Path, default=CURRENT_TEXT_RESULTS_ROOT)
    parser.add_argument("--multimodal_root", type=Path, default=CURRENT_MULTIMODAL_ROOT)
    parser.add_argument("--concat_model_dir", type=Path)
    parser.add_argument("--projectconcat_model_dir", type=Path)
    parser.add_argument("--vector_gating_model_dir", type=Path)
    parser.add_argument("--scalar_gating_model_dir", type=Path)
    parser.add_argument("--targets", nargs="*")
    parser.add_argument(
        "--patient_ids",
        nargs="*",
        help=(
            "Optional patient IDs to generate for each requested target. "
            "When omitted, cases are selected by top predicted risk among positives."
        ),
    )
    parser.add_argument("--top_k_cases", type=int, default=5, help="Top K cases per endpoint/method after ranking by predicted risk. 0 means all.")
    parser.add_argument("--top_k_segments", type=int, default=1)
    parser.add_argument("--top_text_tokens", type=int, default=30)
    parser.add_argument("--lm_model_name", type=str, default="BioBERT")
    parser.add_argument("--lm_checkpoint", type=str, default=None)
    parser.add_argument("--max_text_length", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=64)
    parser.add_argument("--window_sec", type=int, default=30)
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--out_dir", type=Path, default=CURRENT_EXPLANATION_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    requested_pids = {safe_zfill_pid(pid) for pid in args.patient_ids or []}
    device = torch.device(args.device)
    ecg_module = load_repo_module(
        "current_ecg_training",
        PROJECT_ROOT / "3_ECG_Modeling" / "train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py",
    )
    mm_module = load_repo_module(
        "current_multimodal_training",
        PROJECT_ROOT / "6_Multimodal_Modeling" / "train_multimodal_nested_cv.py",
    )
    mil_model, feature_mean, feature_std, feature_cols = load_current_ecg_model(
        ecg_module,
        args.ecg_encoder_root,
        args.val_fold,
        device=device,
    )
    tokenizer_cache: dict[tuple[str, str | None], Any] = {}

    summary_rows: List[Dict[str, object]] = []

    for endpoint, method in targets:
        task = endpoint.lower()
        data, _folds = load_current_pair_data(args, mm_module, task, args.val_fold)
        checkpoint_path = current_checkpoint_path(args.multimodal_root, task, args.val_fold, method)
        model, checkpoint = build_current_fusion_model(mm_module, checkpoint_path, device)
        predictions_path = current_outer_predictions_path(args.multimodal_root, task, args.val_fold, method)
        predictions = pd.read_csv(predictions_path, dtype={"Patient ID": "string"})
        predictions["pid"] = predictions["Patient ID"].map(safe_zfill_pid)
        probability_by_pid = dict(zip(predictions["pid"], predictions["prob"].astype(float)))
        pids = np.array([safe_zfill_pid(value) for value in data.patient_ids_validation.astype(str)])
        y_prob = np.array([probability_by_pid[pid] for pid in pids], dtype=np.float32)
        y_true = data.labels_validation.astype(int)
        text_config = selected_text_config(args.text_model_root, task, args.val_fold)
        encoder_name = text_config["encoder_name"]
        tokenizer_key = (encoder_name, args.lm_checkpoint)
        if tokenizer_key not in tokenizer_cache:
            tokenizer_cache[tokenizer_key] = load_tokenizer_and_model(
                get_lm_checkpoint(encoder_name, args.lm_checkpoint),
                device=device,
            )
        tokenizer, lm_model = tokenizer_cache[tokenizer_key]
        if requested_pids:
            pid_to_index = {pid: index for index, pid in enumerate(pids)}
            present_pids = sorted(pid for pid in requested_pids if pid in pid_to_index)
            if not present_pids:
                print(
                    f"No requested patient IDs are present in validation fold {args.val_fold}; "
                    f"skipping {endpoint}:{method}."
                )
                continue
            missing_pids = sorted(requested_pids - set(present_pids))
            if missing_pids:
                print(
                    f"Validation fold {args.val_fold} does not contain requested patient IDs: "
                    + ", ".join(missing_pids)
                )
            case_indices = np.array([pid_to_index[pid] for pid in present_pids], dtype=np.int64)
            selection_mode = "requested_patient_ids"
        else:
            case_indices, selection_mode = select_case_indices(y_true, y_prob, args.top_k_cases)

        for rank, idx in enumerate(case_indices, start=1):
            pid = pids[idx]
            patient_dir = args.out_dir / f"val_fold_{args.val_fold}" / endpoint / method / pid
            patient_dir.mkdir(parents=True, exist_ok=True)

            feature_csv = args.features_dir / pid / f"{pid}_segment_features.csv"
            if not feature_csv.exists():
                raise FileNotFoundError(f"Missing segment feature CSV: {feature_csv}")
            df_segments = pd.read_csv(feature_csv).sort_values("window_idx").reset_index(drop=True)
            X_segments = current_ecg_features_from_segments(df_segments, feature_cols, feature_mean, feature_std)

            baseline_prob, ecg_deltas, attn_weights = compute_current_ecg_segment_occlusion(
                ecg_model=mil_model,
                multimodal_model=model,
                segment_features=X_segments,
                text_embedding=data.second_validation[idx],
                ecg_mean=data.preprocessing["ecg_mean"],
                ecg_sd=data.preprocessing["ecg_sd"],
                device=device,
            )
            plot_ecg_saliency_strip(patient_dir / "ecg_saliency.png", ecg_deltas, endpoint, method, pid)
            plot_ecg_attention_strip(patient_dir / "ecg_attention.png", attn_weights, endpoint, method, pid)

            saliency_focal_idx = int(np.argmax(ecg_deltas))
            if float(ecg_deltas[saliency_focal_idx]) <= 0.0:
                saliency_focal_idx = int(np.argsort(-ecg_deltas)[0])
            attention_focal_idx = int(np.argmax(attn_weights))

            def build_context_df(focal_idx: int) -> pd.DataFrame:
                left_idx = max(0, focal_idx - 5)
                right_idx = min(len(df_segments) - 1, focal_idx + 5)
                context_idx = np.arange(left_idx, right_idx + 1, dtype=np.int64)
                context_df = df_segments.iloc[context_idx].copy().reset_index(drop=True)
                context_df["saliency"] = ecg_deltas[context_idx]
                context_df["attention_weight"] = attn_weights[context_idx]
                context_df["is_focal"] = 0
                focal_local = int(np.where(context_idx == focal_idx)[0][0])
                context_df.loc[focal_local, "is_focal"] = 1
                return context_df

            saliency_context_df = build_context_df(saliency_focal_idx)
            attention_context_df = build_context_df(attention_focal_idx)
            saliency_context_df.to_csv(patient_dir / "top_ecg_segments_saliency.csv", index=False)
            attention_context_df.to_csv(patient_dir / "top_ecg_segments_attention.csv", index=False)

            raw_npz = find_raw_npz(args.preprocessed_dir, pid)
            raw_signal = np.load(raw_npz, mmap_mode="r")["signal"]
            plot_local_context_window(
                save_path=patient_dir / "top_ecg_segments_saliency.png",
                pid=pid,
                endpoint=endpoint,
                method=method,
                context_df=saliency_context_df,
                raw_signal=raw_signal,
                fs=args.fs,
                window_sec=args.window_sec,
                score_col="saliency",
                score_label="Saliency",
            )
            plot_local_context_window(
                save_path=patient_dir / "top_ecg_segments_attention.png",
                pid=pid,
                endpoint=endpoint,
                method=method,
                context_df=attention_context_df,
                raw_signal=raw_signal,
                fs=args.fs,
                window_sec=args.window_sec,
                score_col="attention_weight",
                score_label="Attention Weight",
            )
            plot_local_context_window(
                save_path=patient_dir / "top_ecg_segments_attention_with_saliency.png",
                pid=pid,
                endpoint=endpoint,
                method=method,
                context_df=attention_context_df,
                raw_signal=raw_signal,
                fs=args.fs,
                window_sec=args.window_sec,
                score_col="saliency",
                score_label="Saliency",
                include_attention_panel=True,
            )
            # Keep the legacy filenames pointing at the saliency-based view.
            saliency_context_df.to_csv(patient_dir / "top_ecg_segments.csv", index=False)
            plot_local_context_window(
                save_path=patient_dir / "top_ecg_segments.png",
                pid=pid,
                endpoint=endpoint,
                method=method,
                context_df=saliency_context_df,
                raw_signal=raw_signal,
                fs=args.fs,
                window_sec=args.window_sec,
                score_col="saliency",
                score_label="Saliency",
            )

            reasoning_text = current_text_for_patient(text_config, task, pid)
            text_result = token_saliency_for_current_reasoning(
                reasoning_text=reasoning_text,
                multimodal_model=model,
                ecg_embedding=data.ecg_validation[idx],
                tokenizer=tokenizer,
                lm_model=lm_model,
                text_mean=data.preprocessing["second_mean"],
                text_sd=data.preprocessing["second_sd"],
                max_length=args.max_text_length,
                chunk_stride=args.chunk_stride,
                device=device,
            )
            if abs(float(text_result["prob"]) - float(y_prob[idx])) > 1e-5:
                raise RuntimeError(
                    f"Chunked text saliency does not reproduce the saved probability for "
                    f"{pid}/{endpoint}/{method}: {text_result['prob']} vs {y_prob[idx]}"
                )
            norm_scores = normalize_scores(text_result["scores"])
            spans = merge_salient_spans(
                reasoning_text,
                text_result["offsets"],
                norm_scores,
                top_tokens=args.top_text_tokens,
            )
            html_text = render_highlighted_html(
                reasoning_text=reasoning_text,
                spans=spans,
                title=f"{pid} | {endpoint} | {method} | Token-attribution visualization",
            )
            (patient_dir / "text_saliency.html").write_text(html_text)
            save_token_scores_csv(
                patient_dir / "text_token_saliency.csv",
                reasoning_text=reasoning_text,
                tokens=text_result["tokens"],
                offsets=text_result["offsets"],
                scores=norm_scores,
            )

            summary = {
                "pid": pid,
                "endpoint": endpoint,
                "method": method,
                "current_multimodal_arm": METHOD_TO_CURRENT_ARM[method],
                "selection_mode": selection_mode,
                "rank_within_target": rank,
                "y_true": int(y_true[idx]),
                "pred_prob": float(y_prob[idx]),
                "ecg_baseline_prob": float(baseline_prob),
                "multimodal_checkpoint": str(checkpoint_path),
                "text_source_name": text_config["source_name"],
                "text_encoder_name": text_config["encoder_name"],
                "text_condition": text_config["condition"],
                "text_chunk_count": text_result["chunk_count"],
                "top_saliency_window_idx": int(df_segments.iloc[saliency_focal_idx]["window_idx"]) if len(df_segments) else None,
                "top_saliency": float(ecg_deltas[saliency_focal_idx]) if len(df_segments) else None,
                "top_attention_window_idx": int(df_segments.iloc[attention_focal_idx]["window_idx"]) if len(df_segments) else None,
                "top_attention_weight": float(attn_weights[attention_focal_idx]) if len(df_segments) else None,
                "top_text_spans": [
                    {
                        "text": reasoning_text[int(span["start"]):int(span["end"])],
                        "score": float(span["score"]),
                    }
                    for span in spans
                ],
            }
            (patient_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            summary_rows.append(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.out_dir / f"multimodal_explain_summary_fold_{args.val_fold}.csv", index=False)
    print(f"Saved outputs to {args.out_dir / f'val_fold_{args.val_fold}'}")


if __name__ == "__main__":
    main()
