#!/usr/bin/env python3
"""Nested-CV MIL-TCN training and fold-specific ECG representation export.

This script implements four explicit stages for the fixed four-year MUSIC
analysis:

1. prepare: validate the cohort/ECG files and create outer-specific inner folds;
2. tune: evaluate one prespecified architecture inside every inner fold of one
   outer-training cohort;
3. summarize: select and tabulate one wave winner per outer-training cohort;
4. final: select the architecture using pooled inner out-of-fold predictions,
   refit it on the complete outer-training cohort, and export train/test ECG
   representations from that fold-specific model; and
5. aggregate: concatenate untouched outer-test predictions and representations.

The two endpoints use masked multitask losses. A competing cardiac-death event
is not treated as a negative outcome for the other endpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PATIENT_ID_COLUMN = "Patient ID"
SCD_LABEL_COLUMN = "SCD_4year_label"
PFD_LABEL_COLUMN = "PFD_4year_label"
OUTER_FOLD_COLUMN = "outer_fold"

NON_FEATURE_COLUMNS = {"patient_id", "window_idx", "start_idx", "duration_sec"}

# Locked before outcome-dependent training. Missing values are retained and
# handled using fold-specific mean imputation plus missingness indicators.
ALLOWED_FEATURES = [
    "HRV_MeanNN",
    "HRV_MedianNN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_SDSD",
    "HRV_CVNN",
    "HRV_CVSD",
    "HRV_IQRNN",
    "HRV_MadNN",
    "HRV_pNN20",
    "HRV_pNN50",
    "HRV_MinNN",
    "HRV_MaxNN",
    "HRV_SD1",
    "HRV_SD2",
    "HRV_SD1SD2",
    "HRV_S",
    "HRV_HF",
    "HRV_LnHF",
    "HRV_TP",
    "HRV_SampEn",
    "n_rpeaks",
    "pvc_burden_pct",
    "longest_rr_pause",
    "hr_mean",
    "hr_std",
    "ecg_quality_mean",
    "ecg_quality_low_frac",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-controlled nested-CV training for ECG MIL-TCN."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "tune", "summarize", "final", "aggregate"],
    )
    parser.add_argument("--folds_csv", type=Path, required=True)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--patient_id_col", default=PATIENT_ID_COLUMN)
    parser.add_argument("--scd_label_col", default=SCD_LABEL_COLUMN)
    parser.add_argument("--pfd_label_col", default=PFD_LABEL_COLUMN)
    parser.add_argument("--outer_fold_col", default=OUTER_FOLD_COLUMN)
    parser.add_argument("--sort_by", default="window_idx", choices=["window_idx", "start_idx"])
    parser.add_argument("--min_segments", type=int, default=3)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    # Optional assertions for the locked four-year cohort.
    parser.add_argument("--expected_patients", type=int)
    parser.add_argument("--expected_controls", type=int)
    parser.add_argument("--expected_scd", type=int)
    parser.add_argument("--expected_pfd", type=int)

    # One candidate configuration for --stage tune.
    parser.add_argument("--outer_fold", type=int)
    parser.add_argument("--config_name")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--tcn_hidden_dim", type=int, default=128)
    parser.add_argument("--tcn_layers", type=int, default=4)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--enc_hidden", type=int, default=128)
    parser.add_argument("--enc_dropout", type=float, default=0.2)
    parser.add_argument("--branch_hidden", type=int)
    parser.add_argument("--branch_dropout", type=float, default=0.0)
    parser.add_argument(
        "--representation_dim",
        type=int,
        default=128,
        help="Fixed exported shared ECG-embedding dimension across configurations.",
    )

    # Training settings are fixed across candidates unless explicitly included
    # in the prespecified launcher grid.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Final selection receives the exact candidate names from the launcher so
    # stale directories cannot silently enter selection.
    parser.add_argument("--candidate_config", action="append", default=[])
    parser.add_argument("--wave_name")
    parser.add_argument("--auc_tolerance", type=float, default=0.005)
    parser.add_argument(
        "--export_selected_inner_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in [logging.FileHandler(log_path), logging.StreamHandler()]:
        handler.setFormatter(formatter)
        root.addHandler(handler)


def patient_path(features_dir: Path, patient_id: str) -> Path:
    pid = str(patient_id).zfill(4)
    return features_dir / pid / f"{pid}_segment_features.csv"


def read_labels(args: argparse.Namespace, prepared: bool = True) -> pd.DataFrame:
    path = (
        args.output_root / "analysis_setup" / "nested_patient_folds.csv"
        if prepared and args.stage != "prepare"
        else args.folds_csv
    )
    if not path.exists():
        raise FileNotFoundError(f"Patient fold file not found: {path}")
    dataframe = pd.read_csv(path, dtype={args.patient_id_col: "string"})
    required = [
        args.patient_id_col,
        args.scd_label_col,
        args.pfd_label_col,
        args.outer_fold_col,
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required patient columns: {missing}")

    if dataframe[args.patient_id_col].isna().any():
        raise ValueError("Patient IDs contain missing values.")
    dataframe[args.patient_id_col] = dataframe[args.patient_id_col].str.strip()
    if dataframe[args.patient_id_col].eq("").any():
        raise ValueError("Patient IDs contain empty values.")
    if not dataframe[args.patient_id_col].is_unique:
        raise ValueError("Patient IDs must be unique.")

    dataframe[args.scd_label_col] = pd.to_numeric(
        dataframe[args.scd_label_col], errors="coerce"
    ).astype("Float64")
    dataframe[args.pfd_label_col] = pd.to_numeric(
        dataframe[args.pfd_label_col], errors="coerce"
    ).astype("Float64")
    dataframe[args.outer_fold_col] = pd.to_numeric(
        dataframe[args.outer_fold_col], errors="raise"
    ).astype(int)

    outcomes = []
    for scd, pfd in zip(dataframe[args.scd_label_col], dataframe[args.pfd_label_col]):
        if pd.notna(scd) and scd == 1 and pd.isna(pfd):
            outcomes.append("SCD")
        elif pd.notna(pfd) and pfd == 1 and pd.isna(scd):
            outcomes.append("PFD")
        elif pd.notna(scd) and pd.notna(pfd) and scd == 0 and pfd == 0:
            outcomes.append("No cardiac death")
        else:
            raise ValueError(
                "Invalid four-year label combination. Expected control=(0,0), "
                "SCD=(1,NA), or PFD=(NA,1)."
            )
    dataframe["four_year_outcome_verified"] = outcomes
    return dataframe


def assert_expected_counts(dataframe: pd.DataFrame, args: argparse.Namespace) -> dict:
    counts = dataframe["four_year_outcome_verified"].value_counts().to_dict()
    checks = {
        "patients": (len(dataframe), args.expected_patients),
        "controls": (counts.get("No cardiac death", 0), args.expected_controls),
        "scd": (counts.get("SCD", 0), args.expected_scd),
        "pfd": (counts.get("PFD", 0), args.expected_pfd),
    }
    for name, (observed, expected) in checks.items():
        if expected is not None and observed != expected:
            raise ValueError(f"Expected {expected} {name}, observed {observed}.")
    return {name: int(observed) for name, (observed, _) in checks.items()}


def stage_prepare(args: argparse.Namespace) -> None:
    dataframe = read_labels(args, prepared=False)
    counts = assert_expected_counts(dataframe, args)
    expected_outer = set(range(args.outer_splits))
    observed_outer = set(dataframe[args.outer_fold_col].unique())
    if observed_outer != expected_outer:
        raise ValueError(
            f"Expected outer folds {sorted(expected_outer)}, observed {sorted(observed_outer)}."
        )

    setup_dir = args.output_root / "analysis_setup"
    setup_dir.mkdir(parents=True, exist_ok=True)

    # Create independent inner folds inside each outer-training cohort.
    for outer_fold in range(args.outer_splits):
        column = f"inner_fold_outer_{outer_fold}"
        dataframe[column] = pd.Series(pd.NA, index=dataframe.index, dtype="Int64")
        train_mask = dataframe[args.outer_fold_col].ne(outer_fold)
        train_indices = dataframe.index[train_mask].to_numpy()
        strata = dataframe.loc[train_mask, "four_year_outcome_verified"]
        splitter = StratifiedKFold(
            n_splits=args.inner_splits,
            shuffle=True,
            random_state=args.seed + outer_fold,
        )
        for inner_fold, (_, validation_positions) in enumerate(
            splitter.split(np.zeros(len(train_indices)), strata)
        ):
            validation_indices = train_indices[validation_positions]
            dataframe.loc[validation_indices, column] = inner_fold

        if dataframe.loc[train_mask, column].isna().any():
            raise RuntimeError(f"Incomplete assignments in {column}.")
        if dataframe.loc[~train_mask, column].notna().any():
            raise RuntimeError(f"Outer-test patients received assignments in {column}.")

    # ECG availability and schema audit. Individual missing features do not
    # exclude a patient; an absent/empty ECG feature file does.
    audit_rows = []
    feature_finite_counts = {feature: 0 for feature in ALLOWED_FEATURES}
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Audit ECG files"):
        patient_id = row[args.patient_id_col]
        path = patient_path(args.features_dir, patient_id)
        record = {
            args.patient_id_col: patient_id,
            "feature_file": str(path.resolve()),
            "file_available": path.exists(),
            "segment_count": 0,
            "meets_min_segments": False,
            "missing_schema_columns": len(ALLOWED_FEATURES),
            "finite_feature_fraction": 0.0,
            "file_size_bytes": pd.NA,
            "file_modified_ns": pd.NA,
        }
        if path.exists():
            stat = path.stat()
            record["file_size_bytes"] = int(stat.st_size)
            record["file_modified_ns"] = int(stat.st_mtime_ns)
            segment_df = pd.read_csv(path)
            record["segment_count"] = int(len(segment_df))
            record["meets_min_segments"] = len(segment_df) >= args.min_segments
            present = [feature for feature in ALLOWED_FEATURES if feature in segment_df.columns]
            record["missing_schema_columns"] = len(ALLOWED_FEATURES) - len(present)
            finite_total = 0
            possible_total = max(len(segment_df) * len(ALLOWED_FEATURES), 1)
            for feature in present:
                values = pd.to_numeric(segment_df[feature], errors="coerce").to_numpy(float)
                count = int(np.isfinite(values).sum())
                feature_finite_counts[feature] += count
                finite_total += count
            record["finite_feature_fraction"] = finite_total / possible_total
        audit_rows.append(record)

    audit = pd.DataFrame(audit_rows)
    audit_path = setup_dir / "patient_ecg_audit.csv"
    audit.to_csv(audit_path, index=False)

    unusable = ~audit["file_available"] | ~audit["meets_min_segments"]
    if unusable.any():
        bad_ids = audit.loc[unusable, args.patient_id_col].tolist()
        raise RuntimeError(
            f"{int(unusable.sum())} selected patients lack a usable ECG feature file. "
            f"See {audit_path}. Example IDs: {bad_ids[:10]}"
        )

    never_observed = [feature for feature, count in feature_finite_counts.items() if count == 0]
    if never_observed:
        raise RuntimeError(
            "Locked ECG features with no finite observations were found: "
            f"{never_observed}. Revise the prespecified feature schema explicitly."
        )

    folds_path = setup_dir / "nested_patient_folds.csv"
    dataframe.to_csv(folds_path, index=False)
    schema = {
        "feature_columns": ALLOWED_FEATURES,
        "missing_data_strategy": "training-fold mean imputation plus one indicator per feature",
        "standardization": "training-fold z-score for continuous feature values only",
        "segment_exclusion_for_partial_missingness": False,
        "minimum_segments": args.min_segments,
        "finite_counts_full_cohort_for_audit_only": feature_finite_counts,
    }
    atomic_json(setup_dir / "feature_schema.json", schema)

    fold_counts = []
    inner_fold_counts = []
    for outer_fold in range(args.outer_splits):
        subset = dataframe[dataframe[args.outer_fold_col].eq(outer_fold)]
        for outcome, count in subset["four_year_outcome_verified"].value_counts().items():
            fold_counts.append(
                {"outer_fold": outer_fold, "outcome": outcome, "patient_count": int(count)}
            )
        inner_column = f"inner_fold_outer_{outer_fold}"
        outer_training = dataframe[dataframe[args.outer_fold_col].ne(outer_fold)]
        grouped = outer_training.groupby(
            [inner_column, "four_year_outcome_verified"], observed=True
        ).size()
        for (inner_fold, outcome), count in grouped.items():
            inner_fold_counts.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": int(inner_fold),
                    "outcome": outcome,
                    "patient_count": int(count),
                }
            )
    pd.DataFrame(fold_counts).to_csv(setup_dir / "outer_fold_counts.csv", index=False)
    pd.DataFrame(inner_fold_counts).to_csv(
        setup_dir / "inner_fold_counts.csv", index=False
    )

    atomic_json(
        setup_dir / "prepare_summary.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "source_folds_csv": str(args.folds_csv.resolve()),
            "source_folds_sha256": sha256_file(args.folds_csv),
            "prepared_folds_csv": str(folds_path.resolve()),
            "patient_counts": counts,
            "outer_splits": args.outer_splits,
            "inner_splits": args.inner_splits,
            "seed": args.seed,
            "ecg_unavailable_count": int(unusable.sum()),
            "feature_count": len(ALLOWED_FEATURES),
        },
    )
    print(f"Prepared nested folds: {folds_path}")
    print(f"ECG audit: {audit_path}")


def compute_training_statistics(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros(len(feature_columns), dtype=np.float64)
    sumsqs = np.zeros(len(feature_columns), dtype=np.float64)
    counts = np.zeros(len(feature_columns), dtype=np.int64)

    for _, row in tqdm(
        dataframe.iterrows(), total=len(dataframe), desc="Training-fold feature statistics", leave=False
    ):
        frame = pd.read_csv(patient_path(args.features_dir, row[args.patient_id_col]))
        values = frame.reindex(columns=feature_columns).apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float64)
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        sums += safe.sum(axis=0)
        sumsqs += (safe * safe).sum(axis=0)
        counts += finite.sum(axis=0)

    if (counts == 0).any():
        dead = [feature_columns[index] for index in np.where(counts == 0)[0]]
        raise RuntimeError(f"Training split has no finite observations for: {dead}")

    means = sums / counts
    variances = np.maximum(sumsqs / counts - means**2, 1e-12)
    standard_deviations = np.sqrt(variances)
    return means.astype(np.float32), standard_deviations.astype(np.float32), counts


class ECGFeatureDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        args: argparse.Namespace,
        feature_columns: list[str],
        means: np.ndarray,
        standard_deviations: np.ndarray,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.args = args
        self.feature_columns = feature_columns
        self.means = means
        self.standard_deviations = standard_deviations

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        patient_id = row[self.args.patient_id_col]
        path = patient_path(self.args.features_dir, patient_id)
        frame = pd.read_csv(path)
        if self.args.sort_by in frame.columns:
            frame = frame.sort_values(self.args.sort_by, kind="stable")
            segment_ids = frame[self.args.sort_by].astype(str).to_numpy()
        else:
            segment_ids = np.arange(len(frame)).astype(str)

        values = frame.reindex(columns=self.feature_columns).apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float32)
        missing = ~np.isfinite(values)
        imputed = np.where(missing, self.means, values)
        standardized = (imputed - self.means) / self.standard_deviations
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.concatenate([standardized, missing.astype(np.float32)], axis=1)

        if len(features) < self.args.min_segments:
            raise RuntimeError(
                f"Patient {patient_id} has {len(features)} segments; "
                f"minimum is {self.args.min_segments}."
            )

        scd_value = row[self.args.scd_label_col]
        pfd_value = row[self.args.pfd_label_col]
        labels = torch.tensor(
            [0.0 if pd.isna(scd_value) else float(scd_value),
             0.0 if pd.isna(pfd_value) else float(pfd_value)],
            dtype=torch.float32,
        )
        masks = torch.tensor(
            [not pd.isna(scd_value), not pd.isna(pfd_value)], dtype=torch.bool
        )
        return torch.from_numpy(features), labels, masks, patient_id, segment_ids


def collate_single(batch):
    if len(batch) != 1:
        raise ValueError("This variable-length MIL implementation requires batch_size=1.")
    return batch[0]


class FeatureEncoder(nn.Module):
    def __init__(self, input_dimension: int, embedding_dimension: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dimension),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class TemporalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.convolution = nn.Conv1d(
            input_channels, output_channels, kernel, padding=padding, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout(F.relu(self.convolution(inputs)))[:, :, : inputs.size(2)]
        residual = inputs if self.downsample is None else self.downsample(inputs)
        return outputs + residual


class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, input_dimension: int, hidden: int, layers: int, kernel: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            *[
                TemporalBlock(
                    input_dimension if layer == 0 else hidden,
                    hidden,
                    kernel,
                    2**layer,
                    dropout,
                )
                for layer in range(layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class AttentionMIL(nn.Module):
    def __init__(self, input_dimension: int, attention_dimension: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dimension, attention_dimension),
            nn.Tanh(),
            nn.Linear(attention_dimension, 1),
        )

    def forward(self, temporal_features: torch.Tensor):
        segments = temporal_features.squeeze(0).transpose(0, 1)
        weights = torch.softmax(self.attention(segments).squeeze(1), dim=0)
        pooled = (segments * weights.unsqueeze(1)).sum(dim=0)
        return pooled, weights


class MILTCNFeatureMultiBranch(nn.Module):
    def __init__(self, input_feature_dimension: int, config: dict):
        super().__init__()
        self.encoder = FeatureEncoder(
            input_feature_dimension,
            config["embedding_dim"],
            config["enc_hidden"],
            config["enc_dropout"],
        )
        self.tcn = TemporalConvolutionNetwork(
            config["embedding_dim"],
            config["tcn_hidden_dim"],
            config["tcn_layers"],
            config["tcn_kernel_size"],
            config["tcn_dropout"],
        )
        self.pool = AttentionMIL(config["tcn_hidden_dim"], config["attn_dim"])
        self.representation_projection = (
            nn.Identity()
            if config["tcn_hidden_dim"] == config["representation_dim"]
            else nn.Linear(config["tcn_hidden_dim"], config["representation_dim"])
        )
        representation_dimension = config["representation_dim"]
        branch_hidden = config["branch_hidden"] or representation_dimension
        branch_dropout = config["branch_dropout"]
        def make_branch():
            return nn.Sequential(
                nn.Linear(representation_dimension, branch_hidden),
                nn.ReLU(),
                nn.Dropout(branch_dropout),
                nn.Linear(branch_hidden, representation_dimension),
                nn.ReLU(),
            )

        self.scd_branch = make_branch()
        self.pfd_branch = make_branch()
        self.scd_head = nn.Linear(representation_dimension, 1)
        self.pfd_head = nn.Linear(representation_dimension, 1)

    def forward(self, features: torch.Tensor):
        encoded = self.encoder(features)
        temporal_input = encoded.transpose(0, 1).unsqueeze(0)
        temporal = self.tcn(temporal_input)
        pooled, attention = self.pool(temporal)
        shared = self.representation_projection(pooled)
        scd_representation = self.scd_branch(shared)
        pfd_representation = self.pfd_branch(shared)
        return (
            self.scd_head(scd_representation),
            self.pfd_head(pfd_representation),
            shared,
            scd_representation,
            pfd_representation,
            attention,
        )


def config_from_args(args: argparse.Namespace) -> dict:
    return {
        "embedding_dim": args.embedding_dim,
        "tcn_hidden_dim": args.tcn_hidden_dim,
        "tcn_layers": args.tcn_layers,
        "tcn_kernel_size": args.tcn_kernel_size,
        "tcn_dropout": args.tcn_dropout,
        "attn_dim": args.attn_dim,
        "enc_hidden": args.enc_hidden,
        "enc_dropout": args.enc_dropout,
        "branch_hidden": args.branch_hidden,
        "branch_dropout": args.branch_dropout,
        "representation_dim": args.representation_dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }


def make_loader(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        ECGFeatureDataset(dataframe, args, ALLOWED_FEATURES, means, standard_deviations),
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_single,
        generator=generator,
    )


def class_weights(dataframe: pd.DataFrame, args: argparse.Namespace) -> tuple[float, float]:
    weights = []
    for column in [args.scd_label_col, args.pfd_label_col]:
        known = dataframe[column].dropna().astype(int)
        positives = int(known.eq(1).sum())
        negatives = int(known.eq(0).sum())
        if positives == 0 or negatives == 0:
            raise RuntimeError(f"Training split has invalid class counts for {column}.")
        weights.append(negatives / positives)
    return float(weights[0]), float(weights[1])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"n": int(len(labels)), "events": int(labels.sum()), "roc_auc": None,
                "pr_auc": None, "brier": None}
    return {
        "n": int(len(labels)),
        "events": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def metrics_from_predictions(predictions: pd.DataFrame) -> dict:
    output = {}
    for task in ["scd", "pfd"]:
        applicable = predictions[f"{task}_label"].notna()
        output[task] = binary_metrics(
            predictions.loc[applicable, f"{task}_label"].to_numpy(),
            predictions.loc[applicable, f"{task}_probability"].to_numpy(),
        )
    aucs = [output[task]["roc_auc"] for task in ["scd", "pfd"]]
    prs = [output[task]["pr_auc"] for task in ["scd", "pfd"]]
    output["mean_roc_auc"] = float(np.mean(aucs)) if all(x is not None for x in aucs) else None
    output["mean_pr_auc"] = float(np.mean(prs)) if all(x is not None for x in prs) else None
    return output


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    outer_fold: int,
    inner_fold: int | None,
    include_attention: bool = False,
):
    model.eval()
    records = []
    patient_ids = []
    shared_embeddings = []
    scd_embeddings = []
    pfd_embeddings = []
    attention_rows = []

    for features, labels, masks, patient_id, segment_ids in tqdm(
        loader, desc="Export predictions/embeddings", leave=False
    ):
        features = features.to(device)
        outputs = model(features)
        scd_logit, pfd_logit, shared, scd_rep, pfd_rep, attention = outputs
        scd_probability = float(torch.sigmoid(scd_logit).item())
        pfd_probability = float(torch.sigmoid(pfd_logit).item())
        record = {
            PATIENT_ID_COLUMN: str(patient_id),
            "outer_fold": int(outer_fold),
            "inner_fold": pd.NA if inner_fold is None else int(inner_fold),
            "scd_label": float(labels[0]) if bool(masks[0]) else pd.NA,
            "pfd_label": float(labels[1]) if bool(masks[1]) else pd.NA,
            "scd_probability": scd_probability,
            "pfd_probability": pfd_probability,
        }
        records.append(record)
        patient_ids.append(str(patient_id))
        shared_embeddings.append(shared.detach().cpu().float().numpy())
        scd_embeddings.append(scd_rep.detach().cpu().float().numpy())
        pfd_embeddings.append(pfd_rep.detach().cpu().float().numpy())

        if include_attention:
            weights = attention.detach().cpu().float().numpy()
            for segment_id, weight in zip(segment_ids, weights):
                attention_rows.append(
                    {
                        PATIENT_ID_COLUMN: str(patient_id),
                        "outer_fold": int(outer_fold),
                        "segment_id": str(segment_id),
                        "attention_weight": float(weight),
                    }
                )

    embeddings = {
        "patient_ids": np.asarray(patient_ids, dtype=str),
        "shared_embeddings": np.asarray(shared_embeddings, dtype=np.float32),
        "scd_branch_embeddings": np.asarray(scd_embeddings, dtype=np.float32),
        "pfd_branch_embeddings": np.asarray(pfd_embeddings, dtype=np.float32),
    }
    return pd.DataFrame(records), embeddings, pd.DataFrame(attention_rows)


def save_embedding_bundle(path: Path, embeddings: dict, provenance: str) -> None:
    atomic_npz(path, **embeddings, provenance=np.asarray([provenance], dtype=str))


def build_model(config: dict, device: torch.device) -> MILTCNFeatureMultiBranch:
    input_dimension = len(ALLOWED_FEATURES) * 2
    return MILTCNFeatureMultiBranch(input_dimension, config).to(device)


def training_loss(
    scd_logit: torch.Tensor,
    pfd_logit: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    scd_criterion: nn.Module,
    pfd_criterion: nn.Module,
) -> torch.Tensor:
    losses = []
    if bool(masks[0]):
        losses.append(scd_criterion(scd_logit.view(-1), labels[0:1]))
    if bool(masks[1]):
        losses.append(pfd_criterion(pfd_logit.view(-1), labels[1:2]))
    if not losses:
        raise RuntimeError("Patient has neither an SCD nor PFD task label.")
    return torch.stack(losses).sum()


def train_split(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    args: argparse.Namespace,
    config: dict,
    output_directory: Path,
    outer_fold: int,
    inner_fold: int,
) -> dict:
    output_directory.mkdir(parents=True, exist_ok=True)
    completion_path = output_directory / "run_complete.json"
    prepared_folds = args.output_root / "analysis_setup" / "nested_patient_folds.csv"
    cache_signature = {
        "prepared_folds_sha256": sha256_file(prepared_folds),
        "training_patient_ids_sha256": sha256_values(
            train_dataframe[args.patient_id_col].tolist()
        ),
        "validation_patient_ids_sha256": sha256_values(
            validation_dataframe[args.patient_id_col].tolist()
        ),
        "config": config,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "script_sha256": sha256_file(Path(__file__)),
    }
    if completion_path.exists() and not args.overwrite:
        with open(completion_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("cache_signature") != cache_signature:
            raise RuntimeError(
                f"Completed output does not match the requested run: {output_directory}. "
                "Use a new output root or rerun with --overwrite."
            )
        logging.info("Using completed run: %s", output_directory)
        return existing

    run_seed = args.seed + outer_fold * 100 + inner_fold
    set_seed(run_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    means, standard_deviations, finite_counts = compute_training_statistics(
        train_dataframe, args, ALLOWED_FEATURES
    )
    atomic_npz(
        output_directory / "preprocessing.npz",
        feature_columns=np.asarray(ALLOWED_FEATURES, dtype=str),
        means=means,
        standard_deviations=standard_deviations,
        finite_training_counts=finite_counts,
    )
    train_loader = make_loader(train_dataframe, args, means, standard_deviations, True)
    validation_loader = make_loader(
        validation_dataframe, args, means, standard_deviations, False
    )

    model = build_model(config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scd_weight, pfd_weight = class_weights(train_dataframe, args)
    scd_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scd_weight], dtype=torch.float32, device=device)
    )
    pfd_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pfd_weight], dtype=torch.float32, device=device)
    )

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for features, labels, masks, _, _ in tqdm(
            train_loader, desc=f"Epoch {epoch}", leave=False
        ):
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            scd_logit, pfd_logit, *_ = model(features)
            loss = training_loss(
                scd_logit, pfd_logit, labels, masks, scd_criterion, pfd_criterion
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        validation_predictions, _, _ = predict(
            model, validation_loader, device, outer_fold, inner_fold, False
        )
        validation_metrics = metrics_from_predictions(validation_predictions)
        score = validation_metrics["mean_roc_auc"]
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / max(steps, 1),
                "scd_roc_auc": validation_metrics["scd"]["roc_auc"],
                "scd_pr_auc": validation_metrics["scd"]["pr_auc"],
                "pfd_roc_auc": validation_metrics["pfd"]["roc_auc"],
                "pfd_pr_auc": validation_metrics["pfd"]["pr_auc"],
                "mean_roc_auc": score,
                "mean_pr_auc": validation_metrics["mean_pr_auc"],
            }
        )
        logging.info(
            "outer=%d inner=%d epoch=%d loss=%.5f mean_auc=%.4f mean_pr=%.4f",
            outer_fold,
            inner_fold,
            epoch,
            history[-1]["training_loss"],
            score,
            validation_metrics["mean_pr_auc"],
        )

        if score is not None and score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            logging.info("Early stopping at epoch %d", epoch)
            break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was produced.")

    pd.DataFrame(history).to_csv(output_directory / "epoch_metrics.csv", index=False)
    checkpoint = {
        "state_dict": best_state,
        "config": config,
        "feature_columns": ALLOWED_FEATURES,
        "best_epoch": best_epoch,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "seed": run_seed,
        "class_weight_scd": scd_weight,
        "class_weight_pfd": pfd_weight,
    }
    atomic_torch_save(output_directory / "best_checkpoint.pt", checkpoint)
    model.load_state_dict(best_state)

    validation_predictions, validation_embeddings, _ = predict(
        model, validation_loader, device, outer_fold, inner_fold, False
    )
    validation_predictions.to_csv(output_directory / "validation_predictions.csv", index=False)
    save_embedding_bundle(
        output_directory / "validation_embeddings.npz",
        validation_embeddings,
        "inner-validation embeddings from an ECG model not trained on these patients",
    )
    validation_metrics = metrics_from_predictions(validation_predictions)
    result = {
        "completed": True,
        "created_at_utc": utc_now(),
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "config_name": args.config_name,
        "config": config,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "parameter_count": parameter_count,
        "training_patient_count": len(train_dataframe),
        "validation_patient_count": len(validation_dataframe),
        "class_weight_scd": scd_weight,
        "class_weight_pfd": pfd_weight,
        "validation_metrics": validation_metrics,
        "seed": run_seed,
        "cache_signature": cache_signature,
    }
    atomic_json(completion_path, result)
    return result


def stage_tune(args: argparse.Namespace) -> None:
    if args.outer_fold is None or args.config_name is None:
        raise ValueError("--stage tune requires --outer_fold and --config_name.")
    dataframe = read_labels(args)
    inner_column = f"inner_fold_outer_{args.outer_fold}"
    if inner_column not in dataframe.columns:
        raise ValueError(f"Missing prepared inner-fold column: {inner_column}")
    outer_train = dataframe[dataframe[args.outer_fold_col].ne(args.outer_fold)].copy()
    config = config_from_args(args)
    base = args.output_root / "tuning" / args.config_name / f"outer_fold_{args.outer_fold}"
    setup_logging(base / "tuning.log")

    results = []
    pooled_predictions = []
    for inner_fold in range(args.inner_splits):
        inner_train = outer_train[outer_train[inner_column].ne(inner_fold)].copy()
        inner_validation = outer_train[outer_train[inner_column].eq(inner_fold)].copy()
        result = train_split(
            inner_train,
            inner_validation,
            args,
            config,
            base / f"inner_fold_{inner_fold}",
            args.outer_fold,
            inner_fold,
        )
        results.append(result)
        pooled_predictions.append(
            pd.read_csv(
                base / f"inner_fold_{inner_fold}" / "validation_predictions.csv",
                dtype={PATIENT_ID_COLUMN: "string"},
            )
        )

    pooled = pd.concat(pooled_predictions, ignore_index=True)
    if len(pooled) != len(outer_train) or not pooled[PATIENT_ID_COLUMN].is_unique:
        raise RuntimeError("Inner out-of-fold predictions do not cover outer-training patients once.")
    pooled.to_csv(base / "pooled_inner_oof_predictions.csv", index=False)
    pooled_metrics = metrics_from_predictions(pooled)
    summary = {
        "completed": True,
        "created_at_utc": utc_now(),
        "outer_fold": args.outer_fold,
        "config_name": args.config_name,
        "config": config,
        "selection_score": pooled_metrics["mean_roc_auc"],
        "secondary_mean_pr_auc": pooled_metrics["mean_pr_auc"],
        "pooled_inner_metrics": pooled_metrics,
        "inner_best_epochs": [result["best_epoch"] for result in results],
        "parameter_count": results[0]["parameter_count"],
        "patient_count": len(pooled),
    }
    atomic_json(base / "tuning_summary.json", summary)
    logging.info("Completed %s outer fold %d: pooled mean AUC %.4f", args.config_name,
                 args.outer_fold, summary["selection_score"])


def load_preprocessing(path: Path) -> tuple[np.ndarray, np.ndarray]:
    artifact = np.load(path)
    columns = artifact["feature_columns"].astype(str).tolist()
    if columns != ALLOWED_FEATURES:
        raise RuntimeError("Stored preprocessing feature schema does not match locked schema.")
    return artifact["means"].astype(np.float32), artifact["standard_deviations"].astype(np.float32)


def export_selected_inner_split(
    args: argparse.Namespace,
    dataframe: pd.DataFrame,
    selected: dict,
    inner_fold: int,
    destination: Path,
    device: torch.device,
) -> None:
    source = (
        args.output_root
        / "tuning"
        / selected["config_name"]
        / f"outer_fold_{args.outer_fold}"
        / f"inner_fold_{inner_fold}"
    )
    checkpoint = torch.load(source / "best_checkpoint.pt", map_location="cpu")
    means, standard_deviations = load_preprocessing(source / "preprocessing.npz")
    model = build_model(checkpoint["config"], device)
    model.load_state_dict(checkpoint["state_dict"])
    inner_column = f"inner_fold_outer_{args.outer_fold}"
    outer_train = dataframe[dataframe[args.outer_fold_col].ne(args.outer_fold)]
    inner_train = outer_train[outer_train[inner_column].ne(inner_fold)]
    inner_validation = outer_train[outer_train[inner_column].eq(inner_fold)]
    destination.mkdir(parents=True, exist_ok=True)
    for name, subset in [("train", inner_train), ("validation", inner_validation)]:
        loader = make_loader(subset, args, means, standard_deviations, False)
        predictions, embeddings, _ = predict(
            model, loader, device, args.outer_fold, inner_fold, False
        )
        predictions.to_csv(destination / f"{name}_predictions.csv", index=False)
        save_embedding_bundle(
            destination / f"{name}_embeddings.npz",
            embeddings,
            f"{name} embeddings from selected inner-fold ECG checkpoint",
        )


def select_candidate(args: argparse.Namespace) -> dict:
    if not args.candidate_config:
        raise ValueError("--stage final requires repeated --candidate_config names.")
    summaries = []
    for name in args.candidate_config:
        path = args.output_root / "tuning" / name / f"outer_fold_{args.outer_fold}" / "tuning_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing tuning summary: {path}")
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
        if not summary.get("completed"):
            raise RuntimeError(f"Incomplete tuning summary: {path}")
        summaries.append(summary)

    maximum_auc = max(summary["selection_score"] for summary in summaries)
    eligible = [
        summary
        for summary in summaries
        if summary["selection_score"] >= maximum_auc - args.auc_tolerance
    ]
    eligible.sort(
        key=lambda summary: (
            summary["parameter_count"],
            -summary["secondary_mean_pr_auc"],
            summary["config_name"],
        )
    )
    selected = copy.deepcopy(eligible[0])
    selected["maximum_candidate_mean_auc"] = maximum_auc
    selected["auc_tolerance"] = args.auc_tolerance
    selected["selection_rule"] = (
        f"Within {args.auc_tolerance:g} of maximum pooled inner mean ROC-AUC, "
        "choose smallest parameter count; then highest pooled mean PR-AUC; "
        "then lexical configuration name."
    )
    selected["all_candidates"] = [
        {
            "config_name": summary["config_name"],
            "selection_score": summary["selection_score"],
            "secondary_mean_pr_auc": summary["secondary_mean_pr_auc"],
            "parameter_count": summary["parameter_count"],
        }
        for summary in summaries
    ]
    return selected


def stage_summarize(args: argparse.Namespace) -> None:
    """Create auditable candidate and winner tables for one tuning wave."""
    if not args.wave_name:
        raise ValueError("--stage summarize requires --wave_name.")
    if not args.candidate_config:
        raise ValueError("--stage summarize requires repeated --candidate_config names.")

    wave_directory = args.output_root / "wave_results"
    wave_directory.mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    selected_rows = []
    selected_by_outer_fold = {}
    original_outer_fold = args.outer_fold

    for outer_fold in range(args.outer_splits):
        args.outer_fold = outer_fold
        selected = select_candidate(args)
        selected_by_outer_fold[str(outer_fold)] = selected
        selected_name = selected["config_name"]

        for candidate_name in args.candidate_config:
            summary_path = (
                args.output_root
                / "tuning"
                / candidate_name
                / f"outer_fold_{outer_fold}"
                / "tuning_summary.json"
            )
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
            config = summary["config"]
            row = {
                "wave": args.wave_name,
                "outer_fold": outer_fold,
                "config_name": candidate_name,
                "selected": candidate_name == selected_name,
                "scd_roc_auc": summary["pooled_inner_metrics"]["scd"]["roc_auc"],
                "pfd_roc_auc": summary["pooled_inner_metrics"]["pfd"]["roc_auc"],
                "mean_roc_auc": summary["selection_score"],
                "scd_pr_auc": summary["pooled_inner_metrics"]["scd"]["pr_auc"],
                "pfd_pr_auc": summary["pooled_inner_metrics"]["pfd"]["pr_auc"],
                "mean_pr_auc": summary["secondary_mean_pr_auc"],
                "parameter_count": summary["parameter_count"],
                "inner_best_epochs": ";".join(
                    str(epoch) for epoch in summary["inner_best_epochs"]
                ),
                **config,
            }
            candidate_rows.append(row)
            if candidate_name == selected_name:
                selected_rows.append(row.copy())

    args.outer_fold = original_outer_fold
    candidate_table = pd.DataFrame(candidate_rows).sort_values(
        ["outer_fold", "mean_roc_auc", "parameter_count"],
        ascending=[True, False, True],
    )
    selected_table = pd.DataFrame(selected_rows).sort_values("outer_fold")
    candidate_table.to_csv(
        wave_directory / f"{args.wave_name}_auc_results.csv", index=False
    )
    selected_table.to_csv(
        wave_directory / f"{args.wave_name}_selected_hyperparameters.csv", index=False
    )

    descriptive = (
        candidate_table.groupby("config_name", as_index=False)
        .agg(
            mean_inner_roc_auc_across_outer_cohorts=("mean_roc_auc", "mean"),
            sd_inner_roc_auc_across_outer_cohorts=("mean_roc_auc", "std"),
            mean_inner_pr_auc_across_outer_cohorts=("mean_pr_auc", "mean"),
            times_selected=("selected", "sum"),
        )
        .sort_values(
            ["times_selected", "mean_inner_roc_auc_across_outer_cohorts"],
            ascending=[False, False],
        )
    )
    descriptive.to_csv(
        wave_directory / f"{args.wave_name}_descriptive_summary.csv", index=False
    )
    atomic_json(
        wave_directory / f"{args.wave_name}_selected_by_outer_fold.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "wave": args.wave_name,
            "selection_scope": "separate within each outer-training cohort",
            "selection_criterion": "equally weighted mean SCD/PFD pooled inner ROC-AUC",
            "auc_tolerance": args.auc_tolerance,
            "selected_by_outer_fold": selected_by_outer_fold,
        },
    )
    print(
        selected_table[
            [
                "outer_fold",
                "config_name",
                "mean_roc_auc",
                "mean_pr_auc",
                "embedding_dim",
                "tcn_hidden_dim",
                "tcn_layers",
                "enc_dropout",
                "tcn_dropout",
                "branch_dropout",
                "lr",
                "weight_decay",
            ]
        ].to_string(index=False)
    )
    print(
        f"Wave tables saved to {wave_directory / f'{args.wave_name}_auc_results.csv'}"
    )


def train_fixed_epochs(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    config: dict,
    epochs: int,
    output_directory: Path,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict]:
    set_seed(args.seed + 1000 + args.outer_fold)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    means, standard_deviations, finite_counts = compute_training_statistics(
        dataframe, args, ALLOWED_FEATURES
    )
    loader = make_loader(dataframe, args, means, standard_deviations, True)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scd_weight, pfd_weight = class_weights(dataframe, args)
    scd_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([scd_weight], device=device)
    )
    pfd_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pfd_weight], device=device)
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels, masks, _, _ in tqdm(
            loader, desc=f"Final epoch {epoch}/{epochs}", leave=False
        ):
            features, labels, masks = features.to(device), labels.to(device), masks.to(device)
            scd_logit, pfd_logit, *_ = model(features)
            loss = training_loss(
                scd_logit, pfd_logit, labels, masks, scd_criterion, pfd_criterion
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        history.append({"epoch": epoch, "training_loss": total_loss / len(loader)})

    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_directory / "final_training_history.csv", index=False)
    atomic_npz(
        output_directory / "preprocessing.npz",
        feature_columns=np.asarray(ALLOWED_FEATURES, dtype=str),
        means=means,
        standard_deviations=standard_deviations,
        finite_training_counts=finite_counts,
    )
    checkpoint_metadata = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": config,
        "feature_columns": ALLOWED_FEATURES,
        "epochs": epochs,
        "outer_fold": args.outer_fold,
        "seed": args.seed + 1000 + args.outer_fold,
        "class_weight_scd": scd_weight,
        "class_weight_pfd": pfd_weight,
    }
    atomic_torch_save(output_directory / "final_checkpoint.pt", checkpoint_metadata)
    return model, means, standard_deviations, checkpoint_metadata


def stage_final(args: argparse.Namespace) -> None:
    if args.outer_fold is None:
        raise ValueError("--stage final requires --outer_fold.")
    dataframe = read_labels(args)
    destination = args.output_root / "final_models" / f"outer_fold_{args.outer_fold}"
    completion_path = destination / "run_complete.json"
    tuning_summary_hashes = {}
    for name in args.candidate_config:
        summary_path = (
            args.output_root
            / "tuning"
            / name
            / f"outer_fold_{args.outer_fold}"
            / "tuning_summary.json"
        )
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing tuning summary: {summary_path}")
        tuning_summary_hashes[name] = sha256_file(summary_path)
    final_cache_signature = {
        "prepared_folds_sha256": sha256_file(
            args.output_root / "analysis_setup" / "nested_patient_folds.csv"
        ),
        "tuning_summary_sha256": tuning_summary_hashes,
        "candidate_configs": args.candidate_config,
        "outer_fold": args.outer_fold,
        "auc_tolerance": args.auc_tolerance,
        "seed": args.seed,
        "script_sha256": sha256_file(Path(__file__)),
    }
    if completion_path.exists() and not args.overwrite:
        with open(completion_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("cache_signature") != final_cache_signature:
            raise RuntimeError(
                f"Completed final output does not match the requested run: {destination}. "
                "Use a new output root or rerun with --overwrite."
            )
        print(f"Using completed final outer fold: {destination}")
        return
    setup_logging(destination / "final.log")
    selected = select_candidate(args)
    best_epochs = selected["inner_best_epochs"]
    final_epochs = int(math.floor(float(np.median(best_epochs)) + 0.5))
    selected["final_training_epochs"] = final_epochs
    selected["epoch_rule"] = "nearest integer to median inner-fold best epoch"
    atomic_json(destination / "selected_configuration.json", selected)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.export_selected_inner_embeddings:
        for inner_fold in range(args.inner_splits):
            export_selected_inner_split(
                args,
                dataframe,
                selected,
                inner_fold,
                destination / "selected_inner_folds" / f"inner_fold_{inner_fold}",
                device,
            )

    outer_train = dataframe[dataframe[args.outer_fold_col].ne(args.outer_fold)].copy()
    outer_test = dataframe[dataframe[args.outer_fold_col].eq(args.outer_fold)].copy()
    model, means, standard_deviations, checkpoint = train_fixed_epochs(
        outer_train,
        args,
        selected["config"],
        final_epochs,
        destination,
    )
    train_loader = make_loader(outer_train, args, means, standard_deviations, False)
    test_loader = make_loader(outer_test, args, means, standard_deviations, False)
    train_predictions, train_embeddings, _ = predict(
        model, train_loader, device, args.outer_fold, None, False
    )
    test_predictions, test_embeddings, attention = predict(
        model, test_loader, device, args.outer_fold, None, True
    )
    train_predictions.to_csv(destination / "outer_train_predictions.csv", index=False)
    test_predictions.to_csv(destination / "outer_test_predictions.csv", index=False)
    save_embedding_bundle(
        destination / "outer_train_embeddings.npz",
        train_embeddings,
        "in-sample outer-training embeddings from the final outer-fold ECG model",
    )
    save_embedding_bundle(
        destination / "outer_test_embeddings.npz",
        test_embeddings,
        "untouched outer-test embeddings from an ECG model trained only on outer training patients",
    )
    attention.to_csv(destination / "outer_test_attention.csv.gz", index=False, compression="gzip")
    test_metrics = metrics_from_predictions(test_predictions)
    atomic_json(destination / "outer_test_metrics.json", test_metrics)
    atomic_json(
        completion_path,
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "outer_fold": args.outer_fold,
            "selected_config": selected["config_name"],
            "final_training_epochs": final_epochs,
            "outer_train_patients": len(outer_train),
            "outer_test_patients": len(outer_test),
            "outer_test_metrics": test_metrics,
            "representation_dim": selected["config"]["representation_dim"],
            "torch_version": torch.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "cache_signature": final_cache_signature,
        },
    )
    logging.info("Completed final outer fold %d", args.outer_fold)


def stage_aggregate(args: argparse.Namespace) -> None:
    labels = read_labels(args)
    prediction_frames = []
    patient_ids = []
    shared = []
    scd_branch = []
    pfd_branch = []
    source_folds = []
    for outer_fold in range(args.outer_splits):
        directory = args.output_root / "final_models" / f"outer_fold_{outer_fold}"
        completion = directory / "run_complete.json"
        if not completion.exists():
            raise FileNotFoundError(f"Final outer fold is incomplete: {directory}")
        predictions = pd.read_csv(
            directory / "outer_test_predictions.csv",
            dtype={PATIENT_ID_COLUMN: "string"},
        )
        prediction_frames.append(predictions)
        embeddings = np.load(directory / "outer_test_embeddings.npz")
        patient_ids.append(embeddings["patient_ids"].astype(str))
        shared.append(embeddings["shared_embeddings"].astype(np.float32))
        scd_branch.append(embeddings["scd_branch_embeddings"].astype(np.float32))
        pfd_branch.append(embeddings["pfd_branch_embeddings"].astype(np.float32))
        source_folds.append(np.full(len(embeddings["patient_ids"]), outer_fold, dtype=np.int16))

    pooled = pd.concat(prediction_frames, ignore_index=True)
    if len(pooled) != len(labels):
        raise RuntimeError(f"Expected {len(labels)} pooled predictions, observed {len(pooled)}.")
    if not pooled[PATIENT_ID_COLUMN].is_unique:
        raise RuntimeError("Pooled outer-test predictions contain duplicate patients.")
    if set(pooled[PATIENT_ID_COLUMN]) != set(labels[args.patient_id_col]):
        raise RuntimeError("Pooled outer-test patient IDs do not match the prepared cohort.")

    aggregate_directory = args.output_root / "pooled_outer_test"
    aggregate_directory.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(aggregate_directory / "pooled_outer_test_predictions.csv", index=False)
    atomic_json(
        aggregate_directory / "pooled_uncalibrated_metrics.json",
        metrics_from_predictions(pooled),
    )
    atomic_npz(
        aggregate_directory / "pooled_outer_test_embeddings_evaluation_only.npz",
        patient_ids=np.concatenate(patient_ids),
        outer_folds=np.concatenate(source_folds),
        shared_embeddings=np.concatenate(shared),
        scd_branch_embeddings=np.concatenate(scd_branch),
        pfd_branch_embeddings=np.concatenate(pfd_branch),
        usage_warning=np.asarray(
            [
                "Evaluation-only pooled embeddings come from different outer-fold models. "
                "Use fold-specific train/test artifacts for downstream model fitting."
            ],
            dtype=str,
        ),
    )
    atomic_json(
        aggregate_directory / "aggregation_manifest.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "patient_count": len(pooled),
            "one_outer_test_prediction_per_patient": True,
            "probabilities_calibrated": False,
            "classification_threshold_selected": False,
            "embedding_use": "evaluation only; preserve outer-fold provenance",
        },
    )

    final_hyperparameter_rows = []
    for outer_fold in range(args.outer_splits):
        selected_path = (
            args.output_root
            / "final_models"
            / f"outer_fold_{outer_fold}"
            / "selected_configuration.json"
        )
        with open(selected_path, encoding="utf-8") as handle:
            selected = json.load(handle)
        final_hyperparameter_rows.append(
            {
                "outer_fold": outer_fold,
                "config_name": selected["config_name"],
                "inner_mean_roc_auc": selected["selection_score"],
                "inner_mean_pr_auc": selected["secondary_mean_pr_auc"],
                "final_training_epochs": selected["final_training_epochs"],
                **selected["config"],
            }
        )
    final_hyperparameters = pd.DataFrame(final_hyperparameter_rows)
    final_hyperparameters.to_csv(
        aggregate_directory / "final_selected_hyperparameters_by_outer_fold.csv",
        index=False,
    )
    frequency_columns = [
        "embedding_dim",
        "tcn_hidden_dim",
        "tcn_layers",
        "enc_dropout",
        "tcn_dropout",
        "branch_dropout",
        "lr",
        "weight_decay",
    ]
    selection_frequency = (
        final_hyperparameters.groupby(frequency_columns, dropna=False)
        .size()
        .reset_index(name="outer_folds_selected")
        .sort_values("outer_folds_selected", ascending=False)
    )
    selection_frequency.to_csv(
        aggregate_directory / "final_hyperparameter_selection_frequency.csv",
        index=False,
    )
    print(f"Pooled outer-test outputs saved to {aggregate_directory}")


def main() -> None:
    args = parse_args()
    args.folds_csv = args.folds_csv.resolve()
    args.features_dir = args.features_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if args.stage in {"tune", "final"}:
            raise RuntimeError("CUDA was requested but is unavailable.")

    if args.stage == "prepare":
        stage_prepare(args)
    elif args.stage == "tune":
        stage_tune(args)
    elif args.stage == "summarize":
        stage_summarize(args)
    elif args.stage == "final":
        stage_final(args)
    elif args.stage == "aggregate":
        stage_aggregate(args)


if __name__ == "__main__":
    main()