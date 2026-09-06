#!/usr/bin/env python3
"""Endpoint-specific nested-CV analysis of frozen patient-level text embeddings.

This revision consumes the detailed-response embedding release. Long responses
are represented by the mean of overlapping chunk-level CLS embeddings so that
content beyond a single 512-token encoder window is not silently discarded.

Stages
------
prepare
    Validate the locked four-year cohort, reuse the ECG inner/outer fold
    assignments, and audit every embedding artifact.
tune
    Evaluate one LLaMA/encoder/classifier candidate within one outer-training
    cohort using its four inner folds.
select
    Select one primary full-response/no-ECG configuration separately inside
    each outer-training cohort.
final
    Hold the selected upstream and classifier configuration fixed and fit all
    prespecified text/ablation arms for one outer fold. Cross-fitted inner
    predictions are exported for calibration and threshold selection.
aggregate
    Pool the untouched outer-test predictions for every analysis arm.
evaluate
    Fit fold-specific Platt calibrators and Youden thresholds using inner OOF
    predictions only; report pooled performance, calibration, threshold
    metrics, and paired comparisons with patient-bootstrap confidence intervals.

SCD and PFD are fitted as independent binary problems. The competing cardiac-
death endpoint is excluded before preprocessing, fitting, calibration, and
evaluation. Every supervised operation is restricted to the appropriate
training fold.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


PATIENT_ID = "Patient ID"
SCD_LABEL = "SCD_4year_label"
PFD_LABEL = "PFD_4year_label"
OUTER_FOLD = "outer_fold"
TASKS = ("scd", "pfd")

PRIMARY_CONDITION = "full_risk_no_ecg_{task}_full"

# The selected LLaMA source and encoder are inherited from the primary arm.
# The deterministic template intentionally has no LLaMA source.
ARMS = {
    "full_risk_no_ecg": {
        "source": "selected",
        "condition": "full_risk_no_ecg_{task}_full",
    },
    "label_only_no_ecg": {
        "source": "selected",
        "condition": "full_risk_no_ecg_{task}_label_only",
    },
    "rationale_only_no_ecg": {
        "source": "selected",
        "condition": "full_risk_no_ecg_{task}_rationale_only",
    },
    "joint_full_risk_no_ecg": {
        "source": "selected",
        "condition": "full_risk_no_ecg_full",
    },
    "neutral_summary_no_ecg": {
        "source": "selected",
        "condition": "neutral_summary_no_ecg",
    },
    "full_risk_with_ecg": {
        "source": "selected",
        "condition": "full_risk_with_ecg_{task}_full",
    },
    "label_only_with_ecg": {
        "source": "selected",
        "condition": "full_risk_with_ecg_{task}_label_only",
    },
    "rationale_only_with_ecg": {
        "source": "selected",
        "condition": "full_risk_with_ecg_{task}_rationale_only",
    },
    "deterministic_template_no_ecg": {
        "source": "DeterministicTemplate",
        "condition": "patient_data_template_no_ecg",
    },
}

PAIRED_COMPARISONS = (
    ("full_vs_template", "full_risk_no_ecg", "deterministic_template_no_ecg"),
    ("full_vs_neutral_summary", "full_risk_no_ecg", "neutral_summary_no_ecg"),
    ("full_vs_label_only", "full_risk_no_ecg", "label_only_no_ecg"),
    ("full_vs_rationale_only", "full_risk_no_ecg", "rationale_only_no_ecg"),
    ("endpoint_specific_vs_joint", "full_risk_no_ecg", "joint_full_risk_no_ecg"),
    ("with_ecg_vs_without_ecg", "full_risk_with_ecg", "full_risk_no_ecg"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested-CV analysis of frozen LLM-derived text embeddings."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "tune", "select", "final", "aggregate", "evaluate", "combine"],
    )
    parser.add_argument("--folds_csv", type=Path, required=True)
    parser.add_argument("--embedding_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--patient_id_col", default=PATIENT_ID)
    parser.add_argument("--scd_label_col", default=SCD_LABEL)
    parser.add_argument("--pfd_label_col", default=PFD_LABEL)
    parser.add_argument("--outer_fold_col", default=OUTER_FOLD)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected_patients", type=int, default=730)
    parser.add_argument("--expected_controls", type=int, default=577)
    parser.add_argument("--expected_scd", type=int, default=71)
    parser.add_argument("--expected_pfd", type=int, default=82)
    parser.add_argument("--expected_pooling", default="cls")
    parser.add_argument("--expected_max_length", type=int, default=512)
    parser.add_argument(
        "--expected_long_text_strategy",
        choices=["mean_chunks", "truncate"],
        default="mean_chunks",
    )

    # One tuning candidate.
    parser.add_argument("--outer_fold", type=int)
    parser.add_argument("--config_name")
    parser.add_argument("--source_name")
    parser.add_argument("--encoder_name", choices=["BioBERT", "ClinicalBERT"])
    parser.add_argument("--classifier", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)

    # Selection/evaluation.
    parser.add_argument("--candidate_config", action="append", default=[])
    parser.add_argument("--auc_tolerance", type=float, default=0.005)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--probability_clip", type=float, default=1e-6)
    parser.add_argument("--platt_C", type=float, default=1e6)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)
    return value % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f".{path.stem}.", dir=path.parent,
        delete=False, encoding="utf-8"
    ) as handle:
        json.dump(value, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".pt", prefix=f".{path.stem}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_"
                   for character in str(value)).strip("_")


def read_folds(args: argparse.Namespace) -> pd.DataFrame:
    path = (
        args.output_root / "analysis_setup" / "nested_patient_folds.csv"
        if args.stage != "prepare"
        else args.folds_csv
    )
    dataframe = pd.read_csv(path, dtype={args.patient_id_col: "string"})
    required = {
        args.patient_id_col,
        args.scd_label_col,
        args.pfd_label_col,
        args.outer_fold_col,
        *[f"inner_fold_outer_{fold}" for fold in range(args.outer_splits)],
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            "The folds CSV must be the nested fold file exported by the ECG analysis. "
            f"Missing columns: {sorted(missing)}"
        )
    if dataframe[args.patient_id_col].isna().any() or not dataframe[
        args.patient_id_col
    ].is_unique:
        raise ValueError("Patient IDs must be complete and unique.")
    dataframe[args.outer_fold_col] = dataframe[args.outer_fold_col].astype(int)
    label_column = args.scd_label_col if args.task == "scd" else args.pfd_label_col
    eligible = dataframe[label_column].notna()
    dataframe = dataframe.loc[eligible].copy().reset_index(drop=True)
    dataframe[label_column] = pd.to_numeric(dataframe[label_column], errors="raise").astype(int)
    expected_task_patients = (
        args.expected_controls + (args.expected_scd if args.task == "scd" else args.expected_pfd)
    )
    if len(dataframe) != expected_task_patients:
        raise ValueError(
            f"Expected {expected_task_patients} eligible {args.task.upper()} patients; "
            f"observed {len(dataframe)}. Competing endpoints must be encoded as missing."
        )
    return dataframe


def outcome_counts(dataframe: pd.DataFrame, args: argparse.Namespace) -> dict:
    label_column = args.scd_label_col if args.task == "scd" else args.pfd_label_col
    labels = dataframe[label_column]
    counts = {
        "patients": int(len(dataframe)),
        "controls": int(labels.eq(0).sum()),
        "events": int(labels.eq(1).sum()),
    }
    expected = {
        "patients": args.expected_controls + (
            args.expected_scd if args.task == "scd" else args.expected_pfd
        ),
        "controls": args.expected_controls,
        "events": args.expected_scd if args.task == "scd" else args.expected_pfd,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and counts[key] != expected_value:
            raise ValueError(
                f"Expected {expected_value} {key}; observed {counts[key]}."
            )
    return counts


def embedding_paths(
    args: argparse.Namespace, source: str, encoder: str, condition: str
) -> tuple[Path, Path]:
    directory = (
        args.embedding_root
        / stable_slug(source)
        / stable_slug(encoder)
        / stable_slug(condition)
    )
    return directory / "embeddings.npz", directory / "manifest.json"


def load_embedding_matrix(
    args: argparse.Namespace,
    dataframe: pd.DataFrame,
    source: str,
    encoder: str,
    condition: str,
) -> tuple[np.ndarray, dict]:
    embedding_path, manifest_path = embedding_paths(
        args, source, encoder, condition
    )
    if not embedding_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing embedding artifact for {source}/{encoder}/{condition}: "
            f"{embedding_path}"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    artifact = np.load(embedding_path, allow_pickle=False)
    required = {
        "patient_ids",
        "embeddings",
        "text_sha256",
        "token_lengths",
        "chunk_counts",
        "was_chunked",
        "was_truncated",
    }
    missing = required - set(artifact.files)
    if missing:
        raise ValueError(f"{embedding_path} lacks arrays: {sorted(missing)}")
    patient_ids = artifact["patient_ids"].astype(str)
    embeddings = artifact["embeddings"].astype(np.float32)
    text_sha256 = artifact["text_sha256"].astype(str)
    token_lengths = artifact["token_lengths"].astype(np.int64)
    chunk_counts = artifact["chunk_counts"].astype(np.int64)
    was_chunked = artifact["was_chunked"].astype(bool)
    was_truncated = artifact["was_truncated"].astype(bool)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(patient_ids):
        raise ValueError(f"Invalid embedding shape in {embedding_path}: {embeddings.shape}")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError(f"Duplicate patient IDs in {embedding_path}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Nonfinite embeddings in {embedding_path}")

    row_count = len(patient_ids)
    for array_name, values in {
        "text_sha256": text_sha256,
        "token_lengths": token_lengths,
        "chunk_counts": chunk_counts,
        "was_chunked": was_chunked,
        "was_truncated": was_truncated,
    }.items():
        if values.ndim != 1 or len(values) != row_count:
            raise ValueError(
                f"{embedding_path}: {array_name} must contain one value per patient."
            )
    if np.any(token_lengths <= 0) or np.any(chunk_counts <= 0):
        raise ValueError(f"Invalid token or chunk counts in {embedding_path}")
    if not np.array_equal(was_chunked, chunk_counts > 1):
        raise ValueError(f"Chunk indicators disagree with chunk counts in {embedding_path}")

    expected_manifest = {
        "artifact_type": "frozen_text_embeddings",
        "source_name": source,
        "encoder_name": encoder,
        "condition": condition,
        "patient_count": row_count,
        "encoder_frozen": True,
        "pooling": args.expected_pooling,
        "max_length": args.expected_max_length,
        "long_text_strategy": args.expected_long_text_strategy,
        "truncated_patient_count": int(was_truncated.sum()),
        "chunked_patient_count": int(was_chunked.sum()),
        "maximum_chunks_per_patient": int(chunk_counts.max()),
        "maximum_untruncated_token_length": int(token_lengths.max()),
    }
    for key, expected_value in expected_manifest.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"{manifest_path}: expected {key}={expected_value!r}; "
                f"observed {manifest.get(key)!r}."
            )
    if args.expected_long_text_strategy == "mean_chunks" and was_truncated.any():
        raise ValueError(
            f"{embedding_path}: detailed responses were unexpectedly truncated."
        )
    if manifest.get("embedding_shape") != list(embeddings.shape):
        raise ValueError(f"Embedding shape disagrees with manifest: {embedding_path}")
    if not isinstance(manifest.get("cache_signature"), dict):
        raise ValueError(f"Missing cache signature in {manifest_path}")

    row_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    requested_ids = dataframe[args.patient_id_col].astype(str).tolist()
    missing_ids = [patient_id for patient_id in requested_ids if patient_id not in row_by_id]
    if missing_ids:
        raise ValueError(
            f"{embedding_path} is missing {len(missing_ids)} patients; "
            f"examples={missing_ids[:10]}"
        )
    aligned = embeddings[[row_by_id[patient_id] for patient_id in requested_ids]]
    return aligned, manifest


class LinearBinary(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(values).squeeze(-1)


class MLPBinary(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, layers: int, dropout: float
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            blocks.extend(
                [nn.Linear(current, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            current = hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(current, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        representation = self.trunk(values)
        return self.head(representation).squeeze(-1)


def model_config(args: argparse.Namespace) -> dict:
    return {
        "task": args.task,
        "source_name": args.source_name,
        "encoder_name": args.encoder_name,
        "condition": PRIMARY_CONDITION.format(task=args.task),
        "classifier": args.classifier,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
    }


def build_model(input_dim: int, config: dict, device: torch.device) -> nn.Module:
    if config["classifier"] == "linear":
        model = LinearBinary(input_dim)
    else:
        model = MLPBinary(
            input_dim,
            int(config["hidden_dim"]),
            int(config["layers"]),
            float(config["dropout"]),
        )
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def labels_for(dataframe: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    column = args.scd_label_col if args.task == "scd" else args.pfd_label_col
    labels = dataframe[column].to_numpy(float)
    if not np.isfinite(labels).all():
        raise ValueError(f"Ineligible competing-endpoint rows remain in {args.task} data")
    return labels


def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviations = values.std(axis=0, dtype=np.float64).astype(np.float32)
    standard_deviations[~np.isfinite(standard_deviations) | (standard_deviations < 1e-8)] = 1.0
    return means, standard_deviations


def transform(values: np.ndarray, means: np.ndarray, sds: np.ndarray) -> np.ndarray:
    transformed = ((values - means) / sds).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Nonfinite standardized text embeddings.")
    return transformed


def task_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("A training split lacks one class for the selected outcome.")
    return torch.tensor(negatives / positives, dtype=torch.float32, device=device)


def masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    return criterion(logits, labels)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"n": len(labels), "events": int(labels.sum()), "roc_auc": None,
                "pr_auc": None, "brier_score": None}
    return {
        "n": int(len(labels)),
        "events": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def metrics_from_arrays(labels: np.ndarray, probabilities: np.ndarray, task: str) -> dict:
    result = binary_metrics(labels.astype(int), probabilities)
    output = {task: result}
    output["mean_roc_auc"] = result["roc_auc"]
    output["mean_pr_auc"] = result["pr_auc"]
    return output


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module, values: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(values).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


@dataclass
class FitResult:
    model: nn.Module
    means: np.ndarray
    standard_deviations: np.ndarray
    best_epoch: int
    history: list[dict]
    validation_probabilities: np.ndarray
    trainable_parameters: int


def fit_model(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    config: dict,
    device: torch.device,
    seed: int,
    fixed_epochs: int | None = None,
) -> FitResult:
    set_seed(seed)
    means, sds = fit_scaler(train_values)
    train_scaled = transform(train_values, means, sds)
    validation_scaled = transform(validation_values, means, sds)
    model = build_model(train_scaled.shape[1], config, device)
    trainable_parameters = count_parameters(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"])
    )
    train_tensor = torch.from_numpy(train_scaled).to(device)
    label_tensor = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    weight = task_pos_weight(train_labels, device)

    maximum_epochs = int(fixed_epochs if fixed_epochs is not None else config["epochs"])
    best_score = -np.inf
    best_epoch = maximum_epochs
    best_state = None
    history = []
    stale = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tensor)
        loss = masked_loss(logits, label_tensor, weight)
        loss.backward()
        optimizer.step()

        validation_probabilities = predict_probabilities(
            model, validation_scaled, device
        )
        metrics = metrics_from_arrays(
            validation_labels, validation_probabilities, config["task"]
        )
        score = metrics["mean_roc_auc"]
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "mean_roc_auc": score,
                "mean_pr_auc": metrics["mean_pr_auc"],
                "outcome": config["task"].upper(),
                "roc_auc": metrics[config["task"]]["roc_auc"],
                "pr_auc": metrics[config["task"]]["pr_auc"],
            }
        )

        if fixed_epochs is None:
            if score > best_score + 1e-12:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if epoch >= int(config["min_epochs"]) and stale >= int(config["patience"]):
                break

    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("No best model state was recorded.")
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum_epochs

    validation_probabilities = predict_probabilities(
        model, validation_scaled, device
    )
    return FitResult(
        model=model,
        means=means,
        standard_deviations=sds,
        best_epoch=best_epoch,
        history=history,
        validation_probabilities=validation_probabilities,
        trainable_parameters=trainable_parameters,
    )


def prediction_frame(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    probabilities: np.ndarray,
    outer_fold: int,
    inner_fold: int | None,
    arm: str | None = None,
) -> pd.DataFrame:
    label_column = args.scd_label_col if args.task == "scd" else args.pfd_label_col
    output = pd.DataFrame(
        {
            args.patient_id_col: dataframe[args.patient_id_col].astype(str).to_numpy(),
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "outcome": args.task.upper(),
            f"{args.task}_label": dataframe[label_column].to_numpy(int),
            f"{args.task}_probability": probabilities,
        }
    )
    if arm is not None:
        output.insert(1, "arm", arm)
    return output


def stage_prepare(args: argparse.Namespace) -> None:
    dataframe = read_folds(args)
    counts = outcome_counts(dataframe, args)
    observed_outer = sorted(dataframe[args.outer_fold_col].unique().tolist())
    if observed_outer != list(range(args.outer_splits)):
        raise ValueError(f"Unexpected outer folds: {observed_outer}")
    for outer_fold in range(args.outer_splits):
        inner_column = f"inner_fold_outer_{outer_fold}"
        outer_train = dataframe[dataframe[args.outer_fold_col].ne(outer_fold)]
        outer_test = dataframe[dataframe[args.outer_fold_col].eq(outer_fold)]
        if outer_train[inner_column].isna().any():
            raise ValueError(f"Missing inner assignments in {inner_column}")
        if outer_test[inner_column].notna().any():
            raise ValueError(f"Outer-test patients have assignments in {inner_column}")
        observed_inner = sorted(outer_train[inner_column].astype(int).unique().tolist())
        if observed_inner != list(range(args.inner_splits)):
            raise ValueError(f"Unexpected assignments in {inner_column}: {observed_inner}")

    audit_rows = []
    sources = ("LLaMA3.1-8B", "LLaMA3.2-3B")
    encoders = ("BioBERT", "ClinicalBERT")
    conditions = sorted(
        {
            spec["condition"].format(task=args.task)
            for spec in ARMS.values()
            if spec["source"] == "selected"
        }
    )
    for source in sources:
        for encoder in encoders:
            for condition in conditions:
                matrix, manifest = load_embedding_matrix(
                    args, dataframe, source, encoder, condition
                )
                audit_rows.append(
                    {
                        "source_name": source,
                        "encoder_name": encoder,
                        "condition": condition,
                        "rows": matrix.shape[0],
                        "embedding_dimension": matrix.shape[1],
                        "encoder_frozen": manifest.get("encoder_frozen"),
                        "resolved_commit": manifest.get("resolved_commit"),
                        "pooling": manifest.get("pooling"),
                        "max_length": manifest.get("max_length"),
                        "long_text_strategy": manifest.get("long_text_strategy"),
                        "chunked_patient_count": manifest.get("chunked_patient_count"),
                        "maximum_chunks_per_patient": manifest.get("maximum_chunks_per_patient"),
                        "maximum_untruncated_token_length": manifest.get(
                            "maximum_untruncated_token_length"
                        ),
                        "truncated_patient_count": manifest.get("truncated_patient_count"),
                    }
                )
    for encoder in encoders:
        matrix, manifest = load_embedding_matrix(
            args, dataframe, "DeterministicTemplate", encoder,
            "patient_data_template_no_ecg"
        )
        audit_rows.append(
            {
                "source_name": "DeterministicTemplate",
                "encoder_name": encoder,
                "condition": "patient_data_template_no_ecg",
                "rows": matrix.shape[0],
                "embedding_dimension": matrix.shape[1],
                "encoder_frozen": manifest.get("encoder_frozen"),
                "resolved_commit": manifest.get("resolved_commit"),
                "pooling": manifest.get("pooling"),
                "max_length": manifest.get("max_length"),
                "long_text_strategy": manifest.get("long_text_strategy"),
                "chunked_patient_count": manifest.get("chunked_patient_count"),
                "maximum_chunks_per_patient": manifest.get("maximum_chunks_per_patient"),
                "maximum_untruncated_token_length": manifest.get(
                    "maximum_untruncated_token_length"
                ),
                "truncated_patient_count": manifest.get("truncated_patient_count"),
            }
        )

    setup = args.output_root / "analysis_setup"
    setup.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(setup / "nested_patient_folds.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(setup / "embedding_audit.csv", index=False)
    atomic_json(
        setup / "prepare_manifest.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "source_folds_csv": str(args.folds_csv.resolve()),
            "source_folds_sha256": sha256_file(args.folds_csv),
            "embedding_root": str(args.embedding_root.resolve()),
            "patient_counts": counts,
            "task": args.task,
            "competing_endpoint_excluded_before_preprocessing": True,
            "outer_splits": args.outer_splits,
            "inner_splits": args.inner_splits,
            "seed": args.seed,
            "embedding_artifacts_audited": len(audit_rows),
            "expected_embedding_policy": {
                "pooling": args.expected_pooling,
                "max_length": args.expected_max_length,
                "long_text_strategy": args.expected_long_text_strategy,
                "truncated_patient_count": 0,
            },
            "analysis_arms": ARMS,
        },
    )
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print(f"Prepared text analysis: {setup}")


def stage_tune(args: argparse.Namespace) -> None:
    required = [args.outer_fold, args.config_name, args.source_name, args.encoder_name]
    if any(value is None for value in required):
        raise ValueError(
            "Tune requires --outer_fold, --config_name, --source_name, and --encoder_name."
        )
    dataframe = read_folds(args)
    config = model_config(args)
    matrix, manifest = load_embedding_matrix(
        args, dataframe, args.source_name, args.encoder_name,
        PRIMARY_CONDITION.format(task=args.task)
    )
    destination = (
        args.output_root / "tuning" / args.config_name
        / f"outer_fold_{args.outer_fold}"
    )
    summary_path = destination / "tuning_summary.json"
    signature = {
        "folds_sha256": sha256_file(
            args.output_root / "analysis_setup" / "nested_patient_folds.csv"
        ),
        "embedding_file": manifest["embedding_file"],
        "cache_signature": manifest["cache_signature"],
        "config": config,
        "outer_fold": args.outer_fold,
        "seed": args.seed,
    }
    if summary_path.exists() and not args.overwrite:
        with open(summary_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("cache_signature") != signature:
            raise RuntimeError(
                f"Completed tuning output does not match request: {destination}. "
                "Use a new output root or --overwrite."
            )
        print(f"Using completed tuning output: {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    outer_train_mask = dataframe[args.outer_fold_col].ne(args.outer_fold)
    outer_train_positions = np.flatnonzero(outer_train_mask.to_numpy())
    inner_column = f"inner_fold_outer_{args.outer_fold}"
    pooled_frames = []
    best_epochs = []
    trainable_parameters = None
    device = torch.device(args.device)

    for inner_fold in range(args.inner_splits):
        validation_mask = outer_train_mask & dataframe[inner_column].eq(inner_fold)
        training_mask = outer_train_mask & dataframe[inner_column].ne(inner_fold)
        training_positions = np.flatnonzero(training_mask.to_numpy())
        validation_positions = np.flatnonzero(validation_mask.to_numpy())
        seed = stable_seed(args.seed, args.config_name, args.outer_fold, inner_fold)
        result = fit_model(
            matrix[training_positions],
            labels_for(dataframe.iloc[training_positions], args),
            matrix[validation_positions],
            labels_for(dataframe.iloc[validation_positions], args),
            config,
            device,
            seed,
        )
        trainable_parameters = result.trainable_parameters
        fold_directory = destination / f"inner_fold_{inner_fold}"
        fold_directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result.history).to_csv(
            fold_directory / "epoch_metrics.csv", index=False
        )
        np.savez_compressed(
            fold_directory / "preprocessing.npz",
            means=result.means,
            standard_deviations=result.standard_deviations,
        )
        atomic_torch_save(
            fold_directory / "best_checkpoint.pt",
            {
                "state_dict": result.model.state_dict(),
                "config": config,
                "input_dim": matrix.shape[1],
                "best_epoch": result.best_epoch,
                "seed": seed,
            },
        )
        frame = prediction_frame(
            dataframe.iloc[validation_positions], args,
            result.validation_probabilities, args.outer_fold, inner_fold
        )
        frame.to_csv(fold_directory / "validation_predictions.csv", index=False)
        atomic_json(
            fold_directory / "run_complete.json",
            {
                "completed": True,
                "best_epoch": result.best_epoch,
                "validation_metrics": metrics_from_arrays(
                    labels_for(dataframe.iloc[validation_positions], args),
                    result.validation_probabilities,
                    args.task,
                ),
                "seed": seed,
            },
        )
        pooled_frames.append(frame)
        best_epochs.append(result.best_epoch)

    pooled = pd.concat(pooled_frames, ignore_index=True)
    expected_ids = set(dataframe.iloc[outer_train_positions][args.patient_id_col].astype(str))
    if set(pooled[args.patient_id_col].astype(str)) != expected_ids:
        raise RuntimeError("Inner OOF predictions do not cover the outer-training cohort.")
    pooled.to_csv(destination / "pooled_inner_oof_predictions.csv", index=False)
    labels = pooled[f"{args.task}_label"].to_numpy(float)
    probabilities = pooled[f"{args.task}_probability"].to_numpy(float)
    metrics = metrics_from_arrays(labels, probabilities, args.task)
    atomic_json(
        summary_path,
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "config_name": args.config_name,
            "config": config,
            "outer_fold": args.outer_fold,
            "inner_best_epochs": best_epochs,
            "trainable_parameters": trainable_parameters,
            "metrics": metrics,
            "selection_score": metrics["mean_roc_auc"],
            "secondary_mean_pr_auc": metrics["mean_pr_auc"],
            "cache_signature": signature,
        },
    )
    print(
        f"Completed {args.config_name} outer={args.outer_fold}: "
        f"mean AUC={metrics['mean_roc_auc']:.4f}, "
        f"mean PR={metrics['mean_pr_auc']:.4f}"
    )


def select_one_outer(args: argparse.Namespace, outer_fold: int) -> dict:
    if not args.candidate_config:
        raise ValueError("Selection requires repeated --candidate_config values.")
    candidates = []
    for name in args.candidate_config:
        path = (
            args.output_root / "tuning" / name
            / f"outer_fold_{outer_fold}" / "tuning_summary.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing tuning summary: {path}")
        with open(path, encoding="utf-8") as handle:
            candidates.append(json.load(handle))
    maximum = max(candidate["selection_score"] for candidate in candidates)
    eligible = [
        candidate for candidate in candidates
        if candidate["selection_score"] >= maximum - args.auc_tolerance
    ]
    eligible.sort(
        key=lambda candidate: (
            candidate["trainable_parameters"],
            -candidate["secondary_mean_pr_auc"],
            candidate["config_name"],
        )
    )
    selected = copy.deepcopy(eligible[0])
    selected["maximum_candidate_mean_auc"] = maximum
    selected["auc_tolerance"] = args.auc_tolerance
    selected["selection_rule"] = (
        "maximum pooled inner endpoint-specific ROC-AUC; within tolerance choose "
        "fewest parameters, then highest endpoint-specific PR-AUC, then configuration name"
    )
    return selected


def stage_select(args: argparse.Namespace) -> None:
    rows = []
    selected_by_fold = {}
    all_rows = []
    for outer_fold in range(args.outer_splits):
        selected = select_one_outer(args, outer_fold)
        selected_by_fold[str(outer_fold)] = selected
        for name in args.candidate_config:
            path = (
                args.output_root / "tuning" / name
                / f"outer_fold_{outer_fold}" / "tuning_summary.json"
            )
            with open(path, encoding="utf-8") as handle:
                candidate = json.load(handle)
            row = {
                "outer_fold": outer_fold,
                "config_name": name,
                "selected": name == selected["config_name"],
                "mean_roc_auc": candidate["selection_score"],
                "mean_pr_auc": candidate["secondary_mean_pr_auc"],
                "trainable_parameters": candidate["trainable_parameters"],
                **candidate["config"],
            }
            all_rows.append(row)
            if row["selected"]:
                rows.append(row)
    selection_directory = args.output_root / "selection"
    selection_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(
        selection_directory / "all_primary_candidate_results.csv", index=False
    )
    pd.DataFrame(rows).to_csv(
        selection_directory / "selected_primary_configuration_by_outer_fold.csv",
        index=False,
    )
    atomic_json(
        selection_directory / "selected_primary_configuration_by_outer_fold.json",
        {
            "completed": True,
            "selected_by_outer_fold": selected_by_fold,
            "task": args.task,
            "selection_condition": PRIMARY_CONDITION.format(task=args.task),
            "outer_test_outcomes_used": False,
        },
    )
    print(pd.DataFrame(rows).to_string(index=False))


def load_selection(args: argparse.Namespace, outer_fold: int) -> dict:
    path = (
        args.output_root / "selection"
        / "selected_primary_configuration_by_outer_fold.json"
    )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["selected_by_outer_fold"][str(outer_fold)]


def arm_embedding_spec(arm: str, selected_config: dict) -> tuple[str, str, str]:
    spec = ARMS[arm]
    source = (
        selected_config["source_name"]
        if spec["source"] == "selected"
        else spec["source"]
    )
    task = selected_config["task"]
    return source, selected_config["encoder_name"], spec["condition"].format(task=task)


def save_fitted_model(
    destination: Path, result: FitResult, config: dict, seed: int
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.history).to_csv(
        destination / "training_history.csv", index=False
    )
    np.savez_compressed(
        destination / "preprocessing.npz",
        means=result.means,
        standard_deviations=result.standard_deviations,
    )
    atomic_torch_save(
        destination / "checkpoint.pt",
        {
            "state_dict": result.model.state_dict(),
            "config": config,
            "input_dim": len(result.means),
            "training_epochs": result.best_epoch,
            "seed": seed,
        },
    )


def stage_final(args: argparse.Namespace) -> None:
    if args.outer_fold is None:
        raise ValueError("Final requires --outer_fold.")
    dataframe = read_folds(args)
    selected = load_selection(args, args.outer_fold)
    config = selected["config"]
    inner_epochs = [int(value) for value in selected["inner_best_epochs"]]
    final_epochs = max(1, int(math.floor(float(np.median(inner_epochs)) + 0.5)))
    outer_train_mask = dataframe[args.outer_fold_col].ne(args.outer_fold)
    outer_test_mask = dataframe[args.outer_fold_col].eq(args.outer_fold)
    outer_train_positions = np.flatnonzero(outer_train_mask.to_numpy())
    outer_test_positions = np.flatnonzero(outer_test_mask.to_numpy())
    inner_column = f"inner_fold_outer_{args.outer_fold}"
    fold_directory = args.output_root / "final_models" / f"outer_fold_{args.outer_fold}"
    completion = fold_directory / "run_complete.json"
    signature = {
        "selected_config_name": selected["config_name"],
        "selected_config": config,
        "arms": ARMS,
        "inner_epochs": inner_epochs,
        "final_epochs": final_epochs,
        "folds_sha256": sha256_file(
            args.output_root / "analysis_setup" / "nested_patient_folds.csv"
        ),
    }
    if completion.exists() and not args.overwrite:
        with open(completion, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("cache_signature") != signature:
            raise RuntimeError(
                f"Completed final output does not match request: {fold_directory}"
            )
        print(f"Using completed final outer fold: {fold_directory}")
        return
    fold_directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        fold_directory / "selected_primary_configuration.json",
        {
            **selected,
            "final_training_epochs": final_epochs,
            "epoch_rule": "nearest integer to median selected inner best epoch",
        },
    )
    device = torch.device(args.device)

    for arm in ARMS:
        source, encoder, condition = arm_embedding_spec(arm, config)
        arm_config = copy.deepcopy(config)
        arm_config.update(
            {
                "arm": arm,
                "source_name": source,
                "encoder_name": encoder,
                "condition": condition,
            }
        )
        matrix, manifest = load_embedding_matrix(
            args, dataframe, source, encoder, condition
        )
        arm_directory = fold_directory / "arms" / arm
        arm_directory.mkdir(parents=True, exist_ok=True)
        inner_frames = []

        for inner_fold in range(args.inner_splits):
            training_mask = outer_train_mask & dataframe[inner_column].ne(inner_fold)
            validation_mask = outer_train_mask & dataframe[inner_column].eq(inner_fold)
            training_positions = np.flatnonzero(training_mask.to_numpy())
            validation_positions = np.flatnonzero(validation_mask.to_numpy())
            seed = stable_seed(
                args.seed, "final_inner", selected["config_name"], arm,
                args.outer_fold, inner_fold
            )
            result = fit_model(
                matrix[training_positions],
                labels_for(dataframe.iloc[training_positions], args),
                matrix[validation_positions],
                labels_for(dataframe.iloc[validation_positions], args),
                arm_config,
                device,
                seed,
                fixed_epochs=inner_epochs[inner_fold],
            )
            inner_destination = arm_directory / "selected_inner_folds" / f"inner_fold_{inner_fold}"
            save_fitted_model(inner_destination, result, arm_config, seed)
            frame = prediction_frame(
                dataframe.iloc[validation_positions], args,
                result.validation_probabilities, args.outer_fold, inner_fold, arm
            )
            frame.to_csv(inner_destination / "validation_predictions.csv", index=False)
            inner_frames.append(frame)

        inner_oof = pd.concat(inner_frames, ignore_index=True)
        if not inner_oof[args.patient_id_col].is_unique:
            raise RuntimeError(f"Duplicate inner OOF patients for {arm}")
        inner_oof.to_csv(arm_directory / "inner_oof_predictions.csv", index=False)

        seed = stable_seed(
            args.seed, "final_outer", selected["config_name"], arm, args.outer_fold
        )
        final_result = fit_model(
            matrix[outer_train_positions],
            labels_for(dataframe.iloc[outer_train_positions], args),
            matrix[outer_test_positions],
            labels_for(dataframe.iloc[outer_test_positions], args),
            arm_config,
            device,
            seed,
            fixed_epochs=final_epochs,
        )
        save_fitted_model(
            arm_directory / "outer_model", final_result, arm_config, seed
        )
        train_scaled = transform(
            matrix[outer_train_positions], final_result.means,
            final_result.standard_deviations
        )
        train_probabilities = predict_probabilities(
            final_result.model, train_scaled, device
        )
        train_frame = prediction_frame(
            dataframe.iloc[outer_train_positions], args, train_probabilities,
            args.outer_fold, None, arm
        )
        test_frame = prediction_frame(
            dataframe.iloc[outer_test_positions], args,
            final_result.validation_probabilities, args.outer_fold, None, arm
        )
        train_frame.to_csv(arm_directory / "outer_train_predictions.csv", index=False)
        test_frame.to_csv(arm_directory / "outer_test_predictions.csv", index=False)
        atomic_json(
            arm_directory / "arm_manifest.json",
            {
                "completed": True,
                "arm": arm,
                "source_name": source,
                "encoder_name": encoder,
                "condition": condition,
                "embedding_manifest": manifest,
                "selected_primary_config_name": selected["config_name"],
                "classifier_config": arm_config,
                "inner_training_epochs": inner_epochs,
                "outer_training_epochs": final_epochs,
                "outer_test_outcomes_used_for_training_or_selection": False,
            },
        )
        print(f"Completed outer={args.outer_fold}, arm={arm}")

    atomic_json(
        completion,
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "outer_fold": args.outer_fold,
            "selected_config_name": selected["config_name"],
            "arms_completed": list(ARMS),
            "cache_signature": signature,
        },
    )


def stage_aggregate(args: argparse.Namespace) -> None:
    dataframe = read_folds(args)
    output_directory = args.output_root / "pooled_outer_test"
    output_directory.mkdir(parents=True, exist_ok=True)
    long_frames = []
    raw_metrics = []
    for arm in ARMS:
        frames = []
        for outer_fold in range(args.outer_splits):
            path = (
                args.output_root / "final_models" / f"outer_fold_{outer_fold}"
                / "arms" / arm / "outer_test_predictions.csv"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing final predictions: {path}")
            frames.append(pd.read_csv(path, dtype={args.patient_id_col: "string"}))
        pooled = pd.concat(frames, ignore_index=True)
        if len(pooled) != len(dataframe) or not pooled[args.patient_id_col].is_unique:
            raise RuntimeError(f"Invalid pooled predictions for {arm}")
        if set(pooled[args.patient_id_col]) != set(dataframe[args.patient_id_col]):
            raise RuntimeError(f"Patient mismatch in pooled predictions for {arm}")
        arm_directory = output_directory / arm
        arm_directory.mkdir(parents=True, exist_ok=True)
        pooled.to_csv(arm_directory / "pooled_outer_test_predictions.csv", index=False)
        long_frames.append(pooled)
        labels = pooled[f"{args.task}_label"].to_numpy(float)
        probabilities = pooled[f"{args.task}_probability"].to_numpy(float)
        metrics = metrics_from_arrays(labels, probabilities, args.task)
        raw_metrics.append(
            {"arm": arm, "outcome": args.task.upper(), **metrics[args.task]}
        )
    pd.concat(long_frames, ignore_index=True).to_csv(
        output_directory / "all_arms_pooled_outer_test_predictions.csv", index=False
    )
    pd.DataFrame(raw_metrics).to_csv(
        output_directory / "all_arms_raw_point_metrics.csv", index=False
    )
    atomic_json(
        output_directory / "aggregation_manifest.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "task": args.task,
            "eligible_patient_count_per_arm": len(dataframe),
            "competing_endpoint_excluded": True,
            "arms": list(ARMS),
            "one_outer_test_prediction_per_patient_per_arm": True,
            "probabilities_calibrated": False,
        },
    )
    print(pd.DataFrame(raw_metrics).to_string(index=False))


def clipped_logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, float), epsilon, 1 - epsilon)
    return np.log(probabilities / (1 - probabilities))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1 + exponent)
    return output


def fit_platt(
    labels: np.ndarray, probabilities: np.ndarray, epsilon: float, c_value: float
) -> dict:
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000)
    model.fit(clipped_logit(probabilities, epsilon).reshape(-1, 1), labels)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def apply_platt(probabilities: np.ndarray, parameters: dict, epsilon: float) -> np.ndarray:
    return sigmoid(
        parameters["intercept"]
        + parameters["slope"] * clipped_logit(probabilities, epsilon)
    )


def calibration_metrics(
    labels: np.ndarray, probabilities: np.ndarray, epsilon: float, c_value: float
) -> dict:
    parameters = fit_platt(labels, probabilities, epsilon, c_value)
    logits = clipped_logit(probabilities, epsilon)

    def score(intercept: float) -> float:
        return float(np.sum(labels - sigmoid(intercept + logits)))

    lower, upper = -30.0, 30.0
    citl = float(brentq(score, lower, upper))
    return {
        "calibration_intercept": parameters["intercept"],
        "calibration_slope": parameters["slope"],
        "calibration_in_the_large": citl,
    }


def select_youden(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    specificity = 1 - false_positive_rate
    youden = true_positive_rate + specificity - 1
    maximum = np.nanmax(youden)
    eligible = np.flatnonzero(np.isclose(youden, maximum, rtol=0, atol=1e-12))
    selected = sorted(
        eligible,
        key=lambda index: (-true_positive_rate[index], thresholds[index]),
    )[0]
    return {
        "threshold": float(thresholds[selected]),
        "youden_j": float(youden[selected]),
        "sensitivity": float(true_positive_rate[selected]),
        "specificity": float(specificity[selected]),
    }


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    divide = lambda numerator, denominator: float(numerator / denominator) if denominator else np.nan
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    ppv = divide(tp, tp + fp)
    npv = divide(tn, tn + fn)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": divide(tp + tn, tp + tn + fp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_performance(
    dataframe: pd.DataFrame,
    task: str,
    probability_column: str,
    replicates: int,
    seed: int,
    epsilon: float,
    c_value: float,
) -> list[dict]:
    applicable = dataframe[f"{task}_label"].notna()
    labels = dataframe.loc[applicable, f"{task}_label"].astype(int).to_numpy()
    probabilities = dataframe.loc[applicable, probability_column].astype(float).to_numpy()
    point = {
        **binary_metrics(labels, probabilities),
        **calibration_metrics(labels, probabilities, epsilon, c_value),
    }
    metric_names = (
        "roc_auc", "pr_auc", "brier_score", "calibration_intercept",
        "calibration_slope", "calibration_in_the_large"
    )
    values = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        try:
            result = {
                **binary_metrics(sampled_labels, sampled_probabilities),
                **calibration_metrics(
                    sampled_labels, sampled_probabilities, epsilon, c_value
                ),
            }
        except (ValueError, RuntimeError):
            continue
        for metric in metric_names:
            values[metric].append(result[metric])
    rows = []
    for metric in metric_names:
        distribution = np.asarray(values[metric], float)
        rows.append(
            {
                "outcome": task.upper(),
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.percentile(distribution, 2.5)),
                "ci_upper": float(np.percentile(distribution, 97.5)),
                "bootstrap_replicates": int(len(distribution)),
                "n": int(len(labels)),
                "events": int(labels.sum()),
            }
        )
    return rows


def bootstrap_threshold_metrics(
    dataframe: pd.DataFrame, task: str, replicates: int, seed: int
) -> tuple[list[dict], dict]:
    applicable = dataframe[f"{task}_label"].notna()
    labels = dataframe.loc[applicable, f"{task}_label"].astype(int).to_numpy()
    predictions = dataframe.loc[applicable, f"{task}_predicted_class"].astype(int).to_numpy()
    point = classification_metrics(labels, predictions)
    rate_names = (
        "sensitivity", "specificity", "ppv", "npv", "accuracy", "f1",
        "balanced_accuracy"
    )
    values = {metric: [] for metric in rate_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        result = classification_metrics(labels[indices], predictions[indices])
        for metric in rate_names:
            if np.isfinite(result[metric]):
                values[metric].append(result[metric])
    rows = []
    for metric in rate_names:
        distribution = np.asarray(values[metric], float)
        rows.append(
            {
                "outcome": task.upper(), "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.percentile(distribution, 2.5)),
                "ci_upper": float(np.percentile(distribution, 97.5)),
                "bootstrap_replicates": int(len(distribution)),
                "n": int(len(labels)), "events": int(labels.sum()),
            }
        )
    counts = {
        "outcome": task.upper(), "n": len(labels), "events": int(labels.sum()),
        **{key: point[key] for key in ("tn", "fp", "fn", "tp")},
    }
    return rows, counts


def metric_value(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(labels, probabilities))
    if metric == "pr_auc":
        return float(average_precision_score(labels, probabilities))
    if metric == "brier_score":
        return float(brier_score_loss(labels, probabilities))
    raise ValueError(metric)


def paired_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    task: str,
    metric: str,
    probability_column: str,
    replicates: int,
    seed: int,
) -> dict:
    columns = [PATIENT_ID, f"{task}_label", probability_column]
    left_subset = left[columns].rename(columns={probability_column: "left_probability"})
    right_subset = right[[PATIENT_ID, probability_column]].rename(
        columns={probability_column: "right_probability"}
    )
    merged = left_subset.merge(right_subset, on=PATIENT_ID, how="inner", validate="one_to_one")
    merged = merged[merged[f"{task}_label"].notna()].reset_index(drop=True)
    labels = merged[f"{task}_label"].astype(int).to_numpy()
    left_probabilities = merged["left_probability"].astype(float).to_numpy()
    right_probabilities = merged["right_probability"].astype(float).to_numpy()
    point = metric_value(labels, left_probabilities, metric) - metric_value(
        labels, right_probabilities, metric
    )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        differences.append(
            metric_value(sampled_labels, left_probabilities[indices], metric)
            - metric_value(sampled_labels, right_probabilities[indices], metric)
        )
    distribution = np.asarray(differences, float)
    p_value = min(
        1.0,
        2 * min(
            (np.sum(distribution <= 0) + 1) / (len(distribution) + 1),
            (np.sum(distribution >= 0) + 1) / (len(distribution) + 1),
        ),
    )
    return {
        "outcome": task.upper(), "metric": metric,
        "difference": point,
        "ci_lower": float(np.percentile(distribution, 2.5)),
        "ci_upper": float(np.percentile(distribution, 97.5)),
        "paired_p_value": float(p_value),
        "bootstrap_replicates": int(len(distribution)),
        "n": int(len(labels)), "events": int(labels.sum()),
    }


def holm_adjust(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def stage_evaluate(args: argparse.Namespace) -> None:
    evaluation = args.output_root / "evaluation"
    if evaluation.exists() and any(evaluation.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Evaluation output already exists: {evaluation}. Use --overwrite intentionally."
        )
    evaluation.mkdir(parents=True, exist_ok=True)
    calibration_rows = []
    threshold_rows = []
    pooled_by_arm = {}

    for arm_index, arm in enumerate(ARMS):
        fold_frames = []
        for outer_fold in range(args.outer_splits):
            arm_directory = (
                args.output_root / "final_models" / f"outer_fold_{outer_fold}"
                / "arms" / arm
            )
            inner = pd.read_csv(
                arm_directory / "inner_oof_predictions.csv",
                dtype={args.patient_id_col: "string"},
            )
            test = pd.read_csv(
                arm_directory / "outer_test_predictions.csv",
                dtype={args.patient_id_col: "string"},
            )
            for task in (args.task,):
                label_column = f"{task}_label"
                probability_column = f"{task}_probability"
                applicable_inner = inner[label_column].notna()
                inner_labels = inner.loc[applicable_inner, label_column].astype(int).to_numpy()
                inner_probabilities = inner.loc[
                    applicable_inner, probability_column
                ].astype(float).to_numpy()
                parameters = fit_platt(
                    inner_labels, inner_probabilities,
                    args.probability_clip, args.platt_C
                )
                calibrated_inner = apply_platt(
                    inner_probabilities, parameters, args.probability_clip
                )
                selected_threshold = select_youden(inner_labels, calibrated_inner)
                calibration_rows.append(
                    {
                        "arm": arm, "outer_fold": outer_fold,
                        "outcome": task.upper(),
                        "platt_intercept": parameters["intercept"],
                        "platt_slope": parameters["slope"],
                        "inner_oof_n": len(inner_labels),
                        "inner_oof_events": int(inner_labels.sum()),
                    }
                )
                threshold_rows.append(
                    {
                        "arm": arm, "outer_fold": outer_fold,
                        "outcome": task.upper(),
                        "selection_rule": "maximize Youden J on calibrated inner OOF predictions",
                        **selected_threshold,
                    }
                )
                test[f"{task}_probability_uncalibrated"] = test[probability_column]
                test[f"{task}_probability_calibrated"] = apply_platt(
                    test[probability_column].astype(float).to_numpy(),
                    parameters, args.probability_clip
                )
                test[f"{task}_selected_threshold"] = selected_threshold["threshold"]
                test[f"{task}_predicted_class"] = (
                    test[f"{task}_probability_calibrated"]
                    >= selected_threshold["threshold"]
                ).astype(int)
            fold_frames.append(test)
        pooled = pd.concat(fold_frames, ignore_index=True)
        if not pooled[args.patient_id_col].is_unique:
            raise RuntimeError(f"Duplicate pooled patients for {arm}")
        pooled_by_arm[arm] = pooled
        arm_directory = evaluation / "arms" / arm
        arm_directory.mkdir(parents=True, exist_ok=True)
        pooled.to_csv(
            arm_directory / "pooled_predictions_calibrated_and_classified.csv",
            index=False,
        )

    pd.DataFrame(calibration_rows).to_csv(
        evaluation / "fold_specific_platt_parameters.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(
        evaluation / "fold_specific_selected_thresholds.csv", index=False
    )
    pd.concat(pooled_by_arm.values(), ignore_index=True).to_csv(
        evaluation / "all_arms_pooled_predictions_calibrated_and_classified.csv",
        index=False,
    )

    performance_rows = []
    threshold_metric_rows = []
    confusion_rows = []
    for arm_index, (arm, pooled) in enumerate(pooled_by_arm.items()):
        for task_index, task in enumerate((args.task,)):
            for probability_type, probability_column in (
                ("uncalibrated", f"{task}_probability_uncalibrated"),
                ("calibrated", f"{task}_probability_calibrated"),
            ):
                rows = bootstrap_performance(
                    pooled, task, probability_column, args.bootstrap_replicates,
                    stable_seed(args.seed, "performance", arm_index, task_index, probability_type),
                    args.probability_clip, args.platt_C
                )
                for row in rows:
                    row.update({"arm": arm, "probability_type": probability_type})
                performance_rows.extend(rows)
            rows, counts = bootstrap_threshold_metrics(
                pooled, task, args.bootstrap_replicates,
                stable_seed(args.seed, "threshold", arm_index, task_index)
            )
            for row in rows:
                row["arm"] = arm
            counts["arm"] = arm
            threshold_metric_rows.extend(rows)
            confusion_rows.append(counts)

    pd.DataFrame(performance_rows).to_csv(
        evaluation / "pooled_performance_calibration_with_95ci.csv", index=False
    )
    pd.DataFrame(threshold_metric_rows).to_csv(
        evaluation / "pooled_threshold_metrics_with_95ci.csv", index=False
    )
    pd.DataFrame(confusion_rows).to_csv(
        evaluation / "pooled_confusion_matrix_counts.csv", index=False
    )

    paired_rows = []
    for comparison_index, (name, left_arm, right_arm) in enumerate(PAIRED_COMPARISONS):
        for task_index, task in enumerate((args.task,)):
            for metric in ("roc_auc", "pr_auc", "brier_score"):
                probability_column = (
                    f"{task}_probability_uncalibrated"
                    if metric in {"roc_auc", "pr_auc"}
                    else f"{task}_probability_calibrated"
                )
                row = paired_comparison(
                    pooled_by_arm[left_arm], pooled_by_arm[right_arm], task,
                    metric, probability_column, args.bootstrap_replicates,
                    stable_seed(args.seed, "paired", comparison_index, task_index, metric)
                )
                row.update(
                    {
                        "comparison": name,
                        "model_a": left_arm,
                        "model_b": right_arm,
                        "difference_definition": "model_a minus model_b",
                        "probability_type": (
                            "uncalibrated" if metric in {"roc_auc", "pr_auc"}
                            else "calibrated"
                        ),
                    }
                )
                paired_rows.append(row)
    paired_table = pd.DataFrame(paired_rows)
    paired_table["holm_adjusted_p_value"] = holm_adjust(
        paired_table["paired_p_value"]
    )
    paired_table.to_csv(
        evaluation / "paired_text_model_comparisons_with_95ci.csv", index=False
    )
    atomic_json(
        evaluation / "evaluation_manifest.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "calibration": "fold-specific Platt fit on selected inner OOF predictions",
            "threshold": "fold-specific Youden J fit on calibrated inner OOF predictions",
            "bootstrap_unit": "patient",
            "bootstrap_replicates": args.bootstrap_replicates,
            "paired_comparisons": PAIRED_COMPARISONS,
            "task": args.task,
            "eligible_cohort": "survivors plus selected cardiac-death endpoint",
            "competing_endpoint_excluded": True,
            "independent_binary_classifier": True,
            "multiplicity_adjustment": "Holm across prespecified text comparisons/outcomes/metrics",
            "outer_test_outcomes_used_for_training_selection_calibration_or_thresholds": False,
            "decision_curve_analysis": (
                "performed post hoc from saved fold-specific calibrated outer-test predictions"
            ),
        },
    )
    print(f"Saved text evaluation outputs to: {evaluation}")


def stage_combine(args: argparse.Namespace) -> None:
    """Combine independent endpoint outputs and apply one cross-endpoint Holm family."""
    destination = args.output_root / "combined_evaluation"
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Combined output already exists: {destination}. Use --overwrite intentionally."
        )
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_files = (
        "pooled_performance_calibration_with_95ci.csv",
        "pooled_threshold_metrics_with_95ci.csv",
        "pooled_confusion_matrix_counts.csv",
        "fold_specific_platt_parameters.csv",
        "fold_specific_selected_thresholds.csv",
    )
    for filename in evaluation_files:
        parts = []
        for task in TASKS:
            path = args.output_root / "tasks" / task / "evaluation" / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing endpoint evaluation file: {path}")
            frame = pd.read_csv(path)
            frame.insert(0, "analysis_task", task.upper())
            parts.append(frame)
        pd.concat(parts, ignore_index=True, sort=False).to_csv(
            destination / filename, index=False
        )

    paired_parts = []
    selection_parts = []
    prediction_parts = []
    raw_metric_parts = []
    for task in TASKS:
        task_root = args.output_root / "tasks" / task
        paired = pd.read_csv(
            task_root / "evaluation" / "paired_text_model_comparisons_with_95ci.csv"
        )
        paired["analysis_task"] = task.upper()
        paired_parts.append(paired)

        selected = pd.read_csv(
            task_root / "selection" / "selected_primary_configuration_by_outer_fold.csv"
        )
        selected.insert(0, "outcome", task.upper())
        selection_parts.append(selected)

        predictions = pd.read_csv(
            task_root / "evaluation" / "all_arms_pooled_predictions_calibrated_and_classified.csv",
            dtype={PATIENT_ID: "string"},
        )
        predictions["analysis_task"] = task.upper()
        prediction_parts.append(predictions)

        raw = pd.read_csv(
            task_root / "pooled_outer_test" / "all_arms_raw_point_metrics.csv"
        )
        raw["analysis_task"] = task.upper()
        raw_metric_parts.append(raw)

    paired_table = pd.concat(paired_parts, ignore_index=True, sort=False)
    paired_table["holm_adjusted_p_value"] = holm_adjust(
        paired_table["paired_p_value"]
    )
    paired_table["holm_family"] = (
        "all prespecified text-arm comparisons across both endpoints and three metrics"
    )
    paired_table.to_csv(
        destination / "paired_text_model_comparisons_with_95ci.csv", index=False
    )
    pd.concat(selection_parts, ignore_index=True, sort=False).to_csv(
        destination / "selected_primary_configuration_by_outer_fold.csv", index=False
    )
    pd.concat(prediction_parts, ignore_index=True, sort=False).to_csv(
        destination / "all_arms_pooled_predictions_calibrated_and_classified.csv",
        index=False,
    )
    pd.concat(raw_metric_parts, ignore_index=True, sort=False).to_csv(
        destination / "all_arms_raw_point_metrics.csv", index=False
    )
    atomic_json(
        destination / "combined_evaluation_manifest.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "tasks": [task.upper() for task in TASKS],
            "independent_binary_classifiers": True,
            "eligible_patients": {"SCD": 648, "PFD": 659},
            "competing_endpoint_excluded_before_preprocessing": True,
            "arms": list(ARMS),
            "paired_comparisons": PAIRED_COMPARISONS,
            "multiplicity_adjustment": (
                "Holm across all prespecified text comparisons, both endpoints, and metrics"
            ),
        },
    )
    print(f"Combined endpoint-specific text results: {destination}")


def main() -> None:
    args = parse_args()
    args.folds_csv = args.folds_csv.resolve()
    args.embedding_root = args.embedding_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.stage != "combine" and args.task is None:
        raise ValueError(f"--task is required for stage {args.stage}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        stage_prepare(args)
    elif args.stage == "tune":
        stage_tune(args)
    elif args.stage == "select":
        stage_select(args)
    elif args.stage == "final":
        stage_final(args)
    elif args.stage == "aggregate":
        stage_aggregate(args)
    elif args.stage == "evaluate":
        stage_evaluate(args)
    elif args.stage == "combine":
        stage_combine(args)


if __name__ == "__main__":
    main()
