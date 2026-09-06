#!/usr/bin/env python3
"""Endpoint-specific nested multimodal fusion for the MUSIC four-year cohort.

This revision consumes the detailed-response text analysis and its mean-pooled
overlapping-chunk embeddings. The preflight rejects legacy text artifacts or
any embedding artifact that indicates silent truncation.

The driver evaluates three modality pairs:

* ECG + full LLM text without ECG impressions;
* ECG + deterministic-template text without ECG impressions; and
* ECG + prompt-matched structured variables without ECG impressions.

For every pair it retains five prespecified fusion arms (direct concatenation,
projected concatenation, patient-specific scalar gating, patient-specific
vector gating, and global weighted sum) plus an overall ``selected_fusion``
arm. SCD and PFD are fit as independent one-head binary models. For each task,
the competing cardiac-death endpoint is excluded before fitting, calibration,
threshold selection, and evaluation. Hyperparameters and the overall fusion
method are selected only within the corresponding outer-training cohort.

The ECG inputs must be the fold-specific embeddings exported by the revised
ECG nested-CV pipeline. Frozen text embeddings are selected separately for
each outer fold using the revised text pipeline's saved selection. Tabular
preprocessing is fitted from scratch inside every training split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm


PATIENT_ID = "Patient ID"
SCD_LABEL = "SCD_4year_label"
PFD_LABEL = "PFD_4year_label"
OUTER_FOLD = "outer_fold"
TASKS = ("scd", "pfd")
FUSION_METHODS = (
    "concatenation",
    "projected_concatenation",
    "scalar_gating",
    "vector_gating",
    "weighted_sum",
)
MODALITY_PAIRS = (
    "ecg_full_text",
    "ecg_deterministic_text",
    "ecg_tabular",
)
TEXT_CONDITIONS = {
    "ecg_full_text": ("selected", "endpoint_specific"),
    "ecg_deterministic_text": (
        "DeterministicTemplate",
        "patient_data_template_no_ecg",
    ),
}
ANTICOAGULANT_FEATURE = "Anticoagulants/antitrombotics  (yes=1)"

PROMPT_CONTINUOUS_FEATURES = [
    "Age",
    "Weight (kg)",
    "Height (cm)",
    "Diastolic blood  pressure (mmHg)",
    "Systolic blood pressure (mmHg)",
    "Albumin (g/L)",
    "ALT or GPT (IU/L)",
    "AST or GOT (IU/L)",
    "Total Cholesterol (mmol/L)",
    "Creatinine (?mol/L)",
    "Gamma-glutamil transpeptidase (IU/L)",
    "Glucose (mmol/L)",
    "Hemoglobin (g/L)",
    "HDL (mmol/L)",
    "Potassium (mEq/L)",
    "LDL (mmol/L)",
    "Sodium (mEq/L)",
    "Pro-BNP (ng/L)",
    "Protein (g/L)",
    "T3 (pg/dL)",
    "T4 (ng/L)",
    "Troponin (ng/mL)",
    "TSH (mIU/L)",
    "Urea (mg/dL)",
    "LVEF (%)",
]

PROMPT_CATEGORICAL_FEATURES = [
    "Gender (male=1)",
    "NYHA class",
    "HF etiology - Diagnosis",
    "Diabetes (yes=1)",
    "History of dyslipemia (yes=1)",
    "Peripheral vascular disease (yes=1)",
    "History of hypertension (yes=1)",
    "Prior Myocardial Infarction (yes=1)",
    "Calcium channel blocker (yes=1)",
    "Diabetes medication (yes=1)",
    "Amiodarone (yes=1)",
    "Angiotensin-II receptor blocker (yes=1)",
    ANTICOAGULANT_FEATURE,
    "Betablockers (yes=1)",
    "Digoxin (yes=1)",
    "Loop diuretics (yes=1)",
    "Spironolactone (yes=1)",
    "Statins (yes=1)",
    "Hidralazina (yes=1)",
    "ACE inhibitor (yes=1)",
    "Nitrovasodilator (yes=1)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested-CV ECG/text/tabular multimodal fusion."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "tune", "select", "final", "aggregate", "evaluate"],
    )
    parser.add_argument("--folds_csv", type=Path, required=True)
    parser.add_argument("--ecg_root", type=Path, required=True)
    parser.add_argument("--text_embedding_root", type=Path, required=True)
    parser.add_argument("--text_results_root", type=Path, required=True)
    parser.add_argument("--tabular_csv", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--patient_id_col", default=PATIENT_ID)
    parser.add_argument("--scd_label_col", default=SCD_LABEL)
    parser.add_argument("--pfd_label_col", default=PFD_LABEL)
    parser.add_argument("--outer_fold_col", default=OUTER_FOLD)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer_fold", type=int)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--modality_pair", choices=MODALITY_PAIRS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--auc_tolerance", type=float, default=0.005)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--platt_C", type=float, default=1e6)
    parser.add_argument("--probability_clip", type=float, default=1e-6)
    parser.add_argument("--expected_patients", type=int, default=730)
    parser.add_argument("--expected_controls", type=int, default=577)
    parser.add_argument("--expected_scd", type=int, default=71)
    parser.add_argument("--expected_pfd", type=int, default=82)
    parser.add_argument("--expected_anticoagulant_yes", type=int, default=610)
    parser.add_argument("--expected_anticoagulant_no", type=int, default=120)
    parser.add_argument("--expected_text_pooling", default="cls")
    parser.add_argument("--expected_text_max_length", type=int, default=512)
    parser.add_argument(
        "--expected_text_long_strategy",
        choices=["mean_chunks", "truncate"],
        default="mean_chunks",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def task_root(args: argparse.Namespace, task: str) -> Path:
    return args.output_root / "tasks" / task


def task_label_column(args: argparse.Namespace, task: str) -> str:
    return args.scd_label_col if task == "scd" else args.pfd_label_col


def competing_label_column(args: argparse.Namespace, task: str) -> str:
    return args.pfd_label_col if task == "scd" else args.scd_label_col


def expected_task_patients(args: argparse.Namespace, task: str) -> int:
    return args.expected_controls + (
        args.expected_scd if task == "scd" else args.expected_pfd
    )


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
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(value, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    ).strip("_")


def canonical_patient_id(value: object) -> str:
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    return str(int(digits)) if digits else text.casefold()


def candidate_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Two prespecified capacity profiles for each of five methods."""
    profiles = (
        {
            "profile": "compact",
            "proj_dim": 128,
            "hidden_dim": 128,
            "layers": 1,
            "dropout": 0.20,
            "lr": 1e-3,
            "weight_decay": 1e-5,
        },
        {
            "profile": "expanded",
            "proj_dim": 256,
            "hidden_dim": 256,
            "layers": 2,
            "dropout": 0.30,
            "lr": 3e-4,
            "weight_decay": 1e-5,
        },
    )
    candidates = []
    for method in FUSION_METHODS:
        for profile in profiles:
            config = {
                "fusion_method": method,
                **profile,
                "epochs": args.epochs,
                "min_epochs": args.min_epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
            }
            config["config_name"] = f"{method}__{profile['profile']}"
            candidates.append(config)
    return candidates


def read_folds(args: argparse.Namespace, setup_copy: bool = True) -> pd.DataFrame:
    path = args.folds_csv
    if args.stage != "prepare":
        copied = args.output_root / "analysis_setup" / "nested_patient_folds.csv"
        if copied.exists():
            path = copied
    frame = pd.read_csv(path, dtype={args.patient_id_col: "string"})
    required = {
        args.patient_id_col,
        args.scd_label_col,
        args.pfd_label_col,
        args.outer_fold_col,
        *[f"inner_fold_outer_{fold}" for fold in range(args.outer_splits)],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Nested folds CSV is missing columns: {sorted(missing)}")
    if frame[args.patient_id_col].isna().any() or not frame[args.patient_id_col].is_unique:
        raise ValueError("Patient IDs in the folds CSV must be complete and unique.")
    frame[args.patient_id_col] = frame[args.patient_id_col].astype(str)
    frame["__patient_key"] = frame[args.patient_id_col].map(canonical_patient_id)
    if not frame["__patient_key"].is_unique:
        raise ValueError("Canonical patient IDs are not unique in the folds CSV.")
    frame[args.outer_fold_col] = pd.to_numeric(
        frame[args.outer_fold_col], errors="raise"
    ).astype(int)
    scd = pd.to_numeric(frame[args.scd_label_col], errors="coerce")
    pfd = pd.to_numeric(frame[args.pfd_label_col], errors="coerce")
    controls = int(((scd == 0) & (pfd == 0)).sum())
    observed = (len(frame), controls, int((scd == 1).sum()), int((pfd == 1).sum()))
    expected = (
        args.expected_patients,
        args.expected_controls,
        args.expected_scd,
        args.expected_pfd,
    )
    if observed != expected:
        raise ValueError(f"Unexpected cohort counts {observed}; expected {expected}.")
    if ((scd == 1) & (pfd == 1)).any():
        raise ValueError("At least one patient is labeled as both SCD and PFD.")
    if sorted(frame[args.outer_fold_col].unique()) != list(range(args.outer_splits)):
        raise ValueError("Outer folds are incomplete or unexpected.")
    for outer_fold in range(args.outer_splits):
        column = f"inner_fold_outer_{outer_fold}"
        train = frame[frame[args.outer_fold_col].ne(outer_fold)]
        test = frame[frame[args.outer_fold_col].eq(outer_fold)]
        if train[column].isna().any() or test[column].notna().any():
            raise ValueError(f"Invalid nested assignments in {column}.")
        if sorted(train[column].astype(int).unique()) != list(range(args.inner_splits)):
            raise ValueError(f"Incomplete inner folds in {column}.")
    if setup_copy:
        destination = args.output_root / "analysis_setup" / "nested_patient_folds.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.stage == "prepare":
            frame.drop(columns="__patient_key").to_csv(destination, index=False)
    return frame


def labels_for_ids(
    frame: pd.DataFrame, ids: np.ndarray, args: argparse.Namespace, task: str
) -> np.ndarray:
    by_key = frame.set_index("__patient_key")
    keys = [canonical_patient_id(value) for value in ids.astype(str)]
    missing = [key for key in keys if key not in by_key.index]
    if missing:
        raise ValueError(f"Embedding bundle contains unknown patients: {missing[:10]}")
    subset = by_key.loc[keys]
    labels = pd.to_numeric(
        subset[task_label_column(args, task)], errors="coerce"
    ).to_numpy(float)
    competing = pd.to_numeric(
        subset[competing_label_column(args, task)], errors="coerce"
    ).to_numpy(float)
    if np.any(competing == 1):
        raise ValueError(
            f"Competing-endpoint patients reached the {task.upper()} model input."
        )
    if not np.isfinite(labels).all() or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError(f"Invalid {task.upper()} labels after endpoint filtering.")
    return labels.astype(np.float32)


def filter_ids_for_task(
    frame: pd.DataFrame,
    ids: np.ndarray,
    values: np.ndarray,
    args: argparse.Namespace,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    by_key = frame.set_index("__patient_key")
    keys = [canonical_patient_id(value) for value in ids.astype(str)]
    competing = pd.to_numeric(
        by_key.loc[keys, competing_label_column(args, task)], errors="coerce"
    ).to_numpy(float)
    keep = competing != 1
    filtered_ids = ids[keep]
    filtered_values = values[keep]
    labels_for_ids(frame, filtered_ids, args, task)
    return filtered_ids, filtered_values


def ecg_bundle_path(
    args: argparse.Namespace,
    outer_fold: int,
    inner_fold: int | None,
    split: str,
) -> Path:
    base = args.ecg_root / "final_models" / f"outer_fold_{outer_fold}"
    if inner_fold is None:
        return base / f"outer_{split}_embeddings.npz"
    return (
        base
        / "selected_inner_folds"
        / f"inner_fold_{inner_fold}"
        / f"{split}_embeddings.npz"
    )


def load_ecg_bundle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ECG embedding bundle: {path}. The revised ECG final stage must "
            "export selected inner-fold train/validation embeddings and outer train/test "
            "embeddings before multimodal fusion."
        )
    artifact = np.load(path, allow_pickle=False)
    required = {"patient_ids", "shared_embeddings"}
    missing = required - set(artifact.files)
    if missing:
        raise ValueError(f"{path} lacks arrays: {sorted(missing)}")
    patient_ids = artifact["patient_ids"].astype(str)
    embeddings = artifact["shared_embeddings"].astype(np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(patient_ids):
        raise ValueError(f"Invalid ECG embedding shape in {path}: {embeddings.shape}")
    if len(set(map(canonical_patient_id, patient_ids))) != len(patient_ids):
        raise ValueError(f"Duplicate patient IDs in {path}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Nonfinite ECG embeddings in {path}")
    return patient_ids, embeddings


def selected_text_spec(
    args: argparse.Namespace, outer_fold: int, pair: str, task: str
) -> tuple[str, str, str]:
    selection_path = (
        args.text_results_root
        / "tasks"
        / task
        / "selection"
        / "selected_primary_configuration_by_outer_fold.json"
    )
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing text selection file: {selection_path}")
    selected = json.loads(selection_path.read_text(encoding="utf-8"))[
        "selected_by_outer_fold"
    ][str(outer_fold)]
    config = selected.get("config", selected)
    encoder = config["encoder_name"]
    source_spec, condition_spec = TEXT_CONDITIONS[pair]
    source = config["source_name"] if source_spec == "selected" else source_spec
    condition = (
        f"full_risk_no_ecg_{task}_full"
        if condition_spec == "endpoint_specific"
        else condition_spec
    )
    return source, encoder, condition


def load_text_matrix(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    outer_fold: int,
    pair: str,
    task: str,
) -> tuple[np.ndarray, dict]:
    source, encoder, condition = selected_text_spec(args, outer_fold, pair, task)
    directory = (
        args.text_embedding_root
        / stable_slug(source)
        / stable_slug(encoder)
        / stable_slug(condition)
    )
    path = directory / "embeddings.npz"
    manifest_path = directory / "manifest.json"
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing text embedding artifact: {path}")
    artifact = np.load(path, allow_pickle=False)
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
        raise ValueError(f"{path} lacks arrays: {sorted(missing)}")
    ids = artifact["patient_ids"].astype(str)
    values = artifact["embeddings"].astype(np.float32)
    text_sha256 = artifact["text_sha256"].astype(str)
    token_lengths = artifact["token_lengths"].astype(np.int64)
    chunk_counts = artifact["chunk_counts"].astype(np.int64)
    was_chunked = artifact["was_chunked"].astype(bool)
    was_truncated = artifact["was_truncated"].astype(bool)
    if values.ndim != 2 or len(values) != len(ids) or not np.isfinite(values).all():
        raise ValueError(f"Invalid text embeddings in {path}")
    row_count = len(ids)
    for array_name, array in {
        "text_sha256": text_sha256,
        "token_lengths": token_lengths,
        "chunk_counts": chunk_counts,
        "was_chunked": was_chunked,
        "was_truncated": was_truncated,
    }.items():
        if array.ndim != 1 or len(array) != row_count:
            raise ValueError(
                f"{path}: {array_name} must contain one value per patient."
            )
    if np.any(token_lengths <= 0) or np.any(chunk_counts <= 0):
        raise ValueError(f"Invalid text token or chunk counts in {path}")
    if not np.array_equal(was_chunked, chunk_counts > 1):
        raise ValueError(f"Chunk indicators disagree with chunk counts in {path}")
    rows = {canonical_patient_id(value): index for index, value in enumerate(ids)}
    if len(rows) != len(ids):
        raise ValueError(f"Duplicate patient IDs in {path}")
    requested = frame["__patient_key"].tolist()
    missing_ids = [key for key in requested if key not in rows]
    if missing_ids:
        raise ValueError(f"Text artifact is missing patients: {missing_ids[:10]}")
    aligned = values[[rows[key] for key in requested]]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_scope = task if pair == "ecg_full_text" else "shared"
    observed_scope = manifest.get("endpoint_scope")
    if observed_scope != expected_scope:
        raise ValueError(
            f"Unexpected endpoint_scope={observed_scope!r} for {path}; "
            f"expected {expected_scope!r}."
        )
    expected_manifest = {
        "artifact_type": "frozen_text_embeddings",
        "source_name": source,
        "encoder_name": encoder,
        "condition": condition,
        "patient_count": row_count,
        "encoder_frozen": True,
        "pooling": args.expected_text_pooling,
        "max_length": args.expected_text_max_length,
        "long_text_strategy": args.expected_text_long_strategy,
        "chunked_patient_count": int(was_chunked.sum()),
        "maximum_chunks_per_patient": int(chunk_counts.max()),
        "maximum_untruncated_token_length": int(token_lengths.max()),
        "truncated_patient_count": int(was_truncated.sum()),
    }
    for key, expected_value in expected_manifest.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"{manifest_path}: expected {key}={expected_value!r}; "
                f"observed {manifest.get(key)!r}."
            )
    if args.expected_text_long_strategy == "mean_chunks" and was_truncated.any():
        raise ValueError(f"Detailed text was unexpectedly truncated in {path}")
    if manifest.get("embedding_shape") != list(values.shape):
        raise ValueError(f"Embedding shape disagrees with manifest: {path}")
    if not isinstance(manifest.get("cache_signature"), dict):
        raise ValueError(f"Missing cache signature in {manifest_path}")
    return aligned, {
        "task": task,
        "source": source,
        "encoder": encoder,
        "condition": condition,
        "embedding_path": str(path.resolve()),
        "embedding_sha256": sha256_file(path),
        "manifest": manifest,
    }


def read_tabular(args: argparse.Namespace, folds: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(
        args.tabular_csv,
        sep=";",
        decimal=",",
        engine="python",
        na_values=["", "NA", "N/A", "nan"],
        keep_default_na=True,
    ).replace(r"\t", ".", regex=True)
    if args.patient_id_col not in source:
        raise ValueError(f"Missing {args.patient_id_col} in {args.tabular_csv}")
    required = [*PROMPT_CONTINUOUS_FEATURES, *PROMPT_CATEGORICAL_FEATURES]
    missing = set(required) - set(source.columns)
    if missing:
        raise ValueError(f"Tabular CSV is missing features: {sorted(missing)}")
    source = source[[args.patient_id_col, *required]].copy()
    source["__patient_key"] = source[args.patient_id_col].map(canonical_patient_id)
    if not source["__patient_key"].is_unique:
        raise ValueError("Canonical IDs are not unique in the tabular CSV.")
    result = folds[["__patient_key"]].merge(
        source.drop(columns=args.patient_id_col),
        on="__patient_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if result["_merge"].ne("both").any():
        raise ValueError("At least one locked patient is absent from the tabular CSV.")
    result = result.drop(columns="_merge")
    for feature in PROMPT_CONTINUOUS_FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
    for feature in PROMPT_CATEGORICAL_FEATURES:
        result[feature] = result[feature].map(
            lambda value: np.nan if pd.isna(value) else str(value).strip()
        )
    anticoagulant = pd.to_numeric(result[ANTICOAGULANT_FEATURE], errors="coerce")
    observed = {
        "yes": int(anticoagulant.eq(1).sum()),
        "no": int(anticoagulant.eq(0).sum()),
        "missing": int(anticoagulant.isna().sum()),
    }
    expected = {
        "yes": args.expected_anticoagulant_yes,
        "no": args.expected_anticoagulant_no,
        "missing": 0,
    }
    if observed != expected:
        raise ValueError(f"Anticoagulant audit failed: {observed}; expected {expected}.")
    return result


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_tabular_preprocessor() -> ColumnTransformer:
    continuous = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("one_hot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", continuous, PROMPT_CONTINUOUS_FEATURES),
            ("categorical", categorical, PROMPT_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = values.std(axis=0, dtype=np.float64).astype(np.float32)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    return mean, sd


def standardize(values: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    result = ((values - mean) / sd).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite standardized embeddings.")
    return result


def align_matrix(ids: np.ndarray, values: np.ndarray, requested_ids: np.ndarray) -> np.ndarray:
    rows = {canonical_patient_id(value): index for index, value in enumerate(ids.astype(str))}
    keys = [canonical_patient_id(value) for value in requested_ids.astype(str)]
    missing = [key for key in keys if key not in rows]
    if missing:
        raise ValueError(f"Embedding bundle is missing patients: {missing[:10]}")
    return values[[rows[key] for key in keys]]


def expected_split_keys(
    folds: pd.DataFrame,
    args: argparse.Namespace,
    outer_fold: int,
    inner_fold: int | None,
    split: str,
) -> set[str]:
    outer_train = folds[args.outer_fold_col].ne(outer_fold)
    if inner_fold is None:
        mask = outer_train if split == "train" else ~outer_train
    else:
        inner_column = f"inner_fold_outer_{outer_fold}"
        mask = outer_train & (
            folds[inner_column].ne(inner_fold)
            if split == "train"
            else folds[inner_column].eq(inner_fold)
        )
    return set(folds.loc[mask, "__patient_key"].astype(str))


def assert_expected_ecg_patients(
    ids: np.ndarray,
    folds: pd.DataFrame,
    args: argparse.Namespace,
    outer_fold: int,
    inner_fold: int | None,
    split: str,
) -> None:
    observed = {canonical_patient_id(value) for value in ids.astype(str)}
    expected = expected_split_keys(folds, args, outer_fold, inner_fold, split)
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        unexpected = sorted(observed - expected)[:10]
        raise ValueError(
            "ECG embedding patient set does not match the locked nested split: "
            f"outer={outer_fold}, inner={inner_fold}, split={split}, "
            f"missing={missing}, unexpected={unexpected}."
        )


@dataclass
class PairData:
    patient_ids_train: np.ndarray
    patient_ids_validation: np.ndarray
    ecg_train: np.ndarray
    ecg_validation: np.ndarray
    second_train: np.ndarray
    second_validation: np.ndarray
    labels_train: np.ndarray
    labels_validation: np.ndarray
    preprocessing: dict[str, Any]


def load_pair_split(
    args: argparse.Namespace,
    folds: pd.DataFrame,
    pair: str,
    task: str,
    outer_fold: int,
    inner_fold: int | None,
    tabular: pd.DataFrame | None,
) -> PairData:
    if inner_fold is None:
        train_name, validation_name = "train", "test"
    else:
        train_name, validation_name = "train", "validation"
    train_ids, ecg_train_raw = load_ecg_bundle(
        ecg_bundle_path(args, outer_fold, inner_fold, train_name)
    )
    validation_ids, ecg_validation_raw = load_ecg_bundle(
        ecg_bundle_path(args, outer_fold, inner_fold, validation_name)
    )
    assert_expected_ecg_patients(
        train_ids, folds, args, outer_fold, inner_fold, "train"
    )
    assert_expected_ecg_patients(
        validation_ids,
        folds,
        args,
        outer_fold,
        inner_fold,
        "test" if inner_fold is None else "validation",
    )
    train_ids, ecg_train_raw = filter_ids_for_task(
        folds, train_ids, ecg_train_raw, args, task
    )
    validation_ids, ecg_validation_raw = filter_ids_for_task(
        folds, validation_ids, ecg_validation_raw, args, task
    )
    ecg_mean, ecg_sd = fit_standardizer(ecg_train_raw)
    ecg_train = standardize(ecg_train_raw, ecg_mean, ecg_sd)
    ecg_validation = standardize(ecg_validation_raw, ecg_mean, ecg_sd)
    preprocessing: dict[str, Any] = {"ecg_mean": ecg_mean, "ecg_sd": ecg_sd}

    if pair in TEXT_CONDITIONS:
        full_text, provenance = load_text_matrix(
            args, folds, outer_fold, pair, task
        )
        full_ids = folds[args.patient_id_col].astype(str).to_numpy()
        second_train_raw = align_matrix(full_ids, full_text, train_ids)
        second_validation_raw = align_matrix(full_ids, full_text, validation_ids)
        second_mean, second_sd = fit_standardizer(second_train_raw)
        second_train = standardize(second_train_raw, second_mean, second_sd)
        second_validation = standardize(second_validation_raw, second_mean, second_sd)
        preprocessing.update(
            {
                "second_mean": second_mean,
                "second_sd": second_sd,
                "text_provenance": provenance,
            }
        )
    else:
        if tabular is None:
            raise ValueError("Tabular data were not loaded.")
        tabular_by_key = tabular.set_index("__patient_key")
        train_keys = [canonical_patient_id(value) for value in train_ids]
        validation_keys = [canonical_patient_id(value) for value in validation_ids]
        train_frame = tabular_by_key.loc[train_keys]
        validation_frame = tabular_by_key.loc[validation_keys]
        preprocessor = build_tabular_preprocessor()
        second_train = np.asarray(preprocessor.fit_transform(train_frame), dtype=np.float32)
        second_validation = np.asarray(
            preprocessor.transform(validation_frame), dtype=np.float32
        )
        if not np.isfinite(second_train).all() or not np.isfinite(second_validation).all():
            raise ValueError("Nonfinite transformed tabular features.")
        preprocessing["tabular_preprocessor"] = preprocessor

    labels_train = labels_for_ids(folds, train_ids, args, task)
    labels_validation = labels_for_ids(folds, validation_ids, args, task)
    return PairData(
        train_ids,
        validation_ids,
        ecg_train,
        ecg_validation,
        second_train,
        second_validation,
        labels_train,
        labels_validation,
        preprocessing,
    )


class FusionNetwork(nn.Module):
    def __init__(self, ecg_dim: int, second_dim: int, config: dict[str, Any]):
        super().__init__()
        self.method = config["fusion_method"]
        proj_dim = int(config["proj_dim"])
        if self.method == "concatenation":
            self.ecg_projection = nn.LayerNorm(ecg_dim)
            self.second_projection = nn.LayerNorm(second_dim)
            fusion_dim = ecg_dim + second_dim
        else:
            self.ecg_projection = nn.Sequential(
                nn.LayerNorm(ecg_dim), nn.Linear(ecg_dim, proj_dim), nn.ReLU()
            )
            self.second_projection = nn.Sequential(
                nn.LayerNorm(second_dim), nn.Linear(second_dim, proj_dim), nn.ReLU()
            )
            fusion_dim = 2 * proj_dim if self.method == "projected_concatenation" else proj_dim
        if self.method == "weighted_sum":
            self.global_alpha = nn.Parameter(torch.tensor(0.0))
        elif self.method == "scalar_gating":
            self.gate = nn.Sequential(
                nn.LayerNorm(2 * proj_dim), nn.Linear(2 * proj_dim, 1)
            )
        elif self.method == "vector_gating":
            self.gate = nn.Sequential(
                nn.LayerNorm(2 * proj_dim), nn.Linear(2 * proj_dim, proj_dim)
            )
        blocks: list[nn.Module] = [nn.LayerNorm(fusion_dim)]
        current = fusion_dim
        for _ in range(int(config["layers"])):
            blocks.extend(
                [
                    nn.Linear(current, int(config["hidden_dim"])),
                    nn.ReLU(),
                    nn.Dropout(float(config["dropout"])),
                ]
            )
            current = int(config["hidden_dim"])
        self.trunk = nn.Sequential(*blocks)
        self.endpoint_head = nn.Linear(current, 1)

    def forward(
        self, ecg: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h_ecg = self.ecg_projection(ecg)
        h_second = self.second_projection(second)
        gate = None
        if self.method in ("concatenation", "projected_concatenation"):
            fused = torch.cat([h_ecg, h_second], dim=1)
        elif self.method == "weighted_sum":
            gate = torch.sigmoid(self.global_alpha)
            fused = gate * h_ecg + (1.0 - gate) * h_second
        else:
            gate_input = torch.cat([h_ecg, h_second], dim=1)
            gate = torch.sigmoid(self.gate(gate_input))
            fused = gate * h_ecg + (1.0 - gate) * h_second
        representation = self.trunk(fused)
        return self.endpoint_head(representation).squeeze(-1), gate


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def task_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        raise ValueError("A training split lacks one class for the active endpoint.")
    return torch.tensor(negative / positive, dtype=torch.float32, device=device)


def endpoint_loss(
    logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return nn.BCEWithLogitsLoss(pos_weight=weight)(logits, labels)


def endpoint_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(labels.astype(int), probabilities))


def predict_model(
    model: nn.Module,
    ecg: np.ndarray,
    second: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    probabilities = []
    gate_values = []
    with torch.no_grad():
        for start in range(0, len(ecg), batch_size):
            stop = min(start + batch_size, len(ecg))
            x1 = torch.as_tensor(ecg[start:stop], dtype=torch.float32, device=device)
            x2 = torch.as_tensor(second[start:stop], dtype=torch.float32, device=device)
            logits, gate = model(x1, x2)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            if gate is not None:
                if gate.ndim == 0:
                    gate_values.append(np.repeat(float(gate.item()), stop - start))
                else:
                    gate_values.append(gate.detach().cpu().numpy().reshape(stop - start, -1))
    gate_stats: dict[str, float] = {}
    if gate_values:
        flat = np.concatenate([np.asarray(value).reshape(-1) for value in gate_values])
        gate_stats = {
            "gate_mean": float(np.mean(flat)),
            "gate_sd": float(np.std(flat)),
            "gate_min": float(np.min(flat)),
            "gate_max": float(np.max(flat)),
        }
    return np.concatenate(probabilities), gate_stats


@dataclass
class FitResult:
    model: FusionNetwork
    best_epoch: int
    probabilities: np.ndarray
    endpoint_auc: float
    parameter_count: int
    gate_stats: dict[str, float]


def fit_with_validation(
    data: PairData,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    progress_label: str,
) -> FitResult:
    set_seed(seed)
    model = FusionNetwork(data.ecg_train.shape[1], data.second_train.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    weight = task_pos_weight(data.labels_train, device)
    x1 = torch.as_tensor(data.ecg_train, dtype=torch.float32)
    x2 = torch.as_tensor(data.second_train, dtype=torch.float32)
    y = torch.as_tensor(data.labels_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x1, x2, y)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(int(config["batch_size"]), len(dataset)),
        shuffle=True,
        generator=generator,
    )
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    iterator = tqdm(
        range(1, int(config["epochs"]) + 1),
        desc=progress_label,
        leave=False,
        dynamic_ncols=True,
    )
    for epoch in iterator:
        model.train()
        for batch_ecg, batch_second, batch_labels in loader:
            batch_ecg = batch_ecg.to(device)
            batch_second = batch_second.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_ecg, batch_second)
            loss = endpoint_loss(logits, batch_labels, weight)
            loss.backward()
            optimizer.step()
        probabilities, _ = predict_model(
            model,
            data.ecg_validation,
            data.second_validation,
            device,
            int(config["batch_size"]),
        )
        validation_auc = endpoint_auc(data.labels_validation, probabilities)
        iterator.set_postfix(roc_auc=f"{validation_auc:.3f}", best=f"{best_auc:.3f}")
        if validation_auc > best_auc + 1e-8:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            epoch >= int(config["min_epochs"])
            and epochs_without_improvement >= int(config["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    probabilities, gate_stats = predict_model(
        model,
        data.ecg_validation,
        data.second_validation,
        device,
        int(config["batch_size"]),
    )
    validation_auc = endpoint_auc(data.labels_validation, probabilities)
    return FitResult(
        model,
        best_epoch,
        probabilities,
        validation_auc,
        parameter_count(model),
        gate_stats,
    )


def fit_fixed_epochs(
    data: PairData,
    config: dict[str, Any],
    epochs: int,
    device: torch.device,
    seed: int,
    progress_label: str,
) -> FusionNetwork:
    set_seed(seed)
    model = FusionNetwork(data.ecg_train.shape[1], data.second_train.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    weight = task_pos_weight(data.labels_train, device)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(data.ecg_train, dtype=torch.float32),
        torch.as_tensor(data.second_train, dtype=torch.float32),
        torch.as_tensor(data.labels_train, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(int(config["batch_size"]), len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in tqdm(range(epochs), desc=progress_label, leave=False, dynamic_ncols=True):
        model.train()
        for batch_ecg, batch_second, batch_labels in loader:
            batch_ecg = batch_ecg.to(device)
            batch_second = batch_second.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_ecg, batch_second)
            loss = endpoint_loss(logits, batch_labels, weight)
            loss.backward()
            optimizer.step()
    return model


def predictions_frame(
    ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    outer_fold: int,
    inner_fold: int | None,
    pair: str,
    arm: str,
    task: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            PATIENT_ID: ids.astype(str),
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "modality_pair": pair,
            "arm": arm,
            "task": task,
            "y": labels,
            "prob": probabilities,
        }
    )


def pooled_selection_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y"].astype(int).to_numpy()
    p = frame["prob"].astype(float).to_numpy()
    roc_auc = float(roc_auc_score(y, p))
    pr_auc = float(average_precision_score(y, p))
    return {
        "endpoint_roc_auc": roc_auc,
        "endpoint_pr_auc": pr_auc,
        "selection_score": roc_auc,
        "secondary_pr_auc": pr_auc,
    }


def require_task_pair_fold(args: argparse.Namespace) -> tuple[str, str, int]:
    if args.task is None or args.modality_pair is None or args.outer_fold is None:
        raise ValueError(
            "This stage requires --task, --modality_pair, and --outer_fold."
        )
    return args.task, args.modality_pair, args.outer_fold


def stage_prepare(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    folds = read_folds(args)
    tabular = read_tabular(args, folds)
    text_analysis_manifests = {}
    expected_text_policy = {
        "pooling": args.expected_text_pooling,
        "max_length": args.expected_text_max_length,
        "long_text_strategy": args.expected_text_long_strategy,
        "truncated_patient_count": 0,
    }
    for task in TASKS:
        path = (
            args.text_results_root
            / "tasks"
            / task
            / "analysis_setup"
            / "prepare_manifest.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing text-analysis preparation manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("completed") is not True or manifest.get("task") != task:
            raise ValueError(f"Incomplete or mismatched text-analysis manifest: {path}")
        if manifest.get("expected_embedding_policy") != expected_text_policy:
            raise ValueError(
                f"{path} does not describe the required detailed-response embeddings."
            )
        if Path(manifest.get("embedding_root", "")).resolve() != args.text_embedding_root:
            raise ValueError(f"Text-analysis embedding root disagrees with launcher: {path}")
        text_analysis_manifests[task] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    audit_rows = []
    for outer_fold in tqdm(range(args.outer_splits), desc="Preflight outer folds"):
        for inner_fold in range(args.inner_splits):
            for split in ("train", "validation"):
                path = ecg_bundle_path(args, outer_fold, inner_fold, split)
                ids, embeddings = load_ecg_bundle(path)
                assert_expected_ecg_patients(
                    ids, folds, args, outer_fold, inner_fold, split
                )
                for task in TASKS:
                    eligible_ids, _ = filter_ids_for_task(
                        folds, ids, embeddings, args, task
                    )
                    audit_rows.append(
                        {
                            "task": task,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "split": split,
                            "path": str(path.resolve()),
                            "patients": len(eligible_ids),
                            "dimension": embeddings.shape[1],
                        }
                    )
        for split in ("train", "test"):
            path = ecg_bundle_path(args, outer_fold, None, split)
            ids, embeddings = load_ecg_bundle(path)
            assert_expected_ecg_patients(
                ids, folds, args, outer_fold, None, split
            )
            for task in TASKS:
                eligible_ids, _ = filter_ids_for_task(
                    folds, ids, embeddings, args, task
                )
                audit_rows.append(
                    {
                        "task": task,
                        "outer_fold": outer_fold,
                        "inner_fold": None,
                        "split": f"outer_{split}",
                        "path": str(path.resolve()),
                        "patients": len(eligible_ids),
                        "dimension": embeddings.shape[1],
                    }
                )
        for task in TASKS:
            for pair in TEXT_CONDITIONS:
                matrix, provenance = load_text_matrix(
                    args, folds, outer_fold, pair, task
                )
                audit_rows.append(
                    {
                        "task": task,
                        "outer_fold": outer_fold,
                        "inner_fold": None,
                        "split": pair,
                        "path": provenance["embedding_path"],
                        "patients": len(matrix),
                        "dimension": matrix.shape[1],
                        "endpoint_scope": provenance["manifest"].get("endpoint_scope"),
                        "long_text_strategy": provenance["manifest"].get(
                            "long_text_strategy"
                        ),
                        "chunked_patient_count": provenance["manifest"].get(
                            "chunked_patient_count"
                        ),
                        "maximum_chunks_per_patient": provenance["manifest"].get(
                            "maximum_chunks_per_patient"
                        ),
                        "truncated_patient_count": provenance["manifest"].get(
                            "truncated_patient_count"
                        ),
                    }
                )
    setup = args.output_root / "analysis_setup"
    atomic_csv(setup / "input_embedding_audit.csv", pd.DataFrame(audit_rows))
    atomic_csv(
        setup / "anticoagulant_feature_audit.csv",
        pd.DataFrame(
            [
                {
                    "feature": ANTICOAGULANT_FEATURE,
                    "yes": args.expected_anticoagulant_yes,
                    "no": args.expected_anticoagulant_no,
                    "missing": 0,
                }
            ]
        ),
    )
    grid = candidate_grid(args)
    atomic_json(setup / "candidate_grid.json", grid)
    atomic_json(
        setup / "analysis_manifest.json",
        {
            "created_at_utc": utc_now(),
            "completed": True,
            "folds_csv": str(args.folds_csv.resolve()),
            "folds_sha256": sha256_file(args.folds_csv),
            "tabular_csv": str(args.tabular_csv.resolve()),
            "tabular_csv_sha256": sha256_file(args.tabular_csv),
            "ecg_root": str(args.ecg_root.resolve()),
            "text_embedding_root": str(args.text_embedding_root.resolve()),
            "text_results_root": str(args.text_results_root.resolve()),
            "text_analysis_manifests": text_analysis_manifests,
            "expected_text_embedding_policy": expected_text_policy,
            "output_root": str(args.output_root.resolve()),
            "patients": args.expected_patients,
            "controls": args.expected_controls,
            "scd": args.expected_scd,
            "pfd": args.expected_pfd,
            "outer_folds": args.outer_splits,
            "inner_folds": args.inner_splits,
            "seed": args.seed,
            "fusion_methods": FUSION_METHODS,
            "modality_pairs": MODALITY_PAIRS,
            "candidate_count_per_pair": len(grid),
            "independent_binary_tasks": True,
            "task_eligible_patients": {
                task: expected_task_patients(args, task) for task in TASKS
            },
            "competing_endpoint_policy": (
                "Exclude PFD from SCD models and exclude SCD from PFD models before "
                "training, selection, calibration, thresholding, and evaluation."
            ),
            "multitask_training": False,
            "scalar_gating_definition": "patient-specific scalar gate",
            "weighted_sum_definition": "single global learned scalar mixture",
            "outer_test_outcomes_used_for_training_or_selection": False,
            "torch_version": torch.__version__,
            "sklearn_version": sklearn.__version__,
        },
    )
    print(f"Preflight complete: {setup}")


def stage_tune(args: argparse.Namespace) -> None:
    task, pair, outer_fold = require_task_pair_fold(args)
    folds = read_folds(args)
    tabular = read_tabular(args, folds) if pair == "ecg_tabular" else None
    device = torch.device(args.device)
    base = task_root(args, task) / "tuning" / pair / f"outer_fold_{outer_fold}"
    rows = []
    for base_config in tqdm(
        candidate_grid(args),
        desc=f"Tune {task.upper()} {pair} outer {outer_fold}",
        dynamic_ncols=True,
    ):
        config = {**base_config, "task": task}
        directory = base / config["config_name"]
        summary_path = directory / "tuning_summary.json"
        predictions_path = directory / "inner_oof_predictions.csv"
        if summary_path.exists() and predictions_path.exists() and not args.overwrite:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(summary)
            continue
        frames = []
        best_epochs = []
        parameter_counts = []
        gate_stats = []
        for inner_fold in range(args.inner_splits):
            data = load_pair_split(
                args, folds, pair, task, outer_fold, inner_fold, tabular
            )
            result = fit_with_validation(
                data,
                config,
                device,
                stable_seed(
                    args.seed, task, pair, config["config_name"], outer_fold, inner_fold
                ),
                f"{task.upper()} {config['config_name']} inner {inner_fold}",
            )
            frames.append(
                predictions_frame(
                    data.patient_ids_validation,
                    data.labels_validation,
                    result.probabilities,
                    outer_fold,
                    inner_fold,
                    pair,
                    config["fusion_method"],
                    task,
                )
            )
            best_epochs.append(result.best_epoch)
            parameter_counts.append(result.parameter_count)
            gate_stats.append(result.gate_stats)
        pooled = pd.concat(frames, ignore_index=True)
        metrics = pooled_selection_metrics(pooled)
        summary = {
            "completed": True,
            "created_at_utc": utc_now(),
            "modality_pair": pair,
            "task": task,
            "outer_fold": outer_fold,
            "config_name": config["config_name"],
            "config": config,
            **metrics,
            "inner_best_epochs": best_epochs,
            "parameter_count": parameter_counts[0],
            "gate_stats_by_inner_fold": gate_stats,
            "patient_count": int(len(pooled)),
        }
        atomic_csv(predictions_path, pooled)
        atomic_json(summary_path, summary)
        rows.append(summary)
    table = pd.DataFrame(
        [
            {
                "modality_pair": row["modality_pair"],
                "task": row["task"],
                "outer_fold": row["outer_fold"],
                "config_name": row["config_name"],
                "fusion_method": row["config"]["fusion_method"],
                "profile": row["config"]["profile"],
                "selection_score": row["selection_score"],
                "secondary_pr_auc": row["secondary_pr_auc"],
                "endpoint_roc_auc": row["endpoint_roc_auc"],
                "endpoint_pr_auc": row["endpoint_pr_auc"],
                "parameter_count": row["parameter_count"],
            }
            for row in rows
        ]
    ).sort_values(["fusion_method", "config_name"])
    atomic_csv(base / "all_candidate_results.csv", table)
    atomic_json(base / "run_complete.json", {"completed": True, "rows": len(table)})
    print(f"Tuning complete: {task.upper()}, {pair}, outer fold {outer_fold}")


def choose_candidate(summaries: list[dict], tolerance: float) -> dict:
    maximum = max(float(summary["selection_score"]) for summary in summaries)
    eligible = [
        summary
        for summary in summaries
        if float(summary["selection_score"]) >= maximum - tolerance
    ]
    eligible.sort(
        key=lambda summary: (
            int(summary["parameter_count"]),
            -float(summary["secondary_pr_auc"]),
            summary["config_name"],
        )
    )
    selected = copy.deepcopy(eligible[0])
    selected["maximum_candidate_endpoint_auc"] = maximum
    selected["auc_tolerance"] = tolerance
    selected["selection_rule"] = (
        "within the prespecified tolerance of maximum pooled inner endpoint-specific "
        "ROC-AUC; then fewest parameters, higher endpoint-specific PR-AUC, and "
        "configuration name"
    )
    return selected


def stage_select(args: argparse.Namespace) -> None:
    combined_rows = []
    for task in TASKS:
        rows = []
        selections: dict[str, dict[str, Any]] = {}
        for pair in MODALITY_PAIRS:
            selections[pair] = {}
            for outer_fold in range(args.outer_splits):
                base = (
                    task_root(args, task)
                    / "tuning"
                    / pair
                    / f"outer_fold_{outer_fold}"
                )
                summaries = []
                for config in candidate_grid(args):
                    path = base / config["config_name"] / "tuning_summary.json"
                    if not path.exists():
                        raise FileNotFoundError(f"Missing tuning summary: {path}")
                    summaries.append(json.loads(path.read_text(encoding="utf-8")))
                selected_by_method = {
                    method: choose_candidate(
                        [
                            summary
                            for summary in summaries
                            if summary["config"]["fusion_method"] == method
                        ],
                        args.auc_tolerance,
                    )
                    for method in FUSION_METHODS
                }
                selected_overall = choose_candidate(summaries, args.auc_tolerance)
                selections[pair][str(outer_fold)] = {
                    "selected_by_method": selected_by_method,
                    "selected_overall": selected_overall,
                }
                for arm, selected in [
                    *selected_by_method.items(),
                    ("selected_fusion", selected_overall),
                ]:
                    rows.append(
                        {
                            "task": task,
                            "modality_pair": pair,
                            "outer_fold": outer_fold,
                            "arm": arm,
                            "selected_config_name": selected["config_name"],
                            "selected_fusion_method": selected["config"]["fusion_method"],
                            "profile": selected["config"]["profile"],
                            "inner_endpoint_roc_auc": selected["selection_score"],
                            "inner_endpoint_pr_auc": selected["secondary_pr_auc"],
                            "parameter_count": selected["parameter_count"],
                        }
                    )
        selection_dir = task_root(args, task) / "selection"
        atomic_json(
            selection_dir / "selected_fusion_configurations_by_outer_fold.json",
            {
                "created_at_utc": utc_now(),
                "task": task,
                "independent_binary_model": True,
                "selections": selections,
            },
        )
        atomic_csv(
            selection_dir / "selected_fusion_configurations_by_outer_fold.csv",
            pd.DataFrame(rows),
        )
        combined_rows.extend(rows)
    atomic_csv(
        args.output_root / "combined_selection" / "selected_fusion_configurations_by_outer_fold.csv",
        pd.DataFrame(combined_rows),
    )
    print(f"Selection complete: {args.output_root / 'tasks'}")


def load_selections(args: argparse.Namespace, task: str) -> dict:
    path = (
        task_root(args, task)
        / "selection"
        / "selected_fusion_configurations_by_outer_fold.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing multimodal selection: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["selections"]


def fit_platt(y: np.ndarray, probabilities: np.ndarray, c_value: float, clip: float) -> dict:
    known = np.isfinite(y)
    y_known = y[known].astype(int)
    p_known = np.clip(probabilities[known], clip, 1 - clip)
    logit = np.log(p_known / (1 - p_known)).reshape(-1, 1)
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=10000)
    model.fit(logit, y_known)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def apply_platt(probabilities: np.ndarray, mapping: dict, clip: float) -> np.ndarray:
    p = np.clip(probabilities, clip, 1 - clip)
    logit = np.log(p / (1 - p))
    calibrated_logit = mapping["intercept"] + mapping["slope"] * logit
    return 1.0 / (1.0 + np.exp(-np.clip(calibrated_logit, -40, 40)))


def select_youden_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    known = np.isfinite(y)
    fpr, tpr, thresholds = roc_curve(y[known].astype(int), probabilities[known])
    finite = np.isfinite(thresholds)
    values = pd.DataFrame(
        {"threshold": thresholds[finite], "sensitivity": tpr[finite], "specificity": 1 - fpr[finite]}
    )
    values["youden"] = values["sensitivity"] + values["specificity"] - 1
    values = values.sort_values(
        ["youden", "sensitivity", "threshold"], ascending=[False, False, True]
    )
    return float(values.iloc[0]["threshold"])


def save_preprocessing(directory: Path, preprocessing: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "embedding_standardization.npz",
        ecg_mean=preprocessing["ecg_mean"],
        ecg_sd=preprocessing["ecg_sd"],
        second_mean=preprocessing.get("second_mean", np.asarray([], dtype=np.float32)),
        second_sd=preprocessing.get("second_sd", np.asarray([], dtype=np.float32)),
    )
    if "tabular_preprocessor" in preprocessing:
        joblib.dump(preprocessing["tabular_preprocessor"], directory / "tabular_preprocessor.joblib")
    if "text_provenance" in preprocessing:
        atomic_json(directory / "text_embedding_provenance.json", preprocessing["text_provenance"])


def stage_final(args: argparse.Namespace) -> None:
    task, pair, outer_fold = require_task_pair_fold(args)
    folds = read_folds(args)
    tabular = read_tabular(args, folds) if pair == "ecg_tabular" else None
    device = torch.device(args.device)
    selected = load_selections(args, task)[pair][str(outer_fold)]
    method_selections = selected["selected_by_method"]
    overall_method = selected["selected_overall"]["config"]["fusion_method"]
    data = load_pair_split(args, folds, pair, task, outer_fold, None, tabular)
    outer_directory = (
        task_root(args, task) / "final_models" / pair / f"outer_fold_{outer_fold}"
    )
    completed_predictions: dict[str, pd.DataFrame] = {}
    for method in tqdm(
        FUSION_METHODS,
        desc=f"Final {task.upper()} {pair} outer {outer_fold}",
        dynamic_ncols=True,
    ):
        selection = method_selections[method]
        config = selection["config"]
        arm_directory = outer_directory / "arms" / method
        completion = arm_directory / "run_complete.json"
        prediction_path = arm_directory / "outer_test_predictions.csv"
        if completion.exists() and prediction_path.exists() and not args.overwrite:
            completed_predictions[method] = pd.read_csv(prediction_path, dtype={PATIENT_ID: "string"})
            continue
        tuning_dir = (
            task_root(args, task)
            / "tuning"
            / pair
            / f"outer_fold_{outer_fold}"
            / selection["config_name"]
        )
        inner_predictions = pd.read_csv(
            tuning_dir / "inner_oof_predictions.csv", dtype={PATIENT_ID: "string"}
        )
        y = inner_predictions["y"].to_numpy(float)
        p = inner_predictions["prob"].to_numpy(float)
        platt = fit_platt(y, p, args.platt_C, args.probability_clip)
        calibrated = apply_platt(p, platt, args.probability_clip)
        threshold = select_youden_threshold(y, calibrated)
        final_epochs = max(
            1,
            int(math.floor(float(np.median(selection["inner_best_epochs"])) + 0.5)),
        )
        model = fit_fixed_epochs(
            data,
            config,
            final_epochs,
            device,
            stable_seed(args.seed, "final", task, pair, method, outer_fold),
            f"{task.upper()} {method} {final_epochs} epochs",
        )
        probabilities, gate_stats = predict_model(
            model,
            data.ecg_validation,
            data.second_validation,
            device,
            int(config["batch_size"]),
        )
        frame = predictions_frame(
            data.patient_ids_validation,
            data.labels_validation,
            probabilities,
            outer_fold,
            None,
            pair,
            method,
            task,
        )
        frame["calibrated_prob"] = apply_platt(
            frame["prob"].to_numpy(float), platt, args.probability_clip
        )
        frame["threshold"] = threshold
        frame["predicted"] = (
            frame["calibrated_prob"] >= threshold
        ).astype(int)
        arm_directory.mkdir(parents=True, exist_ok=True)
        atomic_csv(prediction_path, frame)
        atomic_torch_save(
            arm_directory / "checkpoint.pt",
            {
                "state_dict": model.state_dict(),
                "config": config,
                "ecg_dim": data.ecg_train.shape[1],
                "second_dim": data.second_train.shape[1],
                "final_epochs": final_epochs,
                "task": task,
                "seed": stable_seed(args.seed, "final", task, pair, method, outer_fold),
            },
        )
        save_preprocessing(arm_directory / "preprocessing", data.preprocessing)
        atomic_json(
            arm_directory / "platt_and_thresholds.json",
            {"task": task, "platt": platt, "threshold": threshold},
        )
        atomic_json(
            completion,
            {
                "completed": True,
                "created_at_utc": utc_now(),
                "modality_pair": pair,
                "task": task,
                "outer_fold": outer_fold,
                "arm": method,
                "selected_config_name": selection["config_name"],
                "final_epochs": final_epochs,
                "gate_stats": gate_stats,
                "outer_test_outcomes_used_for_training_selection_calibration_or_thresholds": False,
            },
        )
        completed_predictions[method] = frame
    selected_frame = completed_predictions[overall_method].copy()
    selected_frame["arm"] = "selected_fusion"
    selected_dir = outer_directory / "arms" / "selected_fusion"
    atomic_csv(selected_dir / "outer_test_predictions.csv", selected_frame)
    atomic_json(
        selected_dir / "run_complete.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "selected_from_method": overall_method,
            "task": task,
            "selected_config_name": selected["selected_overall"]["config_name"],
            "outer_test_outcomes_used_for_selection": False,
        },
    )
    atomic_json(
        outer_directory / "run_complete.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "arms": [*FUSION_METHODS, "selected_fusion"],
            "task": task,
            "selected_overall_method": overall_method,
        },
    )
    print(f"Final complete: {task.upper()}, {pair}, outer fold {outer_fold}")


def stage_aggregate(args: argparse.Namespace) -> None:
    all_frames = []
    for task in TASKS:
        destination = task_root(args, task) / "pooled_outer_test"
        expected_patients = expected_task_patients(args, task)
        task_frames = []
        for pair in MODALITY_PAIRS:
            for arm in (*FUSION_METHODS, "selected_fusion"):
                frames = []
                for outer_fold in range(args.outer_splits):
                    path = (
                        task_root(args, task)
                        / "final_models"
                        / pair
                        / f"outer_fold_{outer_fold}"
                        / "arms"
                        / arm
                        / "outer_test_predictions.csv"
                    )
                    if not path.exists():
                        raise FileNotFoundError(f"Missing outer-test predictions: {path}")
                    frames.append(pd.read_csv(path, dtype={PATIENT_ID: "string"}))
                pooled = pd.concat(frames, ignore_index=True)
                if (
                    len(pooled) != expected_patients
                    or not pooled[PATIENT_ID].is_unique
                    or not pooled["task"].eq(task).all()
                ):
                    raise ValueError(f"Invalid pooled predictions for {task}/{pair}/{arm}")
                path = destination / pair / f"{arm}_pooled_outer_test_predictions.csv"
                atomic_csv(path, pooled)
                task_frames.append(pooled)
                all_frames.append(pooled)
        task_combined = pd.concat(task_frames, ignore_index=True)
        atomic_csv(
            destination / "all_multimodal_pooled_outer_test_predictions.csv",
            task_combined,
        )
        atomic_json(
            destination / "aggregation_manifest.json",
            {
                "completed": True,
                "created_at_utc": utc_now(),
                "task": task,
                "independent_binary_model": True,
                "patients_per_arm": expected_patients,
                "events_per_arm": args.expected_scd if task == "scd" else args.expected_pfd,
                "modality_pairs": MODALITY_PAIRS,
                "arms": [*FUSION_METHODS, "selected_fusion"],
                "outer_test_predictions_per_patient_pair_arm": 1,
            },
        )
    combined = pd.concat(all_frames, ignore_index=True)
    expected_rows = sum(expected_task_patients(args, task) for task in TASKS) * len(
        MODALITY_PAIRS
    ) * (len(FUSION_METHODS) + 1)
    if len(combined) != expected_rows:
        raise ValueError(f"Expected {expected_rows} pooled rows; found {len(combined)}")
    destination = args.output_root / "combined_evaluation"
    atomic_csv(destination / "all_multimodal_pooled_outer_test_predictions.csv", combined)
    atomic_json(
        destination / "aggregation_manifest.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "independent_binary_tasks": True,
            "patients_per_arm": {
                task: expected_task_patients(args, task) for task in TASKS
            },
            "modality_pairs": MODALITY_PAIRS,
            "arms": [*FUSION_METHODS, "selected_fusion"],
            "outer_test_predictions_per_patient_pair_arm": 1,
        },
    )
    print(f"Aggregation complete: {destination}")


def metric_values(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, raw)),
        "pr_auc": float(average_precision_score(y, raw)),
        "brier": float(brier_score_loss(y, calibrated)),
    }


def bootstrap_metric_ci(
    y: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    replicates: int,
    seed: int,
    description: str,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    values = {"roc_auc": [], "pr_auc": [], "brier": []}
    for _ in tqdm(range(replicates), desc=description, leave=False, dynamic_ncols=True):
        indices = rng.integers(0, len(y), len(y))
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        metrics = metric_values(sampled_y, raw[indices], calibrated[indices])
        for key, value in metrics.items():
            values[key].append(value)
    return {
        key: (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        for key, samples in values.items()
    }


def stage_evaluate(args: argparse.Namespace) -> None:
    path = (
        args.output_root
        / "combined_evaluation"
        / "all_multimodal_pooled_outer_test_predictions.csv"
    )
    frame = pd.read_csv(path, dtype={PATIENT_ID: "string"})
    performance_rows = []
    threshold_rows = []
    groups = frame.groupby(["task", "modality_pair", "arm"], sort=True)
    for (task, pair, arm), group in tqdm(
        groups, total=len(TASKS) * len(MODALITY_PAIRS) * 6, desc="Evaluate arms"
    ):
        y = group["y"].to_numpy(int)
        raw = group["prob"].to_numpy(float)
        calibrated = group["calibrated_prob"].to_numpy(float)
        estimates = metric_values(y, raw, calibrated)
        intervals = bootstrap_metric_ci(
            y,
            raw,
            calibrated,
            args.bootstrap_replicates,
            stable_seed(args.seed, "bootstrap", task, pair, arm),
            f"{task}/{pair}/{arm}",
        )
        for metric, estimate in estimates.items():
            performance_rows.append(
                {
                    "modality_pair": pair,
                    "arm": arm,
                    "outcome": task,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": intervals[metric][0],
                    "ci_upper": intervals[metric][1],
                    "patients": len(y),
                    "events": int(y.sum()),
                }
            )
        predicted = group["predicted"].to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        npv = tn / (tn + fn) if tn + fn else np.nan
        threshold_rows.append(
            {
                "modality_pair": pair,
                "arm": arm,
                "outcome": task,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    evaluation = args.output_root / "combined_evaluation"
    atomic_csv(evaluation / "pooled_performance_with_95ci.csv", pd.DataFrame(performance_rows))
    atomic_csv(evaluation / "pooled_threshold_metrics.csv", pd.DataFrame(threshold_rows))
    atomic_json(
        evaluation / "evaluation_manifest.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "independent_binary_tasks": True,
            "competing_endpoints_excluded": True,
            "outer_test_outcomes_used_for_training_selection_calibration_or_thresholds": False,
            "paired_cross_model_comparisons_deferred_to_unified_final_analysis": True,
        },
    )
    print(f"Evaluation complete: {evaluation}")


def main() -> None:
    args = parse_args()
    for name in (
        "folds_csv",
        "ecg_root",
        "text_embedding_root",
        "text_results_root",
        "tabular_csv",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
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


if __name__ == "__main__":
    main()
