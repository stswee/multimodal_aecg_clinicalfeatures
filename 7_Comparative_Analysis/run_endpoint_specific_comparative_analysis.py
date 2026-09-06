#!/usr/bin/env python3
"""Final endpoint-specific comparative analysis and multimodal text diagnostics.

This single staged program replaces the former collection of comparative,
plotting, decision-curve, provenance, attribution, and highlighted-response
scripts.  It never retrains or reselects any principal prediction model.

Stages
------
preflight
    Validate all result sources, endpoint cohorts, checkpoints, response files,
    and immutable fold assignments.
compare
    Standardize the seven principal-model prediction sources, calculate pooled
    performance/calibration/threshold metrics, paired patient-bootstrap
    comparisons, feature provenance, exploratory decision curves, and figures.
attribute-fold
    On one GPU, analyze both endpoint-specific ECG + full-LLM-text models for
    one untouched outer fold using integrated gradients through the frozen text
    encoder and the prespecified direct-concatenation checkpoint. Includes refit-seed,
    input-variant, integration-step, masking, and randomization controls.
attribute-aggregate
    Pool the five attribution folds, bootstrap summaries, draw diagnostic and
    de-identified highlighted-response figures, and write the final manifest.

Attributions are technical model-behavior diagnostics.  They are not clinical
explanations and are not clinician validated.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
try:
    from tqdm.auto import tqdm
except ImportError:  # The analysis remains usable in minimal CPU environments.
    def tqdm(iterable, **_kwargs):
        return iterable


PATIENT_ID = "Patient ID"
TASKS = ("scd", "pfd")
EXPECTED_PROMPT_SHA256 = "4676a5e8d918b0ba51df136c59c59062b5b2df1efe97b58d34d30d094b35ec79"
MODELS = (
    "ecg", "tabular", "full_llm_text", "deterministic_text",
    "ecg_full_llm", "ecg_deterministic", "ecg_tabular",
)
MODEL_LABELS = {
    "ecg": "ECG only",
    "tabular": "Tabular only",
    "full_llm_text": "LLM text only",
    "deterministic_text": "Deterministic text only",
    "ecg_full_llm": "ECG + LLM text (DC)",
    "ecg_deterministic": "ECG + deterministic text (DC)",
    "ecg_tabular": "ECG + tabular (DC)",
}
COLORS = {
    "ecg": "#3B6FB6", "tabular": "#D55E00", "full_llm_text": "#8E5EA2",
    "deterministic_text": "#009E73", "ecg_full_llm": "#CC79A7",
    "ecg_deterministic": "#56B4E9", "ecg_tabular": "#E69F00",
}
LINESTYLES = {
    "ecg": "--", "tabular": "-", "full_llm_text": ":",
    "deterministic_text": "-.", "ecg_full_llm": ":",
    "ecg_deterministic": "-.", "ecg_tabular": "-",
}
METRICS = ("roc_auc", "pr_auc", "brier")
THRESHOLD_METRICS = (
    "sensitivity", "specificity", "ppv", "npv", "accuracy", "f1",
    "balanced_accuracy",
)

# One prespecified family across all manuscript-level comparisons.
PRIMARY_COMPARISONS = {
    "incremental_fusion": (
        ("ecg_full_llm", "ecg"), ("ecg_full_llm", "full_llm_text"),
        ("ecg_deterministic", "ecg"),
        ("ecg_deterministic", "deterministic_text"),
        ("ecg_tabular", "ecg"), ("ecg_tabular", "tabular"),
    ),
    "unimodal_representation": (
        ("tabular", "full_llm_text"),
        ("deterministic_text", "full_llm_text"),
        ("tabular", "deterministic_text"),
    ),
    "multimodal_representation": (
        ("ecg_tabular", "ecg_full_llm"),
        ("ecg_deterministic", "ecg_full_llm"),
        ("ecg_tabular", "ecg_deterministic"),
    ),
}

CONTINUOUS = [
    "Age", "Weight (kg)", "Height (cm)",
    "Diastolic blood  pressure (mmHg)", "Systolic blood pressure (mmHg)",
    "Albumin (g/L)", "ALT or GPT (IU/L)", "AST or GOT (IU/L)",
    "Total Cholesterol (mmol/L)", "Creatinine (?mol/L)",
    "Gamma-glutamil transpeptidase (IU/L)", "Glucose (mmol/L)",
    "Hemoglobin (g/L)", "HDL (mmol/L)", "Potassium (mEq/L)",
    "LDL (mmol/L)", "Sodium (mEq/L)", "Pro-BNP (ng/L)", "Protein (g/L)",
    "T3 (pg/dL)", "T4 (ng/L)", "Troponin (ng/mL)", "TSH (mIU/L)",
    "Urea (mg/dL)", "LVEF (%)",
]
CATEGORICAL = [
    "Gender (male=1)", "NYHA class", "HF etiology - Diagnosis",
    "Diabetes (yes=1)", "History of dyslipemia (yes=1)",
    "Peripheral vascular disease (yes=1)", "History of hypertension (yes=1)",
    "Prior Myocardial Infarction (yes=1)", "Calcium channel blocker (yes=1)",
    "Diabetes medication (yes=1)", "Amiodarone (yes=1)",
    "Angiotensin-II receptor blocker (yes=1)",
    "Anticoagulants/antitrombotics  (yes=1)", "Betablockers (yes=1)",
    "Digoxin (yes=1)", "Loop diuretics (yes=1)", "Spironolactone (yes=1)",
    "Statins (yes=1)", "Hidralazina (yes=1)", "ACE inhibitor (yes=1)",
    "Nitrovasodilator (yes=1)",
]
ECG_IMPRESSIONS = [
    "Ventricular Extrasystole", "Ventricular Tachycardia",
    "Non-sustained ventricular tachycardia (CH>10)",
    "Paroxysmal supraventricular tachyarrhythmia", "Bradycardia",
]


def parse_args() -> argparse.Namespace:
    music = Path("/home/sswee/music")
    project = Path("/home/sswee/multimodal_aecg_clinicalfeatures")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=["preflight", "compare", "attribute-fold", "attribute-aggregate"],
    )
    parser.add_argument("--outer_fold", type=int)
    parser.add_argument("--folds_csv", type=Path, default=music / "ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv")
    parser.add_argument("--ecg_predictions", type=Path, default=music / "ecg_nested_4year_three_wave/threshold_analysis/pooled_outer_test_classifications.csv")
    parser.add_argument("--tabular_predictions", type=Path, default=music / "tabular_nested_4year_v2/evaluation/models/prompt_matched_no_ecg/selected_tabular/pooled_predictions_calibrated_and_classified.csv")
    parser.add_argument("--text_predictions", type=Path, default=music / "text_nested_4year_v4_detailed/combined_evaluation/all_arms_pooled_predictions_calibrated_and_classified.csv")
    parser.add_argument("--multimodal_predictions", type=Path, default=music / "multimodal_nested_4year_v4_detailed/combined_evaluation/all_multimodal_pooled_outer_test_predictions.csv")
    parser.add_argument("--subject_info_csv", type=Path, default=music / "subject-info.csv")
    parser.add_argument("--prompt_csv", type=Path, default=music / "subject-info-cleaned-4year-with-prompts.csv")
    parser.add_argument("--ecg_root", type=Path, default=music / "ecg_nested_4year_three_wave")
    parser.add_argument("--text_embedding_root", type=Path, default=music / "text_embeddings_4year_v4_detailed")
    parser.add_argument("--text_results_root", type=Path, default=music / "text_nested_4year_v4_detailed")
    parser.add_argument("--multimodal_root", type=Path, default=music / "multimodal_nested_4year_v4_detailed")
    parser.add_argument("--llama8b_csv", type=Path, default=music / "llm_responses_4year_v3_detailed/LLaMA3.1-8B-4year-responses.csv")
    parser.add_argument("--llama3b_csv", type=Path, default=music / "llm_responses_4year_v3_detailed/LLaMA3.2-3B-4year-responses-postprocessed.csv")
    parser.add_argument("--multimodal_training_script", type=Path, default=project / "6_Multimodal_Modeling/train_multimodal_nested_cv.py")
    parser.add_argument("--output_root", type=Path, default=music / "comparative_analysis_4year_v4_detailed_endpoint_specific")
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--attribution_bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_patients", type=int, default=730)
    parser.add_argument("--expected_controls", type=int, default=577)
    parser.add_argument("--expected_scd", type=int, default=71)
    parser.add_argument("--expected_pfd", type=int, default=82)
    parser.add_argument("--threshold_min", type=float, default=0.02)
    parser.add_argument("--threshold_max", type=float, default=0.25)
    parser.add_argument("--threshold_step", type=float, default=0.0025)
    parser.add_argument("--calibration_groups", type=int, default=5)
    parser.add_argument("--multimodal_arm", default="concatenation")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--samples_per_class", type=int, default=6)
    parser.add_argument("--ig_steps", type=int, default=24)
    parser.add_argument("--refit_seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--mask_fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--random_control_repeats", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--chunk_stride", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.stage == "attribute-fold" and args.outer_fold not in range(5):
        parser.error("--outer_fold must be 0, 1, 2, 3, or 4")
    if not 0 < args.threshold_min < args.threshold_max < 1:
        parser.error("Require 0 < threshold_min < threshold_max < 1")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *map(str, parts)])
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def canonical_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text.casefold()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False) as h:
        frame.to_csv(h, index=False)
        temporary = Path(h.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False) as h:
        json.dump(value, h, indent=2, sort_keys=True)
        h.write("\n")
        temporary = Path(h.name)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=path.parent, delete=False) as h:
        h.write(text)
        temporary = Path(h.name)
    os.replace(temporary, path)


def load_mm(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("multimodal_training_v3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_locked_folds(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.folds_csv, dtype={PATIENT_ID: "string"})
    required = {PATIENT_ID, "outer_fold", "SCD_4year_label", "PFD_4year_label"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Locked folds lack {sorted(missing)}")
    frame["patient_key"] = frame[PATIENT_ID].map(canonical_id)
    frame["scd"] = pd.to_numeric(frame["SCD_4year_label"], errors="coerce")
    frame["pfd"] = pd.to_numeric(frame["PFD_4year_label"], errors="coerce")
    frame["outer_fold"] = pd.to_numeric(frame["outer_fold"], errors="raise").astype(int)
    if len(frame) != args.expected_patients or not frame.patient_key.is_unique:
        raise ValueError("Locked cohort size or patient uniqueness failed.")
    controls = frame.scd.eq(0) & frame.pfd.eq(0)
    observed = (int(controls.sum()), int(frame.scd.eq(1).sum()), int(frame.pfd.eq(1).sum()))
    expected = (args.expected_controls, args.expected_scd, args.expected_pfd)
    if observed != expected:
        raise ValueError(f"Outcome counts {observed} != {expected}")
    if (frame.scd.eq(1) & frame.pfd.eq(1)).any():
        raise ValueError("At least one patient has both endpoints.")
    frame["expected_scd"] = np.where(frame.scd.eq(1), 1.0, np.where(controls, 0.0, np.nan))
    frame["expected_pfd"] = np.where(frame.pfd.eq(1), 1.0, np.where(controls, 0.0, np.nan))
    return frame


def choose_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str:
    for column in candidates:
        if column in frame:
            return column
    raise ValueError(f"None of {list(candidates)} found; columns={frame.columns.tolist()}")


def standardize_wide(path: Path, model: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype={PATIENT_ID: "string"})
    patient = choose_column(raw, [PATIENT_ID, "patient_id", "patient"])
    fold = choose_column(raw, ["outer_fold", "fold", "outer_test_fold"])
    pieces = []
    for task in TASKS:
        label = choose_column(raw, [f"{task}_label", f"y_{task}", f"{task.upper()}_4year_label"])
        score = choose_column(raw, [f"{task}_probability_uncalibrated", f"{task}_probability", f"prob_{task}"])
        calibrated = choose_column(raw, [f"{task}_probability_calibrated", f"calibrated_prob_{task}"])
        threshold = choose_column(raw, [f"{task}_selected_threshold", f"threshold_{task}"])
        predicted = choose_column(raw, [f"{task}_predicted_class", f"predicted_{task}"])
        part = pd.DataFrame({
            "patient_key": raw[patient].map(canonical_id), PATIENT_ID: raw[patient].astype(str),
            "outer_fold": pd.to_numeric(raw[fold], errors="raise").astype(int),
            "task": task, "y": pd.to_numeric(raw[label], errors="coerce"),
            "raw": pd.to_numeric(raw[score], errors="coerce"),
            "calibrated": pd.to_numeric(raw[calibrated], errors="coerce"),
            "threshold": pd.to_numeric(raw[threshold], errors="coerce"),
            "predicted": pd.to_numeric(raw[predicted], errors="coerce"),
        })
        pieces.append(part[part.y.notna()].copy())
    result = pd.concat(pieces, ignore_index=True)
    result["model"] = model
    result["model_label"] = MODEL_LABELS[model]
    return result, {"model": model, "path": str(path), "sha256": sha256_file(path), "format": "wide"}


def standardize_text(path: Path, model: str, arm: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype={PATIENT_ID: "string"})
    if "arm" not in raw:
        raise ValueError(f"Text predictions lack arm: {path}")
    raw = raw[raw.arm.eq(arm)].copy()
    pieces = []
    for task in TASKS:
        task_rows = raw[raw.get("analysis_task", raw.get("outcome", "")).astype(str).str.lower().eq(task)].copy()
        if task_rows.empty:
            raise ValueError(f"No {task}/{arm} text rows")
        part = pd.DataFrame({
            "patient_key": task_rows[PATIENT_ID].map(canonical_id), PATIENT_ID: task_rows[PATIENT_ID].astype(str),
            "outer_fold": pd.to_numeric(task_rows.outer_fold, errors="raise").astype(int), "task": task,
            "y": pd.to_numeric(task_rows[f"{task}_label"], errors="raise"),
            "raw": pd.to_numeric(task_rows[f"{task}_probability_uncalibrated"], errors="raise"),
            "calibrated": pd.to_numeric(task_rows[f"{task}_probability_calibrated"], errors="raise"),
            "threshold": pd.to_numeric(task_rows[f"{task}_selected_threshold"], errors="raise"),
            "predicted": pd.to_numeric(task_rows[f"{task}_predicted_class"], errors="raise"),
        })
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    result["model"] = model; result["model_label"] = MODEL_LABELS[model]
    return result, {"model": model, "path": str(path), "sha256": sha256_file(path), "format": "endpoint-specific long", "filter": f"arm={arm}"}


def standardize_multimodal(path: Path, model: str, pair: str, arm: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, dtype={PATIENT_ID: "string"})
    subset = raw[raw.modality_pair.eq(pair) & raw.arm.eq(arm)].copy()
    result = pd.DataFrame({
        "patient_key": subset[PATIENT_ID].map(canonical_id), PATIENT_ID: subset[PATIENT_ID].astype(str),
        "outer_fold": pd.to_numeric(subset.outer_fold, errors="raise").astype(int),
        "task": subset.task.astype(str).str.lower(), "y": pd.to_numeric(subset.y, errors="raise"),
        "raw": pd.to_numeric(subset.prob, errors="raise"),
        "calibrated": pd.to_numeric(subset.calibrated_prob, errors="raise"),
        "threshold": pd.to_numeric(subset.threshold, errors="raise"),
        "predicted": pd.to_numeric(subset.predicted, errors="raise"),
    })
    result["model"] = model; result["model_label"] = MODEL_LABELS[model]
    return result, {"model": model, "path": str(path), "sha256": sha256_file(path), "format": "endpoint-specific long", "filter": f"modality_pair={pair};arm={arm}"}


def validate_standardized(frame: pd.DataFrame, folds: pd.DataFrame, args: argparse.Namespace) -> None:
    expected_n = {"scd": args.expected_controls + args.expected_scd, "pfd": args.expected_controls + args.expected_pfd}
    expected_events = {"scd": args.expected_scd, "pfd": args.expected_pfd}
    lookup = folds.set_index("patient_key")
    if frame.duplicated(["model", "task", "patient_key"]).any():
        raise ValueError("Duplicate model-task-patient predictions.")
    for task in TASKS:
        expected_ids = set(folds.loc[folds[f"expected_{task}"].notna(), "patient_key"])
        for model in MODELS:
            group = frame[frame.model.eq(model) & frame.task.eq(task)].copy()
            if len(group) != expected_n[task] or set(group.patient_key) != expected_ids:
                raise ValueError(f"Patient set mismatch for {model}/{task}: {len(group)}")
            aligned = lookup.loc[group.patient_key]
            if not np.array_equal(group.outer_fold.to_numpy(int), aligned.outer_fold.to_numpy(int)):
                raise ValueError(f"Outer-fold mismatch for {model}/{task}")
            if not np.array_equal(group.y.to_numpy(int), aligned[f"expected_{task}"].to_numpy(int)):
                raise ValueError(f"Label mismatch for {model}/{task}")
            if int(group.y.sum()) != expected_events[task]:
                raise ValueError(f"Event-count mismatch for {model}/{task}")
            values = group[["raw", "calibrated", "threshold", "predicted"]].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite predictions for {model}/{task}")
            if ((group[["raw", "calibrated"]].to_numpy(float) < 0) | (group[["raw", "calibrated"]].to_numpy(float) > 1)).any():
                raise ValueError(f"Probability outside [0,1] for {model}/{task}")
            recreated = (group.calibrated >= group.threshold).astype(int)
            if not np.array_equal(recreated, group.predicted.astype(int)):
                raise ValueError(f"Threshold/class mismatch for {model}/{task}")


def metric_values(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, float]:
    return {"roc_auc": roc_auc_score(y, raw), "pr_auc": average_precision_score(y, raw), "brier": brier_score_loss(y, calibrated)}


def calibration_values(y: np.ndarray, p: np.ndarray, epsilon: float = 1e-6) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), epsilon, 1 - epsilon)
    x = np.log(p / (1 - p)); design = np.column_stack([np.ones(len(x)), x]); beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-np.clip(design @ beta, -35, 35)))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        try:
            step = np.linalg.solve(design.T @ (w[:, None] * design), design.T @ (y - mu))
        except np.linalg.LinAlgError:
            return {"calibration_intercept": np.nan, "calibration_slope": np.nan, "calibration_in_the_large": np.nan}
        beta += step
        if np.max(np.abs(step)) < 1e-8: break
    citl = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-np.clip(x + citl, -35, 35)))
        step = np.sum(y - mu) / np.sum(np.clip(mu * (1 - mu), 1e-9, None)); citl += step
        if abs(step) < 1e-8: break
    return {"calibration_intercept": float(beta[0]), "calibration_slope": float(beta[1]), "calibration_in_the_large": float(citl)}


def threshold_values(y: np.ndarray, predicted: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else np.nan
    values = {
        "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }
    values["balanced_accuracy"] = (values["sensitivity"] + values["specificity"]) / 2
    return values, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    return tuple(map(float, np.percentile(values, [2.5, 97.5]))) if len(values) else (np.nan, np.nan)


def paired_p(values: np.ndarray) -> float:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    return float(min(1, 2 * min((np.sum(values <= 0) + 1) / (len(values) + 1), (np.sum(values >= 0) + 1) / (len(values) + 1))))


def holm(series: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=series.index); ordered = series.dropna().sort_values(); running = 0.0; m = len(ordered)
    for rank, (index, value) in enumerate(ordered.items()):
        running = max(running, min(1.0, (m - rank) * float(value))); output.loc[index] = running
    return output


def bootstrap_evaluation(frame: pd.DataFrame, args: argparse.Namespace):
    perf_rows, cal_rows, threshold_rows, confusion_rows, threshold_summary = [], [], [], [], []
    replicate_store: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for task_index, task in enumerate(TASKS):
        reference = frame[frame.model.eq(MODELS[0]) & frame.task.eq(task)].sort_values("patient_key")
        keys = reference.patient_key.tolist(); y = reference.y.to_numpy(int); n = len(y)
        rng = np.random.default_rng(stable_seed(args.seed, "bootstrap", task))
        samples = rng.integers(0, n, size=(args.bootstrap_replicates, n), dtype=np.int32)
        replicate_store[task] = {}
        for model in tqdm(MODELS, desc=f"Bootstrap {task.upper()}", unit="model"):
            group = frame[frame.model.eq(model) & frame.task.eq(task)].set_index("patient_key").loc[keys]
            raw = group.raw.to_numpy(float); calibrated = group.calibrated.to_numpy(float); predicted = group.predicted.to_numpy(int)
            point_perf = metric_values(y, raw, calibrated); point_cal = calibration_values(y, calibrated); point_thr, counts = threshold_values(y, predicted)
            perf_reps = {m: np.full(args.bootstrap_replicates, np.nan) for m in METRICS}
            cal_reps = {m: np.full(args.bootstrap_replicates, np.nan) for m in point_cal}
            thr_reps = {m: np.full(args.bootstrap_replicates, np.nan) for m in THRESHOLD_METRICS}
            for index, sampled in enumerate(samples):
                ys = y[sampled]
                if np.unique(ys).size < 2: continue
                for name, value in metric_values(ys, raw[sampled], calibrated[sampled]).items(): perf_reps[name][index] = value
                for name, value in calibration_values(ys, calibrated[sampled]).items(): cal_reps[name][index] = value
                for name, value in threshold_values(ys, predicted[sampled])[0].items(): thr_reps[name][index] = value
            replicate_store[task][model] = perf_reps
            for name, value in point_perf.items():
                lo, hi = ci(perf_reps[name]); perf_rows.append({"model": model, "model_label": MODEL_LABELS[model], "outcome": task.upper(), "metric": name, "estimate": value, "ci_lower": lo, "ci_upper": hi, "patients": n, "events": int(y.sum())})
            for name, value in point_cal.items():
                lo, hi = ci(cal_reps[name]); cal_rows.append({"model": model, "model_label": MODEL_LABELS[model], "outcome": task.upper(), "metric": name, "estimate": value, "ci_lower": lo, "ci_upper": hi, "patients": n, "events": int(y.sum())})
            for name, value in point_thr.items():
                lo, hi = ci(thr_reps[name]); threshold_rows.append({"model": model, "model_label": MODEL_LABELS[model], "outcome": task.upper(), "metric": name, "estimate": value, "ci_lower": lo, "ci_upper": hi, "patients": n, "events": int(y.sum())})
            confusion_rows.append({"model": model, "model_label": MODEL_LABELS[model], "outcome": task.upper(), **counts, "patients": n, "events": int(y.sum())})
            by_fold = group.reset_index().groupby("outer_fold").threshold
            if not by_fold.nunique().eq(1).all(): raise ValueError(f"Multiple thresholds in fold for {model}/{task}")
            values = by_fold.first()
            threshold_summary.append({"model": model, "model_label": MODEL_LABELS[model], "outcome": task.upper(), "fold_thresholds": json.dumps({str(int(k)): float(v) for k, v in values.items()}), "minimum": values.min(), "median": values.median(), "maximum": values.max(), "selection_source": "saved training-only fold-specific threshold"})
    return tuple(map(pd.DataFrame, [perf_rows, cal_rows, threshold_rows, confusion_rows, threshold_summary])), replicate_store


def comparisons(performance: pd.DataFrame, replicates: dict) -> pd.DataFrame:
    estimates = performance.set_index(["model", "outcome", "metric"]).estimate
    rows = []
    for subgroup, pairs in PRIMARY_COMPARISONS.items():
        for left, right in pairs:
            for task in TASKS:
                for metric in METRICS:
                    difference = float(estimates.loc[(left, task.upper(), metric)] - estimates.loc[(right, task.upper(), metric)])
                    reps = replicates[task][left][metric] - replicates[task][right][metric]
                    lo, hi = ci(reps); direction = -1 if metric == "brier" else 1
                    rows.append({"family": "primary_confirmatory", "subgroup": subgroup, "comparator": left, "comparator_label": MODEL_LABELS[left], "reference": right, "reference_label": MODEL_LABELS[right], "outcome": task.upper(), "metric": metric, "difference_comparator_minus_reference": difference, "difference_ci_lower": lo, "difference_ci_upper": hi, "advantage_comparator": direction * difference, "advantage_ci_lower": -hi if direction < 0 else lo, "advantage_ci_upper": -lo if direction < 0 else hi, "paired_bootstrap_p": paired_p(reps)})
    result = pd.DataFrame(rows); result["holm_family_size"] = len(result); result["holm_adjusted_p"] = holm(result.paired_bootstrap_p); result["holm_significant_0_05"] = result.holm_adjusted_p < .05
    return result


def all_pairwise(performance: pd.DataFrame, replicates: dict) -> pd.DataFrame:
    estimates = performance.set_index(["model", "outcome", "metric"]).estimate; rows = []
    for i, left in enumerate(MODELS):
        for right in MODELS[i + 1:]:
            for task in TASKS:
                for metric in METRICS:
                    difference = float(estimates.loc[(left, task.upper(), metric)] - estimates.loc[(right, task.upper(), metric)]); reps = replicates[task][left][metric] - replicates[task][right][metric]; lo, hi = ci(reps); direction = -1 if metric == "brier" else 1
                    rows.append({"family": "all_pairwise_exploratory", "comparator": left, "comparator_label": MODEL_LABELS[left], "reference": right, "reference_label": MODEL_LABELS[right], "outcome": task.upper(), "metric": metric, "difference_comparator_minus_reference": difference, "difference_ci_lower": lo, "difference_ci_upper": hi, "advantage_comparator": direction * difference, "advantage_ci_lower": -hi if direction < 0 else lo, "advantage_ci_upper": -lo if direction < 0 else hi, "paired_bootstrap_p": paired_p(reps)})
    result = pd.DataFrame(rows); result["holm_family_size"] = len(result); result["holm_adjusted_p"] = holm(result.paired_bootstrap_p); result["holm_significant_0_05"] = result.holm_adjusted_p < .05
    return result


def wilson(events: int, n: int) -> tuple[float, float]:
    z = norm.ppf(.975); p = events / n; denominator = 1 + z*z/n; center = (p + z*z/(2*n))/denominator; half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return max(0., center-half), min(1., center+half)


def set_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig: plt.Figure, directory: Path, stem: str, dpi: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory/f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(directory/f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(directory/f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_discrimination(frame: pd.DataFrame, directory: Path, dpi: int) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.4)); rows = []
    for r, task in enumerate(TASKS):
        for model in MODELS:
            g = frame[frame.model.eq(model) & frame.task.eq(task)].sort_values("patient_key"); y = g.y.to_numpy(int); score = g.raw.to_numpy(float)
            fpr, tpr, _ = roc_curve(y, score); precision, recall, _ = precision_recall_curve(y, score); auc = roc_auc_score(y, score); ap = average_precision_score(y, score)
            axes[r,0].plot(fpr,tpr,color=COLORS[model],ls=LINESTYLES[model],lw=1.7,label=f"{MODEL_LABELS[model]} ({auc:.3f})"); axes[r,1].plot(recall,precision,color=COLORS[model],ls=LINESTYLES[model],lw=1.7,label=f"{MODEL_LABELS[model]} ({ap:.3f})")
            rows.extend({"outcome":task.upper(),"model":model,"curve":"ROC","x":x,"y":z,"estimate":auc} for x,z in zip(fpr,tpr)); rows.extend({"outcome":task.upper(),"model":model,"curve":"PR","x":x,"y":z,"estimate":ap} for x,z in zip(recall,precision))
        axes[r,0].plot([0,1],[0,1],"--",color="#777",lw=1,label="Chance"); axes[r,1].axhline(y.mean(),ls="--",color="#777",lw=1,label=f"Prevalence ({y.mean():.3f})")
        axes[r,0].set(title=f"{task.upper()} ROC",xlabel="1 - specificity",ylabel="Sensitivity"); axes[r,1].set(title=f"{task.upper()} precision-recall",xlabel="Recall",ylabel="Precision")
        for ax in axes[r]: ax.set(xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.18); ax.legend(frameon=False)
    fig.suptitle("Seven principal models: pooled untouched outer-test discrimination",fontsize=13); fig.tight_layout(); save_figure(fig,directory,"principal_discrimination_curves",dpi); return pd.DataFrame(rows)


def calibration_points(frame: pd.DataFrame, groups: int) -> pd.DataFrame:
    rows=[]
    for task in TASKS:
        for model in MODELS:
            g=frame[frame.model.eq(model)&frame.task.eq(task)].sort_values("calibrated").copy(); g["risk_group"]=pd.qcut(np.arange(len(g)),groups,labels=False)+1
            for group,v in g.groupby("risk_group"):
                n=len(v); events=int(v.y.sum()); lo,hi=wilson(events,n); rows.append({"outcome":task.upper(),"model":model,"model_label":MODEL_LABELS[model],"risk_group":group,"patients":n,"events":events,"mean_predicted_probability":v.calibrated.mean(),"observed_event_rate":events/n,"observed_ci_lower":lo,"observed_ci_upper":hi})
    return pd.DataFrame(rows)


def plot_calibration(points: pd.DataFrame, directory: Path, dpi: int) -> None:
    fig,axes=plt.subplots(2,2,figsize=(12,9)); panels=(("Unimodal",MODELS[:4]),("Multimodal + ECG reference",("ecg",*MODELS[4:])))
    for r,task in enumerate(TASKS):
        for c,(title,models) in enumerate(panels):
            ax=axes[r,c]; ax.plot([0,.4],[0,.4],"--",color="#777",label="Ideal")
            for model in models:
                v=points[points.outcome.eq(task.upper())&points.model.eq(model)]; low=np.maximum(0,v.observed_event_rate-v.observed_ci_lower); high=np.maximum(0,v.observed_ci_upper-v.observed_event_rate)
                ax.errorbar(v.mean_predicted_probability,v.observed_event_rate,yerr=[low,high],color=COLORS[model],ls=LINESTYLES[model],marker="o",ms=3,capsize=2,label=MODEL_LABELS[model])
            ax.set(title=f"{task.upper()}: {title}",xlabel="Mean predicted probability",ylabel="Observed event proportion",xlim=(0,.4),ylim=(0,.4)); ax.grid(alpha=.18); ax.legend(frameon=False)
    fig.suptitle("Fold-specific calibrated outer-test predictions",fontsize=13); fig.tight_layout(); save_figure(fig,directory,"principal_calibration_plots",dpi)


def decision_curves(frame: pd.DataFrame, args: argparse.Namespace, directory: Path) -> tuple[pd.DataFrame,pd.DataFrame]:
    thresholds=np.arange(args.threshold_min,args.threshold_max+args.threshold_step/2,args.threshold_step); rows=[]
    for task_index,task in enumerate(TASKS):
        ref=frame[frame.model.eq("ecg")&frame.task.eq(task)].sort_values("patient_key"); keys=ref.patient_key.tolist(); y=ref.y.to_numpy(float); n=len(y); rng=np.random.default_rng(stable_seed(args.seed,"dca",task)); samples=rng.integers(0,n,size=(args.bootstrap_replicates,n),dtype=np.int32); weights=np.stack([np.bincount(x,minlength=n) for x in samples]).astype(np.float32); event_counts=weights@y.astype(np.float32); all_point=y.mean()-(1-y.mean())*thresholds/(1-thresholds); all_boot=event_counts[:,None]/n-(1-event_counts[:,None]/n)*thresholds[None,:]/(1-thresholds[None,:])
        for model in tqdm(MODELS,desc=f"DCA {task.upper()}",unit="model"):
            g=frame[frame.model.eq(model)&frame.task.eq(task)].set_index("patient_key").loc[keys]; p=g.calibrated.to_numpy(float); positive=p[:,None]>=thresholds; tp=positive*y[:,None]; fp=positive*(1-y)[:,None]; point=tp.sum(0)/n-fp.sum(0)/n*thresholds/(1-thresholds); boot=weights@tp.astype(np.float32)/n-(weights@fp.astype(np.float32))/n*thresholds[None,:]/(1-thresholds[None,:]); lo,hi=np.percentile(boot,[2.5,97.5],axis=0); da_lo,da_hi=np.percentile(boot-all_boot,[2.5,97.5],axis=0); dn_lo,dn_hi=np.percentile(boot,[2.5,97.5],axis=0)
            for i,t in enumerate(thresholds): rows.append({"outcome":task.upper(),"model":model,"model_label":MODEL_LABELS[model],"threshold":t,"net_benefit":point[i],"net_benefit_ci_lower":lo[i],"net_benefit_ci_upper":hi[i],"treat_all_net_benefit":all_point[i],"treat_none_net_benefit":0.,"difference_vs_treat_all":point[i]-all_point[i],"difference_vs_treat_all_ci_lower":da_lo[i],"difference_vs_treat_all_ci_upper":da_hi[i],"difference_vs_treat_none":point[i],"difference_vs_treat_none_ci_lower":dn_lo[i],"difference_vs_treat_none_ci_upper":dn_hi[i],"supported_over_both":bool(da_lo[i]>0 and dn_lo[i]>0)})
    data=pd.DataFrame(rows); ranges=[]
    for (outcome,model),g in data.groupby(["outcome","model"],sort=False):
        supported=g[g.supported_over_both].threshold.to_numpy(float)
        if not len(supported): ranges.append({"outcome":outcome,"model":model,"model_label":MODEL_LABELS[model],"range_index":0,"threshold_start":np.nan,"threshold_end":np.nan}); continue
        breaks=np.where(np.diff(supported)>args.threshold_step*1.5)[0]+1
        for i,part in enumerate(np.split(supported,breaks),1): ranges.append({"outcome":outcome,"model":model,"model_label":MODEL_LABELS[model],"range_index":i,"threshold_start":part[0],"threshold_end":part[-1]})
    ranges=pd.DataFrame(ranges); atomic_csv(directory/"decision_curve_data_with_pointwise_95ci.csv",data); atomic_csv(directory/"decision_curve_supported_ranges.csv",ranges)
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True); panels=(("Unimodal",MODELS[:4]),("Multimodal + ECG reference",("ecg",*MODELS[4:])))
    for r,task in enumerate(TASKS):
        for c,(title,models) in enumerate(panels):
            ax=axes[r,c]; ref=data[data.outcome.eq(task.upper())&data.model.eq("ecg")]; ax.axhline(0,color="#555",ls="--",label="Treat none"); ax.plot(ref.threshold,ref.treat_all_net_benefit,color="#888",ls=":",label="Treat all")
            for model in models:
                v=data[data.outcome.eq(task.upper())&data.model.eq(model)]; ax.plot(v.threshold,v.net_benefit,color=COLORS[model],ls=LINESTYLES[model],lw=1.6,label=MODEL_LABELS[model])
            ax.set(title=f"{task.upper()}: {title}",xlabel="Threshold probability",ylabel="Net benefit"); ax.grid(alpha=.18); ax.legend(frameon=False)
    fig.suptitle("Exploratory decision curves from calibrated outer-test probabilities",fontsize=13); fig.tight_layout(); save_figure(fig,directory,"principal_exploratory_decision_curves",args.dpi); return data,ranges


def plot_forest(table: pd.DataFrame, directory: Path, dpi: int) -> None:
    v=table[table.subgroup.eq("incremental_fusion")].copy(); fig,axes=plt.subplots(2,3,figsize=(15,10))
    for r,task in enumerate(TASKS):
        for c,metric in enumerate(METRICS):
            g=v[v.outcome.eq(task.upper())&v.metric.eq(metric)]; ax=axes[r,c]; pos=np.arange(len(g))[::-1]; labels=[f"{a} vs\n{b}" for a,b in zip(g.comparator_label,g.reference_label)]
            for p,point,lo,hi,sig in zip(pos,g.advantage_comparator,g.advantage_ci_lower,g.advantage_ci_upper,g.holm_significant_0_05): ax.errorbar(point,p,xerr=[[point-lo],[hi-point]],fmt="o",color="#0072B2" if sig else "#666",capsize=3)
            ax.axvline(0,color="#333",ls="--"); ax.set_yticks(pos,labels if c==0 else [""]*len(labels)); ax.set(title=f"{task.upper()}: {metric}",xlabel="Advantage of first model"); ax.grid(axis="x",alpha=.18)
    fig.suptitle("Incremental-fusion comparisons (95% CI; blue = Holm p < 0.05 across all 72 primary tests)",fontsize=12); fig.tight_layout(); save_figure(fig,directory,"incremental_fusion_difference_forest",dpi)


def provenance(args: argparse.Namespace) -> None:
    subject=pd.read_csv(args.subject_info_csv,sep=";",decimal=",",engine="python",nrows=1); prompts=pd.read_csv(args.prompt_csv,dtype={PATIENT_ID:"string"}); required=set(CONTINUOUS+CATEGORICAL+ECG_IMPRESSIONS)
    if required-set(subject): raise ValueError(f"Structured source lacks {sorted(required-set(subject))}")
    if len(prompts)!=args.expected_patients or not prompts[PATIENT_ID].is_unique: raise ValueError("Prompt cohort mismatch")
    rows=[]
    for group,features in (("continuous clinical",CONTINUOUS),("categorical clinical/medication",CATEGORICAL)):
        for feature in features: rows.append({"variable_group":group,"variable":feature,"tabular_no_ecg":"Yes","deterministic_text_no_ecg":"Yes","full_llm_no_ecg":"Yes","with_ecg_impressions":"Yes","raw_ecg_branch":"No","overlap":"No meaningful overlap","baseline_available":"Yes"})
    for feature in ECG_IMPRESSIONS: rows.append({"variable_group":"Holter ECG impression","variable":feature,"tabular_no_ecg":"No","deterministic_text_no_ecg":"No","full_llm_no_ecg":"No","with_ecg_impressions":"Yes","raw_ecg_branch":"Related waveform information, not the separate field","overlap":"Related physiological information","baseline_available":"Yes"})
    output=args.output_root/"provenance"; table=pd.DataFrame(rows); atomic_csv(output/"cross_representation_feature_provenance.csv",table)
    lines=["# Cross-representation feature provenance","","| Variable | Group | Tabular no ECG | Deterministic text | LLM no ECG | With ECG impressions | Raw ECG branch |","|---|---|---|---|---|---|---|"]
    for _,row in table.iterrows(): lines.append("| "+" | ".join(str(row[x]).replace("|","\\|") for x in ["variable","variable_group","tabular_no_ecg","deterministic_text_no_ecg","full_llm_no_ecg","with_ecg_impressions","raw_ecg_branch"])+" |")
    atomic_text(output/"cross_representation_feature_provenance.md","\n".join(lines)+"\n"); atomic_json(output/"feature_provenance_manifest.json",{"completed":True,"created_at_utc":utc_now(),"patients":len(prompts),"no_ecg_variables":len(CONTINUOUS)+len(CATEGORICAL),"holter_impressions":len(ECG_IMPRESSIONS),"subject_info_sha256":sha256_file(args.subject_info_csv),"prompt_csv_sha256":sha256_file(args.prompt_csv)})


def run_compare(args: argparse.Namespace) -> None:
    completion=args.output_root/"evaluation/comparative_analysis_manifest.json"
    if completion.exists() and not args.overwrite: print(f"Using completed comparative analysis: {completion}"); return
    folds=read_locked_folds(args); frames=[]; audits=[]
    for model,loader in [
        ("ecg",lambda:standardize_wide(args.ecg_predictions,"ecg")),
        ("tabular",lambda:standardize_wide(args.tabular_predictions,"tabular")),
        ("full_llm_text",lambda:standardize_text(args.text_predictions,"full_llm_text","full_risk_no_ecg")),
        ("deterministic_text",lambda:standardize_text(args.text_predictions,"deterministic_text","deterministic_template_no_ecg")),
        ("ecg_full_llm",lambda:standardize_multimodal(args.multimodal_predictions,"ecg_full_llm","ecg_full_text",args.multimodal_arm)),
        ("ecg_deterministic",lambda:standardize_multimodal(args.multimodal_predictions,"ecg_deterministic","ecg_deterministic_text",args.multimodal_arm)),
        ("ecg_tabular",lambda:standardize_multimodal(args.multimodal_predictions,"ecg_tabular","ecg_tabular",args.multimodal_arm)),
    ]:
        frame,audit=loader(); frames.append(frame); audits.append(audit)
    standardized=pd.concat(frames,ignore_index=True); validate_standardized(standardized,folds,args); evaluation=args.output_root/"evaluation"; figures=args.output_root/"figures"; set_style()
    atomic_csv(evaluation/"standardized_principal_outer_test_predictions.csv",standardized); atomic_csv(evaluation/"model_source_audit.csv",pd.DataFrame(audits))
    (perf,cal,thr,conf,thr_summary),reps=bootstrap_evaluation(standardized,args); primary=comparisons(perf,reps); exploratory=all_pairwise(perf,reps)
    for name,table in [("principal_models_performance_with_95ci.csv",perf),("principal_models_calibration_with_95ci.csv",cal),("principal_models_threshold_metrics_with_95ci.csv",thr),("principal_models_confusion_matrices.csv",conf),("saved_fold_specific_threshold_summary.csv",thr_summary),("primary_comparisons_with_95ci.csv",primary),("all_pairwise_exploratory_comparisons_with_95ci.csv",exploratory)]: atomic_csv(evaluation/name,table)
    formatted=perf.copy(); formatted["formatted"]=formatted.apply(lambda r:f"{r.estimate:.3f} ({r.ci_lower:.3f}-{r.ci_upper:.3f})",axis=1); manuscript=formatted.pivot(index=["model","model_label"],columns=["outcome","metric"],values="formatted").reindex(MODELS,level="model").reset_index(); manuscript.columns=["__".join(x for x in c if x) if isinstance(c,tuple) else c for c in manuscript.columns]; atomic_csv(evaluation/"manuscript_ready_principal_performance_table.csv",manuscript)
    atomic_csv(figures/"discrimination_curve_plot_data.csv",plot_discrimination(standardized,figures,args.dpi)); points=calibration_points(standardized,args.calibration_groups); atomic_csv(figures/"calibration_plot_points.csv",points); plot_calibration(points,figures,args.dpi); plot_forest(primary,figures,args.dpi); decision_curves(standardized,args,figures); provenance(args)
    atomic_json(figures/"figure_manifest.json",{"completed":True,"created_at_utc":utc_now(),"source_predictions_sha256":sha256_file(evaluation/"standardized_principal_outer_test_predictions.csv"),"calibration_groups":args.calibration_groups,"decision_curve_threshold_range":[args.threshold_min,args.threshold_max],"decision_curve_interpretation":"exploratory; no intervention-specific range prespecified","bootstrap_replicates":args.bootstrap_replicates})
    atomic_json(completion,{"completed":True,"created_at_utc":utc_now(),"analysis":"prediction-only endpoint-specific comparison","models":list(MODELS),"fixed_multimodal_arm":args.multimodal_arm,"eligible_patients":{"SCD":args.expected_controls+args.expected_scd,"PFD":args.expected_controls+args.expected_pfd},"events":{"SCD":args.expected_scd,"PFD":args.expected_pfd},"bootstrap_replicates":args.bootstrap_replicates,"seed":args.seed,"primary_comparison_family_tests":len(primary),"primary_multiplicity_adjustment":"Holm across all 72 prespecified model-outcome-metric comparisons","outer_test_outcomes_used_for_training_tuning_preprocessing_calibration_threshold_or_model_selection":False,"decision_curves":"exploratory pointwise paired-bootstrap analysis","script_sha256":sha256_file(Path(__file__).resolve()),"folds_sha256":sha256_file(args.folds_csv),"software":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__}})
    print(f"Comparative analysis complete: {args.output_root}"); print(manuscript.to_string(index=False))


# ---------------- Multimodal text attribution diagnostics ----------------

def mm_namespace(args: argparse.Namespace, mm) -> SimpleNamespace:
    return SimpleNamespace(stage="final",folds_csv=args.folds_csv,ecg_root=args.ecg_root,text_embedding_root=args.text_embedding_root,text_results_root=args.text_results_root,tabular_csv=args.subject_info_csv,output_root=args.multimodal_root,patient_id_col=mm.PATIENT_ID,scd_label_col=mm.SCD_LABEL,pfd_label_col=mm.PFD_LABEL,outer_fold_col=mm.OUTER_FOLD,outer_splits=5,inner_splits=4,expected_patients=args.expected_patients,expected_controls=args.expected_controls,expected_scd=args.expected_scd,expected_pfd=args.expected_pfd,expected_anticoagulant_yes=610,expected_anticoagulant_no=120,expected_text_pooling="cls",expected_text_max_length=args.max_length,expected_text_long_strategy="mean_chunks",expected_text_long_text_strategy="mean_chunks",expected_text_truncated_patient_count=0,expected_text_embedding_policy={"pooling":"cls","max_length":args.max_length,"long_text_strategy":"mean_chunks","truncated_patient_count":0})


def load_selected_multimodal(args, mm, task: str, fold: int, device):
    directory=args.multimodal_root/"tasks"/task/"final_models"/"ecg_full_text"/f"outer_fold_{fold}"; method=args.multimodal_arm; arm=directory/"arms"/method; checkpoint=torch_load(arm/"checkpoint.pt")
    model=mm.FusionNetwork(checkpoint["ecg_dim"],checkpoint["second_dim"],checkpoint["config"]); model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval(); return model,method,checkpoint


def torch_load(path: Path):
    import torch
    try: return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(path,map_location="cpu")


def response_text(frame: pd.DataFrame, task: str) -> pd.Series:
    risk=f"full_risk_no_ecg_{task}_risk"; rationale=f"full_risk_no_ecg_{task}_rationale"
    if risk not in frame or rationale not in frame: raise ValueError(f"Response CSV lacks {risk}/{rationale}")
    risk_text=frame[risk].fillna("").astype(str).str.strip()
    rationale_text=frame[rationale].fillna("").astype(str).str.strip()
    if risk_text.eq("").any() or rationale_text.eq("").any():
        raise ValueError(f"Empty endpoint risk/rationale text for {task}")
    text=task.upper()+"_RISK: "+risk_text.str.title()+"\n"+task.upper()+"_RATIONALE: "+rationale_text
    return text


def encoder_assets(args, mm, task: str, fold: int, source: str, encoder_name: str, condition: str, device):
    from transformers import AutoModel, AutoTokenizer
    directory=args.text_embedding_root/mm.stable_slug(source)/mm.stable_slug(encoder_name)/mm.stable_slug(condition); manifest=json.loads((directory/"manifest.json").read_text()); checkpoint=manifest.get("encoder_checkpoint") or {"BioBERT":"dmis-lab/biobert-base-cased-v1.1","ClinicalBERT":"emilyalsentzer/Bio_ClinicalBERT"}[encoder_name]; revision=manifest.get("resolved_commit") or manifest.get("requested_revision")
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,revision=revision,trust_remote_code=False); encoder=AutoModel.from_pretrained(checkpoint,revision=revision,trust_remote_code=False).to(device); encoder.requires_grad_(False); encoder.eval(); return tokenizer,encoder,manifest


def encode(tokenizer,text,max_length,chunk_stride,device):
    value=tokenizer(text,truncation=True,max_length=max_length,stride=chunk_stride,return_overflowing_tokens=True,return_tensors="pt",padding="max_length")
    return {k:v.to(device) for k,v in value.items() if k in {"input_ids","attention_mask","token_type_ids"}}


def multimodal_forward(encoder,model,encoded,ecg,mean,sd):
    import torch
    cls=encoder(**encoded).last_hidden_state[:,0,:].mean(dim=0,keepdim=True); second=(cls-mean)/sd; logit,_=model(ecg,second); return logit,torch.sigmoid(logit)


def integrated_gradients(encoder,model,encoded,ecg,mean,sd,steps,special_ids,pad_id):
    import torch
    layer=encoder.get_input_embeddings(); actual=layer(encoded["input_ids"]).detach(); baseline=layer(torch.full_like(encoded["input_ids"],pad_id)).detach(); delta=actual-baseline; total=torch.zeros_like(actual)
    for alpha in torch.linspace(0,1,steps,device=actual.device):
        value=(baseline+alpha*delta).detach().requires_grad_(True); kwargs={"inputs_embeds":value,"attention_mask":encoded["attention_mask"]};
        if "token_type_ids" in encoded: kwargs["token_type_ids"]=encoded["token_type_ids"]
        cls=encoder(**kwargs).last_hidden_state[:,0,:].mean(dim=0,keepdim=True); second=(cls-mean)/sd; logit,_=model(ecg,second); gradient=torch.autograd.grad(logit.sum(),value)[0]; total+=gradient.detach()
    attribution=(delta*total/steps).sum(-1).detach().cpu().numpy().astype(float).reshape(-1); ids=encoded["input_ids"].detach().cpu().numpy().reshape(-1); mask=encoded["attention_mask"].detach().cpu().numpy().astype(bool).reshape(-1)
    for i,token_id in enumerate(ids):
        if not mask[i] or int(token_id) in special_ids: attribution[i]=0
    with torch.no_grad(): probability=float(multimodal_forward(encoder,model,encoded,ecg,mean,sd)[1].item())
    return attribution,probability


def normalize_token(token: str) -> str: return token.replace("##","").replace("Ġ","").strip().lower()


def attribution_similarity(tokens_a,values_a,tokens_b,values_b,top=.2):
    def aggregate(tokens,values):
        d={}
        for token,value in zip(tokens,values):
            key=normalize_token(token)
            if key and re.search(r"[a-z0-9]",key): d[key]=d.get(key,0)+abs(float(value))
        return d
    a,b=aggregate(tokens_a,values_a),aggregate(tokens_b,values_b); keys=sorted(set(a)|set(b))
    if not keys:return {"spearman":np.nan,"cosine":np.nan,"top_k_jaccard":np.nan}
    x=np.array([a.get(k,0) for k in keys]); y=np.array([b.get(k,0) for k in keys]); denominator=np.linalg.norm(x)*np.linalg.norm(y); count=max(1,math.ceil(top*len(keys))); left=set(np.array(keys)[np.argsort(-x)[:count]]); right=set(np.array(keys)[np.argsort(-y)[:count]])
    return {"spearman":float(scipy.stats.spearmanr(x,y).statistic) if len(keys)>1 else np.nan,"cosine":float(x@y/denominator) if denominator else np.nan,"top_k_jaccard":len(left&right)/len(left|right) if left|right else np.nan}


def variants(text: str) -> dict[str,str]:
    lines=[x.strip() for x in text.splitlines() if x.strip()]; return {"whitespace":"  \n".join(lines),"punctuation":"\n".join(re.sub(r":\s*"," - ",x,count=1) for x in lines),"line_order":"\n".join(reversed(lines)),"neutral_prefix":"Clinical summary follows.\n"+"\n".join(lines)}


def eligible_positions(encoded,special_ids):
    ids=encoded["input_ids"].detach().cpu().numpy().reshape(-1); mask=encoded["attention_mask"].detach().cpu().numpy().astype(bool).reshape(-1); return np.array([i for i,(x,m) in enumerate(zip(ids,mask)) if m and int(x) not in special_ids],int)


def masked_probability(encoder,model,encoded,ecg,mean,sd,positions,replacement,keep_only=False,eligible=None):
    import torch
    ids=encoded["input_ids"].clone(); remove=np.setdiff1d(eligible,positions) if keep_only else positions
    if len(remove): ids.view(-1)[torch.as_tensor(remove,device=ids.device)]=replacement
    value={"input_ids":ids,"attention_mask":encoded["attention_mask"]}
    if "token_type_ids" in encoded: value["token_type_ids"]=encoded["token_type_ids"]
    with torch.no_grad(): return float(multimodal_forward(encoder,model,value,ecg,mean,sd)[1].item())


def run_attribution_fold(args: argparse.Namespace) -> None:
    import torch
    fold=int(args.outer_fold); output=args.output_root/"attribution_diagnostics"/"outer_folds"/f"outer_fold_{fold}"; completion=output/"run_complete.json"
    if completion.exists() and not args.overwrite: print(f"Using completed attribution fold {fold}"); return
    device=torch.device(args.device); mm=load_mm(args.multimodal_training_script); ns=mm_namespace(args,mm); folds=mm.read_folds(ns,setup_copy=False); response_paths={"LLaMA3.1-8B":args.llama8b_csv,"LLaMA3.2-3B":args.llama3b_csv}; pooled=pd.read_csv(args.multimodal_predictions,dtype={PATIENT_ID:"string"})
    stability_rows=[]; perturb_rows=[]; randomization_rows=[]; patient_rows=[]; token_records=[]; reproduction_differences={}
    for task in TASKS:
        data=mm.load_pair_split(ns,folds,"ecg_full_text",task,fold,None,None); source,encoder_name,condition=mm.selected_text_spec(ns,fold,"ecg_full_text",task); response=pd.read_csv(response_paths[source],dtype={PATIENT_ID:"string"}); response["patient_key"]=response[PATIENT_ID].map(canonical_id); response["analysis_text"]=response_text(response,task); response=response.set_index("patient_key")
        model,method,checkpoint=load_selected_multimodal(args,mm,task,fold,device); tokenizer,encoder,encoder_manifest=encoder_assets(args,mm,task,fold,source,encoder_name,condition,device); special=set(map(int,tokenizer.all_special_ids)); pad=tokenizer.pad_token_id; replacement=tokenizer.mask_token_id if tokenizer.mask_token_id is not None else pad
        mean=torch.as_tensor(data.preprocessing["second_mean"],dtype=torch.float32,device=device).unsqueeze(0); sd=torch.as_tensor(data.preprocessing["second_sd"],dtype=torch.float32,device=device).unsqueeze(0)
        saved=pooled[pooled.task.eq(task)&pooled.modality_pair.eq("ecg_full_text")&pooled.arm.eq(args.multimodal_arm)&pooled.outer_fold.eq(fold)].copy(); saved["patient_key"]=saved[PATIENT_ID].map(canonical_id); saved=saved.set_index("patient_key").loc[[canonical_id(x) for x in data.patient_ids_validation]]
        predicted,_=mm.predict_model(model,data.ecg_validation,data.second_validation,device,1024); max_diff=float(np.max(np.abs(predicted-saved.prob.to_numpy(float))))
        reproduction_differences[task.upper()]=max_diff
        if max_diff>1e-5: raise RuntimeError(f"Checkpoint mismatch {task}/fold {fold}: {max_diff}")
        ids=np.array([canonical_id(x) for x in data.patient_ids_validation]); labels=data.labels_validation.astype(int); rng=np.random.default_rng(stable_seed(args.seed,"attribute-sample",task,fold)); chosen=[]
        for label in (0,1):
            positions=np.flatnonzero(labels==label); chosen.extend(rng.choice(positions,size=min(args.samples_per_class,len(positions)),replace=False).tolist())
        refits=[]
        for seed in args.refit_seeds: refits.append((seed,mm.fit_fixed_epochs(data,checkpoint["config"],int(checkpoint["final_epochs"]),device,seed,f"Refit {task.upper()} fold {fold} seed {seed}")))
        random_model=mm.FusionNetwork(checkpoint["ecg_dim"],checkpoint["second_dim"],checkpoint["config"]).to(device).eval()
        for position in tqdm(chosen,desc=f"Attribute {task.upper()} fold {fold}",unit="patient"):
            key=ids[position]; text=response.loc[key,"analysis_text"]; encoded=encode(tokenizer,text,args.max_length,args.chunk_stride,device); ecg=torch.as_tensor(data.ecg_validation[position:position+1],dtype=torch.float32,device=device); tokens=tokenizer.convert_ids_to_tokens(encoded["input_ids"].reshape(-1).tolist()); reference,prob=integrated_gradients(encoder,model,encoded,ecg,mean,sd,args.ig_steps,special,pad)
            saved_probability=float(saved.loc[key,"prob"])
            if abs(prob-saved_probability)>1e-5: raise RuntimeError(f"Chunked text attribution does not reproduce saved probability for {task}/fold {fold}/{key}: {prob} vs {saved_probability}")
            token_records.append({"patient_key":key,"outer_fold":fold,"outcome":task.upper(),"label":int(labels[position]),"selected_method":method,"source":source,"encoder":encoder_name,"tokens":tokens,"signed_integrated_gradients":reference.tolist(),"absolute_integrated_gradients":np.abs(reference).tolist(),"raw_probability":prob,"text_chunks":int(encoded["input_ids"].shape[0])})
            patient_rows.append({"patient_key":key,"outer_fold":fold,"outcome":task.upper(),"label":int(labels[position]),"raw_probability":prob,"selected_method":method,"source":source,"encoder":encoder_name})
            for name,value in variants(text).items():
                enc=encode(tokenizer,value,args.max_length,args.chunk_stride,device); attr,p=integrated_gradients(encoder,model,enc,ecg,mean,sd,args.ig_steps,special,pad); metrics=attribution_similarity(tokens,reference,tokenizer.convert_ids_to_tokens(enc["input_ids"].reshape(-1).tolist()),attr); stability_rows.append({"patient_key":key,"outer_fold":fold,"outcome":task.upper(),"stability_type":"meaning_preserving_input","replicate":name,"probability_reference":prob,"probability_comparison":p,"absolute_probability_change":abs(p-prob),**metrics})
            for steps in sorted(set([max(8,args.ig_steps//2),args.ig_steps*2])):
                attr,p=integrated_gradients(encoder,model,encoded,ecg,mean,sd,steps,special,pad); stability_rows.append({"patient_key":key,"outer_fold":fold,"outcome":task.upper(),"stability_type":"integration_steps","replicate":steps,"probability_reference":prob,"probability_comparison":p,"absolute_probability_change":abs(p-prob),**attribution_similarity(tokens,reference,tokens,attr)})
            for seed,refit in refits:
                attr,p=integrated_gradients(encoder,refit,encoded,ecg,mean,sd,args.ig_steps,special,pad); stability_rows.append({"patient_key":key,"outer_fold":fold,"outcome":task.upper(),"stability_type":"outer_training_refit_seed","replicate":seed,"probability_reference":prob,"probability_comparison":p,"absolute_probability_change":abs(p-prob),**attribution_similarity(tokens,reference,tokens,attr)})
            randomized,p_random=integrated_gradients(encoder,random_model,encoded,ecg,mean,sd,args.ig_steps,special,pad); randomization_rows.append({"patient_key":key,"outer_fold":fold,"outcome":task.upper(),"spearman":attribution_similarity(tokens,reference,tokens,randomized)["spearman"],"cosine":attribution_similarity(tokens,reference,tokens,randomized)["cosine"],"top_k_jaccard":attribution_similarity(tokens,reference,tokens,randomized)["top_k_jaccard"],"trained_probability":prob,"randomized_probability":p_random})
            eligible=eligible_positions(encoded,special); magnitude=np.abs(reference); ordered=eligible[np.argsort(-magnitude[eligible])]; low=eligible[np.argsort(magnitude[eligible])]
            for fraction in args.mask_fractions:
                count=max(1,math.ceil(fraction*len(eligible))); high=ordered[:count]; low_pos=low[:count]; original=prob; high_p=masked_probability(encoder,model,encoded,ecg,mean,sd,high,replacement); low_p=masked_probability(encoder,model,encoded,ecg,mean,sd,low_pos,replacement); sufficient=masked_probability(encoder,model,encoded,ecg,mean,sd,high,replacement,True,eligible)
                for control,p_value in [("high_attribution",high_p),("low_attribution",low_p)]: perturb_rows.append({"patient_key":key,"outer_fold":fold,"outcome":task.upper(),"fraction":fraction,"control":control,"repeat":0,"original_probability":original,"perturbed_probability":p_value,"absolute_probability_change":abs(original-p_value),"comprehensiveness":original-p_value,"sufficiency":np.nan if control!="high_attribution" else original-sufficient})
                for repeat in range(args.random_control_repeats):
                    random_pos=rng.choice(eligible,size=count,replace=False); p_value=masked_probability(encoder,model,encoded,ecg,mean,sd,random_pos,replacement); perturb_rows.append({"patient_key":key,"outer_fold":fold,"outcome":task.upper(),"fraction":fraction,"control":"random","repeat":repeat,"original_probability":original,"perturbed_probability":p_value,"absolute_probability_change":abs(original-p_value),"comprehensiveness":original-p_value,"sufficiency":np.nan})
        del model,encoder,refits,random_model
        if device.type=="cuda": torch.cuda.empty_cache()
    output.mkdir(parents=True,exist_ok=True); atomic_csv(output/"attribution_stability.csv",pd.DataFrame(stability_rows)); atomic_csv(output/"token_perturbation_fidelity.csv",pd.DataFrame(perturb_rows)); atomic_csv(output/"classifier_randomization.csv",pd.DataFrame(randomization_rows)); atomic_csv(output/"attributed_outer_test_predictions.csv",pd.DataFrame(patient_rows))
    with gzip.open(output/"token_attributions.jsonl.gz","wt",encoding="utf-8") as h:
        for record in token_records: h.write(json.dumps(record)+"\n")
    atomic_json(completion,{"completed":True,"created_at_utc":utc_now(),"outer_fold":fold,"patients_attributed":len(patient_rows),"tasks":list(TASKS),"pair":"ecg_full_text","checkpoint_reproduction_max_absolute_probability_difference_by_task":reproduction_differences,"attribution_method":"integrated gradients through frozen encoder and selected multimodal checkpoint","raw_attention_used":False,"clinical_explanation_claimed":False})
    print(f"Attribution fold {fold} complete: {output}")


def bootstrap_mean(values: np.ndarray, reps: int, seed: int):
    values=np.asarray(values,float); values=values[np.isfinite(values)]; rng=np.random.default_rng(seed); means=np.mean(values[rng.integers(0,len(values),size=(reps,len(values)))],axis=1); lo,hi=ci(means); return float(values.mean()),lo,hi


def merge_wordpieces(tokens,values):
    words=[]; scores=[]
    for token,value in zip(tokens,values):
        if token in {"[CLS]","[SEP]","[PAD]","<s>","</s>"}: continue
        if token.startswith("##") and words: words[-1]+=token[2:]; scores[-1]+=value
        else: words.append(token.replace("Ġ","").replace("▁"," ").strip()); scores.append(value)
    return words,np.asarray(scores,float)


def highlighted_figure(record,case_id):
    words,scores=merge_wordpieces(record["tokens"],record["signed_integrated_gradients"]); scale=np.percentile(np.abs(scores),95) if len(scores) else 1; scale=max(scale,1e-12); fig,ax=plt.subplots(figsize=(12,4.2)); ax.axis("off"); x=.02;y=.92;renderer=fig.canvas.get_renderer()
    for word,score in zip(words,scores):
        label=word+" "; color=(1,.25,.25,min(.85,abs(score)/scale*.75+.08)) if score>=0 else (.2,.4,1,min(.85,abs(score)/scale*.75+.08)); text=ax.text(x,y,label,transform=ax.transAxes,fontsize=10,bbox={"facecolor":color,"edgecolor":"none","pad":1.5},va="top"); fig.canvas.draw(); box=text.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted()); width=box.width
        if x+width>.98: text.remove(); x=.02;y-=.095; text=ax.text(x,y,label,transform=ax.transAxes,fontsize=10,bbox={"facecolor":color,"edgecolor":"none","pad":1.5},va="top"); fig.canvas.draw(); box=text.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted()); width=box.width
        x+=width+.006
    ax.set_title(f"{case_id}: {record['outcome']} model-behavior attribution (red increases, blue decreases)",loc="left",fontsize=11); return fig


def aggregate_attribution(args: argparse.Namespace) -> None:
    root=args.output_root/"attribution_diagnostics"; evaluation=root/"evaluation"; completion=evaluation/"attribution_analysis_manifest.json"
    if completion.exists() and not args.overwrite: print(f"Using completed attribution aggregation: {completion}"); return
    stability=[];perturb=[];randomized=[];patients=[];records=[]
    for fold in range(5):
        directory=root/"outer_folds"/f"outer_fold_{fold}"
        if not (directory/"run_complete.json").exists(): raise FileNotFoundError(directory/"run_complete.json")
        stability.append(pd.read_csv(directory/"attribution_stability.csv")); perturb.append(pd.read_csv(directory/"token_perturbation_fidelity.csv")); randomized.append(pd.read_csv(directory/"classifier_randomization.csv")); patients.append(pd.read_csv(directory/"attributed_outer_test_predictions.csv"))
        with gzip.open(directory/"token_attributions.jsonl.gz","rt",encoding="utf-8") as h: records.extend(json.loads(line) for line in h)
    stability=pd.concat(stability,ignore_index=True); perturb=pd.concat(perturb,ignore_index=True); randomized=pd.concat(randomized,ignore_index=True); patients=pd.concat(patients,ignore_index=True); evaluation.mkdir(parents=True,exist_ok=True)
    atomic_csv(evaluation/"all_patient_stability_results.csv",stability); atomic_csv(evaluation/"all_patient_perturbation_results.csv",perturb); atomic_csv(evaluation/"all_patient_randomization_results.csv",randomized); atomic_csv(evaluation/"attributed_outer_test_predictions.csv",patients)
    summary=[]
    for keys,g in stability.groupby(["outcome","stability_type"]):
        for metric in ["spearman","cosine","top_k_jaccard","absolute_probability_change"]:
            estimate,lo,hi=bootstrap_mean(g[metric].to_numpy(float),args.attribution_bootstrap_replicates,stable_seed(args.seed,"attr",*keys,metric)); summary.append({"outcome":keys[0],"stability_type":keys[1],"metric":metric,"estimate":estimate,"ci_lower":lo,"ci_upper":hi,"records":len(g),"patients":g.patient_key.nunique()})
    atomic_csv(evaluation/"attribution_stability_summary_with_95ci.csv",pd.DataFrame(summary))
    patient_perturb=perturb.groupby(["patient_key","outcome","fraction","control"],as_index=False).absolute_probability_change.mean(); paired=[]
    for (outcome,fraction),g in patient_perturb.groupby(["outcome","fraction"]):
        pivot=g.pivot(index="patient_key",columns="control",values="absolute_probability_change").dropna(subset=["high_attribution","random"]); diff=(pivot.high_attribution-pivot.random).to_numpy(); estimate,lo,hi=bootstrap_mean(diff,args.attribution_bootstrap_replicates,stable_seed(args.seed,"mask",outcome,fraction)); paired.append({"outcome":outcome,"fraction":fraction,"comparison":"high_attribution_minus_random","mean_difference":estimate,"ci_lower":lo,"ci_upper":hi,"patients":len(diff)})
    paired=pd.DataFrame(paired); atomic_csv(evaluation/"high_attribution_vs_random_paired_summary.csv",paired)
    random_summary=[]
    for outcome,g in randomized.groupby("outcome"):
        for metric in ["spearman","cosine","top_k_jaccard"]:
            estimate,lo,hi=bootstrap_mean(g[metric].to_numpy(),args.attribution_bootstrap_replicates,stable_seed(args.seed,"random",outcome,metric)); random_summary.append({"outcome":outcome,"metric":metric,"estimate":estimate,"ci_lower":lo,"ci_upper":hi,"patients":g.patient_key.nunique()})
    atomic_csv(evaluation/"classifier_randomization_summary_with_95ci.csv",pd.DataFrame(random_summary))
    figures=root/"figures"; set_style(); fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    stab=pd.DataFrame(summary)
    for outcome,color in [("SCD","#0072B2"),("PFD","#D55E00")]:
        v=stab[stab.outcome.eq(outcome)&stab.metric.eq("top_k_jaccard")]; axes[0].plot(v.stability_type,v.estimate,"o-",label=outcome,color=color); p=paired[paired.outcome.eq(outcome)]; axes[1].plot(p.fraction,p.mean_difference,"o-",label=outcome,color=color)
    axes[0].set(title="Top-token attribution stability",ylabel="Top-k Jaccard",xlabel="Stability control"); axes[0].tick_params(axis="x",rotation=25); axes[1].axhline(0,color="#555",ls="--"); axes[1].set(title="High-attribution versus random masking",ylabel="Difference in absolute probability change",xlabel="Masked token fraction")
    for ax in axes: ax.grid(alpha=.18); ax.legend(frameon=False)
    fig.tight_layout(); save_figure(fig,figures,"attribution_stability_and_perturbation",args.dpi)
    highlight=figures/"highlighted_responses"; highlight.mkdir(parents=True,exist_ok=True); selection=[]
    for outcome in ("SCD","PFD"):
        for label in (0,1):
            candidates=[r for r in records if r["outcome"]==outcome and int(r["label"])==label]; candidates=sorted(candidates,key=lambda r:abs(r["raw_probability"]-.5),reverse=True)[:2]
            for index,record in enumerate(candidates,1):
                case_id=f"{outcome}-{('event' if label else 'control')}-{index}"; fig=highlighted_figure(record,case_id); save_figure(fig,highlight,case_id.lower(),220); selection.append({"case_id":case_id,"outcome":outcome,"label":label,"outer_fold":record["outer_fold"],"selection":"largest absolute distance from 0.5 among attributed cases; de-identified"})
    atomic_csv(highlight/"highlighted_case_selection.csv",pd.DataFrame(selection))
    atomic_json(completion,{"completed":True,"created_at_utc":utc_now(),"pair":"ecg_full_text","tasks":list(TASKS),"folds":5,"patients_attributed":patients.groupby("outcome").patient_key.nunique().to_dict(),"fusion_arm":args.multimodal_arm,"method":"Integrated gradients through the frozen encoder mean-of-chunk CLS representation and the fixed direct-concatenation multimodal checkpoint","stability_controls":["outer-training refit seeds","meaning-preserving input variants","integration-step sensitivity"],"perturbation_controls":["high-attribution masking","low-attribution masking","random masking","classifier randomization"],"raw_attention_presented_as_explanation":False,"clinician_review_available":False,"clinical_validity_claimed":False,"interpretation":"Exploratory technical stability and model-linked fidelity only; highlighted text is not a validated clinical explanation."})
    print(f"Attribution aggregation complete: {root}")


def preflight(args: argparse.Namespace) -> None:
    required=[args.folds_csv,args.ecg_predictions,args.tabular_predictions,args.text_predictions,args.multimodal_predictions,args.subject_info_csv,args.prompt_csv,args.multimodal_training_script,args.llama8b_csv,args.llama3b_csv]
    for path in required:
        if not path.exists(): raise FileNotFoundError(path)
    if sha256_file(args.prompt_csv) != EXPECTED_PROMPT_SHA256:
        raise ValueError("Prompt CSV checksum does not match the locked detailed-response prompt file.")
    for response_path in (args.llama8b_csv, args.llama3b_csv):
        raw_stem = response_path.stem.removesuffix("-postprocessed")
        manifest_path = response_path.with_name(raw_stem + ".manifest.json")
        if not manifest_path.exists(): raise FileNotFoundError(manifest_path)
        response_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if response_manifest.get("input_csv_sha256") != EXPECTED_PROMPT_SHA256:
            raise ValueError(f"Response manifest uses the wrong prompt CSV: {manifest_path}")
        if int(response_manifest.get("patient_count", -1)) != args.expected_patients:
            raise ValueError(f"Response manifest patient count mismatch: {manifest_path}")
    folds=read_locked_folds(args); mm=load_mm(args.multimodal_training_script); ns=mm_namespace(args,mm); mm_folds=mm.read_folds(ns,setup_copy=False); rows=[]
    for task in TASKS:
        for fold in range(5):
            source,encoder,condition=mm.selected_text_spec(ns,fold,"ecg_full_text",task); directory=args.multimodal_root/"tasks"/task/"final_models"/"ecg_full_text"/f"outer_fold_{fold}"; method=args.multimodal_arm; checkpoint=directory/"arms"/method/"checkpoint.pt"; embedding=args.text_embedding_root/mm.stable_slug(source)/mm.stable_slug(encoder)/mm.stable_slug(condition)/"embeddings.npz"
            embedding_manifest=embedding.with_name("manifest.json")
            for path in [checkpoint,embedding,embedding_manifest]:
                if not path.exists(): raise FileNotFoundError(path)
            metadata=json.loads(embedding_manifest.read_text(encoding="utf-8"))
            observed_policy={"pooling":metadata.get("pooling"),"max_length":metadata.get("max_length"),"long_text_strategy":metadata.get("long_text_strategy"),"truncated_patient_count":metadata.get("truncated_patient_count")}
            expected_policy=ns.expected_text_embedding_policy
            if observed_policy != expected_policy: raise ValueError(f"Embedding policy mismatch in {embedding_manifest}: {observed_policy} vs {expected_policy}")
            rows.append({"task":task,"outer_fold":fold,"source":source,"encoder":encoder,"condition":condition,"fusion_method":method,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint),"embedding":str(embedding),"embedding_sha256":sha256_file(embedding)})
    setup=args.output_root/"analysis_setup"; atomic_csv(setup/"attribution_preflight.csv",pd.DataFrame(rows)); atomic_json(setup/"preflight_manifest.json",{"completed":True,"created_at_utc":utc_now(),"patients":len(folds),"eligible_patients":{"SCD":args.expected_controls+args.expected_scd,"PFD":args.expected_controls+args.expected_pfd},"independent_binary_tasks":True,"competing_endpoints_excluded":True,"multimodal_attribution_pair":"ecg_full_text","raw_attention_used":False,"folds_sha256":sha256_file(args.folds_csv),"script_sha256":sha256_file(Path(__file__).resolve())}); print(f"Preflight complete: {setup}")


def main() -> None:
    args=parse_args()
    if args.stage=="preflight": preflight(args)
    elif args.stage=="compare": run_compare(args)
    elif args.stage=="attribute-fold": run_attribution_fold(args)
    else: aggregate_attribution(args)


if __name__ == "__main__":
    main()
