#!/usr/bin/env python3
"""Architecture-matched tabular MLP sensitivity analysis for MUSIC.

The candidate MLPs exactly mirror the six MLP downstream configurations used
in the text-embedding analysis. All preprocessing, selection, early stopping,
calibration, and threshold selection remain inside each outer-training cohort.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


PATIENT_ID = "Patient ID"
SCD_LABEL = "SCD_4year_label"
PFD_LABEL = "PFD_4year_label"
OUTER_FOLD = "outer_fold"
TASKS = ("scd", "pfd")

CONTINUOUS_FEATURES = [
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

ANTICOAGULANT_FEATURE = "Anticoagulants/antitrombotics  (yes=1)"
CATEGORICAL_FEATURES = [
    "Gender (male=1)", "NYHA class", "HF etiology - Diagnosis",
    "Diabetes (yes=1)", "History of dyslipemia (yes=1)",
    "Peripheral vascular disease (yes=1)", "History of hypertension (yes=1)",
    "Prior Myocardial Infarction (yes=1)", "Calcium channel blocker (yes=1)",
    "Diabetes medication (yes=1)", "Amiodarone (yes=1)",
    "Angiotensin-II receptor blocker (yes=1)", ANTICOAGULANT_FEATURE,
    "Betablockers (yes=1)", "Digoxin (yes=1)", "Loop diuretics (yes=1)",
    "Spironolactone (yes=1)", "Statins (yes=1)", "Hidralazina (yes=1)",
    "ACE inhibitor (yes=1)", "Nitrovasodilator (yes=1)",
]

MLP_CANDIDATES = (
    {"name": "mlp_h128_l1_d0p1_lr1e3_wd1e5", "hidden_dim": 128, "layers": 1, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5},
    {"name": "mlp_h128_l2_d0p2_lr3e4_wd1e5", "hidden_dim": 128, "layers": 2, "dropout": 0.2, "lr": 3e-4, "weight_decay": 1e-5},
    {"name": "mlp_h256_l1_d0p2_lr1e3_wd1e5", "hidden_dim": 256, "layers": 1, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5},
    {"name": "mlp_h256_l2_d0p2_lr3e4_wd1e5", "hidden_dim": 256, "layers": 2, "dropout": 0.2, "lr": 3e-4, "weight_decay": 1e-5},
    {"name": "mlp_h256_l2_d0p5_lr1e4_wd1e4", "hidden_dim": 256, "layers": 2, "dropout": 0.5, "lr": 1e-4, "weight_decay": 1e-4},
    {"name": "mlp_h512_l2_d0p2_lr1e4_wd1e5", "hidden_dim": 512, "layers": 2, "dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-5},
)


def parse_args() -> argparse.Namespace:
    music = Path("/home/sswee/music")
    parser = argparse.ArgumentParser(description="Nested architecture-matched tabular MLP sensitivity analysis.")
    parser.add_argument("--stage", required=True, choices=["prepare", "fit-fold", "aggregate"])
    parser.add_argument("--tabular_csv", type=Path, default=music / "subject-info.csv")
    parser.add_argument("--folds_csv", type=Path, default=music / "ecg_nested_4year_three_wave/analysis_setup/nested_patient_folds.csv")
    parser.add_argument("--output_root", type=Path, default=music / "tabular_mlp_matched_4year_v2")
    parser.add_argument("--tabular_reference", type=Path, default=music / "tabular_nested_4year_v2/evaluation/models/prompt_matched_no_ecg/selected_tabular/pooled_predictions_calibrated_and_classified.csv")
    parser.add_argument("--text_reference", type=Path, default=music / "text_nested_4year_v2/evaluation/all_arms_pooled_predictions_calibrated_and_classified.csv")
    parser.add_argument("--outer_fold", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--auc_tolerance", type=float, default=0.005)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_patients", type=int, default=730)
    parser.add_argument("--expected_controls", type=int, default=577)
    parser.add_argument("--expected_scd", type=int, default=71)
    parser.add_argument("--expected_pfd", type=int, default=82)
    parser.add_argument("--expected_anticoagulant_yes", type=int, default=610)
    parser.add_argument("--expected_anticoagulant_no", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *map(str, parts)])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def canonical_id(value: object) -> str:
    """Match the patient-ID normalization used by the successful tabular benchmark."""
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    if digits:
        return str(int(digits))
    return text.casefold()


def read_source(path: Path) -> pd.DataFrame:
    # The original MUSIC table is semicolon-delimited and uses decimal commas.
    frame = pd.read_csv(
        path, sep=";", decimal=",", engine="python",
        na_values=["", "NA", "N/A", "nan"], keep_default_na=True,
    )
    return frame.replace(r"\t", ".", regex=True)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    for path in (args.tabular_csv, args.folds_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = read_source(args.tabular_csv)
    folds = pd.read_csv(args.folds_csv, dtype={PATIENT_ID: "string"})
    required_folds = {PATIENT_ID, SCD_LABEL, PFD_LABEL, OUTER_FOLD, *[f"inner_fold_outer_{fold}" for fold in range(args.outer_splits)]}
    missing = required_folds - set(folds.columns)
    if missing:
        raise ValueError(f"Nested folds file lacks {sorted(missing)}")
    features = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    missing = set([PATIENT_ID, *features]) - set(source.columns)
    if missing:
        raise ValueError(f"Tabular source lacks {sorted(missing)}")
    source = source[[PATIENT_ID, *features]].copy()
    source["patient_key"] = source[PATIENT_ID].map(canonical_id)
    folds["patient_key"] = folds[PATIENT_ID].map(canonical_id)
    if source["patient_key"].duplicated().any() or folds["patient_key"].duplicated().any():
        raise ValueError("Canonical patient IDs are not unique.")
    frame = folds.merge(source.drop(columns=PATIENT_ID), on="patient_key", how="left", validate="one_to_one", indicator=True)
    if not frame["_merge"].eq("both").all():
        unmatched = frame.loc[frame["_merge"].ne("both"), [PATIENT_ID, "patient_key"]]
        source_examples = source[[PATIENT_ID, "patient_key"]].head(10).to_dict("records")
        raise ValueError(
            "At least one locked patient is absent from the tabular source after "
            f"canonicalization. Unmatched locked IDs: {unmatched.head(20).to_dict('records')}. "
            f"First source IDs: {source_examples}. Source file: {args.tabular_csv}"
        )
    frame = frame.drop(columns="_merge")
    for feature in CONTINUOUS_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    for feature in CATEGORICAL_FEATURES:
        frame[feature] = frame[feature].map(lambda value: np.nan if pd.isna(value) else str(value).strip())

    anticoagulant = pd.to_numeric(frame[ANTICOAGULANT_FEATURE], errors="coerce")
    anticoagulant_counts = (int(anticoagulant.eq(1).sum()), int(anticoagulant.eq(0).sum()))
    expected_anticoagulant = (args.expected_anticoagulant_yes, args.expected_anticoagulant_no)
    if anticoagulant_counts != expected_anticoagulant or anticoagulant.isna().any():
        raise ValueError(
            f"Unexpected anticoagulant encoding/counts: yes/no={anticoagulant_counts}, "
            f"missing={int(anticoagulant.isna().sum())}; expected {expected_anticoagulant} and no missing."
        )

    scd = pd.to_numeric(frame[SCD_LABEL], errors="coerce")
    pfd = pd.to_numeric(frame[PFD_LABEL], errors="coerce")
    scd_event, pfd_event = scd.eq(1), pfd.eq(1)
    controls = scd.eq(0) & pfd.eq(0)
    assigned = controls.astype(int) + scd_event.astype(int) + pfd_event.astype(int)
    observed = (len(frame), int(controls.sum()), int(scd_event.sum()), int(pfd_event.sum()))
    expected = (args.expected_patients, args.expected_controls, args.expected_scd, args.expected_pfd)
    if observed != expected or not assigned.eq(1).all():
        raise ValueError(f"Unexpected cohort encoding/counts: {observed}; expected {expected}")
    frame[SCD_LABEL] = np.where(scd_event, 1.0, np.where(controls, 0.0, np.nan))
    frame[PFD_LABEL] = np.where(pfd_event, 1.0, np.where(controls, 0.0, np.nan))
    frame[OUTER_FOLD] = pd.to_numeric(frame[OUTER_FOLD], errors="raise").astype(int)
    for outer_fold in range(args.outer_splits):
        column = f"inner_fold_outer_{outer_fold}"
        train = frame.loc[frame[OUTER_FOLD].ne(outer_fold)]
        test = frame.loc[frame[OUTER_FOLD].eq(outer_fold)]
        if train[column].isna().any() or test[column].notna().any():
            raise ValueError(f"Invalid nested assignments in {column}")
        if sorted(train[column].astype(int).unique()) != list(range(args.inner_splits)):
            raise ValueError(f"Incomplete inner folds in {column}")
    return frame


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    continuous = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("one_hot", one_hot_encoder()),
    ])
    return ColumnTransformer([
        ("continuous", continuous, CONTINUOUS_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop", sparse_threshold=0)


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[[SCD_LABEL, PFD_LABEL]].to_numpy(float)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(current, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.scd_head = nn.Linear(current, 1)
        self.pfd_head = nn.Linear(current, 1)

    def forward(self, values: torch.Tensor):
        representation = self.trunk(values)
        return self.scd_head(representation).squeeze(-1), self.pfd_head(representation).squeeze(-1)


def parameter_count(input_dim: int, config: dict) -> int:
    current, total = input_dim, 0
    for _ in range(config["layers"]):
        total += current * config["hidden_dim"] + config["hidden_dim"]
        current = config["hidden_dim"]
    return total + 2 * (current + 1)


def pos_weights(y: np.ndarray, device: torch.device):
    result = []
    for index in range(2):
        known = np.isfinite(y[:, index])
        positives = int(np.sum(y[known, index] == 1))
        negatives = int(np.sum(y[known, index] == 0))
        if not positives or not negatives:
            raise ValueError("A training split lacks an outcome class.")
        result.append(torch.tensor(negatives / positives, dtype=torch.float32, device=device))
    return result


def loss_value(logits, y_tensor, weights):
    losses = []
    for index in range(2):
        known = torch.isfinite(y_tensor[:, index])
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[index])
        losses.append(criterion(logits[index][known], y_tensor[known, index]))
    return losses[0] + losses[1]


@torch.inference_mode()
def predict(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(values.astype(np.float32)).to(device)
    output = model(tensor)
    return np.column_stack([torch.sigmoid(value).cpu().numpy() for value in output])


def metric_values(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    values = {}
    for index, task in enumerate(TASKS):
        known = np.isfinite(y[:, index])
        task_y = y[known, index].astype(int)
        task_p = probabilities[known, index]
        values[f"{task}_roc_auc"] = roc_auc_score(task_y, task_p)
        values[f"{task}_pr_auc"] = average_precision_score(task_y, task_p)
    values["mean_roc_auc"] = np.mean([values["scd_roc_auc"], values["pfd_roc_auc"]])
    values["mean_pr_auc"] = np.mean([values["scd_pr_auc"], values["pfd_pr_auc"]])
    return {key: float(value) for key, value in values.items()}


def fit_mlp(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray | None,
            validation_y: np.ndarray | None, config: dict, device: torch.device,
            seed: int, epochs: int, min_epochs: int, patience: int,
            fixed_epochs: int | None = None):
    set_seed(seed)
    model = MultiTaskMLP(train_x.shape[1], config["hidden_dim"], config["layers"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    x_tensor = torch.from_numpy(train_x.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(train_y.astype(np.float32)).to(device)
    weights = pos_weights(train_y, device)
    maximum = fixed_epochs or epochs
    best_score, best_epoch, best_state, stale = -np.inf, maximum, None, 0
    history = []
    for epoch in range(1, maximum + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value(model(x_tensor), y_tensor, weights)
        loss.backward()
        optimizer.step()
        row = {"epoch": epoch, "loss": float(loss.detach().cpu())}
        if validation_x is not None:
            probabilities = predict(model, validation_x, device)
            metrics = metric_values(validation_y, probabilities)
            row.update(metrics)
            if metrics["mean_roc_auc"] > best_score + 1e-12:
                best_score, best_epoch = metrics["mean_roc_auc"], epoch
                best_state, stale = copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if epoch >= min_epochs and stale >= patience:
                history.append(row)
                break
        history.append(row)
    if validation_x is not None:
        if best_state is None:
            raise RuntimeError("No best MLP state was recorded.")
        model.load_state_dict(best_state)
    return model, best_epoch, history


def prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray, outer_fold: int, inner_fold: int | None):
    return pd.DataFrame({
        PATIENT_ID: frame[PATIENT_ID].astype(str).to_numpy(),
        "patient_key": frame["patient_key"].to_numpy(),
        "outer_fold": outer_fold, "inner_fold": inner_fold,
        "scd_label": frame[SCD_LABEL].to_numpy(float), "pfd_label": frame[PFD_LABEL].to_numpy(float),
        "scd_probability": probabilities[:, 0], "pfd_probability": probabilities[:, 1],
    })


def stage_prepare(args: argparse.Namespace) -> None:
    frame = load_data(args)
    setup = args.output_root / "analysis_setup"
    setup.mkdir(parents=True, exist_ok=True)
    fold_columns = [
        PATIENT_ID, OUTER_FOLD, SCD_LABEL, PFD_LABEL,
        *[f"inner_fold_outer_{fold}" for fold in range(args.outer_splits)],
    ]
    atomic_csv(setup / "nested_patient_folds.csv", frame[fold_columns])
    missingness = []
    for feature in CONTINUOUS_FEATURES + CATEGORICAL_FEATURES:
        missingness.append({"feature": feature, "feature_type": "continuous" if feature in CONTINUOUS_FEATURES else "categorical", "missing_n": int(frame[feature].isna().sum()), "missing_percent": 100 * frame[feature].isna().mean()})
    atomic_csv(setup / "feature_missingness.csv", pd.DataFrame(missingness))
    atomic_json(setup / "analysis_manifest.json", {
        "completed": True, "created_at_utc": utc_now(), "analysis": "architecture-matched tabular MLP sensitivity",
        "tabular_csv": str(args.tabular_csv), "tabular_csv_sha256": sha256_file(args.tabular_csv),
        "folds_csv": str(args.folds_csv), "folds_csv_sha256": sha256_file(args.folds_csv),
        "features": {"continuous": CONTINUOUS_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "candidate_grid": MLP_CANDIDATES, "same_mlp_grid_as_text_analysis": True,
        "outer_test_outcomes_used_for_preprocessing_tuning_or_selection": False,
    })
    print(f"Preflight complete: {setup}")


def stage_fit_fold(args: argparse.Namespace) -> None:
    if args.outer_fold is None or args.outer_fold not in range(args.outer_splits):
        raise ValueError("--outer_fold is required for fit-fold.")
    frame = load_data(args)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    outer_fold = args.outer_fold
    destination = args.output_root / "outer_folds" / f"outer_fold_{outer_fold}"
    completion = destination / "run_complete.json"
    if completion.exists() and not args.overwrite:
        print(f"Reuse completed outer fold {outer_fold}: {completion}")
        return
    outer_train = frame.loc[frame[OUTER_FOLD].ne(outer_fold)].reset_index(drop=True)
    outer_test = frame.loc[frame[OUTER_FOLD].eq(outer_fold)].reset_index(drop=True)
    inner_column = f"inner_fold_outer_{outer_fold}"
    candidate_rows, candidate_predictions, candidate_epochs = [], {}, {}
    features = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    for candidate in tqdm(MLP_CANDIDATES, desc=f"Outer {outer_fold} MLP candidates", unit="candidate"):
        fold_predictions, fold_epochs, fold_parameter_counts = [], [], []
        for inner_fold in range(args.inner_splits):
            train = outer_train.loc[outer_train[inner_column].ne(inner_fold)].reset_index(drop=True)
            validation = outer_train.loc[outer_train[inner_column].eq(inner_fold)].reset_index(drop=True)
            preprocessor = build_preprocessor()
            train_x = np.asarray(preprocessor.fit_transform(train[features]), dtype=np.float32)
            validation_x = np.asarray(preprocessor.transform(validation[features]), dtype=np.float32)
            fold_parameter_counts.append(parameter_count(train_x.shape[1], candidate))
            model, best_epoch, _ = fit_mlp(
                train_x, labels(train), validation_x, labels(validation), candidate, device,
                stable_seed(args.seed, "inner", outer_fold, candidate["name"], inner_fold),
                args.epochs, args.min_epochs, args.patience,
            )
            probabilities = predict(model, validation_x, device)
            fold_predictions.append(prediction_frame(validation, probabilities, outer_fold, inner_fold))
            fold_epochs.append(best_epoch)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pooled = pd.concat(fold_predictions, ignore_index=True)
        metrics = metric_values(pooled[["scd_label", "pfd_label"]].to_numpy(float), pooled[["scd_probability", "pfd_probability"]].to_numpy(float))
        candidate_rows.append({
            **candidate, **metrics,
            "median_best_epoch": int(np.rint(np.median(fold_epochs))),
            "median_parameter_count": int(np.rint(np.median(fold_parameter_counts))),
        })
        candidate_predictions[candidate["name"]] = pooled
        candidate_epochs[candidate["name"]] = fold_epochs
    candidates = pd.DataFrame(candidate_rows)
    maximum = candidates["mean_roc_auc"].max()
    eligible = candidates.loc[candidates["mean_roc_auc"] >= maximum - args.auc_tolerance].copy()
    eligible = eligible.sort_values(
        ["median_parameter_count", "mean_pr_auc", "name"],
        ascending=[True, False, True],
    )
    selected_row = eligible.iloc[0]
    selected = next(dict(value) for value in MLP_CANDIDATES if value["name"] == selected_row["name"])
    selected_epochs = candidate_epochs[selected["name"]]
    final_epochs = max(1, int(np.rint(np.median(selected_epochs))))
    preprocessor = build_preprocessor()
    train_x = np.asarray(preprocessor.fit_transform(outer_train[features]), dtype=np.float32)
    test_x = np.asarray(preprocessor.transform(outer_test[features]), dtype=np.float32)
    final_model, _, history = fit_mlp(
        train_x, labels(outer_train), None, None, selected, device,
        stable_seed(args.seed, "final", outer_fold, selected["name"]),
        args.epochs, args.min_epochs, args.patience, fixed_epochs=final_epochs,
    )
    outer_probabilities = predict(final_model, test_x, device)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_csv(destination / "candidate_results.csv", candidates)
    atomic_csv(destination / "selected_inner_oof_predictions.csv", candidate_predictions[selected["name"]])
    atomic_csv(destination / "outer_test_predictions_raw.csv", prediction_frame(outer_test, outer_probabilities, outer_fold, None))
    atomic_csv(destination / "final_training_history.csv", pd.DataFrame(history))
    joblib.dump(preprocessor, destination / "tabular_preprocessor.joblib")
    torch.save(final_model.state_dict(), destination / "model_state.pt")
    atomic_json(destination / "selected_configuration.json", {
        **selected, "inner_best_epochs": selected_epochs,
        "outer_training_epochs": final_epochs,
        "final_parameter_count": parameter_count(train_x.shape[1], selected),
        "selection_score": float(selected_row["mean_roc_auc"]),
        "auc_tolerance": args.auc_tolerance,
        "tie_break": "fewest MLP parameters, then higher mean PR-AUC, then candidate name",
    })
    atomic_json(completion, {"completed": True, "created_at_utc": utc_now(), "outer_fold": outer_fold, "selected_candidate": selected["name"], "device": str(device)})
    print(f"Completed architecture-matched MLP outer fold {outer_fold}")


def clipped_logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    return np.log(probabilities / (1 - probabilities))


def fit_platt(y: np.ndarray, p: np.ndarray):
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    model.fit(clipped_logit(p).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def apply_platt(p: np.ndarray, intercept: float, slope: float):
    value = intercept + slope * clipped_logit(p)
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    table = pd.DataFrame({"threshold": thresholds[finite], "sensitivity": tpr[finite], "specificity": 1 - fpr[finite]})
    table["youden"] = table["sensitivity"] + table["specificity"] - 1
    return float(table.sort_values(["youden", "sensitivity", "threshold"], ascending=[False, False, True]).iloc[0]["threshold"])


def performance(y, raw, calibrated):
    return {"roc_auc": roc_auc_score(y, raw), "pr_auc": average_precision_score(y, raw), "brier": brier_score_loss(y, calibrated)}


def threshold_metrics(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    div = lambda a, b: a / b if b else np.nan
    sensitivity, specificity = div(tp, tp + fn), div(tn, tn + fp)
    return {"sensitivity": sensitivity, "specificity": specificity, "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn), "accuracy": div(tp + tn, len(y)), "f1": div(2 * tp, 2 * tp + fp + fn), "balanced_accuracy": (sensitivity + specificity) / 2}, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def bootstrap_ci(y, raw, calibrated, predicted, replicates, seed):
    rng = np.random.default_rng(seed)
    perf = {name: [] for name in ("roc_auc", "pr_auc", "brier")}
    rates = {name: [] for name in ("sensitivity", "specificity", "ppv", "npv", "accuracy", "f1", "balanced_accuracy")}
    for _ in range(replicates):
        indices = rng.integers(0, len(y), len(y))
        sampled_y = y[indices]
        if np.unique(sampled_y).size < 2:
            continue
        for name, value in performance(sampled_y, raw[indices], calibrated[indices]).items():
            perf[name].append(value)
        for name, value in threshold_metrics(sampled_y, predicted[indices])[0].items():
            rates[name].append(value)
    interval = lambda values: (float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5)))
    return {name: interval(values) for name, values in perf.items()}, {name: interval(values) for name, values in rates.items()}


def external_frame(path: Path, arm: str | None = None):
    frame = pd.read_csv(path, dtype={PATIENT_ID: "string"})
    if arm is not None:
        frame = frame.loc[frame["arm"].eq(arm)].copy()
    frame["patient_key"] = frame[PATIENT_ID].map(canonical_id)
    if frame["patient_key"].duplicated().any():
        raise ValueError(f"Duplicate external patients: {path}, arm={arm}")
    return frame


def holm(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index)
    ordered = values.sort_values()
    running, total = 0.0, len(ordered)
    for rank, (index, value) in enumerate(ordered.items()):
        running = max(running, min(1.0, (total - rank) * value))
        output.loc[index] = running
    return output


def paired_p(differences):
    lower = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
    upper = (np.sum(differences >= 0) + 1) / (len(differences) + 1)
    return min(1.0, 2 * min(lower, upper))


def stage_aggregate(args: argparse.Namespace) -> None:
    frame = load_data(args)
    fold_outputs, selection_rows, calibration_rows, threshold_rows = [], [], [], []
    for outer_fold in range(args.outer_splits):
        directory = args.output_root / "outer_folds" / f"outer_fold_{outer_fold}"
        if not (directory / "run_complete.json").is_file():
            raise FileNotFoundError(f"Incomplete outer fold: {directory}")
        inner = pd.read_csv(directory / "selected_inner_oof_predictions.csv", dtype={PATIENT_ID: "string"})
        test = pd.read_csv(directory / "outer_test_predictions_raw.csv", dtype={PATIENT_ID: "string"})
        selected = json.loads((directory / "selected_configuration.json").read_text())
        selection_rows.append({"outer_fold": outer_fold, **selected})
        for task in TASKS:
            inner_known = inner[f"{task}_label"].notna()
            y_inner = inner.loc[inner_known, f"{task}_label"].to_numpy(int)
            p_inner = inner.loc[inner_known, f"{task}_probability"].to_numpy(float)
            intercept, slope = fit_platt(y_inner, p_inner)
            calibrated_inner = apply_platt(p_inner, intercept, slope)
            threshold = select_threshold(y_inner, calibrated_inner)
            test[f"{task}_probability_uncalibrated"] = test[f"{task}_probability"]
            test[f"{task}_probability_calibrated"] = apply_platt(test[f"{task}_probability"].to_numpy(float), intercept, slope)
            test[f"{task}_selected_threshold"] = threshold
            test[f"{task}_predicted_class"] = (test[f"{task}_probability_calibrated"] >= threshold).astype(int)
            calibration_rows.append({"outer_fold": outer_fold, "outcome": task.upper(), "platt_intercept": intercept, "platt_slope": slope, "inner_oof_n": len(y_inner), "inner_oof_events": int(y_inner.sum())})
            threshold_rows.append({"outer_fold": outer_fold, "outcome": task.upper(), "threshold": threshold, "selection_rule": "maximize Youden J on calibrated inner OOF predictions"})
        fold_outputs.append(test)
    pooled = pd.concat(fold_outputs, ignore_index=True)
    if len(pooled) != args.expected_patients or not pooled["patient_key"].is_unique:
        raise ValueError("Invalid pooled architecture-matched MLP predictions.")
    evaluation = args.output_root / "evaluation"
    atomic_csv(args.output_root / "selection" / "selected_mlp_configuration_by_outer_fold.csv", pd.DataFrame(selection_rows))
    atomic_csv(evaluation / "fold_specific_platt_parameters.csv", pd.DataFrame(calibration_rows))
    atomic_csv(evaluation / "fold_specific_selected_thresholds.csv", pd.DataFrame(threshold_rows))
    atomic_csv(evaluation / "pooled_predictions_calibrated_and_classified.csv", pooled)

    performance_rows, rate_rows, confusion_rows = [], [], []
    for task_index, task in enumerate(TASKS):
        known = pooled[f"{task}_label"].notna()
        y = pooled.loc[known, f"{task}_label"].to_numpy(int)
        raw = pooled.loc[known, f"{task}_probability_uncalibrated"].to_numpy(float)
        calibrated = pooled.loc[known, f"{task}_probability_calibrated"].to_numpy(float)
        predicted = pooled.loc[known, f"{task}_predicted_class"].to_numpy(int)
        point = performance(y, raw, calibrated)
        rates, counts = threshold_metrics(y, predicted)
        perf_ci, rate_ci = bootstrap_ci(y, raw, calibrated, predicted, args.bootstrap_replicates, stable_seed(args.seed, "bootstrap", task))
        for metric, estimate in point.items():
            lo, hi = perf_ci[metric]
            performance_rows.append({"model": "architecture_matched_tabular_mlp", "outcome": task.upper(), "metric": metric, "estimate": estimate, "ci_lower": lo, "ci_upper": hi, "n": len(y), "events": int(y.sum())})
        for metric, estimate in rates.items():
            lo, hi = rate_ci[metric]
            rate_rows.append({"model": "architecture_matched_tabular_mlp", "outcome": task.upper(), "metric": metric, "estimate": estimate, "ci_lower": lo, "ci_upper": hi, "n": len(y), "events": int(y.sum())})
        confusion_rows.append({"model": "architecture_matched_tabular_mlp", "outcome": task.upper(), **counts})
    performance_table = pd.DataFrame(performance_rows)
    atomic_csv(evaluation / "pooled_performance_with_95ci.csv", performance_table)
    atomic_csv(evaluation / "pooled_threshold_metrics_with_95ci.csv", pd.DataFrame(rate_rows))
    atomic_csv(evaluation / "pooled_confusion_matrices.csv", pd.DataFrame(confusion_rows))

    references = {
        "selected_conventional_tabular": external_frame(args.tabular_reference),
        "full_llm_text": external_frame(args.text_reference, "full_risk_no_ecg"),
        "deterministic_text": external_frame(args.text_reference, "deterministic_template_no_ecg"),
    }
    comparison_rows = []
    for reference_name, reference in references.items():
        for task_index, task in enumerate(TASKS):
            mlp = pooled.loc[pooled[f"{task}_label"].notna()].copy()
            mlp["patient_key"] = mlp[PATIENT_ID].map(canonical_id)
            ref = reference.copy()
            label_column = f"{task}_label"
            raw_column = f"{task}_probability_uncalibrated"
            calibrated_column = f"{task}_probability_calibrated"
            ref = ref.loc[ref[label_column].notna()].set_index("patient_key").loc[mlp["patient_key"]]
            y = mlp[label_column].to_numpy(int)
            if not np.array_equal(y, ref[label_column].to_numpy(int)):
                raise ValueError(f"Label mismatch for {reference_name}/{task}")
            mlp_raw = mlp[raw_column].to_numpy(float); mlp_cal = mlp[calibrated_column].to_numpy(float)
            ref_raw = ref[raw_column].to_numpy(float); ref_cal = ref[calibrated_column].to_numpy(float)
            estimates_mlp = performance(y, mlp_raw, mlp_cal); estimates_ref = performance(y, ref_raw, ref_cal)
            rng = np.random.default_rng(stable_seed(args.seed, "paired", reference_name, task))
            samples = {metric: [] for metric in ("roc_auc", "pr_auc", "brier")}
            for _ in range(args.bootstrap_replicates):
                indices = rng.integers(0, len(y), len(y)); sampled_y = y[indices]
                if np.unique(sampled_y).size < 2: continue
                left = performance(sampled_y, mlp_raw[indices], mlp_cal[indices]); right = performance(sampled_y, ref_raw[indices], ref_cal[indices])
                for metric in samples: samples[metric].append(left[metric] - right[metric])
            for metric, values in samples.items():
                values = np.asarray(values)
                comparison_rows.append({"comparison": f"architecture_matched_tabular_mlp_vs_{reference_name}", "comparator": "architecture_matched_tabular_mlp", "reference": reference_name, "outcome": task.upper(), "metric": metric, "difference_comparator_minus_reference": estimates_mlp[metric] - estimates_ref[metric], "difference_ci_lower": np.percentile(values, 2.5), "difference_ci_upper": np.percentile(values, 97.5), "paired_bootstrap_p": paired_p(values)})
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["holm_family_size"] = len(comparisons)
    comparisons["holm_adjusted_p"] = holm(comparisons["paired_bootstrap_p"])
    comparisons["holm_significant_0_05"] = comparisons["holm_adjusted_p"] < 0.05
    atomic_csv(evaluation / "paired_architecture_control_comparisons_with_95ci.csv", comparisons)
    atomic_json(evaluation / "evaluation_manifest.json", {
        "completed": True, "created_at_utc": utc_now(), "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed, "comparison_family_size": len(comparisons),
        "outer_test_outcomes_used_for_preprocessing_tuning_selection_calibration_or_thresholds": False,
        "interpretation": "supplementary architecture-matched sensitivity analysis; not a post hoc principal model",
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__},
    })
    print("\nArchitecture-matched tabular MLP evaluation complete:")
    print(performance_table.to_string(index=False))


def main() -> None:
    args = parse_args()
    for name in ("tabular_csv", "folds_csv", "output_root", "tabular_reference", "text_reference"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.stage == "prepare":
        stage_prepare(args)
    elif args.stage == "fit-fold":
        stage_fit_fold(args)
    else:
        stage_aggregate(args)


if __name__ == "__main__":
    main()
