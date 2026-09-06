#!/usr/bin/env python3
"""Leakage-controlled nested-CV structured clinical baselines for MUSIC.

The script fits two conventional baselines:

1. L2-penalized logistic regression.
2. Histogram gradient boosting.

All supervised and data-dependent operations are restricted to the applicable
training split. This includes imputation, missingness indicators, one-hot
encoding, scaling, class weights, hyperparameter selection, Platt calibration,
and classification-threshold selection.

The primary feature set exactly matches the structured clinical variables used
to construct the no-ECG LLM prompt. A sensitivity feature set adds the five
Holter impression variables used by the with-ECG prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import brentq
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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


PATIENT_ID = "Patient ID"
SCD_LABEL = "SCD_4year_label"
PFD_LABEL = "PFD_4year_label"
OUTER_FOLD = "outer_fold"
TASKS = ("scd", "pfd")
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

ECG_IMPRESSION_FEATURES = [
    "Ventricular Extrasystole",
    "Ventricular Tachycardia",
    "Non-sustained ventricular tachycardia (CH>10)",
    "Paroxysmal supraventricular tachyarrhythmia",
    "Bradycardia",
]

# Literature-defined reduced clinical benchmark supplied by the investigator.
# The source paper must be cited in the manuscript; its citation was not encoded
# in the analysis request and is therefore deliberately not invented here.
LITERATURE_NYHA_III = "NYHA class III (derived)"
LITERATURE_ISCHEMIC = "Ischemic etiology (derived)"
LITERATURE_ACE_OR_ARB = "ACE inhibitor or ARB (derived)"
LITERATURE_RAW_DEPENDENCIES = [
    "Age",
    "Gender (male=1)",
    "Diabetes (yes=1)",
    "NYHA class",
    "HF etiology - Diagnosis",
    "ACE inhibitor (yes=1)",
    "Angiotensin-II receptor blocker (yes=1)",
    "Betablockers (yes=1)",
    "Amiodarone (yes=1)",
    "LVEF (%)",
]
LITERATURE_CONTINUOUS_FEATURES = ["Age", "LVEF (%)"]
LITERATURE_CATEGORICAL_FEATURES = [
    "Gender (male=1)",
    "Diabetes (yes=1)",
    LITERATURE_NYHA_III,
    LITERATURE_ISCHEMIC,
    LITERATURE_ACE_OR_ARB,
    "Betablockers (yes=1)",
    "Amiodarone (yes=1)",
]

FEATURE_SETS = {
    "literature_reduced": {
        "continuous": LITERATURE_CONTINUOUS_FEATURES,
        "categorical": LITERATURE_CATEGORICAL_FEATURES,
        "raw_dependencies": LITERATURE_RAW_DEPENDENCIES,
        "description": (
            "Literature-defined reduced clinical set: age, sex, diabetes, "
            "NYHA III, ischemic etiology, ACE-inhibitor-or-ARB use, beta-blocker "
            "use, amiodarone use, and continuous LVEF."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested-CV structured clinical baselines for MUSIC."
    )
    parser.add_argument("--tabular_csv", type=Path, required=True)
    parser.add_argument("--folds_csv", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--feature_set",
        action="append",
        choices=sorted(FEATURE_SETS),
        help="Repeat to run multiple feature sets. Default: both.",
    )
    parser.add_argument("--patient_id_col", default=PATIENT_ID)
    parser.add_argument("--scd_label_col", default=SCD_LABEL)
    parser.add_argument("--pfd_label_col", default=PFD_LABEL)
    parser.add_argument("--outer_fold_col", default=OUTER_FOLD)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auc_tolerance", type=float, default=0.005)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--probability_clip", type=float, default=1e-6)
    parser.add_argument("--platt_C", type=float, default=1e6)
    parser.add_argument("--expected_patients", type=int, default=730)
    parser.add_argument("--expected_controls", type=int, default=577)
    parser.add_argument("--expected_scd", type=int, default=71)
    parser.add_argument("--expected_pfd", type=int, default=82)
    parser.add_argument("--expected_anticoagulant_yes", type=int, default=610)
    parser.add_argument("--expected_anticoagulant_no", type=int, default=120)
    parser.add_argument(
        "--literature_reference",
        default="Citation to the investigator-specified prior paper must be added to the manuscript.",
        help="Citation or stable identifier for the paper defining literature_reduced.",
    )
    parser.add_argument(
        "--sentinel_json",
        type=Path,
        help=(
            "Optional JSON mapping feature names to values that should be treated "
            "as missing. No sentinel values are assumed by default."
        ),
    )
    parser.add_argument(
        "--overwrite_evaluation",
        action="store_true",
        help="Replace evaluation tables while preserving completed tuning/final fits.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore cached tuning and final-model outputs and refit everything.",
    )
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


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f".{path.stem}.",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(value, handle, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def canonical_patient_id(value: object) -> str:
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    if digits:
        return str(int(digits))
    return text.casefold()


def read_tabular(path: Path) -> pd.DataFrame:
    # The original MUSIC file is semicolon-delimited and uses decimal commas.
    dataframe = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        engine="python",
        na_values=["", "NA", "N/A", "nan"],
        keep_default_na=True,
    )
    dataframe = dataframe.replace(r"\t", ".", regex=True)
    return dataframe


def read_folds(args: argparse.Namespace) -> pd.DataFrame:
    dataframe = pd.read_csv(
        args.folds_csv, dtype={args.patient_id_col: "string"}
    )
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
            "The folds CSV must be the nested file exported by the ECG analysis. "
            f"Missing columns: {sorted(missing)}"
        )
    if not dataframe[args.patient_id_col].is_unique:
        raise ValueError("The folds CSV contains duplicate patient IDs.")
    return dataframe


def load_sentinel_map(path: Path | None) -> dict[str, list[Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sentinel_json must contain a JSON object.")
    return {
        str(feature): values if isinstance(values, list) else [values]
        for feature, values in value.items()
    }


def derive_literature_features(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the prespecified published clinical subset without using outcomes."""
    missing = set(LITERATURE_RAW_DEPENDENCIES) - set(source.columns)
    if missing:
        raise ValueError(f"Cannot derive literature_reduced; missing source fields: {sorted(missing)}")
    result = source.copy()
    numeric = {
        name: pd.to_numeric(result[name], errors="coerce")
        for name in LITERATURE_RAW_DEPENDENCIES
    }

    # The MUSIC prompt/codebook maps HF etiology code 2 to ischemic dilated
    # cardiomyopathy. NYHA III is source code 3.
    result[LITERATURE_NYHA_III] = np.where(
        numeric["NYHA class"].isna(), np.nan, numeric["NYHA class"].eq(3).astype(float)
    )
    result[LITERATURE_ISCHEMIC] = np.where(
        numeric["HF etiology - Diagnosis"].isna(),
        np.nan,
        numeric["HF etiology - Diagnosis"].eq(2).astype(float),
    )
    ace = numeric["ACE inhibitor (yes=1)"]
    arb = numeric["Angiotensin-II receptor blocker (yes=1)"]
    ace_or_arb = pd.Series(np.nan, index=result.index, dtype=float)
    ace_or_arb.loc[ace.eq(1) | arb.eq(1)] = 1.0
    ace_or_arb.loc[ace.eq(0) & arb.eq(0)] = 0.0
    result[LITERATURE_ACE_OR_ARB] = ace_or_arb
    definitions = [
        ("Age", "continuous", "Original enrollment value in years"),
        ("Gender (male=1)", "categorical", "Original binary sex field"),
        ("Diabetes (yes=1)", "categorical", "Original binary diabetes field"),
        (LITERATURE_NYHA_III, "categorical", "1 when NYHA class == 3; 0 otherwise"),
        (LITERATURE_ISCHEMIC, "categorical", "1 when HF etiology code == 2; 0 otherwise"),
        (LITERATURE_ACE_OR_ARB, "categorical", "1 when ACE inhibitor == 1 or ARB == 1; 0 when both == 0"),
        ("Betablockers (yes=1)", "categorical", "Original binary beta-blocker field"),
        ("Amiodarone (yes=1)", "categorical", "Original binary amiodarone field"),
        ("LVEF (%)", "continuous", "Original continuous enrollment LVEF"),
    ]
    audit = pd.DataFrame(definitions, columns=["analysis_feature", "feature_type", "derivation"])
    audit["missing_n"] = [int(result[name].isna().sum()) for name, _, _ in definitions]
    audit["nonmissing_n"] = len(result) - audit["missing_n"]
    audit["positive_n_if_binary"] = [
        int(pd.to_numeric(result[name], errors="coerce").eq(1).sum())
        if kind == "categorical" else np.nan
        for name, kind, _ in definitions
    ]
    return result, audit


def prepare_analysis_data(args: argparse.Namespace) -> pd.DataFrame:
    source = read_tabular(args.tabular_csv)
    folds = read_folds(args)
    if args.patient_id_col not in source.columns:
        raise ValueError(f"Missing patient ID column in tabular CSV: {args.patient_id_col}")

    source = source.copy()
    folds = folds.copy()
    source["__patient_key"] = source[args.patient_id_col].map(canonical_patient_id)
    folds["__patient_key"] = folds[args.patient_id_col].map(canonical_patient_id)
    if not source["__patient_key"].is_unique:
        raise ValueError("Canonicalized patient IDs are not unique in the tabular CSV.")
    if not folds["__patient_key"].is_unique:
        raise ValueError("Canonicalized patient IDs are not unique in the folds CSV.")

    requested_feature_sets = args.feature_set or list(FEATURE_SETS)
    literature_audit = None
    if "literature_reduced" in requested_feature_sets:
        source, literature_audit = derive_literature_features(source)
    required_features = sorted(
        {
            feature
            for feature_set in requested_feature_sets
            for kind in ("continuous", "categorical")
            for feature in FEATURE_SETS[feature_set][kind]
        }
    )
    required_raw_features = sorted(
        {
            feature
            for feature_set in requested_feature_sets
            for feature in FEATURE_SETS[feature_set]["raw_dependencies"]
        }
    )
    missing_features = (set(required_features) | set(required_raw_features)) - set(source.columns)
    if missing_features:
        raise ValueError(f"Tabular CSV is missing features: {sorted(missing_features)}")

    source_subset = source[["__patient_key", *required_features]].copy()
    dataframe = folds.merge(
        source_subset,
        on="__patient_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = dataframe.loc[dataframe["_merge"].ne("both"), args.patient_id_col]
    if len(unmatched):
        raise ValueError(f"Patients missing from tabular CSV: {unmatched.tolist()[:10]}")
    dataframe = dataframe.drop(columns=["_merge"])

    sentinel_map = load_sentinel_map(args.sentinel_json)
    unknown_sentinel_features = set(sentinel_map) - set(required_features)
    if unknown_sentinel_features:
        raise ValueError(
            f"Sentinel map contains unused/unknown features: {sorted(unknown_sentinel_features)}"
        )
    for feature, values in sentinel_map.items():
        dataframe[feature] = dataframe[feature].replace(values, np.nan)

    continuous_features = sorted(
        {
            feature
            for feature_set in requested_feature_sets
            for feature in FEATURE_SETS[feature_set]["continuous"]
        }
    )
    categorical_features = sorted(
        {
            feature
            for feature_set in requested_feature_sets
            for feature in FEATURE_SETS[feature_set]["categorical"]
        }
    )
    for feature in continuous_features:
        dataframe[feature] = pd.to_numeric(dataframe[feature], errors="coerce")
    for feature in categorical_features:
        dataframe[feature] = dataframe[feature].map(
            lambda value: np.nan if pd.isna(value) else str(value).strip()
        )

    anticoagulant_audit = (
        validate_anticoagulant_feature(dataframe, args)
        if any(
            ANTICOAGULANT_FEATURE in FEATURE_SETS[name]["categorical"]
            for name in requested_feature_sets
        )
        else None
    )
    validate_cohort(dataframe, args)
    write_setup_outputs(
        dataframe,
        args,
        requested_feature_sets,
        required_features,
        sentinel_map,
        anticoagulant_audit,
        literature_audit,
    )
    return dataframe


def validate_anticoagulant_feature(
    dataframe: pd.DataFrame, args: argparse.Namespace
) -> dict[str, Any]:
    """Lock the corrected prompt-matched medication field and its cohort counts."""
    if ANTICOAGULANT_FEATURE not in dataframe.columns:
        raise ValueError(
            "The corrected prompt-matched benchmark requires the exact source column "
            f"{ANTICOAGULANT_FEATURE!r}."
        )
    values = pd.to_numeric(dataframe[ANTICOAGULANT_FEATURE], errors="coerce")
    invalid = sorted(values.dropna().loc[~values.dropna().isin([0, 1])].unique().tolist())
    if invalid:
        raise ValueError(
            f"Unexpected values in {ANTICOAGULANT_FEATURE!r}: {invalid}. "
            "Expected only 0, 1, or missing."
        )
    audit = {
        "feature": ANTICOAGULANT_FEATURE,
        "yes_n": int(values.eq(1).sum()),
        "no_n": int(values.eq(0).sum()),
        "missing_n": int(values.isna().sum()),
    }
    expected = {
        "yes_n": args.expected_anticoagulant_yes,
        "no_n": args.expected_anticoagulant_no,
        "missing_n": 0,
    }
    observed = {key: audit[key] for key in expected}
    if observed != expected:
        raise ValueError(
            f"Anticoagulant audit failed: observed {observed}; expected {expected}. "
            "Confirm the corrected 730-patient cohort and source column spelling."
        )
    return audit


def validate_cohort(dataframe: pd.DataFrame, args: argparse.Namespace) -> None:
    if len(dataframe) != args.expected_patients:
        raise ValueError(
            f"Expected {args.expected_patients} patients, found {len(dataframe)}."
        )
    scd = pd.to_numeric(dataframe[args.scd_label_col], errors="coerce")
    pfd = pd.to_numeric(dataframe[args.pfd_label_col], errors="coerce")
    controls = int(((scd == 0) & (pfd == 0)).sum())
    scd_events = int((scd == 1).sum())
    pfd_events = int((pfd == 1).sum())
    observed = (controls, scd_events, pfd_events)
    expected = (args.expected_controls, args.expected_scd, args.expected_pfd)
    if observed != expected:
        raise ValueError(f"Unexpected outcome counts {observed}; expected {expected}.")
    if ((scd == 1) & (pfd == 1)).any():
        raise ValueError("At least one patient is labeled as both SCD and PFD.")

    outer_values = sorted(
        pd.to_numeric(dataframe[args.outer_fold_col], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if outer_values != list(range(args.outer_splits)):
        raise ValueError(f"Unexpected outer folds: {outer_values}")
    for outer_fold in range(args.outer_splits):
        column = f"inner_fold_outer_{outer_fold}"
        outer_train = dataframe[dataframe[args.outer_fold_col].ne(outer_fold)]
        outer_test = dataframe[dataframe[args.outer_fold_col].eq(outer_fold)]
        if outer_train[column].isna().any():
            raise ValueError(f"Missing inner assignments in {column} for outer training.")
        if outer_test[column].notna().any():
            raise ValueError(f"Outer-test patients unexpectedly have values in {column}.")
        values = sorted(outer_train[column].astype(int).unique().tolist())
        if values != list(range(args.inner_splits)):
            raise ValueError(f"Unexpected inner folds in {column}: {values}")


def write_setup_outputs(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    feature_sets: list[str],
    required_features: list[str],
    sentinel_map: dict,
    anticoagulant_audit: dict[str, Any] | None,
    literature_audit: pd.DataFrame | None,
) -> None:
    setup = args.output_root / "analysis_setup"
    setup.mkdir(parents=True, exist_ok=True)
    manifest_path = setup / "analysis_manifest.json"
    signature = {
        "tabular_csv_sha256": sha256_file(args.tabular_csv),
        "folds_csv_sha256": sha256_file(args.folds_csv),
        "feature_sets": {name: FEATURE_SETS[name] for name in feature_sets},
        "sentinel_values_treated_as_missing": sentinel_map,
        "seed": args.seed,
        "outer_folds": args.outer_splits,
        "inner_folds": args.inner_splits,
        "anticoagulant_feature_audit": anticoagulant_audit,
        "literature_reference": args.literature_reference,
    }
    if manifest_path.exists() and not args.restart:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_signature = {key: previous.get(key) for key in signature}
        if previous_signature != signature:
            raise RuntimeError(
                "Existing outputs were created from different inputs or settings. "
                "Use a new output root, or use --restart only when a complete refit "
                "is intentional."
            )
    missingness_rows = []
    for feature in required_features:
        missing = int(dataframe[feature].isna().sum())
        missingness_rows.append(
            {
                "feature": feature,
                "missing_n": missing,
                "missing_percent": 100 * missing / len(dataframe),
                "feature_type": (
                    "continuous"
                    if any(feature in FEATURE_SETS[name]["continuous"] for name in feature_sets)
                    else "categorical"
                ),
            }
        )
    pd.DataFrame(missingness_rows).to_csv(
        setup / "feature_missingness.csv", index=False
    )
    if anticoagulant_audit is not None:
        pd.DataFrame([anticoagulant_audit]).to_csv(
            setup / "anticoagulant_feature_audit.csv", index=False
        )
    if literature_audit is not None:
        literature_audit.to_csv(
            setup / "literature_reduced_feature_derivation_audit.csv", index=False
        )
    atomic_json(
        manifest_path,
        {
            "created_at_utc": utc_now(),
            "tabular_csv": str(args.tabular_csv.resolve()),
            "folds_csv": str(args.folds_csv.resolve()),
            **signature,
            "patients": len(dataframe),
            "controls": args.expected_controls,
            "scd_events": args.expected_scd,
            "pfd_events": args.expected_pfd,
            "sklearn_version": sklearn.__version__,
            "outcome_or_followup_fields_used_as_predictors": False,
            "literature_reduced_subset_selected_using_current_outcomes": False,
        },
    )


def candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for c_value in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        candidates.append(
            {
                "name": f"logistic_l2__C{c_value:g}",
                "family": "penalized_logistic",
                "penalty": "l2",
                "C": c_value,
                "complexity": [0, c_value],
            }
        )
    for learning_rate in (0.03, 0.1):
        for max_leaf_nodes in (7, 15):
            for l2_regularization in (0.0, 1.0):
                candidates.append(
                    {
                        "name": (
                            f"histgb__lr{learning_rate:g}__leaf{max_leaf_nodes}"
                            f"__l2{l2_regularization:g}"
                        ),
                        "family": "gradient_boosting",
                        "learning_rate": learning_rate,
                        "max_leaf_nodes": max_leaf_nodes,
                        "min_samples_leaf": 20,
                        "l2_regularization": l2_regularization,
                        "max_iter": 300,
                        "complexity": [1, max_leaf_nodes, learning_rate, -l2_regularization],
                    }
                )
    return candidates


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(continuous: list[str], categorical: list[str]) -> ColumnTransformer:
    continuous_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("one_hot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", continuous_pipeline, continuous),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def task_labels(dataframe: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    return np.column_stack(
        [
            pd.to_numeric(dataframe[args.scd_label_col], errors="coerce").to_numpy(float),
            pd.to_numeric(dataframe[args.pfd_label_col], errors="coerce").to_numpy(float),
        ]
    )


def balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("A training split lacks one outcome class.")
    weights = np.ones(len(labels), dtype=float)
    weights[labels == 1] = negatives / positives
    return weights


def build_estimator(config: dict, seed: int):
    if config["family"] == "penalized_logistic":
        keyword_arguments = dict(
            solver="lbfgs",
            C=float(config["C"]),
            max_iter=10000,
            tol=1e-4,
            random_state=seed,
        )
        penalty_default = inspect.signature(LogisticRegression).parameters[
            "penalty"
        ].default
        if penalty_default != "deprecated":
            keyword_arguments["penalty"] = "l2"
        return LogisticRegression(**keyword_arguments)
    if config["family"] == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(config["learning_rate"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            l2_regularization=float(config["l2_regularization"]),
            max_iter=int(config["max_iter"]),
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unknown model family: {config['family']}")


def fit_estimator_with_convergence_check(
    estimator,
    values: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    context: str,
):
    """Fit once, retry a convergence failure, then fail rather than ignore it."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(values, labels, sample_weight=sample_weights)
    did_not_converge = any(
        issubclass(item.category, ConvergenceWarning) for item in captured
    )
    if not did_not_converge:
        return estimator

    initial_maximum = int(estimator.get_params().get("max_iter", 10000))
    retry_maximum = min(initial_maximum * 4, 200000)
    estimator.set_params(max_iter=retry_maximum)
    print(
        f"Convergence retry for {context}: max_iter "
        f"{initial_maximum} -> {retry_maximum}",
        flush=True,
    )
    with warnings.catch_warnings(record=True) as captured_retry:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(values, labels, sample_weight=sample_weights)
    still_not_converged = any(
        issubclass(item.category, ConvergenceWarning) for item in captured_retry
    )
    if still_not_converged:
        raise RuntimeError(
            f"Estimator failed to converge after {retry_maximum} iterations: {context}. "
            "Do not use this candidate without revising the optimization setup."
        )
    return estimator


def fit_bundle(
    train: pd.DataFrame,
    continuous: list[str],
    categorical: list[str],
    configs_by_task: dict[str, dict],
    args: argparse.Namespace,
    seed: int,
) -> dict:
    """Fit completely separate binary preprocessing/model pipelines by endpoint."""
    features = continuous + categorical
    labels = task_labels(train, args)
    task_bundles = {}
    for task_index, task in enumerate(TASKS):
        applicable = np.isfinite(labels[:, task_index])
        task_train = train.loc[applicable].reset_index(drop=True)
        y = labels[applicable, task_index].astype(int)
        preprocessor = build_preprocessor(continuous, categorical)
        transformed = np.asarray(
            preprocessor.fit_transform(task_train[features]), dtype=np.float64
        )
        config = configs_by_task[task]
        estimator = build_estimator(config, stable_seed(seed, task))
        estimator = fit_estimator_with_convergence_check(
            estimator,
            transformed,
            y,
            balanced_sample_weight(y),
            context=f"{config['name']} task={task} seed={seed}",
        )
        task_bundles[task] = {
            "preprocessor": preprocessor,
            "estimator": estimator,
            "config": config,
            "training_patients": int(applicable.sum()),
        }
    return {
        "task_bundles": task_bundles,
        "features": features,
        "continuous": continuous,
        "categorical": categorical,
        "configs_by_task": configs_by_task,
        "independent_endpoint_preprocessing_and_models": True,
    }


def predict_bundle(bundle: dict, dataframe: pd.DataFrame) -> np.ndarray:
    probabilities = []
    for task in TASKS:
        task_bundle = bundle["task_bundles"][task]
        transformed = np.asarray(
            task_bundle["preprocessor"].transform(dataframe[bundle["features"]]),
            dtype=np.float64,
        )
        probabilities.append(task_bundle["estimator"].predict_proba(transformed)[:, 1])
    return np.column_stack(probabilities)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(labels)) < 2:
        raise ValueError("Both classes are required to calculate discrimination.")
    return {
        "n": int(len(labels)),
        "events": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def multitask_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    result = {}
    for task_index, task in enumerate(TASKS):
        applicable = np.isfinite(labels[:, task_index])
        result[task] = binary_metrics(
            labels[applicable, task_index].astype(int),
            probabilities[applicable, task_index],
        )
    result["mean_roc_auc"] = float(
        np.mean([result[task]["roc_auc"] for task in TASKS])
    )
    result["mean_pr_auc"] = float(
        np.mean([result[task]["pr_auc"] for task in TASKS])
    )
    return result


def prediction_frame(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    args: argparse.Namespace,
    feature_set: str,
    arm: str,
    outer_fold: int,
    inner_fold: int | None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            args.patient_id_col: dataframe[args.patient_id_col].astype(str).to_numpy(),
            "feature_set": feature_set,
            "arm": arm,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "scd_label": pd.to_numeric(
                dataframe[args.scd_label_col], errors="coerce"
            ).to_numpy(float),
            "pfd_label": pd.to_numeric(
                dataframe[args.pfd_label_col], errors="coerce"
            ).to_numpy(float),
            "scd_probability": probabilities[:, 0],
            "pfd_probability": probabilities[:, 1],
        }
    )


def tune_feature_set(
    dataframe: pd.DataFrame,
    feature_set: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    specification = FEATURE_SETS[feature_set]
    candidates = candidate_grid()
    all_rows = []
    for outer_fold in range(args.outer_splits):
        destination = (
            args.output_root / "tuning" / feature_set / f"outer_fold_{outer_fold}"
        )
        destination.mkdir(parents=True, exist_ok=True)
        summary_path = destination / "candidate_results.csv"
        completed = pd.DataFrame()
        if summary_path.exists() and not args.restart:
            completed = pd.read_csv(summary_path)
            if set(completed.get("candidate_name", [])) == {
                candidate["name"] for candidate in candidates
            }:
                print(f"Reuse complete tuning: {feature_set}, outer fold {outer_fold}")
                all_rows.append(completed)
                continue

        rows = [] if args.restart or completed.empty else completed.to_dict("records")
        completed_names = {row["candidate_name"] for row in rows}
        outer_train_mask = dataframe[args.outer_fold_col].ne(outer_fold)
        inner_column = f"inner_fold_outer_{outer_fold}"
        for candidate_index, config in enumerate(candidates):
            if config["name"] in completed_names:
                continue
            print(
                f"Tune {feature_set} outer={outer_fold} "
                f"candidate={candidate_index + 1}/{len(candidates)} {config['name']}",
                flush=True,
            )
            inner_predictions = []
            for inner_fold in range(args.inner_splits):
                train_mask = outer_train_mask & dataframe[inner_column].ne(inner_fold)
                validation_mask = outer_train_mask & dataframe[inner_column].eq(inner_fold)
                train = dataframe.loc[train_mask].reset_index(drop=True)
                validation = dataframe.loc[validation_mask].reset_index(drop=True)
                seed = stable_seed(
                    args.seed, feature_set, outer_fold, config["name"], inner_fold
                )
                bundle = fit_bundle(
                    train,
                    specification["continuous"],
                    specification["categorical"],
                    {task: config for task in TASKS},
                    args,
                    seed,
                )
                probabilities = predict_bundle(bundle, validation)
                inner_predictions.append(
                    prediction_frame(
                        validation,
                        probabilities,
                        args,
                        feature_set,
                        config["name"],
                        outer_fold,
                        inner_fold,
                    )
                )
            pooled = pd.concat(inner_predictions, ignore_index=True)
            labels = pooled[["scd_label", "pfd_label"]].to_numpy(float)
            probabilities = pooled[
                ["scd_probability", "pfd_probability"]
            ].to_numpy(float)
            metrics = multitask_metrics(labels, probabilities)
            row = {
                "feature_set": feature_set,
                "outer_fold": outer_fold,
                "candidate_name": config["name"],
                "family": config["family"],
                "mean_roc_auc": metrics["mean_roc_auc"],
                "mean_pr_auc": metrics["mean_pr_auc"],
                "scd_roc_auc": metrics["scd"]["roc_auc"],
                "scd_pr_auc": metrics["scd"]["pr_auc"],
                "pfd_roc_auc": metrics["pfd"]["roc_auc"],
                "pfd_pr_auc": metrics["pfd"]["pr_auc"],
                "complexity": json.dumps(config["complexity"]),
                "configuration": json.dumps(config, sort_keys=True),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(summary_path, index=False)
        completed = pd.DataFrame(rows)
        all_rows.append(completed)
    table = pd.concat(all_rows, ignore_index=True)
    destination = args.output_root / "selection"
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / f"all_candidates__{feature_set}.csv", index=False)
    return table


def select_candidate(group: pd.DataFrame, tolerance: float) -> pd.Series:
    maximum = group["mean_roc_auc"].max()
    eligible = group[group["mean_roc_auc"].ge(maximum - tolerance)].copy()
    eligible["complexity_key"] = eligible["complexity"].map(
        lambda value: tuple(json.loads(value))
    )
    eligible = eligible.sort_values(
        ["complexity_key", "mean_pr_auc", "candidate_name"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return eligible.iloc[0]


def create_selection_table(
    tuning: pd.DataFrame, feature_set: str, args: argparse.Namespace
) -> pd.DataFrame:
    rows = []
    for outer_fold in range(args.outer_splits):
        fold_results = tuning[tuning["outer_fold"].eq(outer_fold)]
        for task in TASKS:
            task_results = fold_results.copy()
            task_results["selection_roc_auc"] = task_results[f"{task}_roc_auc"]
            task_results["selection_pr_auc"] = task_results[f"{task}_pr_auc"]
            for selection_type, family in (
                ("penalized_logistic", "penalized_logistic"),
                ("gradient_boosting", "gradient_boosting"),
                ("selected_tabular", None),
            ):
                candidates = (
                    task_results
                    if family is None
                    else task_results[task_results["family"].eq(family)]
                ).rename(
                    columns={
                        "selection_roc_auc": "mean_roc_auc_endpoint",
                        "selection_pr_auc": "mean_pr_auc_endpoint",
                    }
                )
                maximum = candidates["mean_roc_auc_endpoint"].max()
                eligible = candidates[
                    candidates["mean_roc_auc_endpoint"].ge(maximum - args.auc_tolerance)
                ].copy()
                eligible["complexity_key"] = eligible["complexity"].map(
                    lambda value: tuple(json.loads(value))
                )
                selected = eligible.sort_values(
                    ["complexity_key", "mean_pr_auc_endpoint", "candidate_name"],
                    ascending=[True, False, True],
                    kind="mergesort",
                ).iloc[0]
                rows.append(
                    {
                        "feature_set": feature_set,
                        "task": task,
                        "outer_fold": outer_fold,
                        "selection_type": selection_type,
                        "candidate_name": selected["candidate_name"],
                        "family": selected["family"],
                        "inner_endpoint_roc_auc": selected["mean_roc_auc_endpoint"],
                        "inner_endpoint_pr_auc": selected["mean_pr_auc_endpoint"],
                        "configuration": selected["configuration"],
                        "selection_rule": (
                            f"endpoint-specific: within {args.auc_tolerance:g} of maximum "
                            "pooled inner ROC-AUC; then lowest prespecified complexity; "
                            "then highest pooled inner PR-AUC; then candidate name"
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(
        args.output_root / "selection" / f"selected_by_outer_fold__{feature_set}.csv",
        index=False,
    )
    return table


def fit_final_models(
    dataframe: pd.DataFrame,
    feature_set: str,
    selection: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    specification = FEATURE_SETS[feature_set]
    for outer_fold in range(args.outer_splits):
        outer_train_mask = dataframe[args.outer_fold_col].ne(outer_fold)
        outer_test_mask = dataframe[args.outer_fold_col].eq(outer_fold)
        inner_column = f"inner_fold_outer_{outer_fold}"
        for arm in ("penalized_logistic", "gradient_boosting", "selected_tabular"):
            selected_rows = selection[
                selection["outer_fold"].eq(outer_fold)
                & selection["selection_type"].eq(arm)
            ]
            if set(selected_rows["task"]) != set(TASKS):
                raise ValueError(f"Missing endpoint-specific selection for {feature_set}/{outer_fold}/{arm}")
            configs_by_task = {
                task: json.loads(
                    selected_rows[selected_rows["task"].eq(task)].iloc[0]["configuration"]
                )
                for task in TASKS
            }
            destination = (
                args.output_root
                / "final_models"
                / feature_set
                / f"outer_fold_{outer_fold}"
                / arm
            )
            complete_path = destination / "run_complete.json"
            if complete_path.exists() and not args.restart:
                print(f"Reuse final model: {feature_set}, outer={outer_fold}, arm={arm}")
                continue
            destination.mkdir(parents=True, exist_ok=True)
            print(f"Final {feature_set} outer={outer_fold} arm={arm}", flush=True)
            inner_predictions = []
            for inner_fold in range(args.inner_splits):
                train = dataframe.loc[
                    outer_train_mask & dataframe[inner_column].ne(inner_fold)
                ].reset_index(drop=True)
                validation = dataframe.loc[
                    outer_train_mask & dataframe[inner_column].eq(inner_fold)
                ].reset_index(drop=True)
                bundle = fit_bundle(
                    train,
                    specification["continuous"],
                    specification["categorical"],
                    configs_by_task,
                    args,
                    stable_seed(args.seed, "final_inner", feature_set, outer_fold, arm, inner_fold),
                )
                probabilities = predict_bundle(bundle, validation)
                inner_predictions.append(
                    prediction_frame(
                        validation,
                        probabilities,
                        args,
                        feature_set,
                        arm,
                        outer_fold,
                        inner_fold,
                    )
                )
            pd.concat(inner_predictions, ignore_index=True).to_csv(
                destination / "inner_oof_predictions.csv", index=False
            )

            outer_train = dataframe.loc[outer_train_mask].reset_index(drop=True)
            outer_test = dataframe.loc[outer_test_mask].reset_index(drop=True)
            final_bundle = fit_bundle(
                outer_train,
                specification["continuous"],
                specification["categorical"],
                configs_by_task,
                args,
                stable_seed(args.seed, "final_outer", feature_set, outer_fold, arm),
            )
            probabilities = predict_bundle(final_bundle, outer_test)
            prediction_frame(
                outer_test,
                probabilities,
                args,
                feature_set,
                arm,
                outer_fold,
                None,
            ).to_csv(destination / "outer_test_predictions.csv", index=False)
            joblib.dump(final_bundle, destination / "model_bundle.joblib")
            atomic_json(
                complete_path,
                {
                    "completed": True,
                    "created_at_utc": utc_now(),
                    "feature_set": feature_set,
                    "outer_fold": outer_fold,
                    "arm": arm,
                    "selected_candidates_by_task": {
                        task: selected_rows[selected_rows["task"].eq(task)].iloc[0]["candidate_name"]
                        for task in TASKS
                    },
                    "configurations_by_task": configs_by_task,
                    "competing_endpoints_excluded_before_preprocessing": True,
                    "outer_test_outcomes_used_for_fitting_or_selection": False,
                },
            )


def clipped_logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, float), epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1 + exponential)
    return output


def fit_platt(
    labels: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float,
    c_value: float,
) -> dict:
    logits = clipped_logit(probabilities, epsilon).reshape(-1, 1)
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=10000)
    model.fit(logits, labels.astype(int))
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


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

    citl = float(brentq(score, -30.0, 30.0))
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

    def divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else np.nan

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
        "accuracy": divide(tp + tn, tp + tn + fp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_performance(
    dataframe: pd.DataFrame,
    task: str,
    probability_column: str,
    probability_type: str,
    replicates: int,
    seed: int,
    epsilon: float,
    c_value: float,
) -> list[dict]:
    applicable = dataframe[f"{task}_label"].notna()
    labels = dataframe.loc[applicable, f"{task}_label"].astype(int).to_numpy()
    probabilities = dataframe.loc[applicable, probability_column].astype(float).to_numpy()
    point = binary_metrics(labels, probabilities)
    if probability_type == "calibrated":
        point.update(calibration_metrics(labels, probabilities, epsilon, c_value))
        metric_names = (
            "roc_auc",
            "pr_auc",
            "brier_score",
            "calibration_intercept",
            "calibration_slope",
            "calibration_in_the_large",
        )
    else:
        metric_names = ("roc_auc", "pr_auc", "brier_score")
    values = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_probabilities = probabilities[indices]
        try:
            result = binary_metrics(sampled_labels, sampled_probabilities)
            if probability_type == "calibrated":
                result.update(
                    calibration_metrics(
                        sampled_labels, sampled_probabilities, epsilon, c_value
                    )
                )
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
                "probability_type": probability_type,
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
    predictions = dataframe.loc[
        applicable, f"{task}_predicted_class"
    ].astype(int).to_numpy()
    point = classification_metrics(labels, predictions)
    metric_names = (
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "accuracy",
        "f1",
        "balanced_accuracy",
    )
    values = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        result = classification_metrics(labels[indices], predictions[indices])
        for metric in metric_names:
            if np.isfinite(result[metric]):
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
    counts = {
        "outcome": task.upper(),
        "n": len(labels),
        "events": int(labels.sum()),
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
    patient_id_col: str,
) -> dict:
    left_subset = left[
        [patient_id_col, f"{task}_label", probability_column]
    ].rename(columns={probability_column: "left_probability"})
    right_subset = right[[patient_id_col, probability_column]].rename(
        columns={probability_column: "right_probability"}
    )
    merged = left_subset.merge(
        right_subset, on=patient_id_col, how="inner", validate="one_to_one"
    )
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
        2
        * min(
            (np.sum(distribution <= 0) + 1) / (len(distribution) + 1),
            (np.sum(distribution >= 0) + 1) / (len(distribution) + 1),
        ),
    )
    return {
        "outcome": task.upper(),
        "metric": metric,
        "difference": point,
        "ci_lower": float(np.percentile(distribution, 2.5)),
        "ci_upper": float(np.percentile(distribution, 97.5)),
        "paired_p_value": float(p_value),
        "bootstrap_replicates": int(len(distribution)),
        "n": int(len(labels)),
        "events": int(labels.sum()),
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


def evaluate_all(
    feature_sets: list[str], args: argparse.Namespace
) -> None:
    evaluation = args.output_root / "evaluation"
    if evaluation.exists() and any(evaluation.iterdir()) and not args.overwrite_evaluation:
        expected = evaluation / "evaluation_manifest.json"
        if expected.exists():
            print(f"Reuse completed evaluation: {evaluation}")
            return
        raise FileExistsError(
            f"Incomplete evaluation directory exists: {evaluation}. "
            "Inspect it or use --overwrite_evaluation."
        )
    evaluation.mkdir(parents=True, exist_ok=True)

    pooled_models: dict[tuple[str, str], pd.DataFrame] = {}
    calibration_rows = []
    threshold_rows = []
    for feature_set in feature_sets:
        for arm in ("penalized_logistic", "gradient_boosting", "selected_tabular"):
            frames = []
            for outer_fold in range(args.outer_splits):
                directory = (
                    args.output_root
                    / "final_models"
                    / feature_set
                    / f"outer_fold_{outer_fold}"
                    / arm
                )
                inner = pd.read_csv(
                    directory / "inner_oof_predictions.csv",
                    dtype={args.patient_id_col: "string"},
                )
                test = pd.read_csv(
                    directory / "outer_test_predictions.csv",
                    dtype={args.patient_id_col: "string"},
                )
                for task in TASKS:
                    applicable = inner[f"{task}_label"].notna()
                    labels = inner.loc[applicable, f"{task}_label"].astype(int).to_numpy()
                    probabilities = inner.loc[
                        applicable, f"{task}_probability"
                    ].astype(float).to_numpy()
                    parameters = fit_platt(
                        labels, probabilities, args.probability_clip, args.platt_C
                    )
                    calibrated_inner = apply_platt(
                        probabilities, parameters, args.probability_clip
                    )
                    threshold = select_youden(labels, calibrated_inner)
                    calibration_rows.append(
                        {
                            "feature_set": feature_set,
                            "arm": arm,
                            "outer_fold": outer_fold,
                            "outcome": task.upper(),
                            "platt_intercept": parameters["intercept"],
                            "platt_slope": parameters["slope"],
                            "inner_oof_n": len(labels),
                            "inner_oof_events": int(labels.sum()),
                        }
                    )
                    threshold_rows.append(
                        {
                            "feature_set": feature_set,
                            "arm": arm,
                            "outer_fold": outer_fold,
                            "outcome": task.upper(),
                            "selection_rule": (
                                "maximize Youden J on calibrated inner OOF predictions"
                            ),
                            **threshold,
                        }
                    )
                    test[f"{task}_probability_uncalibrated"] = test[
                        f"{task}_probability"
                    ]
                    test[f"{task}_probability_calibrated"] = apply_platt(
                        test[f"{task}_probability"].to_numpy(float),
                        parameters,
                        args.probability_clip,
                    )
                    test[f"{task}_selected_threshold"] = threshold["threshold"]
                    test[f"{task}_predicted_class"] = (
                        test[f"{task}_probability_calibrated"] >= threshold["threshold"]
                    ).astype(int)
                frames.append(test)
            pooled = pd.concat(frames, ignore_index=True)
            if not pooled[args.patient_id_col].is_unique:
                raise RuntimeError(f"Duplicate pooled patient predictions: {feature_set}/{arm}")
            pooled_models[(feature_set, arm)] = pooled
            destination = evaluation / "models" / feature_set / arm
            destination.mkdir(parents=True, exist_ok=True)
            pooled.to_csv(
                destination / "pooled_predictions_calibrated_and_classified.csv",
                index=False,
            )

    pd.DataFrame(calibration_rows).to_csv(
        evaluation / "fold_specific_platt_parameters.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(
        evaluation / "fold_specific_selected_thresholds.csv", index=False
    )

    performance_rows = []
    threshold_metric_rows = []
    confusion_rows = []
    for model_index, ((feature_set, arm), pooled) in enumerate(pooled_models.items()):
        for task_index, task in enumerate(TASKS):
            for probability_type in ("uncalibrated", "calibrated"):
                rows = bootstrap_performance(
                    pooled,
                    task,
                    f"{task}_probability_{probability_type}",
                    probability_type,
                    args.bootstrap_replicates,
                    stable_seed(
                        args.seed,
                        "performance",
                        model_index,
                        task_index,
                        probability_type,
                    ),
                    args.probability_clip,
                    args.platt_C,
                )
                for row in rows:
                    row.update({"feature_set": feature_set, "arm": arm})
                performance_rows.extend(rows)
            rows, counts = bootstrap_threshold_metrics(
                pooled,
                task,
                args.bootstrap_replicates,
                stable_seed(args.seed, "threshold", model_index, task_index),
            )
            for row in rows:
                row.update({"feature_set": feature_set, "arm": arm})
            counts.update({"feature_set": feature_set, "arm": arm})
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

    comparisons = []
    for feature_set in feature_sets:
        comparisons.extend(
            [
                (
                    f"{feature_set}__gradient_vs_logistic",
                    (feature_set, "gradient_boosting"),
                    (feature_set, "penalized_logistic"),
                ),
                (
                    f"{feature_set}__selected_vs_logistic",
                    (feature_set, "selected_tabular"),
                    (feature_set, "penalized_logistic"),
                ),
                (
                    f"{feature_set}__selected_vs_gradient",
                    (feature_set, "selected_tabular"),
                    (feature_set, "gradient_boosting"),
                ),
            ]
        )
    if {"prompt_matched_no_ecg", "prompt_matched_with_ecg"}.issubset(feature_sets):
        for arm in ("penalized_logistic", "gradient_boosting", "selected_tabular"):
            comparisons.append(
                (
                    f"with_ecg_vs_without_ecg__{arm}",
                    ("prompt_matched_with_ecg", arm),
                    ("prompt_matched_no_ecg", arm),
                )
            )

    paired_rows = []
    for comparison_index, (name, left_key, right_key) in enumerate(comparisons):
        for task_index, task in enumerate(TASKS):
            for metric in ("roc_auc", "pr_auc", "brier_score"):
                probability_type = (
                    "uncalibrated" if metric in {"roc_auc", "pr_auc"} else "calibrated"
                )
                probability_column = f"{task}_probability_{probability_type}"
                row = paired_comparison(
                    pooled_models[left_key],
                    pooled_models[right_key],
                    task,
                    metric,
                    probability_column,
                    args.bootstrap_replicates,
                    stable_seed(
                        args.seed, "paired", comparison_index, task_index, metric
                    ),
                    args.patient_id_col,
                )
                row.update(
                    {
                        "comparison": name,
                        "model_a_feature_set": left_key[0],
                        "model_a": left_key[1],
                        "model_b_feature_set": right_key[0],
                        "model_b": right_key[1],
                        "difference_definition": "model_a minus model_b",
                        "probability_type": probability_type,
                    }
                )
                paired_rows.append(row)
    paired_table = pd.DataFrame(paired_rows)
    paired_table["holm_adjusted_p_value"] = holm_adjust(
        paired_table["paired_p_value"]
    )
    paired_table.to_csv(
        evaluation / "paired_tabular_model_comparisons_with_95ci.csv", index=False
    )

    atomic_json(
        evaluation / "evaluation_manifest.json",
        {
            "completed": True,
            "created_at_utc": utc_now(),
            "feature_sets": feature_sets,
            "models": ["penalized_logistic", "gradient_boosting", "selected_tabular"],
            "calibration": "fold-specific Platt fit on selected inner OOF predictions",
            "threshold": "fold-specific Youden J fit on calibrated inner OOF predictions",
            "bootstrap_unit": "patient",
            "bootstrap_replicates": args.bootstrap_replicates,
            "multiplicity_adjustment": "Holm across prespecified tabular comparisons",
            "outer_test_outcomes_used_for_preprocessing_selection_calibration_or_thresholds": False,
        },
    )
    print(f"Saved final tabular evaluation to: {evaluation}")


def main() -> None:
    args = parse_args()
    args.tabular_csv = args.tabular_csv.resolve()
    args.folds_csv = args.folds_csv.resolve()
    args.output_root = args.output_root.resolve()
    if args.sentinel_json is not None:
        args.sentinel_json = args.sentinel_json.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.restart:
        args.overwrite_evaluation = True
    set_seed(args.seed)

    feature_sets = args.feature_set or list(FEATURE_SETS)
    dataframe = prepare_analysis_data(args)
    all_selections = []
    for feature_set in feature_sets:
        tuning = tune_feature_set(dataframe, feature_set, args)
        selection = create_selection_table(tuning, feature_set, args)
        all_selections.append(selection)
        fit_final_models(dataframe, feature_set, selection, args)
    pd.concat(all_selections, ignore_index=True).to_csv(
        args.output_root / "selection" / "selected_by_outer_fold_all_feature_sets.csv",
        index=False,
    )
    evaluate_all(feature_sets, args)


if __name__ == "__main__":
    main()
