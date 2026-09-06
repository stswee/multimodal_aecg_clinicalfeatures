#!/usr/bin/env python3
"""Training-only threshold selection for calibrated nested-CV ECG predictions.

For each outer fold and outcome, this script selects a threshold by maximizing
Youden's J on calibrated, selected inner out-of-fold predictions. It then
applies that threshold unchanged to the corresponding calibrated outer-test
probabilities. Pooled sensitivity, specificity, PPV, NPV, accuracy, F1, and
confusion-matrix counts are reported with patient-bootstrap confidence
intervals for the rate metrics.

No outer-test outcome is used for threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = ("scd", "pfd")
RATE_METRICS = (
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "accuracy",
    "f1",
    "balanced_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select fold-specific thresholds from calibrated inner OOF predictions."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--patient_id_col", default="Patient ID")
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clipped_logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1 - epsilon)
    return np.log(probabilities / (1 - probabilities))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1 + exp_values)
    return output


def apply_platt(
    probabilities: np.ndarray,
    intercept: float,
    slope: float,
    epsilon: float,
) -> np.ndarray:
    return sigmoid(intercept + slope * clipped_logit(probabilities, epsilon))


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    ppv = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    precision_plus_recall = ppv + sensitivity
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": safe_divide(tp + tn, tp + tn + fp + fn),
        "f1": (
            float(2 * ppv * sensitivity / precision_plus_recall)
            if np.isfinite(precision_plus_recall) and precision_plus_recall > 0
            else np.nan
        ),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(labels)),
        "events": int(labels.sum()),
    }


def select_youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    specificity = 1 - fpr
    youden_j = tpr + specificity - 1
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise RuntimeError("No finite candidate thresholds were produced.")
    maximum_j = float(np.max(youden_j[finite]))
    eligible = np.flatnonzero(finite & np.isclose(youden_j, maximum_j, atol=1e-12, rtol=0))
    # Prespecified deterministic tie-break: highest sensitivity, then lowest
    # threshold. This favors sensitivity when Youden's J is identical.
    selected_index = sorted(
        eligible,
        key=lambda index: (-tpr[index], thresholds[index]),
    )[0]
    threshold = float(thresholds[selected_index])
    predicted = (probabilities >= threshold).astype(int)
    metrics = classification_metrics(labels, predicted)
    return {
        "threshold": threshold,
        "youden_j": maximum_j,
        **metrics,
    }


def bootstrap_rate_metrics(
    pooled: pd.DataFrame,
    task: str,
    repetitions: int,
    seed: int,
) -> list[dict]:
    label_column = f"{task}_label"
    prediction_column = f"{task}_predicted_class"
    applicable = pooled[label_column].notna() & pooled[prediction_column].notna()
    labels = pooled.loc[applicable, label_column].astype(int).to_numpy()
    predictions = pooled.loc[applicable, prediction_column].astype(int).to_numpy()
    point = classification_metrics(labels, predictions)
    rng = np.random.default_rng(seed)
    bootstrapped = {metric: [] for metric in RATE_METRICS}

    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) != 2:
            continue
        sampled_predictions = predictions[indices]
        values = classification_metrics(sampled_labels, sampled_predictions)
        for metric in RATE_METRICS:
            if np.isfinite(values[metric]):
                bootstrapped[metric].append(values[metric])

    rows = []
    for metric in RATE_METRICS:
        values = np.asarray(bootstrapped[metric], dtype=float)
        if len(values) < max(100, int(0.9 * repetitions)):
            raise RuntimeError(
                f"Too few valid bootstrap replicates for {task}/{metric}: {len(values)}"
            )
        rows.append(
            {
                "outcome": task.upper(),
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
                "bootstrap_replicates": int(len(values)),
                "n": point["n"],
                "events": point["events"],
            }
        )
    return rows


def create_confusion_plot(count_table: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, task in zip(axes, ("SCD", "PFD")):
        row = count_table[count_table["outcome"] == task].iloc[0]
        matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        image = axis.imshow(matrix, cmap="Blues")
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )
        axis.set_xticks([0, 1], ["Predicted control", f"Predicted {task}"])
        axis.set_yticks([0, 1], ["Observed control", f"Observed {task}"])
        axis.set_title(task)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Observed class")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("ECG model pooled outer-fold confusion matrices")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    calibration_directory = output_root / "calibration"
    threshold_directory = output_root / "threshold_analysis"
    if threshold_directory.exists() and any(threshold_directory.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Threshold output already exists: {threshold_directory}. "
            "Use --overwrite only after confirming replacement is intended."
        )
    threshold_directory.mkdir(parents=True, exist_ok=True)

    parameter_path = calibration_directory / "fold_specific_platt_parameters.csv"
    if not parameter_path.exists():
        raise FileNotFoundError(
            f"Run calibrate_ecg_platt.py first; missing {parameter_path}"
        )
    parameters = pd.read_csv(parameter_path)
    source_hashes = {str(parameter_path.relative_to(output_root)): sha256_file(parameter_path)}
    selected_rows = []
    pooled_frames = []

    for outer_fold in range(args.outer_splits):
        fold_calibration_directory = calibration_directory / f"outer_fold_{outer_fold}"
        inner_path = fold_calibration_directory / "inner_oof_predictions_used_for_calibration.csv"
        test_path = fold_calibration_directory / "outer_test_predictions_calibrated.csv"
        for path in (inner_path, test_path):
            if not path.exists():
                raise FileNotFoundError(path)
            source_hashes[str(path.relative_to(output_root))] = sha256_file(path)
        inner_oof = pd.read_csv(inner_path, dtype={args.patient_id_col: "string"})
        outer_test = pd.read_csv(test_path, dtype={args.patient_id_col: "string"})
        if set(inner_oof[args.patient_id_col]) & set(outer_test[args.patient_id_col]):
            raise RuntimeError(f"Outer-test leakage detected for fold {outer_fold}.")

        classified_test = outer_test.copy()
        for task in TASKS:
            parameter_row = parameters[
                (parameters["outer_fold"] == outer_fold)
                & (parameters["outcome"] == task.upper())
            ]
            if len(parameter_row) != 1:
                raise RuntimeError(
                    f"Expected one Platt parameter row for fold {outer_fold}/{task}."
                )
            parameter_row = parameter_row.iloc[0]
            epsilon = float(parameter_row["probability_clip"])
            applicable_inner = inner_oof[f"{task}_label"].notna()
            labels = inner_oof.loc[applicable_inner, f"{task}_label"].astype(int).to_numpy()
            raw_probabilities = inner_oof.loc[
                applicable_inner, f"{task}_probability"
            ].astype(float).to_numpy()
            calibrated_probabilities = apply_platt(
                raw_probabilities,
                float(parameter_row["platt_intercept"]),
                float(parameter_row["platt_slope"]),
                epsilon,
            )
            selected = select_youden_threshold(labels, calibrated_probabilities)
            selected_rows.append(
                {
                    "outer_fold": outer_fold,
                    "outcome": task.upper(),
                    "selection_rule": "maximize Youden J on calibrated selected inner OOF predictions",
                    "tie_break": "highest sensitivity, then lowest threshold",
                    **selected,
                }
            )

            calibrated_column = f"{task}_probability_calibrated"
            if calibrated_column not in classified_test.columns:
                raise ValueError(f"Missing {calibrated_column} in {test_path}")
            applicable_test = classified_test[f"{task}_label"].notna()
            predicted = pd.Series(pd.NA, index=classified_test.index, dtype="Int64")
            predicted.loc[applicable_test] = (
                classified_test.loc[applicable_test, calibrated_column].astype(float)
                >= selected["threshold"]
            ).astype(int)
            classified_test[f"{task}_selected_threshold"] = selected["threshold"]
            classified_test[f"{task}_predicted_class"] = predicted

        fold_output = threshold_directory / f"outer_fold_{outer_fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        classified_test.to_csv(
            fold_output / "outer_test_classifications.csv", index=False
        )
        pooled_frames.append(classified_test)

    selected_thresholds = pd.DataFrame(selected_rows)
    selected_thresholds.to_csv(
        threshold_directory / "fold_specific_selected_thresholds.csv", index=False
    )
    pooled = pd.concat(pooled_frames, ignore_index=True)
    if not pooled[args.patient_id_col].is_unique:
        raise RuntimeError("Pooled threshold classifications contain duplicate patients.")
    pooled = pooled.sort_values(["outer_fold", args.patient_id_col]).reset_index(drop=True)
    pooled.to_csv(
        threshold_directory / "pooled_outer_test_classifications.csv", index=False
    )

    metric_rows = []
    count_rows = []
    for task_index, task in enumerate(TASKS):
        metric_rows.extend(
            bootstrap_rate_metrics(
                pooled,
                task,
                args.bootstrap_replicates,
                args.seed + task_index,
            )
        )
        applicable = pooled[f"{task}_label"].notna() & pooled[
            f"{task}_predicted_class"
        ].notna()
        point = classification_metrics(
            pooled.loc[applicable, f"{task}_label"].astype(int).to_numpy(),
            pooled.loc[applicable, f"{task}_predicted_class"].astype(int).to_numpy(),
        )
        count_rows.append(
            {
                "outcome": task.upper(),
                "n": point["n"],
                "events": point["events"],
                "tn": point["tn"],
                "fp": point["fp"],
                "fn": point["fn"],
                "tp": point["tp"],
            }
        )

    metric_table = pd.DataFrame(metric_rows)
    count_table = pd.DataFrame(count_rows)
    metric_table.to_csv(
        threshold_directory / "pooled_threshold_metrics_with_95ci.csv", index=False
    )
    count_table.to_csv(
        threshold_directory / "pooled_confusion_matrix_counts.csv", index=False
    )
    create_confusion_plot(
        count_table, threshold_directory / "pooled_confusion_matrices.png"
    )

    manifest = {
        "completed": True,
        "threshold_rule": "maximize Youden J",
        "selection_data": "calibrated selected inner OOF predictions within each outer-training cohort",
        "tie_break": "highest sensitivity, then lowest threshold",
        "outer_test_outcomes_used_for_threshold_selection": False,
        "outer_splits": args.outer_splits,
        "patient_count": int(len(pooled)),
        "one_outer_test_classification_per_applicable_patient": True,
        "bootstrap_unit": "patient",
        "bootstrap_replicates_requested": args.bootstrap_replicates,
        "seed": args.seed,
        "clinical_threshold": False,
        "interpretation": "prespecified statistical operating point; not a validated clinical cutoff",
        "source_file_sha256": source_hashes,
    }
    with open(threshold_directory / "threshold_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nFold-specific selected thresholds:")
    print(selected_thresholds.to_string(index=False))
    print("\nPooled threshold-specific metrics with 95% confidence intervals:")
    print(metric_table.to_string(index=False))
    print("\nPooled confusion-matrix counts:")
    print(count_table.to_string(index=False))
    print(f"\nSaved threshold outputs to: {threshold_directory}")


if __name__ == "__main__":
    main()