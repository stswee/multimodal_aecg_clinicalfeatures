#!/usr/bin/env python3
"""Leakage-controlled Platt calibration for the nested-CV MUSIC ECG models.

For each outer fold, this script:

1. pools the selected model's four inner-validation prediction files;
2. fits separate unweighted SCD and PFD Platt mappings on those cross-fitted
   outer-training predictions;
3. applies the mappings unchanged to that outer fold's untouched test scores;
4. concatenates the five calibrated outer-test files; and
5. reports pooled discrimination, Brier score, calibration intercept/slope,
   calibration-in-the-large, patient-bootstrap confidence intervals, and
   calibration plots.

Outer-test outcomes are used only for final evaluation, never for fitting a
calibrator. The ECG encoders are not retrained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = ("scd", "pfd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit fold-specific Platt calibrators using nested-CV inner OOF predictions."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--outer_splits", type=int, default=5)
    parser.add_argument("--inner_splits", type=int, default=4)
    parser.add_argument("--patient_id_col", default="Patient ID")
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probability_clip", type=float, default=1e-6)
    parser.add_argument(
        "--platt_C",
        type=float,
        default=1e6,
        help="Very weak L2 regularization for numerically stable Platt fitting.",
    )
    parser.add_argument("--calibration_bins", type=int, default=5)
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


def applicable_arrays(
    dataframe: pd.DataFrame,
    task: str,
    probability_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    label_column = f"{task}_label"
    required = [label_column, probability_column]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns for {task}: {missing}")
    applicable = dataframe[label_column].notna()
    labels = dataframe.loc[applicable, label_column].astype(int).to_numpy()
    probabilities = dataframe.loc[applicable, probability_column].astype(float).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{task} calibration data do not contain both outcome classes.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{task} probabilities contain non-finite values.")
    return labels, probabilities


def fit_platt(
    labels: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float,
    c_value: float,
) -> dict:
    logits = clipped_logit(probabilities, epsilon).reshape(-1, 1)
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2000,
        class_weight=None,
    )
    model.fit(logits, labels)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
        "C": float(c_value),
        "training_n": int(len(labels)),
        "training_events": int(labels.sum()),
    }


def apply_platt(
    probabilities: np.ndarray,
    parameters: dict,
    epsilon: float,
) -> np.ndarray:
    logits = clipped_logit(probabilities, epsilon)
    return sigmoid(parameters["intercept"] + parameters["slope"] * logits)


def calibration_in_the_large(
    labels: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float,
) -> float:
    logits = clipped_logit(probabilities, epsilon)

    def score(intercept: float) -> float:
        return float(np.sum(sigmoid(logits + intercept)) - np.sum(labels))

    return float(brentq(score, -50.0, 50.0))


def evaluation_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float,
    c_value: float,
) -> dict:
    recalibration = fit_platt(labels, probabilities, epsilon, c_value)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "calibration_intercept": recalibration["intercept"],
        "calibration_slope": recalibration["slope"],
        "calibration_in_the_large": calibration_in_the_large(
            labels, probabilities, epsilon
        ),
    }


def bootstrap_metrics(
    dataframe: pd.DataFrame,
    task: str,
    probability_column: str,
    repetitions: int,
    seed: int,
    epsilon: float,
    c_value: float,
) -> list[dict]:
    labels, probabilities = applicable_arrays(dataframe, task, probability_column)
    point = evaluation_metrics(labels, probabilities, epsilon, c_value)
    rng = np.random.default_rng(seed)
    bootstrapped = {metric: [] for metric in point}

    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) != 2:
            continue
        sampled_probabilities = probabilities[indices]
        try:
            values = evaluation_metrics(
                sampled_labels, sampled_probabilities, epsilon, c_value
            )
        except (ValueError, RuntimeError):
            continue
        for metric, value in values.items():
            if np.isfinite(value):
                bootstrapped[metric].append(value)

    rows = []
    for metric, estimate in point.items():
        values = np.asarray(bootstrapped[metric], dtype=float)
        if len(values) < max(100, int(0.9 * repetitions)):
            raise RuntimeError(
                f"Too few valid bootstrap replicates for {task}/{metric}: {len(values)}"
            )
        rows.append(
            {
                "outcome": task.upper(),
                "probability_type": (
                    "calibrated" if probability_column.endswith("_calibrated") else "uncalibrated"
                ),
                "metric": metric,
                "estimate": estimate,
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
                "bootstrap_replicates": int(len(values)),
                "n": int(len(labels)),
                "events": int(labels.sum()),
            }
        )
    return rows


def clopper_pearson(events: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    lower = 0.0 if events == 0 else float(beta.ppf(alpha / 2, events, total - events + 1))
    upper = 1.0 if events == total else float(
        beta.ppf(1 - alpha / 2, events + 1, total - events)
    )
    return lower, upper


def calibration_bins(
    dataframe: pd.DataFrame,
    task: str,
    probability_column: str,
    number_of_bins: int,
) -> pd.DataFrame:
    labels, probabilities = applicable_arrays(dataframe, task, probability_column)
    working = pd.DataFrame({"label": labels, "probability": probabilities})
    working["bin"] = pd.qcut(
        working["probability"],
        q=number_of_bins,
        labels=False,
        duplicates="drop",
    )
    rows = []
    for bin_number, subset in working.groupby("bin", observed=True):
        events = int(subset["label"].sum())
        total = int(len(subset))
        lower, upper = clopper_pearson(events, total)
        rows.append(
            {
                "outcome": task.upper(),
                "probability_type": (
                    "calibrated" if probability_column.endswith("_calibrated") else "uncalibrated"
                ),
                "bin": int(bin_number) + 1,
                "n": total,
                "events": events,
                "mean_predicted_probability": float(subset["probability"].mean()),
                "observed_event_fraction": events / total,
                "observed_ci_lower": lower,
                "observed_ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def create_calibration_plot(bin_table: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    colors = {"uncalibrated": "#a23b72", "calibrated": "#2070b4"}
    markers = {"uncalibrated": "o", "calibrated": "s"}
    for axis, task in zip(axes, ("SCD", "PFD")):
        axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Ideal")
        for probability_type in ("uncalibrated", "calibrated"):
            subset = bin_table[
                (bin_table["outcome"] == task)
                & (bin_table["probability_type"] == probability_type)
            ].sort_values("bin")
            y = subset["observed_event_fraction"].to_numpy()
            yerr = np.vstack(
                [
                    y - subset["observed_ci_lower"].to_numpy(),
                    subset["observed_ci_upper"].to_numpy() - y,
                ]
            )
            axis.errorbar(
                subset["mean_predicted_probability"],
                y,
                yerr=yerr,
                color=colors[probability_type],
                marker=markers[probability_type],
                linewidth=1.5,
                capsize=3,
                label=probability_type.capitalize(),
            )
        axis.set_title(task)
        axis.set_xlabel("Mean predicted four-year risk")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed four-year event fraction")
    axes[1].legend(loc="upper left")
    figure.suptitle("ECG model calibration from pooled outer-fold predictions")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    calibration_directory = output_root / "calibration"
    if calibration_directory.exists() and any(calibration_directory.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Calibration output already exists: {calibration_directory}. "
            "Use --overwrite only after confirming that replacement is intended."
        )
    calibration_directory.mkdir(parents=True, exist_ok=True)

    fold_calibrator_rows = []
    pooled_test_frames = []
    source_hashes = {}

    for outer_fold in range(args.outer_splits):
        final_directory = output_root / "final_models" / f"outer_fold_{outer_fold}"
        inner_frames = []
        for inner_fold in range(args.inner_splits):
            path = (
                final_directory
                / "selected_inner_folds"
                / f"inner_fold_{inner_fold}"
                / "validation_predictions.csv"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing selected inner-validation predictions: {path}")
            source_hashes[str(path.relative_to(output_root))] = sha256_file(path)
            inner_frames.append(
                pd.read_csv(path, dtype={args.patient_id_col: "string"})
            )

        inner_oof = pd.concat(inner_frames, ignore_index=True)
        if not inner_oof[args.patient_id_col].is_unique:
            raise RuntimeError(f"Duplicate inner OOF patients for outer fold {outer_fold}.")

        outer_train_path = final_directory / "outer_train_predictions.csv"
        outer_test_path = final_directory / "outer_test_predictions.csv"
        for path in (outer_train_path, outer_test_path):
            if not path.exists():
                raise FileNotFoundError(path)
            source_hashes[str(path.relative_to(output_root))] = sha256_file(path)
        outer_train = pd.read_csv(
            outer_train_path, dtype={args.patient_id_col: "string"}
        )
        outer_test = pd.read_csv(
            outer_test_path, dtype={args.patient_id_col: "string"}
        )
        if set(inner_oof[args.patient_id_col]) != set(outer_train[args.patient_id_col]):
            raise RuntimeError(
                f"Inner OOF patients do not match outer-training patients for fold {outer_fold}."
            )
        if set(inner_oof[args.patient_id_col]) & set(outer_test[args.patient_id_col]):
            raise RuntimeError(f"Outer-test leakage detected for fold {outer_fold}.")

        calibrated_test = outer_test.copy()
        for task in TASKS:
            labels, probabilities = applicable_arrays(
                inner_oof, task, f"{task}_probability"
            )
            parameters = fit_platt(
                labels,
                probabilities,
                args.probability_clip,
                args.platt_C,
            )
            raw_test_probabilities = calibrated_test[f"{task}_probability"].astype(float).to_numpy()
            calibrated_test[f"{task}_probability_uncalibrated"] = raw_test_probabilities
            calibrated_test[f"{task}_probability_calibrated"] = apply_platt(
                raw_test_probabilities,
                parameters,
                args.probability_clip,
            )
            fold_calibrator_rows.append(
                {
                    "outer_fold": outer_fold,
                    "outcome": task.upper(),
                    "platt_intercept": parameters["intercept"],
                    "platt_slope": parameters["slope"],
                    "inner_oof_n": parameters["training_n"],
                    "inner_oof_events": parameters["training_events"],
                    "platt_C": parameters["C"],
                    "probability_clip": args.probability_clip,
                }
            )

        fold_directory = calibration_directory / f"outer_fold_{outer_fold}"
        fold_directory.mkdir(parents=True, exist_ok=True)
        inner_oof.to_csv(
            fold_directory / "inner_oof_predictions_used_for_calibration.csv",
            index=False,
        )
        calibrated_test.to_csv(
            fold_directory / "outer_test_predictions_calibrated.csv",
            index=False,
        )
        pooled_test_frames.append(calibrated_test)

    calibrator_table = pd.DataFrame(fold_calibrator_rows)
    calibrator_table.to_csv(
        calibration_directory / "fold_specific_platt_parameters.csv", index=False
    )

    pooled = pd.concat(pooled_test_frames, ignore_index=True)
    if not pooled[args.patient_id_col].is_unique:
        raise RuntimeError("Pooled calibrated predictions contain duplicate patients.")
    pooled = pooled.sort_values(["outer_fold", args.patient_id_col]).reset_index(drop=True)
    pooled.to_csv(
        calibration_directory / "pooled_outer_test_predictions_calibrated.csv",
        index=False,
    )

    metric_rows = []
    for task_index, task in enumerate(TASKS):
        for probability_index, probability_column in enumerate(
            (f"{task}_probability_uncalibrated", f"{task}_probability_calibrated")
        ):
            metric_rows.extend(
                bootstrap_metrics(
                    pooled,
                    task,
                    probability_column,
                    args.bootstrap_replicates,
                    args.seed + 100 * task_index + probability_index,
                    args.probability_clip,
                    args.platt_C,
                )
            )
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(
        calibration_directory / "pooled_calibration_metrics_with_95ci.csv",
        index=False,
    )

    bin_table = pd.concat(
        [
            calibration_bins(pooled, task, probability_column, args.calibration_bins)
            for task in TASKS
            for probability_column in (
                f"{task}_probability_uncalibrated",
                f"{task}_probability_calibrated",
            )
        ],
        ignore_index=True,
    )
    bin_table.to_csv(
        calibration_directory / "calibration_bin_summary.csv", index=False
    )
    create_calibration_plot(
        bin_table, calibration_directory / "ecg_calibration_plot.png"
    )

    manifest = {
        "completed": True,
        "method": "fold-specific Platt calibration on selected inner OOF predictions",
        "outer_test_outcomes_used_for_calibrator_fitting": False,
        "class_weight_used_for_calibration": False,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "patient_count": int(len(pooled)),
        "one_outer_test_prediction_per_patient": True,
        "bootstrap_unit": "patient",
        "bootstrap_replicates_requested": args.bootstrap_replicates,
        "seed": args.seed,
        "source_file_sha256": source_hashes,
    }
    with open(calibration_directory / "calibration_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nFold-specific Platt parameters:")
    print(calibrator_table.to_string(index=False))
    print("\nPooled calibration metrics with 95% confidence intervals:")
    print(metric_table.to_string(index=False))
    print(f"\nSaved calibration outputs to: {calibration_directory}")


if __name__ == "__main__":
    main()