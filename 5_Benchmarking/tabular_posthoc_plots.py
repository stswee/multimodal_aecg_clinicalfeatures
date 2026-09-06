#!/usr/bin/env python3
"""Post hoc figures and exploratory decision curves for MUSIC tabular models.

This script reads only the saved pooled outer-test predictions from the
leakage-controlled nested-CV tabular analysis. It does not refit models,
recalibrate probabilities, or reselect thresholds.

ROC and precision-recall curves use uncalibrated outer-test scores. Calibration
plots and decision curves use fold-specific Platt-calibrated outer-test risks.
The default 2%-25% decision-curve range is exploratory and must not be described
as clinically validated without an independently justified clinical action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATIENT_ID = "Patient ID"
TASKS = ("scd", "pfd")
FEATURE_SETS = ("prompt_matched_no_ecg", "prompt_matched_with_ecg")
ARMS = ("penalized_logistic", "gradient_boosting", "selected_tabular")

DISPLAY_NAMES = {
    ("prompt_matched_no_ecg", "penalized_logistic"): "Logistic, no ECG",
    ("prompt_matched_no_ecg", "gradient_boosting"): "Gradient boosting, no ECG",
    ("prompt_matched_no_ecg", "selected_tabular"): "Selected tabular, no ECG",
    ("prompt_matched_with_ecg", "penalized_logistic"): "Logistic, with ECG",
    ("prompt_matched_with_ecg", "gradient_boosting"): "Gradient boosting, with ECG",
    ("prompt_matched_with_ecg", "selected_tabular"): "Selected tabular, with ECG",
}

COLORS = {
    ("prompt_matched_no_ecg", "penalized_logistic"): "#2166ac",
    ("prompt_matched_no_ecg", "gradient_boosting"): "#b2182b",
    ("prompt_matched_no_ecg", "selected_tabular"): "#1b9e77",
    ("prompt_matched_with_ecg", "penalized_logistic"): "#67a9cf",
    ("prompt_matched_with_ecg", "gradient_boosting"): "#ef8a62",
    ("prompt_matched_with_ecg", "selected_tabular"): "#7570b3",
}

LINESTYLES = {
    "penalized_logistic": "-",
    "gradient_boosting": "--",
    "selected_tabular": "-.",
}

PRINCIPAL_MODELS = (
    ("prompt_matched_no_ecg", "selected_tabular"),
    ("prompt_matched_no_ecg", "gradient_boosting"),
    ("prompt_matched_with_ecg", "selected_tabular"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ROC/PR, calibration, and exploratory decision-curve "
            "outputs from completed tabular nested-CV predictions."
        )
    )
    parser.add_argument("--tabular_root", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Default: TABULAR_ROOT/tabular_posthoc_figures",
    )
    parser.add_argument("--patient_id_col", default=PATIENT_ID)
    parser.add_argument("--calibration_groups", type=int, default=5)
    parser.add_argument("--threshold_min", type=float, default=0.02)
    parser.add_argument("--threshold_max", type=float, default=0.25)
    parser.add_argument("--threshold_points", type=int, default=93)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
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


def validate_args(args: argparse.Namespace) -> np.ndarray:
    if args.calibration_groups < 2:
        raise ValueError("calibration_groups must be at least 2.")
    if not 0 < args.threshold_min < args.threshold_max < 1:
        raise ValueError("Require 0 < threshold_min < threshold_max < 1.")
    if args.threshold_points < 2:
        raise ValueError("threshold_points must be at least 2.")
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    return np.linspace(
        args.threshold_min,
        args.threshold_max,
        args.threshold_points,
    )


def prediction_path(root: Path, feature_set: str, arm: str) -> Path:
    return (
        root
        / "evaluation"
        / "models"
        / feature_set
        / arm
        / "pooled_predictions_calibrated_and_classified.csv"
    )


def load_predictions(
    root: Path, patient_id_col: str
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, dict[str, str]]]:
    models: dict[tuple[str, str], pd.DataFrame] = {}
    sources: dict[str, dict[str, str]] = {}
    reference: pd.DataFrame | None = None
    identity_columns = [
        patient_id_col,
        "outer_fold",
        "scd_label",
        "pfd_label",
    ]
    required_columns = {
        *identity_columns,
        "feature_set",
        "arm",
        "scd_probability_uncalibrated",
        "scd_probability_calibrated",
        "pfd_probability_uncalibrated",
        "pfd_probability_calibrated",
    }

    for feature_set in FEATURE_SETS:
        for arm in ARMS:
            key = (feature_set, arm)
            path = prediction_path(root, feature_set, arm)
            if not path.is_file():
                raise FileNotFoundError(f"Missing pooled prediction file: {path}")
            dataframe = pd.read_csv(path, dtype={patient_id_col: "string"})
            missing = required_columns - set(dataframe.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            if len(dataframe) != 730:
                raise ValueError(f"Expected 730 rows in {path}, found {len(dataframe)}.")
            if not dataframe[patient_id_col].is_unique:
                raise ValueError(f"Duplicate patient IDs in {path}.")
            if set(dataframe["feature_set"].dropna().unique()) != {feature_set}:
                raise ValueError(f"Feature-set metadata mismatch in {path}.")
            if set(dataframe["arm"].dropna().unique()) != {arm}:
                raise ValueError(f"Arm metadata mismatch in {path}.")
            dataframe = dataframe.sort_values(patient_id_col).reset_index(drop=True)

            for task in TASKS:
                for probability_type in ("uncalibrated", "calibrated"):
                    column = f"{task}_probability_{probability_type}"
                    values = pd.to_numeric(dataframe[column], errors="coerce")
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite probabilities in {path}: {column}")
                    if ((values < 0) | (values > 1)).any():
                        raise ValueError(f"Probabilities outside [0, 1] in {path}: {column}")

            identity = dataframe[identity_columns].copy()
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise ValueError(
                    f"Patient IDs, folds, or labels do not match the other models: {path}"
                )
            models[key] = dataframe
            sources[f"{feature_set}__{arm}"] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }

    if reference is None:
        raise RuntimeError("No prediction files were loaded.")
    scd_events = int(reference["scd_label"].eq(1).sum())
    pfd_events = int(reference["pfd_label"].eq(1).sum())
    if (len(reference), scd_events, pfd_events) != (730, 71, 82):
        raise ValueError(
            "Unexpected cohort audit: "
            f"rows={len(reference)}, SCD={scd_events}, PFD={pfd_events}."
        )
    return models, sources


def applicable_arrays(
    dataframe: pd.DataFrame,
    task: str,
    probability_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    applicable = dataframe[f"{task}_label"].notna()
    labels = dataframe.loc[applicable, f"{task}_label"].astype(int).to_numpy()
    probabilities = dataframe.loc[
        applicable, f"{task}_probability_{probability_type}"
    ].astype(float).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{task.upper()} lacks both outcome classes.")
    return labels, probabilities


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def discrimination_rows(
    models: dict[tuple[str, str], pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature_set, arm), dataframe in models.items():
        for task in TASKS:
            labels, probabilities = applicable_arrays(dataframe, task, "uncalibrated")
            rows.extend(
                [
                    {
                        "feature_set": feature_set,
                        "arm": arm,
                        "outcome": task.upper(),
                        "metric": "roc_auc",
                        "estimate": roc_auc_score(labels, probabilities),
                        "n": len(labels),
                        "events": int(labels.sum()),
                    },
                    {
                        "feature_set": feature_set,
                        "arm": arm,
                        "outcome": task.upper(),
                        "metric": "pr_auc",
                        "estimate": average_precision_score(labels, probabilities),
                        "n": len(labels),
                        "events": int(labels.sum()),
                    },
                ]
            )
    return pd.DataFrame(rows)


def plot_roc_pr(
    models: dict[tuple[str, str], pd.DataFrame],
    model_keys: tuple[tuple[str, str], ...],
    title: str,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, task in enumerate(TASKS):
        roc_axis, pr_axis = axes[row]
        prevalence = None
        for key in model_keys:
            labels, probabilities = applicable_arrays(models[key], task, "uncalibrated")
            prevalence = labels.mean()
            fpr, tpr, _ = roc_curve(labels, probabilities)
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            roc_auc = roc_auc_score(labels, probabilities)
            pr_auc = average_precision_score(labels, probabilities)
            name = DISPLAY_NAMES[key]
            style = LINESTYLES[key[1]]
            roc_axis.plot(
                fpr,
                tpr,
                color=COLORS[key],
                linestyle=style,
                linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})",
            )
            pr_axis.plot(
                recall,
                precision,
                color=COLORS[key],
                linestyle=style,
                linewidth=2,
                label=f"{name} (AP={pr_auc:.3f})",
            )
        roc_axis.plot([0, 1], [0, 1], color="0.5", linestyle=":", label="Chance")
        pr_axis.axhline(
            float(prevalence),
            color="0.5",
            linestyle=":",
            label=f"Prevalence ({prevalence:.3f})",
        )
        roc_axis.set(
            title=f"{task.upper()} ROC",
            xlabel="False-positive rate",
            ylabel="True-positive rate",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        pr_axis.set(
            title=f"{task.upper()} precision–recall",
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        for axis in (roc_axis, pr_axis):
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    save_figure(figure, destination)


def deduplicate_curve_keys(
    models: dict[tuple[str, str], pd.DataFrame],
    model_keys: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Avoid drawing exactly overlapping curves as if they were distinct."""
    retained: list[tuple[str, str]] = []
    comparison_columns = [
        f"{task}_probability_{probability_type}"
        for task in TASKS
        for probability_type in ("uncalibrated", "calibrated")
    ]
    for key in model_keys:
        candidate = models[key][comparison_columns].to_numpy(float)
        if any(
            np.array_equal(
                candidate,
                models[existing][comparison_columns].to_numpy(float),
            )
            for existing in retained
        ):
            continue
        retained.append(key)
    return tuple(retained)


def wilson_interval(events: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    proportion = events / n
    denominator = 1 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
        / denominator
    )
    return center - half_width, center + half_width


def calibration_points(
    models: dict[tuple[str, str], pd.DataFrame],
    groups: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (feature_set, arm), dataframe in models.items():
        for task in TASKS:
            labels, probabilities = applicable_arrays(dataframe, task, "calibrated")
            ranks = pd.Series(probabilities).rank(method="first")
            risk_groups = pd.qcut(ranks, q=groups, labels=False).to_numpy() + 1
            for group in range(1, groups + 1):
                included = risk_groups == group
                group_labels = labels[included]
                group_probabilities = probabilities[included]
                n = int(included.sum())
                events = int(group_labels.sum())
                lower, upper = wilson_interval(events, n)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "arm": arm,
                        "outcome": task.upper(),
                        "risk_group": group,
                        "n": n,
                        "events": events,
                        "minimum_predicted_probability": float(group_probabilities.min()),
                        "maximum_predicted_probability": float(group_probabilities.max()),
                        "mean_predicted_probability": float(group_probabilities.mean()),
                        "observed_event_rate": events / n,
                        "observed_ci_lower": lower,
                        "observed_ci_upper": upper,
                    }
                )
    return pd.DataFrame(rows)


def plot_calibration(
    points: pd.DataFrame,
    model_keys: tuple[tuple[str, str], ...],
    title: str,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, task in zip(axes, ("SCD", "PFD")):
        axis.plot([0, 1], [0, 1], color="0.5", linestyle="--", label="Ideal")
        for key in model_keys:
            subset = points[
                points["feature_set"].eq(key[0])
                & points["arm"].eq(key[1])
                & points["outcome"].eq(task)
            ].sort_values("risk_group")
            x = subset["mean_predicted_probability"].to_numpy()
            y = subset["observed_event_rate"].to_numpy()
            yerr = np.vstack(
                [
                    y - subset["observed_ci_lower"].to_numpy(),
                    subset["observed_ci_upper"].to_numpy() - y,
                ]
            )
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                color=COLORS[key],
                linestyle=LINESTYLES[key[1]],
                marker="o",
                capsize=3,
                linewidth=1.8,
                label=DISPLAY_NAMES[key],
            )
        axis.set(
            title=task,
            xlabel="Mean calibrated predicted risk",
            ylabel="Observed event proportion",
            xlim=(-0.01, 0.45),
            ylim=(-0.01, 0.45),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    save_figure(figure, destination)


def contiguous_ranges(thresholds: np.ndarray, condition: np.ndarray) -> list[dict[str, float]]:
    indices = np.flatnonzero(condition)
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return [
        {
            "threshold_start": float(thresholds[start]),
            "threshold_end": float(thresholds[end]),
        }
        for start, end in zip(starts, ends)
    ]


def decision_curve_for_task(
    dataframe: pd.DataFrame,
    task: str,
    thresholds: np.ndarray,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels, probabilities = applicable_arrays(dataframe, task, "calibrated")
    n = len(labels)
    events = int(labels.sum())
    prevalence = events / n
    threshold_odds = thresholds / (1 - thresholds)
    predicted_positive = probabilities[:, None] >= thresholds[None, :]
    positive_matrix = predicted_positive & (labels[:, None] == 1)
    negative_matrix = predicted_positive & (labels[:, None] == 0)
    true_positive = positive_matrix.sum(axis=0)
    false_positive = negative_matrix.sum(axis=0)
    model_net_benefit = true_positive / n - false_positive / n * threshold_odds
    treat_all_net_benefit = prevalence - (1 - prevalence) * threshold_odds
    treat_none_net_benefit = np.zeros_like(thresholds)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=bootstrap_replicates)
    bootstrap_tp = weights @ positive_matrix.astype(np.int16)
    bootstrap_fp = weights @ negative_matrix.astype(np.int16)
    bootstrap_model = bootstrap_tp / n - bootstrap_fp / n * threshold_odds[None, :]
    bootstrap_prevalence = (weights @ labels) / n
    bootstrap_all = (
        bootstrap_prevalence[:, None]
        - (1 - bootstrap_prevalence[:, None]) * threshold_odds[None, :]
    )
    bootstrap_delta_all = bootstrap_model - bootstrap_all
    bootstrap_delta_none = bootstrap_model

    model_lower, model_upper = np.percentile(bootstrap_model, [2.5, 97.5], axis=0)
    all_lower, all_upper = np.percentile(bootstrap_all, [2.5, 97.5], axis=0)
    delta_all_lower, delta_all_upper = np.percentile(
        bootstrap_delta_all, [2.5, 97.5], axis=0
    )
    delta_none_lower, delta_none_upper = np.percentile(
        bootstrap_delta_none, [2.5, 97.5], axis=0
    )

    table = pd.DataFrame(
        {
            "outcome": task.upper(),
            "threshold_probability": thresholds,
            "threshold_odds": threshold_odds,
            "false_positives_equivalent_to_one_true_positive": (1 - thresholds) / thresholds,
            "n": n,
            "events": events,
            "prevalence": prevalence,
            "model_net_benefit": model_net_benefit,
            "model_ci_lower": model_lower,
            "model_ci_upper": model_upper,
            "treat_all_net_benefit": treat_all_net_benefit,
            "treat_all_ci_lower": all_lower,
            "treat_all_ci_upper": all_upper,
            "treat_none_net_benefit": treat_none_net_benefit,
            "delta_model_vs_all": model_net_benefit - treat_all_net_benefit,
            "delta_vs_all_ci_lower": delta_all_lower,
            "delta_vs_all_ci_upper": delta_all_upper,
            "delta_model_vs_none": model_net_benefit,
            "delta_vs_none_ci_lower": delta_none_lower,
            "delta_vs_none_ci_upper": delta_none_upper,
        }
    )
    point_better_both = (
        (model_net_benefit > treat_all_net_benefit)
        & (model_net_benefit > treat_none_net_benefit)
    )
    pointwise_supported_both = (delta_all_lower > 0) & (delta_none_lower > 0)
    ranges = {
        "outcome": task.upper(),
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "point_estimate_better_than_treat_all_and_treat_none": contiguous_ranges(
            thresholds, point_better_both
        ),
        "paired_pointwise_bootstrap_95ci_supports_better_than_both": contiguous_ranges(
            thresholds, pointwise_supported_both
        ),
    }
    return table, ranges


def plot_individual_decision_curve(
    table: pd.DataFrame,
    key: tuple[str, str],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True)
    for axis, task in zip(axes, ("SCD", "PFD")):
        subset = table[table["outcome"].eq(task)].sort_values("threshold_probability")
        x = subset["threshold_probability"].to_numpy()
        model = subset["model_net_benefit"].to_numpy()
        axis.fill_between(
            x,
            subset["model_ci_lower"].to_numpy(),
            subset["model_ci_upper"].to_numpy(),
            color=COLORS[key],
            alpha=0.20,
            label="Model pointwise 95% CI",
        )
        axis.plot(x, model, color=COLORS[key], linewidth=2, label=DISPLAY_NAMES[key])
        axis.plot(
            x,
            subset["treat_all_net_benefit"].to_numpy(),
            color="0.45",
            linestyle="--",
            linewidth=1.7,
            label="Treat all",
        )
        axis.plot(x, np.zeros_like(x), color="black", linestyle=":", label="Treat none")
        axis.set(
            title=task,
            xlabel="Threshold probability",
            ylabel="Net benefit",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(f"Exploratory decision curves: {DISPLAY_NAMES[key]}")
    figure.tight_layout()
    save_figure(figure, destination)


def plot_principal_decision_curves(
    all_tables: pd.DataFrame,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True)
    for axis, task in zip(axes, ("SCD", "PFD")):
        reference = None
        for key in PRINCIPAL_MODELS:
            subset = all_tables[
                all_tables["feature_set"].eq(key[0])
                & all_tables["arm"].eq(key[1])
                & all_tables["outcome"].eq(task)
            ].sort_values("threshold_probability")
            x = subset["threshold_probability"].to_numpy()
            axis.plot(
                x,
                subset["model_net_benefit"].to_numpy(),
                color=COLORS[key],
                linestyle=LINESTYLES[key[1]],
                linewidth=2,
                label=DISPLAY_NAMES[key],
            )
            if reference is None:
                reference = subset
        if reference is None:
            raise RuntimeError("Principal decision-curve reference is missing.")
        x = reference["threshold_probability"].to_numpy()
        axis.plot(
            x,
            reference["treat_all_net_benefit"].to_numpy(),
            color="0.45",
            linestyle="--",
            linewidth=1.7,
            label="Treat all",
        )
        axis.plot(x, np.zeros_like(x), color="black", linestyle=":", label="Treat none")
        axis.set(
            title=task,
            xlabel="Threshold probability",
            ylabel="Net benefit",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Exploratory decision curves for principal tabular models")
    figure.tight_layout()
    save_figure(figure, destination)


def slug(feature_set: str, arm: str) -> str:
    return f"{feature_set}__{arm}"


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Use --overwrite only after confirming replacement is intended."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    thresholds = validate_args(args)
    tabular_root = args.tabular_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else tabular_root / "tabular_posthoc_figures"
    )
    prepare_output_directory(output_dir, args.overwrite)
    models, sources = load_predictions(tabular_root, args.patient_id_col)

    discrimination = discrimination_rows(models)
    discrimination.to_csv(output_dir / "discrimination_point_estimates.csv", index=False)

    plot_roc_pr(
        models,
        PRINCIPAL_MODELS,
        "Pooled outer-test discrimination of principal tabular models",
        output_dir / "principal_tabular_roc_pr_curves",
    )
    for feature_set in FEATURE_SETS:
        keys = deduplicate_curve_keys(
            models,
            tuple((feature_set, arm) for arm in ARMS),
        )
        plot_roc_pr(
            models,
            keys,
            f"Pooled outer-test discrimination: {feature_set}",
            output_dir / f"{feature_set}_roc_pr_curves",
        )

    calibration = calibration_points(models, args.calibration_groups)
    calibration.to_csv(output_dir / "calibration_plot_points.csv", index=False)
    plot_calibration(
        calibration,
        PRINCIPAL_MODELS,
        "Calibration of fold-specific calibrated tabular predictions",
        output_dir / "principal_tabular_calibration_plots",
    )

    decision_tables: list[pd.DataFrame] = []
    ranges: dict[str, dict[str, Any]] = {}
    individual_dir = output_dir / "decision_curves_by_model"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for key, dataframe in models.items():
        key_tables: list[pd.DataFrame] = []
        ranges[slug(*key)] = {}
        for task in TASKS:
            table, task_ranges = decision_curve_for_task(
                dataframe,
                task,
                thresholds,
                args.bootstrap_replicates,
                stable_seed(args.seed, "decision_curve", task),
            )
            table.insert(0, "arm", key[1])
            table.insert(0, "feature_set", key[0])
            key_tables.append(table)
            ranges[slug(*key)][task.upper()] = task_ranges
        combined = pd.concat(key_tables, ignore_index=True)
        decision_tables.append(combined)
        plot_individual_decision_curve(
            combined,
            key,
            individual_dir / f"{slug(*key)}_decision_curves",
        )

    decision_summary = pd.concat(decision_tables, ignore_index=True)
    decision_summary.to_csv(output_dir / "decision_curve_summary.csv", index=False)
    with open(output_dir / "net_benefit_ranges.json", "w", encoding="utf-8") as handle:
        json.dump(ranges, handle, indent=2)
    plot_principal_decision_curves(
        decision_summary,
        output_dir / "principal_tabular_decision_curves",
    )

    manifest = {
        "completed": True,
        "created_at_utc": utc_now(),
        "analysis_status": "exploratory",
        "clinical_action_prespecified": False,
        "tabular_root": str(tabular_root),
        "output_directory": str(output_dir),
        "models": [slug(*key) for key in models],
        "principal_models": [slug(*key) for key in PRINCIPAL_MODELS],
        "roc_pr_probability_source": "uncalibrated pooled outer-test scores",
        "calibration_probability_source": (
            "fold-specific Platt-calibrated pooled outer-test probabilities"
        ),
        "calibration_groups": args.calibration_groups,
        "calibration_interval": "95% Wilson interval for observed event proportion",
        "decision_curve_probability_source": (
            "fold-specific Platt-calibrated pooled outer-test probabilities"
        ),
        "threshold_range_justification": (
            "Exploratory display across 2%-25%; no clinically validated intervention "
            "threshold was supplied. Clinical-utility claims are not supported by "
            "this analysis alone."
        ),
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_points": args.threshold_points,
        "bootstrap_unit": "patient",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "decision_curve_intervals": (
            "paired pointwise percentile 95% bootstrap confidence intervals; "
            "not simultaneous and not adjusted across thresholds"
        ),
        "treat_all_included": True,
        "treat_none_included": True,
        "models_retrained": False,
        "probabilities_recalibrated": False,
        "source_files": sources,
    }
    with open(output_dir / "tabular_posthoc_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Saved tabular post hoc figures and tables to: {output_dir}")
    print(f"Decision-curve ranges: {output_dir / 'net_benefit_ranges.json'}")


if __name__ == "__main__":
    main()
