#!/usr/bin/env python3
"""Create pooled ROC, precision-recall, and calibration plots for text models.

No models or calibrators are refitted. The script reads the saved pooled
outer-test predictions from the completed detailed-response nested text
analysis. ROC and PR curves use uncalibrated scores; calibration plots use the
fold-specific Platt-calibrated probabilities. Five quantile groups are used by
default to avoid unstable decile estimates in this event-limited cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

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


DEFAULT_ARMS = (
    "full_risk_no_ecg",
    "joint_full_risk_no_ecg",
    "deterministic_template_no_ecg",
    "full_risk_with_ecg",
)
ARM_LABELS = {
    "full_risk_no_ecg": "Endpoint-specific full LLM, no ECG",
    "joint_full_risk_no_ecg": "Joint SCD+PFD full LLM, no ECG",
    "deterministic_template_no_ecg": "Deterministic template",
    "full_risk_with_ecg": "Full LLM, with ECG",
    "neutral_summary_no_ecg": "Neutral summary, no ECG",
    "label_only_no_ecg": "Risk label only, no ECG",
    "rationale_only_no_ecg": "Rationale only, no ECG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pooled text-model evaluation results.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--patient_id_col", default="Patient ID")
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--calibration_groups", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_detailed_analysis(root: Path) -> dict[str, dict]:
    """Reject legacy or incomplete analysis roots before plotting."""
    manifests = {}
    expected_counts = {
        "scd": {"patients": 648, "controls": 577, "events": 71},
        "pfd": {"patients": 659, "controls": 577, "events": 82},
    }
    expected_policy = {
        "pooling": "cls",
        "max_length": 512,
        "long_text_strategy": "mean_chunks",
        "truncated_patient_count": 0,
    }
    for task in ("scd", "pfd"):
        path = root / "tasks" / task / "analysis_setup" / "prepare_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing detailed-analysis manifest: {path}")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("completed") is not True:
            raise ValueError(f"Analysis preparation is incomplete: {path}")
        if manifest.get("task") != task:
            raise ValueError(f"Task mismatch in {path}")
        if manifest.get("patient_counts") != expected_counts[task]:
            raise ValueError(
                f"Unexpected {task.upper()} cohort counts in {path}: "
                f"{manifest.get('patient_counts')!r}"
            )
        if manifest.get("expected_embedding_policy") != expected_policy:
            raise ValueError(
                f"{path} is not from the required detailed, chunked embedding analysis."
            )
        manifests[task] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "expected_embedding_policy": expected_policy,
        }
    return manifests


def load_arm(root: Path, task: str, arm: str, patient_id_col: str) -> tuple[pd.DataFrame, Path]:
    path = (
        root / "tasks" / task / "evaluation" / "arms" / arm
        / "pooled_predictions_calibrated_and_classified.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the text evaluate stage first.")
    frame = pd.read_csv(path, dtype={patient_id_col: "string"})
    required = {
        patient_id_col,
        f"{task}_label",
        f"{task}_probability_uncalibrated",
        f"{task}_probability_calibrated",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    expected = 648 if task == "scd" else 659
    if len(frame) != expected or not frame[patient_id_col].is_unique:
        raise ValueError(f"Expected {expected} unique {task.upper()} patients in {path}.")
    return frame, path


def wilson_interval(events: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = events / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def calibration_points(frame: pd.DataFrame, arm: str, task: str, groups: int) -> pd.DataFrame:
    label_column = f"{task}_label"
    probability_column = f"{task}_probability_calibrated"
    subset = frame.loc[frame[label_column].notna(), [label_column, probability_column]].copy()
    subset[label_column] = subset[label_column].astype(int)
    # Rank before qcut so tied calibrated probabilities still produce stable groups.
    subset["risk_group"] = pd.qcut(
        subset[probability_column].rank(method="first"),
        q=groups,
        labels=False,
    )
    rows = []
    for group, part in subset.groupby("risk_group", sort=True):
        n = len(part)
        events = int(part[label_column].sum())
        lower, upper = wilson_interval(events, n)
        rows.append(
            {
                "arm": arm,
                "outcome": task.upper(),
                "risk_group": int(group) + 1,
                "n": n,
                "events": events,
                "mean_predicted_probability": float(part[probability_column].mean()),
                "observed_event_rate": events / n,
                "observed_ci_lower": lower,
                "observed_ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def plot_discrimination(
    frames: dict[str, dict[str, pd.DataFrame]], destination: Path
) -> pd.DataFrame:
    colors = ("#2070b4", "#2a9d8f", "#e76f51", "#9467bd", "#8c564b")
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    metric_rows = []
    for task_index, task in enumerate(("scd", "pfd")):
        roc_axis = axes[0, task_index]
        pr_axis = axes[1, task_index]
        prevalence = None
        for color, (arm, frame) in zip(colors, frames[task].items()):
            applicable = frame[f"{task}_label"].notna()
            labels = frame.loc[applicable, f"{task}_label"].astype(int).to_numpy()
            probabilities = frame.loc[
                applicable, f"{task}_probability_uncalibrated"
            ].astype(float).to_numpy()
            prevalence = labels.mean()
            fpr, tpr, _ = roc_curve(labels, probabilities)
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            roc_auc = roc_auc_score(labels, probabilities)
            pr_auc = average_precision_score(labels, probabilities)
            label = ARM_LABELS.get(arm, arm)
            roc_axis.plot(fpr, tpr, color=color, lw=2, label=f"{label} ({roc_auc:.3f})")
            pr_axis.plot(recall, precision, color=color, lw=2, label=f"{label} ({pr_auc:.3f})")
            metric_rows.extend(
                [
                    {"arm": arm, "outcome": task.upper(), "metric": "roc_auc", "estimate": roc_auc},
                    {"arm": arm, "outcome": task.upper(), "metric": "pr_auc", "estimate": pr_auc},
                ]
            )
        roc_axis.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
        roc_axis.set(title=f"{task.upper()} ROC", xlabel="False-positive rate", ylabel="True-positive rate")
        pr_axis.axhline(prevalence, color="gray", ls="--", lw=1, label=f"Prevalence ({prevalence:.3f})")
        pr_axis.set(title=f"{task.upper()} precision-recall", xlabel="Recall", ylabel="Precision")
        for axis in (roc_axis, pr_axis):
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    figure.suptitle("Pooled outer-test discrimination of principal text representations")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(metric_rows)


def plot_calibration(points: pd.DataFrame, destination: Path) -> None:
    colors = ("#2070b4", "#2a9d8f", "#e76f51", "#9467bd", "#8c564b")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, outcome in zip(axes, ("SCD", "PFD")):
        axis.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.2, label="Ideal")
        for color, arm in zip(colors, points.arm.unique()):
            subset = points[(points.arm == arm) & (points.outcome == outcome)].sort_values("risk_group")
            y = subset.observed_event_rate.to_numpy()
            yerr = np.vstack(
                [y - subset.observed_ci_lower.to_numpy(), subset.observed_ci_upper.to_numpy() - y]
            )
            axis.errorbar(
                subset.mean_predicted_probability,
                y,
                yerr=yerr,
                marker="o",
                ms=4,
                capsize=2,
                lw=1.5,
                color=color,
                label=ARM_LABELS.get(arm, arm),
            )
        axis.set(title=outcome, xlabel="Mean calibrated predicted risk", ylabel="Observed event proportion")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle("Calibration of pooled fold-specific calibrated text predictions")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.calibration_groups < 2:
        raise ValueError("calibration_groups must be at least 2.")
    root = args.output_root.resolve()
    analysis_manifests = validate_detailed_analysis(root)
    destination = root / "text_evaluation_figures_endpoint_specific"
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {destination}. Use --overwrite intentionally.")
    destination.mkdir(parents=True, exist_ok=True)

    frames: dict[str, dict[str, pd.DataFrame]] = {"scd": {}, "pfd": {}}
    sources = {}
    for task in ("scd", "pfd"):
        reference_ids = None
        for arm in args.arms:
            frame, path = load_arm(root, task, arm, args.patient_id_col)
            ids = set(frame[args.patient_id_col])
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                raise ValueError(f"Patient set differs for {task}/{arm}.")
            frames[task][arm] = frame
            sources[f"{task}/{arm}"] = {"path": str(path), "sha256": sha256_file(path)}

    point_metrics = plot_discrimination(frames, destination / "text_roc_pr_curves.png")
    point_metrics.to_csv(destination / "plotted_discrimination_point_estimates.csv", index=False)
    calibration = pd.concat(
        [
            calibration_points(frame, arm, task, args.calibration_groups)
            for task in ("scd", "pfd")
            for arm, frame in frames[task].items()
        ],
        ignore_index=True,
    )
    calibration.to_csv(destination / "calibration_plot_points.csv", index=False)
    plot_calibration(calibration, destination / "text_calibration_plots.png")

    manifest = {
        "completed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": args.arms,
        "independent_binary_tasks": True,
        "eligible_patients": {"SCD": 648, "PFD": 659},
        "roc_pr_probability_source": "uncalibrated pooled outer-test scores",
        "calibration_probability_source": "fold-specific Platt-calibrated outer-test probabilities",
        "calibration_groups": args.calibration_groups,
        "calibration_interval": "95% Wilson interval for observed event proportion",
        "analysis_manifests": analysis_manifests,
        "source_files": sources,
    }
    with open(destination / "text_evaluation_figures_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved text evaluation figures to: {destination}")


if __name__ == "__main__":
    main()
