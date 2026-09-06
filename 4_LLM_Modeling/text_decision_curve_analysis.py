#!/usr/bin/env python3
"""Exploratory decision-curve analysis for nested-CV text models.

The script consumes the fold-specific Platt-calibrated outer-test predictions
already written by train_text_embeddings_nested_cv.py. It does not refit a
model, calibrator, or threshold. Net benefit is calculated for every requested
text arm, treat-all, and treat-none. Patient-level bootstrap samples provide
95% confidence intervals for model net benefit and paired differences from
both default strategies.

The default 2%-25% threshold range is exploratory. It must not be described as
clinically validated unless a concrete intervention and its false-positive /
false-negative tradeoff are justified independently.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = ("scd", "pfd")
DEFAULT_ARMS = (
    "full_risk_no_ecg",
    "joint_full_risk_no_ecg",
    "label_only_no_ecg",
    "rationale_only_no_ecg",
    "neutral_summary_no_ecg",
    "full_risk_with_ecg",
    "label_only_with_ecg",
    "rationale_only_with_ecg",
    "deterministic_template_no_ecg",
)
DEFAULT_PLOT_ARMS = (
    "full_risk_no_ecg",
    "joint_full_risk_no_ecg",
    "deterministic_template_no_ecg",
    "full_risk_with_ecg",
)
ARM_LABELS = {
    "full_risk_no_ecg": "Endpoint-specific full LLM, no ECG",
    "joint_full_risk_no_ecg": "Joint SCD+PFD full LLM, no ECG",
    "label_only_no_ecg": "Risk label only, no ECG",
    "rationale_only_no_ecg": "Rationale only, no ECG",
    "neutral_summary_no_ecg": "Neutral summary, no ECG",
    "full_risk_with_ecg": "Full LLM, with ECG",
    "label_only_with_ecg": "Risk label only, with ECG",
    "rationale_only_with_ecg": "Rationale only, with ECG",
    "deterministic_template_no_ecg": "Deterministic template",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decision curves from calibrated text outer-test predictions."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--patient_id_col", default="Patient ID")
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--plot_arms", nargs="+", default=list(DEFAULT_PLOT_ARMS))
    parser.add_argument("--threshold_min", type=float, default=0.02)
    parser.add_argument("--threshold_max", type=float, default=0.25)
    parser.add_argument("--threshold_points", type=int, default=93)
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


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *map(str, parts)])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def validate_thresholds(args: argparse.Namespace) -> np.ndarray:
    if not 0 < args.threshold_min < args.threshold_max < 1:
        raise ValueError("Require 0 < threshold_min < threshold_max < 1.")
    if args.threshold_points < 2:
        raise ValueError("threshold_points must be at least 2.")
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    return np.linspace(args.threshold_min, args.threshold_max, args.threshold_points)


def source_path(output_root: Path, task: str, arm: str) -> Path:
    return (
        output_root
        / "tasks"
        / task
        / "evaluation"
        / "arms"
        / arm
        / "pooled_predictions_calibrated_and_classified.csv"
    )


def load_arm(
    output_root: Path, task: str, arm: str, patient_id_col: str
) -> tuple[pd.DataFrame, Path]:
    path = source_path(output_root, task, arm)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibrated predictions for {arm}: {path}. "
            "Run the text evaluate stage first."
        )
    frame = pd.read_csv(path, dtype={patient_id_col: "string"})
    required = {
        patient_id_col,
        f"{task}_label",
        f"{task}_probability_calibrated",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    expected = 648 if task == "scd" else 659
    if len(frame) != expected:
        raise ValueError(
            f"Expected {expected} {task.upper()} patients for {arm}, found {len(frame)}."
        )
    if frame[patient_id_col].isna().any() or not frame[patient_id_col].is_unique:
        raise ValueError(f"Patient identifiers are missing or duplicated for {arm}.")
    return frame, path


def contiguous_ranges(thresholds: np.ndarray, condition: np.ndarray) -> list[dict]:
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


def decision_curve(
    frame: pd.DataFrame,
    arm: str,
    task: str,
    thresholds: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    label_column = f"{task}_label"
    probability_column = f"{task}_probability_calibrated"
    applicable = frame[label_column].notna()
    labels = frame.loc[applicable, label_column].astype(int).to_numpy()
    probabilities = frame.loc[applicable, probability_column].astype(float).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{arm}/{task} does not contain both outcome classes.")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{arm}/{task} contains invalid calibrated probabilities.")

    n = len(labels)
    events = int(labels.sum())
    prevalence = events / n
    threshold_odds = thresholds / (1 - thresholds)
    predicted_positive = probabilities[:, None] >= thresholds[None, :]
    positives = labels[:, None] == 1
    negatives = ~positives
    tp = np.sum(predicted_positive & positives, axis=0)
    fp = np.sum(predicted_positive & negatives, axis=0)
    model_nb = tp / n - fp / n * threshold_odds
    all_nb = prevalence - (1 - prevalence) * threshold_odds
    none_nb = np.zeros_like(thresholds)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=replicates)
    boot_tp = weights @ (predicted_positive & positives).astype(np.int16)
    boot_fp = weights @ (predicted_positive & negatives).astype(np.int16)
    boot_model = boot_tp / n - boot_fp / n * threshold_odds[None, :]
    boot_prevalence = (weights @ labels) / n
    boot_all = boot_prevalence[:, None] - (1 - boot_prevalence[:, None]) * threshold_odds
    boot_delta_all = boot_model - boot_all
    boot_delta_none = boot_model

    model_lower, model_upper = np.percentile(boot_model, [2.5, 97.5], axis=0)
    all_lower, all_upper = np.percentile(boot_all, [2.5, 97.5], axis=0)
    delta_all_lower, delta_all_upper = np.percentile(boot_delta_all, [2.5, 97.5], axis=0)
    delta_none_lower, delta_none_upper = np.percentile(boot_delta_none, [2.5, 97.5], axis=0)

    table = pd.DataFrame(
        {
            "arm": arm,
            "outcome": task.upper(),
            "threshold_probability": thresholds,
            "threshold_odds": threshold_odds,
            "false_positives_equivalent_to_one_true_positive": (1 - thresholds) / thresholds,
            "n": n,
            "events": events,
            "prevalence": prevalence,
            "model_net_benefit": model_nb,
            "model_ci_lower": model_lower,
            "model_ci_upper": model_upper,
            "treat_all_net_benefit": all_nb,
            "treat_all_ci_lower": all_lower,
            "treat_all_ci_upper": all_upper,
            "treat_none_net_benefit": none_nb,
            "delta_model_vs_all": model_nb - all_nb,
            "delta_vs_all_ci_lower": delta_all_lower,
            "delta_vs_all_ci_upper": delta_all_upper,
            "delta_model_vs_none": model_nb,
            "delta_vs_none_ci_lower": delta_none_lower,
            "delta_vs_none_ci_upper": delta_none_upper,
        }
    )
    point_better = (model_nb > all_nb) & (model_nb > none_nb)
    ci_better = (delta_all_lower > 0) & (delta_none_lower > 0)
    ranges = {
        "arm": arm,
        "outcome": task.upper(),
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "point_estimate_better_than_treat_all_and_treat_none": contiguous_ranges(
            thresholds, point_better
        ),
        "paired_bootstrap_95ci_supports_better_than_both": contiguous_ranges(
            thresholds, ci_better
        ),
    }
    return table, ranges


def plot_one_arm(table: pd.DataFrame, arm: str, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for axis, outcome in zip(axes, ("SCD", "PFD")):
        subset = table[(table.arm == arm) & (table.outcome == outcome)].sort_values(
            "threshold_probability"
        )
        x = subset.threshold_probability.to_numpy()
        axis.fill_between(
            x,
            subset.model_ci_lower.to_numpy(),
            subset.model_ci_upper.to_numpy(),
            color="#2070b4",
            alpha=0.20,
            label="Model 95% CI",
        )
        axis.plot(x, subset.model_net_benefit, color="#2070b4", lw=2, label="Model")
        axis.plot(x, subset.treat_all_net_benefit, "--", color="#a23b72", lw=1.7, label="Treat all")
        axis.axhline(0, color="black", ls=":", lw=1.5, label="Treat none")
        axis.set_title(outcome)
        axis.set_xlabel("Threshold probability")
        axis.set_ylabel("Net benefit")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(f"Exploratory decision curves: {ARM_LABELS.get(arm, arm)}")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(table: pd.DataFrame, arms: list[str], destination: Path) -> None:
    colors = ("#2070b4", "#2a9d8f", "#e76f51", "#9467bd", "#8c564b")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for axis, outcome in zip(axes, ("SCD", "PFD")):
        outcome_table = table[table.outcome == outcome]
        first = outcome_table[outcome_table.arm == arms[0]].sort_values("threshold_probability")
        x = first.threshold_probability.to_numpy()
        for color, arm in zip(colors, arms):
            subset = outcome_table[outcome_table.arm == arm].sort_values("threshold_probability")
            axis.plot(
                subset.threshold_probability,
                subset.model_net_benefit,
                color=color,
                lw=2,
                label=ARM_LABELS.get(arm, arm),
            )
        axis.plot(x, first.treat_all_net_benefit, "--", color="gray", lw=1.6, label="Treat all")
        axis.axhline(0, color="black", ls=":", lw=1.4, label="Treat none")
        axis.set_title(outcome)
        axis.set_xlabel("Threshold probability")
        axis.set_ylabel("Net benefit")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle("Exploratory decision curves for principal text representations")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    thresholds = validate_thresholds(args)
    output_root = args.output_root.resolve()
    unknown_plot_arms = set(args.plot_arms) - set(args.arms)
    if unknown_plot_arms:
        raise ValueError(f"plot_arms must also be included in arms: {sorted(unknown_plot_arms)}")

    destination = output_root / "text_decision_curve_analysis_endpoint_specific"
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Use --overwrite intentionally."
        )
    destination.mkdir(parents=True, exist_ok=True)

    tables = []
    ranges: dict[str, dict] = {}
    sources = {}
    reference_ids: dict[str, set[str] | None] = {"scd": None, "pfd": None}
    for arm_index, arm in enumerate(args.arms):
        ranges[arm] = {}
        for task_index, task in enumerate(TASKS):
            frame, path = load_arm(output_root, task, arm, args.patient_id_col)
            patient_ids = set(frame[args.patient_id_col])
            if reference_ids[task] is None:
                reference_ids[task] = patient_ids
            elif patient_ids != reference_ids[task]:
                raise ValueError(f"Patient set differs for {task}/{arm}.")
            sources[f"{task}/{arm}"] = {
                "path": str(path), "sha256": sha256_file(path)
            }
            table, summary = decision_curve(
                frame,
                arm,
                task,
                thresholds,
                args.bootstrap_replicates,
                stable_seed(args.seed, "dca", arm_index, task_index),
            )
            tables.append(table)
            ranges[arm][task.upper()] = summary

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(destination / "text_decision_curve_summary.csv", index=False)
    with open(destination / "net_benefit_ranges.json", "w", encoding="utf-8") as handle:
        json.dump(ranges, handle, indent=2)

    plot_directory = destination / "plots_by_arm"
    plot_directory.mkdir(exist_ok=True)
    for arm in args.arms:
        plot_one_arm(combined, arm, plot_directory / f"{arm}_decision_curves.png")
    plot_comparison(
        combined,
        args.plot_arms,
        destination / "principal_text_decision_curves.png",
    )

    manifest = {
        "completed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "exploratory",
        "clinical_action_prespecified": False,
        "threshold_range_justification": (
            "Exploratory display across 2%-25%; no clinically validated intervention "
            "threshold was supplied. Clinical-utility claims are not supported by this analysis alone."
        ),
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_points": args.threshold_points,
        "arms": args.arms,
        "independent_binary_tasks": True,
        "eligible_patients": {"SCD": 648, "PFD": 659},
        "comparison_plot_arms": args.plot_arms,
        "bootstrap_unit": "patient",
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "probability_source": "fold-specific Platt-calibrated outer-test predictions",
        "source_files": sources,
        "treat_all_included": True,
        "treat_none_included": True,
        "paired_net_benefit_differences_from_default_strategies": True,
    }
    with open(destination / "decision_curve_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(ranges, indent=2))
    print(f"\nSaved text decision-curve outputs to: {destination}")


if __name__ == "__main__":
    main()