#!/usr/bin/env python3
"""Exploratory decision-curve analysis for calibrated nested-CV ECG risks.

This script uses only pooled calibrated outer-test predictions. It calculates
net benefit for the ECG model, treat-all, and treat-none strategies, together
with patient-bootstrap confidence intervals and paired net-benefit differences.

The default 2%-25% threshold range is exploratory. It must not be described as
clinically validated unless a concrete intervention and its false-positive /
false-negative tradeoff are justified independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = ("scd", "pfd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decision-curve analysis from calibrated pooled outer-test risks."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--patient_id_col", default="Patient ID")
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


def validate_thresholds(args: argparse.Namespace) -> np.ndarray:
    if not 0 < args.threshold_min < args.threshold_max < 1:
        raise ValueError("Require 0 < threshold_min < threshold_max < 1.")
    if args.threshold_points < 2:
        raise ValueError("threshold_points must be at least 2.")
    return np.linspace(args.threshold_min, args.threshold_max, args.threshold_points)


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


def decision_curve_for_task(
    dataframe: pd.DataFrame,
    task: str,
    thresholds: np.ndarray,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    label_column = f"{task}_label"
    probability_column = f"{task}_probability_calibrated"
    applicable = dataframe[label_column].notna()
    labels = dataframe.loc[applicable, label_column].astype(int).to_numpy()
    probabilities = dataframe.loc[applicable, probability_column].astype(float).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{task} does not contain both classes.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{task} calibrated probabilities contain non-finite values.")

    n = len(labels)
    events = int(labels.sum())
    prevalence = events / n
    threshold_odds = thresholds / (1 - thresholds)
    predicted_positive = probabilities[:, None] >= thresholds[None, :]
    true_positive = np.sum(predicted_positive & (labels[:, None] == 1), axis=0)
    false_positive = np.sum(predicted_positive & (labels[:, None] == 0), axis=0)
    model_net_benefit = true_positive / n - false_positive / n * threshold_odds
    treat_all_net_benefit = prevalence - (1 - prevalence) * threshold_odds
    treat_none_net_benefit = np.zeros_like(thresholds)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n,
        np.full(n, 1 / n),
        size=bootstrap_replicates,
    )
    positive_matrix = predicted_positive & (labels[:, None] == 1)
    negative_matrix = predicted_positive & (labels[:, None] == 0)
    bootstrap_tp = weights @ positive_matrix.astype(np.int16)
    bootstrap_fp = weights @ negative_matrix.astype(np.int16)
    bootstrap_model = (
        bootstrap_tp / n - bootstrap_fp / n * threshold_odds[None, :]
    )
    bootstrap_events = weights @ labels
    bootstrap_prevalence = bootstrap_events / n
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
    uncertainty_supported_better_both = (
        (delta_all_lower > 0) & (delta_none_lower > 0)
    )
    ranges = {
        "outcome": task.upper(),
        "n": n,
        "events": events,
        "prevalence": prevalence,
        "point_estimate_better_than_treat_all_and_treat_none": contiguous_ranges(
            thresholds, point_better_both
        ),
        "paired_bootstrap_95ci_supports_better_than_both": contiguous_ranges(
            thresholds, uncertainty_supported_better_both
        ),
    }
    return table, ranges


def create_plot(table: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for axis, task in zip(axes, ("SCD", "PFD")):
        subset = table[table["outcome"] == task].sort_values("threshold_probability")
        x = subset["threshold_probability"].to_numpy()
        model = subset["model_net_benefit"].to_numpy()
        axis.fill_between(
            x,
            subset["model_ci_lower"].to_numpy(),
            subset["model_ci_upper"].to_numpy(),
            color="#2070b4",
            alpha=0.20,
            label="ECG 95% CI",
        )
        axis.plot(x, model, color="#2070b4", linewidth=2, label="ECG model")
        axis.plot(
            x,
            subset["treat_all_net_benefit"].to_numpy(),
            color="#a23b72",
            linestyle="--",
            linewidth=1.7,
            label="Treat all",
        )
        axis.plot(x, np.zeros_like(x), color="black", linestyle=":", label="Treat none")
        axis.set_title(task)
        axis.set_xlabel("Threshold probability")
        axis.set_ylabel("Net benefit")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Exploratory decision curves from calibrated outer-fold ECG risks")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    thresholds = validate_thresholds(args)
    output_root = args.output_root.resolve()
    source_path = (
        output_root
        / "calibration"
        / "pooled_outer_test_predictions_calibrated.csv"
    )
    if not source_path.exists():
        raise FileNotFoundError(
            f"Run calibrate_ecg_platt.py first; missing {source_path}"
        )
    output_directory = output_root / "decision_curve_analysis"
    if output_directory.exists() and any(output_directory.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Decision-curve output already exists: {output_directory}. "
            "Use --overwrite only after confirming replacement is intended."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(
        source_path,
        dtype={args.patient_id_col: "string"},
    )
    if not predictions[args.patient_id_col].is_unique:
        raise RuntimeError("Pooled calibrated predictions contain duplicate patients.")

    tables = []
    ranges = {}
    for task_index, task in enumerate(TASKS):
        table, task_ranges = decision_curve_for_task(
            predictions,
            task,
            thresholds,
            args.bootstrap_replicates,
            args.seed + task_index,
        )
        tables.append(table)
        ranges[task.upper()] = task_ranges
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(output_directory / "decision_curve_summary.csv", index=False)
    with open(output_directory / "net_benefit_ranges.json", "w", encoding="utf-8") as handle:
        json.dump(ranges, handle, indent=2)
    create_plot(summary, output_directory / "ecg_decision_curves.png")

    manifest = {
        "completed": True,
        "analysis_status": "exploratory",
        "clinical_action_prespecified": False,
        "threshold_range_justification": (
            "Exploratory display across 2%-25%; no clinically validated intervention "
            "threshold was supplied. Clinical-utility claims are not supported by this analysis alone."
        ),
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_points": args.threshold_points,
        "bootstrap_unit": "patient",
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "probability_source": "fold-specific Platt-calibrated outer-test predictions",
        "source_file": str(source_path),
        "source_file_sha256": sha256_file(source_path),
        "treat_all_included": True,
        "treat_none_included": True,
        "paired_net_benefit_differences": True,
    }
    with open(output_directory / "decision_curve_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nDecision-curve net-benefit ranges:")
    print(json.dumps(ranges, indent=2))
    print(f"\nSaved decision-curve outputs to: {output_directory}")


if __name__ == "__main__":
    main()