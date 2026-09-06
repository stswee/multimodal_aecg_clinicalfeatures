#!/usr/bin/env python3
"""Create publication-ready multimodal ROC, PR, calibration, and DCA figures.

The script reads only the pooled untouched outer-test predictions from the
completed multimodal nested-CV run.  Discrimination curves use uncalibrated
scores; calibration and decision curves use the saved fold-specific Platt-
calibrated probabilities.  Decision-curve uncertainty uses paired patient
bootstrap resampling and includes treat-all and treat-none strategies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATIENT_ID = "Patient ID"
TASKS = ("scd", "pfd")
PAIRS = ("ecg_literature_reduced_tabular",)
ARMS = (
    "concatenation", "projected_concatenation", "scalar_gating",
    "vector_gating", "weighted_sum", "selected_fusion",
)
PAIR_LABELS = {
    "ecg_literature_reduced_tabular": "ECG + literature-reduced clinical variables",
}
ARM_LABELS = {
    "concatenation": "Concatenation",
    "projected_concatenation": "Projected concatenation",
    "scalar_gating": "Scalar gating",
    "vector_gating": "Vector gating",
    "weighted_sum": "Weighted sum",
    "selected_fusion": "Nested selected fusion",
}
PAIR_COLORS = {
    "ecg_literature_reduced_tabular": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multimodal_root", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration_bins", type=int, default=5)
    parser.add_argument("--threshold_min", type=float, default=0.02)
    parser.add_argument("--threshold_max", type=float, default=0.25)
    parser.add_argument("--threshold_points", type=int, default=93)
    parser.add_argument("--expected_scd_patients", type=int, default=648)
    parser.add_argument("--expected_pfd_patients", type=int, default=659)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False) as h:
        frame.to_csv(h, index=False); temporary = Path(h.name)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False) as h:
        json.dump(value, h, indent=2); temporary = Path(h.name)
    temporary.replace(path)


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def validate(frame: pd.DataFrame, expected: dict[str, int]) -> None:
    required = {
        PATIENT_ID, "task", "modality_pair", "arm", "y", "prob",
        "calibrated_prob", "threshold", "predicted",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing pooled prediction columns: {sorted(missing)}")
    if frame.duplicated(["task", PATIENT_ID, "modality_pair", "arm"]).any():
        raise ValueError("Duplicate task/patient/modality-pair/arm rows.")
    if set(frame.modality_pair) != set(PAIRS) or set(frame.arm) != set(ARMS):
        raise ValueError("The pooled file does not contain the expected reduced-clinical pair and six arms.")
    if set(frame.task) != set(TASKS):
        raise ValueError(f"Unexpected tasks: {sorted(set(frame.task))}")
    counts = frame.groupby(["task", "modality_pair", "arm"])[PATIENT_ID].nunique()
    for task, patients in expected.items():
        if not counts.xs(task, level="task").eq(patients).all():
            raise ValueError(f"Unexpected patient count for {task}.")


def task_arrays(group: pd.DataFrame, task: str, calibrated: bool = False) -> tuple[np.ndarray, np.ndarray]:
    task_group = group[group["task"].eq(task)]
    probability = "calibrated_prob" if calibrated else "prob"
    return (
        task_group["y"].to_numpy(int),
        task_group[probability].to_numpy(float),
    )


def wilson_interval(events: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = events / n; denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def calibration_points(group: pd.DataFrame, pair: str, arm: str, task: str, bins: int) -> pd.DataFrame:
    y, p = task_arrays(group, task, calibrated=True)
    ranks = pd.Series(p).rank(method="first")
    labels = pd.qcut(ranks, q=min(bins, len(np.unique(p))), labels=False, duplicates="drop")
    rows = []
    for bin_number in sorted(pd.unique(labels)):
        selected = np.asarray(labels == bin_number)
        n = int(selected.sum()); events = int(y[selected].sum())
        lower, upper = wilson_interval(events, n)
        rows.append({
            "modality_pair": pair, "arm": arm, "outcome": task,
            "bin": int(bin_number) + 1, "patients": n, "events": events,
            "mean_predicted_probability": float(np.mean(p[selected])),
            "observed_event_rate": events / n,
            "observed_ci_lower": lower, "observed_ci_upper": upper,
        })
    return pd.DataFrame(rows)


def plot_selected_discrimination(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    selected = frame[frame.arm.eq("selected_fusion")]
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    estimates = []
    for row, task in enumerate(TASKS):
        roc_axis, pr_axis = axes[row]
        prevalence = None
        for pair in PAIRS:
            group = selected[selected.modality_pair.eq(pair)]
            y, p = task_arrays(group, task)
            fpr, tpr, _ = roc_curve(y, p); precision, recall, _ = precision_recall_curve(y, p)
            roc_value = roc_auc_score(y, p); pr_value = average_precision_score(y, p)
            prevalence = float(np.mean(y))
            label = PAIR_LABELS[pair]
            roc_axis.plot(fpr, tpr, color=PAIR_COLORS[pair], lw=2, label=f"{label} ({roc_value:.3f})")
            pr_axis.plot(recall, precision, color=PAIR_COLORS[pair], lw=2, label=f"{label} ({pr_value:.3f})")
            estimates.extend([
                {"modality_pair": pair, "arm": "selected_fusion", "outcome": task, "metric": "roc_auc", "estimate": roc_value},
                {"modality_pair": pair, "arm": "selected_fusion", "outcome": task, "metric": "pr_auc", "estimate": pr_value},
            ])
        roc_axis.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        pr_axis.axhline(prevalence, color="black", ls="--", lw=1, label=f"Prevalence ({prevalence:.3f})")
        roc_axis.set(title=f"{task.upper()} ROC", xlabel="False-positive rate", ylabel="True-positive rate")
        pr_axis.set(title=f"{task.upper()} precision–recall", xlabel="Recall", ylabel="Precision")
        for axis in (roc_axis, pr_axis):
            axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.grid(alpha=0.2); axis.legend(fontsize=8)
    figure.suptitle("Selected multimodal fusion: pooled outer-test discrimination", fontsize=14)
    figure.tight_layout(); save_figure(figure, output / "selected_fusion_roc_pr_curves")
    return pd.DataFrame(estimates)


def plot_method_discrimination(frame: pd.DataFrame, output: Path) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(ARMS)))
    for pair in PAIRS:
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        pair_frame = frame[frame.modality_pair.eq(pair)]
        for row, task in enumerate(TASKS):
            prevalence = None
            for color, arm in zip(colors, ARMS):
                group = pair_frame[pair_frame.arm.eq(arm)]
                y, p = task_arrays(group, task); prevalence = np.mean(y)
                fpr, tpr, _ = roc_curve(y, p); precision, recall, _ = precision_recall_curve(y, p)
                axes[row, 0].plot(fpr, tpr, color=color, lw=1.7, label=f"{ARM_LABELS[arm]} ({roc_auc_score(y,p):.3f})")
                axes[row, 1].plot(recall, precision, color=color, lw=1.7, label=f"{ARM_LABELS[arm]} ({average_precision_score(y,p):.3f})")
            axes[row, 0].plot([0, 1], [0, 1], "k--", lw=1)
            axes[row, 1].axhline(prevalence, color="black", ls="--", lw=1)
            axes[row, 0].set(title=f"{task.upper()} ROC", xlabel="False-positive rate", ylabel="True-positive rate")
            axes[row, 1].set(title=f"{task.upper()} precision–recall", xlabel="Recall", ylabel="Precision")
            for axis in axes[row]:
                axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.grid(alpha=0.2); axis.legend(fontsize=7)
        figure.suptitle(f"{PAIR_LABELS[pair]}: fusion-method sensitivity analysis", fontsize=14)
        figure.tight_layout(); save_figure(figure, output / f"fusion_methods_{pair}_roc_pr_curves")


def plot_selected_calibration(frame: pd.DataFrame, output: Path, bins: int) -> pd.DataFrame:
    selected = frame[frame.arm.eq("selected_fusion")]
    all_points = []
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, task in zip(axes, TASKS):
        for pair in PAIRS:
            group = selected[selected.modality_pair.eq(pair)]
            points = calibration_points(group, pair, "selected_fusion", task, bins)
            all_points.append(points)
            lower_error = np.maximum(
                0.0,
                points.observed_event_rate.to_numpy()
                - points.observed_ci_lower.to_numpy(),
            )
            upper_error = np.maximum(
                0.0,
                points.observed_ci_upper.to_numpy()
                - points.observed_event_rate.to_numpy(),
            )
            axis.errorbar(
                points.mean_predicted_probability, points.observed_event_rate,
                yerr=[lower_error, upper_error],
                color=PAIR_COLORS[pair], marker="o", lw=1.7, capsize=3, label=PAIR_LABELS[pair],
            )
        axis.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        axis.set(title=task.upper(), xlabel="Mean predicted probability", ylabel="Observed event proportion")
        axis.set_xlim(0, 0.45); axis.set_ylim(0, 0.45); axis.grid(alpha=0.2); axis.legend(fontsize=8)
    figure.suptitle("Selected multimodal fusion: pooled calibration", fontsize=14)
    figure.tight_layout(); save_figure(figure, output / "selected_fusion_calibration_plots")
    return pd.concat(all_points, ignore_index=True)


def contiguous_ranges(thresholds: np.ndarray, condition: np.ndarray) -> list[dict[str, float]]:
    indices = np.flatnonzero(condition)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]; ends = np.r_[indices[breaks], indices[-1]]
    return [{"threshold_start": float(thresholds[a]), "threshold_end": float(thresholds[b])} for a, b in zip(starts, ends)]


def dca_values(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    odds = thresholds / (1 - thresholds); positive = p[:, None] >= thresholds[None, :]
    tp = np.sum(positive & (y[:, None] == 1), axis=0)
    fp = np.sum(positive & (y[:, None] == 0), axis=0)
    model = tp / len(y) - fp / len(y) * odds
    treat_all = np.mean(y) - (1 - np.mean(y)) * odds
    return model, treat_all, positive


def selected_dca(
    frame: pd.DataFrame, output: Path, thresholds: np.ndarray, replicates: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    selected = frame[frame.arm.eq("selected_fusion")]
    tables, ranges = [], {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for task_index, (axis, task) in enumerate(zip(axes, TASKS)):
        ranges[task.upper()] = {}
        for pair_index, pair in enumerate(PAIRS):
            group = selected[selected.modality_pair.eq(pair)]
            y, p = task_arrays(group, task, calibrated=True)
            model, treat_all, positive = dca_values(y, p, thresholds)
            odds = thresholds / (1 - thresholds)
            rng = np.random.default_rng(seed + task_index * 100 + pair_index)
            weights = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=replicates)
            bootstrap_tp = weights @ (positive & (y[:, None] == 1)).astype(np.int16)
            bootstrap_fp = weights @ (positive & (y[:, None] == 0)).astype(np.int16)
            bootstrap_model = bootstrap_tp / len(y) - bootstrap_fp / len(y) * odds[None, :]
            bootstrap_prevalence = (weights @ y) / len(y)
            bootstrap_all = bootstrap_prevalence[:, None] - (1 - bootstrap_prevalence[:, None]) * odds[None, :]
            model_low, model_high = np.percentile(bootstrap_model, [2.5, 97.5], axis=0)
            delta_all_low, delta_all_high = np.percentile(bootstrap_model - bootstrap_all, [2.5, 97.5], axis=0)
            delta_none_low, delta_none_high = np.percentile(bootstrap_model, [2.5, 97.5], axis=0)
            tables.append(pd.DataFrame({
                "modality_pair": pair, "arm": "selected_fusion", "outcome": task,
                "threshold_probability": thresholds, "model_net_benefit": model,
                "model_ci_lower": model_low, "model_ci_upper": model_high,
                "treat_all_net_benefit": treat_all, "treat_none_net_benefit": 0.0,
                "delta_model_vs_all": model - treat_all,
                "delta_vs_all_ci_lower": delta_all_low, "delta_vs_all_ci_upper": delta_all_high,
                "delta_model_vs_none": model,
                "delta_vs_none_ci_lower": delta_none_low, "delta_vs_none_ci_upper": delta_none_high,
            }))
            supported = (delta_all_low > 0) & (delta_none_low > 0)
            point_better = (model > treat_all) & (model > 0)
            ranges[task.upper()][pair] = {
                "point_estimate_better_than_treat_all_and_none": contiguous_ranges(thresholds, point_better),
                "paired_bootstrap_95ci_supports_better_than_both": contiguous_ranges(thresholds, supported),
            }
            axis.plot(thresholds, model, color=PAIR_COLORS[pair], lw=2, label=PAIR_LABELS[pair])
            axis.fill_between(thresholds, model_low, model_high, color=PAIR_COLORS[pair], alpha=0.10)
        axis.plot(thresholds, treat_all, color="#7A3E9D", ls="--", lw=1.7, label="Treat all")
        axis.axhline(0, color="black", ls=":", lw=1.5, label="Treat none")
        axis.set(title=task.upper(), xlabel="Threshold probability", ylabel="Net benefit")
        axis.grid(alpha=0.2); axis.legend(fontsize=8)
    figure.suptitle("Exploratory decision curves: selected multimodal fusion", fontsize=14)
    figure.tight_layout(); save_figure(figure, output / "selected_fusion_decision_curves")
    return pd.concat(tables, ignore_index=True), ranges


def all_method_dca_plots(frame: pd.DataFrame, output: Path, thresholds: np.ndarray) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(ARMS)))
    for pair in PAIRS:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, task in zip(axes, TASKS):
            pair_frame = frame[frame.modality_pair.eq(pair)]
            treat_all = None
            for color, arm in zip(colors, ARMS):
                y, p = task_arrays(pair_frame[pair_frame.arm.eq(arm)], task, calibrated=True)
                model, treat_all, _ = dca_values(y, p, thresholds)
                axis.plot(thresholds, model, color=color, lw=1.6, label=ARM_LABELS[arm])
            axis.plot(thresholds, treat_all, color="#7A3E9D", ls="--", lw=1.5, label="Treat all")
            axis.axhline(0, color="black", ls=":", lw=1.5, label="Treat none")
            axis.set(title=task.upper(), xlabel="Threshold probability", ylabel="Net benefit")
            axis.grid(alpha=0.2); axis.legend(fontsize=7)
        figure.suptitle(f"{PAIR_LABELS[pair]}: exploratory fusion-method decision curves", fontsize=14)
        figure.tight_layout(); save_figure(figure, output / f"fusion_methods_{pair}_decision_curves")


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold_min < args.threshold_max < 1:
        raise ValueError("Require 0 < threshold_min < threshold_max < 1.")
    root = args.multimodal_root.resolve()
    source = root / "combined_evaluation" / "all_multimodal_pooled_outer_test_predictions.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    output = root / "multimodal_evaluation_figures"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, dtype={PATIENT_ID: "string"})
    validate(
        frame,
        {"scd": args.expected_scd_patients, "pfd": args.expected_pfd_patients},
    )
    estimates = plot_selected_discrimination(frame, output)
    plot_method_discrimination(frame, output)
    calibration = plot_selected_calibration(frame, output, args.calibration_bins)
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_points)
    dca_table, ranges = selected_dca(frame, output, thresholds, args.bootstrap_replicates, args.seed)
    all_method_dca_plots(frame, output, thresholds)
    atomic_csv(output / "plotted_selected_fusion_discrimination.csv", estimates)
    atomic_csv(output / "selected_fusion_calibration_plot_points.csv", calibration)
    atomic_csv(output / "selected_fusion_decision_curve_summary.csv", dca_table)
    atomic_json(output / "selected_fusion_decision_curve_supported_ranges.json", ranges)
    atomic_json(output / "multimodal_figures_manifest.json", {
        "completed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_predictions": str(source), "source_predictions_sha256": sha256_file(source),
        "bootstrap_replicates_for_decision_curves": args.bootstrap_replicates,
        "seed": args.seed, "calibration_bins": args.calibration_bins,
        "independent_binary_tasks": True,
        "competing_endpoints_excluded": True,
        "decision_curve_threshold_range": [args.threshold_min, args.threshold_max],
        "decision_curve_interpretation": "exploratory unless an intervention-specific threshold range is justified",
        "discrimination_probability": "uncalibrated outer-test score",
        "calibration_and_dca_probability": "fold-specific Platt-calibrated outer-test probability",
    })
    print(f"Multimodal figures complete: {output}")


if __name__ == "__main__":
    main()
