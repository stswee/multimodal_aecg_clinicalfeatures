#!/usr/bin/env python3
"""Create publication-ready multimodal ROC, PR, calibration, and DCA figures.

The script reads only the pooled untouched outer-test predictions from the
completed detailed-response multimodal nested-CV run. Discrimination curves use uncalibrated
scores; calibration and decision curves use the saved fold-specific Platt-
calibrated probabilities.  Decision-curve uncertainty uses paired patient
bootstrap resampling and includes treat-all and treat-none strategies.

When --comparative_predictions is supplied, the principal DCA uses the five
representative configurations displayed in manuscript Figures 2-3: ECG only,
LLaMA8B-BioBERT, tabular only, ECG + tabular (DC), and
ECG + LLaMA8B-BioBERT (DC).
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
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PATIENT_ID = "Patient ID"
TASKS = ("scd", "pfd")
PAIRS = ("ecg_full_text", "ecg_deterministic_text", "ecg_tabular")
ARMS = (
    "concatenation", "projected_concatenation", "scalar_gating",
    "vector_gating", "weighted_sum", "selected_fusion",
)
PAIR_LABELS = {
    "ecg_full_text": "ECG + full LLM text",
    "ecg_deterministic_text": "ECG + deterministic text",
    "ecg_tabular": "ECG + Tabular",
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
    "ecg_full_text": "#D55E00",
    "ecg_deterministic_text": "#0072B2",
    "ecg_tabular": "#009E73",
}

# Keep this order and these colors synchronized with manuscript Figures 2-3.
PRINCIPAL_DCA_MODELS = (
    "ecg",
    "full_llm_text",
    "tabular",
    "ecg_tabular",
    "ecg_full_llm",
)
PRINCIPAL_DCA_LABELS = {
    "ecg": "ECG only",
    "full_llm_text": "LLaMA8B-BioBERT",
    "tabular": "Tabular only",
    "ecg_tabular": "ECG + Tabular (DC)",
    "ecg_full_llm": "ECG + LLaMA8B-BioBERT (DC)",
}
PRINCIPAL_DCA_COLORS = {
    "ecg": "#1f77b4",
    "full_llm_text": "#ff7f0e",
    "tabular": "#2ca02c",
    "ecg_tabular": "#9467bd",
    "ecg_full_llm": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multimodal_root", type=Path, required=True)
    parser.add_argument(
        "--comparative_predictions",
        type=Path,
        default=None,
        help=(
            "Optional standardized_principal_outer_test_predictions.csv. "
            "When supplied, the principal DCA contains ECG only, "
            "LLaMA8B-BioBERT, tabular only, ECG + tabular (DC), and "
            "ECG + LLaMA8B-BioBERT (DC), matching manuscript Figures 2-3."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to MULTIMODAL_ROOT/multimodal_evaluation_figures.",
    )
    parser.add_argument(
        "--dca_only",
        action="store_true",
        help=(
            "Generate only the five-model principal decision-curve figure and tables. "
            "Requires --comparative_predictions and does not read multimodal combined-evaluation files."
        ),
    )
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


def validate_analysis_manifest(root: Path, expected_seed: int) -> tuple[Path, dict]:
    """Reject incomplete or legacy short-response multimodal roots."""
    path = root / "analysis_setup" / "analysis_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing multimodal analysis manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_policy = {
        "pooling": "cls",
        "max_length": 512,
        "long_text_strategy": "mean_chunks",
        "truncated_patient_count": 0,
    }
    expected_values = {
        "completed": True,
        "patients": 730,
        "controls": 577,
        "scd": 71,
        "pfd": 82,
        "outer_folds": 5,
        "inner_folds": 4,
        "seed": expected_seed,
        "independent_binary_tasks": True,
        "expected_text_embedding_policy": expected_policy,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{path}: expected {key}={expected!r}; observed {manifest.get(key)!r}."
            )
    return path, manifest


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
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
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
        raise ValueError("The pooled file does not contain the expected three pairs and six arms.")
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


def canonical_patient_id(value: object) -> str:
    """Normalize numeric-looking patient identifiers without exposing them."""
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def load_principal_dca_predictions(
    path: Path,
    expected: dict[str, int],
) -> pd.DataFrame:
    """Load long- or wide-format standardized comparative predictions."""
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        dtype={PATIENT_ID: "string", "patient_key": "string"},
    )
    patient_column = "patient_key" if "patient_key" in frame else PATIENT_ID
    if patient_column not in frame:
        raise ValueError("Comparative predictions lack patient_key or Patient ID.")
    frame["patient_key"] = frame[patient_column].map(canonical_patient_id)

    wide_required = {
        "model", "y_scd", "y_pfd", "calibrated_scd", "calibrated_pfd",
    }
    long_required = {"model", "task", "y", "calibrated"}
    if wide_required.issubset(frame.columns):
        parts = []
        for task in ("scd", "pfd"):
            part = frame[
                ["model", "patient_key", f"y_{task}", f"calibrated_{task}"]
            ].copy()
            part.columns = ["model", "patient_key", "y", "calibrated"]
            part["task"] = task
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
    elif long_required.issubset(frame.columns):
        frame = frame[["model", "patient_key", "task", "y", "calibrated"]].copy()
        frame["task"] = frame["task"].astype(str).str.lower()
    else:
        raise ValueError(
            "Comparative predictions must contain either wide columns "
            "(model, y_scd, y_pfd, calibrated_scd, calibrated_pfd) or long "
            "columns (model, task, y, calibrated)."
        )

    absent_models = set(PRINCIPAL_DCA_MODELS) - set(frame["model"].astype(str))
    if absent_models:
        raise ValueError(
            f"Comparative predictions lack required models: {sorted(absent_models)}"
        )
    frame = frame.loc[
        frame["model"].isin(PRINCIPAL_DCA_MODELS)
        & frame["task"].isin(("scd", "pfd"))
        & frame["y"].notna()
    ].copy()
    if frame.duplicated(["model", "task", "patient_key"]).any():
        raise ValueError("Duplicate model/task/patient rows in comparative predictions.")

    for task, expected_patients in expected.items():
        reference = frame.loc[
            frame["model"].eq("ecg") & frame["task"].eq(task)
        ].set_index("patient_key")
        known_ids = reference.index
        if len(known_ids) != expected_patients:
            raise ValueError(
                f"Expected {expected_patients} evaluable {task.upper()} patients; "
                f"found {len(known_ids)}."
            )
        reference_y = pd.to_numeric(
            reference.loc[known_ids, "y"], errors="raise"
        ).to_numpy(int)
        for model in PRINCIPAL_DCA_MODELS:
            part = frame.loc[
                frame["model"].eq(model) & frame["task"].eq(task)
            ].set_index("patient_key")
            missing_ids = known_ids.difference(part.index)
            if len(missing_ids):
                raise ValueError(
                    f"{model} lacks {len(missing_ids)} patients for {task.upper()}."
                )
            model_y = pd.to_numeric(
                part.loc[known_ids, "y"], errors="raise"
            ).to_numpy(int)
            if not np.array_equal(reference_y, model_y):
                raise ValueError(f"Outcome mismatch for {model}/{task}.")
            probabilities = pd.to_numeric(
                part.loc[known_ids, "calibrated"], errors="raise"
            ).to_numpy(float)
            if not np.isfinite(probabilities).all() or np.any(
                (probabilities < 0) | (probabilities > 1)
            ):
                raise ValueError(
                    f"Invalid calibrated probabilities for {model}/{task}."
                )
    return frame


def principal_dca_arrays(
    frame: pd.DataFrame,
    model: str,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    reference = (
        frame.loc[frame["model"].eq("ecg") & frame["task"].eq(task)]
        .set_index("patient_key")
        .sort_index()
    )
    ids = reference.index
    part = frame.loc[
        frame["model"].eq(model) & frame["task"].eq(task)
    ].set_index("patient_key").loc[ids]
    return (
        pd.to_numeric(part["y"], errors="raise").to_numpy(int),
        pd.to_numeric(part["calibrated"], errors="raise").to_numpy(float),
    )


def principal_dca(
    frame: pd.DataFrame,
    output: Path,
    thresholds: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Plot the five representative models used in manuscript Figures 2-3."""
    tables: list[pd.DataFrame] = []
    ranges: dict[str, dict] = {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for task_index, (axis, task) in enumerate(zip(axes, TASKS)):
        ranges[task.upper()] = {}
        reference_y, _ = principal_dca_arrays(frame, "ecg", task)
        odds = thresholds / (1 - thresholds)
        prevalence = float(np.mean(reference_y))
        treat_all = prevalence - (1 - prevalence) * odds

        # The same bootstrap patient weights are used for every model within an
        # endpoint, preserving paired uncertainty across model curves.
        rng = np.random.default_rng(seed + task_index * 1000)
        weights = rng.multinomial(
            len(reference_y),
            np.full(len(reference_y), 1 / len(reference_y)),
            size=replicates,
        )
        bootstrap_prevalence = (weights @ reference_y) / len(reference_y)
        bootstrap_all = (
            bootstrap_prevalence[:, None]
            - (1 - bootstrap_prevalence[:, None]) * odds[None, :]
        )

        for model in PRINCIPAL_DCA_MODELS:
            y, probability = principal_dca_arrays(frame, model, task)
            model_nb, model_treat_all, positive = dca_values(
                y, probability, thresholds
            )
            if not np.allclose(model_treat_all, treat_all):
                raise ValueError(f"Prevalence mismatch for {model}/{task}.")

            positives = positive.astype(np.int16)
            bootstrap_tp = weights @ (
                positives * y[:, None]
            ).astype(np.int16)
            bootstrap_fp = weights @ (
                positives * (1 - y[:, None])
            ).astype(np.int16)
            bootstrap_model = (
                bootstrap_tp / len(y)
                - bootstrap_fp / len(y) * odds[None, :]
            )
            model_low, model_high = np.percentile(
                bootstrap_model, [2.5, 97.5], axis=0
            )
            delta_all = bootstrap_model - bootstrap_all
            delta_all_low, delta_all_high = np.percentile(
                delta_all, [2.5, 97.5], axis=0
            )
            delta_none_low, delta_none_high = np.percentile(
                bootstrap_model, [2.5, 97.5], axis=0
            )

            tables.append(pd.DataFrame({
                "model": model,
                "model_label": PRINCIPAL_DCA_LABELS[model],
                "outcome": task,
                "threshold_probability": thresholds,
                "model_net_benefit": model_nb,
                "model_ci_lower": model_low,
                "model_ci_upper": model_high,
                "treat_all_net_benefit": treat_all,
                "treat_none_net_benefit": 0.0,
                "delta_model_vs_all": model_nb - treat_all,
                "delta_vs_all_ci_lower": delta_all_low,
                "delta_vs_all_ci_upper": delta_all_high,
                "delta_model_vs_none": model_nb,
                "delta_vs_none_ci_lower": delta_none_low,
                "delta_vs_none_ci_upper": delta_none_high,
            }))

            point_better = (model_nb > treat_all) & (model_nb > 0)
            supported = (delta_all_low > 0) & (delta_none_low > 0)
            ranges[task.upper()][model] = {
                "point_estimate_better_than_treat_all_and_none": (
                    contiguous_ranges(thresholds, point_better)
                ),
                "paired_bootstrap_95ci_supports_better_than_both": (
                    contiguous_ranges(thresholds, supported)
                ),
            }

            color = PRINCIPAL_DCA_COLORS[model]
            axis.plot(
                thresholds,
                model_nb,
                color=color,
                lw=2,
                label=PRINCIPAL_DCA_LABELS[model],
            )
            axis.fill_between(
                thresholds,
                model_low,
                model_high,
                color=color,
                alpha=0.08,
                linewidth=0,
            )

        # Retain the existing treat-all and treat-none styling.
        axis.plot(
            thresholds,
            treat_all,
            color="#7f7f7f",
            ls="--",
            lw=1.7,
            label="Treat all",
        )
        axis.axhline(
            0,
            color="black",
            ls=":",
            lw=1.5,
            label="Treat none",
        )
        axis.set(
            title=task.upper(),
            xlabel="Threshold probability",
            ylabel="Net benefit",
        )
        axis.set_xlim(float(thresholds.min()), float(thresholds.max()))
        axis.legend(fontsize=8, frameon=True)

    figure.tight_layout()
    save_figure(figure, output / "principal_exploratory_decision_curves")
    return pd.concat(tables, ignore_index=True), ranges


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
        axis.plot(thresholds, treat_all, color="#7f7f7f", ls="--", lw=1.7, label="Treat all")
        axis.axhline(0, color="black", ls=":", lw=1.5, label="Treat none")
        axis.set(title=task.upper(), xlabel="Threshold probability", ylabel="Net benefit")
        axis.legend(fontsize=8)
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
            axis.plot(thresholds, treat_all, color="#7f7f7f", ls="--", lw=1.5, label="Treat all")
            axis.axhline(0, color="black", ls=":", lw=1.5, label="Treat none")
            axis.set(title=task.upper(), xlabel="Threshold probability", ylabel="Net benefit")
            axis.legend(fontsize=7)
        figure.suptitle(f"{PAIR_LABELS[pair]}: exploratory fusion-method decision curves", fontsize=14)
        figure.tight_layout(); save_figure(figure, output / f"fusion_methods_{pair}_decision_curves")


def main() -> None:
    args = parse_args()
    if not 0 < args.threshold_min < args.threshold_max < 1:
        raise ValueError("Require 0 < threshold_min < threshold_max < 1.")
    root = args.multimodal_root.resolve()
    analysis_manifest_path, analysis_manifest = validate_analysis_manifest(
        root, args.seed
    )
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "multimodal_evaluation_figures"
    )
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_points)

    if args.dca_only:
        if args.comparative_predictions is None:
            raise ValueError("--dca_only requires --comparative_predictions.")
        comparative_path = args.comparative_predictions.resolve()
        comparative = load_principal_dca_predictions(
            comparative_path,
            {
                "scd": args.expected_scd_patients,
                "pfd": args.expected_pfd_patients,
            },
        )
        dca_table, ranges = principal_dca(
            comparative,
            output,
            thresholds,
            args.bootstrap_replicates,
            args.seed,
        )
        atomic_csv(output / "principal_exploratory_decision_curve_summary.csv", dca_table)
        atomic_json(
            output / "principal_exploratory_decision_curve_supported_ranges.json",
            ranges,
        )
        atomic_json(output / "principal_decision_curve_manifest.json", {
            "completed": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "comparative_predictions": str(comparative_path),
            "comparative_predictions_sha256": sha256_file(comparative_path),
            "multimodal_analysis_manifest": str(analysis_manifest_path),
            "multimodal_analysis_manifest_sha256": sha256_file(
                analysis_manifest_path
            ),
            "text_embedding_policy": analysis_manifest[
                "expected_text_embedding_policy"
            ],
            "principal_dca_models": list(PRINCIPAL_DCA_MODELS),
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "decision_curve_threshold_range": [args.threshold_min, args.threshold_max],
            "decision_curve_interpretation": "exploratory unless an intervention-specific threshold range is justified",
            "probability_source": "fold-specific calibrated outer-test probabilities",
        })
        print(f"Principal decision curves complete: {output}")
        return

    source = root / "combined_evaluation" / "all_multimodal_pooled_outer_test_predictions.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, dtype={PATIENT_ID: "string"})
    validate(
        frame,
        {"scd": args.expected_scd_patients, "pfd": args.expected_pfd_patients},
    )
    estimates = plot_selected_discrimination(frame, output)
    plot_method_discrimination(frame, output)
    calibration = plot_selected_calibration(frame, output, args.calibration_bins)
    if args.comparative_predictions is not None:
        comparative = load_principal_dca_predictions(
            args.comparative_predictions.resolve(),
            {
                "scd": args.expected_scd_patients,
                "pfd": args.expected_pfd_patients,
            },
        )
        dca_table, ranges = principal_dca(
            comparative,
            output,
            thresholds,
            args.bootstrap_replicates,
            args.seed,
        )
        dca_stem = "principal_exploratory_decision_curve"
    else:
        dca_table, ranges = selected_dca(
            frame,
            output,
            thresholds,
            args.bootstrap_replicates,
            args.seed,
        )
        dca_stem = "selected_fusion_decision_curve"
    all_method_dca_plots(frame, output, thresholds)
    atomic_csv(output / "plotted_selected_fusion_discrimination.csv", estimates)
    atomic_csv(output / "selected_fusion_calibration_plot_points.csv", calibration)
    atomic_csv(output / f"{dca_stem}_summary.csv", dca_table)
    atomic_json(output / f"{dca_stem}_supported_ranges.json", ranges)
    atomic_json(output / "multimodal_figures_manifest.json", {
        "completed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_predictions": str(source), "source_predictions_sha256": sha256_file(source),
        "multimodal_analysis_manifest": str(analysis_manifest_path),
        "multimodal_analysis_manifest_sha256": sha256_file(
            analysis_manifest_path
        ),
        "text_embedding_policy": analysis_manifest[
            "expected_text_embedding_policy"
        ],
        "comparative_predictions": (
            str(args.comparative_predictions.resolve())
            if args.comparative_predictions is not None else None
        ),
        "principal_dca_models": (
            list(PRINCIPAL_DCA_MODELS)
            if args.comparative_predictions is not None else None
        ),
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
