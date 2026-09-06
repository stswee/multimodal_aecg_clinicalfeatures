#!/usr/bin/env python3
"""Post-analysis for completed MUSIC multimodal nested-CV predictions.

This script never trains, selects, recalibrates, or changes a threshold.  It
uses the already saved fold-specific outer-test predictions to add:

* calibration intercept, slope, and calibration-in-the-large (CITL), with
  patient-bootstrap confidence intervals for the primary selected-fusion arms;
* patient-bootstrap confidence intervals for threshold metrics for every arm;
* no cross-pair tests (this focused experiment contains one modality pair).
* a compact summary of fold-specific fusion selection and learned gate values.

ROC-AUC and PR-AUC comparisons use uncalibrated outer-test scores.  Brier-score
comparisons and calibration analyses use the fold-specific Platt-calibrated
outer-test probabilities produced inside the original nested-CV pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from tqdm.auto import tqdm


PATIENT_ID = "Patient ID"
TASKS = ("scd", "pfd")
EXPECTED_PAIRS = ("ecg_literature_reduced_tabular",)
EXPECTED_ARMS = (
    "concatenation",
    "projected_concatenation",
    "scalar_gating",
    "vector_gating",
    "weighted_sum",
    "selected_fusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multimodal_root", type=Path, required=True)
    parser.add_argument("--bootstrap_replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probability_clip", type=float, default=1e-6)
    parser.add_argument("--expected_scd_patients", type=int, default=648)
    parser.add_argument("--expected_pfd_patients", type=int, default=659)
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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False) as h:
        frame.to_csv(h, index=False)
        temporary = Path(h.name)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=path.parent, delete=False) as h:
        json.dump(value, h, indent=2)
        temporary = Path(h.name)
    temporary.replace(path)


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *map(str, parts)])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def validate_predictions(frame: pd.DataFrame, expected_by_task: dict[str, int]) -> None:
    required = {
        PATIENT_ID, "outer_fold", "modality_pair", "arm", "task",
        "y", "prob", "calibrated_prob", "threshold", "predicted",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pooled predictions are missing columns: {missing}")
    if frame.duplicated(["task", PATIENT_ID, "modality_pair", "arm"]).any():
        raise ValueError("Duplicate task/patient/modality-pair/arm rows were found.")
    if set(frame["task"]) != set(TASKS):
        raise ValueError(f"Unexpected tasks: {sorted(set(frame['task']))}")
    observed_pairs = set(frame["modality_pair"])
    observed_arms = set(frame["arm"])
    if observed_pairs != set(EXPECTED_PAIRS):
        raise ValueError(f"Unexpected modality pairs: {sorted(observed_pairs)}")
    if observed_arms != set(EXPECTED_ARMS):
        raise ValueError(f"Unexpected arms: {sorted(observed_arms)}")
    counts = frame.groupby(["task", "modality_pair", "arm"])[PATIENT_ID].nunique()
    for task, expected in expected_by_task.items():
        observed = counts.xs(task, level="task")
        if not observed.eq(expected).all():
            raise ValueError(f"Expected {expected} {task} patients in every pair/arm; got\n{observed}")
    for column in ("prob", "calibrated_prob"):
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid probabilities in {column}.")


def arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        group["y"].to_numpy(int),
        group["prob"].to_numpy(float),
        group["calibrated_prob"].to_numpy(float),
        group["predicted"].to_numpy(int),
    )


def calibration_metrics(y: np.ndarray, p: np.ndarray, clip: float) -> dict[str, float]:
    p = np.clip(p, clip, 1 - clip)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(x, y)

    logits = x[:, 0]
    def score(intercept: float) -> float:
        return float(np.sum(expit(logits + intercept)) - np.sum(y))

    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
        "calibration_in_the_large": float(brentq(score, -50, 50)),
        "mean_predicted_probability": float(np.mean(p)),
        "observed_event_rate": float(np.mean(y)),
    }


def bootstrap_calibration(
    y: np.ndarray, p: np.ndarray, replicates: int, seed: int, clip: float, description: str
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    point = calibration_metrics(y, p, clip)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in point}
    for _ in tqdm(range(replicates), desc=description, leave=False, dynamic_ncols=True):
        index = rng.integers(0, len(y), len(y))
        sampled_y = y[index]
        if np.unique(sampled_y).size < 2:
            continue
        try:
            result = calibration_metrics(sampled_y, p[index], clip)
        except (ValueError, RuntimeError):
            continue
        for key, value in result.items():
            if np.isfinite(value):
                values[key].append(value)
    valid = min(len(v) for v in values.values())
    if valid < max(100, math.floor(0.9 * replicates)):
        raise RuntimeError(f"Only {valid}/{replicates} valid calibration bootstraps for {description}.")
    intervals = {
        key: tuple(map(float, np.percentile(sample, [2.5, 97.5])))
        for key, sample in values.items()
    }
    return point, intervals, valid


def threshold_values(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    positive = y == 1
    negative = ~positive
    tp = int(np.sum(predicted[positive] == 1)); fn = int(np.sum(predicted[positive] == 0))
    tn = int(np.sum(predicted[negative] == 0)); fp = int(np.sum(predicted[negative] == 1))
    safe = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    sensitivity = safe(tp, tp + fn); specificity = safe(tn, tn + fp)
    ppv = safe(tp, tp + fp); npv = safe(tn, tn + fn)
    f1 = safe(2 * tp, 2 * tp + fp + fn)
    return {
        "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def bootstrap_threshold_metrics(
    y: np.ndarray, predicted: np.ndarray, replicates: int, seed: int, description: str
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    point = threshold_values(y, predicted)
    rng = np.random.default_rng(seed)
    metric_names = ("sensitivity", "specificity", "ppv", "npv", "f1")
    values = {key: [] for key in metric_names}
    valid = 0
    for _ in tqdm(range(replicates), desc=description, leave=False, dynamic_ncols=True):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size < 2:
            continue
        result = threshold_values(y[index], predicted[index]); valid += 1
        for key in metric_names:
            if np.isfinite(result[key]):
                values[key].append(result[key])
    intervals = {
        key: tuple(map(float, np.percentile(sample, [2.5, 97.5])))
        if sample else (np.nan, np.nan)
        for key, sample in values.items()
    }
    return point, intervals, valid


def performance(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, raw)),
        "pr_auc": float(average_precision_score(y, raw)),
        "brier": float(brier_score_loss(y, calibrated)),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values); result = np.empty(len(p_values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        result[index] = min(running, 1.0)
    return result


def paired_selected_comparisons(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Cross-pair comparisons are intentionally absent in this one-pair run."""
    if len(EXPECTED_PAIRS) < 2:
        return pd.DataFrame(columns=[
            "outcome", "left_model", "right_model", "metric",
            "difference_left_minus_right", "ci_lower", "ci_upper",
            "p_value_raw", "valid_bootstrap_replicates",
            "p_value_holm_all_selected_fusion_comparisons",
            "difference_favors_left_when",
        ])
    selected = frame[frame["arm"].eq("selected_fusion")].copy()
    comparisons: tuple[tuple[str, str], ...] = ()
    rows = []
    for task in TASKS:
        task_frame = selected[selected["task"].eq(task)]
        by_pair = {
            pair: task_frame[task_frame["modality_pair"].eq(pair)]
            .sort_values(PATIENT_ID)
            .reset_index(drop=True)
            for pair in EXPECTED_PAIRS
        }
        base_ids = by_pair[EXPECTED_PAIRS[0]][PATIENT_ID]
        if not all(item[PATIENT_ID].equals(base_ids) for item in by_pair.values()):
            raise ValueError(
                f"Selected-fusion patient order cannot be aligned for {task}."
            )
        y = by_pair[EXPECTED_PAIRS[0]]["y"].to_numpy(int)
        rng = np.random.default_rng(stable_seed(seed, "paired", task))
        bootstrap_indices = rng.integers(0, len(y), size=(replicates, len(y)))
        for left, right in comparisons:
            left_raw = by_pair[left]["prob"].to_numpy(float)
            right_raw = by_pair[right]["prob"].to_numpy(float)
            left_cal = by_pair[left]["calibrated_prob"].to_numpy(float)
            right_cal = by_pair[right]["calibrated_prob"].to_numpy(float)
            specifications = (
                ("roc_auc", left_raw, right_raw, roc_auc_score),
                ("pr_auc", left_raw, right_raw, average_precision_score),
                ("brier", left_cal, right_cal, brier_score_loss),
            )
            for metric, left_values, right_values, function in specifications:
                point = function(y, left_values) - function(y, right_values)
                differences = []
                for index in tqdm(
                    bootstrap_indices,
                    desc=f"Paired {task}/{left}-{right}/{metric}",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    sampled_y = y[index]
                    if np.unique(sampled_y).size < 2:
                        continue
                    differences.append(
                        function(sampled_y, left_values[index])
                        - function(sampled_y, right_values[index])
                    )
                differences = np.asarray(differences, dtype=float)
                lower, upper = np.percentile(differences, [2.5, 97.5])
                p_value = min(
                    1.0,
                    2 * min(
                        (np.sum(differences <= 0) + 1) / (len(differences) + 1),
                        (np.sum(differences >= 0) + 1) / (len(differences) + 1),
                    ),
                )
                rows.append({
                    "outcome": task, "left_model": left, "right_model": right,
                    "metric": metric, "difference_left_minus_right": float(point),
                    "ci_lower": float(lower), "ci_upper": float(upper),
                    "p_value_raw": float(p_value),
                    "valid_bootstrap_replicates": int(len(differences)),
                })
    result = pd.DataFrame(rows)
    result["p_value_holm_all_selected_fusion_comparisons"] = holm_adjust(
        result["p_value_raw"].to_numpy(float)
    )
    result["difference_favors_left_when"] = np.where(
        result["metric"].eq("brier"), "negative", "positive"
    )
    return result


def gate_and_selection_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows, gate_rows = [], []
    for completion in sorted(root.glob("tasks/*/final_models/*/outer_fold_*/run_complete.json")):
        data = json.loads(completion.read_text())
        pair = completion.parts[-3]; fold = int(completion.parts[-2].split("_")[-1])
        method = data["selected_overall_method"]
        selection_rows.append({"outcome": data["task"], "modality_pair": pair, "outer_fold": fold, "selected_method": method})
    for completion in sorted(root.glob("tasks/*/final_models/*/outer_fold_*/arms/*/run_complete.json")):
        data = json.loads(completion.read_text()); stats = data.get("gate_stats") or {}
        if stats:
            gate_rows.append({
                "outcome": data["task"], "modality_pair": data["modality_pair"], "outer_fold": data["outer_fold"],
                "arm": data["arm"], **stats,
                "gate_convention": "gate * ECG + (1 - gate) * second modality",
            })
    return pd.DataFrame(selection_rows), pd.DataFrame(gate_rows)


def main() -> None:
    args = parse_args(); root = args.multimodal_root.resolve()
    source = root / "combined_evaluation" / "all_multimodal_pooled_outer_test_predictions.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    output = root / "multimodal_posthoc_evaluation"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, dtype={PATIENT_ID: "string"})
    validate_predictions(
        frame,
        {"scd": args.expected_scd_patients, "pfd": args.expected_pfd_patients},
    )

    calibration_point_rows, calibration_ci_rows, threshold_rows = [], [], []
    groups = list(frame.groupby(["task", "modality_pair", "arm"], sort=True))
    for (task, pair, arm), group in tqdm(groups, desc="Posthoc arm evaluation", dynamic_ncols=True):
            y, _, calibrated, predicted = arrays(group)
            point_calibration = calibration_metrics(y, calibrated, args.probability_clip)
            for metric, estimate in point_calibration.items():
                calibration_point_rows.append({
                    "modality_pair": pair, "arm": arm, "outcome": task,
                    "metric": metric, "estimate": estimate,
                    "patients": len(y), "events": int(y.sum()),
                })
            if arm == "selected_fusion":
                point, intervals, valid = bootstrap_calibration(
                    y, calibrated, args.bootstrap_replicates,
                    stable_seed(args.seed, "calibration", pair, task), args.probability_clip,
                    f"Calibration {pair}/{task}",
                )
                for metric, estimate in point.items():
                    calibration_ci_rows.append({
                        "modality_pair": pair, "arm": arm, "outcome": task,
                        "metric": metric, "estimate": estimate,
                        "ci_lower": intervals[metric][0], "ci_upper": intervals[metric][1],
                        "valid_bootstrap_replicates": valid,
                        "patients": len(y), "events": int(y.sum()),
                    })
            point, intervals, valid = bootstrap_threshold_metrics(
                y, predicted, args.bootstrap_replicates,
                stable_seed(args.seed, "threshold", pair, arm, task),
                f"Threshold {pair}/{arm}/{task}",
            )
            threshold = float(group["threshold"].mean())
            row = {
                "modality_pair": pair, "arm": arm, "outcome": task,
                "mean_fold_specific_threshold": threshold,
                "patients": len(y), "events": int(y.sum()),
                "valid_bootstrap_replicates": valid,
                "tn": point["tn"], "fp": point["fp"], "fn": point["fn"], "tp": point["tp"],
            }
            for metric in ("sensitivity", "specificity", "ppv", "npv", "f1"):
                row[metric] = point[metric]
                row[f"{metric}_ci_lower"] = intervals[metric][0]
                row[f"{metric}_ci_upper"] = intervals[metric][1]
            threshold_rows.append(row)

    paired = paired_selected_comparisons(frame, args.bootstrap_replicates, args.seed)
    selections, gates = gate_and_selection_summary(root)
    atomic_csv(output / "calibration_point_estimates_all_arms.csv", pd.DataFrame(calibration_point_rows))
    atomic_csv(output / "selected_fusion_calibration_with_95ci.csv", pd.DataFrame(calibration_ci_rows))
    atomic_csv(output / "threshold_metrics_all_arms_with_95ci.csv", pd.DataFrame(threshold_rows))
    atomic_csv(output / "selected_fusion_paired_comparisons_with_95ci.csv", paired)
    atomic_csv(output / "selected_fusion_method_by_outer_fold.csv", selections)
    atomic_csv(output / "learned_gate_summary_by_outer_fold.csv", gates)
    atomic_json(output / "posthoc_evaluation_manifest.json", {
        "completed": True, "created_at_utc": utc_now(),
        "source_predictions": str(source), "source_predictions_sha256": sha256_file(source),
        "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed,
        "independent_binary_tasks": True,
        "competing_endpoints_excluded": True,
        "calibration_ci_scope": "selected_fusion arms; point estimates supplied for all arms",
        "threshold_ci_scope": "all modality-pair/arm combinations",
        "paired_comparison_scope": "none; focused run contains one modality pair",
        "outer_test_outcomes_used_for_training_selection_calibration_or_threshold_selection": False,
        "note": "All analyses are final evaluation of previously generated outer-test predictions.",
    })
    print(f"Posthoc evaluation complete: {output}")


if __name__ == "__main__":
    main()
