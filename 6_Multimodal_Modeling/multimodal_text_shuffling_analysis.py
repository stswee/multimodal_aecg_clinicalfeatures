#!/usr/bin/env python3
"""Patient-text correspondence permutation test for detailed-response models.

Within each untouched endpoint-specific outer fold, this script permutes the
second-modality text embedding across eligible patients while holding ECG
embeddings, outcomes, model weights, fold membership, Platt mappings, and all
preprocessing fixed. It applies the already trained fold-selected one-head
fusion model without retraining. The resulting null distribution tests whether
correctly paired text contributes patient-specific information beyond the ECG
branch for the active binary endpoint.

The script requires the original server-side checkpoints and embedding roots.
The compact result-sharing archive intentionally does not contain checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from tqdm.auto import tqdm

import train_multimodal_nested_cv as mm


TEXT_PAIRS = ("ecg_full_text", "ecg_deterministic_text")
TASKS = ("scd", "pfd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds_csv", type=Path, required=True)
    parser.add_argument("--ecg_root", type=Path, required=True)
    parser.add_argument("--text_embedding_root", type=Path, required=True)
    parser.add_argument("--text_results_root", type=Path, required=True)
    parser.add_argument("--tabular_csv", type=Path, required=True)
    parser.add_argument("--multimodal_root", type=Path, required=True)
    parser.add_argument("--permutation_replicates", type=int, default=1000)
    parser.add_argument("--permutation_batch", type=int, default=32)
    parser.add_argument("--inference_batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probability_clip", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *map(str, parts)])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2**31 - 1)


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


def loader_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        stage="final", folds_csv=args.folds_csv.resolve(), ecg_root=args.ecg_root.resolve(),
        text_embedding_root=args.text_embedding_root.resolve(),
        text_results_root=args.text_results_root.resolve(), tabular_csv=args.tabular_csv.resolve(),
        output_root=args.multimodal_root.resolve(), patient_id_col=mm.PATIENT_ID,
        scd_label_col=mm.SCD_LABEL, pfd_label_col=mm.PFD_LABEL,
        outer_fold_col=mm.OUTER_FOLD, outer_splits=5, inner_splits=4,
        expected_patients=730, expected_controls=577, expected_scd=71, expected_pfd=82,
        expected_anticoagulant_yes=610, expected_anticoagulant_no=120,
        expected_text_pooling="cls", expected_text_max_length=512,
        expected_text_long_strategy="mean_chunks",
    )


def validate_analysis_manifest(args: argparse.Namespace) -> tuple[Path, dict]:
    root = args.multimodal_root.resolve()
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
        "seed": args.seed,
        "independent_binary_tasks": True,
        "expected_text_embedding_policy": expected_policy,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{path}: expected {key}={expected!r}; observed {manifest.get(key)!r}."
            )
    recorded_roots = {
        "text_embedding_root": args.text_embedding_root.resolve(),
        "text_results_root": args.text_results_root.resolve(),
        "ecg_root": args.ecg_root.resolve(),
    }
    for key, expected in recorded_roots.items():
        if Path(manifest.get(key, "")).resolve() != expected:
            raise ValueError(f"{path}: {key} does not match the supplied input root.")
    return path, manifest


def load_selected_model(
    root: Path, task: str, pair: str, fold: int, device: torch.device
) -> tuple[mm.FusionNetwork, str, dict]:
    fold_directory = root / "tasks" / task / "final_models" / pair / f"outer_fold_{fold}"
    fold_completion = json.loads((fold_directory / "run_complete.json").read_text())
    method = fold_completion["selected_overall_method"]
    arm_directory = fold_directory / "arms" / method
    checkpoint_path = arm_directory / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run shuffling on the original server output, "
            "not the compact result-sharing archive."
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = mm.FusionNetwork(checkpoint["ecg_dim"], checkpoint["second_dim"], checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"]); model.to(device); model.eval()
    mapping = json.loads((arm_directory / "platt_and_thresholds.json").read_text())["platt"]
    return model, method, mapping


def apply_mapping(probabilities: np.ndarray, mapping: dict, clip: float) -> np.ndarray:
    p = np.clip(probabilities, clip, 1 - clip); logit = np.log(p / (1 - p))
    value = mapping["intercept"] + mapping["slope"] * logit
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def permuted_fold_predictions(
    model: mm.FusionNetwork,
    ecg: np.ndarray,
    text: np.ndarray,
    mapping: dict,
    replicates: int,
    permutation_batch: int,
    inference_batch_size: int,
    seed: int,
    device: torch.device,
    clip: float,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.empty((replicates, len(ecg)), dtype=np.float32)
    calibrated = np.empty_like(raw)
    rng = np.random.default_rng(seed)
    iterator = tqdm(range(0, replicates, permutation_batch), desc=description, dynamic_ncols=True)
    for start in iterator:
        stop = min(start + permutation_batch, replicates); count = stop - start
        permutations = np.stack([rng.permutation(len(text)) for _ in range(count)])
        repeated_ecg = np.tile(ecg, (count, 1))
        shuffled_text = text[permutations].reshape(count * len(text), text.shape[1])
        probabilities, _ = mm.predict_model(
            model, repeated_ecg, shuffled_text, device, inference_batch_size
        )
        probabilities = probabilities.reshape(count, len(text))
        raw[start:stop] = probabilities
        calibrated[start:stop] = apply_mapping(probabilities, mapping, clip)
    return raw, calibrated


def metric_values(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, raw)),
        "pr_auc": float(average_precision_score(y, raw)),
        "brier": float(brier_score_loss(y, calibrated)),
    }


def correctly_paired_bootstrap_intervals(
    y: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    replicates: int,
    seed: int,
    description: str,
) -> tuple[dict[str, tuple[float, float]], int]:
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in ("roc_auc", "pr_auc", "brier")}
    valid = 0
    for _ in tqdm(range(replicates), desc=description, leave=False, dynamic_ncols=True):
        index = rng.integers(0, len(y), len(y))
        sampled_y = y[index]
        if np.unique(sampled_y).size < 2:
            continue
        metrics = metric_values(sampled_y, raw[index], calibrated[index])
        for metric, value in metrics.items():
            values[metric].append(value)
        valid += 1
    if valid < max(100, int(0.9 * replicates)):
        raise RuntimeError(
            f"Only {valid}/{replicates} valid correctly paired bootstraps for {description}."
        )
    intervals = {
        metric: tuple(map(float, np.percentile(samples, [2.5, 97.5])))
        for metric, samples in values.items()
    }
    return intervals, valid


def main() -> None:
    args = parse_args(); root = args.multimodal_root.resolve(); device = torch.device(args.device)
    analysis_manifest_path, analysis_manifest = validate_analysis_manifest(args)
    output = root / "multimodal_text_shuffling"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)
    loader_args = loader_namespace(args); folds = mm.read_folds(loader_args, setup_copy=False)
    pooled_path = root / "combined_evaluation" / "all_multimodal_pooled_outer_test_predictions.csv"
    pooled = pd.read_csv(pooled_path, dtype={mm.PATIENT_ID: "string"})
    summary_rows, distribution_rows, fold_audit = [], [], []

    for task in TASKS:
        for pair in TEXT_PAIRS:
            raw_by_fold, calibrated_by_fold = [], []
            labels_by_fold, observed_raw_by_fold, observed_calibrated_by_fold = [], [], []
            for fold in range(5):
                data = mm.load_pair_split(
                    loader_args, folds, pair, task, fold, None, None
                )
                model, method, mapping = load_selected_model(
                    root, task, pair, fold, device
                )
                observed_raw, _ = mm.predict_model(
                    model,
                    data.ecg_validation,
                    data.second_validation,
                    device,
                    args.inference_batch_size,
                )
                saved = pooled[
                    pooled.task.eq(task)
                    & pooled.modality_pair.eq(pair)
                    & pooled.arm.eq("selected_fusion")
                    & pooled.outer_fold.eq(fold)
                ].copy()
                saved["__key"] = saved[mm.PATIENT_ID].map(mm.canonical_patient_id)
                saved = saved.set_index("__key").loc[
                    [mm.canonical_patient_id(value) for value in data.patient_ids_validation]
                ]
                saved_raw = saved["prob"].to_numpy(float)
                max_difference = float(np.max(np.abs(observed_raw - saved_raw)))
                if max_difference > 1e-5:
                    raise RuntimeError(
                        f"Checkpoint reproduction failed for {task}/{pair}/fold {fold}: "
                        f"max difference {max_difference}"
                    )
                shuffled_raw, shuffled_calibrated = permuted_fold_predictions(
                    model,
                    data.ecg_validation,
                    data.second_validation,
                    mapping,
                    args.permutation_replicates,
                    args.permutation_batch,
                    args.inference_batch_size,
                    stable_seed(args.seed, task, pair, fold),
                    device,
                    args.probability_clip,
                    f"Shuffle {task.upper()} {pair} fold {fold}",
                )
                raw_by_fold.append(shuffled_raw)
                calibrated_by_fold.append(shuffled_calibrated)
                labels_by_fold.append(data.labels_validation)
                observed_raw_by_fold.append(saved_raw)
                observed_calibrated_by_fold.append(
                    saved["calibrated_prob"].to_numpy(float)
                )
                fold_audit.append({
                    "outcome": task, "modality_pair": pair, "outer_fold": fold,
                    "selected_method": method,
                    "patients": len(data.patient_ids_validation),
                    "checkpoint_reproduction_max_absolute_probability_difference": max_difference,
                })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            shuffled_raw = np.concatenate(raw_by_fold, axis=1)
            shuffled_calibrated = np.concatenate(calibrated_by_fold, axis=1)
            y = np.concatenate(labels_by_fold).astype(int)
            observed_raw = np.concatenate(observed_raw_by_fold)
            observed_calibrated = np.concatenate(observed_calibrated_by_fold)
            observed_metrics = metric_values(y, observed_raw, observed_calibrated)
            observed_intervals, valid_observed_bootstraps = (
                correctly_paired_bootstrap_intervals(
                    y,
                    observed_raw,
                    observed_calibrated,
                    args.permutation_replicates,
                    stable_seed(args.seed, "correctly_paired", task, pair),
                    f"Correctly paired bootstrap {pair}/{task}",
                )
            )
            null_values = {metric: [] for metric in observed_metrics}
            for replicate in tqdm(range(args.permutation_replicates), desc=f"Metrics {pair}/{task}", dynamic_ncols=True):
                metrics = metric_values(
                    y, shuffled_raw[replicate], shuffled_calibrated[replicate],
                )
                for metric, value in metrics.items():
                    null_values[metric].append(value)
                    distribution_rows.append({
                        "modality_pair": pair, "outcome": task, "metric": metric,
                        "permutation_replicate": replicate, "shuffled_value": value,
                    })
            for metric, observed_value in observed_metrics.items():
                null = np.asarray(null_values[metric]); lower, upper = np.percentile(null, [2.5, 97.5])
                if metric == "brier":
                    p_value = (np.sum(null <= observed_value) + 1) / (len(null) + 1)
                    direction = "lower observed Brier favors correct pairing"
                else:
                    p_value = (np.sum(null >= observed_value) + 1) / (len(null) + 1)
                    direction = "higher observed value favors correct pairing"
                summary_rows.append({
                    "modality_pair": pair, "outcome": task, "metric": metric,
                    "correctly_paired_value": observed_value,
                    "correctly_paired_patient_bootstrap_ci_lower": (
                        observed_intervals[metric][0]
                    ),
                    "correctly_paired_patient_bootstrap_ci_upper": (
                        observed_intervals[metric][1]
                    ),
                    "correctly_paired_valid_bootstrap_replicates": (
                        valid_observed_bootstraps
                    ),
                    "shuffled_mean": float(np.mean(null)), "shuffled_sd": float(np.std(null, ddof=1)),
                    "shuffled_95_percentile_interval_lower": float(lower),
                    "shuffled_95_percentile_interval_upper": float(upper),
                    "correct_minus_shuffled_mean": float(observed_value - np.mean(null)),
                    "one_sided_permutation_p_value": float(p_value),
                    "favorable_direction": direction,
                })

    atomic_csv(output / "text_correspondence_permutation_summary.csv", pd.DataFrame(summary_rows))
    atomic_csv(output / "text_correspondence_null_distributions.csv", pd.DataFrame(distribution_rows))
    atomic_csv(output / "checkpoint_reproduction_and_fold_audit.csv", pd.DataFrame(fold_audit))
    atomic_json(output / "text_shuffling_manifest.json", {
        "completed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "multimodal_analysis_manifest": str(analysis_manifest_path),
        "multimodal_analysis_manifest_sha256": mm.sha256_file(
            analysis_manifest_path
        ),
        "text_embedding_policy": analysis_manifest[
            "expected_text_embedding_policy"
        ],
        "permutation_replicates": args.permutation_replicates, "seed": args.seed,
        "pairs": list(TEXT_PAIRS),
        "independent_binary_tasks": True,
        "competing_endpoints_excluded": True,
        "permutation_scope": "within each untouched endpoint-specific outer fold, across all eligible patients regardless of binary outcome",
        "fixed_components": ["ECG embeddings", "outcomes", "trained fusion weights", "Platt mappings", "fold membership"],
        "retraining_performed": False, "recalibration_performed": False,
        "correctly_paired_interval": (
            "95% percentile confidence interval from patient-level bootstrap resampling"
        ),
        "shuffled_interval": (
            "2.5th-97.5th percentile interval of the permutation null distribution"
        ),
        "interpretation": "Tests patient-specific text correspondence beyond the fixed ECG branch.",
    })
    print(f"Text-shuffling analysis complete: {output}")


if __name__ == "__main__":
    main()
