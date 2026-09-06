#!/usr/bin/env python3
"""Generate manuscript figures and copy-ready aggregate tables from locked outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):
        return iterable


REPO = Path("/home/sswee/multimodal_aecg_clinicalfeatures")
MUSIC = Path("/home/sswee/music")
OUT = REPO / "7_Comparative_Analysis/manuscript_outputs"
FIG_MAIN = OUT / "figures/main"
FIG_SUPP = OUT / "figures/supplementary"
FIG_RESTRICTED = OUT / "figures/patient_examples"
TAB_MAIN = OUT / "tables/main"
TAB_SUPP = OUT / "tables/supplementary"
TAB_NUM = OUT / "tables/numeric_source"
CAPTIONS = OUT / "captions"
MANIFESTS = OUT / "manifests"
LOGS = OUT / "logs"

ROOTS = {
    "ECG_ROOT": MUSIC / "ecg_nested_4year_three_wave",
    "JOINT_TEXT_ROOT": MUSIC / "text_nested_4year_v4_detailed",
    "ENDPOINT_TEXT_ROOT": MUSIC / "text_nested_4year_v4_detailed",
    "JOINT_MULTIMODAL_ROOT": MUSIC / "multimodal_nested_4year_v4_detailed",
    "ENDPOINT_MULTIMODAL_ROOT": MUSIC / "multimodal_nested_4year_v4_detailed",
    "TABULAR_ROOT": MUSIC / "tabular_nested_4year_v2",
    "TABULAR_MLP_ROOT": MUSIC / "tabular_mlp_matched_4year_v2",
    "JOINT_COMPARATIVE_ROOT": MUSIC / "comparative_analysis_4year_v4_detailed_endpoint_specific",
    "ENDPOINT_COMPARATIVE_ROOT": MUSIC / "comparative_analysis_4year_v4_detailed_endpoint_specific",
    "LLM_RESPONSE_ROOT": MUSIC / "llm_responses_4year_v3_detailed",
    "REDUCED_TABULAR_ROOT": MUSIC / "tabular_literature_reduced_continuous_lvef_4year_v1",
    "REDUCED_TABULAR_MLP_ROOT": MUSIC / "tabular_mlp_literature_reduced_continuous_lvef_4year_v1",
    "REDUCED_MULTIMODAL_ROOT": MUSIC / "multimodal_literature_reduced_continuous_lvef_4year_v1",
    "REDUCED_COMPARATIVE_ROOT": MUSIC / "literature_reduced_continuous_lvef_comparative_4year_v1",
}

PRIMARY_MODELS = [
    ("ecg", "ECG only"),
    ("full_llm_text", "LLM text only"),
    ("tabular", "Tabular only"),
    ("ecg_tabular", "ECG + Tabular"),
    ("ecg_full_llm", "ECG + LLM text"),
]
PRIMARY_MODEL_LABELS = dict(PRIMARY_MODELS)
MODEL_COLORS = {
    "ecg": "#0072B2",
    "full_llm_text": "#D55E00",
    "tabular": "#009E73",
    "ecg_tabular": "#CC79A7",
    "ecg_full_llm": "#E69F00",
    "deterministic_text": "#56B4E9",
}
OUTCOMES = {"SCD": ("y_scd", "raw_scd", "calibrated_scd", 648, 71), "PFD": ("y_pfd", "raw_pfd", "calibrated_pfd", 659, 82)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    old_restricted = OUT / "figures/restricted_patient_examples"
    if old_restricted.exists() and not FIG_RESTRICTED.exists():
        old_restricted.rename(FIG_RESTRICTED)
    for d in [FIG_MAIN, FIG_SUPP, FIG_RESTRICTED, TAB_MAIN, TAB_SUPP, TAB_NUM, CAPTIONS, MANIFESTS, LOGS]:
        d.mkdir(parents=True, exist_ok=True)
    for checkpoint_dir in [FIG_MAIN / ".ipynb_checkpoints", FIG_SUPP / ".ipynb_checkpoints"]:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
    obsolete = [
        FIG_MAIN / "Figure_5_paired_performance_differences",
        FIG_MAIN / "Figure_7_multimodal_attribution_diagnostics",
        FIG_MAIN / "Figure_6_exploratory_decision_curves",
        FIG_SUPP / "Figure_S7_complete_decision_curves",
        FIG_SUPP / "Figure_S10_attribution_stability_distributions",
        FIG_SUPP / "Figure_S11_perturbation_results",
        FIG_SUPP / "Figure_S5_all_fusion_architecture_roc_pr_curves",
        FIG_SUPP / "Figure_S8_confusion_matrices",
        FIG_SUPP / "Figure_S12_literature_reduced_sensitivity",
        FIG_SUPP / "Figure_S1_endpoint_specific_roc_pr_curves",
        FIG_SUPP / "Figure_S4_all_fusion_architecture_roc_pr_curves",
        FIG_SUPP / "Figure_S2_principal_calibration_plots",
        FIG_SUPP / "Figure_S3_confusion_matrices",
    ]
    for stem in obsolete:
        for ext in ["svg", "pdf", "png"]:
            stem.with_suffix(f".{ext}").unlink(missing_ok=True)
    for caption in [
        "Figure_5_paired_performance_differences.txt",
        "Figure_7_multimodal_attribution_diagnostics.txt",
        "Figure_6_exploratory_decision_curves.txt",
        "Figure_S7_complete_decision_curves.txt",
        "Figure_S10_attribution_stability_distributions.txt",
        "Figure_S11_perturbation_results.txt",
        "Figure_S5_all_fusion_architecture_roc_pr_curves.txt",
        "Figure_S8_confusion_matrices.txt",
        "Figure_S12_literature_reduced_sensitivity.txt",
        "Figure_S1_endpoint_specific_roc_pr_curves.txt",
        "Figure_S4_all_fusion_architecture_roc_pr_curves.txt",
        "Figure_S2_principal_calibration_plots.txt",
        "Figure_S3_confusion_matrices.txt",
    ]:
        (CAPTIONS / caption).unlink(missing_ok=True)
    if FIG_RESTRICTED.exists():
        shutil.rmtree(FIG_RESTRICTED)
        FIG_RESTRICTED.mkdir(parents=True, exist_ok=True)


def cleanup_generated_dirs() -> None:
    for checkpoint_dir in [FIG_MAIN / ".ipynb_checkpoints", FIG_SUPP / ".ipynb_checkpoints"]:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt_ci(row: pd.Series, estimate: str = "estimate", lo: str = "ci_lower", hi: str = "ci_upper") -> str:
    if pd.isna(row.get(estimate)):
        return "Not available"
    return f"{row[estimate]:.3f} ({row[lo]:.3f}-{row[hi]:.3f})"


def write_caption(name: str, text: str) -> None:
    (CAPTIONS / f"{name}.txt").write_text(text.strip() + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, dest_dir: Path, stem: str) -> list[Path]:
    paths = []
    for ext in ["svg", "pdf", "png"]:
        path = dest_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


class Recorder:
    def __init__(self) -> None:
        self.fig_rows: list[dict[str, str]] = []
        self.table_rows: list[dict[str, str]] = []
        self.blocked: list[dict[str, str]] = []
        self.metric_checks: list[str] = []

    def figure(self, filename: str, source: Path, models: str, outcome: str, role: str, notes: str = "", panel: str = "all", public: str = "public") -> None:
        self.fig_rows.append({
            "figure_filename": filename,
            "panel": panel,
            "source_result_root": source_root_for(source),
            "source_file": str(source),
            "source_sha256": sha256(source) if source.exists() and source.is_file() else "",
            "models_shown": models,
            "outcome": outcome,
            "probability_type": "uncalibrated for ROC/PR; calibrated for calibration/DCA/threshold figures as specified",
            "confidence_interval_source": "locked aggregate tables or locked pointwise coordinate tables",
            "calibration_source": "training-only fold-specific calibration where applicable",
            "threshold_source": "training-only fold-specific thresholds where applicable",
            "scientific_role": role,
            "restricted_public_status": public,
            "notes": notes,
        })

    def table(self, filename: str, source: Path, role: str, notes: str = "") -> None:
        self.table_rows.append({
            "table_filename": filename,
            "source_result_root": source_root_for(source),
            "source_file": str(source),
            "source_sha256": sha256(source) if source.exists() and source.is_file() else "",
            "scientific_role": role,
            "notes": notes,
        })

    def missing(self, name: str, reason: str, needed: str = "") -> None:
        self.blocked.append({"output": name, "reason": reason, "needed_source_or_decision": needed})


def source_root_for(path: Path) -> str:
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path
    for key, root in ROOTS.items():
        try:
            resolved.relative_to(root)
            return key
        except ValueError:
            pass
    return "repository_or_generated"


def preflight() -> list[Path]:
    manifests = []
    optional_roots = {
        "REDUCED_TABULAR_ROOT", "REDUCED_TABULAR_MLP_ROOT",
        "REDUCED_MULTIMODAL_ROOT", "REDUCED_COMPARATIVE_ROOT",
    }
    for name, root in ROOTS.items():
        if not root.exists():
            if name in optional_roots:
                continue
            raise SystemExit(f"Missing required result root: {name}={root}")
        found = sorted(root.rglob("*manifest*.json"))
        if not found:
            raise SystemExit(f"No completion manifest found under {name}={root}")
        manifests.extend(found)
    comparative = json.loads((ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/comparative_analysis_manifest.json").read_text())
    cohorts = comparative["eligible_patients"]
    events = comparative["events"]
    assert cohorts["SCD"] == 648 and events["SCD"] == 71
    assert cohorts["PFD"] == 659 and events["PFD"] == 82
    analysis = json.loads((ROOTS["JOINT_MULTIMODAL_ROOT"] / "analysis_setup/analysis_manifest.json").read_text())
    assert analysis["outer_folds"] == 5 and analysis["inner_folds"] == 4 and analysis["seed"] == 42
    return manifests


def make_flowchart(rec: Recorder) -> None:
    stem = "Figure_1_patient_cohort_flow"
    steps = [
        ("Source MUSIC cohort", 992, ""),
        ("Holter available", 936, "Excluded 56"),
        ("Eligible recorded outcome category", 879, "Excluded 57: outcome outside the prespecified categories"),
        ("No prior implantable cardiac device", 758, "Excluded 121"),
        ("No cardiac transplantation", 746, "Excluded 12"),
        ("Four-year outcome ascertainable", 730, "Excluded 16"),
    ]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.axis("off")
    yvals = np.linspace(0.92, 0.32, len(steps))
    for idx, ((label, n, excl), y) in enumerate(zip(steps, yvals)):
        ax.text(0.5, y, f"{label}\nN={n}", ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#555555"))
        if idx < len(steps) - 1:
            ax.annotate("", xy=(0.5, yvals[idx + 1] + 0.055), xytext=(0.5, y - 0.055),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))
        if excl:
            ax.text(0.78, y - 0.06, excl, ha="left", va="center", fontsize=9, color="#555555")
    outcome_y = 0.17
    outcomes = [("No cardiac death by four years", 577, 0.18), ("SCD", 71, 0.5), ("PFD", 82, 0.82)]
    for label, n, x in outcomes:
        ax.text(x, outcome_y, f"{label}\nN={n}", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F7F7F7", edgecolor="#555555"))
        ax.annotate("", xy=(x, outcome_y + 0.055), xytext=(0.5, yvals[-1] - 0.055),
                    arrowprops=dict(arrowstyle="->", color="#777777", lw=1.2))
    ax.text(0.3, 0.04, "SCD binary analysis: N=648 = 577 controls + 71 SCD; exclude PFD",
            ha="center", va="center", fontsize=9)
    ax.text(0.7, 0.04, "PFD binary analysis: N=659 = 577 controls + 82 PFD; exclude SCD",
            ha="center", va="center", fontsize=9)
    ax.text(0.5, 0.0, "Predictor missingness was handled within each training fold and caused no additional cohort exclusions.",
            ha="center", va="bottom", fontsize=9)
    paths = save_figure(fig, FIG_MAIN, stem)
    for p in paths:
        rec.figure(p.name, ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/comparative_analysis_manifest.json", "cohort", "SCD/PFD", "main")
    exclusions = pd.DataFrame([
        {"Step": s[0], "Remaining N": s[1], "Excluded at step": e, "Exclusion wording": s[2].replace("Excluded ", "")}
        for s, e in zip(steps, [0, 56, 57, 121, 12, 16])
    ])
    path = TAB_SUPP / "Table_S1_cohort_exclusions.csv"
    exclusions.to_csv(path, index=False)
    rec.table(path.name, ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/comparative_analysis_manifest.json", "supplementary")
    write_caption(stem, "Patient cohort selection for four-year SCD and PFD prediction.")


def figure_roc_pr(rec: Recorder, kind: str) -> None:
    pred_path = ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/standardized_principal_outer_test_predictions.csv"
    perf_path = ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/principal_models_performance_with_95ci.csv"
    pred = read_csv(pred_path)
    perf = read_csv(perf_path)
    metric = "roc_auc" if kind == "roc" else "pr_auc"
    stem = "Figure_3_roc_curves" if kind == "roc" else "Figure_4_precision_recall_curves"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, outcome in zip(axes, ["SCD", "PFD"]):
        _, _, _, n, events = OUTCOMES[outcome]
        task = outcome.lower()
        for model, label in PRIMARY_MODELS:
            # The endpoint-specific comparative analysis stores one row per
            # patient, model, and task in long format.  Retain compatibility
            # with an older wide-format export so this reporting step remains
            # safe to rerun against either audited schema.
            if {"task", "y", "raw"}.issubset(pred.columns):
                frame = pred[
                    pred["model"].eq(model)
                    & pred["task"].astype(str).str.lower().eq(task)
                ].dropna(subset=["y", "raw"])
                y = frame["y"].astype(int).to_numpy()
                score = frame["raw"].astype(float).to_numpy()
            else:
                ycol, rawcol, _, _, _ = OUTCOMES[outcome]
                frame = pred[pred["model"].eq(model)].dropna(subset=[ycol, rawcol])
                y = frame[ycol].astype(int).to_numpy()
                score = frame[rawcol].astype(float).to_numpy()

            patient_column = "Patient ID" if "Patient ID" in frame.columns else "patient_key"
            if len(frame) != n or frame[patient_column].nunique() != n:
                raise SystemExit(f"{model} {outcome} predictions do not contain one row per eligible patient")
            rows = perf[
                (perf.model == model)
                & (perf.outcome == outcome)
                & (perf.metric == metric)
            ]
            if rows.empty:
                raise SystemExit(f"Missing authoritative {metric} result for {model} {outcome}")
            row = rows.iloc[0]
            calc = roc_auc_score(y, score) if kind == "roc" else average_precision_score(y, score)
            if abs(calc - row.estimate) > 0.002:
                raise SystemExit(f"Plotted {kind} disagrees with authoritative value for {model} {outcome}: {calc} vs {row.estimate}")
            self_label = f"{label}: {row.estimate:.3f}"
            if kind == "roc":
                xs, ys, _ = roc_curve(y, score)
                ax.plot(xs, ys, label=self_label, color=MODEL_COLORS[model], lw=2)
            else:
                ys, xs, _ = precision_recall_curve(y, score)
                ax.plot(xs, ys, label=self_label, color=MODEL_COLORS[model], lw=2)
            rec.metric_checks.append(f"{stem}: {model} {outcome} calculated {calc:.6f}; authoritative {row.estimate:.6f}")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
        else:
            prevalence = events / n
            ax.axhline(prevalence, ls="--", color="#777777", lw=1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.set_title(f"({chr(97 + list(OUTCOMES).index(outcome))}) {outcome}")
        ax.grid(False)
        ax.legend(fontsize=7, loc="lower right" if kind == "roc" else "best")
    fig.tight_layout()
    paths = save_figure(fig, FIG_MAIN, stem)
    for p in paths:
        rec.figure(p.name, pred_path, ", ".join(label for _, label in PRIMARY_MODELS), "SCD/PFD", "main")
    write_caption(stem, f"Pooled outer-test {'ROC' if kind == 'roc' else 'precision-recall'} curves for four-year SCD and PFD prediction.")


def make_performance_tables(rec: Recorder) -> None:
    perf_path = ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/principal_models_performance_with_95ci.csv"
    perf = read_csv(perf_path)
    rows = []
    numeric = []
    for outcome in ["SCD", "PFD"]:
        _, _, _, n, events = OUTCOMES[outcome]
        for model, label in PRIMARY_MODELS:
            sub = perf[(perf.model == model) & (perf.outcome == outcome)]
            values = {}
            for metric, col in [("roc_auc", "AUROC (95% CI)"), ("pr_auc", "PR-AUC (95% CI)"), ("brier", "Calibrated Brier score (95% CI)")]:
                row = sub[sub.metric == metric]
                if row.empty:
                    values[col] = "Not available"
                else:
                    values[col] = fmt_ci(row.iloc[0])
                    nr = row.iloc[0].to_dict()
                    nr.update({"Outcome": outcome, "Model": label})
                    numeric.append(nr)
            rows.append({"Outcome": outcome, "Model": label, "N": n, "Events": events, **values})
    display = pd.DataFrame(rows)
    display.to_csv(TAB_MAIN / "Table_2_principal_model_performance_display.csv", index=False)
    pd.DataFrame(numeric).to_csv(TAB_NUM / "Table_2_principal_model_performance_numeric.csv", index=False)
    rec.table("Table_2_principal_model_performance_display.csv", perf_path, "main")
    rec.table("Table_2_principal_model_performance_numeric.csv", perf_path, "numeric_source")


def make_threshold_tables(rec: Recorder) -> None:
    thresh_path = ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/principal_models_threshold_metrics_with_95ci.csv"
    cm_path = ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/principal_models_confusion_matrices.csv"
    fold_path = ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/saved_fold_specific_threshold_summary.csv"
    thresh, cm, folds = map(read_csv, [thresh_path, cm_path, fold_path])
    rows, numeric = [], []
    for outcome in ["SCD", "PFD"]:
        for model, label in PRIMARY_MODELS:
            sub = thresh[(thresh.model == model) & (thresh.outcome == outcome)]
            counts = cm[(cm.model == model) & (cm.outcome == outcome)].iloc[0]
            fold = folds[(folds.model == model) & (folds.outcome == outcome)].iloc[0]
            row = {"Outcome": outcome, "Model": label, "Threshold-selection rule": fold.selection_source}
            for metric in ["sensitivity", "specificity", "ppv", "npv", "f1"]:
                m = sub[sub.metric == metric]
                row[metric.upper() if metric != "f1" else "F1"] = fmt_ci(m.iloc[0]) if not m.empty else "Not available"
            row.update({"TP": int(counts.tp), "FP": int(counts.fp), "TN": int(counts.tn), "FN": int(counts.fn)})
            rows.append(row)
            numeric.extend(sub.to_dict("records"))
    pd.DataFrame(rows).to_csv(TAB_MAIN / "Table_3_threshold_operating_characteristics_display.csv", index=False)
    pd.DataFrame(numeric).to_csv(TAB_NUM / "Table_3_threshold_operating_characteristics_numeric.csv", index=False)
    rec.table("Table_3_threshold_operating_characteristics_display.csv", thresh_path, "main")
    rec.table("Table_3_threshold_operating_characteristics_numeric.csv", thresh_path, "numeric_source")


def make_baseline_table(rec: Recorder) -> None:
    src = MUSIC / "subject-info-cleaned-4year.csv"
    df = pd.read_csv(src)
    if len(df) != 730:
        raise SystemExit("Baseline source does not contain N=730")
    groups = {
        "Overall (N=730)": df,
        "No cardiac death (N=577)": df[df["No_cardiac_death_4year"] == 1],
        "SCD (N=71)": df[df["SCD_4year"] == 1],
        "PFD (N=82)": df[df["PFD_4year"] == 1],
    }
    variables = [
        ("Age, years", "Age", "continuous_mean_sd"),
        ("Male sex", "Gender (male=1)", "binary"),
        ("NYHA class III", "NYHA class", "nyha3"),
        ("LVEF, %", "LVEF (%)", "continuous_mean_sd"),
        ("Pro-BNP, ng/L", "Pro-BNP (ng/L)", "continuous_median_iqr"),
        ("Creatinine, umol/L", "Creatinine (?mol/L)", "continuous_median_iqr"),
        ("Diabetes", "Diabetes (yes=1)", "binary"),
        ("Prior myocardial infarction", "Prior Myocardial Infarction (yes=1)", "binary"),
        ("QRS duration, ms", "QRS duration (ms)", "continuous_mean_sd"),
    ]
    display_rows, numeric_rows = [], []
    for label, col, kind in variables:
        row = {"Characteristic": label}
        missing = int(df[col].isna().sum())
        row["Missing, n (%)"] = f"{missing} ({missing / len(df) * 100:.1f})"
        for gname, gdf in groups.items():
            s = gdf[col].dropna()
            if kind == "continuous_mean_sd":
                val = f"{s.mean():.1f} ({s.std():.1f})"
            elif kind == "continuous_median_iqr":
                val = f"{s.median():.1f} [{s.quantile(0.25):.1f}-{s.quantile(0.75):.1f}]"
            elif kind == "nyha3":
                n = int((s == 3).sum())
                val = f"{n} ({n / len(gdf) * 100:.1f})"
            else:
                n = int((s == 1).sum())
                val = f"{n} ({n / len(gdf) * 100:.1f})"
            row[gname] = val
            numeric_rows.append({"Characteristic": label, "Source column": col, "Group": gname, "N": len(gdf), "Nonmissing": len(s), "Mean": s.mean() if pd.api.types.is_numeric_dtype(s) else np.nan, "SD": s.std() if pd.api.types.is_numeric_dtype(s) else np.nan, "Median": s.median() if pd.api.types.is_numeric_dtype(s) else np.nan, "Q1": s.quantile(0.25) if pd.api.types.is_numeric_dtype(s) else np.nan, "Q3": s.quantile(0.75) if pd.api.types.is_numeric_dtype(s) else np.nan})
        display_rows.append(row)
    pd.DataFrame(display_rows).to_csv(TAB_MAIN / "Table_1_baseline_characteristics_display.csv", index=False)
    pd.DataFrame(numeric_rows).to_csv(TAB_NUM / "Table_1_baseline_characteristics_numeric.csv", index=False)
    rec.table("Table_1_baseline_characteristics_display.csv", src, "main", "aggregate baseline table; no patient rows")
    rec.table("Table_1_baseline_characteristics_numeric.csv", src, "numeric_source", "aggregate baseline table; no patient rows")


def copy_table(src: Path, dest: Path, rec: Recorder, role: str) -> None:
    df = read_csv(src)
    bad_cols = {"Patient ID", "patient_id", "patient_key"}
    if bad_cols.intersection(df.columns):
        raise SystemExit(f"Refusing to write patient-identifier table: {src}")
    df.to_csv(dest, index=False)
    rec.table(dest.name, src, role)


def make_supp_tables(rec: Recorder) -> None:
    mappings = {
        "Table_S5_variable_provenance_and_overlap.csv": ROOTS["JOINT_COMPARATIVE_ROOT"] / "provenance/cross_representation_feature_provenance.csv",
        "Table_S7_text_ablation_performance.csv": ROOTS["JOINT_TEXT_ROOT"] / "combined_evaluation/pooled_performance_calibration_with_95ci.csv",
        "Table_S8_fusion_architecture_performance.csv": ROOTS["JOINT_MULTIMODAL_ROOT"] / "combined_evaluation/pooled_performance_with_95ci.csv",
        "Table_S9_calibration_metrics.csv": ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/principal_models_calibration_with_95ci.csv",
        "Table_S10_complete_threshold_metrics.csv": ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/principal_models_threshold_metrics_with_95ci.csv",
        "Table_S11_confusion_matrix_counts.csv": ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/principal_models_confusion_matrices.csv",
        "Table_S12_paired_model_comparisons.csv": ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/primary_comparisons_with_95ci.csv",
        "Table_S13_decision_curve_summary.csv": ROOTS["JOINT_COMPARATIVE_ROOT"] / "figures/decision_curve_supported_ranges.csv",
        "Table_S14_text_shuffling.csv": ROOTS["JOINT_MULTIMODAL_ROOT"] / "multimodal_text_shuffling/text_correspondence_permutation_summary.csv",
    }
    for name, src in tqdm(mappings.items(), desc="Copying supplementary tables"):
        copy_table(src, TAB_SUPP / name, rec, "supplementary")
    sel_src = ROOTS["JOINT_MULTIMODAL_ROOT"] / "multimodal_posthoc_evaluation/selected_fusion_method_by_outer_fold.csv"
    sel = read_csv(sel_src)
    sel_rows = []
    for r in sel.itertuples(index=False):
        sel_rows.append({
            "Outcome": "Endpoint-specific SCD or PFD selection",
            "Outer fold": r.outer_fold,
            "Text representation": "endpoint-specific LLM representation" if r.modality_pair == "ecg_full_text" else "deterministic text" if r.modality_pair == "ecg_deterministic_text" else "Not applicable",
            "LLaMA model": "Selected within the outer-training cohort; see text model manifests",
            "Encoder": "Nested-selected; see text model manifests",
            "Classifier": "Nested-selected; see text model manifests",
            "Modality pair": r.modality_pair,
            "Fusion method": r.selected_method,
            "Configuration": "selected_fusion",
            "Selection source": str(sel_src),
        })
    table_s6 = TAB_SUPP / "Table_S6_fold_specific_model_selection.csv"
    pd.DataFrame(sel_rows).to_csv(table_s6, index=False)
    rec.table(table_s6.name, sel_src, "supplementary")
    stab_src = ROOTS["JOINT_COMPARATIVE_ROOT"] / "attribution_diagnostics/evaluation/attribution_stability_summary_with_95ci.csv"
    pert_src = ROOTS["JOINT_COMPARATIVE_ROOT"] / "attribution_diagnostics/evaluation/high_attribution_vs_random_paired_summary.csv"
    stab = read_csv(stab_src).assign(diagnostic_family="stability")
    pert = read_csv(pert_src).assign(diagnostic_family="perturbation")
    table_s15 = TAB_SUPP / "Table_S15_attribution_stability_perturbation.csv"
    pd.concat([stab, pert], ignore_index=True, sort=False).to_csv(table_s15, index=False)
    rec.table(table_s15.name, stab_src, "supplementary", f"Combined with {pert_src}")
    outcome_defs = pd.DataFrame([
        {"Concept": "Four-year horizon", "Definition": "Events assessed within four years after baseline Holter recording."},
        {"Concept": "SCD definition", "Definition": "Sudden cardiac death endpoint as encoded in locked MUSIC four-year cohort labels."},
        {"Concept": "PFD definition", "Definition": "Pump failure death endpoint as encoded in locked MUSIC four-year cohort labels."},
        {"Concept": "Control definition", "Definition": "No cardiac death by four years."},
        {"Concept": "Competing-endpoint exclusion", "Definition": "PFD excluded from SCD binary task; SCD excluded from PFD binary task."},
        {"Concept": "Cardiac transplantation", "Definition": "Excluded before final four-year cohort construction."},
        {"Concept": "Non-cardiac death", "Definition": "Not treated as SCD or PFD event in final binary task cohorts."},
        {"Concept": "Insufficient follow-up", "Definition": "Excluded when four-year outcome ascertainment was unavailable."},
    ])
    outcome_defs.to_csv(TAB_SUPP / "Table_S2_outcome_definitions.csv", index=False)
    rec.table("Table_S2_outcome_definitions.csv", REPO / "2_Labeling/Labeling_MUSIC_Updated.ipynb", "supplementary")
    shutil.copy2(TAB_MAIN / "Table_1_baseline_characteristics_display.csv", TAB_SUPP / "Table_S3_expanded_baseline_characteristics.csv")
    rec.table("Table_S3_expanded_baseline_characteristics.csv", MUSIC / "subject-info-cleaned-4year.csv", "supplementary")
    miss = pd.DataFrame([
        {"Variable": c, "Missing n": int(pd.read_csv(MUSIC / "subject-info-cleaned-4year.csv", usecols=[c])[c].isna().sum()), "Missing percent": "", "Modality": "Tabular", "Imputation method": "Model-specific training-fold imputation", "Fitted within training data": "Yes", "Missingness caused exclusion": "No"}
        for c in ["Age", "LVEF (%)", "Pro-BNP (ng/L)", "Creatinine (?mol/L)", "QRS duration (ms)"]
    ])
    total = 730
    miss["Missing percent"] = miss["Missing n"].map(lambda x: f"{x / total * 100:.1f}")
    miss.to_csv(TAB_SUPP / "Table_S4_missingness_and_imputation.csv", index=False)
    rec.table("Table_S4_missingness_and_imputation.csv", MUSIC / "subject-info-cleaned-4year.csv", "supplementary")
    llm_rows = []
    for stem in ["LLaMA3.1-8B-4year-responses", "LLaMA3.2-3B-4year-responses"]:
        manifest_path = ROOTS["LLM_RESPONSE_ROOT"] / f"{stem}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generation = manifest["generation"]
        software = manifest["software"]
        llm_rows.append({
            "Model name": manifest["model_id"],
            "Immutable revision": manifest["resolved_model_commit"],
            "Inference library/version": f"Transformers {software['transformers']}",
            "Quantization": manifest["quantization"],
            "Precision": manifest["torch_dtype"],
            "Device mapping": manifest["device_map_strategy"],
            "Decoding method": "greedy",
            "Temperature or not applicable": "not applicable",
            "do_sample": generation["do_sample"],
            "max_new_tokens": generation["max_new_tokens"],
            "seed": generation["seed"],
            "Prompt checksum": manifest["input_csv_sha256"],
            "Prompt version/condition": manifest["run_label"],
            "Hardware": "; ".join(manifest["visible_gpu_names"]),
            "Generation status counts": json.dumps(manifest["generation_status_counts"], sort_keys=True),
            "Whether prompt development used outcome labels": "No",
        })
    llm = pd.DataFrame(llm_rows)
    llm.to_csv(TAB_SUPP / "Table_S16_llm_reproducibility.csv", index=False)
    rec.table("Table_S16_llm_reproducibility.csv", ROOTS["JOINT_TEXT_ROOT"] / "analysis_setup/prepare_manifest.json", "supplementary")


def make_dca(rec: Recorder) -> None:
    src = ROOTS["JOINT_COMPARATIVE_ROOT"] / "figures/decision_curve_data_with_pointwise_95ci.csv"
    dca = read_csv(src)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    keep = ["ecg", "full_llm_text", "tabular", "ecg_tabular", "ecg_full_llm"]
    for ax, outcome in zip(axes, ["SCD", "PFD"]):
        sub = dca[(dca.outcome == outcome) & (dca.threshold.between(0.02, 0.25))]
        for model in keep:
            m = sub[sub.model == model]
            if m.empty:
                continue
            ax.plot(m.threshold, m.net_benefit, color=MODEL_COLORS[model], lw=2, label=PRIMARY_MODEL_LABELS.get(model, m.model_label.iloc[0]))
            ax.fill_between(m.threshold, m.net_benefit_ci_lower, m.net_benefit_ci_upper, color=MODEL_COLORS[model], alpha=0.12)
        first = sub.iloc[0]
        ax.plot(sub.threshold.unique(), sub.groupby("threshold").treat_all_net_benefit.first(), color="#666666", lw=1.5, ls="--", label="Treat all")
        ax.axhline(0, color="#222222", lw=1.2, ls=":", label="Treat none")
        ax.set_title(f"({chr(97 + list(OUTCOMES).index(outcome))}) {outcome}")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.grid(False)
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    paths = save_figure(fig, FIG_MAIN, "Figure_5_exploratory_decision_curves")
    for p in paths:
        rec.figure(p.name, src, "ECG, LLM text only, tabular, ECG + tabular, ECG + LLM text, treat all, treat none", "SCD/PFD", "main")
    write_caption("Figure_5_exploratory_decision_curves", "Exploratory decision curves for calibrated four-year SCD and PFD risk predictions.")


def make_forest_table_only(rec: Recorder) -> None:
    src = ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/primary_comparisons_with_95ci.csv"
    df = read_csv(src)
    dest = TAB_SUPP / "Table_S18_paired_performance_differences_display.csv"
    display = df.copy()
    for col in ["comparator_label", "reference_label"]:
        if col in display.columns:
            display[col] = display[col].replace(PRIMARY_MODEL_LABELS)
    display.to_csv(dest, index=False)
    rec.table(dest.name, src, "supplementary", "Former main Figure 5 content moved to a supplementary table.")


def make_calibration(rec: Recorder) -> None:
    src = ROOTS["JOINT_COMPARATIVE_ROOT"] / "figures/calibration_plot_points.csv"
    df = read_csv(src)
    # Current comparative outputs use ``mean_predicted_probability``.  Keep
    # the former reporting name as a read-only compatibility fallback.
    predicted_column = (
        "mean_predicted_probability"
        if "mean_predicted_probability" in df.columns
        else "mean_predicted_risk"
    )
    if predicted_column not in df.columns:
        raise SystemExit(
            "Calibration data lack a mean-prediction column; "
            f"available columns: {df.columns.tolist()}"
        )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, outcome in zip(axes, ["SCD", "PFD"]):
        for model, label in PRIMARY_MODELS:
            sub = df[(df.outcome == outcome) & (df.model == model)]
            if sub.empty:
                continue
            lower_error = np.maximum(0, sub.observed_event_rate - sub.observed_ci_lower)
            upper_error = np.maximum(0, sub.observed_ci_upper - sub.observed_event_rate)
            ax.errorbar(sub[predicted_column], sub.observed_event_rate,
                        yerr=[lower_error, upper_error],
                        marker="o", lw=1.5, label=label, color=MODEL_COLORS[model])
        ax.plot([0, 1], [0, 1], color="#777777", ls="--", lw=1)
        ax.set_xlim(0, max(0.35, df[predicted_column].max() * 1.1))
        ax.set_ylim(0, max(0.45, df.observed_ci_upper.max() * 1.1))
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed event proportion")
        ax.grid(False)
        ax.legend(fontsize=7, loc="lower right")
    paths = save_figure(fig, FIG_SUPP, "Figure_S3_principal_calibration_plots")
    for p in paths:
        rec.figure(p.name, src, "principal models", "SCD/PFD", "supplementary")
    write_caption("Figure_S3_principal_calibration_plots", "Principal calibration plots for pooled outer-test predictions using fold-specific calibrated probabilities. Points show calibration groups with locked observed confidence intervals. No recalibration or bootstrap recomputation was performed.")


def make_confusion(rec: Recorder) -> None:
    src = ROOTS["JOINT_COMPARATIVE_ROOT"] / "evaluation/principal_models_confusion_matrices.csv"
    cm = read_csv(src)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.6))
    for i, outcome in enumerate(["SCD", "PFD"]):
        for j, (model, label) in enumerate(PRIMARY_MODELS):
            ax = axes[i, j]
            row = cm[(cm.outcome == outcome) & (cm.model == model)].iloc[0]
            mat = np.array([[row.tp, row.fn], [row.fp, row.tn]])
            ax.imshow(mat, cmap="Blues")
            threshold = float(mat.max()) * 0.5
            for r in range(2):
                for c in range(2):
                    text_color = "white" if mat[r, c] > threshold else "black"
                    ax.text(c, r, f"{[['TP','FN'],['FP','TN']][r][c]}\n{int(mat[r,c])}", ha="center", va="center", fontsize=8, color=text_color)
            if int(mat.sum()) != OUTCOMES[outcome][3]:
                raise SystemExit(f"Confusion matrix sum mismatch for {model} {outcome}")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Positive", "Negative"] if i == 1 else [])
            ax.set_yticklabels(["Positive", "Negative"] if j == 0 else [])
            ax.tick_params(length=0, labelsize=8)
            if j == 0:
                ax.set_ylabel("Ground Truth", fontsize=9)
            if i == 1:
                ax.set_xlabel("Predicted", fontsize=9)
            ax.set_title(f"{outcome}: {label}", fontsize=8, pad=6)
    fig.tight_layout()
    paths = save_figure(fig, FIG_SUPP, "Figure_S4_confusion_matrices")
    for p in paths:
        rec.figure(p.name, src, "principal models", "SCD/PFD", "supplementary")
    write_caption("Figure_S4_confusion_matrices", "Confusion matrices for pooled outer-test classifications using fold-specific thresholds selected from inner out-of-fold training predictions. Ground-truth positives are shown on the top row and predicted positives on the left column. No global threshold is displayed because fold-specific thresholds were used.")


def make_endpoint_specific_s1(rec: Recorder) -> None:
    src = ROOTS["ENDPOINT_COMPARATIVE_ROOT"] / "figures/discrimination_curve_plot_data.csv"
    df = read_csv(src)
    specs = [
        ("ROC", "Figure_S1_endpoint_specific_auroc_curves", "False positive rate", "True positive rate", "lower right"),
        ("PR", "Figure_S2_endpoint_specific_prauc_curves", "Recall", "Precision", "best"),
    ]
    for curve, stem, xlabel, ylabel, legend_loc in specs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
        for ax, outcome in zip(axes, ["SCD", "PFD"]):
            for model, label in PRIMARY_MODELS:
                sub = df[(df.outcome == outcome) & (df.model == model) & (df.curve == curve)]
                if sub.empty:
                    continue
                ax.plot(sub.x, sub.y, color=MODEL_COLORS[model], lw=2, ls="-", label=f"{label}: {sub.estimate.iloc[0]:.3f}")
            ax.set_title(f"({chr(97 + list(OUTCOMES).index(outcome))}) {outcome}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(False)
            ax.legend(fontsize=7, loc=legend_loc)
        fig.tight_layout()
        paths = save_figure(fig, FIG_SUPP, stem)
        for p in paths:
            rec.figure(p.name, src, ", ".join(label for _, label in PRIMARY_MODELS), "SCD/PFD", "supplementary", "Endpoint-specific sensitivity curves; solid lines only; no chance baseline.")
        write_caption(stem, f"Endpoint-specific sensitivity {'ROC' if curve == 'ROC' else 'precision-recall'} curves for SCD and PFD. Curves use solid lines and point-estimate legend values only; confidence intervals are reported in tables.")


def copy_existing_figures(rec: Recorder) -> None:
    copies = {}
    for stem, srcstem in tqdm(copies.items(), desc="Copying existing figures"):
        made = False
        copied_png = None
        for ext in ["pdf", "png"]:
            src = srcstem.with_suffix(f".{ext}")
            if src.exists():
                dest = FIG_SUPP / f"{stem}.{ext}"
                shutil.copy2(src, dest)
                rec.figure(dest.name, src, "locked existing figure", "SCD/PFD", "supplementary")
                made = True
                if ext == "png":
                    copied_png = dest
        if copied_png is not None:
            arr = plt.imread(copied_png)
            height, width = arr.shape[0], arr.shape[1]
            encoded = base64.b64encode(copied_png.read_bytes()).decode("ascii")
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">\n'
                f'<image href="data:image/png;base64,{encoded}" width="{width}" height="{height}"/>\n'
                "</svg>\n"
            )
            dest = FIG_SUPP / f"{stem}.svg"
            dest.write_text(svg, encoding="utf-8")
            rec.figure(dest.name, copied_png, "locked existing figure", "SCD/PFD", "supplementary", "SVG wrapper around locked PNG; not redrawn")
        if made:
            write_caption(stem, f"Supplementary locked-output figure copied from {source_root_for(srcstem)}. Scientific role follows the source analysis hierarchy; no data or statistics were regenerated.")
        else:
            rec.missing(stem, "No existing locked PDF/PNG figure found", str(srcstem))


def make_patient_examples(rec: Recorder) -> None:
    src_root = ROOTS["ENDPOINT_MULTIMODAL_ROOT"] / "explain_multimodal_ecg_text"
    summaries = sorted(src_root.glob("val_fold_*/*/*/*/summary.json"))
    if not summaries:
        rec.missing("patient_examples", "No nested multimodal explainability example directories found", str(src_root))
        return
    manifest_rows = []
    examples = []
    for summary_path in summaries:
        data = json.loads(summary_path.read_text())
        if data.get("selection_mode") in {"positive", "top_positive_cases"} and int(data.get("y_true", 0)) == 1 and data.get("endpoint") in {"SCD", "PFD"}:
            examples.append((summary_path, data))
    examples.sort(key=lambda item: (item[1].get("endpoint", ""), str(item[0].parts[-4]), item[1].get("method", ""), item[1].get("rank_within_target", 9999), item[1].get("pid", "")))
    required = [
        "ecg_attention.png",
        "ecg_attention.svg",
        "text_saliency.html",
        "text_token_saliency.csv",
        "top_ecg_segments_attention_with_saliency.png",
        "top_ecg_segments_attention_with_saliency.svg",
    ]
    for idx, (summary_path, data) in tqdm(list(enumerate(examples, start=1)), desc="Copying patient examples"):
        source_dir = summary_path.parent
        fold = source_dir.parts[-4].replace("val_fold_", "fold_")
        endpoint = data["endpoint"].lower()
        method = data.get("method", source_dir.parts[-2])
        example_id = f"{endpoint}_example_{idx:03d}_{fold}_{method}_{data.get('pid', source_dir.name)}"
        dest_dir = FIG_RESTRICTED / example_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in required:
            src = source_dir / name
            if not src.exists():
                rec.missing(f"patient_examples/{example_id}/{name}", "Requested patient-example asset missing", str(src))
                continue
            dest = dest_dir / name
            shutil.copy2(src, dest)
            copied.append(dest)
            rec.figure(str(dest.relative_to(OUT)), src, "ECG + LLM-text model-behavior visualization", data["endpoint"], "patient_example", "Patient-level local example copied from locked explanation output.", public="restricted_patient_level")
        shutil.copy2(summary_path, dest_dir / "summary.json")
        copied.append(dest_dir / "summary.json")
        manifest_rows.append({
            "example_id": example_id,
            "source_pid": data.get("pid"),
            "endpoint": data.get("endpoint"),
            "source_fold": fold,
            "method": data.get("method"),
            "selection_mode": data.get("selection_mode"),
            "rank_within_target": data.get("rank_within_target"),
            "y_true": data.get("y_true"),
            "pred_prob": data.get("pred_prob"),
            "ecg_baseline_prob": data.get("ecg_baseline_prob"),
            "copied_asset_count": len(copied),
            "source_directory": str(source_dir),
            "destination_directory": str(dest_dir),
        })
    manifest = MANIFESTS / "patient_examples_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    rec.table(manifest.name, src_root, "patient_example", "Patient-level manifest for local review; not a public aggregate table.")



def make_attribution_2x2(rec: Recorder) -> None:
    eval_root = ROOTS["JOINT_COMPARATIVE_ROOT"] / "attribution_diagnostics/evaluation"

    stability_src = eval_root / "all_patient_stability_results.csv"
    randomization_src = eval_root / "all_patient_randomization_results.csv"
    perturbation_src = eval_root / "all_patient_perturbation_results.csv"

    stability = read_csv(stability_src)
    randomization = read_csv(randomization_src)
    perturbation = read_csv(perturbation_src)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --------------------------------------------------------------
    # Top row:
    #   top-left  = SCD attribution
    #   top-right = PFD attribution
    # --------------------------------------------------------------
    attribution_labels = [
        ("Repeated-\nseed refit", "refit"),
        ("Whitespace", "whitespace"),
        ("Punctuation", "punctuation"),
        ("Line order", "line_order"),
        ("Neutral\nPrefix", "neutral_prefix"),
        ("Randomized\nClassifier", "randomized"),
    ]

    for ax, outcome in zip(axes[0], ["SCD", "PFD"]):
        values = []
        lower = []
        upper = []

        # Repeated-seed refit.
        x = stability[
            stability["outcome"].eq(outcome)
            & stability["stability_type"].eq("outer_training_refit_seed")
        ]["spearman"].dropna().to_numpy(float)
        values.append(float(np.mean(x)))
        se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        lower.append(values[-1] - 1.96 * se)
        upper.append(values[-1] + 1.96 * se)

        # Meaning-preserving input variants.
        input_rows = stability[
            stability["outcome"].eq(outcome)
            & stability["stability_type"].eq("meaning_preserving_input")
        ]
        for variant in ["whitespace", "punctuation", "line_order", "neutral_prefix"]:
            x = input_rows[
                input_rows["replicate"].astype(str).eq(variant)
            ]["spearman"].dropna().to_numpy(float)
            values.append(float(np.mean(x)))
            se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
            lower.append(values[-1] - 1.96 * se)
            upper.append(values[-1] + 1.96 * se)

        # Classifier randomization.
        x = randomization[
            randomization["outcome"].eq(outcome)
        ]["spearman"].dropna().to_numpy(float)
        values.append(float(np.mean(x)))
        se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
        lower.append(values[-1] - 1.96 * se)
        upper.append(values[-1] + 1.96 * se)

        values = np.asarray(values, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        positions = np.arange(len(values))

        ax.errorbar(
            positions,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="o",
            color="#333333",
            ecolor="#333333",
            capsize=3,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([label for label, _ in attribution_labels])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Spearman correlation")
        ax.set_title(f"{outcome}: Attribution Spearman correlation")
        ax.grid(False)

    # --------------------------------------------------------------
    # Bottom row:
    #   bottom-left  = SCD token masking
    #   bottom-right = PFD token masking
    # --------------------------------------------------------------
    controls = [
        ("high_attribution", "High Attribution", "#b2182b"),
        ("frequency_matched", "Frequency Matched", "#ef8a62"),
        ("random", "Random", "#2166ac"),
        ("low_attribution", "Low Attribution", "#67a9cf"),
    ]

    # Average repeated random masks within patient before plotting.
    patient_masking = (
        perturbation[
            perturbation["control"].isin([item[0] for item in controls])
        ]
        .groupby(
            ["patient_key", "outcome", "fraction", "control"],
            as_index=False,
        )["absolute_probability_change"]
        .mean()
    )

    for ax, outcome in zip(axes[1], ["SCD", "PFD"]):
        outcome_data = patient_masking[
            patient_masking["outcome"].eq(outcome)
        ]

        for control, label, color in controls:
            data = outcome_data[
                outcome_data["control"].eq(control)
            ]
            if data.empty:
                continue

            rows = []
            for fraction, group in data.groupby("fraction"):
                x = group["absolute_probability_change"].dropna().to_numpy(float)
                mean = float(np.mean(x))
                se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
                rows.append(
                    {
                        "fraction": float(fraction),
                        "mean": mean,
                        "lower": mean - 1.96 * se,
                        "upper": mean + 1.96 * se,
                    }
                )

            summary = pd.DataFrame(rows).sort_values("fraction")
            x = summary["fraction"].to_numpy(float) * 100.0
            y = summary["mean"].to_numpy(float)
            lo = summary["lower"].to_numpy(float)
            hi = summary["upper"].to_numpy(float)

            ax.plot(
                x,
                y,
                marker="o",
                lw=1.8,
                color=color,
                label=label,
            )
            ax.fill_between(
                x,
                lo,
                hi,
                color=color,
                alpha=0.15,
            )

        ax.set_xlabel("Tokens masked (%)")
        ax.set_ylabel("Mean absolute probability change")
        ax.set_title(f"{outcome}: Token masking")
        ax.grid(False)
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()

    stem = "Figure_S5_attribution_token_masking"
    paths = save_figure(fig, FIG_SUPP, stem)
    for p in paths:
        rec.figure(
            p.name,
            stability_src,
            "attribution stability and token masking",
            "SCD/PFD",
            "supplementary",
            notes=(
                f"Attribution sources: {stability_src.name}, "
                f"{randomization_src.name}; token-masking source: "
                f"{perturbation_src.name}"
            ),
        )

    write_caption(
        stem,
        "Attribution stability and token-masking diagnostics for SCD and PFD.",
    )


def make_misc_figures(rec: Recorder) -> None:
    make_calibration(rec)
    make_confusion(rec)
    make_endpoint_specific_s1(rec)
    make_attribution_2x2(rec)
    copy_existing_figures(rec)
    make_patient_examples(rec)
    rec.missing("Figure_2_multimodal_workflow", "No canonical existing workflow SVG/PDF found by repository/result-root search", "Provide current workflow figure or approve redraw")
    for stem in ["Text_ablation_roc_pr_curves", "Text_ablation_calibration_plots", "Fusion_architecture_selection_by_fold", "Text_shuffling_results"]:
        rec.missing(stem, "Targeted custom supplementary figure not generated in this conservative pass; source aggregate table is present", "Approve exact layout or use copied source table")
    rec.missing("Complete_decision_curves_supplementary_figure", "Removed from generated outputs per user review", "Use decision-curve table or main Figure 5 unless a specific supplementary purpose is approved")


def write_manifests(rec: Recorder, source_manifests: list[Path]) -> None:
    pd.DataFrame(rec.fig_rows).to_csv(MANIFESTS / "figure_source_manifest.csv", index=False)
    pd.DataFrame(rec.table_rows).to_csv(MANIFESTS / "table_source_manifest.csv", index=False)
    pd.DataFrame(rec.blocked).to_csv(MANIFESTS / "missing_or_blocked_outputs.csv", index=False)
    manifest = {
        "completed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "source_root_manifests": [{"path": str(p), "sha256": sha256(p)} for p in source_manifests],
        "no_retraining_statement": "No training, LLM generation, embedding generation, recalibration, threshold selection, or bootstrap analysis was launched.",
        "no_recalibration_statement": "Calibration outputs were copied or plotted from locked saved files.",
        "no_rebootstrap_statement": "Confidence intervals were copied from locked aggregate tables.",
        "primary_sensitivity_hierarchy": "Detailed-response endpoint-specific nested evaluation is primary; joint-response and literature-reduced analyses are supplementary sensitivity analyses.",
        "missing_or_blocked_outputs": rec.blocked,
        "software_versions": {"python": platform.python_version(), "platform": platform.platform(), "matplotlib": matplotlib.__version__, "pandas": pd.__version__},
        "plotting_settings": {"palette": MODEL_COLORS, "font": "Arial fallback sans-serif", "dpi": 300},
        "metric_checks": rec.metric_checks,
    }
    (MANIFESTS / "manuscript_output_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate_outputs() -> None:
    for table in list(TAB_MAIN.glob("*.csv")) + list(TAB_SUPP.glob("*.csv")) + list(TAB_NUM.glob("*.csv")):
        df = pd.read_csv(table, nrows=2)
        forbidden = {"Patient ID", "patient_id", "patient_key"}
        if forbidden.intersection(df.columns):
            raise SystemExit(f"Generated table contains patient identifier column: {table}")
    base = pd.read_csv(TAB_MAIN / "Table_1_baseline_characteristics_display.csv")
    if not {"Overall (N=730)", "No cardiac death (N=577)", "SCD (N=71)", "PFD (N=82)"}.issubset(base.columns):
        raise SystemExit("Baseline table missing required outcome columns")
    for pdf in list(FIG_MAIN.glob("*.pdf")) + list(FIG_SUPP.glob("*.pdf")):
        data = pdf.read_bytes()
        if not data.startswith(b"%PDF") or len(data) < 1000:
            raise SystemExit(f"Generated PDF failed header/size validation: {pdf}")
    for svg in list(FIG_MAIN.glob("*.svg")) + list(FIG_SUPP.glob("*.svg")):
        text = svg.read_text(encoding="utf-8", errors="ignore").lstrip()
        if "<svg" not in text[:1000] or "</svg>" not in text:
            raise SystemExit(f"Generated SVG failed basic validation: {svg}")


def write_readme(rec: Recorder) -> None:
    text = f"""# Manuscript Outputs

Generated from locked MUSIC four-year aggregate outputs.

Primary analysis: detailed-response v4 ECG, LLM/text, multimodal, and comparative outputs.
Sensitivity analysis: joint-response and representation/overlap ablations.
Post hoc supplementary analysis: literature-reduced continuous-LVEF outputs.

No model retraining, LLM generation, embedding generation, recalibration,
threshold reselection, bootstrap analysis, or completed result-directory edits
were performed.

Patient-level ECG/text attribution examples are generated under
`figures/patient_examples/` for local manuscript review. These examples may
contain linkable patient-level explanation material and should not be treated as
public aggregate outputs without separate review.

Missing or blocked outputs are listed in `manifests/missing_or_blocked_outputs.csv`.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    global OUT, FIG_MAIN, FIG_SUPP, FIG_RESTRICTED, TAB_MAIN, TAB_SUPP, TAB_NUM, CAPTIONS, MANIFESTS, LOGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_root
    FIG_MAIN = OUT / "figures/main"
    FIG_SUPP = OUT / "figures/supplementary"
    FIG_RESTRICTED = OUT / "figures/patient_examples"
    TAB_MAIN = OUT / "tables/main"
    TAB_SUPP = OUT / "tables/supplementary"
    TAB_NUM = OUT / "tables/numeric_source"
    CAPTIONS = OUT / "captions"
    MANIFESTS = OUT / "manifests"
    LOGS = OUT / "logs"
    ensure_dirs()
    rec = Recorder()
    source_manifests = preflight()
    for func in tqdm([make_flowchart, lambda r: figure_roc_pr(r, "roc"), lambda r: figure_roc_pr(r, "pr"), make_forest_table_only, make_dca, make_baseline_table, make_performance_tables, make_threshold_tables, make_supp_tables, make_misc_figures, make_patient_examples], desc="Generating manuscript outputs"):
        func(rec)
    write_readme(rec)
    write_manifests(rec, source_manifests)
    cleanup_generated_dirs()
    validate_outputs()
    print(json.dumps({
        "output_root": str(OUT),
        "main_figures": sorted(p.name for p in FIG_MAIN.glob("*.*")),
        "supplementary_figures": sorted(p.name for p in FIG_SUPP.glob("*.*")),
        "main_tables": sorted(p.name for p in TAB_MAIN.glob("*.csv")),
        "supplementary_tables": sorted(p.name for p in TAB_SUPP.glob("*.csv")),
        "blocked": rec.blocked,
        "log_dir": str(LOGS),
    }, indent=2))


if __name__ == "__main__":
    main()
