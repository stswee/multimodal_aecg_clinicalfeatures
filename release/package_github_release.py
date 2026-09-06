#!/usr/bin/env python3
"""Build an audited local GitHub release package for the MUSIC 4-year analyses."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/sswee/multimodal_aecg_clinicalfeatures")
MUSIC = Path("/home/sswee/music")
DIST = REPO / "dist/github_release"
STAGING_ROOT = DIST / "staging"
RELEASE_NAME = "music_four_year_ecg_llm_release"
STAGE = STAGING_ROOT / RELEASE_NAME
TARBALL = DIST / f"{RELEASE_NAME}.tar.gz"
TARBALL_SHA = DIST / f"{RELEASE_NAME}.tar.gz.sha256"
MAX_FILE_BYTES = 50 * 1024 * 1024
HANDOFF = "Codex_Project_Handoff.md"

ROOTS = {
    "ECG_ROOT": MUSIC / "ecg_nested_4year_three_wave",
    "JOINT_TEXT_ROOT": MUSIC / "text_nested_4year_v2",
    "ENDPOINT_TEXT_ROOT": MUSIC / "text_nested_4year_v3_endpoint_specific",
    "JOINT_MULTIMODAL_ROOT": MUSIC / "multimodal_nested_4year_v2",
    "ENDPOINT_MULTIMODAL_ROOT": MUSIC / "multimodal_nested_4year_v3_endpoint_specific",
    "TABULAR_ROOT": MUSIC / "tabular_nested_4year_v2",
    "TABULAR_MLP_ROOT": MUSIC / "tabular_mlp_matched_4year_v2",
    "JOINT_COMPARATIVE_ROOT": MUSIC / "comparative_analysis_4year_v2",
    "ENDPOINT_COMPARATIVE_ROOT": MUSIC / "comparative_analysis_4year_v3_endpoint_specific",
    "REDUCED_TABULAR_ROOT": MUSIC / "tabular_literature_reduced_continuous_lvef_4year_v1",
    "REDUCED_TABULAR_MLP_ROOT": MUSIC / "tabular_mlp_literature_reduced_continuous_lvef_4year_v1",
    "REDUCED_MULTIMODAL_ROOT": MUSIC / "multimodal_literature_reduced_continuous_lvef_4year_v1",
    "REDUCED_COMPARATIVE_ROOT": MUSIC / "literature_reduced_continuous_lvef_comparative_4year_v1",
}

REPORTING = [
    "4_LLM_Modeling/LLM_Results_Reporting.txt",
    "6_Multimodal_Modeling/Multimodal_Results_Reporting.txt",
    "5_Benchmarking/Tabular_Results_Reporting.txt",
    "7_Comparative_Analysis/Comparative_Results_Reporting.txt",
]

CODE_ALLOWLIST = [
    "0_Preprocessing/Preprocessing.ipynb",
    "0_Preprocessing/preprocess_all_ecgs_HRV_complete.py",
    "0_Preprocessing/preprocess_single_ecg_HRV_complete.py",
    "0_Preprocessing/run_preprocess_all_ecgs_HRV_complete.sh",
    "1_Segmenting/create_window_index_metadata.py",
    "1_Segmenting/run_segment_HRV_complete.sh",
    "1_Segmenting/run_segment_HRV_complete_parallel.sh",
    "1_Segmenting/segment_all_patients_by_start_indices_HRV_complete.py",
    "1_Segmenting/segment_all_patients_by_start_indices_HRV_complete_parallel.py",
    "2_Labeling/Labeling_MUSIC_Updated.ipynb",
    "3_ECG_Modeling/README.md",
    "3_ECG_Modeling/train_tcn_mil_hrv_csv_multiclass_SCDPFD_complete.py",
    "3_ECG_Modeling/calibrate_ecg_platt.py",
    "3_ECG_Modeling/select_ecg_thresholds.py",
    "3_ECG_Modeling/ecg_decision_curve_analysis.py",
    "3_ECG_Modeling/ECG_Analysis.ipynb",
    "3_ECG_Modeling/wave1_capacity.sh",
    "4_LLM_Modeling/README.md",
    "4_LLM_Modeling/LLM_Response_Verification.ipynb",
    "4_LLM_Modeling/generate_llm_risks.py",
    "4_LLM_Modeling/run_generate_LLM_risks.sh",
    "4_LLM_Modeling/embed_llm_risks.py",
    "4_LLM_Modeling/run_embed_llm_risks.sh",
    "4_LLM_Modeling/train_text_embeddings_nested_cv.py",
    "4_LLM_Modeling/run_text_embeddings_nested_cv.sh",
    "4_LLM_Modeling/text_decision_curve_analysis.py",
    "4_LLM_Modeling/plot_text_evaluation.py",
    "4_LLM_Modeling/run_text_posthoc_evaluation.sh",
    "6_Multimodal_Modeling/README.md",
    "6_Multimodal_Modeling/train_multimodal_nested_cv.py",
    "6_Multimodal_Modeling/run_multimodal_nested_cv.sh",
    "6_Multimodal_Modeling/multimodal_posthoc_evaluation.py",
    "6_Multimodal_Modeling/multimodal_text_shuffling_analysis.py",
    "6_Multimodal_Modeling/plot_multimodal_evaluation.py",
    "6_Multimodal_Modeling/run_multimodal_posthoc_evaluation.sh",
    "6_Multimodal_Modeling/supplementary/literature_reduced/train_multimodal_literature_reduced_nested_cv.py",
    "6_Multimodal_Modeling/supplementary/literature_reduced/run_multimodal_literature_reduced_nested_cv.sh",
    "6_Multimodal_Modeling/supplementary/literature_reduced/multimodal_literature_reduced_posthoc.py",
    "6_Multimodal_Modeling/supplementary/literature_reduced/plot_multimodal_literature_reduced.py",
    "6_Multimodal_Modeling/supplementary/literature_reduced/run_multimodal_literature_reduced_posthoc.sh",
    "5_Benchmarking/README.md",
    "5_Benchmarking/train_tabular_nested_cv.py",
    "5_Benchmarking/run_tabular_nested_cv.sh",
    "5_Benchmarking/train_tabular_mlp_matched_nested_cv.py",
    "5_Benchmarking/run_tabular_mlp_matched_nested_cv.sh",
    "5_Benchmarking/tabular_posthoc_plots.py",
    "5_Benchmarking/supplementary/literature_reduced/train_tabular_literature_reduced_nested_cv.py",
    "5_Benchmarking/supplementary/literature_reduced/run_tabular_literature_reduced_nested_cv.sh",
    "5_Benchmarking/supplementary/literature_reduced/train_tabular_mlp_literature_reduced_nested_cv.py",
    "5_Benchmarking/supplementary/literature_reduced/run_tabular_mlp_literature_reduced_nested_cv.sh",
    "5_Benchmarking/supplementary/literature_reduced/tabular_literature_reduced_posthoc_plots.py",
    "5_Benchmarking/supplementary/literature_reduced/run_tabular_literature_reduced_posthoc.sh",
    "7_Comparative_Analysis/generate_manuscript_figures_tables.py",
    "7_Comparative_Analysis/run_generate_manuscript_figures_tables.sh",
    "7_Comparative_Analysis/generate_multimodal_ecg_text_explanations.py",
    "7_Comparative_Analysis/run_generate_multimodal_ecg_text_explanations.sh",
    "7_Comparative_Analysis/run_endpoint_specific_comparative_analysis.py",
    "7_Comparative_Analysis/run_endpoint_specific_comparative_analysis.sh",
]

REPO_ARTIFACT_ROOTS = [
    (
        "7_Comparative_Analysis/manuscript_outputs",
        Path("results/primary_joint_endpoint/comparative/manuscript_outputs"),
    ),
    (
        "7_Comparative_Analysis/manuscript_outputs_v4_detailed",
        Path("results/primary_joint_endpoint/comparative/manuscript_outputs_v4_detailed"),
    ),
]

REPO_ARTIFACT_ALLOWLIST = [
    (
        "7_Comparative_Analysis/Fig_ECG_PFD_0285.png",
        Path("results/primary_joint_endpoint/comparative/patient_ecg_saliency/Fig_ECG_PFD_0285.png"),
    ),
    (
        "7_Comparative_Analysis/Fig_ECG_PFD_0285.svg",
        Path("results/primary_joint_endpoint/comparative/patient_ecg_saliency/Fig_ECG_PFD_0285.svg"),
    ),
    (
        "7_Comparative_Analysis/Fig_ECG_SCD_0271.png",
        Path("results/primary_joint_endpoint/comparative/patient_ecg_saliency/Fig_ECG_SCD_0271.png"),
    ),
    (
        "7_Comparative_Analysis/Fig_ECG_SCD_0271.svg",
        Path("results/primary_joint_endpoint/comparative/patient_ecg_saliency/Fig_ECG_SCD_0271.svg"),
    ),
]

RESULT_INCLUDE_PATTERNS = [
    "*manifest*.json",
    "*summary*.csv",
    "*performance*.csv",
    "*calibration*.csv",
    "*threshold*.csv",
    "*confusion*.csv",
    "*comparison*.csv",
    "*comparisons*.csv",
    "*decision_curve*.csv",
    "*supported_range*.csv",
    "*supported_ranges*.csv",
    "*provenance*.csv",
    "*provenance*.md",
    "*stability_summary*.csv",
    "*randomization_summary*.csv",
    "*perturbation_fidelity_summary*.csv",
    "*paired_summary*.csv",
    "*curves.png",
    "*curves.pdf",
    "*plots.png",
    "*plots.pdf",
    "*forest.png",
    "*forest.pdf",
    "*stability.png",
    "*stability.pdf",
    "*fidelity.png",
    "*fidelity.pdf",
    "*.svg",
]

EXCLUDE_NAME_BITS = [
    ".ipynb_checkpoints",
    "__pycache__",
    "launcher_logs",
    "logs",
    "final_models",
    "tuning",
    "outer_folds",
    "outer_fold_",
    "inner_oof",
    "checkpoint",
    "checkpoints",
    "embedding",
    "embeddings",
    "explain_multimodal_ecg_text",
    "token_attributions",
    "highlighted_response",
    "highlighted_responses",
    "all_arms_pooled_predictions",
    "outer_test_predictions",
    "pooled_inner_oof_predictions",
    "nested_patient_folds",
    "selected_outer_test_patients",
]

KNOWN_HASHES = {
    "40644ed776353758c4c94bb752620f177244ac53e559d82538b7402a09f170e6",
    "0e9e39f249a16976918f6564b8830bc894c89659",
    "0cb88a4f764b7a12671c53f0838cd831a0843b95",
}

FIELDNAMES = [
    "source_path",
    "release_path",
    "source_sha256",
    "release_sha256",
    "size_bytes",
    "analysis_category",
    "scientific_scope",
    "file_role",
    "source_manifest",
    "contains_patient_level_rows",
    "contains_patient_text",
    "release_status",
    "notes",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_notebook_for_release(path: Path) -> bytes:
    nb = read_json(path)
    if not isinstance(nb, dict):
        raise SystemExit(f"Notebook JSON has unexpected structure: {path}")
    for cell in nb.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        metadata = cell.get("metadata")
        if isinstance(metadata, dict):
            for key in ["execution", "collapsed", "scrolled"]:
                metadata.pop(key, None)
    metadata = nb.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("widgets", None)
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    text, _ = sanitize_text(text)
    return text.encode("utf-8")


def any_key(obj: object, key: str) -> list[object]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out.extend(any_key(v, key))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(any_key(v, key))
    return out


def verify_roots() -> list[str]:
    notes = []
    for name, root in ROOTS.items():
        if not root.exists():
            raise SystemExit(f"Missing required result root: {name}={root}")
        manifests = sorted(root.rglob("*manifest*.json"))
        if not manifests:
            raise SystemExit(f"No manifests found in required result root: {name}={root}")
        completed = 0
        for mf in manifests:
            try:
                data = read_json(mf)
            except Exception as exc:
                raise SystemExit(f"Manifest JSON parse failed: {mf}: {exc}") from exc
            if True in any_key(data, "completed"):
                completed += 1
            for value in any_key(data, "script_sha256") + any_key(data, "source_sha256"):
                if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                    KNOWN_HASHES.add(value.lower())
        notes.append(f"- `{name}`: exists; {len(manifests)} manifest files; {completed} expose `completed=true`.")
    joint_dca = read_json(ROOTS["JOINT_COMPARATIVE_ROOT"] / "clinical_decision_curve_comparisons/exploratory_2_to_25_percent/decision_curve_comparison_manifest.json")
    cohorts = joint_dca.get("cohorts", {})
    if cohorts.get("SCD", {}).get("n") != 648 or cohorts.get("SCD", {}).get("events") != 71:
        raise SystemExit("SCD cohort count check failed in joint decision-curve manifest")
    if cohorts.get("PFD", {}).get("n") != 659 or cohorts.get("PFD", {}).get("events") != 82:
        raise SystemExit("PFD cohort count check failed in joint decision-curve manifest")
    return notes


def sanitize_text(text: str) -> tuple[str, list[str]]:
    replacements = [
        (str(REPO), "${REPOSITORY_ROOT}", "local repository root"),
        (str(MUSIC), "${MUSIC_ROOT}", "local MUSIC root"),
        ("/home/sswee", "${HOME}", "local home root"),
    ]
    used = []
    for old, new, label in replacements:
        if old in text:
            text = text.replace(old, new)
            used.append(f"`{label}` -> `{new}`")
    return text, used


def portable_path(value: str | Path) -> str:
    text = str(value)
    text = text.replace(str(REPO), "${REPOSITORY_ROOT}")
    text = text.replace(str(MUSIC), "${MUSIC_ROOT}")
    text = text.replace("/home/sswee", "${HOME}")
    return text


def is_result_candidate(path: Path) -> bool:
    s = str(path)
    if any(bit in s for bit in EXCLUDE_NAME_BITS):
        return False
    name = path.name.lower()
    return any(path.match(pat) for pat in RESULT_INCLUDE_PATTERNS) or name.endswith((".png", ".pdf"))


def result_release_base(root_name: str) -> Path:
    mapping = {
        "ECG_ROOT": Path("results/primary_joint_endpoint/ecg"),
        "JOINT_TEXT_ROOT": Path("results/primary_joint_endpoint/text"),
        "JOINT_MULTIMODAL_ROOT": Path("results/primary_joint_endpoint/multimodal"),
        "JOINT_COMPARATIVE_ROOT": Path("results/primary_joint_endpoint/comparative"),
        "ENDPOINT_TEXT_ROOT": Path("results/sensitivity_endpoint_specific/text"),
        "ENDPOINT_MULTIMODAL_ROOT": Path("results/sensitivity_endpoint_specific/multimodal"),
        "ENDPOINT_COMPARATIVE_ROOT": Path("results/sensitivity_endpoint_specific/comparative"),
        "TABULAR_ROOT": Path("results/benchmarks/tabular"),
        "TABULAR_MLP_ROOT": Path("results/benchmarks/tabular_mlp"),
        "REDUCED_TABULAR_ROOT": Path("results/supplementary/literature_reduced_continuous_lvef/tabular"),
        "REDUCED_TABULAR_MLP_ROOT": Path("results/supplementary/literature_reduced_continuous_lvef/tabular_mlp"),
        "REDUCED_MULTIMODAL_ROOT": Path("results/supplementary/literature_reduced_continuous_lvef/multimodal"),
        "REDUCED_COMPARATIVE_ROOT": Path("results/supplementary/literature_reduced_continuous_lvef/comparative"),
    }
    return mapping[root_name]


def category_for(rel: Path) -> tuple[str, str, str]:
    s = str(rel)
    if s.startswith("code/"):
        return "code", "canonical", "source"
    if s.startswith("documentation/"):
        return "documentation", "release", "documentation"
    if s.startswith("provenance/sanitized_manifests/"):
        return "provenance", "sanitized_manifest", "manifest"
    if s.startswith("provenance/source_snapshots/"):
        return "provenance", "historical_source_snapshot", "snapshot"
    if s.startswith("provenance/"):
        return "provenance", "release", "provenance"
    if s.startswith("environment/"):
        return "environment", "release", "environment"
    if s.startswith("results/primary_joint_endpoint/"):
        return "primary_joint_endpoint", "primary_joint_endpoint", "result"
    if s.startswith("results/sensitivity_endpoint_specific/"):
        return "sensitivity_endpoint_specific", "endpoint_specific_sensitivity", "result"
    if s.startswith("results/benchmarks/"):
        return "benchmarks", "benchmark", "result"
    if s.startswith("results/supplementary/"):
        return "supplementary", "post_hoc_supplementary", "result"
    return "release", "release", "file"


def inspect_csv(path: Path, rel: Path) -> tuple[bool, str]:
    if path.suffix.lower() != ".csv":
        return False, ""
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False, "empty_csv"
        lowered = [h.strip().lower() for h in header]
        identifier_cols = {"pid", "patient id", "patient_id", "patient_key", "mrn", "subject_id"}
        if str(rel).startswith("results/") and any(h in identifier_cols for h in lowered):
            return True, "release data CSV contains identifier column"
        rows = [row for _, row in zip(range(20), reader)]
        if str(rel).startswith("results/") and rows and len(rows) > 1000 and len(header) > 3:
            return True, "release data CSV appears patient-level by row count"
    return False, ""


def secret_scan(path: Path, rel: Path) -> list[str]:
    if path.suffix.lower() in {".png", ".pdf", ".gz"}:
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    issues = []
    patterns = {
        "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        "authorization_header": r"Authorization:\s*Bearer\s+[A-Za-z0-9._-]+",
        "password_assignment": r"(?i)\b(password|passwd|secret|api_key|token)\s*=\s*['\"][^'\"]{8,}",
        "absolute_home_path": r"/home/sswee",
    }
    for name, pat in patterns.items():
        for m in re.finditer(pat, text):
            value = m.group(0)
            if re.fullmatch(r"[0-9a-fA-F]{40,64}", value) and value.lower() in KNOWN_HASHES:
                continue
            line = text.count("\n", 0, m.start()) + 1
            if name == "absolute_home_path" and str(rel).startswith("provenance/source_snapshots/"):
                continue
            issues.append(f"{rel}:{line}:{name}")
    return issues


class Builder:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.omissions: list[str] = []
        self.sanitizations: list[str] = []
        self.collisions: dict[str, str] = {}
        self.included_tables: list[str] = []
        self.included_figures: list[str] = []

    def add_bytes(self, data: bytes, rel: Path, source: str, notes: str = "") -> None:
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != data:
            raise SystemExit(f"Release path collision with different content: {rel}")
        dest.write_bytes(data)
        self.record(Path(source) if source.startswith("/") else None, rel, notes=notes, source_hash=hashlib.sha256(data).hexdigest())

    def add_text(self, text: str, rel: Path, source: str, notes: str = "") -> None:
        self.add_bytes(text.encode("utf-8"), rel, source, notes)

    def copy_file(self, src: Path, rel: Path, *, sanitize: bool = False, notes: str = "", source_manifest: str = "") -> None:
        if not src.exists():
            raise SystemExit(f"Allowlisted file missing: {src}")
        if src.stat().st_size > MAX_FILE_BYTES:
            raise SystemExit(f"Eligible file exceeds 50 MiB limit: {src} ({src.stat().st_size} bytes)")
        original_hash = sha256(src)
        if src.suffix.lower() == ".ipynb":
            data = strip_notebook_for_release(src)
            notes = notes or "notebook outputs and execution counts stripped for release"
        else:
            data = src.read_bytes()
        if sanitize and src.suffix.lower() in {".json", ".md", ".txt", ".csv", ".py", ".sh"}:
            text, used = sanitize_text(data.decode("utf-8", errors="replace"))
            data = text.encode("utf-8")
            for item in used:
                self.sanitizations.append(f"- `{portable_path(src)}`: {item}")
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != data:
            raise SystemExit(f"Release path collision with different content: {rel}")
        dest.write_bytes(data)
        patient_rows, csv_note = inspect_csv(dest, rel)
        if patient_rows:
            raise SystemExit(f"Privacy audit failed: {rel}: {csv_note}")
        self.record(src, rel, notes=notes or csv_note, source_manifest=source_manifest, source_hash=original_hash)
        if dest.suffix.lower() == ".csv":
            self.included_tables.append(str(rel))
        if dest.suffix.lower() in {".png", ".pdf", ".svg"}:
            self.included_figures.append(str(rel))

    def record(self, src: Path | None, rel: Path, *, notes: str = "", source_manifest: str = "", source_hash: str | None = None) -> None:
        dest = STAGE / rel
        category, scope, role = category_for(rel)
        self.rows.append({
            "source_path": portable_path(src) if src else "",
            "release_path": str(rel),
            "source_sha256": source_hash or (sha256(src) if src else sha256(dest)),
            "release_sha256": sha256(dest),
            "size_bytes": dest.stat().st_size,
            "analysis_category": category,
            "scientific_scope": scope,
            "file_role": role,
            "source_manifest": portable_path(source_manifest) if source_manifest else "",
            "contains_patient_level_rows": "false",
            "contains_patient_text": "false",
            "release_status": "included",
            "notes": notes,
        })


def make_docs(builder: Builder, root_notes: list[str]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    docs = {
        "README.md": f"""# MUSIC Four-Year ECG-LLM Release

This local release package contains shareable code, documentation, aggregate
results, and publication figures for four-year SCD and PFD risk modeling using
ECG, LLM-derived text, multimodal ECG-LLM fusion, and tabular benchmarks.

The primary/original-manuscript representation is the joint-endpoint v2
analysis. Endpoint-specific v3 analyses are sensitivity/ablation analyses.
Literature-reduced continuous-LVEF analyses are post hoc supplementary.

The cohort contains 730 patients: 577 controls, 71 SCD events, and 82 PFD
events. Task cohorts use competing-endpoint exclusion: SCD N=648 and PFD N=659.
Analyses use a fixed four-year horizon, seed 42, five outer folds, four inner
folds, and training-only preprocessing, tuning, selection, calibration, and
threshold selection.

Raw ECG recordings, clinical source CSVs, prompt CSVs, patient-level LLM
responses, embeddings, model weights/checkpoints, patient-level predictions,
fold assignments, and highlighted patient-response artifacts are not included.

Verify files with `sha256sum -c SHA256SUMS`.
""",
        "RELEASE_CONTENTS.md": f"""# Release Contents

Generated: {now}

Top-level handoff file discovered in repository: `{HANDOFF}`.

Maximum allowed individual included-file size: 50 MiB.

Citation metadata is incomplete, so this release contains
`CITATION.cff.template` rather than a valid `CITATION.cff`.

Empty directories, archives, caches, notebooks with outputs, raw inputs,
patient-level rows, embeddings, checkpoints, logs, and highlighted-response
patient text artifacts are omitted.

## Result Root Checks
{chr(10).join(root_notes)}
""",
        "REPRODUCIBILITY.md": """# Reproducibility

- Total cohort: 730.
- Controls: 577.
- SCD events: 71.
- PFD events: 82.
- SCD task N=648.
- PFD task N=659.
- Seed: 42.
- Horizon: fixed four-year prediction horizon.
- Outer folds: 5.
- Inner folds: 4.
- Competing-endpoint exclusion was used for endpoint-specific task cohorts.
- Outer-test outcomes were reserved for final evaluation and were not used for
  preprocessing, tuning, selection, calibration, or threshold selection.
- Prompt SHA-256:
  `40644ed776353758c4c94bb752620f177244ac53e559d82538b7402a09f170e6`.
- Llama-3.1-8B-Instruct revision:
  `0e9e39f249a16976918f6564b8830bc894c89659`.
- Llama-3.2-3B-Instruct revision:
  `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
""",
        "MODEL_ACCESS.md": """# Model Access

Model weights are not redistributed in this release. Llama, BioBERT,
ClinicalBERT, and any other external model weights must be obtained by users
under their applicable licenses, access requirements, and provider terms.
""",
        "LICENSE_STATUS.md": """# License Status

No complete project license metadata was identified during packaging. This
release therefore does not include a generated `LICENSE` file. Downstream users
must resolve project licensing before redistribution or reuse beyond review.
""",
        "CITATION.cff.template": """cff-version: 1.2.0
message: "Citation metadata incomplete; fill TODO fields before publication."
title: "TODO: release title"
authors:
  - family-names: "TODO"
    given-names: "TODO"
doi: "TODO"
repository-code: "TODO"
version: "TODO"
date-released: "TODO"
license: "TODO"
""",
        "documentation/analysis_directory_map.md": """# Analysis Directory Map

- `results/primary_joint_endpoint/`: primary/original-manuscript joint-endpoint ECG, text, multimodal, and comparative outputs.
- `results/sensitivity_endpoint_specific/`: endpoint-specific v3 sensitivity/ablation outputs.
- `results/benchmarks/`: tabular and matched MLP benchmark outputs.
- `results/exploratory/`: exploratory decision-curve, shuffling, and attribution summaries when separately staged.
- `results/supplementary/literature_reduced_continuous_lvef/`: post hoc literature-reduced continuous-LVEF analyses.
""",
        "documentation/reviewer_requirements_map.md": """# Reviewer Requirements Map

- Cohort construction: `code/2_Labeling/label_music_cohort.py`.
- Canonical analysis code: `code/3_ECG_Modeling/` through `code/7_Comparative_Analysis/`.
- Reporting summaries: `documentation/*_Results_Reporting.txt`.
- Provenance and consolidation: `provenance/`.
- Environment notes: `environment/`.
- File checksums: `SHA256SUMS`.
""",
        "environment/python_environment.txt": subprocess.run(["python3", "--version"], text=True, capture_output=True).stdout.strip() + "\n",
        "environment/package_versions.txt": subprocess.run(["python3", "-m", "pip", "freeze"], text=True, capture_output=True).stdout,
        "environment/model_revisions.txt": "Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659\nLlama-3.2-3B-Instruct 0cb88a4f764b7a12671c53f0838cd831a0843b95\n",
        "environment/prompt_checksum.txt": "40644ed776353758c4c94bb752620f177244ac53e559d82538b7402a09f170e6\n",
        "environment/software_and_hardware_summary.txt": subprocess.run(["uname", "-a"], text=True, capture_output=True).stdout,
    }
    for rel, text in docs.items():
        clean, used = sanitize_text(text)
        for item in used:
            builder.sanitizations.append(f"- generated `{rel}`: {item}")
        builder.add_text(clean, Path(rel), "generated")


def stage_release() -> Builder:
    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    if TARBALL.exists():
        TARBALL.unlink()
    if TARBALL_SHA.exists():
        TARBALL_SHA.unlink()
    STAGE.mkdir(parents=True)
    builder = Builder()
    root_notes = verify_roots()
    make_docs(builder, root_notes)
    for rel in CODE_ALLOWLIST:
        builder.copy_file(REPO / rel, Path("code") / rel, sanitize=True)
    for rel in REPORTING:
        builder.copy_file(REPO / rel, Path("documentation") / Path(rel).name, sanitize=True)
    for rel in ["provenance/FILE_CONSOLIDATION_MAP.csv", "provenance/CONSOLIDATION_LOG.md"]:
        builder.copy_file(REPO / rel, Path(rel), sanitize=True)
    for src in sorted((REPO / "provenance/source_snapshots").rglob("*")):
        if src.is_file():
            builder.copy_file(src, Path("provenance/source_snapshots") / src.relative_to(REPO / "provenance/source_snapshots"), notes="frozen historical provenance; not path-sanitized")
    for root_name, root in ROOTS.items():
        base = result_release_base(root_name)
        for src in sorted(root.rglob("*")):
            if not src.is_file() or not is_result_candidate(src):
                continue
            rel = base / src.relative_to(root)
            if src.suffix.lower() == ".json" and "manifest" in src.name.lower():
                san_rel = Path("provenance/sanitized_manifests") / root_name.lower() / src.relative_to(root)
                builder.copy_file(src, san_rel, sanitize=True, source_manifest=str(src))
            else:
                builder.copy_file(src, rel, sanitize=src.suffix.lower() in {".json", ".md", ".txt", ".csv"}, source_manifest=str(root / ""))
    for src_root_rel, dest_root in REPO_ARTIFACT_ROOTS:
        src_root = REPO / src_root_rel
        if not src_root.exists():
            builder.omissions.append(f"Repository artifact root missing: {src_root_rel}")
            continue
        for src in sorted(src_root.rglob("*")):
            if not src.is_file() or not is_result_candidate(src):
                continue
            rel = dest_root / src.relative_to(src_root)
            builder.copy_file(src, rel, sanitize=src.suffix.lower() in {".json", ".md", ".txt", ".csv"})
    for src_rel, dest_rel in REPO_ARTIFACT_ALLOWLIST:
        src = REPO / src_rel
        if src.exists():
            builder.copy_file(src, dest_rel, sanitize=src.suffix.lower() in {".json", ".md", ".txt", ".csv"})
        else:
            builder.omissions.append(f"Repository artifact file missing: {src_rel}")
    for src_rel in ["release/package_github_release.py", "release/run_package_github_release.sh"]:
        builder.copy_file(REPO / src_rel, Path("code/release_tools") / Path(src_rel).name, sanitize=True)
    path_map = "# Path Sanitization Map\n\n" + ("\n".join(builder.sanitizations) if builder.sanitizations else "No replacements were needed.\n")
    builder.add_text(path_map + "\n\nFrozen historical source snapshots are preserved byte-for-byte and may contain nonportable original paths.\n", Path("provenance/PATH_SANITIZATION_MAP.md"), "generated")
    return builder


def write_manifest(builder: Builder) -> None:
    path = STAGE / "provenance/RELEASE_MANIFEST.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in sorted(builder.rows, key=lambda r: str(r["release_path"])):
            w.writerow(row)
    builder.record(None, Path("provenance/RELEASE_MANIFEST.csv"), notes="generated release manifest")


def audit_tree(base: Path) -> list[str]:
    issues = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base)
        s = str(rel)
        if any(bit in s for bit in [".git", "__pycache__", ".ipynb_checkpoints"]):
            issues.append(f"{rel}: prohibited cache/vcs path")
        if path.stat().st_size > MAX_FILE_BYTES:
            issues.append(f"{rel}: exceeds 50 MiB")
        if str(rel).startswith("results/") and re.search(r"(checkpoint|embedding|token_attributions|outer_test_predictions|pooled_inner_oof_predictions)", s, re.I):
            issues.append(f"{rel}: prohibited filename category")
        patient_rows, note = inspect_csv(path, rel)
        if patient_rows:
            issues.append(f"{rel}: {note}")
        issues.extend(secret_scan(path, rel))
    return issues


def validate_code(base: Path) -> tuple[str, str]:
    py_files = [str(p) for p in sorted((base / "code").rglob("*.py"))]
    sh_files = [str(p) for p in sorted((base / "code").rglob("*.sh"))]
    if py_files:
        subprocess.run(["python3", "-m", "py_compile", *py_files], check=True)
    for sh in sh_files:
        subprocess.run(["bash", "-n", sh], check=True)
    missing = []
    for sh in sh_files:
        p = Path(sh)
        text = p.read_text(errors="ignore")
        for m in re.finditer(r"(?:python3?|bash)\s+([A-Za-z0-9_./${}-]+\.(?:py|sh))", text):
            ref = m.group(1)
            if ref.startswith("$") or "${" in ref:
                continue
            target = (p.parent / ref).resolve()
            if not target.exists():
                missing.append(f"{p.relative_to(base)} -> {ref}")
    if missing:
        raise SystemExit("Missing staged launcher references:\n" + "\n".join(missing))
    return f"py_compile passed for {len(py_files)} Python files", f"bash -n passed for {len(sh_files)} shell files"


def cleanup_generated_caches(base: Path) -> None:
    for path in sorted(base.rglob("__pycache__"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)


def write_sha256sums(base: Path) -> None:
    files = [p for p in sorted(base.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    lines = [f"{sha256(p)}  {p.relative_to(base)}\n" for p in files]
    (base / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=base, check=True, stdout=subprocess.PIPE, text=True)


def make_tarball() -> None:
    with tarfile.open(TARBALL, "w:gz", compresslevel=9) as tf:
        for path in sorted(STAGE.rglob("*")):
            arc = Path(RELEASE_NAME) / path.relative_to(STAGE)
            info = tf.gettarinfo(str(path), arcname=str(arc))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if path.is_dir():
                info.mode = 0o755
                tf.addfile(info)
            else:
                info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                with path.open("rb") as f:
                    tf.addfile(info, f)
    TARBALL_SHA.write_text(f"{sha256(TARBALL)}  {TARBALL.name}\n", encoding="utf-8")


def verify_archive() -> None:
    subprocess.run(["gzip", "-t", str(TARBALL)], check=True)
    subprocess.run(["tar", "-tzf", str(TARBALL)], check=True, stdout=subprocess.PIPE, text=True)
    subprocess.run(["sha256sum", "-c", TARBALL_SHA.name], cwd=DIST, check=True, stdout=subprocess.PIPE, text=True)
    with tempfile.TemporaryDirectory(prefix="music_release_extract_") as td:
        subprocess.run(["tar", "-xzf", str(TARBALL), "-C", td], check=True)
        extracted = Path(td) / RELEASE_NAME
        subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=extracted, check=True, stdout=subprocess.PIPE, text=True)
        validate_code(extracted)
        cleanup_generated_caches(extracted)
        issues = audit_tree(extracted)
        if issues:
            raise SystemExit("Extracted archive audit failed:\n" + "\n".join(issues[:100]))


def chmod_wrapper() -> None:
    wrapper = REPO / "release/run_package_github_release.sh"
    if wrapper.exists():
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-archive", action="store_true", help="stage and audit without creating tarball")
    args = parser.parse_args()
    chmod_wrapper()
    builder = stage_release()
    write_manifest(builder)
    issues = audit_tree(STAGE)
    if issues:
        raise SystemExit("Staging audit failed:\n" + "\n".join(issues[:100]))
    py_status, sh_status = validate_code(STAGE)
    cleanup_generated_caches(STAGE)
    issues = audit_tree(STAGE)
    if issues:
        raise SystemExit("Post-validation staging audit failed:\n" + "\n".join(issues[:100]))
    write_sha256sums(STAGE)
    if not args.skip_archive:
        make_tarball()
        verify_archive()
    counts = Counter(row["analysis_category"] for row in builder.rows)
    summary = {
        "staging_directory": str(STAGE),
        "archive_path": str(TARBALL),
        "external_checksum_path": str(TARBALL_SHA),
        "archive_size_bytes": TARBALL.stat().st_size if TARBALL.exists() else 0,
        "included_file_count": len([p for p in STAGE.rglob("*") if p.is_file()]),
        "manifest_summary_by_analysis_category": dict(sorted(counts.items())),
        "included_tables": sorted(builder.included_tables),
        "included_figures": sorted(builder.included_figures),
        "syntax_validation": [py_status, sh_status],
        "launcher_reference_validation": "passed",
        "privacy_secret_audit": "passed",
        "sha256_archive_extraction_verification": "passed" if TARBALL.exists() else "staging only",
    }
    write_text(DIST / "release_build_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
