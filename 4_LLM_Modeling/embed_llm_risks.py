#!/usr/bin/env python3

"""Create reusable patient-level embeddings with frozen BERT encoders.

This is an outcome-independent feature-extraction step. It does not fit,
fine-tune, scale, select, or compare encoders. Each condition is saved as one
auditable NPZ matrix plus a JSON manifest. Optional per-patient NPY files are
available for compatibility with older downstream code. Texts longer than the
encoder limit are split into overlapping chunks, and the chunk-level CLS
embeddings are averaged so that the complete response contributes.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


ENCODER_MODELS = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed frozen LLM-derived text with BioBERT/ClinicalBERT."
    )
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--patient_id_col", default="Patient ID")
    parser.add_argument("--output_root", default="text_embeddings_4year")
    parser.add_argument("--source_name", required=True)
    parser.add_argument(
        "--risk_prefix",
        action="append",
        default=[],
        help=(
            "Parsed risk-response prefix, such as full_risk_no_ecg. "
            "Each prefix produces endpoint-specific SCD and PFD full, "
            "label-only, and rationale-only representations plus one joint "
            "SCD+PFD full-response ablation. May be repeated."
        ),
    )
    parser.add_argument(
        "--text_spec",
        action="append",
        nargs=2,
        metavar=("CONDITION", "COLUMN"),
        default=[],
        help="Embed a text column under a stable condition name. May be repeated.",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=sorted(ENCODER_MODELS),
        default=["BioBERT", "ClinicalBERT"],
    )
    parser.add_argument(
        "--encoder_revision",
        action="append",
        default=[],
        metavar="ENCODER=COMMIT",
        help=(
            "Immutable encoder revision, repeated once per requested encoder. "
            "Example: BioBERT=<40-character commit hash>."
        ),
    )
    parser.add_argument("--pooling", choices=["cls"], default="cls")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--long_text_strategy",
        choices=["mean_chunks", "truncate"],
        default="mean_chunks",
        help=(
            "For text exceeding max_length, either average CLS embeddings "
            "from overlapping chunks or truncate the text."
        ),
    )
    parser.add_argument(
        "--chunk_stride",
        type=int,
        default=64,
        help=(
            "Number of overlapping tokens between adjacent chunks when "
            "--long_text_strategy=mean_chunks."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--save_per_patient", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    revisions = {}
    for value in args.encoder_revision:
        if "=" not in value:
            parser.error(f"Invalid --encoder_revision value: {value!r}")
        encoder, revision = value.split("=", 1)
        if encoder not in ENCODER_MODELS or not re.fullmatch(r"[0-9a-f]{40}", revision):
            parser.error(
                "--encoder_revision must be ENCODER=<40-character lowercase commit hash>"
            )
        revisions[encoder] = revision
    missing = set(args.encoders) - set(revisions)
    if missing:
        parser.error(f"Missing immutable revisions for encoders: {sorted(missing)}")
    if args.max_length < 8:
        parser.error("--max_length must be at least 8.")
    if args.chunk_stride < 0:
        parser.error("--chunk_stride must be nonnegative.")
    if args.chunk_stride >= args.max_length - 2:
        parser.error("--chunk_stride must be smaller than max_length - 2.")
    args.encoder_revisions = revisions
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_slug(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def require_columns(dataframe, columns, context):
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def clean_text_series(series, condition):
    if series.isna().any():
        raise ValueError(
            f"Condition '{condition}' contains {int(series.isna().sum())} missing texts."
        )

    cleaned = series.astype(str).str.strip()
    empty_count = int(cleaned.eq("").sum())
    if empty_count:
        raise ValueError(
            f"Condition '{condition}' contains {empty_count} empty texts."
        )
    return cleaned


def canonical_risk_specs(dataframe, prefix):
    columns = {
        "scd_risk": f"{prefix}_scd_risk",
        "scd_rationale": f"{prefix}_scd_rationale",
        "pfd_risk": f"{prefix}_pfd_risk",
        "pfd_rationale": f"{prefix}_pfd_rationale",
    }
    require_columns(dataframe, list(columns.values()), prefix)

    status_column = f"{prefix}_postprocessed_status"
    if status_column in dataframe.columns:
        incomplete = ~dataframe[status_column].eq("complete")
        if incomplete.any():
            raise ValueError(
                f"{prefix} has {int(incomplete.sum())} responses not marked complete."
            )

    for column in columns.values():
        if dataframe[column].isna().any():
            raise ValueError(
                f"{prefix} has missing parsed values in '{column}'."
            )

    scd_risk = dataframe[columns["scd_risk"]].astype(str).str.title()
    pfd_risk = dataframe[columns["pfd_risk"]].astype(str).str.title()
    scd_rationale = dataframe[columns["scd_rationale"]].astype(str).str.strip()
    pfd_rationale = dataframe[columns["pfd_rationale"]].astype(str).str.strip()

    joint_full_text = (
        "SCD_RISK: "
        + scd_risk
        + "\nSCD_RATIONALE: "
        + scd_rationale
        + "\nPFD_RISK: "
        + pfd_risk
        + "\nPFD_RATIONALE: "
        + pfd_rationale
    )
    source_columns = list(columns.values())
    return [
        {
            "condition": f"{prefix}_full",
            "texts": joint_full_text,
            "construction": "canonical_joint_endpoint_full_risk_response",
            "source_columns": source_columns,
            "endpoint_scope": "joint_scd_pfd",
        },
        {
            "condition": f"{prefix}_scd_full",
            "texts": "SCD_RISK: " + scd_risk + "\nSCD_RATIONALE: " + scd_rationale,
            "construction": "canonical_endpoint_specific_full_risk_response",
            "source_columns": [columns["scd_risk"], columns["scd_rationale"]],
            "endpoint_scope": "scd",
        },
        {
            "condition": f"{prefix}_scd_label_only",
            "texts": "SCD_RISK: " + scd_risk,
            "construction": "canonical_endpoint_specific_label_only_response",
            "source_columns": [columns["scd_risk"]],
            "endpoint_scope": "scd",
        },
        {
            "condition": f"{prefix}_scd_rationale_only",
            "texts": "SCD_RATIONALE: " + scd_rationale,
            "construction": "canonical_endpoint_specific_rationale_only_response",
            "source_columns": [columns["scd_rationale"]],
            "endpoint_scope": "scd",
        },
        {
            "condition": f"{prefix}_pfd_full",
            "texts": "PFD_RISK: " + pfd_risk + "\nPFD_RATIONALE: " + pfd_rationale,
            "construction": "canonical_endpoint_specific_full_risk_response",
            "source_columns": [columns["pfd_risk"], columns["pfd_rationale"]],
            "endpoint_scope": "pfd",
        },
        {
            "condition": f"{prefix}_pfd_label_only",
            "texts": "PFD_RISK: " + pfd_risk,
            "construction": "canonical_endpoint_specific_label_only_response",
            "source_columns": [columns["pfd_risk"]],
            "endpoint_scope": "pfd",
        },
        {
            "condition": f"{prefix}_pfd_rationale_only",
            "texts": "PFD_RATIONALE: " + pfd_rationale,
            "construction": "canonical_endpoint_specific_rationale_only_response",
            "source_columns": [columns["pfd_rationale"]],
            "endpoint_scope": "pfd",
        },
    ]


def build_text_specs(dataframe, risk_prefixes, text_specs):
    specs = []
    seen_conditions = set()

    for prefix in risk_prefixes:
        for spec in canonical_risk_specs(dataframe, prefix):
            if spec["condition"] in seen_conditions:
                raise ValueError(f"Duplicate condition: {spec['condition']}")
            seen_conditions.add(spec["condition"])
            spec["texts"] = clean_text_series(
                spec["texts"], spec["condition"]
            )
            specs.append(spec)

    for condition, column in text_specs:
        require_columns(dataframe, [column], condition)
        if condition in seen_conditions:
            raise ValueError(f"Duplicate condition: {condition}")
        seen_conditions.add(condition)
        specs.append(
            {
                "condition": condition,
                "texts": clean_text_series(dataframe[column], condition),
                "construction": "direct_text_column",
                "source_columns": [column],
                "endpoint_scope": "shared",
            }
        )

    if not specs:
        raise ValueError("Specify at least one --risk_prefix or --text_spec.")

    return specs


def aggregate_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", prefix=f".{path.stem}.", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f".{path.stem}.",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def artifact_matches(manifest_path, embedding_path, expected_signature):
    if not manifest_path.exists() or not embedding_path.exists():
        return False
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("cache_signature") == expected_signature


def embed_condition(
    model,
    tokenizer,
    texts,
    patient_ids,
    device,
    batch_size,
    max_length,
    long_text_strategy,
    chunk_stride,
):
    tokenized_untruncated = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    token_lengths = np.asarray(
        [len(input_ids) for input_ids in tokenized_untruncated["input_ids"]],
        dtype=np.int32,
    )
    exceeded_single_window = token_lengths > max_length

    batches = []
    chunk_count_batches = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding",
        leave=False,
    ):
        batch_texts = texts[start : start + batch_size]
        if long_text_strategy == "mean_chunks":
            model_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                stride=chunk_stride,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            overflow_mapping = model_inputs.pop(
                "overflow_to_sample_mapping"
            ).cpu().numpy()
        else:
            model_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            overflow_mapping = np.arange(
                len(batch_texts),
                dtype=np.int64,
            )

        model_inputs = model_inputs.to(device)

        with torch.inference_mode():
            outputs = model(**model_inputs)
            chunk_embeddings = (
                outputs.last_hidden_state[:, 0, :]
                .detach()
                .cpu()
                .float()
                .numpy()
            )

        chunk_counts = np.bincount(
            overflow_mapping,
            minlength=len(batch_texts),
        ).astype(np.int32)

        if np.any(chunk_counts == 0):
            raise RuntimeError(
                "At least one text produced no encoder chunks."
            )

        patient_embeddings = np.stack(
            [
                chunk_embeddings[
                    overflow_mapping == local_index
                ].mean(axis=0)
                for local_index in range(len(batch_texts))
            ],
            axis=0,
        )

        batches.append(patient_embeddings)
        chunk_count_batches.append(chunk_counts)

    embedding_matrix = np.concatenate(batches, axis=0).astype(
        np.float32, copy=False
    )
    chunk_counts = np.concatenate(chunk_count_batches).astype(
        np.int32, copy=False
    )

    was_chunked = chunk_counts > 1
    was_truncated = (
        exceeded_single_window
        if long_text_strategy == "truncate"
        else np.zeros(len(texts), dtype=bool)
    )

    if embedding_matrix.shape[0] != len(patient_ids):
        raise RuntimeError("Embedding row count does not match patient count.")
    if not np.isfinite(embedding_matrix).all():
        raise RuntimeError("Embedding matrix contains non-finite values.")
    if chunk_counts.shape[0] != len(patient_ids):
        raise RuntimeError("Chunk-count row count does not match patient count.")

    return (
        embedding_matrix,
        token_lengths,
        chunk_counts,
        was_chunked,
        was_truncated,
    )


def save_per_patient_embeddings(
    root,
    patient_ids,
    embeddings,
    source_name,
    encoder_name,
    condition,
):
    root.mkdir(parents=True, exist_ok=True)
    for patient_id, embedding in zip(patient_ids, embeddings):
        filename = (
            f"{stable_slug(patient_id)}_{stable_slug(source_name)}_"
            f"{stable_slug(encoder_name)}_{stable_slug(condition)}.npy"
        )
        np.save(root / filename, embedding)


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    csv_path = Path(args.csv_path).resolve()
    dataframe = pd.read_csv(
        csv_path,
        dtype={args.patient_id_col: "string"},
    )
    require_columns(dataframe, [args.patient_id_col], "patient IDs")

    patient_ids = dataframe[args.patient_id_col]
    if patient_ids.isna().any() or patient_ids.str.strip().eq("").any():
        raise ValueError("Patient IDs contain missing or empty values.")
    patient_ids = patient_ids.astype(str)
    if not patient_ids.is_unique:
        raise ValueError("Patient IDs must be unique.")

    text_specs = build_text_specs(
        dataframe,
        risk_prefixes=args.risk_prefix,
        text_specs=args.text_spec,
    )

    input_file_sha256 = sha256_file(csv_path)
    patient_order_sha256 = aggregate_hash(patient_ids.tolist())
    output_root = Path(args.output_root).resolve()
    source_dir = output_root / stable_slug(args.source_name)
    index_records = []

    print(f"Loaded {len(dataframe):,} patients from {csv_path}")
    print(f"Source: {args.source_name}")
    print(f"Conditions: {[spec['condition'] for spec in text_specs]}")
    print(f"Frozen encoders: {args.encoders}")
    print(f"Device: {args.device}")

    for encoder_name in args.encoders:
        checkpoint = ENCODER_MODELS[encoder_name]
        requested_revision = args.encoder_revisions[encoder_name]
        print(f"\nLoading frozen {encoder_name}: {checkpoint}")

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=requested_revision,
            trust_remote_code=False,
        )
        model = AutoModel.from_pretrained(
            checkpoint,
            revision=requested_revision,
            trust_remote_code=False,
        )
        model.requires_grad_(False)
        model.to(args.device)
        model.eval()

        resolved_commit = getattr(model.config, "_commit_hash", None)
        if resolved_commit != requested_revision:
            raise RuntimeError(
                f"{encoder_name} resolved to {resolved_commit!r}, not pinned revision "
                f"{requested_revision!r}."
            )

        for spec in text_specs:
            condition = spec["condition"]
            texts = spec["texts"].tolist()
            text_hashes = [sha256_text(text) for text in texts]
            text_collection_sha256 = aggregate_hash(text_hashes)

            condition_dir = (
                source_dir
                / stable_slug(encoder_name)
                / stable_slug(condition)
            )
            embedding_path = condition_dir / "embeddings.npz"
            manifest_path = condition_dir / "manifest.json"

            cache_signature = {
                "input_file_sha256": input_file_sha256,
                "patient_order_sha256": patient_order_sha256,
                "text_collection_sha256": text_collection_sha256,
                "condition": condition,
                "source_columns": spec["source_columns"],
                "construction": spec["construction"],
                "endpoint_scope": spec["endpoint_scope"],
                "encoder_checkpoint": checkpoint,
                "requested_revision": requested_revision,
                "resolved_commit": resolved_commit,
                "pooling": args.pooling,
                "max_length": args.max_length,
                "long_text_strategy": args.long_text_strategy,
                "chunk_stride": args.chunk_stride,
            }

            if (
                not args.overwrite
                and artifact_matches(
                    manifest_path, embedding_path, cache_signature
                )
            ):
                print(
                    f"Skipping cached {encoder_name} / {condition}: "
                    f"{embedding_path}"
                )
                with open(manifest_path, encoding="utf-8") as handle:
                    manifest = json.load(handle)
                index_records.append(manifest)
                continue

            print(f"Embedding {encoder_name} / {condition}")
            (
                embedding_matrix,
                token_lengths,
                chunk_counts,
                was_chunked,
                was_truncated,
            ) = embed_condition(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                patient_ids=patient_ids,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                long_text_strategy=args.long_text_strategy,
                chunk_stride=args.chunk_stride,
            )

            atomic_save_npz(
                embedding_path,
                patient_ids=patient_ids.to_numpy(dtype=str),
                embeddings=embedding_matrix,
                text_sha256=np.asarray(text_hashes, dtype=str),
                token_lengths=token_lengths,
                chunk_counts=chunk_counts,
                was_chunked=was_chunked,
                was_truncated=was_truncated,
            )

            if args.save_per_patient:
                save_per_patient_embeddings(
                    root=condition_dir / "per_patient",
                    patient_ids=patient_ids.tolist(),
                    embeddings=embedding_matrix,
                    source_name=args.source_name,
                    encoder_name=encoder_name,
                    condition=condition,
                )

            manifest = {
                "artifact_type": "frozen_text_embeddings",
                "source_name": args.source_name,
                "source_csv": str(csv_path),
                "condition": condition,
                "construction": spec["construction"],
                "endpoint_scope": spec["endpoint_scope"],
                "source_columns": spec["source_columns"],
                "patient_id_column": args.patient_id_col,
                "patient_count": int(len(patient_ids)),
                "embedding_shape": list(embedding_matrix.shape),
                "embedding_dtype": str(embedding_matrix.dtype),
                "encoder_name": encoder_name,
                "encoder_checkpoint": checkpoint,
                "requested_revision": requested_revision,
                "resolved_commit": resolved_commit,
                "encoder_frozen": True,
                "pooling": args.pooling,
                "max_length": int(args.max_length),
                "long_text_strategy": args.long_text_strategy,
                "chunk_stride": int(args.chunk_stride),
                "chunked_patient_count": int(was_chunked.sum()),
                "maximum_chunks_per_patient": int(chunk_counts.max()),
                "total_encoder_chunks": int(chunk_counts.sum()),
                "truncated_patient_count": int(was_truncated.sum()),
                "maximum_untruncated_token_length": int(token_lengths.max()),
                "batch_size": int(args.batch_size),
                "seed": int(args.seed),
                "device": args.device,
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if args.device.startswith("cuda")
                    else None
                ),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "numpy_version": np.__version__,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "embedding_file": str(embedding_path),
                "per_patient_files_saved": bool(args.save_per_patient),
                "cache_signature": cache_signature,
            }
            atomic_save_json(manifest_path, manifest)
            index_records.append(manifest)

            print(
                f"Saved {embedding_matrix.shape} to {embedding_path}; "
                f"chunked={int(was_chunked.sum())}; "
                f"truncated={int(was_truncated.sum())}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    index_path = source_dir / "embedding_index.csv"
    index_dataframe = pd.DataFrame(
        [
            {
                "source_name": record["source_name"],
                "encoder_name": record["encoder_name"],
                "encoder_checkpoint": record["encoder_checkpoint"],
                "resolved_commit": record["resolved_commit"],
                "condition": record["condition"],
                "patient_count": record["patient_count"],
                "embedding_shape": str(record["embedding_shape"]),
                "pooling": record["pooling"],
                "max_length": record["max_length"],
                "long_text_strategy": record[
                    "long_text_strategy"
                ],
                "chunk_stride": record["chunk_stride"],
                "chunked_patient_count": record[
                    "chunked_patient_count"
                ],
                "maximum_chunks_per_patient": record[
                    "maximum_chunks_per_patient"
                ],
                "truncated_patient_count": record[
                    "truncated_patient_count"
                ],
                "embedding_file": record["embedding_file"],
            }
            for record in index_records
        ]
    )
    index_dataframe.to_csv(index_path, index=False)
    print(f"\nEmbedding index saved to {index_path}")
    print("All requested frozen embeddings are complete.")


if __name__ == "__main__":
    main()
