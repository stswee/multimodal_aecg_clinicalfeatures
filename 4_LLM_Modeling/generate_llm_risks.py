#!/usr/bin/env python3

"""Generate four-year SCD/PFD LLM responses from paired chat messages.

For each condition, the input CSV contains a system-message column and a
patient-specific user-message column. The script passes these messages to the
tokenizer with their correct chat roles, parses the structured response fields,
and checkpoints progress so interrupted runs can resume safely.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


DEFAULT_PROMPT_COLUMNS = [
    "Full_Risk_No_ECG_Prompt",
    "Neutral_Summary_No_ECG_Prompt",
    "Full_Risk_With_ECG_Prompt",
]

PROMPT_SLUGS = {
    "Full_Risk_No_ECG_Prompt": "full_risk_no_ecg",
    "Neutral_Summary_No_ECG_Prompt": "neutral_summary_no_ecg",
    "Full_Risk_With_ECG_Prompt": "full_risk_with_ecg",
}

SYSTEM_MESSAGE_COLUMNS = {
    "Full_Risk_No_ECG_Prompt": "Full_Risk_No_ECG_System_Message",
    "Neutral_Summary_No_ECG_Prompt": (
        "Neutral_Summary_No_ECG_System_Message"
    ),
    "Full_Risk_With_ECG_Prompt": "Full_Risk_With_ECG_System_Message",
}

STRUCTURED_FIELDS = (
    "SCD_RISK",
    "SCD_RATIONALE",
    "PFD_RISK",
    "PFD_RATIONALE",
    "CLINICAL_SUMMARY",
)

ALLOWED_RISKS = {"low", "moderate", "high"}


def prompt_slug(prompt_column):
    """Return a stable output-column prefix for a prompt column."""
    if prompt_column in PROMPT_SLUGS:
        return PROMPT_SLUGS[prompt_column]

    slug = re.sub(r"[^a-z0-9]+", "_", prompt_column.lower()).strip("_")
    return slug.removesuffix("_prompt")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_chat_pair(system_message, user_message):
    """Hash both chat messages with an unambiguous serialization."""
    payload = json.dumps(
        {
            "system_message": system_message,
            "user_message": user_message,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 checksum of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_structured_field(text, field):
    """Extract one field, allowing its value to wrap across lines."""
    field_alternatives = "|".join(re.escape(name) for name in STRUCTURED_FIELDS)
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(field)}\s*:\s*(.*?)"
        rf"(?=^\s*(?:{field_alternatives})\s*:|\Z)"
    )
    match = pattern.search(text or "")
    return match.group(1).strip() if match else None


def normalize_risk(value):
    if not value:
        return None

    match = re.search(r"\b(low|moderate|high)\b", value.lower())
    if not match:
        return None

    risk = match.group(1)
    return risk if risk in ALLOWED_RISKS else None


def parse_response(prompt_column, raw_response):
    """Parse one response and construct the requested ablation texts."""
    slug = prompt_slug(prompt_column)
    parsed = {
        f"{slug}_generation_status": "ok" if raw_response else "empty_output"
    }

    if "full_risk" in slug:
        scd_risk = normalize_risk(
            extract_structured_field(raw_response, "SCD_RISK")
        )
        pfd_risk = normalize_risk(
            extract_structured_field(raw_response, "PFD_RISK")
        )
        scd_rationale = extract_structured_field(
            raw_response, "SCD_RATIONALE"
        )
        pfd_rationale = extract_structured_field(
            raw_response, "PFD_RATIONALE"
        )

        parsed.update(
            {
                f"{slug}_scd_risk": scd_risk,
                f"{slug}_pfd_risk": pfd_risk,
                f"{slug}_scd_rationale": scd_rationale,
                f"{slug}_pfd_rationale": pfd_rationale,
                f"{slug}_label_only_response": (
                    f"SCD_RISK: {scd_risk.title()}\n"
                    f"PFD_RISK: {pfd_risk.title()}"
                    if scd_risk and pfd_risk
                    else None
                ),
                f"{slug}_rationale_only_response": (
                    f"SCD_RATIONALE: {scd_rationale}\n"
                    f"PFD_RATIONALE: {pfd_rationale}"
                    if scd_rationale and pfd_rationale
                    else None
                ),
            }
        )

        if raw_response and not all(
            [scd_risk, pfd_risk, scd_rationale, pfd_rationale]
        ):
            parsed[f"{slug}_generation_status"] = "format_error"

    elif "neutral_summary" in slug:
        summary = extract_structured_field(raw_response, "CLINICAL_SUMMARY")
        parsed[f"{slug}_clinical_summary"] = summary
        if raw_response and not summary:
            parsed[f"{slug}_generation_status"] = "format_error"

    return parsed


def build_chat_text(tokenizer, system_message, user_message):
    """Apply the model chat template to the paired system and user messages."""
    messages = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    model,
    tokenizer,
    system_messages,
    user_messages,
    max_new_tokens,
    repetition_penalty,
):
    chat_texts = [
        build_chat_text(tokenizer, system_message, user_message)
        for system_message, user_message in zip(
            system_messages,
            user_messages,
        )
    ]
    model_inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    input_device = next(model.parameters()).device
    model_inputs = {
        key: value.to(input_device) for key, value in model_inputs.items()
    }
    input_width = model_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for output in outputs:
        generated_ids = output[input_width:]
        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        responses.append(response)

    return responses


def generate_with_retries(
    model,
    tokenizer,
    system_messages,
    user_messages,
    max_new_tokens,
    repetition_penalty,
    max_retries,
):
    """Retry only chat pairs that produced an empty response."""
    if len(system_messages) != len(user_messages):
        raise ValueError(
            "system_messages and user_messages must have equal lengths."
        )

    responses = [""] * len(user_messages)
    attempts = [0] * len(user_messages)
    pending = list(range(len(user_messages)))

    while pending:
        current_indices = pending
        current_system_messages = [
            system_messages[index] for index in current_indices
        ]
        current_user_messages = [
            user_messages[index] for index in current_indices
        ]

        for index in current_indices:
            attempts[index] += 1

        generated = generate_batch(
            model=model,
            tokenizer=tokenizer,
            system_messages=current_system_messages,
            user_messages=current_user_messages,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

        next_pending = []
        for index, response in zip(current_indices, generated):
            if response:
                responses[index] = response
            elif attempts[index] < max_retries:
                next_pending.append(index)

        pending = next_pending
        if pending:
            time.sleep(0.2)

    return responses, attempts


def atomic_save_csv(dataframe, output_path):
    """Replace the checkpoint only after the new CSV is written successfully."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix=f".{output_path.stem}.",
        dir=output_path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary_path = Path(handle.name)
        dataframe.to_csv(handle, index=False, na_rep="")

    os.replace(temporary_path, output_path)


def atomic_save_json(payload, output_path):
    """Atomically write a JSON manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f".{output_path.stem}.",
        dir=output_path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    os.replace(temporary_path, output_path)


def load_results(input_df, output_path, overwrite):
    """Initialize results or resume a prior output file."""
    patient_ids = input_df[["Patient ID"]].copy()
    patient_ids["Patient ID"] = patient_ids["Patient ID"].astype("string")

    if overwrite or not output_path.exists():
        return patient_ids

    existing = pd.read_csv(output_path, dtype={"Patient ID": "string"})
    if "Patient ID" not in existing.columns or not existing["Patient ID"].is_unique:
        raise ValueError("Existing output must contain unique 'Patient ID' values.")

    unexpected_ids = set(existing["Patient ID"]) - set(patient_ids["Patient ID"])
    if unexpected_ids:
        raise ValueError("Existing output contains Patient IDs absent from the input CSV.")

    return patient_ids.merge(
        existing,
        on="Patient ID",
        how="left",
        validate="one_to_one",
    )


def main(args):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError(
            "--revision must be an immutable 40-character Hugging Face "
            "commit hash; aliases such as 'main' are not allowed."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    manifest_path = output_path.with_suffix(".manifest.json")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    prompt_file_sha256 = sha256_file(input_path)
    if prompt_file_sha256.lower() != args.expected_prompt_sha256.lower():
        raise ValueError(
            "Prompt CSV checksum mismatch. "
            f"Expected {args.expected_prompt_sha256}, observed "
            f"{prompt_file_sha256}."
        )

    input_df = pd.read_csv(input_path, dtype={"Patient ID": "string"})
    if "Patient ID" not in input_df.columns:
        raise ValueError("Input CSV must contain a 'Patient ID' column.")
    if not input_df["Patient ID"].is_unique:
        raise ValueError("Input CSV contains duplicate Patient IDs.")
    if input_df["Patient ID"].isna().any():
        raise ValueError("Input CSV contains missing Patient IDs.")
    if len(input_df) != args.expected_patients:
        raise ValueError(
            f"Expected {args.expected_patients} patients, observed "
            f"{len(input_df)}."
        )

    unknown_prompt_columns = [
        column
        for column in args.prompt_columns
        if column not in SYSTEM_MESSAGE_COLUMNS
    ]
    if unknown_prompt_columns:
        raise ValueError(
            "No system-message mapping is defined for prompt columns: "
            f"{unknown_prompt_columns}"
        )

    system_message_columns = [
        SYSTEM_MESSAGE_COLUMNS[column] for column in args.prompt_columns
    ]
    required_message_columns = list(args.prompt_columns) + system_message_columns

    missing_columns = [
        column
        for column in required_message_columns
        if column not in input_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required chat-message columns: "
            f"{missing_columns}"
        )

    empty_message_counts = {
        column: int(
            input_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
        for column in required_message_columns
    }
    if any(empty_message_counts.values()):
        raise ValueError(
            "Input CSV contains empty chat messages: "
            f"{empty_message_counts}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    resolved_model_commit = getattr(model.config, "_commit_hash", None)
    resolved_tokenizer_commit = tokenizer.init_kwargs.get("_commit_hash")
    if resolved_model_commit and resolved_model_commit.lower() != args.revision.lower():
        raise RuntimeError(
            "Resolved model commit does not match --revision: "
            f"{resolved_model_commit} != {args.revision}"
        )

    generation_started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_status": "running",
        "run_label": args.run_label,
        "generation_started_at_utc": generation_started_at,
        "input_csv": str(input_path.resolve()),
        "input_csv_sha256": prompt_file_sha256,
        "patient_count": int(len(input_df)),
        "prompt_columns": list(args.prompt_columns),
        "system_message_columns": {
            column: SYSTEM_MESSAGE_COLUMNS[column]
            for column in args.prompt_columns
        },
        "chat_roles": ["system", "user"],
        "model_id": args.model,
        "requested_model_revision": args.revision,
        "resolved_model_commit": resolved_model_commit,
        "resolved_tokenizer_commit": resolved_tokenizer_commit,
        "torch_dtype": "float16",
        "quantization": "none",
        "device_map_strategy": args.device_map,
        "visible_gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "generation": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "sampling_note": (
                "Temperature, top-p, and top-k are not applicable because "
                "greedy decoding is used."
            ),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "max_empty_output_retries": args.max_retries,
            "batch_size": args.batch_size,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "output_csv": str(output_path.resolve()),
    }
    atomic_save_json(manifest, manifest_path)

    results = load_results(input_df, output_path, args.overwrite)
    results["model_id"] = args.model
    results["model_revision"] = args.revision
    results["resolved_model_commit"] = resolved_model_commit
    results["resolved_tokenizer_commit"] = resolved_tokenizer_commit
    results["prompt_csv_sha256"] = prompt_file_sha256
    results["run_label"] = args.run_label
    results["transformers_version"] = transformers.__version__
    results["torch_version"] = torch.__version__
    results["cuda_version"] = torch.version.cuda
    results["visible_gpu_names"] = "; ".join(
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    )
    results["device_map_strategy"] = args.device_map
    results["generation_seed"] = args.seed
    results["max_new_tokens"] = args.max_new_tokens
    results["repetition_penalty"] = args.repetition_penalty
    results["do_sample"] = False
    results["temperature"] = "not_applicable_greedy_decoding"
    results["top_p"] = "not_applicable_greedy_decoding"
    results["top_k"] = "not_applicable_greedy_decoding"

    completed_since_save = 0

    for prompt_column in args.prompt_columns:
        slug = prompt_slug(prompt_column)
        system_message_column = SYSTEM_MESSAGE_COLUMNS[prompt_column]
        response_column = f"{slug}_raw_response"
        attempts_column = f"{slug}_generation_attempts"
        timestamp_column = f"{slug}_generated_at_utc"
        system_hash_column = f"{slug}_system_message_sha256"
        user_hash_column = f"{slug}_user_message_sha256"
        chat_hash_column = f"{slug}_chat_messages_sha256"

        for column in (response_column, attempts_column, timestamp_column):
            if column not in results.columns:
                results[column] = pd.NA

        current_system_hashes = input_df[system_message_column].map(sha256_text)
        current_user_hashes = input_df[prompt_column].map(sha256_text)
        current_chat_hashes = pd.Series(
            [
                sha256_chat_pair(system_message, user_message)
                for system_message, user_message in zip(
                    input_df[system_message_column].astype(str),
                    input_df[prompt_column].astype(str),
                )
            ],
            index=input_df.index,
        )

        # Resume only when the response and the exact system/user message pair
        # match. A change to either message forces regeneration.
        if chat_hash_column in results.columns:
            matching_hash = (
                results[chat_hash_column]
                .fillna("")
                .astype(str)
                .eq(current_chat_hashes)
            )
        else:
            matching_hash = pd.Series(False, index=results.index)

        results[system_hash_column] = current_system_hashes
        results[user_hash_column] = current_user_hashes
        results[chat_hash_column] = current_chat_hashes

        # Parse any completed responses loaded from a checkpoint.
        existing_mask = (
            results[response_column].fillna("").astype(str).str.len().gt(0)
            & matching_hash
        )
        for index in results.index[existing_mask]:
            parsed = parse_response(prompt_column, results.at[index, response_column])
            for column, value in parsed.items():
                results.at[index, column] = value

        pending_indices = results.index[~existing_mask].tolist()
        if not pending_indices:
            print(f"{prompt_column}: already complete; skipping.")
            continue

        # Prevent stale parsed fields from surviving when a changed prompt is
        # regenerated in a resumed output file.
        stale_columns = [
            column
            for column in results.columns
            if column.startswith(f"{slug}_")
            and column
            not in {
                response_column,
                attempts_column,
                timestamp_column,
                system_hash_column,
                user_hash_column,
                chat_hash_column,
            }
        ]
        if stale_columns:
            results.loc[pending_indices, stale_columns] = pd.NA

        progress = tqdm(
            total=len(pending_indices),
            desc=f"{args.model} | {slug}",
        )

        for start in range(0, len(pending_indices), args.batch_size):
            batch_indices = pending_indices[start : start + args.batch_size]
            system_messages = (
                input_df.loc[batch_indices, system_message_column]
                .astype(str)
                .tolist()
            )
            user_messages = (
                input_df.loc[batch_indices, prompt_column]
                .astype(str)
                .tolist()
            )

            responses, attempts = generate_with_retries(
                model=model,
                tokenizer=tokenizer,
                system_messages=system_messages,
                user_messages=user_messages,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                max_retries=args.max_retries,
            )

            generated_at = datetime.now(timezone.utc).isoformat()
            for index, response, attempt_count in zip(
                batch_indices, responses, attempts
            ):
                results.at[index, response_column] = response
                results.at[index, attempts_column] = attempt_count
                results.at[index, timestamp_column] = generated_at

                parsed = parse_response(prompt_column, response)
                for column, value in parsed.items():
                    results.at[index, column] = value

            completed_since_save += len(batch_indices)
            progress.update(len(batch_indices))

            if completed_since_save >= args.save_every:
                atomic_save_csv(results, output_path)
                completed_since_save = 0

        progress.close()
        atomic_save_csv(results, output_path)
        completed_since_save = 0

    atomic_save_csv(results, output_path)

    status_counts = {}
    for prompt_column in args.prompt_columns:
        slug = prompt_slug(prompt_column)
        status_column = f"{slug}_generation_status"
        if status_column in results.columns:
            status_counts[status_column] = {
                str(key): int(value)
                for key, value in results[status_column]
                .value_counts(dropna=False)
                .items()
            }

    manifest["run_status"] = "complete"
    manifest["generation_finished_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    manifest["generation_status_counts"] = status_counts
    manifest["output_csv_sha256"] = sha256_file(output_path)
    atomic_save_json(manifest, manifest_path)

    print(f"\nSaved results to {output_path}")
    for prompt_column in args.prompt_columns:
        slug = prompt_slug(prompt_column)
        status_column = f"{slug}_generation_status"
        if status_column in results.columns:
            print(f"\n{status_column}:")
            print(results[status_column].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable 40-character Hugging Face commit hash.",
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--expected_prompt_sha256",
        required=True,
        help="Required SHA-256 checksum of the complete input prompt CSV.",
    )
    parser.add_argument("--expected_patients", type=int, default=730)
    parser.add_argument("--run_label", default="four_year_v2")
    parser.add_argument(
        "--prompt_columns",
        nargs="+",
        default=DEFAULT_PROMPT_COLUMNS,
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--device_map",
        default="balanced",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help=(
            "Maximum generated tokens per response. This is a safety ceiling, "
            "not a requested response length."
        ),
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing output file and start from the first patient.",
    )

    main(parser.parse_args())
