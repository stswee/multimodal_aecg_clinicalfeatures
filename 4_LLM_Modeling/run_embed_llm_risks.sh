#!/usr/bin/env bash

# Generate reusable, outcome-independent text representations for the revised
# four-year analysis. This launcher addresses the reviewer's representation
# comparisons by embedding, for both LLaMA checkpoints:
#   1. endpoint-specific SCD full/label/rationale text;
#   2. endpoint-specific PFD full/label/rationale text;
#   3. one joint SCD+PFD full-response ablation;
#   4. neutral, non-reasoning summary; and
#   5. the same endpoint-specific conditions with ECG impressions.
# It also embeds the deterministic, non-LLM patient-data template as a separate
# comparison arm. BioBERT and ClinicalBERT are used off the shelf, kept frozen,
# and applied with prespecified CLS pooling. No outcomes, folds, imputation,
# model selection, or performance results enter this feature-extraction stage.
# Encoder/model selection must occur later within the inner cross-validation
# loop. Responses exceeding the 512-token encoder window are divided into
# overlapping chunks, and their chunk-level CLS embeddings are averaged. Each
# artifact includes patient IDs, text hashes, token and chunk counts, the
# resolved encoder revision, software versions, and an audit manifest. Existing
# exactly matching artifacts are reused automatically.

set -euo pipefail

# ---------------- USER CONFIG ----------------

SESSION_NAME="embed_4year_text_v4_detailed"
CONDA_ENV="shdb-af-analysis"

GPU_ID_8B=0
GPU_ID_3B=1
GPU_ID_TEMPLATE=2

BATCH_SIZE=32
MAX_LENGTH=512
LONG_TEXT_STRATEGY="mean_chunks"
CHUNK_STRIDE=64
SEED=42
EXPECTED_PATIENTS=730

# Source LLM identifiers and immutable revisions.
MODEL_8B="meta-llama/Llama-3.1-8B-Instruct"
MODEL_3B="meta-llama/Llama-3.2-3B-Instruct"

REVISION_8B="0e9e39f249a16976918f6564b8830bc894c89659"
REVISION_3B="0cb88a4f764b7a12671c53f0838cd831a0843b95"

# These are the frozen text encoders, not the source LLMs.
BIOBERT_MODEL="dmis-lab/biobert-base-cased-v1.1"
CLINICALBERT_MODEL="emilyalsentzer/Bio_ClinicalBERT"

BIOBERT_REVISION="924f12e0c3db7f156a765ad53fb6b11e7afedbc8"
CLINICALBERT_REVISION="d5892b39a4adaed74b92212a44081509db72f87b"

EXPECTED_PROMPT_SHA256="4676a5e8d918b0ba51df136c59c59062b5b2df1efe97b58d34d30d094b35ec79"

# ---------------- PATHS ----------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/embed_llm_risks.py"

MUSIC_DIR="${SCRIPT_DIR}/../../music"

PROMPT_CSV="${MUSIC_DIR}/subject-info-cleaned-4year-with-prompts.csv"

RESPONSE_DIR="${MUSIC_DIR}/llm_responses_4year_v3_detailed"

# The 8B responses required no formatting repair. The 3B responses use the
# separately verified postprocessed file; its raw-response fields are unchanged.
LLAMA8B_CSV="${RESPONSE_DIR}/LLaMA3.1-8B-4year-responses.csv"
LLAMA3B_CSV="${RESPONSE_DIR}/LLaMA3.2-3B-4year-responses-postprocessed.csv"

OUTPUT_ROOT="${MUSIC_DIR}/text_embeddings_4year_v4_detailed"
LOG_DIR="${OUTPUT_ROOT}/logs"
METADATA_DIR="${OUTPUT_ROOT}/run_metadata"

LOG_8B="${LOG_DIR}/LLaMA3.1-8B.log"
LOG_3B="${LOG_DIR}/LLaMA3.2-3B.log"
LOG_TEMPLATE="${LOG_DIR}/DeterministicTemplate.log"

# ---------------- FILE CHECKS ----------------

mkdir -p \
    "${OUTPUT_ROOT}" \
    "${LOG_DIR}" \
    "${METADATA_DIR}"

for required_file in \
    "${PYTHON_SCRIPT}" \
    "${PROMPT_CSV}" \
    "${LLAMA8B_CSV}" \
    "${LLAMA3B_CSV}"; do

    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: Required file not found:" >&2
        echo "${required_file}" >&2
        exit 1
    fi
done

# Confirm that the corrected prompt CSV is being used.
ACTUAL_PROMPT_SHA256="$(
    sha256sum "${PROMPT_CSV}" | awk '{print $1}'
)"

if [[ "${ACTUAL_PROMPT_SHA256}" != "${EXPECTED_PROMPT_SHA256}" ]]; then
    echo "ERROR: Prompt CSV checksum mismatch." >&2
    echo "Expected: ${EXPECTED_PROMPT_SHA256}" >&2
    echo "Observed: ${ACTUAL_PROMPT_SHA256}" >&2
    exit 1
fi

echo "Prompt CSV checksum verified:"
echo "${ACTUAL_PROMPT_SHA256}"

# ---------------- COHORT AND RESPONSE AUDIT ----------------

python - \
    "${LLAMA8B_CSV}" \
    "${LLAMA3B_CSV}" \
    "${PROMPT_CSV}" \
    "${EXPECTED_PATIENTS}" <<'PY'
import sys
import pandas as pd

llama8b_path = sys.argv[1]
llama3b_path = sys.argv[2]
prompt_path = sys.argv[3]
expected_patients = int(sys.argv[4])

paths = {
    "LLaMA3.1-8B": llama8b_path,
    "LLaMA3.2-3B": llama3b_path,
    "Prompt CSV": prompt_path,
}

tables = {
    name: pd.read_csv(
        path,
        dtype={"Patient ID": "string"},
    )
    for name, path in paths.items()
}

for name, table in tables.items():
    if len(table) != expected_patients:
        raise ValueError(
            f"{name}: expected {expected_patients} rows, "
            f"found {len(table)}"
        )

    if "Patient ID" not in table.columns:
        raise ValueError(
            f"{name}: Patient ID column is missing"
        )

    if table["Patient ID"].isna().any():
        raise ValueError(
            f"{name}: missing Patient IDs"
        )

    if not table["Patient ID"].is_unique:
        raise ValueError(
            f"{name}: duplicate Patient IDs"
        )

reference_ids = set(tables["Prompt CSV"]["Patient ID"])

for name in ["LLaMA3.1-8B", "LLaMA3.2-3B"]:
    table = tables[name]
    observed_ids = set(table["Patient ID"])

    if observed_ids != reference_ids:
        missing = sorted(reference_ids - observed_ids)
        unexpected = sorted(observed_ids - reference_ids)

        raise ValueError(
            f"{name}: patient mismatch; "
            f"missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    for prefix in [
        "full_risk_no_ecg",
        "full_risk_with_ecg",
    ]:
        required_columns = [
            f"{prefix}_scd_risk",
            f"{prefix}_scd_rationale",
            f"{prefix}_pfd_risk",
            f"{prefix}_pfd_rationale",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in table.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{name}: missing columns "
                f"{missing_columns}"
            )

        postprocessed_status = f"{prefix}_postprocessed_status"
        generation_status = f"{prefix}_generation_status"

        if postprocessed_status in table.columns:
            complete = table[postprocessed_status].eq("complete")
            status_used = postprocessed_status
        elif generation_status in table.columns:
            complete = table[generation_status].eq("ok")
            status_used = generation_status
        else:
            raise ValueError(
                f"{name}: neither {postprocessed_status} nor "
                f"{generation_status} is present"
            )

        if not complete.all():
            count = int((~complete).sum())
            raise ValueError(
                f"{name}: {count} incomplete {prefix} responses "
                f"according to {status_used}"
            )

        if not table[required_columns].notna().all().all():
            raise ValueError(
                f"{name}: missing parsed fields "
                f"for {prefix}"
            )

        for risk_column in [
            f"{prefix}_scd_risk",
            f"{prefix}_pfd_risk",
        ]:
            normalized = (
                table[risk_column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            valid = normalized.isin(
                ["low", "moderate", "high"]
            )

            if not valid.all():
                invalid = sorted(
                    normalized[~valid].unique()
                )
                raise ValueError(
                    f"{name}: invalid values in "
                    f"{risk_column}: {invalid}"
                )

    neutral_status = (
        "neutral_summary_no_ecg_generation_status"
    )
    neutral_text = (
        "neutral_summary_no_ecg_clinical_summary"
    )

    if not table[neutral_status].eq("ok").all():
        raise ValueError(
            f"{name}: neutral summaries are incomplete"
        )

    if (
        table[neutral_text].isna().any()
        or table[neutral_text]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            f"{name}: missing or empty neutral summaries"
        )

# Verify the exact deterministic format repairs applied to the 3B file.
expected_3b_repairs = {
    "full_risk_no_ecg": 52,
    "full_risk_with_ecg": 31,
}

table_3b = tables["LLaMA3.2-3B"]
for prefix, expected_count in expected_3b_repairs.items():
    repair_column = f"{prefix}_format_repair_applied"
    reason_column = f"{prefix}_format_repair_reason"

    for column in [repair_column, reason_column]:
        if column not in table_3b.columns:
            raise ValueError(
                f"LLaMA3.2-3B: missing repair audit column {column}"
            )

    repaired = (
        table_3b[repair_column]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    if int(repaired.sum()) != expected_count:
        raise ValueError(
            f"LLaMA3.2-3B: expected {expected_count} repairs for "
            f"{prefix}, observed {int(repaired.sum())}"
        )

    expected_reason = (
        "Second PFD_RISK field interpreted as PFD_RATIONALE"
    )
    if not table_3b.loc[repaired, reason_column].eq(expected_reason).all():
        raise ValueError(
            f"LLaMA3.2-3B: unexpected repair reason for {prefix}"
        )

template_column = "Patient_Data_Template_No_ECG"
prompt_table = tables["Prompt CSV"]

if template_column not in prompt_table.columns:
    raise ValueError(
        f"Prompt CSV is missing {template_column}"
    )

if (
    prompt_table[template_column].isna().any()
    or prompt_table[template_column]
    .astype(str)
    .str.strip()
    .eq("")
    .any()
):
    raise ValueError(
        "Deterministic templates contain missing text"
    )

print(
    "Preflight audit passed: "
    f"{expected_patients} aligned patients."
)
PY

# ---------------- TMUX CHECK ----------------

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION_NAME}' already exists." >&2
    echo "Attach with:" >&2
    echo "tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

# ---------------- REPRODUCIBILITY METADATA ----------------

{
    echo "launch_time_utc=$(date -u --iso-8601=seconds)"
    echo "hostname=$(hostname)"

    echo "python_script=$(readlink -f "${PYTHON_SCRIPT}")"
    echo "python_script_sha256=$(sha256sum "${PYTHON_SCRIPT}" | awk '{print $1}')"

    echo "launcher_script=$(readlink -f "${BASH_SOURCE[0]}")"
    echo "launcher_script_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"

    echo "prompt_csv=$(readlink -f "${PROMPT_CSV}")"
    echo "prompt_csv_sha256=${ACTUAL_PROMPT_SHA256}"

    echo "llama8b_csv=$(readlink -f "${LLAMA8B_CSV}")"
    echo "llama8b_csv_sha256=$(sha256sum "${LLAMA8B_CSV}" | awk '{print $1}')"

    echo "llama3b_csv=$(readlink -f "${LLAMA3B_CSV}")"
    echo "llama3b_csv_sha256=$(sha256sum "${LLAMA3B_CSV}" | awk '{print $1}')"

    echo "source_model_8b=${MODEL_8B}"
    echo "source_revision_8b=${REVISION_8B}"

    echo "source_model_3b=${MODEL_3B}"
    echo "source_revision_3b=${REVISION_3B}"

    echo "biobert_model=${BIOBERT_MODEL}"
    echo "clinicalbert_model=${CLINICALBERT_MODEL}"
    echo "biobert_revision=${BIOBERT_REVISION}"
    echo "clinicalbert_revision=${CLINICALBERT_REVISION}"

    echo "expected_patients=${EXPECTED_PATIENTS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "max_length=${MAX_LENGTH}"
    echo "long_text_strategy=${LONG_TEXT_STRATEGY}"
    echo "chunk_stride=${CHUNK_STRIDE}"
    echo "seed=${SEED}"

    echo "gpu_id_8b=${GPU_ID_8B}"
    echo "gpu_id_3b=${GPU_ID_3B}"
    echo "gpu_id_template=${GPU_ID_TEMPLATE}"
} > "${METADATA_DIR}/launcher_configuration.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=index,name,driver_version,memory.total \
        --format=csv,noheader \
        > "${METADATA_DIR}/gpu_inventory.csv"
fi

# ---------------- EMBEDDING ARGUMENTS ----------------

COMMON_ARGS=(
    --patient_id_col "Patient ID"
    --output_root "${OUTPUT_ROOT}"
    --encoders BioBERT ClinicalBERT
    --encoder_revision "BioBERT=${BIOBERT_REVISION}"
    --encoder_revision "ClinicalBERT=${CLINICALBERT_REVISION}"
    --pooling cls
    --batch_size "${BATCH_SIZE}"
    --max_length "${MAX_LENGTH}"
    --long_text_strategy "${LONG_TEXT_STRATEGY}"
    --chunk_stride "${CHUNK_STRIDE}"
    --seed "${SEED}"
    --device cuda
)

make_command() {
    local gpu_id="$1"
    local log_file="$2"
    local job_name="$3"

    shift 3

    local quoted_bashrc
    local quoted_env
    local quoted_log
    local python_command

    printf -v quoted_bashrc '%q' "${HOME}/.bashrc"
    printf -v quoted_env '%q' "${CONDA_ENV}"
    printf -v quoted_log '%q' "${log_file}"

    printf -v python_command '%q ' \
        python \
        "${PYTHON_SCRIPT}" \
        "$@"

    printf \
        'source %s && conda activate %s && export CUDA_VISIBLE_DEVICES=%q TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=%q && { set -o pipefail; %s2>&1 | tee -a %s; status=${PIPESTATUS[0]}; echo; echo "%s exit status: ${status}"; echo "Finished at: $(date --iso-8601=seconds)"; }; exec bash' \
        "${quoted_bashrc}" \
        "${quoted_env}" \
        "${gpu_id}" \
        "${SEED}" \
        "${python_command}" \
        "${quoted_log}" \
        "${job_name}"
}

LLAMA8B_COMMAND="$(
    make_command \
        "${GPU_ID_8B}" \
        "${LOG_8B}" \
        "LLaMA 3.1 8B embedding" \
        --csv_path "${LLAMA8B_CSV}" \
        --source_name "LLaMA3.1-8B" \
        --risk_prefix full_risk_no_ecg \
        --risk_prefix full_risk_with_ecg \
        --text_spec \
            neutral_summary_no_ecg \
            neutral_summary_no_ecg_clinical_summary \
        "${COMMON_ARGS[@]}"
)"

LLAMA3B_COMMAND="$(
    make_command \
        "${GPU_ID_3B}" \
        "${LOG_3B}" \
        "LLaMA 3.2 3B embedding" \
        --csv_path "${LLAMA3B_CSV}" \
        --source_name "LLaMA3.2-3B" \
        --risk_prefix full_risk_no_ecg \
        --risk_prefix full_risk_with_ecg \
        --text_spec \
            neutral_summary_no_ecg \
            neutral_summary_no_ecg_clinical_summary \
        "${COMMON_ARGS[@]}"
)"

TEMPLATE_COMMAND="$(
    make_command \
        "${GPU_ID_TEMPLATE}" \
        "${LOG_TEMPLATE}" \
        "Deterministic-template embedding" \
        --csv_path "${PROMPT_CSV}" \
        --source_name "DeterministicTemplate" \
        --text_spec \
            patient_data_template_no_ecg \
            Patient_Data_Template_No_ECG \
        "${COMMON_ARGS[@]}"
)"

# ---------------- LAUNCH JOBS ----------------

tmux new-session \
    -d \
    -s "${SESSION_NAME}" \
    -n "llama8b"

tmux send-keys \
    -t "${SESSION_NAME}:llama8b" \
    "${LLAMA8B_COMMAND}" \
    C-m

tmux new-window \
    -t "${SESSION_NAME}" \
    -n "llama3b"

tmux send-keys \
    -t "${SESSION_NAME}:llama3b" \
    "${LLAMA3B_COMMAND}" \
    C-m

tmux new-window \
    -t "${SESSION_NAME}" \
    -n "template"

tmux send-keys \
    -t "${SESSION_NAME}:template" \
    "${TEMPLATE_COMMAND}" \
    C-m

tmux select-window \
    -t "${SESSION_NAME}:llama8b"

echo
echo "Embedding jobs launched successfully."
echo "Session: ${SESSION_NAME}"
echo "GPU ${GPU_ID_8B}: LLaMA 3.1 8B conditions"
echo "GPU ${GPU_ID_3B}: LLaMA 3.2 3B conditions"
echo "GPU ${GPU_ID_TEMPLATE}: deterministic template"
echo "Output: ${OUTPUT_ROOT}"
echo
echo "Attach:"
echo "tmux attach -t ${SESSION_NAME}"
echo
echo "List windows:"
echo "tmux list-windows -t ${SESSION_NAME}"
echo
echo "Follow logs:"
echo "tail -f ${LOG_8B}"
echo "tail -f ${LOG_3B}"
echo "tail -f ${LOG_TEMPLATE}"

tmux attach -t "${SESSION_NAME}"
