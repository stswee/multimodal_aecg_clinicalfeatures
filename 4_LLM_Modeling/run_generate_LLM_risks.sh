#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LLM GENERATION REPRODUCIBILITY RECORD
# ============================================================
#
# Study task:
#   Qualitative prediction of sudden cardiac death (SCD) and
#   pump failure death (PFD) within four years of the baseline
#   24-hour Holter recording.
#
# Analysis cohort:
#   - 730 patients with an ascertainable four-year outcome
#   - 577 without cardiac death by four years
#   - 71 with SCD by four years
#   - 82 with PFD by four years
#
# Input file:
#   subject-info-cleaned-4year-with-prompts.csv
#
# Prompt conditions:
#   1. Detailed four-year risk assessment without ECG impressions
#   2. Neutral, non-reasoning summary without ECG impressions
#   3. Detailed four-year risk assessment with ECG impressions
#
# Chat-message structure:
#   - Each condition has a dedicated system-message column.
#   - Each patient has a corresponding user-message column.
#   - The Python generator passes these using their actual system
#     and user roles in the model chat template.
#   - No word-count or sentence-count instruction is imposed.
#
# Derived conditions requiring no additional LLM generation:
#   - Risk-label-only response
#   - Rationale-only response
#   - Deterministic patient-data template
#   - Shuffled response analysis, performed later within each
#     outer test fold
#
# Models:
#   - meta-llama/Llama-3.1-8B-Instruct
#   - meta-llama/Llama-3.2-3B-Instruct
#
# Model revisions:
#   Immutable Hugging Face commit hashes are specified separately
#   for the 8B and 3B models below.
#   The resolved model commit is saved in each output CSV.
#
# Parallel GPU allocation:
#   - LLaMA 3.1 8B: GPUs 0,1,2,3
#   - LLaMA 3.2 3B: GPUs 4,5,6,7
#   - Device placement: balanced across visible GPUs
#
# Generation settings:
#   - Deterministic greedy decoding
#   - do_sample=False
#   - random seed=42
#   - max_new_tokens=1024 (truncation safeguard only)
#   - repetition_penalty=1.1
#   - maximum retries=3
#   - batch size=4
#
# Checkpointing:
#   - Results saved every 20 responses
#   - Existing outputs resumed using Patient ID and the combined
#     system-message/user-message hash
#   - A change to either message forces regeneration
#
# Response fields retained:
#   - Complete raw response
#   - SCD risk category and rationale
#   - PFD risk category and rationale
#   - Neutral clinical summary
#   - Label-only and rationale-only response variants
#   - Format-validation status
#   - Number of generation attempts
#   - UTC generation timestamp
#
# Reproducibility metadata saved in output:
#   - Exact model identifier
#   - Requested model revision
#   - Resolved model commit
#   - Prompt SHA-256 checksum
#   - Transformers version
#   - PyTorch version
#   - CUDA version
#   - GPU names
#   - Device-map strategy
#   - Seed and decoding parameters
#
# Information to record separately for the manuscript/supplement:
#   - Date of generation
#   - Computing cluster and node
#   - GPU model and number of GPUs
#   - Conda environment/package export
#   - Prompt-generation code version or Git commit
#   - Inference-script version or Git commit
#   - Whether Hugging Face model files were cached or downloaded
#   - Any generation failures, format errors, or manual exclusions
#
# Important safeguards:
#   - Prompts contain no outcome labels, follow-up duration,
#     cause of death, study-exit status, or fold assignments.
#   - Prompt wording and generation settings are locked before
#     examining model performance.
#   - LLM responses are generated once per patient and condition;
#     downstream nested cross-validation does not regenerate them.
#
# Expected workload:
#   - 730 patients x 3 conditions = 2,190 responses per model
#   - 4,380 total responses across both LLaMA models
#
# Output files:
#   - LLaMA3.1-8B-4year-responses.csv
#   - LLaMA3.2-3B-4year-responses.csv
#
# ============================================================

# ============================================================
# tmux launcher for four-year SCD/PFD LLM response generation
# Parallel execution:
#   LLaMA 3.1 8B -> GPUs 0,1,2,3
#   LLaMA 3.2 3B -> GPUs 4,5,6,7
# ============================================================

# ---------------- USER CONFIG ----------------

SESSION_NAME="llama_4year_scd_pfd_detailed_v3"

GPU_IDS_8B="0,1,2,3"
GPU_IDS_3B="4,5,6,7"

CONDA_ENV="shdb-af-analysis"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/generate_llm_risks.py"

CSV_PATH="${SCRIPT_DIR}/../../music/subject-info-cleaned-4year-with-prompts.csv"
# Replace this placeholder with the Prompt CSV SHA-256 printed by the
# final notebook save-and-verification cell.
EXPECTED_PROMPT_SHA256="4676a5e8d918b0ba51df136c59c59062b5b2df1efe97b58d34d30d094b35ec79"

# Keep this detailed-rationale rerun separate from all earlier responses.
OUTPUT_DIR="${SCRIPT_DIR}/../../music/llm_responses_4year_v3_detailed"
LOG_DIR="${OUTPUT_DIR}/logs"
METADATA_DIR="${OUTPUT_DIR}/run_metadata"

OUTPUT_8B="${OUTPUT_DIR}/LLaMA3.1-8B-4year-responses.csv"
OUTPUT_3B="${OUTPUT_DIR}/LLaMA3.2-3B-4year-responses.csv"

LOG_8B="${LOG_DIR}/LLaMA3.1-8B-generation.log"
LOG_3B="${LOG_DIR}/LLaMA3.2-3B-generation.log"

MODEL_8B="meta-llama/Llama-3.1-8B-Instruct"
MODEL_3B="meta-llama/Llama-3.2-3B-Instruct"
REVISION_8B="0e9e39f249a16976918f6564b8830bc894c89659"
REVISION_3B="0cb88a4f764b7a12671c53f0838cd831a0843b95"

PROMPT_COLUMNS=(
    "Full_Risk_No_ECG_Prompt"
    "Neutral_Summary_No_ECG_Prompt"
    "Full_Risk_With_ECG_Prompt"
)

BATCH_SIZE_8B=4
BATCH_SIZE_3B=4
# This is only a truncation safeguard. The prompt does not request a
# particular word count or sentence count, and generation may stop earlier.
MAX_NEW_TOKENS=1024
MAX_RETRIES=3
SAVE_EVERY=20
SEED=42
REPETITION_PENALTY=1.1
EXPECTED_PATIENTS=730
RUN_LABEL="four_year_v3_detailed_rationales"

# ---------------- END CONFIG ----------------

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${METADATA_DIR}"

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "ERROR: Python script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

if [[ ! -f "${CSV_PATH}" ]]; then
    echo "ERROR: Input CSV not found: ${CSV_PATH}" >&2
    exit 1
fi

if [[ ! "${EXPECTED_PROMPT_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "ERROR: EXPECTED_PROMPT_SHA256 has not been configured." >&2
    echo "Copy the Prompt CSV SHA-256 printed by the notebook into:" >&2
    echo "  EXPECTED_PROMPT_SHA256=\"...\"" >&2
    exit 1
fi

ACTUAL_PROMPT_SHA256="$(sha256sum "${CSV_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_PROMPT_SHA256}" != "${EXPECTED_PROMPT_SHA256}" ]]; then
    echo "ERROR: Prompt CSV checksum mismatch." >&2
    echo "Expected: ${EXPECTED_PROMPT_SHA256}" >&2
    echo "Observed: ${ACTUAL_PROMPT_SHA256}" >&2
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION_NAME}' already exists." >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

printf -v PROMPT_ARGS " %q" "${PROMPT_COLUMNS[@]}"

COMMON_ARGS="--device_map balanced --input_csv ${CSV_PATH} --expected_prompt_sha256 ${EXPECTED_PROMPT_SHA256} --expected_patients ${EXPECTED_PATIENTS} --run_label ${RUN_LABEL} --max_new_tokens ${MAX_NEW_TOKENS} --max_retries ${MAX_RETRIES} --save_every ${SAVE_EVERY} --seed ${SEED} --repetition_penalty ${REPETITION_PENALTY} --prompt_columns${PROMPT_ARGS}"

COMMAND_8B="source ~/.bashrc && conda activate ${CONDA_ENV} && export CUDA_VISIBLE_DEVICES=${GPU_IDS_8B} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 && { set -o pipefail; python ${SCRIPT_PATH} --model ${MODEL_8B} --revision ${REVISION_8B} --output_csv ${OUTPUT_8B} --batch_size ${BATCH_SIZE_8B} ${COMMON_ARGS} 2>&1 | tee -a ${LOG_8B}; status=\${PIPESTATUS[0]}; echo; echo 'LLaMA 3.1 8B exit status:' \${status}; echo 'Finished at:' \$(date --iso-8601=seconds); }; exec bash"

COMMAND_3B="source ~/.bashrc && conda activate ${CONDA_ENV} && export CUDA_VISIBLE_DEVICES=${GPU_IDS_3B} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 && { set -o pipefail; python ${SCRIPT_PATH} --model ${MODEL_3B} --revision ${REVISION_3B} --output_csv ${OUTPUT_3B} --batch_size ${BATCH_SIZE_3B} ${COMMON_ARGS} 2>&1 | tee -a ${LOG_3B}; status=\${PIPESTATUS[0]}; echo; echo 'LLaMA 3.2 3B exit status:' \${status}; echo 'Finished at:' \$(date --iso-8601=seconds); }; exec bash"

{
    echo "launch_time_utc=$(date -u --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "input_csv=$(readlink -f "${CSV_PATH}")"
    echo "input_csv_sha256=${ACTUAL_PROMPT_SHA256}"
    echo "python_script=$(readlink -f "${SCRIPT_PATH}")"
    echo "python_script_sha256=$(sha256sum "${SCRIPT_PATH}" | awk '{print $1}')"
    echo "launcher_script=$(readlink -f "${BASH_SOURCE[0]}")"
    echo "launcher_script_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
    echo "model_8b=${MODEL_8B}"
    echo "revision_8b=${REVISION_8B}"
    echo "model_3b=${MODEL_3B}"
    echo "revision_3b=${REVISION_3B}"
    echo "gpu_ids_8b=${GPU_IDS_8B}"
    echo "gpu_ids_3b=${GPU_IDS_3B}"
    echo "seed=${SEED}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "repetition_penalty=${REPETITION_PENALTY}"
    echo "batch_size_8b=${BATCH_SIZE_8B}"
    echo "batch_size_3b=${BATCH_SIZE_3B}"
} > "${METADATA_DIR}/launcher_configuration.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total \
        --format=csv,noheader \
        > "${METADATA_DIR}/gpu_inventory.csv"
fi

echo "Launching tmux session: ${SESSION_NAME}"
echo "LLaMA 3.1 8B GPUs: ${GPU_IDS_8B}"
echo "LLaMA 3.2 3B GPUs: ${GPU_IDS_3B}"
echo "Prompt CSV checksum verified: ${ACTUAL_PROMPT_SHA256}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Existing output files will be resumed automatically."

tmux new-session -d -s "${SESSION_NAME}" -n "llama8b"
tmux send-keys -t "${SESSION_NAME}:llama8b" "${COMMAND_8B}" C-m

tmux new-window -t "${SESSION_NAME}" -n "llama3b"
tmux send-keys -t "${SESSION_NAME}:llama3b" "${COMMAND_3B}" C-m

tmux select-window -t "${SESSION_NAME}:llama8b"
tmux attach -t "${SESSION_NAME}"
