#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_gedi_targets_slurm.sh --targets PATH [options]

Options:
  --targets PATH        targets.csv from `--mode design`. Required.
  --partition NAME      Slurm partition. Default: CPU (or GEDI_PARTITION).
  --qos NAME            Slurm QOS. Default: minor_student on the CPU partition.
  --time HH:MM:SS       Slurm time limit. Default: 03:00:00 (or GEDI_TIME).
  --mem SIZE            Slurm memory, for example 8G. Default: 8G (or GEDI_MEM).
  --cpus-per-task N     Slurm CPUs per task. Default: 1.
  --gedi-python PATH    Sidecar interpreter. Default: .venv-gedi/bin/python.
  --base-seed N         Deterministic base seed. Default: 2024.
  --n-trials N          SMAC trials per target. Default: 50.
  --max-attempts N      Generation attempts per target. Default: 3.
  --timeout-seconds N   Per-GEDI-call timeout. Default: 1800.
  --array RANGE         Slurm array range, for example 0-199 or 0-199%20.
  --row-offset N        Targets row offset added to each Slurm task id.
  --all-rows            Submit the full targets file in chunked Slurm arrays.
  --max-array-tasks N   Maximum raw Slurm array tasks per submission. Default: 1000.
  --max-rows N          Submit rows 0 through N-1 for a small smoke sweep.
  --row N               Submit exactly one zero-based targets row.
  --array-start N       First array index to submit. Use with --array-end.
  --array-end N         Last array index to submit. Use with --array-start.
  --exclude NODES       Exclude Slurm node list.
  --job-name NAME       Slurm job name. Default: gedi_targets.
  --output PATH         Slurm stdout pattern. Default: logs/slurm/gedi_targets_%A_%a.out.
  --error PATH          Slurm stderr pattern. Default: logs/slurm/gedi_targets_%A_%a.err.
  --dry-run             Validate and print the sbatch command without submitting.
  -h, --help            Show this help.

GEDI synthesis jobs are CPU-only and default to the CPU partition with the
minor_student QOS the CPU partition requires. One array task runs exactly one
targets.csv row via slurm/run_gedi_target.sh. Requires the GEDI sidecar venv
(see docs/feature_space_generation.md) and anchor_features.csv next to the
targets file.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

is_nonnegative_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

is_positive_integer() {
  [[ "${1}" =~ ^[1-9][0-9]*$ ]]
}

# The site's CPU partition requires an explicit student QOS. When the caller
# did not set a QOS, derive one from the resolved partition.
default_qos_for_partition() {
  case "$1" in
    CPU|cpu) printf '%s' "minor_student" ;;
    *) printf '%s' "" ;;
  esac
}

validate_targets_file() {
  local targets_path="$1"
  local python_bin="$2"
  "${python_bin}" - "${targets_path}" <<'PY'
from __future__ import annotations

import csv
import sys

required = [
    "target_id",
    "band",
    "concurrency",
    "noise_level",
    "feasible",
    "nearest_real_distance",
    "target_num_traces",
    "target_avg_trace_length",
    "target_num_activities",
    "target_variant_ratio",
    "target_dfg_density",
    "target_repetition_prevalence",
]
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise SystemExit(f"targets file missing required column(s): {', '.join(missing)}")
    rows = list(reader)
if not rows:
    raise SystemExit("targets file has no data rows")
PY
}

parse_array_bounds() {
  local array_spec="$1"
  local array_range="${array_spec%%%*}"
  local start
  local end
  if [[ "${array_range}" == *-* ]]; then
    start="${array_range%%-*}"
    end="${array_range##*-}"
  else
    start="${array_range}"
    end="${array_range}"
  fi
  is_nonnegative_integer "${start}" || fail "--array must start with a nonnegative integer"
  is_nonnegative_integer "${end}" || fail "--array must end with a nonnegative integer"
  [[ "${start}" -le "${end}" ]] || fail "--array start must be <= end"
  ARRAY_FIRST="${start}"
  ARRAY_LAST="${end}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_PROJECT_ROOT}/scripts/lib/env.sh"
pdcash_load_dotenv "${PDCASH_DOTENV_PATH:-${SCRIPT_PROJECT_ROOT}/.env}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_PROJECT_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/slurm}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
GEDI_PYTHON="${GEDI_PYTHON:-${PROJECT_ROOT}/.venv-gedi/bin/python}"

TARGETS_PATH=""
PARTITION="${GEDI_PARTITION:-}"
QOS="${GEDI_QOS:-}"
TIME_LIMIT="${GEDI_TIME:-}"
MEMORY="${GEDI_MEM:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
BASE_SEED="${GEDI_BASE_SEED:-2024}"
N_TRIALS="${GEDI_N_TRIALS:-50}"
MAX_ATTEMPTS="${GEDI_MAX_ATTEMPTS:-3}"
TIMEOUT_SECONDS="${GEDI_TIMEOUT_SECONDS:-1800}"
MAX_ROWS=""
ROW_INDEX=""
ARRAY_START=""
ARRAY_END=""
ARRAY_SPEC=""
ROW_OFFSET="0"
ROW_OFFSET_EXPLICIT="false"
ALL_ROWS="false"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1000}"
EXCLUDE="${GEDI_EXCLUDE:-}"
JOB_NAME=""
JOB_NAME_EXPLICIT="false"
OUTPUT_PATH=""
OUTPUT_PATH_EXPLICIT="false"
ERROR_PATH=""
ERROR_PATH_EXPLICIT="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --targets)
      [[ $# -ge 2 ]] || fail "--targets requires a path"
      TARGETS_PATH="$2"
      shift 2
      ;;
    --partition)
      [[ $# -ge 2 ]] || fail "--partition requires a name"
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      [[ $# -ge 2 ]] || fail "--qos requires a name"
      [[ -n "$2" ]] || fail "--qos requires a non-empty name"
      QOS="$2"
      shift 2
      ;;
    --qos=*)
      QOS="${1#--qos=}"
      [[ -n "${QOS}" ]] || fail "--qos requires a non-empty name"
      shift
      ;;
    --time)
      [[ $# -ge 2 ]] || fail "--time requires a Slurm time limit"
      TIME_LIMIT="$2"
      shift 2
      ;;
    --mem|--memory)
      [[ $# -ge 2 ]] || fail "--mem requires a memory value"
      MEMORY="$2"
      shift 2
      ;;
    --cpus-per-task)
      [[ $# -ge 2 ]] || fail "--cpus-per-task requires a positive integer"
      CPUS_PER_TASK="$2"
      shift 2
      ;;
    --gedi-python)
      [[ $# -ge 2 ]] || fail "--gedi-python requires a path"
      GEDI_PYTHON="$2"
      shift 2
      ;;
    --base-seed)
      [[ $# -ge 2 ]] || fail "--base-seed requires an integer"
      BASE_SEED="$2"
      shift 2
      ;;
    --n-trials)
      [[ $# -ge 2 ]] || fail "--n-trials requires a positive integer"
      N_TRIALS="$2"
      shift 2
      ;;
    --max-attempts)
      [[ $# -ge 2 ]] || fail "--max-attempts requires a positive integer"
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --timeout-seconds)
      [[ $# -ge 2 ]] || fail "--timeout-seconds requires a positive integer"
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --array)
      [[ $# -ge 2 ]] || fail "--array requires a range"
      ARRAY_SPEC="$2"
      shift 2
      ;;
    --array=*)
      ARRAY_SPEC="${1#--array=}"
      shift
      ;;
    --row-offset)
      [[ $# -ge 2 ]] || fail "--row-offset requires a nonnegative integer"
      ROW_OFFSET="$2"
      ROW_OFFSET_EXPLICIT="true"
      shift 2
      ;;
    --row-offset=*)
      ROW_OFFSET="${1#--row-offset=}"
      ROW_OFFSET_EXPLICIT="true"
      shift
      ;;
    --all-rows)
      ALL_ROWS="true"
      shift
      ;;
    --max-array-tasks)
      [[ $# -ge 2 ]] || fail "--max-array-tasks requires a positive integer"
      MAX_ARRAY_TASKS="$2"
      shift 2
      ;;
    --max-array-tasks=*)
      MAX_ARRAY_TASKS="${1#--max-array-tasks=}"
      shift
      ;;
    --max-rows)
      [[ $# -ge 2 ]] || fail "--max-rows requires a positive integer"
      MAX_ROWS="$2"
      shift 2
      ;;
    --row)
      [[ $# -ge 2 ]] || fail "--row requires a zero-based row index"
      ROW_INDEX="$2"
      shift 2
      ;;
    --array-start)
      [[ $# -ge 2 ]] || fail "--array-start requires a zero-based row index"
      ARRAY_START="$2"
      shift 2
      ;;
    --array-end)
      [[ $# -ge 2 ]] || fail "--array-end requires a zero-based row index"
      ARRAY_END="$2"
      shift 2
      ;;
    --exclude)
      [[ $# -ge 2 ]] || fail "--exclude requires a node list"
      EXCLUDE="$2"
      shift 2
      ;;
    --exclude=*)
      EXCLUDE="${1#--exclude=}"
      shift
      ;;
    --job-name)
      [[ $# -ge 2 ]] || fail "--job-name requires a name"
      JOB_NAME="$2"
      JOB_NAME_EXPLICIT="true"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || fail "--output requires a path pattern"
      OUTPUT_PATH="$2"
      OUTPUT_PATH_EXPLICIT="true"
      shift 2
      ;;
    --error)
      [[ $# -ge 2 ]] || fail "--error requires a path pattern"
      ERROR_PATH="$2"
      ERROR_PATH_EXPLICIT="true"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ -n "${TARGETS_PATH}" ]] || fail "--targets is required"
is_positive_integer "${CPUS_PER_TASK}" || fail "--cpus-per-task must be a positive integer"
is_positive_integer "${MAX_ARRAY_TASKS}" || fail "--max-array-tasks must be a positive integer"
is_nonnegative_integer "${ROW_OFFSET}" || fail "--row-offset must be a nonnegative integer"
is_positive_integer "${N_TRIALS}" || fail "--n-trials must be a positive integer"
is_positive_integer "${MAX_ATTEMPTS}" || fail "--max-attempts must be a positive integer"
is_positive_integer "${TIMEOUT_SECONDS}" || fail "--timeout-seconds must be a positive integer"

case "${PARTITION}" in
  all)
    fail "partition 'all' is not allowed for student accounts. Use a valid site partition from sinfo."
    ;;
esac

if [[ -n "${MAX_ROWS}" && -n "${ROW_INDEX}" ]]; then
  fail "--max-rows and --row are mutually exclusive"
fi
if [[ "${ALL_ROWS}" == "true" && ( -n "${ARRAY_SPEC}" || -n "${ROW_INDEX}" || -n "${MAX_ROWS}" || -n "${ARRAY_START}" || -n "${ARRAY_END}" ) ]]; then
  fail "--all-rows cannot be combined with --array, --row, --max-rows, --array-start, or --array-end"
fi
if [[ "${ALL_ROWS}" == "true" && "${ROW_OFFSET_EXPLICIT}" == "true" ]]; then
  fail "--all-rows cannot be combined with --row-offset"
fi
if [[ -n "${ROW_INDEX}" && "${ROW_OFFSET_EXPLICIT}" == "true" ]]; then
  fail "--row-offset cannot be combined with --row"
fi
if [[ -n "${MAX_ROWS}" && "${ROW_OFFSET_EXPLICIT}" == "true" ]]; then
  fail "--row-offset cannot be combined with --max-rows"
fi
if [[ -n "${ROW_INDEX}" && ( -n "${ARRAY_START}" || -n "${ARRAY_END}" || -n "${ARRAY_SPEC}" ) ]]; then
  fail "--row cannot be combined with --array/--array-start/--array-end"
fi
if [[ -n "${MAX_ROWS}" && ( -n "${ARRAY_START}" || -n "${ARRAY_END}" || -n "${ARRAY_SPEC}" ) ]]; then
  fail "--max-rows cannot be combined with --array/--array-start/--array-end"
fi
if [[ -n "${ARRAY_SPEC}" && ( -n "${ARRAY_START}" || -n "${ARRAY_END}" ) ]]; then
  fail "--array cannot be combined with --array-start/--array-end"
fi

if [[ -n "${MAX_ROWS}" ]] && ! is_positive_integer "${MAX_ROWS}"; then
  fail "--max-rows must be a positive integer"
fi
if [[ -n "${ROW_INDEX}" ]] && ! is_nonnegative_integer "${ROW_INDEX}"; then
  fail "--row must be a nonnegative integer"
fi
if [[ -n "${ARRAY_START}" || -n "${ARRAY_END}" ]]; then
  [[ -n "${ARRAY_START}" && -n "${ARRAY_END}" ]] || \
    fail "--array-start and --array-end must be provided together"
  is_nonnegative_integer "${ARRAY_START}" || fail "--array-start must be a nonnegative integer"
  is_nonnegative_integer "${ARRAY_END}" || fail "--array-end must be a nonnegative integer"
  [[ "${ARRAY_START}" -le "${ARRAY_END}" ]] || fail "--array-start must be <= --array-end"
fi

if [[ "${PROJECT_ROOT}" == "/var/lib/slurm/slurmd" || "${PROJECT_ROOT}" == /var/lib/slurm/slurmd/* ]]; then
  fail "PROJECT_ROOT points inside the Slurm spool directory: ${PROJECT_ROOT}"
fi
if [[ ! -d "${PROJECT_ROOT}/src" ]] \
  || [[ ! -d "${PROJECT_ROOT}/scripts" ]] \
  || [[ ! -d "${PROJECT_ROOT}/slurm" ]] \
  || [[ ! -f "${PROJECT_ROOT}/slurm/run_gedi_target.sh" ]]; then
  fail "PROJECT_ROOT does not look like the cash-process-discovery training_data project: ${PROJECT_ROOT}"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}"

export PROJECT_ROOT DATA_ROOT RESULTS_ROOT LOG_ROOT
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "Python executable not found: ${PYTHON_BIN}"
fi

TARGETS_ABS="$(pdcash_resolve_data_path "${TARGETS_PATH}")"
TARGETS_ARG="${TARGETS_ABS}"
[[ -f "${TARGETS_ABS}" ]] || fail "targets file does not exist: ${TARGETS_PATH}"

ANCHOR_ABS="$(dirname "${TARGETS_ABS}")/anchor_features.csv"
[[ -f "${ANCHOR_ABS}" ]] || fail "anchor_features.csv not found next to the targets file: ${ANCHOR_ABS}. Generate it with --mode design."

validate_targets_file "${TARGETS_ABS}" "${PYTHON_BIN}"

NUM_ROWS=$(($(wc -l < "${TARGETS_ABS}") - 1))
[[ "${NUM_ROWS}" -gt 0 ]] || fail "targets file has no data rows: ${TARGETS_PATH}"
LAST_AVAILABLE=$((NUM_ROWS - 1))

[[ -n "${PARTITION}" ]] || PARTITION="CPU"
[[ -n "${TIME_LIMIT}" ]] || TIME_LIMIT="03:00:00"
[[ -n "${MEMORY}" ]] || MEMORY="8G"
[[ -n "${QOS}" ]] || QOS="$(default_qos_for_partition "${PARTITION}")"

case "${PARTITION}" in
  "")
    fail "partition resolved to empty; pass --partition or set GEDI_PARTITION"
    ;;
  all)
    fail "partition 'all' is not allowed for student accounts. Use a valid site partition from sinfo."
    ;;
esac

build_sbatch_command() {
  local array_spec="$1"
  local row_offset="$2"
  local force_offset_suffix="$3"
  local offset_suffix=""
  local chunk_job_name
  local chunk_output_path
  local chunk_error_path

  if [[ "${force_offset_suffix}" == "true" || "${row_offset}" -ne 0 ]]; then
    offset_suffix="_o${row_offset}"
  fi

  if [[ "${JOB_NAME_EXPLICIT}" == "true" ]]; then
    chunk_job_name="${JOB_NAME}"
  else
    chunk_job_name="gedi_targets${offset_suffix}"
  fi

  if [[ "${OUTPUT_PATH_EXPLICIT}" == "true" ]]; then
    chunk_output_path="${OUTPUT_PATH}"
  else
    chunk_output_path="${LOG_ROOT}/gedi_targets${offset_suffix}_%A_%a.out"
  fi

  if [[ "${ERROR_PATH_EXPLICIT}" == "true" ]]; then
    chunk_error_path="${ERROR_PATH}"
  else
    chunk_error_path="${LOG_ROOT}/gedi_targets${offset_suffix}_%A_%a.err"
  fi

  SBATCH_COMMAND=(
    sbatch
    --partition="${PARTITION}"
    --time="${TIME_LIMIT}"
    --mem="${MEMORY}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --job-name="${chunk_job_name}"
    --output="${chunk_output_path}"
    --error="${chunk_error_path}"
    --chdir "${PROJECT_ROOT}"
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},GEDI_PYTHON=${GEDI_PYTHON},GEDI_ROW_OFFSET=${row_offset},GEDI_BASE_SEED=${BASE_SEED},GEDI_N_TRIALS=${N_TRIALS},GEDI_MAX_ATTEMPTS=${MAX_ATTEMPTS},GEDI_TIMEOUT_SECONDS=${TIMEOUT_SECONDS}"
    --array="${array_spec}"
  )

  if [[ -n "${EXCLUDE}" ]]; then
    SBATCH_COMMAND+=(--exclude="${EXCLUDE}")
  fi
  if [[ -n "${QOS}" ]]; then
    SBATCH_COMMAND+=(--qos="${QOS}")
  fi

  SBATCH_COMMAND+=(
    slurm/run_gedi_target.sh
    "${TARGETS_ARG}"
  )
}

print_submission_summary() {
  local array_spec="$1"
  local row_offset="$2"
  local chunk_number="$3"
  local chunk_count="$4"
  local actual_first="$5"
  local actual_last="$6"

  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "Targets=${TARGETS_ARG}"
  echo "Targets rows=${NUM_ROWS}"
  echo "Chunk=${chunk_number}/${chunk_count}"
  echo "Submitting array=${array_spec}"
  echo "GEDI_ROW_OFFSET=${row_offset}"
  echo "Targets row range=${actual_first}-${actual_last}"
  echo "Partition=${PARTITION}"
  echo "QOS=${QOS}"
  echo "Time=${TIME_LIMIT}"
  echo "Memory=${MEMORY}"
  echo "CPUs per task=${CPUS_PER_TASK}"
  echo "GEDI_PYTHON=${GEDI_PYTHON}"
  echo "Base seed=${BASE_SEED}"
  echo "SMAC trials=${N_TRIALS}"
  echo "Max attempts=${MAX_ATTEMPTS}"
  echo "Timeout seconds=${TIMEOUT_SECONDS}"
  echo "Exclude=${EXCLUDE}"
  echo "${SBATCH_COMMAND[*]}"
}

submit_or_print() {
  local array_spec="$1"
  local row_offset="$2"
  local force_offset_suffix="$3"
  local chunk_number="$4"
  local chunk_count="$5"
  local actual_first="$6"
  local actual_last="$7"

  build_sbatch_command "${array_spec}" "${row_offset}" "${force_offset_suffix}"
  print_submission_summary \
    "${array_spec}" \
    "${row_offset}" \
    "${chunk_number}" \
    "${chunk_count}" \
    "${actual_first}" \
    "${actual_last}"

  if [[ "${DRY_RUN}" != "true" ]]; then
    "${SBATCH_COMMAND[@]}"
  fi
}

validate_raw_array_limit() {
  local array_last="$1"
  local max_task_id=$((MAX_ARRAY_TASKS - 1))
  if [[ "${array_last}" -gt "${max_task_id}" ]]; then
    fail "array task id ${array_last} exceeds --max-array-tasks ${MAX_ARRAY_TASKS}; use --row-offset or --all-rows"
  fi
}

SUBMIT_ARRAY_SPECS=()
SUBMIT_ROW_OFFSETS=()
SUBMIT_ACTUAL_FIRST=()
SUBMIT_ACTUAL_LAST=()
FORCE_OFFSET_SUFFIX="false"

if [[ -n "${ROW_INDEX}" ]]; then
  [[ "${ROW_INDEX}" -le "${LAST_AVAILABLE}" ]] || \
    fail "--row ${ROW_INDEX} is outside targets range 0-${LAST_AVAILABLE}"
  ARRAY_SPEC="0"
  ROW_OFFSET="${ROW_INDEX}"
  SUBMIT_ARRAY_SPECS+=("${ARRAY_SPEC}")
  SUBMIT_ROW_OFFSETS+=("${ROW_OFFSET}")
  SUBMIT_ACTUAL_FIRST+=("${ROW_INDEX}")
  SUBMIT_ACTUAL_LAST+=("${ROW_INDEX}")
elif [[ -n "${MAX_ROWS}" ]]; then
  [[ "${MAX_ROWS}" -le "${NUM_ROWS}" ]] || \
    fail "--max-rows ${MAX_ROWS} exceeds targets row count ${NUM_ROWS}"
  [[ "${MAX_ROWS}" -le "${MAX_ARRAY_TASKS}" ]] || \
    fail "--max-rows ${MAX_ROWS} exceeds --max-array-tasks ${MAX_ARRAY_TASKS}; use --all-rows"
  LAST_INDEX=$((MAX_ROWS - 1))
  ARRAY_SPEC="0-${LAST_INDEX}"
  SUBMIT_ARRAY_SPECS+=("${ARRAY_SPEC}")
  SUBMIT_ROW_OFFSETS+=("0")
  SUBMIT_ACTUAL_FIRST+=("0")
  SUBMIT_ACTUAL_LAST+=("${LAST_INDEX}")
elif [[ "${ALL_ROWS}" == "true" ]]; then
  FORCE_OFFSET_SUFFIX="true"
  offset=0
  while [[ "${offset}" -lt "${NUM_ROWS}" ]]; do
    remaining=$((NUM_ROWS - offset))
    chunk_size="${MAX_ARRAY_TASKS}"
    if [[ "${remaining}" -lt "${chunk_size}" ]]; then
      chunk_size="${remaining}"
    fi
    array_last=$((chunk_size - 1))
    actual_last=$((offset + chunk_size - 1))
    SUBMIT_ARRAY_SPECS+=("0-${array_last}")
    SUBMIT_ROW_OFFSETS+=("${offset}")
    SUBMIT_ACTUAL_FIRST+=("${offset}")
    SUBMIT_ACTUAL_LAST+=("${actual_last}")
    offset=$((offset + chunk_size))
  done
elif [[ -n "${ARRAY_SPEC}" ]]; then
  parse_array_bounds "${ARRAY_SPEC}"
  validate_raw_array_limit "${ARRAY_LAST}"
  actual_first=$((ROW_OFFSET + ARRAY_FIRST))
  actual_last=$((ROW_OFFSET + ARRAY_LAST))
  [[ "${actual_first}" -le "${LAST_AVAILABLE}" && "${actual_last}" -le "${LAST_AVAILABLE}" ]] || \
    fail "array ${ARRAY_SPEC} with row offset ${ROW_OFFSET} is outside targets range 0-${LAST_AVAILABLE}"
  SUBMIT_ARRAY_SPECS+=("${ARRAY_SPEC}")
  SUBMIT_ROW_OFFSETS+=("${ROW_OFFSET}")
  SUBMIT_ACTUAL_FIRST+=("${actual_first}")
  SUBMIT_ACTUAL_LAST+=("${actual_last}")
elif [[ -n "${ARRAY_START}" || -n "${ARRAY_END}" ]]; then
  array_length=$((ARRAY_END - ARRAY_START + 1))
  [[ "${array_length}" -le "${MAX_ARRAY_TASKS}" ]] || \
    fail "array range ${ARRAY_START}-${ARRAY_END} exceeds --max-array-tasks ${MAX_ARRAY_TASKS}; use --row-offset or --all-rows"
  [[ "${ARRAY_END}" -le "${LAST_AVAILABLE}" ]] || \
    fail "--array-end ${ARRAY_END} is outside targets range 0-${LAST_AVAILABLE}"
  array_last=$((array_length - 1))
  SUBMIT_ARRAY_SPECS+=("0-${array_last}")
  SUBMIT_ROW_OFFSETS+=("${ARRAY_START}")
  SUBMIT_ACTUAL_FIRST+=("${ARRAY_START}")
  SUBMIT_ACTUAL_LAST+=("${ARRAY_END}")
else
  if [[ "${NUM_ROWS}" -gt "${MAX_ARRAY_TASKS}" ]]; then
    fail "targets file has ${NUM_ROWS} rows, exceeding --max-array-tasks ${MAX_ARRAY_TASKS}; use --all-rows"
  fi
  LAST_INDEX=$((NUM_ROWS - 1))
  SUBMIT_ARRAY_SPECS+=("0-${LAST_INDEX}")
  SUBMIT_ROW_OFFSETS+=("0")
  SUBMIT_ACTUAL_FIRST+=("0")
  SUBMIT_ACTUAL_LAST+=("${LAST_INDEX}")
fi

chunk_count="${#SUBMIT_ARRAY_SPECS[@]}"
for index in "${!SUBMIT_ARRAY_SPECS[@]}"; do
  chunk_number=$((index + 1))
  submit_or_print \
    "${SUBMIT_ARRAY_SPECS[$index]}" \
    "${SUBMIT_ROW_OFFSETS[$index]}" \
    "${FORCE_OFFSET_SUFFIX}" \
    "${chunk_number}" \
    "${chunk_count}" \
    "${SUBMIT_ACTUAL_FIRST[$index]}" \
    "${SUBMIT_ACTUAL_LAST[$index]}"
done
