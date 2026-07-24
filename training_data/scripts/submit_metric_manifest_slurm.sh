#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_metric_manifest_slurm.sh --manifest PATH [options]

Options:
  --manifest PATH       Generated metric manifest CSV to submit. Required.
  --profile NAME        Metric profile used for defaults and log naming.
  --partition NAME      Slurm partition. Overrides profile/default partition.
  --qos NAME            Slurm QOS. Overrides profile/default QOS.
  --time HH:MM:SS       Slurm time limit. Overrides profile/default time.
  --mem SIZE            Slurm memory, for example 8G. Overrides profile/default memory.
  --cpus-per-task N     Slurm CPUs per task. Default: 1.
  --array RANGE         Slurm array range, for example 0-999 or 0-999%20.
  --row-offset N        Metric manifest row offset added to each Slurm task id.
  --all-rows            Submit the full metric manifest in chunked Slurm arrays.
  --max-array-tasks N   Maximum raw Slurm array tasks per submission. Default: 1000.
  --max-rows N          Submit rows 0 through N-1 for a small smoke sweep.
  --row N               Submit exactly one zero-based metric manifest row.
  --array-start N       First array index to submit. Use with --array-end.
  --array-end N         Last array index to submit. Use with --array-start.
  --exclude NODES       Exclude Slurm node list.
  --job-name NAME       Slurm job name. Default includes metric profile.
  --output PATH         Slurm stdout pattern. Default: logs/slurm/metrics_<profile>_%A_%a.out.
  --error PATH          Slurm stderr pattern. Default: logs/slurm/metrics_<profile>_%A_%a.err.
  --dry-run             Validate and print the sbatch command without submitting.
  -h, --help            Show this help.

Metric jobs are CPU-only and default to the CPU partition with the
minor_student QOS the CPU partition requires. Override with --partition or
METRIC_PARTITION, and --qos or METRIC_QOS.

Profile defaults:
  token:         CPU, minor_student QOS, 06:00:00, 16G
  pm4py_default: CPU, minor_student QOS, 08:00:00, 16G
  alignment:     CPU, minor_student QOS, 24:00:00, 32G
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

sanitize_name() {
  printf '%s' "${1}" | tr -cs '[:alnum:]_.-' '_' | sed -e 's/^_//' -e 's/_$//'
}

env_prefix_for_profile() {
  printf '%s' "${1}" | tr '[:lower:]-' '[:upper:]_' | tr -cs '[:alnum:]_' '_'
}

env_value() {
  local name="$1"
  printf '%s' "${!name:-}"
}

profile_default() {
  local profile="$1"
  local setting="$2"
  local env_prefix
  local profile_env
  env_prefix="$(env_prefix_for_profile "${profile}")"
  profile_env="$(env_value "${env_prefix}_METRIC_${setting}")"

  if [[ -n "${profile_env}" ]]; then
    printf '%s' "${profile_env}"
    return
  fi

  case "${profile}:${setting}" in
    token:PARTITION) printf '%s' "${METRIC_PARTITION:-CPU}" ;;
    token:QOS) printf '%s' "${METRIC_QOS:-}" ;;
    token:TIME) printf '%s' "${METRIC_TIME:-06:00:00}" ;;
    token:MEM) printf '%s' "${METRIC_MEM:-16G}" ;;
    token:EXCLUDE) printf '%s' "${METRIC_EXCLUDE:-}" ;;

    pm4py_default:PARTITION) printf '%s' "${METRIC_PARTITION:-CPU}" ;;
    pm4py_default:QOS) printf '%s' "${METRIC_QOS:-}" ;;
    pm4py_default:TIME) printf '%s' "${METRIC_TIME:-08:00:00}" ;;
    pm4py_default:MEM) printf '%s' "${METRIC_MEM:-16G}" ;;
    pm4py_default:EXCLUDE) printf '%s' "${METRIC_EXCLUDE:-}" ;;

    alignment:PARTITION) printf '%s' "${METRIC_PARTITION:-CPU}" ;;
    alignment:QOS) printf '%s' "${METRIC_QOS:-}" ;;
    alignment:TIME) printf '%s' "${METRIC_TIME:-24:00:00}" ;;
    alignment:MEM) printf '%s' "${METRIC_MEM:-32G}" ;;
    alignment:EXCLUDE) printf '%s' "${METRIC_EXCLUDE:-}" ;;

    *:PARTITION) printf '%s' "${METRIC_PARTITION:-CPU}" ;;
    *:QOS) printf '%s' "${METRIC_QOS:-}" ;;
    *:TIME) printf '%s' "${METRIC_TIME:-08:00:00}" ;;
    *:MEM) printf '%s' "${METRIC_MEM:-16G}" ;;
    *:EXCLUDE) printf '%s' "${METRIC_EXCLUDE:-}" ;;
    *) return 1 ;;
  esac
}

# The site's CPU partition requires an explicit student QOS. When the caller
# did not set a QOS (via --qos or a *_QOS env default), derive one from the
# resolved partition. Other partitions keep their prior no-default behavior.
default_qos_for_partition() {
  case "$1" in
    CPU|cpu) printf '%s' "minor_student" ;;
    *) printf '%s' "" ;;
  esac
}

infer_metric_profile() {
  local manifest_path="$1"
  local python_bin="$2"
  "${python_bin}" - "${manifest_path}" <<'PY'
from __future__ import annotations

import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

profiles = {(row.get("metric_profile") or "").strip() for row in rows}
profiles.discard("")
if len(profiles) == 1:
    print(next(iter(profiles)))
else:
    print("metrics")
PY
}

validate_metric_manifest() {
  local manifest_path="$1"
  local python_bin="$2"
  "${python_bin}" - "${manifest_path}" <<'PY'
from __future__ import annotations

import csv
import sys

required = [
    "source_result_path",
    "source_config_hash",
    "experiment_id",
    "log_id",
    "algorithm_name",
    "test_log_path",
    "metric_profile",
    "metric_names_json",
    "log_dir",
    "output_path",
]
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise SystemExit(f"metric manifest missing required column(s): {', '.join(missing)}")
    rows = list(reader)
if not rows:
    raise SystemExit("metric manifest has no data rows")
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

MANIFEST_PATH=""
PROFILE=""
PARTITION=""
QOS=""
TIME_LIMIT=""
MEMORY=""
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MAX_ROWS=""
ROW_INDEX=""
ARRAY_START=""
ARRAY_END=""
ARRAY_SPEC=""
ROW_OFFSET="0"
ROW_OFFSET_EXPLICIT="false"
ALL_ROWS="false"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1000}"
EXCLUDE=""
EXCLUDE_EXPLICIT="false"
JOB_NAME=""
JOB_NAME_EXPLICIT="false"
OUTPUT_PATH=""
OUTPUT_PATH_EXPLICIT="false"
ERROR_PATH=""
ERROR_PATH_EXPLICIT="false"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      [[ $# -ge 2 ]] || fail "--manifest requires a path"
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || fail "--profile requires a name"
      PROFILE="$2"
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
      EXCLUDE_EXPLICIT="true"
      shift 2
      ;;
    --exclude=*)
      EXCLUDE="${1#--exclude=}"
      EXCLUDE_EXPLICIT="true"
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

[[ -n "${MANIFEST_PATH}" ]] || fail "--manifest is required"
is_positive_integer "${CPUS_PER_TASK}" || fail "--cpus-per-task must be a positive integer"
is_positive_integer "${MAX_ARRAY_TASKS}" || fail "--max-array-tasks must be a positive integer"
is_nonnegative_integer "${ROW_OFFSET}" || fail "--row-offset must be a nonnegative integer"

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
  || [[ ! -f "${PROJECT_ROOT}/slurm/run_metric_row.sh" ]]; then
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

if [[ "${MANIFEST_PATH}" = /* ]]; then
  MANIFEST_ABS="${MANIFEST_PATH}"
  MANIFEST_ARG="${MANIFEST_PATH}"
else
  MANIFEST_ABS="${PROJECT_ROOT}/${MANIFEST_PATH}"
  MANIFEST_ARG="${MANIFEST_PATH}"
fi
[[ -f "${MANIFEST_ABS}" ]] || fail "metric manifest does not exist: ${MANIFEST_PATH}"

validate_metric_manifest "${MANIFEST_ABS}" "${PYTHON_BIN}"

NUM_ROWS=$(($(wc -l < "${MANIFEST_ABS}") - 1))
[[ "${NUM_ROWS}" -gt 0 ]] || fail "metric manifest has no data rows: ${MANIFEST_PATH}"
LAST_AVAILABLE=$((NUM_ROWS - 1))

if [[ -z "${PROFILE}" ]]; then
  PROFILE="$(infer_metric_profile "${MANIFEST_ABS}" "${PYTHON_BIN}")"
fi
PROFILE_SAFE="$(sanitize_name "${PROFILE}")"
[[ -n "${PROFILE_SAFE}" ]] || PROFILE_SAFE="metrics"

[[ -n "${PARTITION}" ]] || PARTITION="$(profile_default "${PROFILE}" "PARTITION")"
[[ -n "${TIME_LIMIT}" ]] || TIME_LIMIT="$(profile_default "${PROFILE}" "TIME")"
[[ -n "${MEMORY}" ]] || MEMORY="$(profile_default "${PROFILE}" "MEM")"
if [[ "${EXCLUDE_EXPLICIT}" != "true" ]]; then
  EXCLUDE="$(profile_default "${PROFILE}" "EXCLUDE")"
fi
[[ -n "${QOS}" ]] || QOS="$(profile_default "${PROFILE}" "QOS")"
[[ -n "${QOS}" ]] || QOS="$(default_qos_for_partition "${PARTITION}")"

case "${PARTITION}" in
  "")
    fail "partition resolved to empty; pass --partition or set METRIC_PARTITION"
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
    chunk_job_name="metrics_${PROFILE_SAFE}${offset_suffix}"
  fi

  if [[ "${OUTPUT_PATH_EXPLICIT}" == "true" ]]; then
    chunk_output_path="${OUTPUT_PATH}"
  else
    chunk_output_path="${LOG_ROOT}/metrics_${PROFILE_SAFE}${offset_suffix}_%A_%a.out"
  fi

  if [[ "${ERROR_PATH_EXPLICIT}" == "true" ]]; then
    chunk_error_path="${ERROR_PATH}"
  else
    chunk_error_path="${LOG_ROOT}/metrics_${PROFILE_SAFE}${offset_suffix}_%A_%a.err"
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
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},METRIC_PROFILE=${PROFILE},METRIC_MANIFEST_ROW_OFFSET=${row_offset}"
    --array="${array_spec}"
  )

  if [[ -n "${EXCLUDE}" ]]; then
    SBATCH_COMMAND+=(--exclude="${EXCLUDE}")
  fi
  if [[ -n "${QOS}" ]]; then
    SBATCH_COMMAND+=(--qos="${QOS}")
  fi

  SBATCH_COMMAND+=(
    slurm/run_metric_row.sh
    "${MANIFEST_ARG}"
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
  echo "Metric manifest=${MANIFEST_ARG}"
  echo "Metric manifest rows=${NUM_ROWS}"
  echo "Metric profile=${PROFILE}"
  echo "Chunk=${chunk_number}/${chunk_count}"
  echo "Submitting array=${array_spec}"
  echo "METRIC_MANIFEST_ROW_OFFSET=${row_offset}"
  echo "Metric manifest row range=${actual_first}-${actual_last}"
  echo "Partition=${PARTITION}"
  echo "QOS=${QOS}"
  echo "Time=${TIME_LIMIT}"
  echo "Memory=${MEMORY}"
  echo "CPUs per task=${CPUS_PER_TASK}"
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
    fail "--row ${ROW_INDEX} is outside metric manifest range 0-${LAST_AVAILABLE}"
  ARRAY_SPEC="0"
  ROW_OFFSET="${ROW_INDEX}"
  SUBMIT_ARRAY_SPECS+=("${ARRAY_SPEC}")
  SUBMIT_ROW_OFFSETS+=("${ROW_OFFSET}")
  SUBMIT_ACTUAL_FIRST+=("${ROW_INDEX}")
  SUBMIT_ACTUAL_LAST+=("${ROW_INDEX}")
elif [[ -n "${MAX_ROWS}" ]]; then
  [[ "${MAX_ROWS}" -le "${NUM_ROWS}" ]] || \
    fail "--max-rows ${MAX_ROWS} exceeds metric manifest row count ${NUM_ROWS}"
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
    fail "array ${ARRAY_SPEC} with row offset ${ROW_OFFSET} is outside metric manifest range 0-${LAST_AVAILABLE}"
  SUBMIT_ARRAY_SPECS+=("${ARRAY_SPEC}")
  SUBMIT_ROW_OFFSETS+=("${ROW_OFFSET}")
  SUBMIT_ACTUAL_FIRST+=("${actual_first}")
  SUBMIT_ACTUAL_LAST+=("${actual_last}")
elif [[ -n "${ARRAY_START}" || -n "${ARRAY_END}" ]]; then
  array_length=$((ARRAY_END - ARRAY_START + 1))
  [[ "${array_length}" -le "${MAX_ARRAY_TASKS}" ]] || \
    fail "array range ${ARRAY_START}-${ARRAY_END} exceeds --max-array-tasks ${MAX_ARRAY_TASKS}; use --row-offset or --all-rows"
  [[ "${ARRAY_END}" -le "${LAST_AVAILABLE}" ]] || \
    fail "--array-end ${ARRAY_END} is outside metric manifest range 0-${LAST_AVAILABLE}"
  array_last=$((array_length - 1))
  SUBMIT_ARRAY_SPECS+=("0-${array_last}")
  SUBMIT_ROW_OFFSETS+=("${ARRAY_START}")
  SUBMIT_ACTUAL_FIRST+=("${ARRAY_START}")
  SUBMIT_ACTUAL_LAST+=("${ARRAY_END}")
else
  if [[ "${NUM_ROWS}" -gt "${MAX_ARRAY_TASKS}" ]]; then
    fail "metric manifest has ${NUM_ROWS} rows, exceeding --max-array-tasks ${MAX_ARRAY_TASKS}; use --all-rows"
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
