#!/usr/bin/env bash
#SBATCH --job-name=process-metrics-dynamic
#SBATCH --output=logs/slurm/dynamic_metric_workers/bootstrap_%j.out
#SBATCH --error=logs/slurm/dynamic_metric_workers/bootstrap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash slurm/run_dynamic_metric_manifest.slurm [sbatch options] <metric-manifest.csv>

This wrapper reads the metric manifest path, creates a mirrored logs/slurm/.../
dynamic_workers/ directory, and resubmits itself with matching --output/--error
paths. Inside Slurm the job payload receives exactly one positional argument:
<metric-manifest.csv>.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${SCRIPT_PROJECT_ROOT}/scripts/lib/env.sh" ]]; then
  if [[ -n "${PROJECT_ROOT:-}" ]]; then
    SCRIPT_PROJECT_ROOT="${PROJECT_ROOT}"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/lib/env.sh" ]]; then
    SCRIPT_PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  fi
fi
source "${SCRIPT_PROJECT_ROOT}/scripts/lib/env.sh"
pdcash_load_dotenv "${PDCASH_DOTENV_PATH:-${SCRIPT_PROJECT_ROOT}/.env}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_PROJECT_ROOT}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/slurm}"
export PROJECT_ROOT LOG_ROOT
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
STATE_DIR="${DYNAMIC_METRIC_STATE_DIR:-}"
RESULTS_DIR="${DYNAMIC_METRIC_RESULTS_DIR:-}"
WORKER_WALLTIME_SECONDS="${WORKER_WALLTIME_SECONDS:-14400}"
SAFETY_MARGIN_SECONDS="${SAFETY_MARGIN_SECONDS:-600}"
MAX_RUNS_PER_WORKER="${MAX_RUNS_PER_WORKER:-}"
RECLAIM_STALE_AFTER_SECONDS="${RECLAIM_STALE_AFTER_SECONDS:-}"
RETRY_FAILED="${RETRY_FAILED:-0}"
RETRY_FAILED_ONLY="${RETRY_FAILED_ONLY:-0}"
RETRY_FAILED_WITH_MORE_MEMORY="${RETRY_FAILED_WITH_MORE_MEMORY:-0}"
ISOLATE_RUNS="${ISOLATE_RUNS:-1}"
MAX_ABNORMAL_ATTEMPTS="${MAX_ABNORMAL_ATTEMPTS:-3}"
HEARTBEAT_INTERVAL_SECONDS="${HEARTBEAT_INTERVAL_SECONDS:-60}"
# Leave the per-row metric_timeout_seconds manifest column authoritative by
# default. Set this to a positive number only when every row should use an
# explicit submission-level override; 0 also means no override.
METRIC_TIMEOUT_SECONDS="${METRIC_TIMEOUT_SECONDS:-}"
# Internal metric workers per Slurm job. Defaults to SLURM_CPUS_PER_TASK so a
# multi-CPU allocation can evaluate several independent metric rows at once.
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

resolve_manifest_log_dir() {
  local manifest_path="$1"
  local python_bin="${PYTHON:-python}"
  command -v "${python_bin}" >/dev/null 2>&1 || fail "Python executable not found: ${python_bin}"
  "${python_bin}" - "${manifest_path}" <<'PY'
from __future__ import annotations

import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    if "log_dir" not in fieldnames:
        raise SystemExit("metric manifest must contain a log_dir column")
    rows = list(reader)

log_dirs = {
    (row.get("log_dir") or "").strip()
    for row in rows
    if any((value or "").strip() for value in row.values())
}
if not log_dirs:
    raise SystemExit("metric manifest has no log_dir values")
if "" in log_dirs:
    raise SystemExit("metric manifest rows must define a non-empty log_dir")
if len(log_dirs) != 1:
    configured = ", ".join(repr(value) for value in sorted(log_dirs))
    raise SystemExit(f"metric manifest rows define inconsistent log_dir values: {configured}")
print(next(iter(log_dirs)))
PY
}

resolve_worker_log_dir() {
  local manifest_path="$1"
  local fallback_log_dir="$2"
  local python_bin="${PYTHON:-python}"
  command -v "${python_bin}" >/dev/null 2>&1 || fail "Python executable not found: ${python_bin}"
  "${python_bin}" - "${manifest_path}" "${fallback_log_dir}" <<'PY'
from __future__ import annotations

from pathlib import Path
import re
import sys


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "manifest"


manifest = Path(sys.argv[1])
fallback_log_dir = Path(sys.argv[2])
parts = manifest.parts
marker = ("experiments", "manifests")
for index in range(len(parts) - len(marker) + 1):
    if parts[index:index + len(marker)] != marker:
        continue
    suffix = list(parts[index + len(marker):])
    if suffix:
        suffix[-1] = Path(suffix[-1]).stem
        if (
            len(suffix) >= 3
            and suffix[1] == "metrics"
            and suffix[-1].endswith("_metrics")
        ):
            profile = suffix[-1][: -len("_metrics")]
            suffix = [suffix[0], "model", *suffix[2:-1], "metrics", profile]
        mirrored = Path("logs/slurm") / Path(*(safe_path_part(part) for part in suffix))
        print((mirrored / "dynamic_workers").as_posix())
        break
else:
    print((fallback_log_dir / "dynamic_workers").as_posix())
PY
}

submit_with_manifest_logs() {
  [[ $# -ge 1 ]] || {
    usage
    fail "missing metric manifest path"
  }
  local manifest_path="${!#}"
  local -a sbatch_args=()
  if (( $# > 1 )); then
    local sbatch_arg_count=$(( $# - 1 ))
    sbatch_args=("${@:1:sbatch_arg_count}")
  fi

  local manifest_abs
  if [[ "${manifest_path}" = /* ]]; then
    manifest_abs="${manifest_path}"
  else
    manifest_abs="${PROJECT_ROOT}/${manifest_path}"
  fi
  [[ -f "${manifest_abs}" ]] || fail "metric manifest does not exist: ${manifest_path}"

  local manifest_log_dir
  manifest_log_dir="$(resolve_manifest_log_dir "${manifest_abs}")"
  local worker_log_dir
  worker_log_dir="$(resolve_worker_log_dir "${manifest_abs}" "${manifest_log_dir}")"
  local worker_log_dir_abs
  worker_log_dir_abs="$(pdcash_resolve_log_path "${worker_log_dir}")"
  mkdir -p "${worker_log_dir_abs}"
  if [[ -n "${STATE_DIR}" ]]; then
    mkdir -p "${STATE_DIR}"
  fi

  echo "MANIFEST_PATH=${manifest_abs}"
  echo "Metric manifest log directory=${manifest_log_dir}"
  echo "Dynamic metric worker log directory=${worker_log_dir}"

  local sbatch_bin="${SBATCH_BIN:-sbatch}"
  local -a submit_cmd=(
    "${sbatch_bin}"
    "${sbatch_args[@]}"
    "--output=${worker_log_dir_abs}/%A_%a.out"
    "--error=${worker_log_dir_abs}/%A_%a.err"
    "${SCRIPT_PATH}"
    "${manifest_path}"
  )
  printf 'Submitting:'
  printf ' %q' "${submit_cmd[@]}"
  printf '\n'
  "${submit_cmd[@]}"
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  submit_with_manifest_logs "$@"
  exit 0
fi

MANIFEST_PATH="${1:-}"
[[ -n "${MANIFEST_PATH}" ]] || fail "dynamic metric worker payload requires <metric-manifest.csv>"
[[ $# -eq 1 ]] || fail "dynamic metric worker payload accepts only <metric-manifest.csv>"

# Cap each row subprocess so runaway metrics fail with a catchable MemoryError
# (clean failed JSON) before the cgroup OOM killer fires. With NUM_WORKERS
# workers sharing the allocation the auto-derived cap is split evenly so one
# metric run cannot starve its siblings. Set CHILD_MEMORY_LIMIT_MB explicitly
# (treated as a per-run value) to override, or to 0 to disable.
CHILD_MEMORY_LIMIT_MB="${CHILD_MEMORY_LIMIT_MB:-}"
if [[ -z "${CHILD_MEMORY_LIMIT_MB}" ]]; then
  ALLOCATION_MEMORY_MB=""
  if [[ -n "${SLURM_MEM_PER_NODE:-}" ]]; then
    ALLOCATION_MEMORY_MB="${SLURM_MEM_PER_NODE}"
  elif [[ -n "${SLURM_MEM_PER_CPU:-}" ]]; then
    ALLOCATION_MEMORY_MB="$(( SLURM_MEM_PER_CPU * ${SLURM_CPUS_PER_TASK:-1} ))"
  fi
  if [[ -n "${ALLOCATION_MEMORY_MB}" ]]; then
    if [[ "${NUM_WORKERS}" -gt 1 ]]; then
      CHILD_MEMORY_LIMIT_MB="$(( ALLOCATION_MEMORY_MB / NUM_WORKERS ))"
    else
      CHILD_MEMORY_LIMIT_MB="${ALLOCATION_MEMORY_MB}"
    fi
  fi
fi

# Pin numeric-library threads so NUM_WORKERS independent metric evaluations
# share the CPUs cleanly. The Python pool applies this too; exporting here
# covers anything imported before the pool starts.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

if [[ ! -d "${PROJECT_ROOT}/src" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/run_dynamic_metric_worker.py" ]]; then
  fail "PROJECT_ROOT does not look like the process-mining-cash repository: ${PROJECT_ROOT}"
fi

cd "${PROJECT_ROOT}"
if [[ -n "${STATE_DIR}" ]]; then
  mkdir -p "${STATE_DIR}"
fi

if hostname | grep -qi "madeira"; then
  fail "Refusing to run metrics on login node madeira. Submit this script from the shell instead."
fi

if [[ "${MANIFEST_PATH}" = /* ]]; then
  MANIFEST_ABS="${MANIFEST_PATH}"
else
  MANIFEST_ABS="${PROJECT_ROOT}/${MANIFEST_PATH}"
fi
[[ -f "${MANIFEST_ABS}" ]] || fail "metric manifest does not exist: ${MANIFEST_PATH}"

[[ -f "${VENV_PATH}/bin/activate" ]] || fail "virtual environment not found: ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MANIFEST_PATH=${MANIFEST_ABS}"
echo "STATE_DIR=${STATE_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "WORKER_WALLTIME_SECONDS=${WORKER_WALLTIME_SECONDS}"
echo "SAFETY_MARGIN_SECONDS=${SAFETY_MARGIN_SECONDS}"
echo "MAX_RUNS_PER_WORKER=${MAX_RUNS_PER_WORKER}"
echo "RECLAIM_STALE_AFTER_SECONDS=${RECLAIM_STALE_AFTER_SECONDS}"
echo "RETRY_FAILED=${RETRY_FAILED}"
echo "RETRY_FAILED_ONLY=${RETRY_FAILED_ONLY}"
echo "RETRY_FAILED_WITH_MORE_MEMORY=${RETRY_FAILED_WITH_MORE_MEMORY}"
echo "ISOLATE_RUNS=${ISOLATE_RUNS}"
echo "CHILD_MEMORY_LIMIT_MB=${CHILD_MEMORY_LIMIT_MB}"
echo "MAX_ABNORMAL_ATTEMPTS=${MAX_ABNORMAL_ATTEMPTS}"
echo "HEARTBEAT_INTERVAL_SECONDS=${HEARTBEAT_INTERVAL_SECONDS}"
echo "METRIC_TIMEOUT_SECONDS=${METRIC_TIMEOUT_SECONDS}"
echo "NUM_WORKERS=${NUM_WORKERS}"

worker_args=(
  --manifest "${MANIFEST_ABS}"
  --worker-walltime-seconds "${WORKER_WALLTIME_SECONDS}"
  --safety-margin-seconds "${SAFETY_MARGIN_SECONDS}"
)
if [[ -n "${STATE_DIR}" ]]; then
  worker_args+=(--state-dir "${STATE_DIR}")
fi
if [[ -n "${RESULTS_DIR}" ]]; then
  worker_args+=(--results-dir "${RESULTS_DIR}")
fi
if [[ -n "${MAX_RUNS_PER_WORKER}" ]]; then
  worker_args+=(--max-runs-per-worker "${MAX_RUNS_PER_WORKER}")
fi
if [[ -n "${RECLAIM_STALE_AFTER_SECONDS}" ]]; then
  worker_args+=(--reclaim-stale-after-seconds "${RECLAIM_STALE_AFTER_SECONDS}")
fi
if [[ "${RETRY_FAILED}" == "1" || "${RETRY_FAILED}" == "true" ]]; then
  worker_args+=(--retry-failed)
fi
if [[ "${RETRY_FAILED_ONLY}" == "1" || "${RETRY_FAILED_ONLY}" == "true" ]]; then
  worker_args+=(--retry-failed-only)
fi
if [[ "${RETRY_FAILED_WITH_MORE_MEMORY}" == "1" || "${RETRY_FAILED_WITH_MORE_MEMORY}" == "true" ]]; then
  worker_args+=(--retry-failed-with-more-memory)
fi
if [[ "${ISOLATE_RUNS}" == "0" || "${ISOLATE_RUNS}" == "false" ]]; then
  worker_args+=(--no-isolate-runs)
fi
if [[ -n "${CHILD_MEMORY_LIMIT_MB}" && "${CHILD_MEMORY_LIMIT_MB}" != "0" ]]; then
  worker_args+=(--child-memory-limit-mb "${CHILD_MEMORY_LIMIT_MB}")
fi
worker_args+=(--max-abnormal-attempts "${MAX_ABNORMAL_ATTEMPTS}")
worker_args+=(--heartbeat-interval-seconds "${HEARTBEAT_INTERVAL_SECONDS}")
if [[ -n "${METRIC_TIMEOUT_SECONDS}" && "${METRIC_TIMEOUT_SECONDS}" != "0" ]]; then
  worker_args+=(--metric-timeout-seconds "${METRIC_TIMEOUT_SECONDS}")
fi
worker_args+=(--num-workers "${NUM_WORKERS}")

"${PYTHON_BIN}" scripts/run_dynamic_metric_worker.py "${worker_args[@]}"
