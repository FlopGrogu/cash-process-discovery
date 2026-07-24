#!/usr/bin/env bash
#SBATCH --job-name=process-discovery-hpo
#SBATCH --output=logs/slurm/hpo_studies/bootstrap_%j.out
#SBATCH --error=logs/slurm/hpo_studies/bootstrap_%j.err
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
  bash slurm/run_hpo_study.slurm [sbatch options] <studies.csv>

One array task = one HPO study (one log x algorithm pair; a row of the study
manifest written by pdcash-generate-hpo-studies). Inside the allocation,
NUM_WORKERS internal workers (defaults to SLURM_CPUS_PER_TASK) share the
study's Optuna journal and evaluate proposed configurations concurrently.

This wrapper reads the study manifest path, creates a mirrored logs/slurm/...
hpo_workers/ directory, and resubmits itself with matching --output/--error
paths. Inside Slurm the job payload receives exactly one positional argument:
<studies.csv>.

Example (12 studies, 4 parallel trial evaluations per study):

  WORKER_WALLTIME_SECONDS=28800 bash slurm/run_hpo_study.slurm \
    --cpus-per-task=4 --mem=32G \
    --time=08:00:00 --array=0-11 --partition=CPU --qos=minor_student \
    build/manifests/v6/hpo/heuristic_plusplus/v1/studies.csv

Match WORKER_WALLTIME_SECONDS to --time; workers stop asking for new trials
when the remaining walltime no longer fits one trial. Re-submitting the same
array resumes each study from its journal and existing result files.
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
WORKER_WALLTIME_SECONDS="${WORKER_WALLTIME_SECONDS:-14400}"
SAFETY_MARGIN_SECONDS="${SAFETY_MARGIN_SECONDS:-600}"
MAX_TRIALS_PER_WORKER="${MAX_TRIALS_PER_WORKER:-}"
ISOLATE_RUNS="${ISOLATE_RUNS:-1}"
REQUIRE_ARTIFACTS="${REQUIRE_ARTIFACTS:-0}"
# Internal workers per Slurm job: one worker per CPU evaluates one proposed
# configuration at a time. All workers of a job share one Optuna study.
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

resolve_worker_log_dir() {
  local manifest_path="$1"
  local python_bin="${PYTHON:-python}"
  command -v "${python_bin}" >/dev/null 2>&1 || fail "Python executable not found: ${python_bin}"
  "${python_bin}" - "${manifest_path}" <<'PY'
from __future__ import annotations

from pathlib import Path
import re
import sys


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "manifest"


manifest = Path(sys.argv[1])
parts = manifest.parts
marker = ("experiments", "manifests")
for index in range(len(parts) - len(marker) + 1):
    if parts[index:index + len(marker)] != marker:
        continue
    suffix = list(parts[index + len(marker):])
    if suffix:
        suffix[-1] = Path(suffix[-1]).stem
        mirrored = Path("logs/slurm") / Path(
            *(safe_path_part(part) for part in suffix)
        )
        print((mirrored / "hpo_workers").as_posix())
        break
else:
    print((Path("logs/slurm/hpo_studies") / safe_path_part(manifest.stem)).as_posix())
PY
}

submit_with_manifest_logs() {
  [[ $# -ge 1 ]] || {
    usage
    fail "missing study manifest path"
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
  [[ -f "${manifest_abs}" ]] || fail "study manifest does not exist: ${manifest_path}"

  local worker_log_dir
  worker_log_dir="$(resolve_worker_log_dir "${manifest_abs}")"
  local worker_log_dir_abs
  worker_log_dir_abs="$(pdcash_resolve_log_path "${worker_log_dir}")"
  mkdir -p "${worker_log_dir_abs}"

  echo "STUDY_MANIFEST_PATH=${manifest_abs}"
  echo "HPO worker log directory=${worker_log_dir}"

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
[[ -n "${MANIFEST_PATH}" ]] || fail "HPO study payload requires <studies.csv>"
[[ $# -eq 1 ]] || fail "HPO study payload accepts only <studies.csv>"

# Cap each trial subprocess so a runaway configuration fails with a catchable
# MemoryError (clean failed JSON, objective 0.0) before the cgroup OOM killer
# fires. With NUM_WORKERS workers sharing the allocation the auto-derived cap
# is split evenly. Set CHILD_MEMORY_LIMIT_MB explicitly (per-trial value) to
# override, or to 0 to disable.
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

# Pin numeric-library threads so NUM_WORKERS independent trial evaluations
# share the CPUs cleanly; parallelism comes from evaluating many
# configurations, not from threading inside one of them.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

if [[ ! -d "${PROJECT_ROOT}/src" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/run_hpo_study.py" ]]; then
  fail "PROJECT_ROOT does not look like the cash-process-discovery training_data project: ${PROJECT_ROOT}"
fi

cd "${PROJECT_ROOT}"

if hostname | grep -qi "madeira"; then
  fail "Refusing to run HPO trials on login node madeira. Submit this script from the shell instead."
fi

if [[ "${MANIFEST_PATH}" = /* ]]; then
  MANIFEST_ABS="${MANIFEST_PATH}"
else
  MANIFEST_ABS="${PROJECT_ROOT}/${MANIFEST_PATH}"
fi
[[ -f "${MANIFEST_ABS}" ]] || fail "study manifest does not exist: ${MANIFEST_PATH}"

[[ -f "${VENV_PATH}/bin/activate" ]] || fail "virtual environment not found: ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "STUDY_MANIFEST_PATH=${MANIFEST_ABS}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "WORKER_WALLTIME_SECONDS=${WORKER_WALLTIME_SECONDS}"
echo "SAFETY_MARGIN_SECONDS=${SAFETY_MARGIN_SECONDS}"
echo "MAX_TRIALS_PER_WORKER=${MAX_TRIALS_PER_WORKER}"
echo "ISOLATE_RUNS=${ISOLATE_RUNS}"
echo "REQUIRE_ARTIFACTS=${REQUIRE_ARTIFACTS}"
echo "CHILD_MEMORY_LIMIT_MB=${CHILD_MEMORY_LIMIT_MB}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"

study_args=(
  --study-manifest "${MANIFEST_ABS}"
  --study-index "${SLURM_ARRAY_TASK_ID}"
  --worker-walltime-seconds "${WORKER_WALLTIME_SECONDS}"
  --safety-margin-seconds "${SAFETY_MARGIN_SECONDS}"
  --num-workers "${NUM_WORKERS}"
)
if [[ -n "${MAX_TRIALS_PER_WORKER}" ]]; then
  study_args+=(--max-trials-per-worker "${MAX_TRIALS_PER_WORKER}")
fi
if [[ "${ISOLATE_RUNS}" == "0" || "${ISOLATE_RUNS}" == "false" ]]; then
  study_args+=(--no-isolate-runs)
fi
if [[ "${REQUIRE_ARTIFACTS}" == "1" || "${REQUIRE_ARTIFACTS}" == "true" ]]; then
  study_args+=(--require-artifacts)
fi
if [[ -n "${CHILD_MEMORY_LIMIT_MB}" && "${CHILD_MEMORY_LIMIT_MB}" != "0" ]]; then
  study_args+=(--child-memory-limit-mb "${CHILD_MEMORY_LIMIT_MB}")
fi

"${PYTHON_BIN}" scripts/run_hpo_study.py "${study_args[@]}"
