#!/usr/bin/env bash
# Generic GEDI log-synthesis payload: one array task = one targets.csv row.
# Resource settings belong on the sbatch command or in
# scripts/submit_gedi_targets_slurm.sh, not in this row runner.
#SBATCH --job-name=gedi-target
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

TARGETS_PATH="${1:-}"
[[ -n "${TARGETS_PATH}" ]] || fail "Usage: sbatch --array=0-N slurm/run_gedi_target.sh <targets.csv>"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "SLURM_ARRAY_TASK_ID is not set"
GEDI_ROW_OFFSET="${GEDI_ROW_OFFSET:-0}"
[[ "${GEDI_ROW_OFFSET}" =~ ^[0-9]+$ ]] || fail "GEDI_ROW_OFFSET must be a nonnegative integer"
ACTUAL_ROW=$((GEDI_ROW_OFFSET + SLURM_ARRAY_TASK_ID))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/slurm}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
GEDI_PYTHON="${GEDI_PYTHON:-${PROJECT_ROOT}/.venv-gedi/bin/python}"
GEDI_OUTPUT_ROOT="${GEDI_OUTPUT_ROOT:-${DATA_ROOT}/synthetic/gedi}"
GEDI_RESULTS_DIR="${GEDI_RESULTS_DIR:-${RESULTS_ROOT}/gedi}"

if [[ "${PROJECT_ROOT}" == "/var/lib/slurm/slurmd" || "${PROJECT_ROOT}" == /var/lib/slurm/slurmd/* ]]; then
  fail "PROJECT_ROOT points inside the Slurm spool directory: ${PROJECT_ROOT}"
fi
if [[ ! -d "${PROJECT_ROOT}/src" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/run_gedi_target.py" ]] \
  || [[ ! -f "${PROJECT_ROOT}/slurm/run_gedi_target.sh" ]]; then
  fail "PROJECT_ROOT does not look like the cash-process-discovery training_data project: ${PROJECT_ROOT}"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}" "${GEDI_RESULTS_DIR}"

if hostname | grep -qi "madeira"; then
  fail "Refusing to run GEDI synthesis on login node madeira. Submit this script with sbatch."
fi

TARGETS_ABS="$(pdcash_resolve_data_path "${TARGETS_PATH}")"
[[ -f "${TARGETS_ABS}" ]] || fail "targets file does not exist: ${TARGETS_PATH}"
[[ -x "${GEDI_PYTHON}" ]] || fail "GEDI sidecar interpreter not found: ${GEDI_PYTHON}. Run 'make env' or export GEDI_PYTHON."

[[ -f "${VENV_PATH}/bin/activate" ]] || fail "virtual environment not found: ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

export PROJECT_ROOT DATA_ROOT RESULTS_ROOT LOG_ROOT GEDI_PYTHON
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "GEDI_PYTHON=${GEDI_PYTHON}"
echo "GEDI_OUTPUT_ROOT=${GEDI_OUTPUT_ROOT}"
echo "GEDI_RESULTS_DIR=${GEDI_RESULTS_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "GEDI_ROW_OFFSET=${GEDI_ROW_OFFSET}"
echo "ACTUAL_GEDI_ROW_INDEX=${ACTUAL_ROW}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "TARGETS_PATH=${TARGETS_PATH}"

"${PYTHON_BIN}" - "${TARGETS_ABS}" "${ACTUAL_ROW}" <<'PY'
from __future__ import annotations

import csv
import sys

targets_path = sys.argv[1]
row_index = int(sys.argv[2])

with open(targets_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if row_index < 0 or row_index >= len(rows):
    raise SystemExit(
        f"Actual GEDI row index {row_index} is outside targets range 0-{len(rows) - 1}"
    )

row = rows[row_index]
print(f"SELECTED_ROW_INDEX={row_index}")
for key in ["target_id", "band", "concurrency", "noise_level", "feasible"]:
    value = row.get(key)
    if value:
        print(f"SELECTED_{key.upper()}={value}")
PY

"${PYTHON_BIN}" scripts/run_gedi_target.py \
  --targets "${TARGETS_ABS}" \
  --row-index "${ACTUAL_ROW}" \
  --output-root "${GEDI_OUTPUT_ROOT}" \
  --results-dir "${GEDI_RESULTS_DIR}" \
  --gedi-python "${GEDI_PYTHON}" \
  --base-seed "${GEDI_BASE_SEED:-2024}" \
  --n-trials "${GEDI_N_TRIALS:-50}" \
  --max-attempts "${GEDI_MAX_ATTEMPTS:-3}" \
  --timeout-seconds "${GEDI_TIMEOUT_SECONDS:-1800}"
