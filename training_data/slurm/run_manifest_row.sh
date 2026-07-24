#!/usr/bin/env bash
# Generic discovery job payload. Resource settings belong on the sbatch command
# or in scripts/submit_manifest_slurm.sh, not in this row runner.
#SBATCH --job-name=process-discovery
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

MANIFEST_PATH="${1:-}"
[[ -n "${MANIFEST_PATH}" ]] || fail "Usage: sbatch --array=0-N slurm/run_manifest_row.sh <manifest.csv>"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "SLURM_ARRAY_TASK_ID is not set"
MANIFEST_ROW_OFFSET="${MANIFEST_ROW_OFFSET:-0}"
[[ "${MANIFEST_ROW_OFFSET}" =~ ^[0-9]+$ ]] || fail "MANIFEST_ROW_OFFSET must be a nonnegative integer"
ACTUAL_ARRAY_TASK_ID=$((MANIFEST_ROW_OFFSET + SLURM_ARRAY_TASK_ID))

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

if [[ "${PROJECT_ROOT}" == "/var/lib/slurm/slurmd" || "${PROJECT_ROOT}" == /var/lib/slurm/slurmd/* ]]; then
  fail "PROJECT_ROOT points inside the Slurm spool directory: ${PROJECT_ROOT}"
fi
if [[ ! -d "${PROJECT_ROOT}/src" ]] \
  || [[ ! -f "${PROJECT_ROOT}/scripts/run_discovery.py" ]] \
  || [[ ! -f "${PROJECT_ROOT}/slurm/run_manifest_row.sh" ]]; then
  fail "PROJECT_ROOT does not look like the process-mining-cash repository: ${PROJECT_ROOT}"
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}"

if hostname | grep -qi "madeira"; then
  fail "Refusing to run discovery on login node madeira. Submit this script with sbatch."
fi

if [[ "${MANIFEST_PATH}" = /* ]]; then
  MANIFEST_ABS="${MANIFEST_PATH}"
else
  MANIFEST_ABS="${PROJECT_ROOT}/${MANIFEST_PATH}"
fi
[[ -f "${MANIFEST_ABS}" ]] || fail "manifest does not exist: ${MANIFEST_PATH}"

[[ -f "${VENV_PATH}/bin/activate" ]] || fail "virtual environment not found: ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

export PROJECT_ROOT DATA_ROOT RESULTS_ROOT LOG_ROOT
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python executable not found: ${PYTHON_BIN}"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "MANIFEST_ROW_OFFSET=${MANIFEST_ROW_OFFSET}"
echo "ACTUAL_MANIFEST_ROW_INDEX=${ACTUAL_ARRAY_TASK_ID}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"

"${PYTHON_BIN}" - "${MANIFEST_ABS}" "${ACTUAL_ARRAY_TASK_ID}" <<'PY'
from __future__ import annotations

import csv
import sys

manifest_path = sys.argv[1]
row_index = int(sys.argv[2])

with open(manifest_path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if row_index < 0 or row_index >= len(rows):
    raise SystemExit(
        f"Actual manifest row index {row_index} is outside manifest range 0-{len(rows) - 1}"
    )

row = rows[row_index]
print(f"SELECTED_ROW_INDEX={row_index}")
for key in [
    "experiment_id",
    "log_id",
    "seed",
    "algorithm_id",
    "algorithm",
    "algorithm_variant",
    "log_dir",
    "output_path",
]:
    value = row.get(key)
    if value:
        print(f"SELECTED_{key.upper()}={value}")
PY

"${PYTHON_BIN}" scripts/run_discovery.py \
  --manifest "${MANIFEST_PATH}" \
  --row-index "${ACTUAL_ARRAY_TASK_ID}"
