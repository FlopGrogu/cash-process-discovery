#!/usr/bin/env bash
# Single-row compatibility wrapper around the generic row runner.
#SBATCH --job-name=process-discovery
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

MANIFEST_PATH="${1:?Usage: sbatch slurm/run_single.sh <manifest.csv> [row-index]}"
ROW_INDEX="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-${ROW_INDEX}}"
exec "${SCRIPT_DIR}/run_manifest_row.sh" "${MANIFEST_PATH}"
