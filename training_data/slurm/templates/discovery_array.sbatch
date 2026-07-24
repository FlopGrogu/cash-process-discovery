#!/usr/bin/env bash
# Template for direct sbatch use. Prefer scripts/submit_manifest_slurm.sh when
# you want manifest validation and uniform discovery defaults.
#SBATCH --job-name=process-discovery
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G

set -euo pipefail

# Example:
# sbatch --partition=minor --time=24:00:00 --mem=16G --array=0-999 \
#   slurm/templates/discovery_array.sbatch experiments/manifests/test_01_inductive_miner.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../run_manifest_row.sh" "$@"
