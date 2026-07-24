# Slurm entry points

Only the v6 workflow is supported. The complete cluster runbook is in
[`docs/cluster.md`](../docs/cluster.md).

| Entry point | Scheduling unit | Intended use |
|---|---|---|
| `run_dynamic_manifest.slurm` | one resumable pull worker per array task | preferred for large discovery manifests |
| `run_manifest_row.sh` | one discovery manifest row per array task | smoke tests and strict row isolation |
| `run_dynamic_metric_manifest.slurm` | one resumable metric pull worker per array task | preferred for large metric manifests |
| `run_metric_row.sh` | one metric manifest row per array task | smoke tests and strict row isolation |
| `run_hpo_study.slurm` | optional: one HPO study row per array task | parallel trials inside each study |
| `run_gedi_target.sh` | one deterministic GEDI target per array task | synthetic-log generation |
| `run_single.sh` | one selected discovery row | direct one-row helper |
| `templates/discovery_array.sbatch` | site-customizable static array | direct `sbatch` deployments |

Use `scripts/submit_manifest_slurm.sh`,
`scripts/submit_metric_manifest_slurm.sh`, and
`scripts/submit_gedi_targets_slurm.sh` for validated static submissions.
Dynamic and HPO wrappers resubmit themselves with the caller's `sbatch`
options and derived log paths. Discovery submissions default uniformly to a
24-hour algorithm timeout, 24-hour Slurm walltime, 16G of memory, and one CPU
per row or dynamic worker. Metric, HPO, and GEDI stages retain their documented
stage-specific resources.
