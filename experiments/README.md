# Experiments

Experiment YAML files live in `configs/experiments/`.

Generated manifests are CSV files with one independent job per row. Slurm array
jobs use the row index selected by `SLURM_ARRAY_TASK_ID`.

The `experiments/generated/` directory is for generated helper files and is
ignored by Git except for placeholders.
