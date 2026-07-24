# Deprecated hyperparameter-optimization tooling

HPO is a deprecated leftover retained only for compatibility with earlier v6
runs. It is not part of the supported workflow and is never required by data
generation, `make manifests`, `make manifests-all`, `make check`, release
validation, or submission.

The six legacy configs reference the canonical 215-log
real/augmented/synthetic inventory. The commands below are preserved only for
reproducing an earlier HPO run.

Install the frozen legacy dependencies before using these commands:

```bash
.venv/bin/python -m pip install -r requirements-hpo.txt
```

Generate a study manifest:

```bash
.venv/bin/python scripts/generate_hpo_studies.py \
  --config configs/experiments/v6/hpo/heuristic_plusplus/v1.yaml \
  --output-root build/manifests/v6
```

The result is
`build/manifests/v6/hpo/heuristic_plusplus/v1/studies.csv` with exactly 215
rows. Each row identifies one log/algorithm study, journal, summary, results
directory, and Slurm log directory.

Run locally or submit the legacy `slurm/run_hpo_study.slurm` entry point.
Optuna has a fixed sampler seed and stops at `n_trials`; trial timeouts are
deterministic failure boundaries. Journals make studies resumable. Completed
discovery result hashes are reused.

Export completed trials to an ordinary discovery manifest:

```bash
.venv/bin/python scripts/export_hpo_manifest.py \
  --config configs/experiments/v6/hpo/heuristic_plusplus/v1.yaml
```

The generic `pdcash-generate-manifest` command intentionally rejects HPO
configs because expanding the static search space is not an HPO study.
