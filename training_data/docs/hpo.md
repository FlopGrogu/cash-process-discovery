# Hyperparameter optimization

HPO is an optional workflow. It is not invoked by data generation,
`make manifests`, `make check`, or the primary release-validation sequence.

Six v6 algorithms use HPO: genetic, heuristic classic, heuristic plus-plus,
ILP, inductive IMF, and Split Miner. Each config references the canonical
215-log real/augmented/synthetic inventory.

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

Run locally or submit `slurm/run_hpo_study.slurm`. Optuna has a fixed sampler
seed and stops at `n_trials`; trial timeouts are deterministic failure
boundaries. Journals make studies resumable. Completed discovery result hashes
are reused.

Export completed trials to an ordinary discovery manifest:

```bash
.venv/bin/python scripts/export_hpo_manifest.py \
  --config configs/experiments/v6/hpo/heuristic_plusplus/v1.yaml
```

The generic `pdcash-generate-manifest` command intentionally rejects HPO
configs because expanding the static search space is not an HPO study.
