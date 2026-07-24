# v6 experiment interfaces

Only `configs/experiments/v6/` is supported:

| Stage | Configs | Workflow | Purpose |
|---|---:|---|---|
| `baseline` | 10 | primary | one configuration on 21 real logs |
| `explore` | 10 | primary | real plus accepted augmented exploration |
| `explore_synthetic` | 10 | primary | accepted GEDI synthetic exploration |
| `hpo` | 6 | optional | per-log Optuna studies |
| `default_run_survey` | 10 | optional | default configurations on 21 real logs |

Generic generation accepts ordinary configs only:

```bash
.venv/bin/python scripts/generate_experiment_manifest.py \
  --config configs/experiments/v6/baseline/alpha_classic/v1.yaml \
  --output build/alpha-classic.csv
```

Passing an HPO config fails immediately with the correct routing command.

Generate the 30 primary baseline and explore manifests without changing the
repository:

```bash
.venv/bin/python scripts/generate_v6_manifests.py \
  --primary --output-root build/manifests/v6
```

Check primary receipts in an automatic temporary directory:

```bash
.venv/bin/python scripts/manifest_receipts.py --check --scope primary
```

The optional scopes are explicit:

```bash
make manifests-survey
make manifests-hpo
make manifests-all
```

`--all` remains available on `generate_v6_manifests.py` for the 40 ordinary
primary-plus-survey manifests. HPO study generation continues to use
`generate_hpo_studies.py`.

Manifests contain only portable `data/`, `results/`, and `logs/slurm/` paths.
`log_path` and `test_log_path` are authoritative XES references. Optimized
cache paths and preprocessing fingerprints are not part of new manifest
identities. Scientific result JSON can contain volatile runtime/resource
metadata.

The default-configuration runtime and failure survey has exactly
10 × 21 = 210 discovery rows. It characterizes environment-specific runtime
and failure behavior under declared defaults; it is not a comparative
performance experiment. Large generated CSVs are ignored;
`release/v6-manifest-receipts.json` is committed instead.
