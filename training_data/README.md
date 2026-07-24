# Process Discovery CASH — v6 reproducibility artifact

This repository contains the v6 process-discovery benchmark: deterministic
augmentation, feature-space synthesis, experiment manifests, discovery, and
quality-metric evaluation. Python 3.11 and pip are the only required setup
tools. Docker, Apptainer, conda, and uv are not required.

## Quick start

Use Python 3.11.15. The core benchmark and GEDI 1.0.8 need separate
environments because they intentionally pin incompatible NumPy, SciPy, and
PM4Py versions.

```bash
git clone https://github.com/FlopGrogu/cash-process-discovery.git
cd cash-process-discovery/training_data

python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .

python -m venv .venv-gedi
.venv-gedi/bin/python -m pip install -r environments/gedi/requirements.txt
```

`make env` performs the same four installation steps. All commands below call
repository scripts with the core Python interpreter, so activating a virtual
environment is optional.

## XES-first workflow

Experiment manifests always reference the canonical `.xes` or `.xes.gz`
inputs. Preprocessed Parquet files are optional local caches: creating or
deleting them does not change a manifest or its experiment identity.

The following sequence covers the complete local workflow:

```bash
# Verify manually acquired real inputs.
.venv/bin/python scripts/verify_inputs.py --all

# Generate deterministic augmented XES logs.
.venv/bin/python scripts/augment_logs.py --all

# Design targets and generate deterministic GEDI synthetic XES logs.
.venv/bin/python scripts/generate_feature_space_logs.py \
  --mode design --n-targets 200 --seed 2024 --compute-anchor
.venv/bin/python scripts/generate_feature_space_logs.py \
  --mode main --n-targets 200 --seed 2024 --n-trials 50 \
  --gedi-python .venv-gedi/bin/python

# Generate an XES-backed discovery manifest and run a row.
.venv/bin/python scripts/generate_experiment_manifest.py \
  --config configs/experiments/v6/baseline/alpha_classic/v1.yaml \
  --output build/alpha-classic.csv
.venv/bin/python scripts/run_discovery.py \
  --manifest build/alpha-classic.csv --row-index 0

# Derive and run the corresponding metric manifest.
.venv/bin/python scripts/generate_metric_manifest.py \
  --source-manifest build/alpha-classic.csv \
  --output build/alpha-classic-metrics.csv \
  --output-root results/local/metrics/alpha-classic
.venv/bin/python scripts/run_metric.py \
  --manifest build/alpha-classic-metrics.csv --row-index 0

# Optional: build validated Parquet caches without changing either manifest.
.venv/bin/python scripts/preprocess_logs.py --manifest build/alpha-classic.csv
```

`scripts/preprocess_logs.py` also accepts `--logs PATH [PATH ...]` and
`--config PATH`. Each cache is bound to the source path, size, modification
time, and SHA-256. Discovery and metric workers use a valid cache
transparently and parse the XES source when no current cache exists.

## Input data and generated inventory

Acquire the 21 real event logs listed in
[`configs/datasets/processmining_org.yaml`](configs/datasets/processmining_org.yaml).
Place each file at the exact `data/raw/...` destination in that catalog, which
also pins the official source, access terms, size, and SHA-256. Do not rename
or decompress the files.

The full data workflow is resumable and produces:

| Stage | Expected count |
|---|---:|
| manually verified real logs | 21 |
| accepted augmented logs | 77 |
| deterministic GEDI targets | 200 |
| accepted GEDI synthetic logs | 117 |
| total real, augmented, and synthetic logs | 215 |

Accepted augmented logs are written to `data/augmented/logs/`; accepted GEDI
synthetic logs are written to `data/synthetic/gedi/logs/`.

Real inputs, the external JAR, generated data, and large manifests are
intentionally ignored by Git and are not included in a normal clone. Existing
valid outputs are reused unless the relevant command receives `--overwrite` or
`--force`.

## Manifests, discovery, and metrics

Generate the 30 primary v6 manifests—10 real-log baselines, 10 augmented
explorations, and 10 synthetic explorations—and check their committed receipt
ledger:

```bash
.venv/bin/python scripts/generate_v6_manifests.py \
  --primary --output-root build/manifests/v6
.venv/bin/python scripts/manifest_receipts.py \
  --check --scope primary --manifest-root build/manifests/v6
```

`make manifests` runs those commands. The default-configuration survey and HPO
remain available through the explicit `make manifests-survey` and
`make manifests-hpo` targets; `make manifests-all` verifies the complete
46-config ledger.

Every discovery manifest contains portable `data/`, `results/`, and
`logs/slurm/` paths. The required event-log fields are `log_path` and
`test_log_path`; both point to XES sources. Legacy artifact columns remain in
the CSV schema for backward compatibility but are empty in newly generated
manifests.

Metric manifests are deterministic transformations of discovery manifests.
They preserve `test_log_path`, resolve the exported model at runtime, and keep
failed or missing discovery rows visible in the evaluation denominator.

The Python-backed algorithms need no external runtime. Split Miner remains an
optional exception: it requires Java 8 and the separately acquired
`data/external/split-miner-1.7.1-all.jar`. See
[`docs/algorithms.md`](docs/algorithms.md).

## Verification

Run the complete non-external acceptance suite:

```bash
make check
```

This runs Ruff, all non-external tests, exact manifest receipt verification,
and the submission audit using `.venv/bin/python`. It does not invoke uv or a
container runtime.

Further details:

- [`docs/setup.md`](docs/setup.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/data.md`](docs/data.md)
- [`docs/experiments.md`](docs/experiments.md)
- [`docs/metrics.md`](docs/metrics.md)
- Optional: [`docs/hpo.md`](docs/hpo.md)
- [`docs/cluster.md`](docs/cluster.md)

Third-party provenance is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
