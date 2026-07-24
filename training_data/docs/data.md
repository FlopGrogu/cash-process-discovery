# Data generation and preprocessing

## 1. Real inputs

The catalog `configs/datasets/processmining_org.yaml` is authoritative for all
21 real logs. Each entry pins:

- the official landing page and DOI;
- license or repository access terms;
- the exact path below `DATA_ROOT`;
- byte size and SHA-256;
- event schema and companion-file roles.

Place files manually, then run:

```bash
.venv/bin/python scripts/verify_inputs.py --all
```

The verifier performs no network request. Missing, misplaced, wrong-size, or
wrong-hash inputs produce a nonzero exit status. Omit the JAR check only for
non-Split-Miner work with `--no-split-miner`.

## 2. Deterministic augmentation

```bash
.venv/bin/python scripts/augment_logs.py --all
```

The fixed base seed is 1001. A child seed is derived from the parent ID and
canonical augmentation specification. Operators include balanced
variant-aware subsampling, frequency coverage, event noise, and conditional
large/long-log transforms. Rejected children are recorded deterministically.

Outputs:

```text
data/augmented/
├── logs/*.xes.gz
└── manifest.csv
```

Expected accepted children: 77. Every accepted row records parent and output
SHA-256 values. The canonical XES writer fixes trace/event order, XML metadata
and attribute order, UTC timestamp formatting, gzip level/header, and
`mtime=0`.

## 3. Feature anchor and GEDI design

The 48-feature extractor is pinned to `origin/test-anitan` commit `04d91d52`.
Its golden vector is `tests/golden/anitan_tiny_features.json`. NetworkX is a
required dependency.

Build the anchor and 200-target design:

```bash
.venv/bin/python scripts/generate_feature_space_logs.py \
  --mode design --n-targets 200 --seed 2024 --compute-anchor
```

Outputs:

```text
data/synthetic/gedi/
├── real_features.csv
└── targets.csv
```

Target IDs, bands, feasibility records, and values are deterministic. The
`feasible` CSV field is parsed strictly; malformed booleans are rejected.

## 4. GEDI generation

```bash
.venv/bin/python scripts/generate_feature_space_logs.py \
  --mode main \
  --n-targets 200 \
  --seed 2024 \
  --n-trials 50 \
  --gedi-python .venv-gedi/bin/python
```

Each target is optimized sequentially. Python, NumPy, ConfigSpace/SMAC, GEDI,
and subprocess seeds derive from the target/base seed. BLAS thread counts are
fixed to one. Termination is by exactly 50 trials, not elapsed optimizer time;
the subprocess timeout is only a failure boundary.

Outputs:

```text
data/synthetic/gedi/
├── logs/*.xes.gz
├── rejected/*.xes.gz
├── work/
├── manifest.csv
└── coverage.json
```

Expected inventory: 200 targets and 117 accepted logs. The manifest contains
stable seeds/configuration/attainment/checksum data. Runtime timestamps and
worker-local paths are not part of deterministic records.

On Slurm, generate `targets.csv` locally, submit one target per job using
`scripts/submit_gedi_targets_slurm.sh`, then aggregate:

```bash
.venv/bin/python scripts/generate_feature_space_logs.py --mode aggregate
```

Aggregation is idempotent. Missing targets are reported, and unknown stale
target results are not silently accepted.

## 5. Optional XES caches

```bash
.venv/bin/python scripts/preprocess_logs.py \
  --logs data/example/tiny_log.xes

# Or cache every unique XES path referenced by a manifest:
.venv/bin/python scripts/preprocess_logs.py \
  --manifest build/manifests/v6/model/default_run_survey/alpha_classic/v1.csv
```

`preprocess_logs.py` builds optional Zstandard-compressed Parquet caches from
explicit XES paths, experiment configs, or manifests. Manifests continue to
reference their source XES files, and every discovery or metric command works
without a cache. A cache is used only while its source path, size,
modification time, and SHA-256 match its metadata sidecar.

The default cache layout is:

```text
data/processed/log_cache/
├── <log-id>.parquet
└── <log-id>.parquet.json
```

The older `preprocess_event_logs.py` command remains available for release
compatibility and Split Miner projections, but ordinary manifests never
require or identify those algorithm-specific artifacts.

## Inventory gate

Before primary manifest generation or release:

```bash
.venv/bin/python scripts/verify_generated_data.py
# compact machine-readable receipt:
.venv/bin/python scripts/verify_generated_data.py --json
```

The verifier checks the committed expected counts (21 real, 77 augmented, 200
GEDI targets, 117 accepted synthetic, and 215 total event logs), portable
`data/...` output paths, uniqueness, and every generated file against the
`artifact_sha256` in its deterministic manifest. The JSON receipt calls the
combined count `total_event_logs`; HPO is not a data-generation stage. The
verifier also emits a combined generated-artifact receipt hash for the release
record. Generated inputs, manifests, and results are intentionally ignored by
Git.
