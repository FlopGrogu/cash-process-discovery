# Slurm execution

Only the v6 workflow is supported. Clone the repository onto a filesystem
visible to every compute node, run `make env`, and generate manifests on the
login node. Do not parse full logs, preprocess the full inventory, run GEDI, or
run discovery on a login node.

## Configure portable roots

Copy `.env.example` to the untracked `.env` file and set cluster-local roots:

```bash
cp .env.example .env
```

```dotenv
PROJECT_ROOT=/shared/projects/cash-process-discovery/training_data
DATA_ROOT=/shared/data/process-mining-cash-v6
RESULTS_ROOT=/scratch/process-mining-cash-v6
LOG_ROOT=/scratch/process-mining-cash-v6/slurm
SPLIT_MINER_JAR=/shared/tools/split-miner-1.7.1-all.jar
GEDI_PYTHON=/shared/projects/cash-process-discovery/training_data/.venv-gedi/bin/python
DISCOVERY_PARTITION=CPU
DISCOVERY_QOS=minor_student
DISCOVERY_TIME=24:00:00
DISCOVERY_MEM=16G
METRIC_PARTITION=CPU
METRIC_QOS=minor_student
GEDI_PARTITION=CPU
GEDI_QOS=minor_student
```

Real environment variables and command-line flags override `.env`. The
submission wrappers print the resolved roots and requested resources before
calling `sbatch`. Partition names, QOS names, walltime, memory, and concurrency
remain site policy; inspect them with `sinfo` and `scontrol`.

Verify the 21 catalog files and Split Miner JAR before submitting work:

```bash
.venv/bin/python scripts/verify_inputs.py --all
.venv/bin/python scripts/validate_splitminer_java.py
```

## Discovery: dynamic workers for scale

Dynamic workers are the recommended mode for large manifests. Each Slurm array
task is a pull worker that repeatedly claims uncompleted manifest rows until its
walltime is nearly exhausted. The array size therefore controls the number of
worker allocations, not the number of manifest rows:

```bash
bash slurm/run_dynamic_manifest.slurm \
  --partition=CPU --qos=minor_student \
  --array=0-3 \
  build/manifests/v6/model/baseline/alpha_classic/v1.csv
```

That command uses the uniform discovery defaults: a 24-hour discovery timeout,
24-hour Slurm walltime, 16G of memory, one CPU, and therefore one worker.
`NUM_WORKERS` defaults to `SLURM_CPUS_PER_TASK`. For four simultaneous workers,
request `--cpus-per-task=4 --mem=64G`; memory is shared and the automatic
per-run child limit remains 16 GiB. Keep `NUM_WORKERS` at or below the requested
CPUs and allocate 16G for each simultaneous run.

Important worker controls are exported when submitting:

- `WORKER_WALLTIME_SECONDS` and `SAFETY_MARGIN_SECONDS` control graceful exit;
- `DYNAMIC_STATE_DIR` and `DYNAMIC_RESULTS_DIR` override derived state/results;
- `MAX_RUNS_PER_WORKER` limits work performed by one worker;
- `RECLAIM_STALE_AFTER_SECONDS` enables explicit stale-claim recovery;
- `RETRY_FAILED`, `RETRY_FAILED_ONLY`, and
  `RETRY_FAILED_WITH_MORE_MEMORY` select failed-row retry behavior;
- `CHILD_MEMORY_LIMIT_MB` overrides the automatically divided memory limit.

When overriding `--time`, set `WORKER_WALLTIME_SECONDS` to the same allocation
length in seconds. The default pair is `24:00:00` and `86400`.

Successful hash-addressed results are skipped. Interrupted workers leave
claim/state records, and resubmitting the same command continues unfinished
rows. Do not enable stale-claim recovery until the original allocation is known
to have ended.

## Discovery: one row per job

Use a static array for smoke tests, strict row isolation, heterogeneous
resubmission, or simple one-task/one-result accounting:

```bash
bash scripts/submit_manifest_slurm.sh \
  --manifest build/manifests/v6/model/baseline/alpha_classic/v1.csv \
  --row 0 --dry-run

bash scripts/submit_manifest_slurm.sh \
  --manifest build/manifests/v6/model/baseline/alpha_classic/v1.csv \
  --all-rows
```

One array task maps to exactly one zero-based manifest row. `--all-rows`
automatically chunks arrays larger than `--max-array-tasks`; `--array` plus
`--row-offset` submits an explicit chunk. Use `%N` in an array specification,
such as `--array=0-20%8`, to cap concurrent tasks. Always inspect a
`--dry-run` before a large static submission. Every primary baseline manifest
has 21 rows; explore manifests can be substantially larger. Static rows use
the same 24-hour walltime, 16G memory, and one CPU defaults as a one-worker
dynamic job.

## Optional survey overview

The default-configuration survey runs 10 algorithm variants on all 21 real
logs using declared default parameters. Its 210 discovery records describe
runtime and failure behavior in the current environment; they are not a
performance ranking, scalability study, or basis for cross-system runtime
comparisons.

After all survey manifests finish, aggregate their structured results:

```bash
set -a
source .env
set +a
.venv/bin/python scripts/aggregate_results.py \
  --results-root "$RESULTS_ROOT/cluster/v6/model/default_run_survey" \
  --output "$RESULTS_ROOT/aggregated/v6/default_run_survey.csv"
```

The overview CSV retains `status`, `runtime_seconds`, `error_message`, detailed
timing and memory fields, and captured Slurm metadata. Preserve it with the
cluster execution evidence.

## Optional quality metrics

Quality evaluation is separate from the runtime and failure survey. Derive a
metric manifest from a non-survey discovery manifest:

```bash
.venv/bin/python scripts/generate_metric_manifest.py \
  --source-manifest build/manifests/v6/model/baseline/alpha_classic/v1.csv \
  --output build/manifests/v6/metrics/baseline/alpha_classic/v1/token_metrics.csv \
  --output-root results/cluster/v6/metrics/baseline/alpha_classic/v1/token
```

For scale, use the resumable metric pull pool:

```bash
bash slurm/run_dynamic_metric_manifest.slurm \
  --partition=CPU --qos=minor_student \
  --cpus-per-task=4 --mem=32G --time=04:00:00 \
  --array=0-3 \
  build/manifests/v6/metrics/baseline/alpha_classic/v1/token_metrics.csv
```

For one row per array task:

```bash
bash scripts/submit_metric_manifest_slurm.sh \
  --manifest build/manifests/v6/metrics/baseline/alpha_classic/v1/token_metrics.csv \
  --profile token --row 0 --dry-run

bash scripts/submit_metric_manifest_slurm.sh \
  --manifest build/manifests/v6/metrics/baseline/alpha_classic/v1/token_metrics.csv \
  --profile token --all-rows
```

Metric workers have the same walltime, worker-count, memory-division, claim,
retry, and resubmission model as discovery workers. Their optional state and
result overrides are `DYNAMIC_METRIC_STATE_DIR` and
`DYNAMIC_METRIC_RESULTS_DIR`.

## GEDI row semantics

GEDI is part of primary synthetic-log generation and uses one deterministic
target per array task:

```bash
bash scripts/submit_gedi_targets_slurm.sh \
  --targets data/synthetic/gedi/targets.csv \
  --all-rows --dry-run

bash scripts/submit_gedi_targets_slurm.sh \
  --targets data/synthetic/gedi/targets.csv \
  --all-rows
```

The wrapper validates `targets.csv` and `anchor_features.csv`, exports the
fixed seed/trial controls, and writes one result record per target. Re-submit
missing or failed row ranges, then aggregate once all 200 targets have records.

## Deprecated HPO compatibility

This legacy workflow is not required for validation or submission. When
reproducing an earlier HPO run, its study manifests contain 215 rows. One
Slurm array task owns one study row (one log/algorithm pair); `NUM_WORKERS`
controls concurrent Optuna trials inside that study:

```bash
WORKER_WALLTIME_SECONDS=28800 bash slurm/run_hpo_study.slurm \
  --partition=CPU --qos=minor_student \
  --cpus-per-task=4 --mem=32G --time=08:00:00 \
  --array=0-214 \
  build/manifests/v6/hpo/heuristic_plusplus/v1/studies.csv
```

The journal and existing result hashes make a study resumable. Match
`WORKER_WALLTIME_SECONDS` to the allocation walltime and retain the default
safety margin. This section is not part of the primary cluster workflow.

## Monitoring and evidence

Use `squeue` for live state and `sacct` for resource evidence:

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,NodeList%30
```

Worker logs are below the resolved `LOG_ROOT`, mirrored by v6 manifest path.
Preserve the submission command, Slurm job/array IDs, stdout/stderr, resource
summary, retry settings, and final validation receipts for the release record.

## Fresh-root release validation

Run the release workflow with new `DATA_ROOT`, `RESULTS_ROOT`, and `LOG_ROOT`
directories. Do not reuse the stale ignored artifacts from another code
revision.

1. Stage the 21 immutable catalog files and Split Miner JAR at their configured
   paths; run `make env` and `make verify-inputs`.
2. In a compute allocation, run
   `.venv/bin/python scripts/augment_logs.py --all`.
3. In a compute allocation, design the anchor and targets:

   ```bash
   .venv/bin/python scripts/generate_feature_space_logs.py \
     --mode design --n-targets 200 --seed 2024 --compute-anchor
   ```

4. Submit all GEDI target rows with the wrapper above. When every target has a
   result, aggregate:

   ```bash
   .venv/bin/python scripts/generate_feature_space_logs.py \
     --mode aggregate --n-targets 200 --seed 2024
   ```

5. Optionally build local Parquet caches from XES-backed manifests:

   ```bash
   .venv/bin/python scripts/preprocess_logs.py \
     --manifest build/manifests/v6/model/baseline/alpha_classic/v1.csv
   ```

6. Require `.venv/bin/python scripts/verify_generated_data.py --json` to report
   `21/77/200/117/215`, then run `make manifests` and `make check`.
7. Run one real GEDI target plus one Split Miner discovery smoke row.
8. Separately run one baseline Split Miner metric row and the tiny integration
   test specified in [external-validation.md](external-validation.md).
9. Return the generated-data JSON receipt, manifest-receipt result, smoke
   outcomes, Slurm resource evidence, and final archive checksum with the
   submission.

The survey may be run afterwards with `make manifests-survey`. Deprecated HPO
tooling is retained only for compatibility and is not a release prerequisite.
