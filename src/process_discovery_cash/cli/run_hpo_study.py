from __future__ import annotations

import argparse
import os
import sys

from process_discovery_cash.experiments.worker_pool import (
    apply_single_thread_env,
    run_worker_pool,
)
from process_discovery_cash.hpo.study import (
    completed_trial_count,
    journal_path_for,
    open_study,
    study_name_for,
    worker_sampler_seed,
)
from process_discovery_cash.hpo.study_manifest import load_study_manifest_rows
from process_discovery_cash.hpo.summary import write_study_summary
from process_discovery_cash.hpo.trial_runner import StudyContext
from process_discovery_cash.hpo.worker import run_hpo_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one HPO study: N workers share an Optuna TPE study (journal file) "
            "and iteratively evaluate discovery configurations for one log/algorithm."
        )
    )
    selector = parser.add_argument_group("study selection")
    selector.add_argument("--study-manifest", help="Study manifest CSV.")
    selector.add_argument(
        "--study-index",
        type=int,
        default=None,
        help="Row of the study manifest. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    selector.add_argument("--config", help="Experiment config YAML (direct selection).")
    selector.add_argument("--log-id", help="Log id (direct selection).")
    selector.add_argument("--algorithm", help="Algorithm name (direct selection).")
    parser.add_argument(
        "--worker-walltime-seconds",
        type=float,
        default=14400,
        help="Total worker allocation in seconds. Default: 14400.",
    )
    parser.add_argument(
        "--safety-margin-seconds",
        type=float,
        default=600,
        help="Do not start new trials inside this final margin. Default: 600.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Concurrent evaluation workers inside this allocation. Set to "
            "SLURM_CPUS_PER_TASK to evaluate that many configurations in parallel."
        ),
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help=(
            "Base worker id (sampler-seed offset). Give distinct values when several "
            "Slurm jobs work on the same study concurrently. Default: 0."
        ),
    )
    parser.add_argument(
        "--max-trials-per-worker",
        type=int,
        help="Optional cap on trials attempted by this worker.",
    )
    parser.add_argument(
        "--child-memory-limit-mb",
        type=float,
        default=None,
        help=(
            "Address-space limit (RLIMIT_AS, in MB) applied inside each trial subprocess. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help=(
            "Deprecated compatibility flag; ignored. HPO uses source XES files "
            "and optional runtime-local caches."
        ),
    )
    parser.add_argument(
        "--no-isolate-runs",
        action="store_true",
        help=(
            "Run trials in the worker process instead of an isolated subprocess. "
            "Metric hangs and OOM kills then take the worker down with them."
        ),
    )
    parser.add_argument(
        "--no-write-summary",
        action="store_true",
        help="Skip writing the study summary JSON after the worker(s) finish.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the study and report its state without running any trials.",
    )
    return parser


def _resolve_study_selection(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[str, str, str]:
    """Return (experiment_config_path, log_id, algorithm_name)."""
    if args.study_manifest:
        study_index = args.study_index
        if study_index is None:
            task_id = os.getenv("SLURM_ARRAY_TASK_ID")
            if task_id is None:
                parser.error("--study-index is required (or set SLURM_ARRAY_TASK_ID)")
            study_index = int(task_id)
        rows = load_study_manifest_rows(args.study_manifest)
        if study_index < 0 or study_index >= len(rows):
            parser.error(f"--study-index {study_index} outside 0..{len(rows) - 1}")
        row = rows[study_index]
        return row["experiment_config_path"], row["log_id"], row["algorithm_name"]
    if args.config and args.log_id and args.algorithm:
        return args.config, args.log_id, args.algorithm
    parser.error(
        "Select a study with --study-manifest [--study-index] or with --config --log-id --algorithm"
    )
    raise AssertionError("unreachable")


def _child_argv(args: argparse.Namespace, config: str, log_id: str, algorithm: str) -> list[str]:
    """Command line for one pooled child: a single-worker run of the same study."""
    argv: list[str] = [
        sys.executable,
        sys.argv[0],
        "--config",
        config,
        "--log-id",
        log_id,
        "--algorithm",
        algorithm,
        "--worker-walltime-seconds",
        repr(args.worker_walltime_seconds),
        "--safety-margin-seconds",
        repr(args.safety_margin_seconds),
        "--num-workers",
        "1",
        "--no-write-summary",
    ]
    if args.max_trials_per_worker is not None:
        argv += ["--max-trials-per-worker", str(args.max_trials_per_worker)]
    if args.child_memory_limit_mb is not None:
        argv += ["--child-memory-limit-mb", repr(args.child_memory_limit_mb)]
    if args.require_artifacts:
        argv += ["--require-artifacts"]
    if args.no_isolate_runs:
        argv += ["--no-isolate-runs"]
    return argv


def _open_study_for(ctx: StudyContext, worker_id: int):
    study_name = study_name_for(
        ctx.experiment.experiment_id, ctx.log_ref.log_id, ctx.algorithm_ref.name
    )
    journal_path = journal_path_for(ctx.hpo.storage_root, ctx.experiment.experiment_id, study_name)
    return open_study(
        study_name=study_name,
        journal_path=journal_path,
        sampler_seed=worker_sampler_seed(ctx.hpo.sampler_seed, worker_id),
        n_startup_trials=ctx.hpo.n_startup_trials,
        multivariate=ctx.hpo.multivariate,
        group=ctx.hpo.group,
        constant_liar=ctx.hpo.constant_liar,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num-workers must be a positive integer")

    config, log_id, algorithm = _resolve_study_selection(parser, args)
    ctx = StudyContext.from_experiment(
        config, log_id, algorithm, require_artifacts=args.require_artifacts
    )
    study_name = study_name_for(
        ctx.experiment.experiment_id, ctx.log_ref.log_id, ctx.algorithm_ref.name
    )

    if args.dry_run:
        study = _open_study_for(ctx, args.worker_id)
        print(
            f"[dry-run] study={study_name} n_trials={ctx.hpo.n_trials} "
            f"completed={completed_trial_count(study)} "
            f"num_workers={args.num_workers} "
            f"per_trial_walltime_seconds={ctx.hpo.per_trial_walltime_seconds}",
            flush=True,
        )
        return

    if args.num_workers > 1:
        # Pin numeric-library threads, then supervise N single-worker children
        # that share the study's journal file.
        apply_single_thread_env()
        print(
            f"study={study_name} pool of {args.num_workers} workers; "
            f"base_worker_id={args.worker_id}",
            flush=True,
        )
        result = run_worker_pool(
            num_workers=args.num_workers,
            base_worker_id=args.worker_id,
            build_argv=lambda wid: [
                *_child_argv(args, config, log_id, algorithm),
                "--worker-id",
                str(wid),
            ],
        )
        if not args.no_write_summary:
            summary_path = write_study_summary(_open_study_for(ctx, args.worker_id), ctx)
            print(f"Wrote study summary {summary_path}", flush=True)
        if not result.ok:
            failed_ids = ", ".join(str(item.worker_id) for item in result.failed)
            print(f"Worker pool: {len(result.failed)} worker(s) failed ({failed_ids})", flush=True)
            sys.exit(1)
        return

    study = _open_study_for(ctx, args.worker_id)
    print(
        f"study={study_name} worker_id={args.worker_id} "
        f"n_trials={ctx.hpo.n_trials} completed={completed_trial_count(study)}",
        flush=True,
    )
    stats = run_hpo_worker(
        ctx=ctx,
        study=study,
        worker_id=args.worker_id,
        worker_walltime_seconds=args.worker_walltime_seconds,
        safety_margin_seconds=args.safety_margin_seconds,
        max_trials_per_worker=args.max_trials_per_worker,
        isolate_runs=not args.no_isolate_runs,
        child_memory_limit_mb=args.child_memory_limit_mb,
    )
    print(
        f"Worker summary: told_complete={stats.told_complete} told_failed={stats.told_failed} "
        f"cached={stats.cached} executed={stats.executed} stopped={stats.stopped_reason}",
        flush=True,
    )
    if not args.no_write_summary:
        summary_path = write_study_summary(study, ctx)
        print(f"Wrote study summary {summary_path}", flush=True)


if __name__ == "__main__":
    main()
