from __future__ import annotations

import argparse
import os
import sys

from process_discovery_cash.experiments.dynamic_metric_worker import (
    DynamicMetricManifestEntry,
    load_dynamic_metric_manifest_entries,
    run_dynamic_metric_worker,
)
from process_discovery_cash.experiments.dynamic_paths import default_dynamic_metric_state_dir
from process_discovery_cash.experiments.worker_pool import (
    apply_single_thread_env,
    run_worker_pool,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dynamically claim and execute unfinished metric manifest rows."
    )
    parser.add_argument("--manifest", required=True, help="Path to the full metric manifest CSV.")
    parser.add_argument(
        "--state-dir",
        help=(
            "Shared worker state directory. Defaults to an algorithm-specific "
            "subdirectory under runs/metric_state."
        ),
    )
    parser.add_argument(
        "--results-dir",
        help=(
            "Optional metric result-root override. Paths below a manifest 'results/' "
            "component are rebased below this directory."
        ),
    )
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
        help="Do not claim new runs inside this final margin. Default: 600.",
    )
    parser.add_argument(
        "--max-runs-per-worker",
        type=int,
        help="Optional cap on runs claimed by this worker.",
    )
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed metric result rows. Successes are still skipped.",
    )
    retry_group.add_argument(
        "--retry-failed-only",
        action="store_true",
        help=(
            "Retry only existing metric result rows with top-level status='failed'. "
            "Timed-out rows are skipped."
        ),
    )
    retry_group.add_argument(
        "--retry-failed-with-more-memory",
        action="store_true",
        help=(
            "Retry failed or incomplete metric results only when the current "
            "--child-memory-limit-mb exceeds the effective per-child limit "
            "recorded by the previous result. Timed-out rows are skipped."
        ),
    )
    parser.add_argument(
        "--reclaim-stale-after-seconds",
        type=float,
        help="Reclaim claims older than this threshold. Disabled by default.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="Deterministic scan offset. Defaults to SLURM_ARRAY_TASK_ID or 0.",
    )
    parser.add_argument(
        "--child-memory-limit-mb",
        type=float,
        default=None,
        help=(
            "Address-space limit (RLIMIT_AS, in MB) applied inside each row subprocess so "
            "runaway runs fail with MemoryError instead of an OOM SIGKILL. Disabled by default."
        ),
    )
    parser.add_argument(
        "--no-isolate-runs",
        action="store_true",
        help=(
            "Run claimed rows in the worker process instead of an isolated subprocess. "
            "A row that is OOM-killed then takes the whole worker down with it."
        ),
    )
    parser.add_argument(
        "--max-abnormal-attempts",
        type=int,
        default=3,
        help=(
            "Write a terminal failed result after this many abnormal attempts (previous "
            "worker died or the run was killed at the walltime boundary). Default: 3."
        ),
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=60,
        help="Refresh the claim's liveness timestamp this often while a run is in flight.",
    )
    parser.add_argument(
        "--metric-timeout-seconds",
        type=float,
        default=None,
        help=(
            "Override the per-row metric evaluation timeout (seconds), superseding the "
            "manifest's metric_timeout_seconds column. Falls back to the "
            "METRIC_TIMEOUT_SECONDS environment variable when omitted; if neither is set, "
            "each row's manifest value is used. Effective timeouts are always capped to the "
            "worker's remaining walltime."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of internal metric workers to run concurrently inside this "
            "process's allocation. Default 1 keeps the classic single-worker "
            "behavior. Set this to SLURM_CPUS_PER_TASK to run that many independent "
            "metric evaluations in parallel within one Slurm job."
        ),
    )
    return parser


def _resolve_base_worker_id(worker_id: int | None) -> int:
    if worker_id is not None:
        return worker_id
    return int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))


def _child_base_argv(args: argparse.Namespace, state_dir: str) -> list[str]:
    """Command line for one pooled child: a single metric worker with fixed state."""
    argv: list[str] = [
        sys.executable,
        sys.argv[0],
        "--manifest",
        args.manifest,
        "--state-dir",
        state_dir,
        "--worker-walltime-seconds",
        repr(args.worker_walltime_seconds),
        "--safety-margin-seconds",
        repr(args.safety_margin_seconds),
        "--max-abnormal-attempts",
        str(args.max_abnormal_attempts),
        "--heartbeat-interval-seconds",
        repr(args.heartbeat_interval_seconds),
        "--num-workers",
        "1",
    ]
    if args.results_dir:
        argv += ["--results-dir", args.results_dir]
    if args.max_runs_per_worker is not None:
        argv += ["--max-runs-per-worker", str(args.max_runs_per_worker)]
    if args.reclaim_stale_after_seconds is not None:
        argv += ["--reclaim-stale-after-seconds", repr(args.reclaim_stale_after_seconds)]
    if args.child_memory_limit_mb is not None:
        argv += ["--child-memory-limit-mb", repr(args.child_memory_limit_mb)]
    if args.metric_timeout_seconds is not None:
        argv += ["--metric-timeout-seconds", repr(args.metric_timeout_seconds)]
    if args.retry_failed:
        argv += ["--retry-failed"]
    if args.retry_failed_only:
        argv += ["--retry-failed-only"]
    if args.retry_failed_with_more_memory:
        argv += ["--retry-failed-with-more-memory"]
    if args.no_isolate_runs:
        argv += ["--no-isolate-runs"]
    return argv


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num-workers must be a positive integer")
    if args.retry_failed_with_more_memory and args.child_memory_limit_mb is None:
        parser.error("--retry-failed-with-more-memory requires --child-memory-limit-mb")
    worker_id = _resolve_base_worker_id(args.worker_id)
    metric_timeout_seconds = _resolve_metric_timeout_seconds(args.metric_timeout_seconds)
    command_args = sys.argv if argv is None else ["pdcash-run-dynamic-metric-worker", *argv]

    entries = load_dynamic_metric_manifest_entries(
        args.manifest,
        results_dir=args.results_dir,
    )
    state_dir = args.state_dir or default_metric_state_dir(args.manifest, entries)
    if args.num_workers > 1:
        apply_single_thread_env()
        print(
            f"Loaded {len(entries)} metric manifest rows; pool of {args.num_workers} "
            f"workers; base_worker_id={worker_id}; retry_failed={args.retry_failed}; "
            f"retry_failed_only={args.retry_failed_only}; state_dir={state_dir}",
            flush=True,
        )
        result = run_worker_pool(
            num_workers=args.num_workers,
            base_worker_id=worker_id,
            build_argv=lambda wid, state_dir=state_dir: [
                *_child_base_argv(args, state_dir),
                "--worker-id",
                str(wid),
            ],
        )
        if not result.ok:
            failed_ids = ", ".join(str(item.worker_id) for item in result.failed)
            print(f"Metric worker pool: {len(result.failed)} worker(s) failed ({failed_ids})")
            sys.exit(1)
        return

    print(
        f"Loaded {len(entries)} metric manifest rows; worker_id={worker_id}; "
        f"retry_failed={args.retry_failed}; retry_failed_only={args.retry_failed_only}; "
        f"retry_failed_with_more_memory={args.retry_failed_with_more_memory}; "
        f"state_dir={state_dir}",
        flush=True,
    )
    stats = run_dynamic_metric_worker(
        entries,
        state_dir=state_dir,
        worker_walltime_seconds=args.worker_walltime_seconds,
        safety_margin_seconds=args.safety_margin_seconds,
        worker_id=worker_id,
        max_runs_per_worker=args.max_runs_per_worker,
        retry_failed=args.retry_failed,
        retry_failed_only=args.retry_failed_only,
        retry_failed_with_more_memory=args.retry_failed_with_more_memory,
        reclaim_stale_after_seconds=args.reclaim_stale_after_seconds,
        command_args=command_args,
        isolate_runs=not args.no_isolate_runs,
        child_memory_limit_mb=args.child_memory_limit_mb,
        max_abnormal_attempts=args.max_abnormal_attempts,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        metric_timeout_seconds=metric_timeout_seconds,
    )
    print(
        f"Worker summary: completed={stats.completed} failed={stats.failed} "
        f"skipped={stats.skipped} claimed={stats.claimed} "
        f"stale_reclaimed={stats.stale_reclaimed}",
        flush=True,
    )


def _resolve_metric_timeout_seconds(cli_value: float | None) -> float | None:
    if cli_value is not None:
        return cli_value
    env_value = os.getenv("METRIC_TIMEOUT_SECONDS")
    if env_value is None or env_value.strip() == "":
        return None
    try:
        return float(env_value)
    except ValueError as exc:
        raise SystemExit(
            f"METRIC_TIMEOUT_SECONDS must be a number of seconds, got {env_value!r}"
        ) from exc


def default_metric_state_dir(
    manifest_path: str,
    entries: list[DynamicMetricManifestEntry],
) -> str:
    algorithm_names = {entry.row.get("algorithm_name") or "" for entry in entries}
    return default_dynamic_metric_state_dir(manifest_path, algorithm_names)


if __name__ == "__main__":
    main()
