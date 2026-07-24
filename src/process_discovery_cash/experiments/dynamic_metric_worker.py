from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from process_discovery_cash.experiments.dynamic_worker import (
    AttemptTracker,
    Claim,
    ClaimManager,
    WorkerStats,
    _result_written_since,
    _termination_error_message,
    _termination_metadata,
    _validate_worker_limits,
    deterministic_scan_order,
)
from process_discovery_cash.experiments.metric_manifest import (
    run_metric_manifest_row,
    source_model_is_available,
)
from process_discovery_cash.experiments.metric_timeout import METRIC_TIMEOUT_FIELD
from process_discovery_cash.experiments.run_isolation import (
    ROW_KIND_METRIC,
    RowExecutionOutcome,
    run_row_in_subprocess,
)
from process_discovery_cash.experiments.runner import _write_json_atomically
from process_discovery_cash.experiments.saved_model_metrics import _load_json
from process_discovery_cash.utils.hashing import stable_hash
from process_discovery_cash.utils.paths import portable_project_path


@dataclass(frozen=True)
class DynamicMetricManifestEntry:
    row_index: int
    row: dict[str, str]
    run_id: str
    malformed_error: str | None = None


@dataclass(frozen=True)
class MetricResultInspection:
    state: str
    payload: dict[str, Any] | None = None


def load_dynamic_metric_manifest_entries(
    manifest_path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[DynamicMetricManifestEntry]:
    entries: list[DynamicMetricManifestEntry] = []
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw_row in enumerate(reader):
            raw = {key: "" if value is None else str(value) for key, value in raw_row.items()}
            try:
                row = _normalize_metric_manifest_row(raw, row_index, results_dir=results_dir)
                entries.append(
                    DynamicMetricManifestEntry(
                        row_index=row_index,
                        row=row,
                        run_id=_metric_run_id(row),
                    )
                )
            except Exception as exc:
                row = _fallback_malformed_metric_row(raw, row_index, results_dir=results_dir)
                entries.append(
                    DynamicMetricManifestEntry(
                        row_index=row_index,
                        row=row,
                        run_id=_metric_run_id(row),
                        malformed_error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return entries


def run_dynamic_metric_worker(
    entries: list[DynamicMetricManifestEntry],
    *,
    state_dir: str | Path,
    worker_walltime_seconds: float,
    safety_margin_seconds: float,
    worker_id: int = 0,
    max_runs_per_worker: int | None = None,
    retry_failed: bool = False,
    retry_failed_only: bool = False,
    retry_failed_with_more_memory: bool = False,
    reclaim_stale_after_seconds: float | None = None,
    command_args: list[str] | None = None,
    isolate_runs: bool = True,
    child_memory_limit_mb: float | None = None,
    max_abnormal_attempts: int | None = 3,
    heartbeat_interval_seconds: float = 60.0,
    metric_timeout_seconds: float | None = None,
    run_row=None,
    execute_row: Callable[..., RowExecutionOutcome] | None = None,
    monotonic=time.monotonic,
) -> WorkerStats:
    retry_modes = sum((retry_failed, retry_failed_only, retry_failed_with_more_memory))
    if retry_modes > 1:
        raise ValueError("failed retry modes are mutually exclusive")
    if retry_failed_with_more_memory and child_memory_limit_mb is None:
        raise ValueError("retry_failed_with_more_memory requires child_memory_limit_mb")
    _validate_worker_limits(
        worker_walltime_seconds=worker_walltime_seconds,
        safety_margin_seconds=safety_margin_seconds,
        max_runs_per_worker=max_runs_per_worker,
        reclaim_stale_after_seconds=reclaim_stale_after_seconds,
        max_abnormal_attempts=max_abnormal_attempts,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        child_memory_limit_mb=child_memory_limit_mb,
    )
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)
    entries = [_ensure_malformed_output_path(entry, state_path) for entry in entries]
    claims = ClaimManager(state_path)
    attempts = AttemptTracker(state_path)
    # An injected run_row (tests, --no-isolate-runs) executes in-process as
    # before; the default path runs each row in an isolated child process.
    use_isolation = run_row is None and isolate_runs
    if run_row is None:
        run_row = run_metric_manifest_row
    if use_isolation and execute_row is None:

        def execute_row(
            effective_row: Mapping[str, str],
            *,
            deadline_monotonic: float,
            on_tick: Callable[[], None],
            force: bool = False,
        ) -> RowExecutionOutcome:
            return run_row_in_subprocess(
                effective_row,
                kind=ROW_KIND_METRIC,
                command_args=command_args,
                deadline_monotonic=deadline_monotonic,
                memory_limit_mb=child_memory_limit_mb,
                on_tick=on_tick,
                tick_interval_seconds=heartbeat_interval_seconds,
                scratch_dir=state_path,
                force=force,
                monotonic=monotonic,
            )

    stats = WorkerStats()
    started = monotonic()
    claim_deadline = started + worker_walltime_seconds - safety_margin_seconds

    for entry_index in deterministic_scan_order(len(entries), worker_id):
        elapsed = monotonic() - started
        if monotonic() >= claim_deadline:
            _progress("walltime_stop", None, stats, elapsed, detail="safety margin reached")
            break
        if max_runs_per_worker is not None and stats.claimed >= max_runs_per_worker:
            _progress("max_runs_stop", None, stats, elapsed)
            break

        entry = entries[entry_index]
        inspection = _inspect_metric_entry(entry)
        skip_reason = _result_skip_reason(
            inspection,
            retry_failed=retry_failed,
            retry_failed_only=retry_failed_only,
            retry_failed_with_more_memory=retry_failed_with_more_memory,
            child_memory_limit_mb=child_memory_limit_mb,
        )
        if skip_reason is not None:
            _record_result_skip(stats, skip_reason)
            _progress(f"skipped_{skip_reason}", entry.run_id, stats, elapsed)
            continue

        claim_attempt = claims.try_claim(
            entry.run_id,
            _claim_metadata(
                entry,
                worker_id,
                retry_failed,
                retry_failed_only,
                retry_failed_with_more_memory,
            ),
            reclaim_stale_after_seconds=reclaim_stale_after_seconds,
            success_exists=lambda entry=entry: (
                _inspect_metric_entry(entry).state == "success_complete"
            ),
        )
        claim = claim_attempt.claim
        if claim is None:
            if claim_attempt.stale_detected:
                _progress("stale_claim_detected", entry.run_id, stats, elapsed)
            if claim_attempt.reason == "completed_with_stale_claim":
                stats.skipped_success += 1
                _progress("skipped_success", entry.run_id, stats, elapsed)
            else:
                stats.skipped_claimed += 1
                _progress("skipped_claimed", entry.run_id, stats, elapsed)
            continue

        stats.claimed += 1
        if claim.stale_reclaimed:
            stats.stale_reclaimed += 1
            _progress("stale_claim_reclaimed", entry.run_id, stats, elapsed)
        _progress("claimed", entry.run_id, stats, elapsed)

        stop_after_release = False
        try:
            inspection = _inspect_metric_entry(entry)
            skip_reason = _result_skip_reason(
                inspection,
                retry_failed=retry_failed,
                retry_failed_only=retry_failed_only,
                retry_failed_with_more_memory=retry_failed_with_more_memory,
                child_memory_limit_mb=child_memory_limit_mb,
            )
            if skip_reason is not None:
                _record_result_skip(stats, skip_reason)
                _progress(f"skipped_{skip_reason}", entry.run_id, stats, monotonic() - started)
                continue

            if claim.stale_reclaimed:
                attempts.record(entry.run_id, "stale_claim_reclaimed")
            abnormal_count = attempts.count(entry.run_id)
            if max_abnormal_attempts is not None and abnormal_count >= max_abnormal_attempts:
                _write_abnormal_attempts_metric_result(entry, abnormal_count, worker_id)
                attempts.clear(entry.run_id)
                stats.failed += 1
                _progress(
                    "failed",
                    entry.run_id,
                    stats,
                    monotonic() - started,
                    detail=f"gave up after {abnormal_count} abnormal attempts",
                )
                continue

            effective_row = entry.row
            if monotonic() >= claim_deadline:
                stop_after_release = True
                run_isolated = False
                run_in_process = False
                force_row = False
            else:
                if _should_retry_failed_metric_result(
                    inspection,
                    retry_failed=retry_failed,
                    retry_failed_only=retry_failed_only,
                    retry_failed_with_more_memory=retry_failed_with_more_memory,
                ):
                    _archive_previous_attempt(entry, inspection, state_path, claim)
                force_row = _should_retry_failed_metric_result(
                    inspection,
                    retry_failed=retry_failed,
                    retry_failed_only=retry_failed_only,
                    retry_failed_with_more_memory=retry_failed_with_more_memory,
                )
                run_isolated = use_isolation and not entry.malformed_error
                run_in_process = not run_isolated
                effective_row = _with_capped_metric_timeout(
                    entry.row,
                    remaining_run_seconds=claim_deadline - monotonic(),
                    override_seconds=metric_timeout_seconds,
                )

            if run_isolated:
                child_started_wall = time.time()
                outcome = execute_row(
                    effective_row,
                    deadline_monotonic=claim_deadline,
                    on_tick=lambda claim=claim: claims.heartbeat(claim),
                    force=force_row,
                )
                if outcome.killed_by_parent:
                    abnormal_count = attempts.record(entry.run_id, "walltime_kill")
                    if (
                        max_abnormal_attempts is not None
                        and abnormal_count >= max_abnormal_attempts
                    ):
                        _write_abnormal_attempts_metric_result(entry, abnormal_count, worker_id)
                        attempts.clear(entry.run_id)
                        stats.failed += 1
                        _progress(
                            "failed",
                            entry.run_id,
                            stats,
                            monotonic() - started,
                            detail=f"gave up after {abnormal_count} abnormal attempts",
                        )
                    else:
                        _progress(
                            "walltime_kill",
                            entry.run_id,
                            stats,
                            monotonic() - started,
                            detail=(
                                f"killed child after {outcome.duration_seconds:.0f}s; "
                                f"row left for retry (abnormal attempt {abnormal_count})"
                            ),
                        )
                    stop_after_release = True
                else:
                    final_inspection = _inspect_metric_entry(entry)
                    result_is_fresh = _result_written_since(
                        entry.row["output_path"],
                        child_started_wall,
                    )
                    if final_inspection.state == "success_complete":
                        attempts.clear(entry.run_id)
                        stats.completed += 1
                        _progress("completed", entry.run_id, stats, monotonic() - started)
                    elif final_inspection.state == "failed" and (
                        not outcome.abnormal or result_is_fresh
                    ):
                        attempts.clear(entry.run_id)
                        stats.failed += 1
                        _progress(
                            "failed",
                            entry.run_id,
                            stats,
                            monotonic() - started,
                            detail=f"result_state={final_inspection.state}",
                        )
                    elif outcome.abnormal:
                        error = _termination_error_message(outcome, child_memory_limit_mb)
                        _write_metric_termination_result(
                            entry,
                            outcome,
                            worker_id,
                            child_memory_limit_mb,
                        )
                        attempts.clear(entry.run_id)
                        stats.failed += 1
                        _progress(
                            "failed",
                            entry.run_id,
                            stats,
                            monotonic() - started,
                            detail=error,
                        )
                    else:
                        stats.failed += 1
                        _progress(
                            "failed",
                            entry.run_id,
                            stats,
                            monotonic() - started,
                            detail=f"result_state={final_inspection.state}",
                        )
            elif run_in_process:
                if entry.malformed_error:
                    _write_malformed_result(entry, worker_id)
                    written_path = Path(entry.row["output_path"])
                else:
                    written_path = run_row(
                        effective_row, command_args=command_args, force=force_row
                    )

                final_inspection = inspect_metric_result_file(entry.row, written_path)
                if final_inspection.state == "success_complete":
                    attempts.clear(entry.run_id)
                    stats.completed += 1
                    _progress("completed", entry.run_id, stats, monotonic() - started)
                else:
                    if final_inspection.state == "failed":
                        attempts.clear(entry.run_id)
                    stats.failed += 1
                    _progress(
                        "failed",
                        entry.run_id,
                        stats,
                        monotonic() - started,
                        detail=f"result_state={final_inspection.state}",
                    )
        except Exception as exc:
            stats.failed += 1
            _write_worker_exception_result(entry, exc, worker_id)
            _progress(
                "failed",
                entry.run_id,
                stats,
                monotonic() - started,
                detail=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if not claims.release(claim):
                _progress(
                    "claim_release_skipped",
                    entry.run_id,
                    stats,
                    monotonic() - started,
                    detail="claim ownership changed or guard was busy",
                )
        if stop_after_release:
            _progress(
                "walltime_stop",
                None,
                stats,
                monotonic() - started,
                detail="safety margin reached after claim",
            )
            break

    _progress("worker_finished", None, stats, monotonic() - started)
    return stats


def inspect_metric_result_file(
    row: dict[str, str],
    output_path: str | Path,
) -> MetricResultInspection:
    path = Path(output_path)
    if not path.exists():
        return MetricResultInspection("missing")
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return MetricResultInspection("corrupt")

    status = payload.get("status")
    if status == "failed":
        return MetricResultInspection("failed", payload)
    if status == "timeout":
        return MetricResultInspection("timeout", payload)
    if status not in {"success", "success_missing"}:
        return MetricResultInspection("incomplete", payload)
    if not _metric_payload_matches_row(row, payload):
        return MetricResultInspection("identity_mismatch", payload)
    if status == "success_missing":
        if source_model_is_available(row):
            # A stale placeholder written when the source discovery result was
            # missing/failed: the model is now available, so this can be recomputed
            # into real metrics rather than being treated as a completed result.
            return MetricResultInspection("recoverable", payload)
        if _all_requested_metrics_have_status(row, payload, "missing_model"):
            return MetricResultInspection("success_complete", payload)
        return MetricResultInspection("failed", payload)
    if not _all_requested_metrics_succeeded(row, payload):
        return MetricResultInspection("failed", payload)
    return MetricResultInspection("success_complete", payload)


def _metric_payload_matches_row(row: dict[str, str], payload: dict[str, Any]) -> bool:
    expected = {
        "source_config_hash": row.get("source_config_hash") or "",
        "metric_profile": row.get("metric_profile") or "",
        "source_result_path": row.get("source_result_path") or "",
    }
    return all(str(payload.get(key) or "") == value for key, value in expected.items())


def _inspect_metric_entry(entry: DynamicMetricManifestEntry) -> MetricResultInspection:
    return inspect_metric_result_file(entry.row, entry.row["output_path"])


def _all_requested_metrics_succeeded(row: dict[str, str], payload: dict[str, Any]) -> bool:
    return _all_requested_metrics_have_status(row, payload, "success")


def _all_requested_metrics_have_status(
    row: dict[str, str],
    payload: dict[str, Any],
    status: str,
) -> bool:
    metric_statuses = payload.get("metric_statuses")
    if not isinstance(metric_statuses, dict):
        return False
    for name in _safe_metric_names_from_row(row):
        record = metric_statuses.get(name)
        if not isinstance(record, dict) or record.get("status") != status:
            return False
    return True


def _result_skip_reason(
    inspection: MetricResultInspection,
    *,
    retry_failed: bool,
    retry_failed_only: bool = False,
    retry_failed_with_more_memory: bool = False,
    child_memory_limit_mb: float | None = None,
) -> str | None:
    if inspection.state == "success_complete":
        return "success"
    if inspection.state == "failed":
        if retry_failed_with_more_memory:
            previous_limit = _recorded_child_memory_limit_mb(inspection.payload)
            assert child_memory_limit_mb is not None
            if previous_limit is None or child_memory_limit_mb > previous_limit:
                return None
            return "failed"
        if retry_failed or retry_failed_only:
            return None
        return "failed"
    if inspection.state == "timeout":
        if retry_failed:
            return None
        return "failed"
    if retry_failed_with_more_memory:
        return "failed"
    return None


def _recorded_child_memory_limit_mb(payload: dict[str, Any] | None) -> float | None:
    """Return the effective per-child memory limit recorded by a metric result."""
    if not payload:
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    dynamic_worker = metadata.get("dynamic_worker")
    if isinstance(dynamic_worker, dict):
        termination = dynamic_worker.get("termination")
        if isinstance(termination, dict):
            try:
                value = float(termination.get("child_memory_limit_mb"))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
    command_args = metadata.get("command_args")
    if isinstance(command_args, list):
        try:
            index = command_args.index("--child-memory-limit-mb")
            value = float(command_args[index + 1])
            if value > 0:
                return value
        except (ValueError, IndexError, TypeError):
            pass

    slurm = metadata.get("slurm")
    if not isinstance(slurm, dict):
        return None
    requested_bytes = slurm.get("requested_memory_bytes")
    try:
        allocation_mb = float(requested_bytes) / (1024 * 1024)
    except (TypeError, ValueError):
        return None
    workers: int | None = None
    if isinstance(command_args, list):
        try:
            workers = int(command_args[command_args.index("--num-workers") + 1])
        except (ValueError, IndexError, TypeError):
            pass
    if workers is None:
        workers = 1
    return allocation_mb / max(workers, 1)


def _should_retry_failed_metric_result(
    inspection: MetricResultInspection,
    *,
    retry_failed: bool,
    retry_failed_only: bool,
    retry_failed_with_more_memory: bool = False,
) -> bool:
    return inspection.state == "failed" and (
        retry_failed or retry_failed_only or retry_failed_with_more_memory
    )


def _record_result_skip(stats: WorkerStats, reason: str) -> None:
    if reason == "success":
        stats.skipped_success += 1
    elif reason == "failed":
        stats.skipped_failed += 1


def _claim_metadata(
    entry: DynamicMetricManifestEntry,
    worker_id: int,
    retry_failed: bool,
    retry_failed_only: bool,
    retry_failed_with_more_memory: bool = False,
) -> dict[str, Any]:
    return {
        "row_index": entry.row_index,
        "worker_id": worker_id,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "retry_failed": retry_failed,
        "retry_failed_only": retry_failed_only,
        "retry_failed_with_more_memory": retry_failed_with_more_memory,
        "output_path": entry.row.get("output_path"),
        "metric_profile": entry.row.get("metric_profile"),
    }


def _archive_previous_attempt(
    entry: DynamicMetricManifestEntry,
    inspection: MetricResultInspection,
    state_dir: Path,
    claim: Claim,
) -> Path | None:
    if inspection.payload is None:
        return None
    source_path = Path(entry.row["output_path"])
    if not source_path.exists():
        return None
    attempts_dir = state_dir / "attempts" / _claim_key(entry.run_id)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive_path = attempts_dir / f"{timestamp}_{inspection.state}_{claim.token}.json"
    try:
        with archive_path.open("xb") as destination:
            with source_path.open("rb") as source:
                shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError:
        return None
    return archive_path


def _write_malformed_result(entry: DynamicMetricManifestEntry, worker_id: int) -> None:
    error = f"Malformed metric manifest row {entry.row_index}: {entry.malformed_error}"
    output_path = Path(entry.row["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(output_path, _failed_metric_result_payload(entry, error, worker_id))


def _write_worker_exception_result(
    entry: DynamicMetricManifestEntry,
    exc: Exception,
    worker_id: int,
) -> None:
    error = f"Dynamic metric worker error: {type(exc).__name__}: {exc}"
    try:
        output_path = Path(entry.row["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(output_path, _failed_metric_result_payload(entry, error, worker_id))
    except Exception:
        pass


def _write_metric_termination_result(
    entry: DynamicMetricManifestEntry,
    outcome: RowExecutionOutcome,
    worker_id: int,
    child_memory_limit_mb: float | None,
) -> None:
    error = _termination_error_message(outcome, child_memory_limit_mb)
    try:
        payload = _failed_metric_result_payload(entry, error, worker_id)
        payload["metadata"]["dynamic_worker"]["termination"] = _termination_metadata(
            outcome,
            child_memory_limit_mb,
        )
        output_path = Path(entry.row["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(output_path, payload)
    except Exception:
        pass


def _write_abnormal_attempts_metric_result(
    entry: DynamicMetricManifestEntry,
    abnormal_count: int,
    worker_id: int,
) -> None:
    error = (
        f"Giving up after {abnormal_count} abnormal attempts: previous workers died or hit "
        "the walltime boundary while running this row. Rerun with --retry-failed, more "
        "memory, or a longer worker walltime."
    )
    try:
        payload = _failed_metric_result_payload(entry, error, worker_id)
        payload["metadata"]["dynamic_worker"]["abnormal_attempts"] = abnormal_count
        output_path = Path(entry.row["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(output_path, payload)
    except Exception:
        pass


def _failed_metric_result_payload(
    entry: DynamicMetricManifestEntry,
    error: str,
    worker_id: int,
) -> dict[str, Any]:
    row = entry.row
    metric_names = _safe_metric_names_from_row(row)
    metrics = {name: None for name in metric_names}
    metric_statuses = {
        name: {"status": "not_computed", "error": error, "value": None} for name in metric_names
    }
    return {
        "status": "failed",
        "metric_runtime_seconds": None,
        "discovery_runtime_seconds": None,
        "discovery_status": "failed",
        "source_result_path": row.get("source_result_path"),
        "source_config_hash": row.get("source_config_hash"),
        "config_hash": row.get("source_config_hash"),
        "experiment_id": row.get("experiment_id"),
        "log_id": row.get("log_id"),
        "log_path": "",
        "algorithm_name": row.get("algorithm_name"),
        "backend": "saved_model",
        "hyperparameters": {},
        "warnings": [],
        "model_path": None,
        "discovered_model_type": None,
        "test_log_path": row.get("test_log_path"),
        "seed": 0,
        "metric_profile": row.get("metric_profile"),
        "metric_names": metric_names,
        "metrics": metrics,
        "metric_statuses": metric_statuses,
        "error_message": error,
        "metadata": {
            "config_hash": row.get("source_config_hash") or "",
            "manifest_row_index": entry.row_index,
            "dynamic_worker": {
                "worker_id": worker_id,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                "hostname": socket.gethostname(),
            },
        },
        "source_metadata": {},
    }


def _normalize_metric_manifest_row(
    row: dict[str, str],
    row_index: int,
    *,
    results_dir: str | Path | None,
) -> dict[str, str]:
    required = [
        "source_result_path",
        "experiment_id",
        "log_id",
        "algorithm_name",
        "test_log_path",
        "metric_profile",
        "metric_names_json",
        "output_path",
    ]
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(
            f"Metric manifest row {row_index} is missing required fields: {', '.join(missing)}"
        )
    normalized = dict(row)
    if results_dir is not None:
        normalized["output_path"] = portable_project_path(
            _rebase_result_path(normalized["output_path"], results_dir)
        )
    _metric_names_from_row(normalized)
    return normalized


def _fallback_malformed_metric_row(
    raw: dict[str, str],
    row_index: int,
    *,
    results_dir: str | Path | None,
) -> dict[str, str]:
    row = dict(raw)
    row["source_result_path"] = row.get("source_result_path") or ""
    row["source_config_hash"] = row.get("source_config_hash") or ""
    row["experiment_id"] = row.get("experiment_id") or "metric_manifest"
    row["log_id"] = row.get("log_id") or f"row_{row_index}"
    row["algorithm_name"] = row.get("algorithm_name") or "unknown"
    row["test_log_path"] = row.get("test_log_path") or ""
    row["log_cache_key"] = row.get("log_cache_key") or row["log_id"]
    row["metric_profile"] = row.get("metric_profile") or "unknown"
    row["metric_names_json"] = row.get("metric_names_json") or "[]"
    output_path = row.get("output_path")
    if output_path and results_dir is not None:
        row["output_path"] = portable_project_path(_rebase_result_path(output_path, results_dir))
    return row


def _ensure_malformed_output_path(
    entry: DynamicMetricManifestEntry,
    state_dir: Path,
) -> DynamicMetricManifestEntry:
    if entry.row.get("output_path"):
        return entry
    row = dict(entry.row)
    row["output_path"] = (
        state_dir / "malformed_metric_results" / f"{_claim_key(entry.run_id)}.json"
    ).as_posix()
    return DynamicMetricManifestEntry(
        row_index=entry.row_index,
        row=row,
        run_id=entry.run_id,
        malformed_error=entry.malformed_error,
    )


def _with_capped_metric_timeout(
    row: Mapping[str, str],
    *,
    remaining_run_seconds: float,
    override_seconds: float | None = None,
) -> dict[str, str]:
    """Return a row whose ``metric_timeout_seconds`` respects the worker walltime.

    An ``override_seconds`` (from ``--metric-timeout-seconds`` / the
    ``METRIC_TIMEOUT_SECONDS`` env) supersedes the manifest column, then the
    value is capped to the worker's remaining walltime so a per-metric timeout
    never outlives the allocation. Mirrors
    ``dynamic_worker._with_capped_execution_timeout``.
    """
    effective = dict(row)
    cap_seconds = max(0.001, remaining_run_seconds)

    configured: float | None = None
    if override_seconds is not None:
        configured = override_seconds
    else:
        raw = row.get(METRIC_TIMEOUT_FIELD)
        if raw not in (None, ""):
            try:
                configured = float(raw)
            except (TypeError, ValueError):
                configured = None

    if configured is None or configured <= 0:
        # No per-metric timeout requested; leave the row untouched (the worker
        # walltime kill still bounds the run), mirroring discovery's cap.
        return effective

    effective[METRIC_TIMEOUT_FIELD] = _format_metric_timeout(min(configured, cap_seconds))
    return effective


def _format_metric_timeout(seconds: float) -> str:
    return str(int(seconds)) if float(seconds).is_integer() else repr(float(seconds))


def _metric_run_id(row: dict[str, str]) -> str:
    source_config_hash = row.get("source_config_hash")
    if source_config_hash:
        return stable_hash(
            {
                "source_config_hash": source_config_hash,
                "metric_profile": row.get("metric_profile") or "",
                "output_path": row.get("output_path") or "",
            },
            length=32,
        )
    return stable_hash(
        {
            "source_result_path": row.get("source_result_path") or "",
            "metric_profile": row.get("metric_profile") or "",
            "output_path": row.get("output_path") or "",
            "row": row,
        },
        length=32,
    )


def _metric_names_from_row(row: dict[str, str]) -> list[str]:
    raw = row.get("metric_names_json") or "[]"
    parsed = json.loads(raw)
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("metric_names_json must contain a JSON list or string")
    return [str(name) for name in parsed]


def _safe_metric_names_from_row(row: dict[str, str]) -> list[str]:
    try:
        return _metric_names_from_row(row)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _rebase_result_path(output_path: str | Path, results_dir: str | Path) -> Path:
    path = Path(output_path)
    parts = path.parts
    try:
        results_index = parts.index("results")
    except ValueError:
        if path.is_absolute():
            return path
        suffix = path
    else:
        suffix = Path(*parts[results_index + 1 :])
    return Path(results_dir) / suffix


def _claim_key(run_id: str) -> str:
    if (
        run_id
        and len(run_id) <= 180
        and run_id not in {".", ".."}
        and all(character.isalnum() or character in "._-" for character in run_id)
    ):
        return run_id
    return stable_hash({"run_id": run_id}, length=32)


def _progress(
    event: str,
    run_id: str | None,
    stats: WorkerStats,
    elapsed_seconds: float,
    *,
    detail: str | None = None,
) -> None:
    fields = [
        f"event={event}",
        f"completed={stats.completed}",
        f"failed={stats.failed}",
        f"skipped={stats.skipped}",
        f"claimed={stats.claimed}",
        f"elapsed_seconds={max(0.0, elapsed_seconds):.3f}",
    ]
    if run_id is not None:
        fields.insert(1, f"run_id={run_id}")
    if detail:
        fields.append(f"detail={json.dumps(detail)}")
    print(" ".join(fields), flush=True)
