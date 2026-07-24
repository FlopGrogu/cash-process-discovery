from __future__ import annotations

import csv
import json
import os
import shutil
import socket
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from process_discovery_cash.experiments.identity import EXECUTION_CONTROL_FIELDS
from process_discovery_cash.experiments.local_manifest_runner import normalize_manifest_row
from process_discovery_cash.experiments.run_isolation import (
    ROW_KIND_DISCOVERY,
    RowExecutionOutcome,
    run_row_in_subprocess,
)
from process_discovery_cash.experiments.runner import (
    ResultFileInspection,
    ResultFileState,
    _write_json_atomically,
    inspect_result_file,
)
from process_discovery_cash.experiments.runner import (
    run_manifest_row as _run_manifest_row,
)
from process_discovery_cash.utils.hashing import stable_hash
from process_discovery_cash.utils.paths import portable_project_path

TERMINAL_FAILURE_STATES = frozenset(
    {
        ResultFileState.FAILED,
        ResultFileState.TIMEOUT,
        ResultFileState.UNSUPPORTED,
    }
)
COMPLETED_RESULT_STATES = frozenset(
    {
        ResultFileState.SUCCESS,
        ResultFileState.SKIPPED,
    }
)
TIMEOUT_CONTROL_FIELDS = frozenset(
    field
    for field in EXECUTION_CONTROL_FIELDS
    if field in {"discovery_timeout_seconds", "timeout_seconds"}
)


@dataclass(frozen=True)
class DynamicManifestEntry:
    row_index: int
    row: dict[str, str]
    run_id: str
    malformed_error: str | None = None


@dataclass(frozen=True)
class Claim:
    path: Path
    run_id: str
    token: str
    stale_reclaimed: bool = False


@dataclass(frozen=True)
class ClaimAttempt:
    claim: Claim | None
    reason: str
    stale_detected: bool = False


@dataclass
class WorkerStats:
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    skipped_success: int = 0
    skipped_failed: int = 0
    skipped_claimed: int = 0
    stale_reclaimed: int = 0

    @property
    def skipped(self) -> int:
        return self.skipped_success + self.skipped_failed + self.skipped_claimed

    @property
    def finished(self) -> int:
        return self.completed + self.failed


class ClaimManager:
    """Filesystem-backed exclusive claims for dynamic manifest workers."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        guard_timeout_seconds: float = 1.0,
        stale_guard_after_seconds: float = 60.0,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.claims_dir = self.state_dir / "claims"
        self.guards_dir = self.state_dir / "claim_guards"
        self.claims_dir.mkdir(parents=True, exist_ok=True)
        self.guards_dir.mkdir(parents=True, exist_ok=True)
        self.guard_timeout_seconds = guard_timeout_seconds
        self.stale_guard_after_seconds = stale_guard_after_seconds

    def claim_path(self, run_id: str) -> Path:
        return self.claims_dir / f"{_claim_key(run_id)}.claim"

    def try_claim(
        self,
        run_id: str,
        metadata: Mapping[str, Any],
        *,
        reclaim_stale_after_seconds: float | None = None,
        success_exists: Callable[[], bool] | None = None,
    ) -> ClaimAttempt:
        key = _claim_key(run_id)
        guard_path = self.guards_dir / f"{key}.guard"
        if not self._acquire_guard(guard_path):
            return ClaimAttempt(None, "guard_busy")

        try:
            claim_path = self.claims_dir / f"{key}.claim"
            stale_detected = False
            stale_reclaimed = False
            if claim_path.exists():
                if reclaim_stale_after_seconds is None:
                    return ClaimAttempt(None, "already_claimed")
                age_seconds = _claim_age_seconds(claim_path)
                if age_seconds is None or age_seconds < reclaim_stale_after_seconds:
                    return ClaimAttempt(None, "already_claimed")
                stale_detected = True
                if success_exists is not None and success_exists():
                    return ClaimAttempt(None, "completed_with_stale_claim", True)
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
                stale_reclaimed = True

            token = uuid.uuid4().hex
            claim_metadata = dict(metadata)
            claim_metadata.update(
                {
                    "run_id": run_id,
                    "claim_token": token,
                    "claimed_at": _utc_now(),
                    "claimed_at_epoch_seconds": time.time(),
                }
            )
            try:
                descriptor = os.open(
                    claim_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                return ClaimAttempt(None, "already_claimed", stale_detected)

            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(claim_metadata, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                claim_path.unlink(missing_ok=True)
                raise
            return ClaimAttempt(
                Claim(
                    path=claim_path,
                    run_id=run_id,
                    token=token,
                    stale_reclaimed=stale_reclaimed,
                ),
                "claimed",
                stale_detected,
            )
        finally:
            self._release_guard(guard_path)

    def heartbeat(self, claim: Claim) -> bool:
        """Refresh the claim's liveness timestamp while its run is in flight."""
        try:
            os.utime(claim.path)
            return True
        except OSError:
            return False

    def release(self, claim: Claim) -> bool:
        key = _claim_key(claim.run_id)
        guard_path = self.guards_dir / f"{key}.guard"
        if not self._acquire_guard(guard_path):
            return False
        try:
            current_token = _read_claim_token(claim.path)
            if current_token != claim.token:
                return False
            claim.path.unlink(missing_ok=True)
            return True
        finally:
            self._release_guard(guard_path)

    def _acquire_guard(self, guard_path: Path) -> bool:
        deadline = time.monotonic() + self.guard_timeout_seconds
        while True:
            try:
                guard_path.mkdir()
                return True
            except FileExistsError:
                try:
                    age_seconds = max(0.0, time.time() - guard_path.stat().st_mtime)
                    if age_seconds >= self.stale_guard_after_seconds:
                        guard_path.rmdir()
                        continue
                except (FileNotFoundError, OSError):
                    pass
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)

    @staticmethod
    def _release_guard(guard_path: Path) -> None:
        try:
            guard_path.rmdir()
        except FileNotFoundError:
            pass


class AttemptTracker:
    """Counts abnormal attempts per run so poisoned rows terminate eventually.

    An abnormal attempt is one where the previous owner demonstrably died
    mid-run (its stale claim had to be reclaimed) or where the parent had to
    kill the run at the worker walltime boundary. Counter files live under
    the shared state directory and are only written while holding the row's
    claim, so there is a single writer at a time.
    """

    def __init__(self, state_dir: str | Path) -> None:
        self.counts_dir = Path(state_dir) / "attempt_counts"

    def count(self, run_id: str) -> int:
        try:
            with self._path(run_id).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return int(payload.get("count", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return 0

    def record(self, run_id: str, event: str) -> int:
        count = self.count(run_id) + 1
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(
            path,
            {
                "run_id": run_id,
                "count": count,
                "last_event": event,
                "updated_at": _utc_now(),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            },
        )
        return count

    def clear(self, run_id: str) -> None:
        self._path(run_id).unlink(missing_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.counts_dir / f"{_claim_key(run_id)}.json"


def load_dynamic_manifest_entries(
    manifest_path: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[DynamicManifestEntry]:
    entries: list[DynamicManifestEntry] = []
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw_row in enumerate(reader):
            raw = {key: "" if value is None else str(value) for key, value in raw_row.items()}
            try:
                row = normalize_manifest_row(raw, row_index, output_root=None)
                run_id = (
                    raw.get("run_id")
                    or raw.get("config_hash")
                    or row.get("config_hash")
                    or row["config_id"]
                )
                row["run_id"] = run_id
                if results_dir is not None:
                    row["output_path"] = portable_project_path(
                        _rebase_result_path(row["output_path"], results_dir)
                    )
                entries.append(
                    DynamicManifestEntry(
                        row_index=row_index,
                        row=row,
                        run_id=run_id,
                    )
                )
            except Exception as exc:
                run_id = (
                    raw.get("run_id")
                    or raw.get("config_hash")
                    or raw.get("config_id")
                    or stable_hash({"row_index": row_index, "row": raw})
                )
                fallback = _fallback_malformed_row(raw, row_index, run_id, results_dir)
                entries.append(
                    DynamicManifestEntry(
                        row_index=row_index,
                        row=fallback,
                        run_id=run_id,
                        malformed_error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return entries


@dataclass(frozen=True)
class DynamicRunPlan:
    """A no-execute preview of what a worker would do with the current results.

    ``would_run`` lists rows a worker would claim and execute; ``skipped`` maps
    each already-resolved run_id to the reason it would be skipped (``success``
    or ``failed``). Nothing is claimed or executed while building this.
    """

    would_run: list[DynamicManifestEntry] = dataclass_field(default_factory=list)
    skipped: dict[str, str] = dataclass_field(default_factory=dict)
    malformed: list[DynamicManifestEntry] = dataclass_field(default_factory=list)


def plan_dynamic_runs(
    entries: list[DynamicManifestEntry],
    *,
    retry_failed: bool = False,
    retry_failed_only: bool = False,
    retry_failed_with_more_memory: bool = False,
    child_memory_limit_mb: float | None = None,
) -> DynamicRunPlan:
    if sum((retry_failed, retry_failed_only, retry_failed_with_more_memory)) > 1:
        raise ValueError("retry modes are mutually exclusive")
    if retry_failed_with_more_memory and child_memory_limit_mb is None:
        raise ValueError("retry_failed_with_more_memory requires child_memory_limit_mb")
    plan = DynamicRunPlan()
    for entry in entries:
        if entry.malformed_error:
            plan.malformed.append(entry)
            plan.would_run.append(entry)
            continue
        inspection = _inspect_entry(entry)
        skip_reason = _result_skip_reason(
            inspection,
            retry_failed=retry_failed,
            retry_failed_only=retry_failed_only,
            retry_failed_with_more_memory=retry_failed_with_more_memory,
            child_memory_limit_mb=child_memory_limit_mb,
        )
        if skip_reason is not None:
            plan.skipped[entry.run_id] = skip_reason
        else:
            plan.would_run.append(entry)
    return plan


def deterministic_scan_order(row_count: int, worker_id: int) -> list[int]:
    if row_count <= 0:
        return []
    start = worker_id % row_count
    return [((start + offset) % row_count) for offset in range(row_count)]


def run_dynamic_worker(
    entries: list[DynamicManifestEntry],
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
    run_row: Callable[..., Path] | None = None,
    execute_row: Callable[..., RowExecutionOutcome] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> WorkerStats:
    if sum((retry_failed, retry_failed_only, retry_failed_with_more_memory)) > 1:
        raise ValueError("retry modes are mutually exclusive")
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
        run_row = _run_manifest_row
    if use_isolation and execute_row is None:

        def execute_row(
            effective_row: Mapping[str, str],
            *,
            deadline_monotonic: float,
            on_tick: Callable[[], None],
        ) -> RowExecutionOutcome:
            return run_row_in_subprocess(
                effective_row,
                kind=ROW_KIND_DISCOVERY,
                command_args=command_args,
                deadline_monotonic=deadline_monotonic,
                memory_limit_mb=child_memory_limit_mb,
                on_tick=on_tick,
                tick_interval_seconds=heartbeat_interval_seconds,
                scratch_dir=state_path,
                monotonic=monotonic,
            )

    stats = WorkerStats()
    started = monotonic()
    worker_deadline = started + worker_walltime_seconds
    claim_deadline = worker_deadline - safety_margin_seconds

    for entry_index in deterministic_scan_order(len(entries), worker_id):
        elapsed = monotonic() - started
        if monotonic() >= claim_deadline:
            _progress(
                "walltime_stop",
                None,
                stats,
                elapsed,
                detail="safety margin reached",
            )
            break
        if max_runs_per_worker is not None and stats.claimed >= max_runs_per_worker:
            _progress("max_runs_stop", None, stats, elapsed)
            break

        entry = entries[entry_index]
        inspection = _inspect_entry(entry)
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
                _inspect_entry(entry).state in COMPLETED_RESULT_STATES
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
            inspection = _inspect_entry(entry)
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
                _write_abnormal_attempts_result(entry, abnormal_count, worker_id)
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

            remaining_run_seconds = worker_deadline - monotonic()
            if remaining_run_seconds <= 0:
                stop_after_release = True
                run_isolated = False
                run_in_process = False
            else:
                if _should_retry_existing_result(
                    inspection,
                    retry_failed=retry_failed,
                    retry_failed_only=retry_failed_only,
                    retry_failed_with_more_memory=retry_failed_with_more_memory,
                    child_memory_limit_mb=child_memory_limit_mb,
                ):
                    _archive_previous_attempt(entry, inspection, state_path, claim)
                run_isolated = use_isolation and not entry.malformed_error
                run_in_process = not run_isolated

            if run_isolated:
                effective_row = _with_capped_execution_timeout(
                    entry.row,
                    remaining_run_seconds,
                )
                child_started_wall = time.time()
                outcome = execute_row(
                    effective_row,
                    deadline_monotonic=worker_deadline,
                    on_tick=lambda claim=claim: claims.heartbeat(claim),
                )
                if outcome.killed_by_parent:
                    abnormal_count = attempts.record(entry.run_id, "walltime_kill")
                    if (
                        max_abnormal_attempts is not None
                        and abnormal_count >= max_abnormal_attempts
                    ):
                        _write_abnormal_attempts_result(entry, abnormal_count, worker_id)
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
                    final_inspection = _inspect_entry(entry)
                    result_is_fresh = _result_written_since(
                        entry.row["output_path"],
                        child_started_wall,
                    )
                    if final_inspection.state in COMPLETED_RESULT_STATES:
                        attempts.clear(entry.run_id)
                        stats.completed += 1
                        _progress("completed", entry.run_id, stats, monotonic() - started)
                    elif final_inspection.state in TERMINAL_FAILURE_STATES and (
                        not outcome.abnormal or result_is_fresh
                    ):
                        attempts.clear(entry.run_id)
                        stats.failed += 1
                        _progress(
                            "failed",
                            entry.run_id,
                            stats,
                            monotonic() - started,
                            detail=f"result_state={final_inspection.state.value}",
                        )
                    elif outcome.abnormal:
                        error = _termination_error_message(outcome, child_memory_limit_mb)
                        _write_termination_result(
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
                            detail=f"result_state={final_inspection.state.value}",
                        )
            elif run_in_process:
                if entry.malformed_error:
                    _write_malformed_result(entry, worker_id)
                    written_path = Path(entry.row["output_path"])
                else:
                    effective_row = _with_capped_execution_timeout(
                        entry.row,
                        remaining_run_seconds,
                    )
                    written_path = run_row(
                        effective_row,
                        command_args=command_args,
                        force=False,
                    )

                final_inspection = inspect_result_file(entry.row, written_path)
                if final_inspection.state in COMPLETED_RESULT_STATES:
                    attempts.clear(entry.run_id)
                    stats.completed += 1
                    _progress("completed", entry.run_id, stats, monotonic() - started)
                else:
                    if final_inspection.state in TERMINAL_FAILURE_STATES:
                        attempts.clear(entry.run_id)
                    stats.failed += 1
                    _progress(
                        "failed",
                        entry.run_id,
                        stats,
                        monotonic() - started,
                        detail=f"result_state={final_inspection.state.value}",
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


def _inspect_entry(entry: DynamicManifestEntry) -> ResultFileInspection:
    return inspect_result_file(entry.row, entry.row["output_path"])


def _result_skip_reason(
    inspection: ResultFileInspection,
    *,
    retry_failed: bool,
    retry_failed_only: bool,
    retry_failed_with_more_memory: bool = False,
    child_memory_limit_mb: float | None = None,
) -> str | None:
    if inspection.state in COMPLETED_RESULT_STATES:
        return "success"
    if inspection.state in TERMINAL_FAILURE_STATES:
        if retry_failed_with_more_memory:
            if inspection.state is not ResultFileState.FAILED:
                return "failed"
            previous_limit = _recorded_child_memory_limit_mb(inspection.payload)
            assert child_memory_limit_mb is not None
            if previous_limit is None or child_memory_limit_mb > previous_limit:
                return None
            return "failed"
        if retry_failed:
            return None
        if retry_failed_only and inspection.state is ResultFileState.FAILED:
            return None
        return "failed"
    return None


def _should_retry_existing_result(
    inspection: ResultFileInspection,
    *,
    retry_failed: bool,
    retry_failed_only: bool,
    retry_failed_with_more_memory: bool = False,
    child_memory_limit_mb: float | None = None,
) -> bool:
    if inspection.state not in TERMINAL_FAILURE_STATES:
        return False
    if retry_failed_with_more_memory:
        if inspection.state is not ResultFileState.FAILED:
            return False
        previous_limit = _recorded_child_memory_limit_mb(inspection.payload)
        assert child_memory_limit_mb is not None
        return previous_limit is None or child_memory_limit_mb > previous_limit
    if retry_failed:
        return True
    return retry_failed_only and inspection.state is ResultFileState.FAILED


def _recorded_child_memory_limit_mb(payload: dict[str, Any] | None) -> float | None:
    """Return the effective per-child memory limit recorded by a discovery result."""
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
            value = float(command_args[command_args.index("--child-memory-limit-mb") + 1])
            if value > 0:
                return value
        except (ValueError, IndexError, TypeError):
            pass
    slurm = metadata.get("slurm")
    if not isinstance(slurm, dict):
        return None
    try:
        allocation_mb = float(slurm.get("requested_memory_bytes")) / (1024 * 1024)
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


def _record_result_skip(stats: WorkerStats, reason: str) -> None:
    if reason == "success":
        stats.skipped_success += 1
    elif reason == "failed":
        stats.skipped_failed += 1


def _claim_metadata(
    entry: DynamicManifestEntry,
    worker_id: int,
    retry_failed: bool,
    retry_failed_only: bool,
    retry_failed_with_more_memory: bool,
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
    }


def _archive_previous_attempt(
    entry: DynamicManifestEntry,
    inspection: ResultFileInspection,
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
    archive_path = attempts_dir / (f"{timestamp}_{inspection.state.value}_{claim.token}.json")
    try:
        with archive_path.open("xb") as destination:
            with source_path.open("rb") as source:
                shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError:
        return None
    return archive_path


def _with_capped_execution_timeout(
    row: Mapping[str, str],
    remaining_run_seconds: float,
) -> dict[str, str]:
    effective = dict(row)
    raw_params = row.get("algorithm_params_json") or row.get("params_json") or "{}"
    try:
        params = json.loads(raw_params)
    except (TypeError, json.JSONDecodeError):
        return effective
    if not isinstance(params, dict):
        return effective

    changed = False
    cap_seconds = max(0.001, remaining_run_seconds)
    for field in TIMEOUT_CONTROL_FIELDS:
        if field not in params:
            continue
        try:
            configured_seconds = float(params[field])
        except (TypeError, ValueError):
            continue
        if (
            configured_seconds <= 0
            or configured_seconds <= cap_seconds
            or configured_seconds - cap_seconds < 1.0
        ):
            continue
        params[field] = cap_seconds
        changed = True
    if changed:
        params_json = json.dumps(params, sort_keys=True)
        effective["algorithm_params_json"] = params_json
        effective["params_json"] = params_json
    return effective


def _write_malformed_result(entry: DynamicManifestEntry, worker_id: int) -> None:
    error = f"Malformed manifest row {entry.row_index}: {entry.malformed_error}"
    output_path = Path(entry.row["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(
        output_path,
        _failed_result_payload(entry, error, worker_id),
    )


def _write_worker_exception_result(
    entry: DynamicManifestEntry,
    exc: Exception,
    worker_id: int,
) -> None:
    error = f"Dynamic worker error: {type(exc).__name__}: {exc}"
    try:
        output_path = Path(entry.row["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(
            output_path,
            _failed_result_payload(entry, error, worker_id),
        )
    except Exception:
        pass


def _result_written_since(output_path: str | Path, started_wall_seconds: float) -> bool:
    try:
        return Path(output_path).stat().st_mtime >= started_wall_seconds - 1.0
    except OSError:
        return False


def _termination_metadata(
    outcome: RowExecutionOutcome,
    child_memory_limit_mb: float | None,
) -> dict[str, Any]:
    return {
        "exit_code": outcome.exit_code,
        "signal": outcome.signal_name,
        "killed_by_parent": outcome.killed_by_parent,
        "oom_suspected": outcome.oom_suspected,
        "child_peak_rss_bytes": outcome.child_peak_rss_bytes,
        "child_memory_limit_mb": child_memory_limit_mb,
        "duration_seconds": outcome.duration_seconds,
    }


def _termination_error_message(
    outcome: RowExecutionOutcome,
    child_memory_limit_mb: float | None,
) -> str:
    if outcome.signal_name:
        cause = f"killed by {outcome.signal_name}"
    else:
        cause = f"exited with code {outcome.exit_code}"
    message = (
        f"Worker subprocess {cause} after {outcome.duration_seconds:.0f}s without writing a result"
    )
    details = []
    if outcome.oom_suspected:
        details.append("probable OOM kill")
    if outcome.child_peak_rss_bytes:
        details.append(f"child peak RSS ~ {outcome.child_peak_rss_bytes / 1024**3:.1f} GB")
    if child_memory_limit_mb:
        details.append(f"memory limit {child_memory_limit_mb:g} MB")
    if details:
        message = f"{message} ({'; '.join(details)})"
    return message


def _write_termination_result(
    entry: DynamicManifestEntry,
    outcome: RowExecutionOutcome,
    worker_id: int,
    child_memory_limit_mb: float | None,
) -> None:
    error = _termination_error_message(outcome, child_memory_limit_mb)
    try:
        payload = _failed_result_payload(entry, error, worker_id)
        payload["metadata"]["dynamic_worker"]["termination"] = _termination_metadata(
            outcome,
            child_memory_limit_mb,
        )
        output_path = Path(entry.row["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(output_path, payload)
    except Exception:
        pass


def _write_abnormal_attempts_result(
    entry: DynamicManifestEntry,
    abnormal_count: int,
    worker_id: int,
) -> None:
    error = (
        f"Giving up after {abnormal_count} abnormal attempts: previous workers died or hit "
        "the walltime boundary while running this row. Rerun with --retry-failed, more "
        "memory, or a longer worker walltime."
    )
    try:
        payload = _failed_result_payload(entry, error, worker_id)
        payload["metadata"]["dynamic_worker"]["abnormal_attempts"] = abnormal_count
        output_path = Path(entry.row["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(output_path, payload)
    except Exception:
        pass


def _failed_result_payload(
    entry: DynamicManifestEntry,
    error: str,
    worker_id: int,
) -> dict[str, Any]:
    row = entry.row
    try:
        seed = int(row.get("seed") or 0)
    except ValueError:
        seed = 0
    return {
        "experiment_id": row.get("experiment_id") or "manifest",
        "log_id": row.get("log_id") or f"row_{entry.row_index}",
        "log_path": row.get("log_path") or "",
        "test_log_path": row.get("test_log_path") or row.get("log_path") or "",
        "seed": seed,
        "algorithm_name": row.get("algorithm_id") or row.get("algorithm") or "unknown",
        "backend": row.get("backend") or "unknown",
        "hyperparameters": {},
        "discovered_model_type": "unknown",
        "model_path": None,
        "metrics": {},
        "metric_statuses": {},
        "runtime_seconds": None,
        "status": "failed",
        "error_message": error,
        "warnings": [],
        "metadata": {
            "config_hash": row.get("config_hash") or entry.run_id,
            "manifest_row_index": entry.row_index,
            "dynamic_worker": {
                "worker_id": worker_id,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                "hostname": socket.gethostname(),
            },
        },
    }


def _fallback_malformed_row(
    raw: Mapping[str, str],
    row_index: int,
    run_id: str,
    results_dir: str | Path | None,
) -> dict[str, str]:
    row = dict(raw)
    row["experiment_id"] = row.get("experiment_id") or "manifest"
    row["log_id"] = row.get("log_id") or f"row_{row_index}"
    row["log_path"] = row.get("log_path") or ""
    row["test_log_path"] = row.get("test_log_path") or row["log_path"]
    row["seed"] = row.get("seed") or "0"
    row["algorithm_id"] = row.get("algorithm_id") or row.get("algorithm") or "unknown"
    row["algorithm"] = row.get("algorithm") or row["algorithm_id"]
    row["backend"] = row.get("backend") or "unknown"
    row["config_id"] = row.get("config_id") or run_id
    row["config_hash"] = row.get("config_hash") or run_id
    row["run_id"] = row.get("run_id") or run_id
    output_path = row.get("output_path")
    if output_path and results_dir is not None:
        row["output_path"] = portable_project_path(_rebase_result_path(output_path, results_dir))
    return row


def _ensure_malformed_output_path(
    entry: DynamicManifestEntry,
    state_dir: Path,
) -> DynamicManifestEntry:
    if entry.row.get("output_path"):
        return entry
    row = dict(entry.row)
    row["output_path"] = (
        state_dir / "malformed_results" / f"{_claim_key(entry.run_id)}.json"
    ).as_posix()
    return DynamicManifestEntry(
        row_index=entry.row_index,
        row=row,
        run_id=entry.run_id,
        malformed_error=entry.malformed_error,
    )


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


def _claim_age_seconds(claim_path: Path) -> float | None:
    # Heartbeats refresh the claim file's mtime, so liveness is the newest of
    # the recorded claim timestamp and the mtime.
    timestamps: list[float] = []
    try:
        with claim_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_timestamp = payload.get("claimed_at_epoch_seconds")
        if raw_timestamp is not None:
            timestamps.append(float(raw_timestamp))
    except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
        pass
    try:
        timestamps.append(claim_path.stat().st_mtime)
    except (FileNotFoundError, OSError):
        pass
    if not timestamps:
        return None
    return max(0.0, time.time() - max(timestamps))


def _read_claim_token(claim_path: Path) -> str | None:
    try:
        with claim_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    token = payload.get("claim_token") if isinstance(payload, dict) else None
    return str(token) if token else None


def _validate_worker_limits(
    *,
    worker_walltime_seconds: float,
    safety_margin_seconds: float,
    max_runs_per_worker: int | None,
    reclaim_stale_after_seconds: float | None,
    max_abnormal_attempts: int | None = None,
    heartbeat_interval_seconds: float = 60.0,
    child_memory_limit_mb: float | None = None,
) -> None:
    if worker_walltime_seconds <= 0:
        raise ValueError("worker_walltime_seconds must be greater than 0")
    if safety_margin_seconds < 0:
        raise ValueError("safety_margin_seconds must be greater than or equal to 0")
    if max_runs_per_worker is not None and max_runs_per_worker <= 0:
        raise ValueError("max_runs_per_worker must be greater than 0")
    if reclaim_stale_after_seconds is not None and reclaim_stale_after_seconds <= 0:
        raise ValueError("reclaim_stale_after_seconds must be greater than 0")
    if max_abnormal_attempts is not None and max_abnormal_attempts <= 0:
        raise ValueError("max_abnormal_attempts must be greater than 0")
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be greater than 0")
    if child_memory_limit_mb is not None and child_memory_limit_mb <= 0:
        raise ValueError("child_memory_limit_mb must be greater than 0")


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
