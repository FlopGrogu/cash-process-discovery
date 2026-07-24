from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from process_discovery_cash.cli.run_dynamic_worker import (
    build_parser as build_dynamic_worker_parser,
)
from process_discovery_cash.experiments.dynamic_paths import default_dynamic_state_dir
from process_discovery_cash.experiments.dynamic_worker import (
    AttemptTracker,
    ClaimManager,
    _recorded_child_memory_limit_mb,
    _with_capped_execution_timeout,
    deterministic_scan_order,
    load_dynamic_manifest_entries,
    run_dynamic_worker,
)
from process_discovery_cash.experiments.run_isolation import RowExecutionOutcome


def test_dynamic_worker_default_walltime_is_24_hours() -> None:
    args = build_dynamic_worker_parser().parse_args(["--manifest", "manifest.csv"])

    assert args.worker_walltime_seconds == 86400


def test_two_workers_cannot_claim_the_same_run(tmp_path: Path) -> None:
    manager = ClaimManager(tmp_path / "state")
    barrier = Barrier(2)

    def claim(worker_id: int):
        barrier.wait()
        return manager.try_claim("run-1", {"worker_id": worker_id})

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = list(executor.map(claim, [1, 2]))

    assert sum(attempt.claim is not None for attempt in attempts) == 1
    assert sum(attempt.reason == "already_claimed" for attempt in attempts) == 1


def test_existing_successful_result_is_skipped(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    _write_success(Path(entries[0].row["output_path"]), entries[0].row)

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: _raise("successful row should not run"),
    )

    assert stats.skipped_success == 1
    assert stats.claimed == 0


def test_failed_result_is_skipped_by_default(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    _write_status(Path(entries[0].row["output_path"]), "failed")

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: _raise("failed row should not run by default"),
    )

    assert stats.skipped_failed == 1
    assert stats.claimed == 0


def test_failed_result_can_be_retried_and_is_archived(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_status(output_path, "failed")
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["config_hash"])
        _write_success(Path(row["output_path"]), row)
        return Path(row["output_path"])

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed=True,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1
    assert len(list((tmp_path / "state" / "attempts" / "cfg_0").glob("*.json"))) == 1


def test_failed_result_can_be_retried_with_failed_only_and_is_archived(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_status(output_path, "failed")
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["config_hash"])
        _write_success(Path(row["output_path"]), row)
        return Path(row["output_path"])

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed_only=True,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1
    assert len(list((tmp_path / "state" / "attempts" / "cfg_0").glob("*.json"))) == 1


def test_timeout_result_is_skipped_with_failed_only_retry(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    _write_status(Path(entries[0].row["output_path"]), "timeout")

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed_only=True,
        run_row=lambda *_args, **_kwargs: _raise("timeout row should not run"),
    )

    assert stats.skipped_failed == 1
    assert stats.claimed == 0


def test_timeout_result_still_retries_with_broad_retry_failed(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    _write_status(Path(entries[0].row["output_path"]), "timeout")
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["config_hash"])
        _write_success(Path(row["output_path"]), row)
        return Path(row["output_path"])

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed=True,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1
    assert len(list((tmp_path / "state" / "attempts" / "cfg_0").glob("*.json"))) == 1


def test_dynamic_worker_parser_accepts_retry_failed_only() -> None:
    args = build_dynamic_worker_parser().parse_args(
        ["--manifest", "manifest.csv", "--retry-failed-only"]
    )

    assert args.retry_failed is False
    assert args.retry_failed_only is True


def test_dynamic_worker_parser_accepts_memory_aware_failed_retry() -> None:
    args = build_dynamic_worker_parser().parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--retry-failed-with-more-memory",
            "--child-memory-limit-mb",
            "8192",
        ]
    )

    assert args.retry_failed_with_more_memory is True
    assert args.child_memory_limit_mb == 8192


def test_memory_aware_retry_only_reruns_failure_from_smaller_child_limit(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 2)
    for entry, previous_limit in zip(entries, (4096, 8192), strict=True):
        output_path = Path(entry.row["output_path"])
        _write_status(output_path, "failed")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        payload["metadata"] = {
            "command_args": ["worker", "--child-memory-limit-mb", str(previous_limit)]
        }
        output_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["config_hash"])
        _write_success(Path(row["output_path"]), row)
        return Path(row["output_path"])

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed_with_more_memory=True,
        child_memory_limit_mb=8192,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1
    assert stats.skipped_failed == 1


def test_memory_aware_retry_requires_child_memory_limit(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    try:
        run_dynamic_worker(
            entries,
            state_dir=tmp_path / "state",
            worker_walltime_seconds=100,
            safety_margin_seconds=10,
            retry_failed_with_more_memory=True,
        )
    except ValueError as exc:
        assert "requires child_memory_limit_mb" in str(exc)
    else:
        raise AssertionError("missing child memory limit should be rejected")


def test_recorded_child_limit_does_not_infer_unrecorded_worker_count() -> None:
    payload = {
        "metadata": {
            "slurm": {
                "requested_memory_bytes": 64 * 1024**3,
                "requested_cpus_per_task": 8,
            }
        }
    }

    assert _recorded_child_memory_limit_mb(payload) == 65536


def test_recorded_child_limit_reads_termination_metadata() -> None:
    payload = {"metadata": {"dynamic_worker": {"termination": {"child_memory_limit_mb": 4096}}}}

    assert _recorded_child_memory_limit_mb(payload) == 4096


def test_worker_stops_claiming_when_safety_margin_is_reached(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 2)
    clock = _Clock()
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["config_hash"])
        _write_success(Path(row["output_path"]), row)
        clock.value = 95
        return Path(row["output_path"])

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=fake_run,
        monotonic=clock,
    )

    assert calls == ["cfg_0"]
    assert stats.claimed == 1
    assert not Path(entries[1].row["output_path"]).exists()


def test_deterministic_scan_order_uses_worker_offset() -> None:
    assert deterministic_scan_order(5, 0) == [0, 1, 2, 3, 4]
    assert deterministic_scan_order(5, 2) == [2, 3, 4, 0, 1]
    assert deterministic_scan_order(5, 7) == [2, 3, 4, 0, 1]


def test_default_dynamic_state_dir_mirrors_manifest_path() -> None:
    assert (
        default_dynamic_state_dir("experiments/manifests/v6/model/baseline/split/v1.csv")
        == "runs/state/v6/model/baseline/split/v1"
    )


def test_rerun_after_partial_completion_continues_remaining_rows(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 3)
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["config_hash"])
        _write_success(Path(row["output_path"]), row)
        return Path(row["output_path"])

    first = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        max_runs_per_worker=1,
        run_row=fake_run,
    )
    second = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=fake_run,
    )

    assert first.completed == 1
    assert second.completed == 2
    assert second.skipped_success == 1
    assert calls == ["cfg_0", "cfg_1", "cfg_2"]


def test_stale_claim_can_be_reclaimed_explicitly(tmp_path: Path) -> None:
    manager = ClaimManager(tmp_path / "state")
    first = manager.try_claim("run-1", {"worker_id": 1})
    assert first.claim is not None
    payload = json.loads(first.claim.path.read_text(encoding="utf-8"))
    payload["claimed_at_epoch_seconds"] = 0
    first.claim.path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(first.claim.path, (0, 0))

    second = manager.try_claim(
        "run-1",
        {"worker_id": 2},
        reclaim_stale_after_seconds=1,
        success_exists=lambda: False,
    )

    assert second.claim is not None
    assert second.claim.stale_reclaimed is True
    assert second.claim.token != first.claim.token
    assert manager.release(first.claim) is False
    assert manager.release(second.claim) is True


def test_remaining_time_caps_runtime_control_without_changing_hash() -> None:
    row = {
        "config_hash": "stable-hash",
        "params_json": json.dumps(
            {"variant": "im", "discovery_timeout_seconds": 900},
            sort_keys=True,
        ),
        "algorithm_params_json": json.dumps(
            {"variant": "im", "discovery_timeout_seconds": 900},
            sort_keys=True,
        ),
    }

    effective = _with_capped_execution_timeout(row, 120)

    assert effective["config_hash"] == "stable-hash"
    assert json.loads(effective["params_json"])["discovery_timeout_seconds"] == 120
    assert json.loads(row["params_json"])["discovery_timeout_seconds"] == 900


def test_subsecond_worker_startup_does_not_reduce_24_hour_timeout() -> None:
    row = {
        "config_hash": "stable-hash",
        "params_json": json.dumps(
            {"discovery_timeout_seconds": 86400},
            sort_keys=True,
        ),
        "algorithm_params_json": json.dumps(
            {"discovery_timeout_seconds": 86400},
            sort_keys=True,
        ),
    }

    effective = _with_capped_execution_timeout(row, 86399.5)

    assert json.loads(effective["params_json"])["discovery_timeout_seconds"] == 86400


def test_remaining_time_does_not_cap_recursion_limit() -> None:
    row = {
        "config_hash": "stable-hash",
        "params_json": json.dumps(
            {
                "variant": "im",
                "discovery_timeout_seconds": 900,
                "recursion_limit": 10000,
            },
            sort_keys=True,
        ),
        "algorithm_params_json": json.dumps(
            {
                "variant": "im",
                "discovery_timeout_seconds": 900,
                "recursion_limit": 10000,
            },
            sort_keys=True,
        ),
    }

    effective = _with_capped_execution_timeout(row, 120)

    assert effective["config_hash"] == "stable-hash"
    assert json.loads(effective["params_json"])["discovery_timeout_seconds"] == 120
    assert json.loads(effective["params_json"])["recursion_limit"] == 10000


def test_malformed_row_writes_failed_result_and_does_not_abort(tmp_path: Path) -> None:
    row = _manifest_row(tmp_path, 0)
    row["params_json"] = "not: [valid"
    row["algorithm_params_json"] = "not: [valid"
    manifest_path = tmp_path / "malformed.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    entries = load_dynamic_manifest_entries(manifest_path)

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: _raise("malformed row must not reach runner"),
    )

    payload = json.loads(Path(entries[0].row["output_path"]).read_text(encoding="utf-8"))
    assert stats.failed == 1
    assert payload["status"] == "failed"
    assert "Malformed manifest row 0" in payload["error_message"]


def test_abnormal_child_death_writes_synthetic_failed_result(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        child_memory_limit_mb=16000,
        execute_row=lambda *_args, **_kwargs: _outcome(
            exit_code=-9,
            signal_name="SIGKILL",
            oom_suspected=True,
            child_peak_rss_bytes=14 * 1024**3,
        ),
    )

    assert stats.failed == 1
    payload = json.loads(Path(entries[0].row["output_path"]).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "SIGKILL" in payload["error_message"]
    assert "probable OOM kill" in payload["error_message"]
    termination = payload["metadata"]["dynamic_worker"]["termination"]
    assert termination["signal"] == "SIGKILL"
    assert termination["oom_suspected"] is True
    assert termination["child_memory_limit_mb"] == 16000

    rerun = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        execute_row=lambda *_args, **_kwargs: _raise("terminal row must not rerun"),
    )
    assert rerun.skipped_failed == 1


def test_abnormal_death_keeps_fresh_result_written_by_child(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)

    def fake_execute(row, **_kwargs) -> RowExecutionOutcome:
        _write_status(Path(row["output_path"]), "failed")
        return _outcome(exit_code=-9, signal_name="SIGKILL", oom_suspected=True)

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        execute_row=fake_execute,
    )

    assert stats.failed == 1
    payload = json.loads(Path(entries[0].row["output_path"]).read_text(encoding="utf-8"))
    assert "error_message" not in payload


def test_successful_isolated_run_completes_and_clears_attempts(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    tracker = AttemptTracker(tmp_path / "state")
    tracker.record(entries[0].run_id, "stale_claim_reclaimed")

    def fake_execute(row, **_kwargs) -> RowExecutionOutcome:
        _write_success(Path(row["output_path"]), row)
        return _outcome()

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        execute_row=fake_execute,
    )

    assert stats.completed == 1
    assert tracker.count(entries[0].run_id) == 0


def test_walltime_kill_leaves_row_missing_and_records_attempt(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 2)

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        execute_row=lambda *_args, **_kwargs: _outcome(killed_by_parent=True),
    )

    assert stats.claimed == 1
    assert stats.completed == 0
    assert stats.failed == 0
    assert not Path(entries[0].row["output_path"]).exists()
    assert AttemptTracker(tmp_path / "state").count(entries[0].run_id) == 1
    assert not list((tmp_path / "state" / "claims").glob("*.claim"))


def test_repeated_walltime_kills_give_up_with_terminal_result(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    kill = lambda *_args, **_kwargs: _outcome(killed_by_parent=True)  # noqa: E731

    first = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        max_abnormal_attempts=2,
        execute_row=kill,
    )
    second = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        max_abnormal_attempts=2,
        execute_row=kill,
    )

    assert first.failed == 0
    assert second.failed == 1
    payload = json.loads(Path(entries[0].row["output_path"]).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "Giving up after 2 abnormal attempts" in payload["error_message"]
    assert payload["metadata"]["dynamic_worker"]["abnormal_attempts"] == 2
    assert AttemptTracker(tmp_path / "state").count(entries[0].run_id) == 0


def test_stale_reclaim_counts_as_abnormal_attempt_and_can_give_up(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    manager = ClaimManager(tmp_path / "state")
    stale = manager.try_claim(entries[0].run_id, {"worker_id": 99})
    assert stale.claim is not None
    stale_payload = json.loads(stale.claim.path.read_text(encoding="utf-8"))
    stale_payload["claimed_at_epoch_seconds"] = 0
    stale.claim.path.write_text(json.dumps(stale_payload), encoding="utf-8")
    os.utime(stale.claim.path, (0, 0))

    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        reclaim_stale_after_seconds=1,
        max_abnormal_attempts=1,
        execute_row=lambda *_args, **_kwargs: _raise("row past attempt cap must not run"),
    )

    assert stats.stale_reclaimed == 1
    assert stats.failed == 1
    payload = json.loads(Path(entries[0].row["output_path"]).read_text(encoding="utf-8"))
    assert "Giving up after 1 abnormal attempts" in payload["error_message"]


def test_heartbeat_keeps_claim_from_being_reclaimed(tmp_path: Path) -> None:
    manager = ClaimManager(tmp_path / "state")
    first = manager.try_claim("run-1", {"worker_id": 1})
    assert first.claim is not None
    payload = json.loads(first.claim.path.read_text(encoding="utf-8"))
    payload["claimed_at_epoch_seconds"] = 0
    first.claim.path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(first.claim.path, (0, 0))

    assert manager.heartbeat(first.claim) is True
    second = manager.try_claim(
        "run-1",
        {"worker_id": 2},
        reclaim_stale_after_seconds=1,
        success_exists=lambda: False,
    )

    assert second.claim is None
    assert second.reason == "already_claimed"


def _outcome(
    *,
    exit_code: int | None = 0,
    signal_name: str | None = None,
    killed_by_parent: bool = False,
    oom_suspected: bool = False,
    child_peak_rss_bytes: int | None = None,
    duration_seconds: float = 12.0,
) -> RowExecutionOutcome:
    return RowExecutionOutcome(
        exit_code=exit_code,
        signal_name=signal_name,
        killed_by_parent=killed_by_parent,
        oom_suspected=oom_suspected,
        child_peak_rss_bytes=child_peak_rss_bytes,
        duration_seconds=duration_seconds,
    )


def _entries(tmp_path: Path, row_count: int):
    manifest_path = tmp_path / "manifest.csv"
    rows = [_manifest_row(tmp_path, index) for index in range(row_count)]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return load_dynamic_manifest_entries(manifest_path)


def _manifest_row(tmp_path: Path, index: int) -> dict[str, str]:
    params_json = json.dumps({"variant": "classic"}, sort_keys=True)
    return {
        "experiment_id": "dynamic_test",
        "log_id": f"log_{index}",
        "log_path": "data/example/tiny_log.xes",
        "test_log_path": "data/example/tiny_log.xes",
        "seed": "0",
        "algorithm_id": "alpha_miner",
        "algorithm": "alpha_miner",
        "backend": "pm4py",
        "algorithm_params_json": params_json,
        "params_json": params_json,
        "metrics_json": json.dumps({"enabled": False}),
        "config_id": f"cfg_{index}",
        "config_hash": f"cfg_{index}",
        "output_path": (tmp_path / "results" / f"result_{index}.json").as_posix(),
    }


def _write_status(path: Path, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status}) + "\n", encoding="utf-8")


def _write_success(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "log_path": row["log_path"],
        "test_log_path": row["test_log_path"],
        "seed": int(row["seed"]),
        "algorithm_name": row["algorithm_id"],
        "backend": row["backend"],
        "hyperparameters": json.loads(row["params_json"]),
        "discovered_model_type": "petri_net",
        "metrics": {},
        "metric_statuses": {},
        "status": "success",
        "metadata": {"config_hash": row["config_hash"]},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class _Clock:
    value = 0.0

    def __call__(self) -> float:
        return self.value


def _raise(message: str):
    raise AssertionError(message)
