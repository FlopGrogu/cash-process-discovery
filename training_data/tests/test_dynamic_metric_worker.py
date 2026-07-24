from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from process_discovery_cash.cli.run_dynamic_metric_worker import (
    _child_base_argv,
    default_metric_state_dir,
)
from process_discovery_cash.cli.run_dynamic_metric_worker import (
    build_parser as build_dynamic_metric_worker_parser,
)
from process_discovery_cash.experiments.dynamic_metric_worker import (
    _recorded_child_memory_limit_mb,
    _result_skip_reason,
    inspect_metric_result_file,
    load_dynamic_metric_manifest_entries,
    run_dynamic_metric_worker,
)
from process_discovery_cash.experiments.dynamic_worker import AttemptTracker, ClaimManager
from process_discovery_cash.experiments.run_isolation import RowExecutionOutcome


def test_two_workers_cannot_claim_the_same_metric_run(tmp_path: Path) -> None:
    manager = ClaimManager(tmp_path / "state")
    barrier = Barrier(2)

    def claim(worker_id: int):
        barrier.wait()
        return manager.try_claim("metric-run-1", {"worker_id": worker_id})

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = list(executor.map(claim, [1, 2]))

    assert sum(attempt.claim is not None for attempt in attempts) == 1
    assert sum(attempt.reason == "already_claimed" for attempt in attempts) == 1


def test_existing_successful_metric_result_is_skipped(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    _write_metric_status(Path(entries[0].row["output_path"]), entries[0].row, "success")

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: _raise("successful row should not run"),
    )

    assert stats.skipped_success == 1
    assert stats.claimed == 0


def test_failed_metric_result_is_skipped_by_default(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    _write_metric_status(Path(entries[0].row["output_path"]), entries[0].row, "failed")

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: _raise("failed row should not run by default"),
    )

    assert stats.skipped_failed == 1
    assert stats.claimed == 0


def test_failed_metric_result_can_be_retried_and_is_archived(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_metric_status(output_path, entries[0].row, "failed")
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["source_config_hash"])
        _write_metric_status(Path(row["output_path"]), row, "success")
        return Path(row["output_path"])

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed=True,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1
    assert len(list((tmp_path / "state" / "attempts" / entries[0].run_id).glob("*.json"))) == 1


def test_failed_metric_result_can_be_retried_with_failed_only_and_is_archived(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_metric_status(output_path, entries[0].row, "failed")
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["source_config_hash"])
        _write_metric_status(Path(row["output_path"]), row, "success")
        return Path(row["output_path"])

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed_only=True,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1
    assert len(list((tmp_path / "state" / "attempts" / entries[0].run_id).glob("*.json"))) == 1


def test_top_level_metric_timeout_is_skipped_with_failed_only_retry(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_metric_status(output_path, entries[0].row, "timeout")

    inspection = inspect_metric_result_file(entries[0].row, output_path)
    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed_only=True,
        run_row=lambda *_args, **_kwargs: _raise("timeout row should not run"),
    )

    assert inspection.state == "timeout"
    assert stats.skipped_failed == 1
    assert stats.claimed == 0


def test_top_level_metric_timeout_still_retries_with_broad_retry_failed(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_metric_status(output_path, entries[0].row, "timeout")
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["source_config_hash"])
        _write_metric_status(Path(row["output_path"]), row, "success")
        return Path(row["output_path"])

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed=True,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1


def test_dynamic_metric_worker_parser_accepts_retry_failed_only() -> None:
    args = build_dynamic_metric_worker_parser().parse_args(
        ["--manifest", "metrics.csv", "--retry-failed-only"]
    )

    assert args.retry_failed is False
    assert args.retry_failed_only is True


def test_dynamic_metric_worker_parser_accepts_memory_aware_failed_retry() -> None:
    args = build_dynamic_metric_worker_parser().parse_args(
        [
            "--manifest",
            "metrics.csv",
            "--retry-failed-with-more-memory",
            "--child-memory-limit-mb",
            "16384",
        ]
    )

    assert args.retry_failed_with_more_memory is True
    assert args.child_memory_limit_mb == 16384


def test_memory_aware_retry_only_reruns_failure_from_smaller_child_limit(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 2)
    for entry, previous_limit in zip(entries, (4096, 16384), strict=True):
        output_path = Path(entry.row["output_path"])
        _write_metric_status(output_path, entry.row, "failed")
        _record_child_memory_limit(output_path, previous_limit)
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["source_config_hash"])
        _write_metric_status(Path(row["output_path"]), row, "success")
        return Path(row["output_path"])

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed_with_more_memory=True,
        child_memory_limit_mb=16384,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1
    assert stats.skipped_failed == 1


def test_metric_recorded_limit_does_not_infer_unrecorded_worker_count() -> None:
    payload = {
        "metadata": {
            "slurm": {
                "requested_memory_bytes": 64 * 1024**3,
                "requested_cpus_per_task": 8,
            }
        }
    }

    assert _recorded_child_memory_limit_mb(payload) == 65536


def test_metric_recorded_limit_reads_termination_metadata() -> None:
    payload = {"metadata": {"dynamic_worker": {"termination": {"child_memory_limit_mb": 4096}}}}

    assert _recorded_child_memory_limit_mb(payload) == 4096


def test_memory_aware_retry_applies_to_partial_success_metric_failures(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_partial_metric_success(output_path, entries[0].row, failed_status="backend_error")
    _record_child_memory_limit(output_path, 4096)
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["source_config_hash"])
        _write_metric_status(Path(row["output_path"]), row, "success")
        return Path(row["output_path"])

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed_with_more_memory=True,
        child_memory_limit_mb=16384,
        run_row=fake_run,
    )

    assert calls == ["cfg_0"]
    assert stats.completed == 1


def test_memory_aware_retry_skips_partial_success_at_same_child_limit(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_partial_metric_success(output_path, entries[0].row, failed_status="backend_error")
    _record_child_memory_limit(output_path, 16384)

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed_with_more_memory=True,
        child_memory_limit_mb=16384,
        run_row=lambda *_args, **_kwargs: _raise(
            "partial success at the same child limit should not rerun"
        ),
    )

    assert stats.claimed == 0
    assert stats.skipped_failed == 1


def test_dynamic_metric_worker_parser_accepts_num_workers() -> None:
    args = build_dynamic_metric_worker_parser().parse_args(
        ["--manifest", "metrics.csv", "--num-workers", "4"]
    )

    assert args.num_workers == 4


def test_metric_pool_child_argv_forwards_single_worker_options() -> None:
    args = build_dynamic_metric_worker_parser().parse_args(
        [
            "--manifest",
            "metrics.csv",
            "--results-dir",
            "alt-results",
            "--worker-walltime-seconds",
            "100",
            "--safety-margin-seconds",
            "5",
            "--max-runs-per-worker",
            "2",
            "--retry-failed-only",
            "--reclaim-stale-after-seconds",
            "30",
            "--child-memory-limit-mb",
            "4096",
            "--metric-timeout-seconds",
            "120",
            "--no-isolate-runs",
            "--num-workers",
            "4",
        ]
    )

    argv = _child_base_argv(args, "state-dir")

    assert "--num-workers" in argv
    assert argv[argv.index("--num-workers") + 1] == "1"
    assert argv[argv.index("--state-dir") + 1] == "state-dir"
    assert argv[argv.index("--results-dir") + 1] == "alt-results"
    assert argv[argv.index("--child-memory-limit-mb") + 1] == "4096.0"
    assert argv[argv.index("--metric-timeout-seconds") + 1] == "120.0"
    assert "--retry-failed-only" in argv
    assert "--no-isolate-runs" in argv


def test_dynamic_metric_slurm_wrapper_forwards_num_workers() -> None:
    script = Path("slurm/run_dynamic_metric_manifest.slurm").read_text(encoding="utf-8")

    assert 'NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"' in script
    assert 'CHILD_MEMORY_LIMIT_MB="$(( ALLOCATION_MEMORY_MB / NUM_WORKERS ))"' in script
    assert 'worker_args+=(--num-workers "${NUM_WORKERS}")' in script


def test_partial_success_metric_result_is_skipped_as_failed_by_default(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    _write_partial_metric_success(
        Path(entries[0].row["output_path"]),
        entries[0].row,
        failed_status="backend_error",
    )

    inspection = inspect_metric_result_file(entries[0].row, entries[0].row["output_path"])
    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: _raise("partial row should not run by default"),
    )

    assert inspection.state == "failed"
    assert _result_skip_reason(inspection, retry_failed=False) == "failed"
    assert stats.skipped_failed == 1
    assert stats.claimed == 0


def test_partial_success_missing_metric_status_is_skipped_as_failed_by_default(
    tmp_path: Path,
) -> None:
    row = _metric_row(tmp_path, 0)
    row["metric_names_json"] = json.dumps(["fitness", "precision"])
    manifest_path = tmp_path / "metrics.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    entries = load_dynamic_metric_manifest_entries(manifest_path)
    _write_metric_status(Path(entries[0].row["output_path"]), entries[0].row, "success")

    inspection = inspect_metric_result_file(entries[0].row, entries[0].row["output_path"])
    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: _raise("partial row should not run by default"),
    )

    assert inspection.state == "failed"
    assert stats.skipped_failed == 1
    assert stats.claimed == 0


def test_partial_success_metric_result_can_be_retried_and_forces_recompute(
    tmp_path: Path,
) -> None:
    entries = _entries(tmp_path, 1)
    output_path = Path(entries[0].row["output_path"])
    _write_partial_metric_success(output_path, entries[0].row, failed_status="backend_error")
    calls: list[bool] = []

    def fake_run(row: dict[str, str], **kwargs) -> Path:
        calls.append(bool(kwargs.get("force")))
        _write_metric_status(Path(row["output_path"]), row, "success")
        return Path(row["output_path"])

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed=True,
        run_row=fake_run,
    )

    assert calls == [True]
    assert stats.completed == 1
    assert len(list((tmp_path / "state" / "attempts" / entries[0].run_id).glob("*.json"))) == 1


def test_metric_worker_stops_claiming_when_safety_margin_is_reached(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 2)
    clock = _Clock()
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["source_config_hash"])
        _write_metric_status(Path(row["output_path"]), row, "success")
        clock.value = 95
        return Path(row["output_path"])

    stats = run_dynamic_metric_worker(
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


def test_metric_rerun_after_partial_completion_continues_remaining_rows(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 3)
    calls: list[str] = []

    def fake_run(row: dict[str, str], **_kwargs) -> Path:
        calls.append(row["source_config_hash"])
        _write_metric_status(Path(row["output_path"]), row, "success")
        return Path(row["output_path"])

    first = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        max_runs_per_worker=1,
        run_row=fake_run,
    )
    second = run_dynamic_metric_worker(
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


def test_stale_metric_claim_can_be_reclaimed_explicitly(tmp_path: Path) -> None:
    manager = ClaimManager(tmp_path / "state")
    first = manager.try_claim("metric-run-1", {"worker_id": 1})
    assert first.claim is not None
    payload = json.loads(first.claim.path.read_text(encoding="utf-8"))
    payload["claimed_at_epoch_seconds"] = 0
    first.claim.path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(first.claim.path, (0, 0))

    second = manager.try_claim(
        "metric-run-1",
        {"worker_id": 2},
        reclaim_stale_after_seconds=1,
        success_exists=lambda: False,
    )

    assert second.claim is not None
    assert second.claim.stale_reclaimed is True
    assert second.claim.token != first.claim.token
    assert manager.release(first.claim) is False
    assert manager.release(second.claim) is True


def test_malformed_metric_row_writes_failed_result_and_does_not_abort(tmp_path: Path) -> None:
    row = _metric_row(tmp_path, 0)
    row["metric_names_json"] = "{bad"
    manifest_path = tmp_path / "malformed_metrics.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    entries = load_dynamic_metric_manifest_entries(manifest_path)

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: _raise("malformed row must not reach runner"),
    )

    payload = json.loads(Path(entries[0].row["output_path"]).read_text(encoding="utf-8"))
    assert stats.failed == 1
    assert payload["status"] == "failed"
    assert "Malformed metric manifest row 0" in payload["error_message"]
    assert payload["metadata"]["dynamic_worker"]["worker_id"] == 0


def test_dynamic_metric_worker_accepts_missing_model_path_and_completes(tmp_path: Path) -> None:
    row = _metric_row(tmp_path, 0)
    row["model_path"] = ""
    source_result_path = Path(row["source_result_path"])
    source_result_path.parent.mkdir(parents=True, exist_ok=True)
    source_result_path.write_text(
        json.dumps(
            {
                "status": "success",
                "experiment_id": row["experiment_id"],
                "log_id": row["log_id"],
                "algorithm_name": row["algorithm_name"],
                "backend": "pm4py",
                "hyperparameters": {},
                "runtime_seconds": 1.0,
                "log_path": "data/train.xes",
                "test_log_path": row["test_log_path"],
                "seed": 0,
                "warnings": [],
                "metadata": {"config_hash": row["source_config_hash"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "metrics.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    entries = load_dynamic_metric_manifest_entries(manifest_path)

    assert entries[0].malformed_error is None

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda row, **_kwargs: _write_zero_metric_success(Path(row["output_path"]), row),
    )

    payload = json.loads(Path(entries[0].row["output_path"]).read_text(encoding="utf-8"))
    assert stats.completed == 1
    assert payload["status"] == "success_missing"
    assert payload["metrics"] == {"fitness": 0.0}
    assert payload["metric_statuses"]["fitness"]["status"] == "missing_model"


def test_metric_claim_identity_distinguishes_profile_and_output_path(tmp_path: Path) -> None:
    manifest_path = tmp_path / "metrics.csv"
    first = _metric_row(tmp_path, 0)
    second = dict(first)
    second["metric_profile"] = "alignment"
    second["output_path"] = (tmp_path / "results" / "metric_alignment.json").as_posix()
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerow(second)

    entries = load_dynamic_metric_manifest_entries(manifest_path)

    assert entries[0].run_id != entries[1].run_id


def test_metric_result_identity_mismatch_is_not_treated_as_success(tmp_path: Path) -> None:
    row = _metric_row(tmp_path, 0)
    output_path = Path(row["output_path"])
    _write_metric_status(output_path, {**row, "source_config_hash": "other"}, "success")

    inspection = inspect_metric_result_file(row, output_path)

    assert inspection.state == "identity_mismatch"


def test_stale_success_missing_metric_result_is_recoverable(tmp_path: Path) -> None:
    row = _metric_row(tmp_path, 0)
    _write_success_source_with_model(row)
    output_path = Path(row["output_path"])
    _write_zero_metric_success(output_path, row)

    inspection = inspect_metric_result_file(row, output_path)

    assert inspection.state == "recoverable"
    # A recoverable placeholder must not be skipped so the worker reruns it.
    assert _result_skip_reason(inspection, retry_failed=False) is None


def test_genuine_success_missing_metric_result_stays_terminal(tmp_path: Path) -> None:
    # No source discovery result on disk -> no model can be recovered, so the
    # success_missing output is a legitimate completed result.
    row = _metric_row(tmp_path, 0)
    row["model_path"] = ""
    output_path = Path(row["output_path"])
    _write_zero_metric_success(output_path, row)

    inspection = inspect_metric_result_file(row, output_path)

    assert inspection.state == "success_complete"
    assert _result_skip_reason(inspection, retry_failed=False) == "success"


def test_worker_recomputes_stale_success_missing_when_model_available(tmp_path: Path) -> None:
    row = _metric_row(tmp_path, 0)
    _write_success_source_with_model(row)
    output_path = Path(row["output_path"])
    _write_zero_metric_success(output_path, row)

    manifest_path = tmp_path / "metrics.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    entries = load_dynamic_metric_manifest_entries(manifest_path)

    def rerun(row, **_kwargs):
        _write_metric_status(Path(row["output_path"]), row, "success")
        return Path(row["output_path"])

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=rerun,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert stats.completed == 1
    assert payload["status"] == "success"
    assert payload["metrics"] == {"fitness": 1.0}


def test_default_metric_state_dir_rejects_pre_v6_manifest_path(tmp_path: Path) -> None:
    manifest_path = (
        tmp_path
        / "experiments"
        / "manifests"
        / "v5"
        / "metrics"
        / "inductiveim"
        / "synthetic"
        / "v2"
        / "token_metrics.csv"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row = _metric_row(tmp_path, 0)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    entries = load_dynamic_metric_manifest_entries(manifest_path)

    with pytest.raises(ValueError, match="Only v6 manifest paths"):
        default_metric_state_dir(manifest_path.as_posix(), entries)


def test_default_metric_state_dir_falls_back_to_algorithm_name(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)

    assert (
        default_metric_state_dir((tmp_path / "metrics.csv").as_posix(), entries)
        == "runs/metric_state/alpha_miner"
    )


def test_abnormal_metric_child_death_writes_synthetic_failed_result(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        execute_row=lambda *_args, **_kwargs: _metric_outcome(
            exit_code=-9,
            signal_name="SIGKILL",
            oom_suspected=True,
        ),
    )

    assert stats.failed == 1
    payload = json.loads(Path(entries[0].row["output_path"]).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "SIGKILL" in payload["error_message"]
    termination = payload["metadata"]["dynamic_worker"]["termination"]
    assert termination["oom_suspected"] is True

    rerun = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        execute_row=lambda *_args, **_kwargs: _raise("terminal row must not rerun"),
    )
    assert rerun.skipped_failed == 1


def test_metric_walltime_kill_leaves_row_missing_and_records_attempt(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        execute_row=lambda *_args, **_kwargs: _metric_outcome(
            exit_code=None,
            killed_by_parent=True,
        ),
    )

    assert stats.claimed == 1
    assert stats.failed == 0
    assert not Path(entries[0].row["output_path"]).exists()
    assert AttemptTracker(tmp_path / "state").count(entries[0].run_id) == 1


def test_successful_isolated_metric_run_completes(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)

    def fake_execute(row, **_kwargs) -> RowExecutionOutcome:
        _write_metric_status(Path(row["output_path"]), row, "success")
        return _metric_outcome()

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        execute_row=fake_execute,
    )

    assert stats.completed == 1


def test_isolated_partial_success_retry_passes_force(tmp_path: Path) -> None:
    entries = _entries(tmp_path, 1)
    _write_partial_metric_success(
        Path(entries[0].row["output_path"]),
        entries[0].row,
        failed_status="backend_error",
    )
    force_values: list[bool] = []

    def fake_execute(row, **kwargs) -> RowExecutionOutcome:
        force_values.append(bool(kwargs.get("force")))
        _write_metric_status(Path(row["output_path"]), row, "success")
        return _metric_outcome()

    stats = run_dynamic_metric_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        retry_failed=True,
        execute_row=fake_execute,
    )

    assert force_values == [True]
    assert stats.completed == 1


def _metric_outcome(
    *,
    exit_code: int | None = 0,
    signal_name: str | None = None,
    killed_by_parent: bool = False,
    oom_suspected: bool = False,
) -> RowExecutionOutcome:
    return RowExecutionOutcome(
        exit_code=exit_code,
        signal_name=signal_name,
        killed_by_parent=killed_by_parent,
        oom_suspected=oom_suspected,
        child_peak_rss_bytes=None,
        duration_seconds=5.0,
    )


def _entries(tmp_path: Path, row_count: int) -> list:
    manifest_path = tmp_path / "metrics.csv"
    rows = [_metric_row(tmp_path, index) for index in range(row_count)]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return load_dynamic_metric_manifest_entries(manifest_path)


def _metric_row(tmp_path: Path, index: int) -> dict[str, str]:
    return {
        "source_result_path": (tmp_path / "discovery" / f"result_{index}.json").as_posix(),
        "source_config_hash": f"cfg_{index}",
        "experiment_id": "metrics_dynamic_test",
        "log_id": f"log_{index}",
        "algorithm_name": "alpha_miner",
        "model_path": (tmp_path / f"model_{index}.pnml").as_posix(),
        "metric_model_path": "",
        "test_log_path": "data/example/tiny_log.xes",
        "log_cache_key": f"log_{index}",
        "metric_profile": "token",
        "metric_names_json": json.dumps(["fitness"]),
        "log_dir": "logs/slurm/metrics/test/token",
        "output_path": (tmp_path / "results" / f"metric_{index}.json").as_posix(),
    }


def _write_success_source_with_model(row: dict[str, str]) -> Path:
    model_path = Path(row["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.touch()
    source_result_path = Path(row["source_result_path"])
    source_result_path.parent.mkdir(parents=True, exist_ok=True)
    source_result_path.write_text(
        json.dumps(
            {
                "status": "success",
                "experiment_id": row["experiment_id"],
                "log_id": row["log_id"],
                "algorithm_name": row["algorithm_name"],
                "backend": "pm4py",
                "hyperparameters": {},
                "runtime_seconds": 1.0,
                "log_path": "data/train.xes",
                "test_log_path": row["test_log_path"],
                "model_path": model_path.as_posix(),
                "seed": 0,
                "warnings": [],
                "metadata": {"config_hash": row["source_config_hash"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source_result_path


def _write_metric_status(path: Path, row: dict[str, str], status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "runtime_seconds": 0.1,
        "source_result_path": row["source_result_path"],
        "source_config_hash": row["source_config_hash"],
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "algorithm_name": row["algorithm_name"],
        "model_path": row["model_path"],
        "model_type": "petri_net",
        "test_log_path": row["test_log_path"],
        "metric_profile": row["metric_profile"],
        "metric_names": ["fitness"],
        "metrics": {"fitness": 1.0 if status == "success" else None},
        "metric_statuses": {
            "fitness": {
                "status": "success" if status == "success" else "not_computed",
                "value": 1.0 if status == "success" else None,
                "error": None if status == "success" else "failed",
            }
        },
        "error_message": None if status == "success" else "failed",
        "metadata": {},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _record_child_memory_limit(path: Path, limit_mb: float) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["command_args"] = [
        "scripts/run_dynamic_metric_worker.py",
        "--child-memory-limit-mb",
        str(limit_mb),
    ]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_partial_metric_success(
    path: Path,
    row: dict[str, str],
    *,
    failed_status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "success",
        "runtime_seconds": 0.1,
        "source_result_path": row["source_result_path"],
        "source_config_hash": row["source_config_hash"],
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "algorithm_name": row["algorithm_name"],
        "model_path": row["model_path"],
        "model_type": "petri_net",
        "test_log_path": row["test_log_path"],
        "metric_profile": row["metric_profile"],
        "metric_names": ["fitness"],
        "metrics": {"fitness": None},
        "metric_statuses": {
            "fitness": {
                "status": failed_status,
                "value": None,
                "error": "metric backend failed",
            }
        },
        "error_message": None,
        "metadata": {},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_zero_metric_success(path: Path, row: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "success_missing",
        "metric_runtime_seconds": 0.1,
        "discovery_runtime_seconds": 1.0,
        "discovery_status": "success",
        "source_result_path": row["source_result_path"],
        "source_config_hash": row["source_config_hash"],
        "config_hash": row["source_config_hash"],
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "log_path": "data/train.xes",
        "algorithm_name": row["algorithm_name"],
        "backend": "saved_model",
        "hyperparameters": {},
        "warnings": [],
        "model_path": row["model_path"],
        "discovered_model_type": None,
        "test_log_path": row["test_log_path"],
        "seed": 0,
        "metric_profile": row["metric_profile"],
        "metric_names": ["fitness"],
        "metrics": {"fitness": 0.0},
        "metric_statuses": {
            "fitness": {
                "status": "missing_model",
                "value": 0.0,
                "error": "defaulted metrics to 0",
            }
        },
        "error_message": None,
        "metadata": {},
        "source_metadata": {},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _raise(message: str):
    raise AssertionError(message)
