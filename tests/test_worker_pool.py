from __future__ import annotations

import io
from pathlib import Path

import pytest

from process_discovery_cash.experiments.dynamic_worker import (
    DynamicManifestEntry,
    plan_dynamic_runs,
)
from process_discovery_cash.experiments.worker_pool import (
    SINGLE_THREAD_ENV_VARS,
    PoolResult,
    apply_single_thread_env,
    run_worker_pool,
    single_thread_env,
    worker_ids_for_pool,
)


class _FakePopen:
    """Minimal Popen stand-in that records argv/env and yields canned output."""

    instances: list[_FakePopen] = []

    def __init__(self, argv, *, stdout, stderr, text, env, exit_code=0, output=""):
        self.argv = list(argv)
        self.env = dict(env)
        self.stdout = io.StringIO(output)
        self._exit_code = exit_code
        self.returncode = None
        _FakePopen.instances.append(self)

    def wait(self) -> int:
        self.returncode = self._exit_code
        return self._exit_code


def _make_popen(exit_codes: dict[int, int] | None = None, output: str = ""):
    """Return a popen factory whose exit code depends on the child's --worker-id."""
    exit_codes = exit_codes or {}

    def factory(argv, **kwargs):
        worker_id = int(argv[argv.index("--worker-id") + 1])
        return _FakePopen(
            argv,
            exit_code=exit_codes.get(worker_id, 0),
            output=output,
            **kwargs,
        )

    return factory


def setup_function() -> None:
    _FakePopen.instances = []


# --- thread pinning ---------------------------------------------------------


def test_single_thread_env_sets_missing_vars() -> None:
    env = single_thread_env({})
    for name in SINGLE_THREAD_ENV_VARS:
        assert env[name] == "1"


def test_single_thread_env_preserves_existing_unless_override() -> None:
    base = {"OMP_NUM_THREADS": "8"}
    assert single_thread_env(base)["OMP_NUM_THREADS"] == "8"
    assert single_thread_env(base, override=True)["OMP_NUM_THREADS"] == "1"


def test_apply_single_thread_env_mutates_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SINGLE_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    apply_single_thread_env()
    import os

    for name in SINGLE_THREAD_ENV_VARS:
        assert os.environ[name] == "1"


# --- worker id layout -------------------------------------------------------


def test_worker_ids_are_distinct_and_non_overlapping_across_jobs() -> None:
    job0 = worker_ids_for_pool(0, 4)
    job1 = worker_ids_for_pool(1, 4)
    assert job0 == [0, 1, 2, 3]
    assert job1 == [4, 5, 6, 7]
    assert set(job0).isdisjoint(job1)


# --- pool supervision -------------------------------------------------------


def test_pool_launches_one_child_per_worker_with_distinct_ids() -> None:
    stream = io.StringIO()
    result = run_worker_pool(
        num_workers=3,
        base_worker_id=2,
        build_argv=lambda wid: ["python", "worker", "--worker-id", str(wid)],
        env={},
        stream=stream,
        popen=_make_popen(),
    )
    assert result.ok
    launched_ids = sorted(
        int(inst.argv[inst.argv.index("--worker-id") + 1]) for inst in _FakePopen.instances
    )
    assert launched_ids == [6, 7, 8]


def test_pool_continues_after_one_child_fails() -> None:
    stream = io.StringIO()
    result = run_worker_pool(
        num_workers=3,
        base_worker_id=0,
        build_argv=lambda wid: ["python", "worker", "--worker-id", str(wid)],
        env={},
        stream=stream,
        popen=_make_popen(exit_codes={1: 7}),
    )
    # All three still launched; the pool reports failure without stopping siblings.
    assert len(_FakePopen.instances) == 3
    assert not result.ok
    assert [item.worker_id for item in result.failed] == [1]


def test_pool_forwards_prefixed_child_output() -> None:
    stream = io.StringIO()
    run_worker_pool(
        num_workers=1,
        base_worker_id=5,
        build_argv=lambda wid: ["python", "worker", "--worker-id", str(wid)],
        env={},
        stream=stream,
        popen=_make_popen(output="event=completed\n"),
    )
    assert "[worker 5] event=completed" in stream.getvalue()


def test_pool_rejects_zero_workers() -> None:
    with pytest.raises(ValueError):
        run_worker_pool(
            num_workers=0,
            base_worker_id=0,
            build_argv=lambda wid: ["python"],
        )


def test_pool_result_ok_when_empty() -> None:
    assert PoolResult().ok


# --- dry-run planning -------------------------------------------------------


def _entry(tmp_path: Path, run_id: str, *, result_status: str | None) -> DynamicManifestEntry:
    import json

    output_path = tmp_path / f"{run_id}.json"
    row = {
        "output_path": str(output_path),
        "experiment_id": "dynamic_test",
        "algorithm_id": "alpha_miner",
        "backend": "pm4py",
        "log_id": run_id,
        "log_path": "data/example/tiny_log.xes",
        "test_log_path": "data/example/tiny_log.xes",
        "seed": "0",
        "config_hash": run_id,
        "run_id": run_id,
    }
    if result_status == "success":
        output_path.write_text(
            json.dumps(
                {
                    "experiment_id": "dynamic_test",
                    "log_id": run_id,
                    "log_path": row["log_path"],
                    "test_log_path": row["test_log_path"],
                    "seed": 0,
                    "algorithm_name": "alpha_miner",
                    "backend": "pm4py",
                    "hyperparameters": {},
                    "discovered_model_type": "petri_net",
                    "metrics": {},
                    "metric_statuses": {},
                    "status": "success",
                    "metadata": {"config_hash": run_id},
                }
            ),
            encoding="utf-8",
        )
    elif result_status is not None:
        output_path.write_text(json.dumps({"status": result_status}), encoding="utf-8")
    return DynamicManifestEntry(row_index=0, row=row, run_id=run_id)


def test_plan_dynamic_runs_categorizes_rows(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, "pending", result_status=None),
        _entry(tmp_path, "done", result_status="success"),
        _entry(tmp_path, "broken", result_status="failed"),
    ]
    plan = plan_dynamic_runs(entries, retry_failed=False)
    would_run = {entry.run_id for entry in plan.would_run}
    assert would_run == {"pending"}
    assert plan.skipped == {"done": "success", "broken": "failed"}


def test_plan_dynamic_runs_retries_failed(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path, "done", result_status="success"),
        _entry(tmp_path, "broken", result_status="failed"),
    ]
    plan = plan_dynamic_runs(entries, retry_failed=True)
    would_run = {entry.run_id for entry in plan.would_run}
    assert would_run == {"broken"}
    assert plan.skipped == {"done": "success"}
