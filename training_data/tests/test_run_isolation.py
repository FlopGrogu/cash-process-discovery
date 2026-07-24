from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from process_discovery_cash.experiments.run_isolation import (
    ROW_KIND_DISCOVERY,
    run_row_in_subprocess,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _python_child(code: str):
    return lambda _row_path: [sys.executable, "-c", code]


def test_normal_child_exit_is_not_abnormal() -> None:
    outcome = run_row_in_subprocess(
        {"x": "y"},
        kind=ROW_KIND_DISCOVERY,
        child_argv_builder=_python_child("pass"),
    )

    assert outcome.exit_code == 0
    assert outcome.signal_name is None
    assert outcome.killed_by_parent is False
    assert outcome.abnormal is False


def test_sigkilled_child_is_reported_as_suspected_oom() -> None:
    outcome = run_row_in_subprocess(
        {"x": "y"},
        kind=ROW_KIND_DISCOVERY,
        child_argv_builder=_python_child("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
    )

    assert outcome.exit_code == -9
    assert outcome.signal_name == "SIGKILL"
    assert outcome.killed_by_parent is False
    assert outcome.oom_suspected is True
    assert outcome.abnormal is True


def test_nonzero_exit_is_abnormal_but_not_oom() -> None:
    outcome = run_row_in_subprocess(
        {"x": "y"},
        kind=ROW_KIND_DISCOVERY,
        child_argv_builder=_python_child("import sys; sys.exit(3)"),
    )

    assert outcome.exit_code == 3
    assert outcome.signal_name is None
    assert outcome.oom_suspected is False
    assert outcome.abnormal is True


def test_deadline_kill_reports_killed_by_parent() -> None:
    outcome = run_row_in_subprocess(
        {"x": "y"},
        kind=ROW_KIND_DISCOVERY,
        child_argv_builder=_python_child("import time; time.sleep(60)"),
        deadline_monotonic=time.monotonic() + 0.5,
        tick_interval_seconds=0.1,
    )

    assert outcome.killed_by_parent is True
    assert outcome.abnormal is True
    assert outcome.duration_seconds < 40


def test_on_tick_fires_while_child_runs() -> None:
    ticks: list[int] = []

    outcome = run_row_in_subprocess(
        {"x": "y"},
        kind=ROW_KIND_DISCOVERY,
        child_argv_builder=_python_child("import time; time.sleep(1)"),
        tick_interval_seconds=0.1,
        on_tick=lambda: ticks.append(1),
    )

    assert outcome.exit_code == 0
    assert ticks


def test_row_json_is_passed_to_child_and_cleaned_up(tmp_path: Path) -> None:
    captured: dict[str, Path] = {}

    def builder(row_path: Path) -> list[str]:
        captured["path"] = row_path
        code = (
            f"import json; payload = json.load(open({str(row_path)!r}));assert payload['x'] == 'y'"
        )
        return [sys.executable, "-c", code]

    outcome = run_row_in_subprocess(
        {"x": "y"},
        kind=ROW_KIND_DISCOVERY,
        child_argv_builder=builder,
        scratch_dir=tmp_path,
    )

    assert outcome.exit_code == 0
    assert captured["path"].parent == tmp_path
    assert not captured["path"].exists()


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_row_in_subprocess({}, kind="nope")


def test_apply_memory_limit_sets_rlimit_as() -> None:
    code = (
        "import resource;"
        "from process_discovery_cash.experiments.run_isolation import apply_memory_limit_mb;"
        "applied = apply_memory_limit_mb(512);"
        "soft, _hard = resource.getrlimit(resource.RLIMIT_AS);"
        "print(applied, soft)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SRC_PATH}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    applied, soft = result.stdout.split()
    if sys.platform == "linux":
        assert applied == soft == str(512 * 1024 * 1024)
    else:
        # Platforms that refuse RLIMIT_AS changes (e.g. macOS) must degrade
        # to a no-op instead of crashing the child at startup.
        assert applied in {"None", str(512 * 1024 * 1024)}


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only enforced on Linux")
def test_memory_limit_turns_runaway_allocation_into_memory_error() -> None:
    code = (
        "from process_discovery_cash.experiments.run_isolation import apply_memory_limit_mb;"
        "apply_memory_limit_mb(256);"
        "blob = bytearray(1024 * 1024 * 1024)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SRC_PATH}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "MemoryError" in result.stderr


def test_child_entry_writes_failed_result_for_broken_discovery_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{SRC_PATH}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )
    output_path = tmp_path / "result.json"
    params_json = json.dumps({"variant": "classic"}, sort_keys=True)
    row = {
        "experiment_id": "iso_test",
        "log_id": "log_x",
        "log_path": (tmp_path / "does_not_exist.xes").as_posix(),
        "test_log_path": (tmp_path / "does_not_exist.xes").as_posix(),
        "seed": "0",
        "algorithm_id": "alpha_miner",
        "algorithm": "alpha_miner",
        "backend": "pm4py",
        "algorithm_params_json": params_json,
        "params_json": params_json,
        "metrics_json": json.dumps({"enabled": False}),
        "config_id": "cfg_x",
        "config_hash": "cfg_x",
        "output_path": output_path.as_posix(),
    }

    outcome = run_row_in_subprocess(row, kind=ROW_KIND_DISCOVERY, scratch_dir=tmp_path)

    assert outcome.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["metadata"]["config_hash"] == "cfg_x"
