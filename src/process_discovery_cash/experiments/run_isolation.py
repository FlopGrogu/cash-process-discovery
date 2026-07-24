"""Run one manifest row in an isolated child process.

The dynamic workers execute every claimed row through this module so that an
OOM kill (or any other fatal signal) terminates only the child process. The
parent worker survives, observes the exit status, and can write a synthetic
failed result JSON instead of disappearing without a trace.

The module doubles as the child entry point:

    python -m process_discovery_cash.experiments.run_isolation \
        --row-json /path/to/row.json --kind discovery
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROW_KIND_DISCOVERY = "discovery"
ROW_KIND_METRIC = "metric"

_TERM_GRACE_SECONDS = 30.0
_DEFAULT_TICK_SECONDS = 15.0
_CHILD_OOM_SCORE_ADJ = 900


@dataclass(frozen=True)
class RowExecutionOutcome:
    """What happened to a row subprocess, as observed by the parent."""

    exit_code: int | None
    signal_name: str | None
    killed_by_parent: bool
    oom_suspected: bool
    child_peak_rss_bytes: int | None
    duration_seconds: float

    @property
    def abnormal(self) -> bool:
        return self.killed_by_parent or self.exit_code != 0


def run_row_in_subprocess(
    row: Mapping[str, str],
    *,
    kind: str,
    command_args: list[str] | None = None,
    deadline_monotonic: float | None = None,
    memory_limit_mb: float | None = None,
    on_tick: Callable[[], None] | None = None,
    tick_interval_seconds: float = _DEFAULT_TICK_SECONDS,
    scratch_dir: str | Path | None = None,
    child_argv_builder: Callable[[Path], list[str]] | None = None,
    force: bool = False,
    monotonic: Callable[[], float] = time.monotonic,
) -> RowExecutionOutcome:
    """Execute one manifest row in a child interpreter and report its fate.

    The child writes the result JSON itself (via ``run_manifest_row`` /
    ``run_metric_manifest_row``); the parent only observes the exit status.
    ``on_tick`` fires roughly every ``tick_interval_seconds`` while the child
    runs (used for claim heartbeats). When ``deadline_monotonic`` passes, the
    child gets SIGTERM, a grace period, then SIGKILL, and the outcome reports
    ``killed_by_parent=True``.
    """
    if kind not in {ROW_KIND_DISCOVERY, ROW_KIND_METRIC}:
        raise ValueError(f"Unknown row kind: {kind!r}")

    row_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="pdcash_row_",
        suffix=".json",
        dir=None if scratch_dir is None else str(scratch_dir),
        delete=False,
    )
    row_path = Path(row_file.name)
    try:
        with row_file:
            json.dump(dict(row), row_file)

        if child_argv_builder is not None:
            argv = child_argv_builder(row_path)
        else:
            argv = _default_child_argv(
                row_path,
                kind=kind,
                command_args=command_args,
                memory_limit_mb=memory_limit_mb,
                force=force,
            )

        rss_before = _children_peak_rss_bytes()
        oom_kills_before = _cgroup_oom_kill_count()
        started = monotonic()
        process = subprocess.Popen(argv)
        killed_by_parent = _wait_with_ticks(
            process,
            deadline_monotonic=deadline_monotonic,
            on_tick=on_tick,
            tick_interval_seconds=tick_interval_seconds,
            monotonic=monotonic,
        )
        duration_seconds = monotonic() - started
        returncode = process.returncode
        signal_name = _signal_name(returncode)
        oom_kills_after = _cgroup_oom_kill_count()
        cgroup_oom_kill = (
            oom_kills_before is not None
            and oom_kills_after is not None
            and oom_kills_after > oom_kills_before
        )
        oom_suspected = cgroup_oom_kill or (signal_name == "SIGKILL" and not killed_by_parent)
        return RowExecutionOutcome(
            exit_code=returncode,
            signal_name=signal_name,
            killed_by_parent=killed_by_parent,
            oom_suspected=oom_suspected,
            child_peak_rss_bytes=_children_peak_rss_delta(rss_before),
            duration_seconds=duration_seconds,
        )
    finally:
        row_path.unlink(missing_ok=True)


def _default_child_argv(
    row_path: Path,
    *,
    kind: str,
    command_args: list[str] | None,
    memory_limit_mb: float | None,
    force: bool,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "process_discovery_cash.experiments.run_isolation",
        "--row-json",
        str(row_path),
        "--kind",
        kind,
    ]
    if memory_limit_mb is not None and memory_limit_mb > 0:
        argv.extend(["--memory-limit-mb", str(memory_limit_mb)])
    if command_args:
        argv.extend(["--command-args-json", json.dumps(list(command_args))])
    if force:
        argv.append("--force")
    return argv


def _wait_with_ticks(
    process: subprocess.Popen[Any],
    *,
    deadline_monotonic: float | None,
    on_tick: Callable[[], None] | None,
    tick_interval_seconds: float,
    monotonic: Callable[[], float],
) -> bool:
    tick_interval_seconds = max(0.1, tick_interval_seconds)
    while True:
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            _terminate_child(process)
            return True
        wait_seconds = tick_interval_seconds
        if deadline_monotonic is not None:
            wait_seconds = min(wait_seconds, max(0.1, deadline_monotonic - monotonic()))
        try:
            process.wait(timeout=wait_seconds)
            return False
        except subprocess.TimeoutExpired:
            pass
        if on_tick is not None:
            on_tick()


def _terminate_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_TERM_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    process.wait()


def _signal_name(returncode: int | None) -> str | None:
    if returncode is None or returncode >= 0:
        return None
    number = -returncode
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal_{number}"


def _children_peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    try:
        max_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except (OSError, ValueError):
        return None
    if max_rss <= 0:
        return None
    if sys.platform == "darwin":
        return int(max_rss)
    return int(max_rss * 1024)


def _children_peak_rss_delta(rss_before: int | None) -> int | None:
    rss_after = _children_peak_rss_bytes()
    if rss_after is None:
        return None
    # ru_maxrss for RUSAGE_CHILDREN is a high-water mark across all waited
    # children; only a new maximum is attributable to the child we just ran.
    if rss_before is not None and rss_after <= rss_before:
        return None
    return rss_after


def _cgroup_oom_kill_count() -> int | None:
    try:
        cgroup_text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in cgroup_text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            events_path = Path("/sys/fs/cgroup") / parts[2].lstrip("/") / "memory.events"
            try:
                events_text = events_path.read_text(encoding="utf-8")
            except OSError:
                return None
            for events_line in events_text.splitlines():
                fields = events_line.split()
                if len(fields) == 2 and fields[0] == "oom_kill":
                    try:
                        return int(fields[1])
                    except ValueError:
                        return None
            return None
    return None


def apply_memory_limit_mb(limit_mb: float | None) -> int | None:
    """Cap the process address space via RLIMIT_AS; returns the applied bytes.

    Enforcement is only reliable on Linux (on macOS the kernel largely ignores
    RLIMIT_AS), but setting it is harmless everywhere.
    """
    if limit_mb is None or limit_mb <= 0:
        return None
    try:
        import resource
    except ImportError:
        return None
    limit_bytes = int(limit_mb * 1024 * 1024)
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY and hard < limit_bytes:
            limit_bytes = hard
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
    except (OSError, ValueError):
        return None
    return limit_bytes


def raise_oom_score_adj(value: int = _CHILD_OOM_SCORE_ADJ) -> bool:
    """Make the OOM killer prefer this process over its parent (Linux only)."""
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as handle:
            handle.write(str(value))
        return True
    except OSError:
        return False


def _child_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Internal child entry point: run one manifest row in isolation."
    )
    parser.add_argument("--row-json", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=[ROW_KIND_DISCOVERY, ROW_KIND_METRIC],
    )
    parser.add_argument("--memory-limit-mb", type=float, default=None)
    parser.add_argument("--command-args-json", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    # Apply resource limits before the heavy pm4py imports allocate anything.
    apply_memory_limit_mb(args.memory_limit_mb)
    raise_oom_score_adj()

    with open(args.row_json, encoding="utf-8") as handle:
        row = json.load(handle)
    command_args: list[str] | None = None
    if args.command_args_json:
        command_args = [str(item) for item in json.loads(args.command_args_json)]

    if args.kind == ROW_KIND_DISCOVERY:
        from process_discovery_cash.experiments.runner import run_manifest_row

        run_manifest_row(row, command_args=command_args, force=args.force)
    else:
        from process_discovery_cash.experiments.metric_manifest import (
            run_metric_manifest_row,
        )

        run_metric_manifest_row(row, command_args=command_args, force=args.force)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _child_main(argv)
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
