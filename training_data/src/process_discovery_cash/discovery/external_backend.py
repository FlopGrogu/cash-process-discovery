from __future__ import annotations

import gzip
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from process_discovery_cash.utils.paths import resolve_project_path


def prepare_xes_input(event_log: Any, input_log_path: str | None, work_dir: Path) -> Path:
    if input_log_path:
        source = resolve_project_path(input_log_path)
        if source.exists():
            if source.name.lower().endswith(".xes.gz"):
                target = work_dir / source.name[:-3]
                with gzip.open(source, "rb") as source_handle:
                    with target.open("wb") as target_handle:
                        shutil.copyfileobj(source_handle, target_handle)
                return target
            target = work_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            return target

    target = work_dir / "input_log.xes"
    try:
        import pm4py

        pm4py.write_xes(event_log, str(target))
        return target
    except Exception as exc:
        raise RuntimeError(
            "Could not prepare XES input for external backend. Provide input_log_path "
            "or install a pm4py version that can export the in-memory log."
        ) from exc


def run_command(
    command: list[str],
    timeout_seconds: int | float,
    cwd: Path,
) -> tuple[int, str, str, float, bool]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return (
            completed.returncode,
            completed.stdout,
            completed.stderr,
            time.perf_counter() - started,
            False,
        )
    except subprocess.TimeoutExpired as exc:
        return (
            -1,
            _subprocess_text(exc.stdout),
            _subprocess_text(exc.stderr),
            time.perf_counter() - started,
            True,
        )
    except FileNotFoundError as exc:
        return (
            127,
            "",
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - started,
            False,
        )


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
