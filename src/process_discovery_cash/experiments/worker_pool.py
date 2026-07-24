"""Run several single-worker dynamic-manifest processes inside one allocation.

The dynamic worker (``experiments/dynamic_worker.py``) already claims rows
through an atomic, multi-writer-safe ``ClaimManager`` so that many independent
processes can race for distinct rows without duplicating work or corrupting
result files. This module simply launches ``N`` such workers as child
processes of a single Slurm job, so one submitted job can execute ``N``
experiments concurrently.

The cluster limits the user to ~100 running/submitted jobs, so the way to get
more concurrency without more jobs is to give each job several CPUs and run one
worker per CPU. Each internal worker executes one experiment at a time and then
pulls the next pending row, exactly as an independent Slurm array task would.

Nothing here changes what an experiment does or how it is measured — it only
controls how many run at once inside a job.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

# Numeric libraries used by pm4py/numpy/polars default to one thread *per*
# logical core, which would oversubscribe the CPUs when several workers run in
# parallel. Pin every worker (and its row subprocess, which inherits the
# environment) to a single thread so that parallelism comes from running many
# independent experiments rather than from threading inside one of them.
SINGLE_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
)


def single_thread_env(
    base: Mapping[str, str] | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """Return a copy of ``base`` (or ``os.environ``) with thread vars pinned to 1.

    Values already present are preserved unless ``override`` is set, so an
    operator who deliberately exports ``OMP_NUM_THREADS=2`` keeps their choice.
    """
    env = dict(os.environ if base is None else base)
    for name in SINGLE_THREAD_ENV_VARS:
        if override or name not in env:
            env[name] = "1"
    return env


def apply_single_thread_env(*, override: bool = False) -> None:
    """Pin the thread vars in the current process's ``os.environ`` in place."""
    for name in SINGLE_THREAD_ENV_VARS:
        if override or name not in os.environ:
            os.environ[name] = "1"


def worker_ids_for_pool(base_worker_id: int, num_workers: int) -> list[int]:
    """Distinct scan-offset ids for the workers of one pool.

    Multiplying the base id by ``num_workers`` keeps the id ranges of separate
    Slurm jobs from overlapping when the base is the array task id, so the
    workers of different jobs also start their deterministic scans at different
    manifest rows and contend less.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be greater than 0")
    return [base_worker_id * num_workers + index for index in range(num_workers)]


@dataclass(frozen=True)
class WorkerProcessResult:
    worker_id: int
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass
class PoolResult:
    results: list[WorkerProcessResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.results)

    @property
    def failed(self) -> list[WorkerProcessResult]:
        return [result for result in self.results if not result.ok]


def _pump_output(
    stream_in,
    prefix: str,
    stream_out,
    lock: threading.Lock,
) -> None:
    """Forward a child's combined output to ``stream_out``, one prefixed line at a time."""
    try:
        for raw_line in iter(stream_in.readline, ""):
            line = raw_line.rstrip("\n")
            with lock:
                stream_out.write(f"{prefix}{line}\n")
                stream_out.flush()
    finally:
        try:
            stream_in.close()
        except Exception:
            pass


def run_worker_pool(
    *,
    num_workers: int,
    base_worker_id: int,
    build_argv: Callable[[int], Sequence[str]],
    env: Mapping[str, str] | None = None,
    stream=None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> PoolResult:
    """Launch ``num_workers`` single-worker children and wait for all of them.

    ``build_argv(worker_id)`` returns the full command line for one child (a
    ``--num-workers 1`` invocation of the worker CLI). Each child inherits
    ``env`` (defaults to a thread-pinned copy of the current environment) and
    streams its output back with a ``[worker N]`` prefix.

    The pool is fault-tolerant: a child that crashes or exits non-zero is
    recorded and reported, but its siblings keep running and the pool waits for
    every child before returning. Duplicate execution and result corruption are
    prevented by the shared ``ClaimManager`` the children already use, not by
    this supervisor.
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be greater than 0")
    if stream is None:
        stream = sys.stdout
    if env is None:
        env = single_thread_env()

    lock = threading.Lock()
    worker_ids = worker_ids_for_pool(base_worker_id, num_workers)
    processes: list[tuple[int, subprocess.Popen, threading.Thread | None]] = []

    for worker_id in worker_ids:
        argv = list(build_argv(worker_id))
        with lock:
            stream.write(f"[pool] launching worker_id={worker_id}: {' '.join(argv)}\n")
            stream.flush()
        try:
            process = popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=dict(env),
            )
        except Exception as exc:  # pragma: no cover - defensive: spawn failure
            with lock:
                stream.write(f"[pool] worker_id={worker_id} failed to start: {exc}\n")
                stream.flush()
            processes.append((worker_id, _FailedToStart(), None))
            continue

        pump: threading.Thread | None = None
        if process.stdout is not None:
            pump = threading.Thread(
                target=_pump_output,
                args=(process.stdout, f"[worker {worker_id}] ", stream, lock),
                daemon=True,
            )
            pump.start()
        processes.append((worker_id, process, pump))

    result = PoolResult()
    for worker_id, process, pump in processes:
        exit_code = process.wait()
        if pump is not None:
            pump.join()
        result.results.append(WorkerProcessResult(worker_id=worker_id, exit_code=exit_code))
        with lock:
            stream.write(f"[pool] worker_id={worker_id} exited with code {exit_code}\n")
            stream.flush()

    completed = len(result.results)
    failed = len(result.failed)
    with lock:
        stream.write(
            f"[pool] all workers finished: {completed - failed}/{completed} succeeded, "
            f"{failed} failed\n"
        )
        stream.flush()
    return result


class _FailedToStart:
    """Stand-in for a process that never launched, so the pool can still report it."""

    stdout = None

    def wait(self) -> int:
        return 1
