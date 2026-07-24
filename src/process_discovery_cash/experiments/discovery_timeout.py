from __future__ import annotations

import multiprocessing
import pickle
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

from process_discovery_cash.discovery.base import DiscoveryAlgorithm, DiscoveryResult
from process_discovery_cash.utils.recursion import recursion_limit

DISCOVERY_TIMEOUT_FIELD = "discovery_timeout_seconds"
DEFAULT_DISCOVERY_TIMEOUT_SECONDS = 86400
_PROCESS_EXIT_GRACE_SECONDS = 1.0


def extract_discovery_timeout_seconds(config: dict[str, Any]) -> int | float | None:
    value = config.get(DISCOVERY_TIMEOUT_FIELD)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{DISCOVERY_TIMEOUT_FIELD} must be a positive number of seconds")
    try:
        timeout_seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{DISCOVERY_TIMEOUT_FIELD} must be a positive number of seconds, got {value!r}"
        ) from exc
    if timeout_seconds <= 0:
        raise ValueError(f"{DISCOVERY_TIMEOUT_FIELD} must be greater than zero")
    if timeout_seconds.is_integer():
        return int(timeout_seconds)
    return timeout_seconds


def discover_with_timeout(
    algorithm: DiscoveryAlgorithm,
    train_log: Any,
    config: dict[str, Any],
    timeout_seconds: int | float,
) -> DiscoveryResult:
    context = _multiprocessing_context()
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_discovery,
        args=(send_connection, algorithm, train_log, config),
    )
    started = time.perf_counter()
    process.start()
    send_connection.close()

    try:
        if receive_connection.poll(timeout_seconds):
            payload = receive_connection.recv_bytes()
            process.join(_PROCESS_EXIT_GRACE_SECONDS)
            if process.is_alive():
                _stop_process(process)
            with recursion_limit(config):
                result = pickle.loads(payload)
            if not isinstance(result, DiscoveryResult):
                raise TypeError(
                    f"Discovery worker returned an unexpected payload type: {type(result).__name__}"
                )
            result.metadata.setdefault("timeout_seconds", timeout_seconds)
            return result

        elapsed = time.perf_counter() - started
        _stop_process(process)
        return DiscoveryResult(
            algorithm_name=algorithm.algorithm_name,
            backend_name=algorithm.backend_name,
            hyperparameters=dict(config),
            runtime_seconds=elapsed,
            status="timeout",
            model_type=algorithm.default_model_type,
            error_message=(
                f"TimeoutError: discovery exceeded {_format_seconds(timeout_seconds)} seconds."
            ),
            metadata={"timeout_seconds": timeout_seconds},
        )
    finally:
        receive_connection.close()
        if process.is_alive():
            _stop_process(process)
        else:
            process.join()


def _run_discovery(
    connection: Connection,
    algorithm: DiscoveryAlgorithm,
    train_log: Any,
    config: dict[str, Any],
) -> None:
    started = time.perf_counter()
    try:
        with recursion_limit(config):
            result = algorithm.discover(train_log, config)
            payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException as exc:
        fallback = DiscoveryResult(
            algorithm_name=getattr(algorithm, "algorithm_name", "unknown"),
            backend_name=getattr(algorithm, "backend_name", "unknown"),
            hyperparameters=dict(config),
            runtime_seconds=time.perf_counter() - started,
            status="failed",
            model_type=getattr(algorithm, "default_model_type", "unknown"),
            error_message=f"{type(exc).__name__}: {exc}",
        )
        payload = pickle.dumps(fallback, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        connection.send_bytes(payload)
    finally:
        connection.close()


def _multiprocessing_context() -> Any:
    methods = multiprocessing.get_all_start_methods()
    can_fork_safely = (
        "fork" in methods
        and threading.current_thread() is threading.main_thread()
        and threading.active_count() == 1
    )
    if can_fork_safely:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _stop_process(process: Any) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(_PROCESS_EXIT_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join()


def _format_seconds(timeout_seconds: int | float) -> str:
    if isinstance(timeout_seconds, int):
        return str(timeout_seconds)
    return f"{timeout_seconds:g}"
