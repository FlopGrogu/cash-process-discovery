from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any

from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.evaluation.quality_metrics import (
    DEFAULT_METRICS,
    evaluate_discovery_result,
)
from process_discovery_cash.experiments.discovery_timeout import (
    _PROCESS_EXIT_GRACE_SECONDS,
    _format_seconds,
    _multiprocessing_context,
    _stop_process,
)

METRIC_TIMEOUT_FIELD = "metric_timeout_seconds"
DEFAULT_METRIC_TIMEOUT_SECONDS = 1800


@dataclass
class TimedMetricEvaluation:
    """Result of a (possibly timed-out) metric evaluation.

    ``metrics``/``statuses`` mirror the return of ``evaluate_discovery_result``;
    on a timeout every metric is ``None`` with a ``"timeout"`` status. ``timings``
    carries the per-metric timings on success or a timeout marker otherwise.
    ``timed_out`` lets the caller record the top-level ``"timeout"`` status.
    """

    metrics: dict[str, float | None]
    statuses: dict[str, dict[str, Any]]
    timings: dict[str, Any] = field(default_factory=dict)
    timed_out: bool = False


def extract_metric_timeout_seconds(config: Mapping[str, Any]) -> int | float | None:
    value = config.get(METRIC_TIMEOUT_FIELD)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{METRIC_TIMEOUT_FIELD} must be a positive number of seconds")
    try:
        timeout_seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{METRIC_TIMEOUT_FIELD} must be a positive number of seconds, got {value!r}"
        ) from exc
    if timeout_seconds <= 0:
        raise ValueError(f"{METRIC_TIMEOUT_FIELD} must be greater than zero")
    if timeout_seconds.is_integer():
        return int(timeout_seconds)
    return timeout_seconds


def evaluate_with_timeout(
    discovery_result: DiscoveryResult,
    test_log: Any,
    metric_names: list[str] | None,
    *,
    metric_profile: str,
    timeout_seconds: int | float,
) -> TimedMetricEvaluation:
    resolved_metric_names = [
        name for name in (metric_names or DEFAULT_METRICS) if name != "composite_score"
    ]
    context = _multiprocessing_context()
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_evaluation,
        args=(
            send_connection,
            discovery_result,
            test_log,
            resolved_metric_names,
            metric_profile,
        ),
    )
    process.start()
    send_connection.close()

    try:
        if receive_connection.poll(timeout_seconds):
            payload = receive_connection.recv_bytes()
            process.join(_PROCESS_EXIT_GRACE_SECONDS)
            if process.is_alive():
                _stop_process(process)
            result = pickle.loads(payload)
            if not isinstance(result, tuple) or len(result) != 3:
                raise TypeError(
                    f"Metric worker returned an unexpected payload type: {type(result).__name__}"
                )
            metrics, statuses, timings = result
            return TimedMetricEvaluation(
                metrics=metrics,
                statuses=statuses,
                timings=timings,
                timed_out=False,
            )

        _stop_process(process)
        return _timeout_evaluation(resolved_metric_names, timeout_seconds)
    finally:
        receive_connection.close()
        if process.is_alive():
            _stop_process(process)
        else:
            process.join()


def _timeout_evaluation(
    metric_names: list[str],
    timeout_seconds: int | float,
) -> TimedMetricEvaluation:
    error_message = (
        f"TimeoutError: metric evaluation exceeded {_format_seconds(timeout_seconds)} seconds."
    )
    metrics: dict[str, float | None] = {name: None for name in metric_names}
    statuses: dict[str, dict[str, Any]] = {
        name: {"status": "timeout", "error": error_message, "value": None} for name in metric_names
    }
    timings: dict[str, Any] = {"timed_out": True, "timeout_seconds": timeout_seconds}
    return TimedMetricEvaluation(
        metrics=metrics,
        statuses=statuses,
        timings=timings,
        timed_out=True,
    )


def _run_evaluation(
    connection: Connection,
    discovery_result: DiscoveryResult,
    test_log: Any,
    metric_names: list[str],
    metric_profile: str,
) -> None:
    try:
        metrics, statuses, timings = evaluate_discovery_result(
            discovery_result,
            test_log,
            metric_names=metric_names,
            metric_profile=metric_profile,
            include_timings=True,
        )
        payload = pickle.dumps((metrics, statuses, timings), protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        metrics = {name: None for name in metric_names}
        statuses = {
            name: {"status": "backend_error", "error": error_message, "value": None}
            for name in metric_names
        }
        timings = {"error": error_message}
        payload = pickle.dumps((metrics, statuses, timings), protocol=pickle.HIGHEST_PROTOCOL)
    try:
        connection.send_bytes(payload)
    finally:
        connection.close()
