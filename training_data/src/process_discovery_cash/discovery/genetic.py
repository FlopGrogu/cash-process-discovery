from __future__ import annotations

import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from process_discovery_cash.data.loading import (
    ACTIVITY_COLUMN,
    CASE_ID_COLUMN,
    TIMESTAMP_COLUMN,
    ensure_pm4py_event_log,
)
from process_discovery_cash.discovery.base import DiscoveryAlgorithm, DiscoveryResult
from process_discovery_cash.discovery.pm4py_backend import (
    UnsupportedAlgorithmError,
    discover_genetic_miner,
)
from process_discovery_cash.experiments.discovery_timeout import (
    DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
)

NESTED_PARAM_FIELDS = ("default_params", "params", "hyperparameters")
DEFAULT_TIMEOUT_SECONDS = DEFAULT_DISCOVERY_TIMEOUT_SECONDS
DISCOVERY_TIMEOUT_FIELD = "discovery_timeout_seconds"
REMOVED_TIMEOUT_FIELD = "timeout_seconds"
RESERVED_CONFIG_FIELDS = {
    "algorithm",
    "algorithm_id",
    "algorithm_name",
    "backend",
    "display_name",
    "model_type",
    "pm4py_function",
    "pm4py_api",
    "search_space",
    "conditional_search_space",
    "parameter_mapping",
    "supported",
    "notes",
    "external",
    "input_log_path",
    "output_dir",
    DISCOVERY_TIMEOUT_FIELD,
    REMOVED_TIMEOUT_FIELD,
    "recursion_limit",
    *NESTED_PARAM_FIELDS,
}


class GeneticMinerTimeout(TimeoutError):
    """Raised when the Genetic Miner backend exceeds the configured runtime."""


class GeneticMiner(DiscoveryAlgorithm):
    algorithm_name = "genetic_miner"
    backend_name = "pm4py"
    default_model_type = "petri_net"

    def supports_algorithm(self) -> bool:
        try:
            from pm4py.algo.discovery.genetic import algorithm as genetic_miner

            return hasattr(genetic_miner, "apply")
        except Exception:
            return False

    def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
        started = time.perf_counter()
        result_config = _safe_result_config(config)
        try:
            params = extract_genetic_miner_params(config)
            timeout_seconds = extract_timeout_seconds(config)
        except TypeError as exc:
            return self._result(
                config=result_config,
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                error_message=f"Invalid Genetic Miner config: {exc}",
            )

        try:
            normalized_log = prepare_genetic_miner_log(train_log)
        except Exception as exc:
            return self._result(
                config=result_config,
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                error_message=(
                    "Invalid Genetic Miner input log: "
                    f"could not prepare {type(train_log).__name__} for pm4py Genetic Miner: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        try:
            with time_limit(timeout_seconds):
                payload = discover_genetic_miner(normalized_log, params)
            return self._result(
                config=params,
                status="success",
                runtime_seconds=time.perf_counter() - started,
                model_type=payload["model_type"],
                discovered_model=payload["model"],
                warnings=payload.get("metadata", {}).get("warnings", []),
                metadata=payload.get("metadata", {}),
            )
        except UnsupportedAlgorithmError as exc:
            return self._result(
                config=params,
                status="unsupported",
                runtime_seconds=time.perf_counter() - started,
                error_message=str(exc),
                warnings=[
                    "Genetic Miner is optional in pm4py and may be unavailable or "
                    "removed depending on the installed version."
                ],
            )
        except GeneticMinerTimeout as exc:
            return self._result(
                config=params,
                status="timeout",
                runtime_seconds=time.perf_counter() - started,
                error_message=str(exc),
                warnings=[
                    "Genetic Miner exceeded its configured timeout. Increase "
                    "discovery_timeout_seconds in the algorithm config for longer runs."
                ],
            )
        except Exception as exc:
            return self._result(
                config=params,
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                error_message=f"Genetic Miner backend failed: {type(exc).__name__}: {exc}",
            )


def extract_genetic_miner_params(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"expected config to be a dict, got {type(config).__name__}")

    params: dict[str, Any] = {}
    for field in NESTED_PARAM_FIELDS:
        value = config.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"expected '{field}' to be a dict, got {type(value).__name__}")
        params.update(value)

    for key, value in config.items():
        if key not in RESERVED_CONFIG_FIELDS:
            params[key] = value
    return params


def extract_timeout_seconds(config: Any) -> int:
    if not isinstance(config, dict):
        return DEFAULT_TIMEOUT_SECONDS
    removed_value = _extract_timeout_field(config, REMOVED_TIMEOUT_FIELD)
    if removed_value is not None:
        raise TypeError(
            f"unsupported timeout field '{REMOVED_TIMEOUT_FIELD}'; "
            f"use '{DISCOVERY_TIMEOUT_FIELD}'"
        )
    value = _extract_timeout_field(config, DISCOVERY_TIMEOUT_FIELD)
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout_seconds = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "expected "
            f"'{DISCOVERY_TIMEOUT_FIELD}'"
            " to be an integer number of seconds, got "
            f"{value!r}"
        ) from exc
    if timeout_seconds <= 0:
        raise TypeError(f"expected '{DISCOVERY_TIMEOUT_FIELD}' to be greater than zero")
    return timeout_seconds


def _extract_timeout_field(config: dict[str, Any], field_name: str) -> Any:
    value = config.get(field_name)
    if value is not None:
        return value
    for nested_field in NESTED_PARAM_FIELDS:
        nested = config.get(nested_field)
        if isinstance(nested, dict) and nested.get(field_name) is not None:
            return nested[field_name]
    return None


def _safe_result_config(config: Any) -> dict[str, Any]:
    return dict(config) if isinstance(config, dict) else {}


def prepare_genetic_miner_log(train_log: Any) -> Any:
    import pandas as pd
    from pm4py.objects.conversion.log import converter as log_converter

    if _looks_like_dataframe(train_log):
        dataframe = train_log.copy()
    else:
        event_log = ensure_pm4py_event_log(train_log)
        dataframe = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)

    dataframe = _normalize_genetic_dataframe_columns(dataframe)
    dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
        dataframe[TIMESTAMP_COLUMN],
        errors="raise",
        utc=True,
    )
    return dataframe


def _normalize_genetic_dataframe_columns(dataframe: Any) -> Any:
    rename_map = {}
    if CASE_ID_COLUMN not in dataframe.columns and "case_id" in dataframe.columns:
        rename_map["case_id"] = CASE_ID_COLUMN
    if ACTIVITY_COLUMN not in dataframe.columns and "activity" in dataframe.columns:
        rename_map["activity"] = ACTIVITY_COLUMN
    if TIMESTAMP_COLUMN not in dataframe.columns and "timestamp" in dataframe.columns:
        rename_map["timestamp"] = TIMESTAMP_COLUMN

    normalized = dataframe.rename(columns=rename_map).copy() if rename_map else dataframe.copy()
    required = [CASE_ID_COLUMN, ACTIVITY_COLUMN, TIMESTAMP_COLUMN]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise TypeError(
            "Genetic Miner DataFrame is missing required columns: "
            f"{', '.join(missing)}. Required columns are: {', '.join(required)}."
        )
    return normalized


def _looks_like_dataframe(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "copy")


@contextmanager
def time_limit(timeout_seconds: int) -> Iterator[None]:
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(_signum: int, _frame: Any) -> None:
        raise GeneticMinerTimeout(f"Genetic Miner timed out after {timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
