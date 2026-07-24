from __future__ import annotations

import sys
import time
import traceback
from importlib import metadata as importlib_metadata
from typing import Any

from process_discovery_cash.discovery.base import DiscoveryAlgorithm, DiscoveryResult
from process_discovery_cash.discovery.pm4py_backend import (
    UnsupportedAlgorithmError,
    discover_inductive_miner,
)
from process_discovery_cash.utils.recursion import configured_recursion_limit


class InductiveMiner(DiscoveryAlgorithm):
    algorithm_name = "inductive_miner"
    backend_name = "pm4py"
    default_model_type = "petri_net"

    def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
        started = time.perf_counter()
        previous_recursion_limit = sys.getrecursionlimit()
        recursion_limit = previous_recursion_limit
        try:
            recursion_limit = _configured_recursion_limit(config)
            sys.setrecursionlimit(recursion_limit)
            payload = discover_inductive_miner(train_log, config)
            metadata = dict(payload.get("metadata", {}))
            metadata.update(
                {
                    "pm4py_version": _pm4py_version(),
                    "previous_recursion_limit": previous_recursion_limit,
                    "recursion_limit_used": recursion_limit,
                }
            )
            return self._result(
                config=config,
                status="success",
                runtime_seconds=time.perf_counter() - started,
                model_type=payload["model_type"],
                discovered_model=payload["model"],
                warnings=metadata.get("warnings", []),
                metadata=metadata,
            )
        except UnsupportedAlgorithmError as exc:
            return self._result(
                config=config,
                status="unsupported",
                runtime_seconds=time.perf_counter() - started,
                error_message=str(exc),
            )
        except RecursionError as exc:
            return self._result(
                config=config,
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                error_message=f"{type(exc).__name__}: {exc}",
                metadata=_failure_metadata(
                    config=config,
                    recursion_limit=recursion_limit,
                    previous_recursion_limit=previous_recursion_limit,
                    error=exc,
                ),
            )
        except Exception as exc:
            return self._result(
                config=config,
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                error_message=f"{type(exc).__name__}: {exc}",
                metadata=_failure_metadata(
                    config=config,
                    recursion_limit=recursion_limit,
                    previous_recursion_limit=previous_recursion_limit,
                    error=exc,
                ),
            )
        finally:
            sys.setrecursionlimit(previous_recursion_limit)


def _configured_recursion_limit(config: dict[str, Any]) -> int:
    return configured_recursion_limit(config)


def _pm4py_version() -> str | None:
    try:
        return importlib_metadata.version("pm4py")
    except importlib_metadata.PackageNotFoundError:
        return None


def _failure_metadata(
    *,
    config: dict[str, Any],
    recursion_limit: int,
    previous_recursion_limit: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "algorithm_name": InductiveMiner.algorithm_name,
        "backend": InductiveMiner.backend_name,
        "error_message": str(error),
        "error_type": type(error).__name__,
        "hyperparameters": dict(config),
        "input_log_path": config.get("input_log_path"),
        "log_id": config.get("log_id"),
        "pm4py_version": _pm4py_version(),
        "previous_recursion_limit": previous_recursion_limit,
        "recursion_limit_used": recursion_limit,
        "traceback": traceback.format_exc(),
    }
