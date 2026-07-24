from __future__ import annotations

import time
from typing import Any

from process_discovery_cash.discovery.base import DiscoveryAlgorithm, DiscoveryResult
from process_discovery_cash.discovery.pm4py_backend import (
    UnsupportedAlgorithmError,
    discover_alpha_miner,
)


class AlphaMiner(DiscoveryAlgorithm):
    algorithm_name = "alpha_miner"
    backend_name = "pm4py"
    default_model_type = "petri_net"

    def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
        started = time.perf_counter()
        try:
            payload = discover_alpha_miner(train_log, config)
            return self._result(
                config=config,
                status="success",
                runtime_seconds=time.perf_counter() - started,
                model_type=payload["model_type"],
                discovered_model=payload["model"],
                warnings=payload.get("metadata", {}).get("warnings", []),
                metadata=payload.get("metadata", {}),
            )
        except UnsupportedAlgorithmError as exc:
            return self._result(
                config=config,
                status="unsupported",
                runtime_seconds=time.perf_counter() - started,
                error_message=str(exc),
            )
        except Exception as exc:
            return self._result(
                config=config,
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                error_message=f"{type(exc).__name__}: {exc}",
            )
