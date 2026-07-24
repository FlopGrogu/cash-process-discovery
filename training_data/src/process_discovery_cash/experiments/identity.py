from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# These fields control how long discovery may execute; they do not change the
# semantic experiment identity or the model/metrics requested by the row.
EXECUTION_CONTROL_FIELDS = frozenset(
    {
        "discovery_timeout_seconds",
        "metric_timeout_seconds",
        "recursion_limit",
        "timeout_seconds",
    }
)


def semantic_algorithm_parameters(
    params: Mapping[str, Any],
    runtime_fields: Iterable[str] = (),
) -> dict[str, Any]:
    excluded_fields = EXECUTION_CONTROL_FIELDS | frozenset(runtime_fields)
    return {key: value for key, value in params.items() if key not in excluded_fields}


def semantic_run_identity(
    *,
    log_id: str | None,
    log_path: str | None,
    test_log_path: str | None,
    seed: int | str,
    algorithm_id: str | None,
    backend: str | None,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    runtime_fields: Iterable[str] = (),
    dataset_id: str | None = None,
    artifact_sha256: str | None = None,
    preprocessing_fingerprint: str | None = None,
) -> dict[str, Any]:
    identity = {
        "log_id": log_id,
        "seed": int(seed),
        "algorithm": algorithm_id,
        "backend": backend,
        "params": semantic_algorithm_parameters(params, runtime_fields),
        "metrics": dict(metrics or {}),
    }
    if preprocessing_fingerprint:
        identity["dataset_id"] = dataset_id or log_id
        identity["preprocessing_fingerprint"] = preprocessing_fingerprint
    else:
        identity["log_path"] = log_path
        identity["test_log_path"] = test_log_path
    return identity


def execution_config(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: params[key]
        for key in sorted(EXECUTION_CONTROL_FIELDS)
        if key in params and params[key] not in (None, "")
    }
