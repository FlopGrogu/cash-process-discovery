from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def aggregate_results(
    results_root: str | Path,
    output_path: str | Path,
) -> Path:
    rows = [_flatten_result(path) for path in sorted(Path(results_root).rglob("*.json"))]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row}) if rows else ["result_path"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _flatten_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    metadata = payload.get("metadata") or {}
    source_metadata = payload.get("source_metadata") or {}
    is_metric_output = "metric_profile" in payload and "source_result_path" in payload
    runtime_seconds = payload.get("runtime_seconds")
    status = payload.get("status")
    error_message = payload.get("error_message")
    metric_runtime_seconds = payload.get("metric_runtime_seconds")
    if is_metric_output:
        runtime_seconds = payload.get("discovery_runtime_seconds")
        status = payload.get("discovery_status") or payload.get("status")
        metric_runtime_seconds = payload.get("metric_runtime_seconds")
    row: dict[str, Any] = {
        "result_path": path.as_posix(),
        "experiment_id": payload.get("experiment_id"),
        "log_id": payload.get("log_id"),
        "log_path": payload.get("log_path"),
        "test_log_path": payload.get("test_log_path"),
        "seed": payload.get("seed"),
        "algorithm_name": payload.get("algorithm_name"),
        "backend": payload.get("backend"),
        "status": status,
        "runtime_seconds": runtime_seconds,
        "discovered_model_type": payload.get("discovered_model_type") or payload.get("model_type"),
        "model_path": payload.get("model_path"),
        "error_message": error_message,
        "metric_profile": payload.get("metric_profile"),
        "source_result_path": payload.get("source_result_path"),
        "source_config_hash": payload.get("source_config_hash"),
        "config_hash": payload.get("config_hash") or metadata.get("config_hash"),
        "metric_run_status": payload.get("status") if is_metric_output else None,
        "metric_runtime_seconds": metric_runtime_seconds,
        "discovery_status": payload.get("discovery_status"),
        "discovery_runtime_seconds": payload.get("discovery_runtime_seconds"),
        "timestamp_utc": metadata.get("timestamp_utc"),
    }
    for key, value in (payload.get("hyperparameters") or {}).items():
        row[f"param_{key}"] = value
    for key, value in (payload.get("metrics") or {}).items():
        row[f"metric_{key}"] = value
    for key, value in (payload.get("metric_statuses") or {}).items():
        if isinstance(value, dict):
            row[f"metric_status_{key}"] = value.get("status")
            row[f"metric_error_{key}"] = value.get("error")
    _flatten_metadata(metadata.get("timings"), row, "timing")
    _flatten_metadata(metadata.get("memory"), row, "memory")
    _flatten_metadata(metadata.get("slurm"), row, "slurm")
    _flatten_metadata(source_metadata, row, "source")
    return row


def _flatten_metadata(value: Any, row: dict[str, Any], prefix: str) -> None:
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        column = f"{prefix}_{key}"
        if isinstance(child, dict):
            _flatten_metadata(child, row, column)
        elif isinstance(child, int | float | str | bool) or child is None:
            row[column] = child
