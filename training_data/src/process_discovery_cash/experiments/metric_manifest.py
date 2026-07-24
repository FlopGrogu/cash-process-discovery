from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from process_discovery_cash.data.loading import load_event_log
from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.evaluation.quality_metrics import (
    DEFAULT_METRICS,
    evaluate_discovery_result,
)
from process_discovery_cash.experiments.metric_timeout import (
    DEFAULT_METRIC_TIMEOUT_SECONDS,
    evaluate_with_timeout,
    extract_metric_timeout_seconds,
)
from process_discovery_cash.experiments.resource_tracking import (
    attach_resource_metadata,
    current_peak_rss_bytes,
)
from process_discovery_cash.experiments.result_schema import MetricRecord, MetricResult
from process_discovery_cash.experiments.saved_model_metrics import (
    _load_json,
    _load_model_artifact_with_fallback,
    _resolve_metric_model_path,
    _resolve_model_path,
)
from process_discovery_cash.experiments.tracking import collect_reproducibility_metadata
from process_discovery_cash.utils.paths import (
    ensure_parent,
    portable_project_path,
    project_root,
    resolve_portable_path,
)

METRIC_MANIFEST_COLUMNS = [
    "source_result_path",
    "source_config_hash",
    "experiment_id",
    "log_id",
    "algorithm_name",
    "test_log_path",
    "log_cache_key",
    "metric_profile",
    "metric_names_json",
    "metric_timeout_seconds",
    "log_dir",
    "output_path",
]


@dataclass(frozen=True)
class MetricManifestStats:
    """Metric-manifest row counts retained for API compatibility.

    Generation does not inspect discovery outputs. Every source row is marked
    schedulable (``full``); the worker resolves discovery/model availability at
    run time.
    """

    full: int = 0
    fallback_missing_result: int = 0
    fallback_malformed_json: int = 0
    fallback_failed_discovery: int = 0
    fallback_missing_model: int = 0

    @property
    def total(self) -> int:
        return (
            self.full
            + self.fallback_missing_result
            + self.fallback_malformed_json
            + self.fallback_failed_discovery
            + self.fallback_missing_model
        )

    def increment(self, field_name: str) -> MetricManifestStats:
        values = self.as_dict()
        values[field_name] += 1
        return MetricManifestStats(**values)

    def as_dict(self) -> dict[str, int]:
        return {
            "full": self.full,
            "fallback_missing_result": self.fallback_missing_result,
            "fallback_malformed_json": self.fallback_malformed_json,
            "fallback_failed_discovery": self.fallback_failed_discovery,
            "fallback_missing_model": self.fallback_missing_model,
        }


@dataclass(frozen=True)
class MetricManifestGenerationResult:
    rows: list[dict[str, str]]
    stats: MetricManifestStats


def generate_metric_manifest_rows_from_source_manifest(
    source_manifest_path: str | Path,
    *,
    metric_profile: str,
    metric_names: list[str] | None = None,
    output_root: str | Path,
    metric_timeout_seconds: int | float | None = DEFAULT_METRIC_TIMEOUT_SECONDS,
) -> MetricManifestGenerationResult:
    source_manifest_path = Path(source_manifest_path)
    output_root = Path(output_root)
    root = project_root()
    metric_names_json = json.dumps(_metric_names(metric_names), sort_keys=True)
    metric_timeout_str = _metric_timeout_str(metric_timeout_seconds)

    rows: list[dict[str, str]] = []
    stats = MetricManifestStats()
    with source_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [
            column
            for column in ("output_path", "log_dir", "config_hash")
            if column not in fieldnames
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"source manifest must contain column(s): {joined}")
        source_rows = list(reader)

    log_dir = metric_log_dir_from_source_manifest_rows(
        source_rows,
        metric_profile=metric_profile,
    )
    source_results_root = metric_output_source_root_from_source_manifest_rows(source_rows)

    for source_row in source_rows:
        source_result_path = Path(source_row.get("output_path") or "")
        if not source_result_path.as_posix():
            raise ValueError(
                "source manifest row is missing an output_path; every discovery "
                "row must define output_path so it can be preserved in the metric "
                "manifest"
            )
        output_path = metric_output_path_from_source_result_path(
            source_result_path,
            output_root=output_root,
            source_results_root=source_results_root,
        )
        row = _metric_manifest_row_from_source_manifest_fallback(
            source_result_path,
            source_row=source_row,
            metric_profile=metric_profile,
            metric_names_json=metric_names_json,
            metric_timeout_str=metric_timeout_str,
            log_dir=log_dir,
            output_path=output_path,
            project_root=root,
        )
        rows.append(row)
        stats = stats.increment("full")

    if len(rows) != len(source_rows):  # pragma: no cover - defensive invariant guard
        raise RuntimeError(
            f"metric manifest row count {len(rows)} does not match source manifest "
            f"row count {len(source_rows)}; every source row must be preserved"
        )
    _validate_metric_rows_preserve_source_manifest_identity(
        rows,
        source_rows,
        project_root=root,
    )
    return MetricManifestGenerationResult(rows=rows, stats=stats)


def _metric_manifest_row_from_source_manifest_fallback(
    source_result_path: Path,
    *,
    source_row: dict[str, str],
    metric_profile: str,
    metric_names_json: str,
    metric_timeout_str: str,
    log_dir: str,
    output_path: Path,
    project_root: Path,
) -> dict[str, str]:
    algorithm_name = str(
        source_row.get("algorithm_name")
        or source_row.get("algorithm")
        or source_row.get("algorithm_id")
        or ""
    )
    source_log_path = str(source_row.get("log_path") or "")
    metric_log_path = str(
        source_row.get("test_log_path")
        or source_log_path
        or ""
    )
    log_id = str(source_row.get("log_id") or "")
    return {
        "source_result_path": _portable_path(source_result_path, project_root),
        "source_config_hash": _source_manifest_config_hash(source_row),
        "experiment_id": str(source_row.get("experiment_id") or ""),
        "log_id": log_id,
        "algorithm_name": algorithm_name,
        "test_log_path": metric_log_path,
        "log_cache_key": _metric_log_cache_key(log_id, source_log_path, metric_log_path),
        "metric_profile": metric_profile,
        "metric_names_json": metric_names_json,
        "metric_timeout_seconds": metric_timeout_str,
        "log_dir": log_dir,
        "output_path": _portable_path(output_path, project_root),
    }


def _metric_timeout_str(metric_timeout_seconds: int | float | None) -> str:
    """Render the metric timeout for a manifest cell (empty string means no timeout)."""
    if metric_timeout_seconds is None or metric_timeout_seconds == "":
        return ""
    validated = extract_metric_timeout_seconds({"metric_timeout_seconds": metric_timeout_seconds})
    return "" if validated is None else str(validated)


def _source_manifest_config_hash(source_row: dict[str, str]) -> str:
    config_hash = str(source_row.get("config_hash") or "").strip()
    if not config_hash:
        raise ValueError(
            "source manifest rows must define a non-empty config_hash; "
            "metric manifest identity is copied from the source manifest"
        )
    return config_hash


def _validate_metric_rows_preserve_source_manifest_identity(
    rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    *,
    project_root: Path,
) -> None:
    for row_index, (metric_row, source_row) in enumerate(zip(rows, source_rows, strict=True)):
        expected_source_path = _portable_path(
            Path(source_row.get("output_path") or ""),
            project_root,
        )
        expected_config_hash = _source_manifest_config_hash(source_row)
        if metric_row.get("source_result_path") != expected_source_path:
            raise RuntimeError(
                f"metric manifest row {row_index} changed source_result_path: "
                f"expected {expected_source_path!r}, got "
                f"{metric_row.get('source_result_path')!r}"
            )
        if metric_row.get("source_config_hash") != expected_config_hash:
            raise RuntimeError(
                f"metric manifest row {row_index} changed source_config_hash: "
                f"expected {expected_config_hash!r}, got "
                f"{metric_row.get('source_config_hash')!r}"
            )


def metric_log_dir_from_source_manifest_rows(
    rows: list[dict[str, str]],
    *,
    metric_profile: str,
) -> str:
    log_dirs = {
        (row.get("log_dir") or "").strip()
        for row in rows
        if any(value.strip() for value in row.values())
    }
    if not log_dirs:
        raise ValueError("source manifest has no log_dir values")
    if "" in log_dirs:
        raise ValueError("source manifest rows must define a non-empty log_dir")
    if len(log_dirs) != 1:
        configured = ", ".join(repr(value) for value in sorted(log_dirs))
        raise ValueError(f"source manifest rows define inconsistent log_dir values: {configured}")
    return (Path(next(iter(log_dirs))) / "metrics" / metric_profile).as_posix()


def metric_output_source_root_from_source_manifest_rows(rows: list[dict[str, str]]) -> Path:
    parent_dirs = sorted(
        {
            Path(output_path).parent
            for row in rows
            if (output_path := (row.get("output_path") or "").strip())
        },
        key=lambda path: path.as_posix(),
    )
    if not parent_dirs:
        raise ValueError("source manifest has no output_path values")
    root = Path(os.path.commonpath([parent_dir.as_posix() for parent_dir in parent_dirs]))
    if not root.as_posix():
        raise ValueError("unable to infer common source result root from source manifest")
    return root


def metric_output_path_from_source_result_path(
    source_result_path: str | Path,
    *,
    output_root: str | Path,
    source_results_root: str | Path,
) -> Path:
    source_result_path = Path(source_result_path)
    output_root = Path(output_root)
    source_results_root = Path(source_results_root)
    try:
        relative_result_path = source_result_path.relative_to(source_results_root)
    except ValueError as exc:
        raise ValueError(
            f"source result path {source_result_path.as_posix()} is not under "
            f"{source_results_root.as_posix()}"
        ) from exc
    return output_root / relative_result_path


def write_metric_manifest(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=METRIC_MANIFEST_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def generate_metric_manifest_from_source_manifest(
    source_manifest_path: str | Path,
    *,
    metric_profile: str,
    output_path: str | Path,
    output_root: str | Path,
    metric_names: list[str] | None = None,
    metric_timeout_seconds: int | float | None = DEFAULT_METRIC_TIMEOUT_SECONDS,
) -> tuple[Path, MetricManifestStats]:
    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile=metric_profile,
        metric_names=metric_names,
        output_root=output_root,
        metric_timeout_seconds=metric_timeout_seconds,
    )
    return write_metric_manifest(result.rows, output_path), result.stats


def load_metric_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_metric_manifest_index(
    manifest_path: str | Path,
    row_index: int,
    *,
    command_args: list[str] | None = None,
    force: bool = False,
) -> Path:
    rows = load_metric_manifest_rows(manifest_path)
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"Metric manifest row index {row_index} outside 0..{len(rows) - 1}")
    return run_metric_manifest_row(rows[row_index], command_args=command_args, force=force)


def run_metric_slurm_array_task(
    manifest_path: str | Path,
    *,
    command_args: list[str] | None = None,
    force: bool = False,
) -> Path:
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not set")
    row_offset = int(os.getenv("METRIC_MANIFEST_ROW_OFFSET", os.getenv("MANIFEST_ROW_OFFSET", "0")))
    return run_metric_manifest_index(
        manifest_path,
        row_offset + int(task_id),
        command_args=command_args,
        force=force,
    )


def run_metric_manifest_row(
    row: dict[str, str],
    *,
    command_args: list[str] | None = None,
    force: bool = False,
) -> Path:
    started = time.perf_counter()
    memory_started = current_peak_rss_bytes()
    output_path = ensure_parent(row["output_path"])
    if (
        not force
        and _metric_output_is_reusable_success(output_path, row)
        and not _metric_output_is_stale_success_missing(output_path, row)
    ):
        print(f"Skipped existing successful metric result {output_path}")
        return output_path

    metric_names = _metric_names_from_row(row)
    metric_profile = row["metric_profile"]
    metric_timeout_seconds = extract_metric_timeout_seconds(row)
    timings: dict[str, Any] = {}
    source_payload, source_metadata = _source_payload_from_row(row)
    source_result_path = Path(row.get("source_result_path") or "")
    metadata = collect_reproducibility_metadata(
        config_hash=row.get("source_config_hash") or "",
        seed=0,
        command_args=command_args or sys.argv,
    )
    metadata["command_args"] = command_args or sys.argv

    try:
        model_path, metric_model_path, fallback_reason = _resolve_metric_model_inputs(
            source_payload,
            source_result_path,
        )
        if fallback_reason is not None:
            model_type = _model_type_from_payload(source_payload)
            metrics, metric_statuses = _missing_model_metric_results(metric_names, fallback_reason)
            timings["model_load_backend"] = "fallback_zero"
            timings["metrics"] = {
                "profile": metric_profile,
                "fallback_zero": True,
                "missing_model": True,
            }
            status = "success_missing"
            error_message = None
        else:
            (
                load_backend,
                model_type,
                discovered_model,
                model_timings,
            ) = _load_model_artifact_with_fallback(
                model_path,
                metric_model_path=metric_model_path,
            )
            timings.update(model_timings)
            timings["model_load_backend"] = load_backend
            with _record_timing(timings, "load_test_log_seconds"):
                test_log = load_event_log(
                    row["test_log_path"],
                    cache_key=row.get("log_cache_key") or row.get("log_id"),
                )
            discovery_result = DiscoveryResult(
                algorithm_name=row.get("algorithm_name") or "unknown",
                backend_name="saved_model",
                hyperparameters={},
                runtime_seconds=None,
                status="success",
                model_type=model_type,
                model_path=model_path.as_posix() if model_path is not None else None,
                discovered_model=discovered_model,
            )
            with _record_timing(timings, "metrics_eval_call_seconds"):
                if metric_timeout_seconds is None:
                    metrics, metric_statuses, metric_timings = evaluate_discovery_result(
                        discovery_result,
                        test_log,
                        metric_names=metric_names,
                        metric_profile=metric_profile,
                        include_timings=True,
                    )
                    timed_out = False
                else:
                    evaluation = evaluate_with_timeout(
                        discovery_result,
                        test_log,
                        metric_names,
                        metric_profile=metric_profile,
                        timeout_seconds=metric_timeout_seconds,
                    )
                    metrics = evaluation.metrics
                    metric_statuses = evaluation.statuses
                    metric_timings = evaluation.timings
                    timed_out = evaluation.timed_out
            timings["metrics"] = metric_timings
            if timed_out:
                status = "timeout"
                error_message = (
                    f"TimeoutError: metric evaluation exceeded {metric_timeout_seconds} seconds."
                )
            else:
                status = "success"
                error_message = None
    except Exception as exc:
        model_type = None
        metrics, metric_statuses = _not_computed_metric_results(
            f"Metric evaluation failed: {type(exc).__name__}: {exc}",
            metric_names,
        )
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"

    runtime_seconds = time.perf_counter() - started
    timings["total_seconds_before_write"] = runtime_seconds
    metadata["timings"] = timings
    attach_resource_metadata(metadata, memory_started)
    output_record = _metric_output_record(
        row,
        source_payload=source_payload,
        metric_names=metric_names,
        metrics=metrics,
        metric_statuses=metric_statuses,
        model_type=model_type,
        status=status,
        error_message=error_message,
        runtime_seconds=runtime_seconds,
        metadata=metadata,
        source_metadata=source_metadata,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _metric_log_cache_key(log_id: str, source_log_path: str, test_log_path: str) -> str:
    if source_log_path and source_log_path != test_log_path:
        return f"{log_id}_test"
    return log_id


@contextmanager
def _record_timing(timings: dict[str, Any], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - started


def _metric_names(metric_names: list[str] | None) -> list[str]:
    return [name for name in (metric_names or DEFAULT_METRICS) if name != "composite_score"]


def _metric_names_from_row(row: dict[str, str]) -> list[str]:
    raw = row.get("metric_names_json") or json.dumps(DEFAULT_METRICS)
    parsed = json.loads(raw)
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("metric_names_json must contain a JSON list or string")
    return _metric_names([str(name) for name in parsed])


def _not_computed_metric_results(
    reason: str,
    metric_names: list[str],
) -> tuple[dict[str, None], dict[str, dict[str, Any]]]:
    metrics = {name: None for name in metric_names}
    statuses = {
        name: {"status": "not_computed", "error": reason, "value": None} for name in metric_names
    }
    return metrics, statuses


def _zero_metric_results(
    metric_names: list[str],
    reason: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics = {name: 0.0 for name in metric_names}
    statuses = {name: {"status": "success", "error": reason, "value": 0.0} for name in metric_names}
    return metrics, statuses


def _missing_model_metric_results(
    metric_names: list[str],
    reason: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics = {name: 0.0 for name in metric_names}
    statuses = {
        name: {"status": "missing_model", "error": reason, "value": 0.0} for name in metric_names
    }
    return metrics, statuses


def _resolve_metric_model_inputs(
    source_payload: dict[str, Any],
    source_result_path: Path,
) -> tuple[Path | None, Path | None, str | None]:
    """Resolve the model / metric-model paths to evaluate for a metric row.

    Artifact paths are derived exclusively from the source discovery payload.
    Returns ``(model_path, metric_model_path, fallback_reason)`` where a non-None
    ``fallback_reason`` means no usable model is available and metrics should
    default to zero (``success_missing``).
    """
    discovery_status = str(source_payload.get("status") or "")
    if discovery_status != "success":
        return (
            None,
            None,
            (
                f"Source discovery status is '{discovery_status or 'missing'}'; "
                "defaulted metrics to 0 because no model is available"
            ),
        )

    metric_model_path = None
    candidate = _resolve_metric_model_path(source_payload, source_result_path)
    if candidate is not None and candidate.exists():
        metric_model_path = candidate

    model_path = None
    try:
        candidate = _resolve_model_path(source_payload, source_result_path)
    except ValueError:
        candidate = None
    if candidate is not None and candidate.exists():
        model_path = candidate

    if metric_model_path is not None or model_path is not None:
        return model_path, metric_model_path, None

    raw_model = str(source_payload.get("model_path") or "").strip()
    if not raw_model:
        return None, None, "Source discovery result has no exported model; defaulted metrics to 0"
    return None, None, f"Exported model path does not exist: {raw_model}; defaulted metrics to 0"


def source_model_is_available(
    row: dict[str, str],
    source_payload: dict[str, Any] | None = None,
) -> bool:
    """True if the row's source discovery result now resolves to a usable model.

    Used to decide whether a stale ``success_missing`` metric output can be
    recomputed into real metrics (the source discovery became available after
    the placeholder was written).
    """
    source_result_path = resolve_portable_path(row.get("source_result_path") or "")
    if source_payload is None:
        if not source_result_path.exists():
            return False
        try:
            source_payload = _load_json(source_result_path)
        except (OSError, json.JSONDecodeError, ValueError):
            return False
    _, _, fallback_reason = _resolve_metric_model_inputs(source_payload, source_result_path)
    return fallback_reason is None


def _metric_output_is_stale_success_missing(
    output_path: str | Path,
    row: dict[str, str],
) -> bool:
    """True if an existing metric output is a ``success_missing`` placeholder that
    can now be recomputed because the source discovery result became available."""
    try:
        payload = _load_json(resolve_portable_path(output_path))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if payload.get("status") != "success_missing":
        return False
    return source_model_is_available(row)


def _metric_output_is_reusable_success(
    output_path: str | Path,
    row: dict[str, str],
) -> bool:
    try:
        payload = _load_json(resolve_portable_path(output_path))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    status = payload.get("status")
    if status == "success":
        metric_statuses = payload.get("metric_statuses")
        if not isinstance(metric_statuses, dict):
            return False
        try:
            metric_names = _metric_names_from_row(row)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        for name in metric_names:
            record = metric_statuses.get(name)
            if not isinstance(record, dict) or record.get("status") != "success":
                return False
        return True
    if status == "success_missing":
        return _success_missing_output_is_terminal_placeholder(payload, row)
    return False


def _success_missing_output_is_terminal_placeholder(
    payload: dict[str, Any],
    row: dict[str, str],
) -> bool:
    metric_statuses = payload.get("metric_statuses")
    if not isinstance(metric_statuses, dict):
        return False
    try:
        metric_names = _metric_names_from_row(row)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    for name in metric_names:
        record = metric_statuses.get(name)
        if not isinstance(record, dict) or record.get("status") != "missing_model":
            return False
    return True


def _source_payload_from_row(
    row: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_result_path = resolve_portable_path(row["source_result_path"])
    if source_result_path.exists():
        try:
            source_payload = _load_json(source_result_path)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        else:
            return source_payload, _source_metadata(source_payload)

    payload = {
        "status": "failed",
        "experiment_id": row.get("experiment_id") or "",
        "log_id": row.get("log_id") or "",
        "log_path": "",
        "test_log_path": row.get("test_log_path") or "",
        "algorithm_name": row.get("algorithm_name") or "unknown",
        "backend": "saved_model",
        "hyperparameters": {},
        "runtime_seconds": None,
        "model_path": "",
        "seed": 0,
        "warnings": [],
        "metadata": {"config_hash": row.get("source_config_hash") or ""},
    }
    return payload, _source_metadata(payload)


def _metric_output_record(
    row: dict[str, str],
    *,
    source_payload: dict[str, Any],
    metric_names: list[str],
    metrics: dict[str, float | None],
    metric_statuses: dict[str, dict[str, Any]],
    model_type: str | None,
    status: str,
    error_message: str | None,
    runtime_seconds: float,
    metadata: dict[str, Any],
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    metric_result = MetricResult(
        experiment_id=str(source_payload.get("experiment_id") or row.get("experiment_id") or ""),
        log_id=str(source_payload.get("log_id") or row.get("log_id") or ""),
        log_path=str(source_payload.get("log_path") or ""),
        test_log_path=str(source_payload.get("test_log_path") or row.get("test_log_path") or ""),
        seed=int(source_payload.get("seed") or 0),
        algorithm_name=str(
            source_payload.get("algorithm_name") or row.get("algorithm_name") or "unknown"
        ),
        backend=str(source_payload.get("backend") or "saved_model"),
        hyperparameters=_mapping_copy(source_payload.get("hyperparameters")),
        discovered_model_type=model_type or _model_type_from_payload(source_payload),
        model_path=str(source_payload.get("model_path") or ""),
        metric_profile=str(row.get("metric_profile") or ""),
        metric_names=metric_names,
        metrics=metrics,
        metric_statuses={
            name: MetricRecord(**metric_status) for name, metric_status in metric_statuses.items()
        },
        metric_runtime_seconds=runtime_seconds,
        discovery_runtime_seconds=_optional_float(source_payload.get("runtime_seconds")),
        status=status,
        discovery_status=str(source_payload.get("status") or "failed"),
        error_message=error_message,
        warnings=_warnings_from_payload(source_payload),
        source_result_path=row.get("source_result_path"),
        source_config_hash=row.get("source_config_hash"),
        config_hash=str(row.get("source_config_hash") or source_metadata.get("config_hash") or ""),
        metadata=metadata,
        source_metadata=source_metadata,
    )
    return metric_result.to_json_record()


def _source_metadata(source_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = source_payload.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _warnings_from_payload(payload: dict[str, Any]) -> list[str]:
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [str(warning) for warning in warnings]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_type_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("discovered_model_type") or payload.get("model_type")
    return None if value in (None, "") else str(value)


def _looks_like_metric_output(payload: dict[str, Any]) -> bool:
    return "source_result_path" in payload and "metric_profile" in payload


def _portable_path(value: str | Path, project_root: Path) -> str:
    return portable_project_path(value)
