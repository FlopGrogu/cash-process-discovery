from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib

from process_discovery_cash.data.loading import load_event_log
from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.evaluation.quality_metrics import (
    DEFAULT_METRICS,
    evaluate_discovery_result,
)
from process_discovery_cash.experiments.result_schema import MetricRecord, MetricResult
from process_discovery_cash.utils.paths import ensure_parent, resolve_portable_path

METRIC_MODEL_CACHE_VERSION = 1
METRIC_MODEL_ARTIFACT_TYPE = "petri_net_joblib"


def evaluate_saved_result_model(
    result_path: str | Path,
    *,
    metric_profile: str = "alignment",
    metric_names: list[str] | None = None,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    result_path = Path(result_path)
    output_path = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(
            result_path,
            metric_profile,
        )
    )
    if output_path.exists() and not force and _is_successful_metric_output(output_path):
        return output_path

    started = time.perf_counter()
    source_payload = _load_json(result_path)
    model_path = _resolve_model_path(source_payload, result_path)
    metric_model_path = _resolve_metric_model_path(source_payload, result_path)
    test_log_path = source_payload.get("test_log_path") or source_payload.get("log_path")
    if not test_log_path:
        raise ValueError(f"Source result does not contain a test log path: {result_path}")

    metadata = (
        source_payload.get("metadata") if isinstance(source_payload.get("metadata"), dict) else {}
    )
    resolved_metric_names = _metric_names(metric_names)
    fallback_reason = _missing_model_reason(
        source_payload,
        model_path=model_path,
        metric_model_path=metric_model_path,
    )
    if fallback_reason is not None:
        model_type = _model_type_from_payload(source_payload)
        metrics, metric_statuses = _missing_model_metric_results(
            resolved_metric_names,
            fallback_reason,
        )
        load_backend = "fallback_zero"
        model_timings: dict[str, Any] = {}
        log_load_seconds = 0.0
        metric_timings: dict[str, Any] = {
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

        log_load_started = time.perf_counter()
        test_log = load_event_log(test_log_path)
        log_load_seconds = time.perf_counter() - log_load_started

        discovery_result = DiscoveryResult(
            algorithm_name=source_payload.get("algorithm_name", "unknown"),
            backend_name=source_payload.get("backend", "unknown"),
            hyperparameters=source_payload.get("hyperparameters") or {},
            runtime_seconds=source_payload.get("runtime_seconds"),
            status="success",
            model_type=model_type,
            model_path=model_path.as_posix() if model_path is not None else None,
            discovered_model=discovered_model,
        )
        metrics, metric_statuses, metric_timings = evaluate_discovery_result(
            discovery_result,
            test_log,
            metric_names=resolved_metric_names,
            metric_profile=metric_profile,
            include_timings=True,
        )
        status = "success"
        error_message = None

    metric_runtime_seconds = time.perf_counter() - started
    output_record = MetricResult(
        experiment_id=str(source_payload.get("experiment_id") or ""),
        log_id=str(source_payload.get("log_id") or ""),
        log_path=str(source_payload.get("log_path") or ""),
        test_log_path=str(test_log_path),
        seed=int(source_payload.get("seed") or 0),
        algorithm_name=str(source_payload.get("algorithm_name") or "unknown"),
        backend=str(source_payload.get("backend") or "unknown"),
        hyperparameters=_mapping_copy(source_payload.get("hyperparameters")),
        discovered_model_type=model_type,
        model_path=model_path.as_posix() if model_path is not None else None,
        metric_profile=metric_profile,
        metric_names=resolved_metric_names,
        metrics=metrics,
        metric_statuses={
            name: MetricRecord(**metric_status) for name, metric_status in metric_statuses.items()
        },
        metric_runtime_seconds=metric_runtime_seconds,
        discovery_runtime_seconds=_optional_float(source_payload.get("runtime_seconds")),
        status=status,
        discovery_status=str(source_payload.get("status") or "failed"),
        error_message=error_message,
        warnings=_warnings_from_payload(source_payload),
        source_result_path=result_path.as_posix(),
        source_config_hash=str(metadata.get("config_hash") or ""),
        config_hash=str(metadata.get("config_hash") or ""),
        metadata={
            "timings": {
                **model_timings,
                "model_load_backend": load_backend,
                "load_test_log_seconds": log_load_seconds,
                "metrics": metric_timings,
                "total_seconds_before_write": metric_runtime_seconds,
            }
        },
        source_metadata=dict(metadata),
    ).to_json_record()

    output_path = ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def evaluate_saved_result_tree(
    results_root: str | Path,
    output_dir: str | Path,
    *,
    metric_profile: str = "alignment",
    metric_names: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    written: list[Path] = []
    for result_path in sorted(results_root.rglob("*.json")):
        try:
            if _is_saved_metric_output(result_path):
                continue
            relative_path = result_path.relative_to(results_root)
            output_path = output_dir / relative_path
            written.append(
                evaluate_saved_result_model(
                    result_path,
                    metric_profile=metric_profile,
                    metric_names=metric_names,
                    output_path=output_path,
                    force=force,
                )
            )
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return written


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_model_path(source_payload: dict[str, Any], result_path: Path) -> Path | None:
    metadata = (
        source_payload.get("metadata") if isinstance(source_payload.get("metadata"), dict) else {}
    )
    discovery_metadata = (
        metadata.get("discovery") if isinstance(metadata.get("discovery"), dict) else {}
    )
    raw_model_path = source_payload.get("model_path") or discovery_metadata.get(
        "model_artifact_path"
    )
    if not raw_model_path:
        return None

    model_path = resolve_portable_path(str(raw_model_path), base_path=result_path)
    if model_path.exists():
        return model_path
    raw_path = Path(str(raw_model_path))
    result_relative_path = result_path.parent / raw_path
    if result_relative_path.exists():
        return result_relative_path
    return model_path


def _resolve_metric_model_path(source_payload: dict[str, Any], result_path: Path) -> Path | None:
    metadata = (
        source_payload.get("metadata") if isinstance(source_payload.get("metadata"), dict) else {}
    )
    discovery_metadata = (
        metadata.get("discovery") if isinstance(metadata.get("discovery"), dict) else {}
    )
    raw_path = discovery_metadata.get("metric_model_artifact_path")
    if not raw_path:
        return None
    metric_model_path = resolve_portable_path(str(raw_path), base_path=result_path)
    if metric_model_path.exists():
        return metric_model_path
    raw_metric_path = Path(str(raw_path))
    result_relative_path = result_path.parent / raw_metric_path
    if result_relative_path.exists():
        return result_relative_path
    return metric_model_path


def _load_model_artifact(model_path: Path) -> tuple[str, Any]:
    import pm4py

    suffix = model_path.suffix.lower()
    if suffix == ".pnml":
        return "petri_net", pm4py.read_pnml(model_path.as_posix())
    if suffix == ".bpmn":
        return "bpmn", pm4py.read_bpmn(model_path.as_posix())
    raise ValueError(f"Unsupported model artifact format: {model_path}")


def _load_model_artifact_with_fallback(
    model_path: Path | None,
    *,
    metric_model_path: Path | None = None,
) -> tuple[str, str, Any, dict[str, float]]:
    if metric_model_path is not None and metric_model_path.exists():
        started = time.perf_counter()
        try:
            model_type, discovered_model = _load_metric_model_artifact(metric_model_path)
        except Exception:
            pass
        else:
            return (
                "joblib",
                model_type,
                discovered_model,
                {"load_metric_model_seconds": time.perf_counter() - started},
            )

    if model_path is None:
        raise ValueError("Source discovery result has no portable model artifact")

    started = time.perf_counter()
    model_type, discovered_model = _load_model_artifact(model_path)
    backend = model_path.suffix.lower().lstrip(".") or "unknown"
    return (
        backend,
        model_type,
        discovered_model,
        {"load_model_seconds": time.perf_counter() - started},
    )


def _load_metric_model_artifact(metric_model_path: Path) -> tuple[str, Any]:
    payload = joblib.load(metric_model_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Metric model artifact must be a dict: {metric_model_path}")
    if payload.get("artifact_type") != METRIC_MODEL_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported metric model artifact type: {metric_model_path}")
    if payload.get("cache_version") != METRIC_MODEL_CACHE_VERSION:
        raise ValueError(f"Unsupported metric model artifact version: {metric_model_path}")
    if payload.get("model_type") != "petri_net":
        raise ValueError(f"Unsupported metric model type: {metric_model_path}")
    try:
        return "petri_net", (payload["net"], payload["initial_marking"], payload["final_marking"])
    except KeyError as exc:
        raise ValueError(
            f"Metric model artifact is missing field {exc.args[0]}: {metric_model_path}"
        ) from exc


def dump_metric_model_artifact(
    metric_model_path: str | Path,
    petri_net: tuple[Any, Any, Any],
    *,
    source_model_path: str | None,
    source_model_type: str | None,
    algorithm_name: str,
    config_hash: str | None = None,
) -> Path:
    net, initial_marking, final_marking = petri_net
    metric_model_path = Path(metric_model_path)
    metric_model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": METRIC_MODEL_ARTIFACT_TYPE,
        "cache_version": METRIC_MODEL_CACHE_VERSION,
        "model_type": "petri_net",
        "net": net,
        "initial_marking": initial_marking,
        "final_marking": final_marking,
        "source_model_path": source_model_path,
        "source_model_type": source_model_type,
        "algorithm_name": algorithm_name,
        "config_hash": config_hash,
    }
    joblib.dump(payload, metric_model_path)
    return metric_model_path


def _metric_names(metric_names: list[str] | None) -> list[str]:
    return [name for name in (metric_names or DEFAULT_METRICS) if name != "composite_score"]


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _missing_model_metric_results(
    metric_names: list[str],
    reason: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    metrics = {name: 0.0 for name in metric_names}
    statuses = {
        name: {"status": "missing_model", "error": reason, "value": 0.0} for name in metric_names
    }
    return metrics, statuses


def _missing_model_reason(
    source_payload: dict[str, Any],
    *,
    model_path: Path | None,
    metric_model_path: Path | None,
) -> str | None:
    discovery_status = str(source_payload.get("status") or "")
    if discovery_status != "success":
        return (
            f"Source discovery status is '{discovery_status or 'missing'}'; "
            "defaulted metrics to 0 because no model is available"
        )
    if metric_model_path is not None and metric_model_path.exists():
        return None
    if model_path is None:
        return "Source discovery result has no exported model; defaulted metrics to 0"
    if not model_path.exists():
        return f"Exported model path does not exist: {model_path}; defaulted metrics to 0"
    return None


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


def _default_output_path(result_path: Path, metric_profile: str) -> Path:
    return result_path.with_name(f"{result_path.stem}_metrics_{metric_profile}.json")


def _is_successful_metric_output(output_path: Path) -> bool:
    try:
        payload = _load_json(output_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return payload.get("status") in {"success", "success_missing"}


def _is_saved_metric_output(path: Path) -> bool:
    payload = _load_json(path)
    return "source_result_path" in payload and "metric_profile" in payload


def _model_type_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("discovered_model_type") or payload.get("model_type")
    return None if value in (None, "") else str(value)
