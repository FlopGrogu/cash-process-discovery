from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from process_discovery_cash.data.loading import load_event_log
from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.discovery.conversion import to_petri_net
from process_discovery_cash.discovery.registry import get_algorithm
from process_discovery_cash.evaluation.quality_metrics import (
    DEFAULT_METRICS,
    evaluate_discovery_result,
)
from process_discovery_cash.experiments.discovery_timeout import (
    discover_with_timeout,
    extract_discovery_timeout_seconds,
)
from process_discovery_cash.experiments.identity import execution_config
from process_discovery_cash.experiments.metric_timeout import (
    evaluate_with_timeout,
    extract_metric_timeout_seconds,
)
from process_discovery_cash.experiments.resource_tracking import (
    attach_resource_metadata,
    current_peak_rss_bytes,
)
from process_discovery_cash.experiments.result_schema import ExperimentResult, MetricRecord
from process_discovery_cash.experiments.saved_model_metrics import (
    METRIC_MODEL_ARTIFACT_TYPE,
    METRIC_MODEL_CACHE_VERSION,
    dump_metric_model_artifact,
)
from process_discovery_cash.experiments.tracking import collect_reproducibility_metadata
from process_discovery_cash.utils.paths import (
    ensure_parent,
    portable_project_path,
    resolve_portable_path,
)
from process_discovery_cash.utils.random import seed_everything
from process_discovery_cash.utils.recursion import (
    configured_recursion_limit,
)
from process_discovery_cash.utils.recursion import (
    recursion_limit as recursion_limit_context,
)


class ResultFileState(StrEnum):
    MISSING = "missing"
    CORRUPT = "corrupt"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    SKIPPED = "skipped"
    IDENTITY_MISMATCH = "identity_mismatch"
    SUCCESS = "success"


_SUCCESS_RESULT_REQUIRED_FIELDS = frozenset(
    {
        "algorithm_name",
        "backend",
        "discovered_model_type",
        "experiment_id",
        "hyperparameters",
        "log_id",
        "log_path",
        "metadata",
        "metric_statuses",
        "metrics",
        "seed",
        "test_log_path",
    }
)


@dataclass(frozen=True)
class ResultFileInspection:
    state: ResultFileState
    payload: dict[str, Any] | None = None


def load_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_manifest_index(
    manifest_path: str | Path,
    row_index: int,
    command_args: list[str] | None = None,
    force: bool = False,
) -> Path:
    rows = load_manifest_rows(manifest_path)
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"Manifest row index {row_index} outside 0..{len(rows) - 1}")
    return run_manifest_row(rows[row_index], command_args=command_args, force=force)


def run_slurm_array_task(
    manifest_path: str | Path,
    command_args: list[str] | None = None,
    force: bool = False,
) -> Path:
    task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not set")
    return run_manifest_index(
        manifest_path,
        int(task_id),
        command_args=command_args,
        force=force,
    )


def run_manifest_row(
    row: dict[str, str],
    command_args: list[str] | None = None,
    force: bool = False,
) -> Path:
    row_started = time.perf_counter()
    memory_started = current_peak_rss_bytes()
    timings: dict[str, Any] = {}
    output_path = ensure_parent(row["output_path"])
    if not force and is_successfully_completed(row, output_path):
        print(f"Skipped existing successful result {output_path}")
        return output_path

    params: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    metrics_config = _metrics_config_from_row(row)
    metric_names = _metric_names(metrics_config)

    seed = int(row["seed"])
    seed_everything(seed)

    algorithm_id = row.get("algorithm_id") or row["algorithm"]
    params = json.loads(row.get("algorithm_params_json") or row.get("params_json") or "{}")
    discovery_timeout_seconds: int | float | None = None
    metadata = collect_reproducibility_metadata(
        config_hash=row["config_hash"],
        seed=seed,
        command_args=command_args or sys.argv,
    )
    metadata["metrics_config"] = metrics_config
    metadata["execution_config"] = execution_config(params)
    metadata["preprocessing"] = {
        key: row.get(key)
        for key in [
            "dataset_id",
            "source_log_path",
            "discovery_log_path",
            "test_discovery_log_path",
            "artifact_kind",
            "artifact_sha256",
            "preprocessing_fingerprint",
            "preprocessing_metadata_path",
        ]
        if row.get(key)
    }

    try:
        metadata["execution_config"].setdefault(
            "recursion_limit",
            configured_recursion_limit(params),
        )
        discovery_timeout_seconds = extract_discovery_timeout_seconds(params)
        if discovery_timeout_seconds is not None:
            metadata["discovery"] = {"timeout_seconds": discovery_timeout_seconds}
        metric_timeout_seconds = extract_metric_timeout_seconds(params)
        train_log_path = _train_log_path(row)
        if _can_discover_from_path_without_loading(algorithm_id, metrics_config):
            train_log = None
            timings["load_train_log_seconds"] = 0.0
            timings["train_log_load_skipped"] = True
        else:
            with _record_timing(timings, "load_train_log_seconds"):
                train_log = load_event_log(
                    train_log_path,
                    cache_key=row.get("log_id"),
                )
            timings["train_log_load_skipped"] = False
        with _record_timing(timings, "get_algorithm_seconds"):
            algorithm = get_algorithm(algorithm_id)
        runtime_config = dict(params)
        runtime_config.setdefault("log_id", row["log_id"])
        runtime_config.setdefault("input_log_path", train_log_path)
        if train_log_path == row.get("discovery_log_path"):
            runtime_config.setdefault("input_artifact_kind", row.get("artifact_kind"))
        runtime_config.setdefault("output_dir", str(output_path.with_suffix("")))
        if metrics_config["export_model"] and algorithm_id == "split_miner":
            runtime_config.setdefault("keep_output_files", True)
        with recursion_limit_context(runtime_config):
            with _record_timing(timings, "discovery_call_seconds"):
                if discovery_timeout_seconds is None:
                    discovery_result = algorithm.discover(train_log, runtime_config)
                else:
                    discovery_result = discover_with_timeout(
                        algorithm,
                        train_log,
                        runtime_config,
                        discovery_timeout_seconds,
                    )
            timings["discovery_reported_runtime_seconds"] = discovery_result.runtime_seconds
            if discovery_result.status == "success":
                if metrics_config["export_model"]:
                    with _record_timing(timings, "model_export_seconds"):
                        _export_discovered_model(discovery_result, output_path)
                else:
                    timings["model_export_seconds"] = 0.0

                if metrics_config["enabled"]:
                    test_log_path = _test_log_path(row)
                    if test_log_path == train_log_path:
                        test_log = train_log
                        timings["load_test_log_seconds"] = 0.0
                        timings["test_log_reused_train_log"] = True
                    else:
                        with _record_timing(timings, "load_test_log_seconds"):
                            test_log = load_event_log(
                                test_log_path,
                                cache_key=_test_log_cache_key(row, train_log_path, test_log_path),
                            )
                        timings["test_log_reused_train_log"] = False
                    if metric_timeout_seconds is not None:
                        metadata.setdefault("metrics", {})["timeout_seconds"] = (
                            metric_timeout_seconds
                        )
                    with _record_timing(timings, "metrics_eval_call_seconds"):
                        metrics, metric_statuses, metric_timings = _evaluate_with_timings(
                            discovery_result,
                            test_log,
                            metric_names=metric_names,
                            metric_profile=metrics_config["profile"],
                            metric_timeout_seconds=metric_timeout_seconds,
                        )
                    timings["metrics"] = metric_timings
                else:
                    timings["load_test_log_seconds"] = 0.0
                    timings["test_log_reused_train_log"] = None
                    timings["metrics_eval_call_seconds"] = 0.0
                    timings["metrics"] = {}
                    metrics, metric_statuses = _not_computed_metric_results(
                        "Metrics disabled by manifest metrics config.",
                        metric_names,
                    )
            else:
                timings["model_export_seconds"] = 0.0
                timings["load_test_log_seconds"] = 0.0
                timings["test_log_reused_train_log"] = None
                timings["metrics_eval_call_seconds"] = 0.0
                timings["metrics"] = {}
                metrics, metric_statuses = _not_computed_metric_results(
                    f"Discovery status is {discovery_result.status}; metrics were skipped.",
                    metric_names,
                )
        timings["total_row_seconds_before_write"] = time.perf_counter() - row_started
        metadata["timings"] = timings
        attach_resource_metadata(metadata, memory_started)
        result = _build_result(row, params, discovery_result, metrics, metric_statuses, metadata)
    except Exception as exc:
        discovery_result = DiscoveryResult(
            algorithm_name=row.get("algorithm_id") or row.get("algorithm", "unknown"),
            backend_name=row.get("backend", "unknown"),
            hyperparameters=params,
            runtime_seconds=None,
            status="failed",
            model_type="unknown",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        metrics, metric_statuses = _not_computed_metric_results(
            "Experiment failed before evaluation",
            metric_names,
        )
        timings["total_row_seconds_before_write"] = time.perf_counter() - row_started
        metadata.setdefault("metrics_config", metrics_config)
        metadata["timings"] = timings
        attach_resource_metadata(metadata, memory_started)
        result = _build_result(row, params, discovery_result, metrics, metric_statuses, metadata)

    _write_json_atomically(output_path, result.to_json_record())
    return output_path


def _can_discover_from_path_without_loading(
    algorithm_id: str,
    metrics_config: Mapping[str, Any],
) -> bool:
    return algorithm_id == "split_miner" and not metrics_config["enabled"]


def _test_log_cache_key(
    row: Mapping[str, str],
    train_log_path: str,
    test_log_path: str,
) -> str | None:
    log_id = row.get("log_id")
    if not log_id or test_log_path == train_log_path:
        return log_id
    return f"{log_id}_test"


def _write_json_atomically(output_path: Path, payload: Mapping[str, Any]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@contextmanager
def _record_timing(timings: dict[str, Any], name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - started


def _evaluate_with_timings(
    discovery_result: DiscoveryResult,
    test_log: Any,
    *,
    metric_names: list[str] | None = None,
    metric_profile: str = "pm4py_default",
    metric_timeout_seconds: int | float | None = None,
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]], dict[str, Any]]:
    if metric_timeout_seconds is not None:
        # Same process-isolated timeout the post-hoc metric worker uses: a slow
        # metric yields a clean per-metric "timeout" status instead of taking the
        # whole trial down, matching the explore metric pass.
        evaluation = evaluate_with_timeout(
            discovery_result,
            test_log,
            metric_names,
            metric_profile=metric_profile,
            timeout_seconds=metric_timeout_seconds,
        )
        return evaluation.metrics, evaluation.statuses, evaluation.timings
    result = evaluate_discovery_result(
        discovery_result,
        test_log,
        metric_names=metric_names,
        metric_profile=metric_profile,
        include_timings=True,
    )
    metrics, metric_statuses, metric_timings = result
    return metrics, metric_statuses, metric_timings


def _export_discovered_model(discovery_result: DiscoveryResult, output_path: Path) -> None:
    if discovery_result.model_path:
        existing_artifact_path = Path(discovery_result.model_path)
        if existing_artifact_path.exists():
            portable_artifact_path = portable_project_path(existing_artifact_path)
            discovery_result.model_path = portable_artifact_path
            discovery_result.metadata["model_artifact_path"] = portable_artifact_path
            discovery_result.metadata["model_artifact_type"] = _existing_model_artifact_type(
                existing_artifact_path,
                discovery_result.model_type,
            )
            _export_metric_model_artifact(discovery_result, output_path)
            return

    if discovery_result.discovered_model is None:
        _record_model_export_warning(discovery_result, "No discovered model available")
        return

    artifact_dir = output_path.with_suffix("")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        if discovery_result.model_type == "bpmn":
            import pm4py

            artifact_path = artifact_dir / "discovered_model.bpmn"
            pm4py.write_bpmn(discovery_result.discovered_model, artifact_path.as_posix())
            artifact_type = "bpmn"
        else:
            petri_net, conversion_error = to_petri_net(
                discovery_result.discovered_model,
                discovery_result.model_type,
            )
            if petri_net is None:
                _record_model_export_warning(
                    discovery_result,
                    conversion_error or "Could not convert model to Petri net",
                )
                return
            net, initial_marking, final_marking = petri_net
            import pm4py

            artifact_path = artifact_dir / "discovered_model.pnml"
            pm4py.write_pnml(
                net,
                initial_marking,
                final_marking,
                artifact_path.as_posix(),
            )
            artifact_type = "pnml"
    except Exception as exc:
        _record_model_export_warning(discovery_result, f"{type(exc).__name__}: {exc}")
        return

    portable_artifact_path = portable_project_path(artifact_path)
    discovery_result.model_path = portable_artifact_path
    discovery_result.metadata["model_artifact_path"] = portable_artifact_path
    discovery_result.metadata["model_artifact_type"] = artifact_type
    _export_metric_model_artifact(discovery_result, output_path)


def _existing_model_artifact_type(artifact_path: Path, model_type: str) -> str:
    suffix = artifact_path.suffix.lower()
    if suffix == ".bpmn":
        return "bpmn"
    if suffix == ".pnml":
        return "pnml"
    return model_type


def _record_model_export_warning(discovery_result: DiscoveryResult, reason: str) -> None:
    message = f"Model export skipped: {reason}"
    if message not in discovery_result.warnings:
        discovery_result.warnings.append(message)
    discovery_result.metadata["model_artifact_error"] = reason


def _export_metric_model_artifact(discovery_result: DiscoveryResult, output_path: Path) -> None:
    try:
        petri_net, conversion_error = to_petri_net(
            discovery_result.discovered_model,
            discovery_result.model_type,
        )
    except Exception as exc:
        _record_metric_model_export_warning(discovery_result, f"{type(exc).__name__}: {exc}")
        return
    if petri_net is None:
        _record_metric_model_export_warning(
            discovery_result,
            conversion_error or "Could not convert model to Petri net",
        )
        return

    artifact_dir = output_path.with_suffix("")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metric_model_path = artifact_dir / "discovered_model.metric.joblib"
    source_model_path = (
        discovery_result.metadata.get("model_artifact_path") or discovery_result.model_path
    )
    try:
        dump_metric_model_artifact(
            metric_model_path,
            petri_net,
            source_model_path=(
                portable_project_path(source_model_path) if source_model_path else None
            ),
            source_model_type=str(discovery_result.metadata.get("model_artifact_type") or ""),
            algorithm_name=discovery_result.algorithm_name,
            config_hash=str(discovery_result.metadata.get("config_hash") or ""),
        )
    except Exception as exc:
        _record_metric_model_export_warning(discovery_result, f"{type(exc).__name__}: {exc}")
        return

    discovery_result.metadata["metric_model_artifact_path"] = portable_project_path(
        metric_model_path
    )
    discovery_result.metadata["metric_model_artifact_type"] = METRIC_MODEL_ARTIFACT_TYPE
    discovery_result.metadata["metric_model_cache_version"] = METRIC_MODEL_CACHE_VERSION


def _record_metric_model_export_warning(discovery_result: DiscoveryResult, reason: str) -> None:
    message = f"Metric model export skipped: {reason}"
    if message not in discovery_result.warnings:
        discovery_result.warnings.append(message)
    discovery_result.metadata["metric_model_artifact_error"] = reason


def is_successfully_completed(row: dict[str, str], output_path: str | Path | None = None) -> bool:
    result_path = Path(output_path or row["output_path"])
    inspection = inspect_result_file(row, result_path)
    return inspection.state in {ResultFileState.SUCCESS, ResultFileState.SKIPPED}


def inspect_result_file(
    row: Mapping[str, str],
    output_path: str | Path,
    *,
    require_hash_match: bool = True,
) -> ResultFileInspection:
    result_path = Path(output_path)
    if not result_path.exists():
        return ResultFileInspection(ResultFileState.MISSING)
    try:
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ResultFileInspection(ResultFileState.CORRUPT)
    if not isinstance(payload, dict):
        return ResultFileInspection(ResultFileState.CORRUPT)

    status = payload.get("status")
    if status == "failed":
        return ResultFileInspection(ResultFileState.FAILED, payload)
    if status == "timeout":
        return ResultFileInspection(ResultFileState.TIMEOUT, payload)
    if status == "unsupported":
        return ResultFileInspection(ResultFileState.UNSUPPORTED, payload)
    if status == "skipped":
        return ResultFileInspection(ResultFileState.SKIPPED, payload)
    if status != "success":
        return ResultFileInspection(ResultFileState.INCOMPLETE, payload)
    if not _is_complete_success_payload(payload):
        return ResultFileInspection(ResultFileState.INCOMPLETE, payload)

    row_hash = row.get("config_hash") or row.get("config_id")
    metadata = payload.get("metadata")
    result_hash = metadata.get("config_hash") if isinstance(metadata, dict) else None
    if require_hash_match and row_hash and result_hash and str(row_hash) != str(result_hash):
        return ResultFileInspection(ResultFileState.IDENTITY_MISMATCH, payload)
    return ResultFileInspection(ResultFileState.SUCCESS, payload)


def _is_complete_success_payload(payload: Mapping[str, Any]) -> bool:
    if not _SUCCESS_RESULT_REQUIRED_FIELDS.issubset(payload):
        return False
    return all(
        isinstance(payload.get(field), Mapping)
        for field in ["hyperparameters", "metadata", "metric_statuses", "metrics"]
    )


def _build_result(
    row: dict[str, str],
    params: dict[str, Any],
    discovery_result: DiscoveryResult,
    metrics: dict[str, float | int | None],
    metric_statuses: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> ExperimentResult:
    merged_metadata = dict(metadata)
    discovery_metadata = dict(merged_metadata.get("discovery") or {})
    discovery_metadata.update(discovery_result.metadata)
    merged_metadata["discovery"] = discovery_metadata
    warnings = list(discovery_result.warnings)
    for warning in merged_metadata.get("warnings", []):
        if warning not in warnings:
            warnings.append(warning)
    return ExperimentResult(
        experiment_id=row["experiment_id"],
        log_id=row["log_id"],
        log_path=_source_log_path(row),
        test_log_path=_source_test_log_path(row),
        seed=int(row["seed"]),
        algorithm_name=discovery_result.algorithm_name,
        backend=discovery_result.backend_name,
        hyperparameters=params,
        discovered_model_type=discovery_result.model_type,
        model_path=discovery_result.model_path,
        metrics=metrics,
        metric_statuses={name: MetricRecord(**status) for name, status in metric_statuses.items()},
        runtime_seconds=discovery_result.runtime_seconds,
        status=discovery_result.status,
        error_message=discovery_result.error_message,
        warnings=warnings,
        metadata=merged_metadata,
    )


def _not_computed_metric_results(
    reason: str,
    metric_names: list[str] | None = None,
) -> tuple[dict[str, None], dict[str, dict[str, Any]]]:
    names = [name for name in (metric_names or DEFAULT_METRICS) if name != "composite_score"]
    metrics = {name: None for name in names}
    metric_statuses = {
        name: {"status": "not_computed", "error": reason, "value": None} for name in names
    }
    return metrics, metric_statuses


def _metrics_config_from_row(row: dict[str, str]) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "profile": "pm4py_default",
        "names": DEFAULT_METRICS,
        "export_model": False,
    }
    raw = row.get("metrics_json") or "{}"
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("metrics_json must contain a JSON object")

    merged = dict(defaults)
    merged.update({key: value for key, value in parsed.items() if value is not None})
    merged["enabled"] = bool(merged["enabled"])
    merged["export_model"] = bool(merged["export_model"])
    merged["profile"] = str(merged["profile"])
    merged["names"] = _metric_names(merged)
    return merged


def _metric_names(metrics_config: dict[str, Any]) -> list[str]:
    names = metrics_config.get("names") or DEFAULT_METRICS
    if isinstance(names, str):
        names = [names]
    return [str(name) for name in names if str(name) != "composite_score"]


def _train_log_path(row: dict[str, str]) -> str:
    artifact_path = row.get("discovery_log_path")
    if artifact_path and resolve_portable_path(artifact_path).exists():
        return artifact_path
    return row.get("train_log_path") or row.get("log_path") or row["source_log_path"]


def _test_log_path(row: dict[str, str]) -> str:
    artifact_path = row.get("test_discovery_log_path")
    if artifact_path and resolve_portable_path(artifact_path).exists():
        return artifact_path
    return row.get("test_log_path") or row.get("evaluation_log_path") or _train_log_path(row)


def _source_log_path(row: Mapping[str, str]) -> str:
    return row.get("log_path") or row.get("train_log_path") or row["source_log_path"]


def _source_test_log_path(row: Mapping[str, str]) -> str:
    return row.get("test_log_path") or row.get("evaluation_log_path") or _source_log_path(row)
