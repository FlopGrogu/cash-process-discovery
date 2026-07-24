from __future__ import annotations

import time
from typing import Any

from process_discovery_cash.data.loading import ensure_pm4py_event_log
from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.discovery.conversion import to_petri_net

DEFAULT_METRICS = ["fitness", "precision", "generalization", "simplicity"]
LOG_METRICS = {"fitness", "precision", "generalization"}
METRIC_PROFILES = {"pm4py_default", "token", "alignment"}


def evaluate_discovery_result(
    discovery_result: DiscoveryResult,
    test_log: Any,
    metric_names: list[str] | None = None,
    *,
    metric_profile: str = "pm4py_default",
    include_timings: bool = False,
) -> (
    tuple[dict[str, float | None], dict[str, dict[str, Any]]]
    | tuple[dict[str, float | None], dict[str, dict[str, Any]], dict[str, Any]]
):
    started = time.perf_counter()
    metric_profile = _normalize_metric_profile(metric_profile)
    timings: dict[str, Any] = {}
    timings["profile"] = metric_profile
    metric_names = [name for name in (metric_names or DEFAULT_METRICS) if name != "composite_score"]
    metrics: dict[str, float | None] = {name: None for name in metric_names}
    statuses: dict[str, dict[str, Any]] = {
        name: {"status": "not_computed", "error": None, "value": None} for name in metric_names
    }

    if discovery_result.status != "success":
        timings["total_seconds"] = time.perf_counter() - started
        if include_timings:
            return metrics, statuses, timings
        return metrics, statuses

    conversion_started = time.perf_counter()
    petri_net, conversion_error = to_petri_net(
        discovery_result.discovered_model,
        discovery_result.model_type,
    )
    timings["model_conversion_seconds"] = time.perf_counter() - conversion_started
    if petri_net is None:
        for name in metric_names:
            statuses[name] = {
                "status": "unsupported_model_type",
                "error": conversion_error,
                "value": None,
            }
        timings["total_seconds"] = time.perf_counter() - started
        if include_timings:
            return metrics, statuses, timings
        return metrics, statuses

    log_type_before = _type_name(test_log)
    log_type_after: str | None = None
    normalized_log: Any | None = None
    log_conversion_error: str | None = None
    if any(metric_name in LOG_METRICS for metric_name in metric_names):
        log_conversion_started = time.perf_counter()
        try:
            normalized_log = ensure_pm4py_event_log(test_log)
            log_type_after = _type_name(normalized_log)
        except Exception as exc:
            log_conversion_error = f"{type(exc).__name__}: {exc}"
        timings["log_conversion_seconds"] = time.perf_counter() - log_conversion_started

    model_object_type = _type_name(discovery_result.discovered_model)
    metric_timings: dict[str, float] = {}
    for metric_name in metric_names:
        metric_started = time.perf_counter()
        if metric_name in LOG_METRICS and log_conversion_error is not None:
            value, status, error = (
                None,
                "backend_error",
                _metric_error_context(
                    metric_name=metric_name,
                    model_type=discovery_result.model_type,
                    log_type_before=log_type_before,
                    log_type_after=log_type_after,
                    model_object_type=model_object_type,
                    error=f"Log conversion failed: {log_conversion_error}",
                ),
            )
        else:
            metric_log = normalized_log if metric_name in LOG_METRICS else None
            value, status, error = _compute_petri_net_metric(
                metric_name,
                metric_log,
                petri_net,
                model_type=discovery_result.model_type,
                log_type_before=log_type_before,
                log_type_after=log_type_after,
                model_object_type=model_object_type,
                metric_profile=metric_profile,
            )
        metrics[metric_name] = value
        statuses[metric_name] = {"status": status, "error": error, "value": value}
        metric_timings[metric_name] = time.perf_counter() - metric_started

    timings["metric_seconds"] = metric_timings
    timings["total_seconds"] = time.perf_counter() - started
    if include_timings:
        return metrics, statuses, timings
    return metrics, statuses


def _compute_petri_net_metric(
    metric_name: str,
    event_log: Any | None,
    petri_net: tuple[Any, Any, Any],
    *,
    model_type: str,
    log_type_before: str,
    log_type_after: str | None,
    model_object_type: str,
    metric_profile: str,
) -> tuple[float | None, str, str | None]:
    try:
        net, initial_marking, final_marking = petri_net
    except Exception as exc:
        return (
            None,
            "unsupported_model_type",
            _metric_error_context(
                metric_name=metric_name,
                model_type=model_type,
                log_type_before=log_type_before,
                log_type_after=log_type_after,
                model_object_type=model_object_type,
                error=f"Expected Petri net tuple: {exc}",
            ),
        )

    try:
        if metric_name == "fitness":
            from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

            variant = _replay_fitness_variant(replay_fitness, metric_profile)
            if variant is None:
                raw_value = replay_fitness.apply(
                    event_log,
                    net,
                    initial_marking,
                    final_marking,
                )
            else:
                raw_value = replay_fitness.apply(
                    event_log,
                    net,
                    initial_marking,
                    final_marking,
                    variant=variant,
                )
            if isinstance(raw_value, dict):
                value = raw_value.get("log_fitness", raw_value.get("averageFitness"))
            else:
                value = raw_value
            return _as_float(value), "success", None

        if metric_name == "precision":
            from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

            variant = _precision_variant(precision_evaluator, metric_profile)
            if variant is None:
                value = precision_evaluator.apply(
                    event_log,
                    net,
                    initial_marking,
                    final_marking,
                )
            else:
                value = precision_evaluator.apply(
                    event_log,
                    net,
                    initial_marking,
                    final_marking,
                    variant=variant,
                )
            return (
                _as_float(value),
                "success",
                None,
            )

        if metric_name == "generalization":
            from pm4py.algo.evaluation.generalization import algorithm as generalization

            return (
                _as_float(generalization.apply(event_log, net, initial_marking, final_marking)),
                "success",
                None,
            )

        if metric_name == "simplicity":
            from pm4py.algo.evaluation.simplicity import algorithm as simplicity

            return (
                _as_float(simplicity.apply(net)),
                "success",
                None,
            )

        return None, "not_computed", f"Unknown metric: {metric_name}"
    except ModuleNotFoundError as exc:
        return (
            None,
            "backend_error",
            _metric_error_context(
                metric_name=metric_name,
                model_type=model_type,
                log_type_before=log_type_before,
                log_type_after=log_type_after,
                model_object_type=model_object_type,
                error=f"pm4py metric backend unavailable: {exc}",
            ),
        )
    except Exception as exc:
        return (
            None,
            "backend_error",
            _metric_error_context(
                metric_name=metric_name,
                model_type=model_type,
                log_type_before=log_type_before,
                log_type_after=log_type_after,
                model_object_type=model_object_type,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalize_metric_profile(metric_profile: str) -> str:
    if metric_profile == "default":
        return "pm4py_default"
    if metric_profile not in METRIC_PROFILES:
        allowed = ", ".join(sorted(METRIC_PROFILES))
        raise ValueError(f"Unknown metric profile '{metric_profile}'. Expected one of: {allowed}")
    return metric_profile


def _replay_fitness_variant(replay_fitness: Any, metric_profile: str) -> Any | None:
    if metric_profile == "token":
        return replay_fitness.Variants.TOKEN_BASED
    if metric_profile == "alignment":
        return replay_fitness.Variants.ALIGNMENT_BASED
    return None


def _precision_variant(precision_evaluator: Any, metric_profile: str) -> Any | None:
    if metric_profile == "token":
        return precision_evaluator.Variants.ETCONFORMANCE_TOKEN
    if metric_profile == "alignment":
        return precision_evaluator.Variants.ALIGN_ETCONFORMANCE
    return None


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _metric_error_context(
    *,
    metric_name: str,
    model_type: str,
    log_type_before: str,
    log_type_after: str | None,
    model_object_type: str,
    error: str,
) -> str:
    return (
        f"metric={metric_name}; model_type={model_type}; "
        f"log_type_before={log_type_before}; "
        f"log_type_after={log_type_after or 'unavailable'}; "
        f"discovered_model_object_type={model_object_type}; error={error}"
    )
