"""Scalar objective computed from a trial's quality metrics.

The objective is a weighted mean of the configured quality metrics, each in
[0, 1]. Trials whose result is not a full success (discovery failed/timed out,
or any weighted metric is missing or non-success) receive ``failed_value``
(default 0.0) — the "crashed cost" convention from SMAC (Hutter et al., 2011),
so the sampler learns to avoid those regions instead of ignoring them. With
equal weights the objective equals the mean score used by
``v6.select_best_v6_configs``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObjectiveOutcome:
    value: float
    run_status: str
    metric_values: dict[str, float | None] = field(default_factory=dict)


def compute_objective(
    metrics: Mapping[str, Any] | None,
    metric_statuses: Mapping[str, Any] | None,
    weights: Mapping[str, float],
    *,
    failed_value: float = 0.0,
) -> float:
    """Weighted mean of the weighted metrics, or ``failed_value`` if any is unusable."""
    metrics = metrics or {}
    metric_statuses = metric_statuses or {}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Objective weights must sum to a positive value")

    weighted_sum = 0.0
    for name, weight in weights.items():
        value = _numeric_metric_value(metrics.get(name))
        if value is None:
            return failed_value
        status = _metric_status(metric_statuses.get(name))
        if status is not None and status != "success":
            return failed_value
        weighted_sum += weight * value
    return weighted_sum / total_weight


def objective_from_result_payload(
    payload: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    failed_value: float = 0.0,
) -> ObjectiveOutcome:
    """Objective for a completed run's result JSON payload (``ExperimentResult`` shape)."""
    run_status = str(payload.get("status") or "unknown")
    metrics = payload.get("metrics") or {}
    metric_values = {name: _numeric_metric_value(metrics.get(name)) for name in weights}
    if run_status != "success":
        return ObjectiveOutcome(
            value=failed_value, run_status=run_status, metric_values=metric_values
        )
    value = compute_objective(
        metrics,
        payload.get("metric_statuses") or {},
        weights,
        failed_value=failed_value,
    )
    return ObjectiveOutcome(value=value, run_status=run_status, metric_values=metric_values)


def _numeric_metric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric_status(record: Any) -> str | None:
    if record is None:
        return None
    if isinstance(record, Mapping):
        status = record.get("status")
        return None if status is None else str(status)
    status = getattr(record, "status", None)
    return None if status is None else str(status)
