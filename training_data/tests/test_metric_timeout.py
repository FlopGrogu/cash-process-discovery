from __future__ import annotations

import multiprocessing
import time

import pytest

from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.experiments import metric_timeout
from process_discovery_cash.experiments.dynamic_metric_worker import _with_capped_metric_timeout
from process_discovery_cash.experiments.metric_timeout import (
    METRIC_TIMEOUT_FIELD,
    evaluate_with_timeout,
    extract_metric_timeout_seconds,
)

_FORK_AVAILABLE = "fork" in multiprocessing.get_all_start_methods()
requires_fork = pytest.mark.skipif(not _FORK_AVAILABLE, reason="fork start method unavailable")


def _discovery_result() -> DiscoveryResult:
    return DiscoveryResult(
        algorithm_name="alpha_miner",
        backend_name="saved_model",
        hyperparameters={},
        runtime_seconds=None,
        status="success",
        model_type="petri_net",
        discovered_model=object(),
    )


def test_extract_metric_timeout_seconds_empty_is_none() -> None:
    assert extract_metric_timeout_seconds({}) is None
    assert extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: ""}) is None
    assert extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: None}) is None


def test_extract_metric_timeout_seconds_normalizes_integral_float() -> None:
    assert extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: "1800"}) == 1800
    assert isinstance(extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: 1800.0}), int)
    assert extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: "0.5"}) == 0.5


def test_extract_metric_timeout_seconds_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: "0"})
    with pytest.raises(ValueError):
        extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: "-5"})
    with pytest.raises(ValueError):
        extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: True})
    with pytest.raises(ValueError):
        extract_metric_timeout_seconds({METRIC_TIMEOUT_FIELD: "abc"})


@requires_fork
def test_evaluate_with_timeout_returns_metrics_on_success(monkeypatch) -> None:
    def fake_evaluate(discovery_result, test_log, *, metric_names, metric_profile, include_timings):
        assert include_timings is True
        metrics = {name: 1.0 for name in metric_names}
        statuses = {
            name: {"status": "success", "value": 1.0, "error": None} for name in metric_names
        }
        return metrics, statuses, {"profile": metric_profile}

    monkeypatch.setattr(metric_timeout, "evaluate_discovery_result", fake_evaluate)
    monkeypatch.setattr(
        metric_timeout, "_multiprocessing_context", lambda: multiprocessing.get_context("fork")
    )

    evaluation = evaluate_with_timeout(
        _discovery_result(),
        object(),
        ["fitness", "precision"],
        metric_profile="token",
        timeout_seconds=5,
    )

    assert evaluation.timed_out is False
    assert evaluation.metrics == {"fitness": 1.0, "precision": 1.0}
    assert evaluation.statuses["fitness"]["status"] == "success"
    assert evaluation.timings["profile"] == "token"


@requires_fork
def test_evaluate_with_timeout_marks_timeout_on_slow_evaluation(monkeypatch) -> None:
    def slow_evaluate(*_args, **_kwargs):
        time.sleep(5)
        raise AssertionError("timeout should terminate evaluation before completion")

    monkeypatch.setattr(metric_timeout, "evaluate_discovery_result", slow_evaluate)
    monkeypatch.setattr(
        metric_timeout, "_multiprocessing_context", lambda: multiprocessing.get_context("fork")
    )

    evaluation = evaluate_with_timeout(
        _discovery_result(),
        object(),
        ["fitness", "precision"],
        metric_profile="token",
        timeout_seconds=0.05,
    )

    assert evaluation.timed_out is True
    assert evaluation.metrics == {"fitness": None, "precision": None}
    assert all(status["status"] == "timeout" for status in evaluation.statuses.values())
    assert "exceeded" in evaluation.statuses["fitness"]["error"]
    assert evaluation.timings["timed_out"] is True
    assert evaluation.timings["timeout_seconds"] == 0.05


def test_with_capped_metric_timeout_override_supersedes_row() -> None:
    row = {METRIC_TIMEOUT_FIELD: "1800", "output_path": "x"}
    effective = _with_capped_metric_timeout(row, remaining_run_seconds=10_000, override_seconds=120)
    assert effective[METRIC_TIMEOUT_FIELD] == "120"


def test_with_capped_metric_timeout_caps_to_remaining_walltime() -> None:
    row = {METRIC_TIMEOUT_FIELD: "1800", "output_path": "x"}
    effective = _with_capped_metric_timeout(row, remaining_run_seconds=300)
    assert effective[METRIC_TIMEOUT_FIELD] == "300"


def test_with_capped_metric_timeout_keeps_smaller_row_value() -> None:
    row = {METRIC_TIMEOUT_FIELD: "60", "output_path": "x"}
    effective = _with_capped_metric_timeout(row, remaining_run_seconds=1800)
    assert effective[METRIC_TIMEOUT_FIELD] == "60"


def test_with_capped_metric_timeout_absent_field_is_untouched() -> None:
    row = {"output_path": "x"}
    effective = _with_capped_metric_timeout(row, remaining_run_seconds=300)
    assert METRIC_TIMEOUT_FIELD not in effective
