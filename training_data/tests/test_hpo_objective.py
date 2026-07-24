import pytest

from process_discovery_cash.hpo.objective import (
    compute_objective,
    objective_from_result_payload,
)

EQUAL_WEIGHTS = {
    "fitness": 1.0,
    "precision": 1.0,
    "generalization": 1.0,
    "simplicity": 1.0,
}

SUCCESS_STATUSES = {
    name: {"status": "success", "value": None, "error": None} for name in EQUAL_WEIGHTS
}


def test_equal_weights_match_v6_mean_score() -> None:
    metrics = {
        "fitness": 0.9,
        "precision": 0.7,
        "generalization": 0.5,
        "simplicity": 0.3,
    }

    value = compute_objective(metrics, SUCCESS_STATUSES, EQUAL_WEIGHTS)

    assert value == pytest.approx(sum(metrics.values()) / len(metrics))


def test_custom_weights() -> None:
    metrics = {"fitness": 1.0, "precision": 0.5}
    statuses = {name: {"status": "success"} for name in metrics}

    value = compute_objective(metrics, statuses, {"fitness": 3.0, "precision": 1.0})

    assert value == pytest.approx((3.0 * 1.0 + 1.0 * 0.5) / 4.0)


def test_missing_metric_value_yields_failed_value() -> None:
    metrics = {"fitness": 0.9, "precision": None}
    statuses = {name: {"status": "success"} for name in metrics}

    assert compute_objective(metrics, statuses, {"fitness": 1.0, "precision": 1.0}) == 0.0


def test_non_numeric_metric_value_yields_failed_value() -> None:
    metrics = {"fitness": "0.9"}

    assert compute_objective(metrics, {}, {"fitness": 1.0}) == 0.0


def test_non_success_metric_status_yields_failed_value() -> None:
    metrics = {"fitness": 0.9}
    statuses = {"fitness": {"status": "backend_error"}}

    assert compute_objective(metrics, statuses, {"fitness": 1.0}, failed_value=-1.0) == -1.0


def test_unweighted_metrics_are_ignored() -> None:
    metrics = {"fitness": 0.8, "precision": None}
    statuses = {"fitness": {"status": "success"}, "precision": {"status": "backend_error"}}

    assert compute_objective(metrics, statuses, {"fitness": 1.0}) == pytest.approx(0.8)


def test_missing_status_record_is_tolerated() -> None:
    metrics = {"fitness": 0.8}

    assert compute_objective(metrics, {}, {"fitness": 1.0}) == pytest.approx(0.8)


def test_zero_total_weight_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_objective({"fitness": 0.5}, {}, {"fitness": 0.0})


def test_failed_payload_yields_failed_value() -> None:
    payload = {"status": "timeout", "metrics": {}, "metric_statuses": {}}

    outcome = objective_from_result_payload(payload, EQUAL_WEIGHTS)

    assert outcome.value == 0.0
    assert outcome.run_status == "timeout"
    assert outcome.metric_values == {name: None for name in EQUAL_WEIGHTS}


def test_success_payload_computes_weighted_mean() -> None:
    metrics = {
        "fitness": 1.0,
        "precision": 0.5,
        "generalization": 0.5,
        "simplicity": 1.0,
    }
    payload = {"status": "success", "metrics": metrics, "metric_statuses": SUCCESS_STATUSES}

    outcome = objective_from_result_payload(payload, EQUAL_WEIGHTS)

    assert outcome.value == pytest.approx(0.75)
    assert outcome.run_status == "success"
    assert outcome.metric_values == metrics
