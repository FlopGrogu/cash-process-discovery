from __future__ import annotations

import pandas as pd
import pytest

from process_discovery_cash.data.loading import ensure_pm4py_event_log, load_event_log
from process_discovery_cash.discovery.alpha import AlphaMiner
from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.evaluation.quality_metrics import evaluate_discovery_result


def test_pm4py_event_log_can_be_evaluated_without_conversion_errors() -> None:
    event_log = _tiny_event_log()
    discovery_result = AlphaMiner().discover(event_log, {"variant": "classic"})

    metrics, statuses = evaluate_discovery_result(discovery_result, event_log)

    assert discovery_result.status == "success"
    assert statuses["fitness"]["status"] == "success"
    assert statuses["precision"]["status"] == "success"
    assert metrics["fitness"] is not None
    assert metrics["precision"] is not None


def test_pandas_dataframe_log_is_converted_to_event_log() -> None:
    from pm4py.objects.log.obj import EventLog

    dataframe = pd.DataFrame(
        [
            {
                "case:concept:name": "case_1",
                "concept:name": "A",
                "time:timestamp": pd.Timestamp("2026-01-01T00:00:00"),
            },
            {
                "case:concept:name": "case_1",
                "concept:name": "B",
                "time:timestamp": pd.Timestamp("2026-01-01T00:01:00"),
            },
        ]
    )

    converted = ensure_pm4py_event_log(dataframe)

    assert isinstance(converted, EventLog)
    assert hasattr(converted, "attributes")
    assert len(converted) == 1


def test_fitness_and_precision_receive_event_log_not_plain_list(monkeypatch) -> None:
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
    from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
    from pm4py.objects.log.obj import EventLog

    def fake_fitness_apply(event_log, *_args, **_kwargs):
        assert isinstance(event_log, EventLog)
        return {"log_fitness": 1.0}

    def fake_precision_apply(event_log, *_args, **_kwargs):
        assert isinstance(event_log, EventLog)
        return 0.75

    monkeypatch.setattr(replay_fitness, "apply", fake_fitness_apply)
    monkeypatch.setattr(precision_evaluator, "apply", fake_precision_apply)
    discovery_result = DiscoveryResult(
        algorithm_name="alpha_miner",
        backend_name="pm4py",
        hyperparameters={"variant": "classic"},
        runtime_seconds=0.0,
        status="success",
        model_type="petri_net",
        discovered_model=(object(), object(), object()),
    )
    list_log = [
        [
            {"concept:name": "A", "time:timestamp": "2026-01-01T00:00:00+00:00"},
            {"concept:name": "B", "time:timestamp": "2026-01-01T00:01:00+00:00"},
        ]
    ]

    metrics, statuses = evaluate_discovery_result(
        discovery_result,
        list_log,
        metric_names=["fitness", "precision"],
    )

    assert metrics["fitness"] == 1.0
    assert metrics["precision"] == 0.75
    assert statuses["fitness"]["status"] == "success"
    assert statuses["precision"]["status"] == "success"


@pytest.mark.parametrize(
    ("profile", "expected_fitness_variant", "expected_precision_variant"),
    [
        ("token", "TOKEN_BASED", "ETCONFORMANCE_TOKEN"),
        ("alignment", "ALIGNMENT_BASED", "ALIGN_ETCONFORMANCE"),
    ],
)
def test_metric_profiles_select_pm4py_variants(
    monkeypatch,
    profile: str,
    expected_fitness_variant: str,
    expected_precision_variant: str,
) -> None:
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
    from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

    captured = {}

    def fake_fitness_apply(
        _event_log,
        _net,
        _initial_marking,
        _final_marking,
        parameters=None,
        variant=None,
        align_variant=None,
    ):
        captured["fitness_variant"] = variant
        return {"log_fitness": 1.0}

    def fake_precision_apply(
        _event_log,
        _net,
        _initial_marking,
        _final_marking,
        parameters=None,
        variant=None,
    ):
        captured["precision_variant"] = variant
        return 0.75

    monkeypatch.setattr(replay_fitness, "apply", fake_fitness_apply)
    monkeypatch.setattr(precision_evaluator, "apply", fake_precision_apply)
    discovery_result = DiscoveryResult(
        algorithm_name="alpha_miner",
        backend_name="pm4py",
        hyperparameters={"variant": "classic"},
        runtime_seconds=0.0,
        status="success",
        model_type="petri_net",
        discovered_model=(object(), object(), object()),
    )

    metrics, statuses, timings = evaluate_discovery_result(
        discovery_result,
        _tiny_event_log(),
        metric_names=["fitness", "precision"],
        metric_profile=profile,
        include_timings=True,
    )

    assert captured["fitness_variant"] == getattr(
        replay_fitness.Variants,
        expected_fitness_variant,
    )
    assert captured["precision_variant"] == getattr(
        precision_evaluator.Variants,
        expected_precision_variant,
    )
    assert metrics["fitness"] == 1.0
    assert metrics["precision"] == 0.75
    assert statuses["fitness"]["status"] == "success"
    assert statuses["precision"]["status"] == "success"
    assert timings["profile"] == profile


def test_alpha_miner_tiny_xes_has_structured_fitness_and_precision_statuses() -> None:
    event_log = load_event_log("data/example/tiny_log.xes")
    discovery_result = AlphaMiner().discover(event_log, {"variant": "classic"})

    metrics, statuses = evaluate_discovery_result(discovery_result, event_log)

    assert discovery_result.status == "success"
    assert statuses["fitness"]["status"] == "success"
    assert statuses["precision"]["status"] == "success"
    assert statuses["fitness"]["error"] is None
    assert statuses["precision"]["error"] is None
    assert metrics["fitness"] is not None
    assert metrics["precision"] is not None


def test_default_metrics_do_not_include_composite_score() -> None:
    event_log = _tiny_event_log()
    discovery_result = AlphaMiner().discover(event_log, {"variant": "classic"})

    metrics, statuses = evaluate_discovery_result(discovery_result, event_log)

    assert set(metrics) == {"fitness", "precision", "generalization", "simplicity"}
    assert "composite_score" not in statuses


def test_metric_evaluation_can_report_timings() -> None:
    event_log = _tiny_event_log()
    discovery_result = AlphaMiner().discover(event_log, {"variant": "classic"})

    metrics, statuses, timings = evaluate_discovery_result(
        discovery_result,
        event_log,
        include_timings=True,
    )

    assert statuses["fitness"]["status"] == "success"
    assert metrics["fitness"] is not None
    assert timings["model_conversion_seconds"] >= 0
    assert timings["log_conversion_seconds"] >= 0
    assert timings["metric_seconds"]["fitness"] >= 0
    assert timings["total_seconds"] >= 0


def test_metric_failure_does_not_add_composite_score(monkeypatch) -> None:
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

    def failing_precision(*_args, **_kwargs):
        raise RuntimeError("forced precision failure")

    monkeypatch.setattr(precision_evaluator, "apply", failing_precision)
    event_log = _tiny_event_log()
    discovery_result = AlphaMiner().discover(event_log, {"variant": "classic"})

    metrics, statuses = evaluate_discovery_result(discovery_result, event_log)

    assert metrics["precision"] is None
    assert statuses["precision"]["status"] == "backend_error"
    assert "metric=precision" in statuses["precision"]["error"]
    assert "log_type_before=" in statuses["precision"]["error"]
    assert "log_type_after=pm4py.objects.log.obj.EventLog" in statuses["precision"]["error"]
    assert "discovered_model_object_type=builtins.tuple" in statuses["precision"]["error"]
    assert "composite_score" not in metrics
    assert "composite_score" not in statuses


def _tiny_event_log():
    from pm4py.objects.log.obj import Event, EventLog, Trace

    event_log = EventLog()
    for case_id, activities in [
        ("case_1", ["A", "B", "C"]),
        ("case_2", ["A", "B", "C"]),
        ("case_3", ["A", "D", "C"]),
    ]:
        trace = Trace(attributes={"concept:name": case_id})
        for index, activity in enumerate(activities):
            trace.append(
                Event(
                    {
                        "concept:name": activity,
                        "time:timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=index),
                    }
                )
            )
        event_log.append(trace)
    return event_log
