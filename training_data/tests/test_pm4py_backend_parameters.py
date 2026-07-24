from __future__ import annotations

import pandas as pd

from process_discovery_cash.discovery import pm4py_backend
from process_discovery_cash.discovery.pm4py_backend import (
    discover_alpha_miner,
    discover_genetic_miner,
    discover_heuristics_miner,
    discover_ilp_miner,
    discover_inductive_miner,
)


def test_alpha_parameters_are_passed_to_lower_level_pm4py_call(monkeypatch) -> None:
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner

    captured = _capture_apply(monkeypatch, alpha_miner, ("net", "im", "fm"))

    payload = discover_alpha_miner(
        [],
        {
            "variant": "classic",
            "activity_key": "activity",
            "timestamp_key": "timestamp",
            "case_id_key": "case_id",
        },
    )

    assert captured["variant"] == alpha_miner.Variants.ALPHA_VERSION_CLASSIC
    assert captured["parameters"] == {
        alpha_miner.Parameters.ACTIVITY_KEY: "activity",
        alpha_miner.Parameters.TIMESTAMP_KEY: "timestamp",
        alpha_miner.Parameters.CASE_ID_KEY: "case_id",
    }
    assert payload["metadata"]["backend_function"] == ("pm4py.algo.discovery.alpha.algorithm.apply")
    assert payload["metadata"]["ignored_parameters"] == {}


def test_alpha_classic_passes_dataframe_directly_to_pm4py(monkeypatch) -> None:
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner

    dataframe = pd.DataFrame(
        {
            "case:concept:name": ["case_1"],
            "concept:name": ["A"],
            "time:timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        }
    )
    captured = _capture_apply(monkeypatch, alpha_miner, ("net", "im", "fm"))

    discover_alpha_miner(dataframe, {"variant": "classic"})

    assert captured["event_log"] is dataframe


def test_inductive_variant_and_parameters_are_passed_to_lower_level_api(
    monkeypatch,
) -> None:
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner

    captured = _capture_apply(monkeypatch, inductive_miner, object())
    _capture_process_tree_conversion(monkeypatch)

    payload = discover_inductive_miner(
        [],
        {
            "variant": "imf",
            "noise_threshold": 0.2,
            "multi_processing": False,
            "disable_fallthroughs": True,
            "activity_key": "activity",
            "timestamp_key": "timestamp",
            "case_id_key": "case_id",
            "discovery_timeout_seconds": 1800,
        },
    )

    assert captured["variant"] == inductive_miner.Variants.IMf
    assert captured["parameters"] == {
        inductive_miner.Parameters.ACTIVITY_KEY: "activity",
        inductive_miner.Parameters.TIMESTAMP_KEY: "timestamp",
        inductive_miner.Parameters.CASE_ID_KEY: "case_id",
        "noise_threshold": 0.2,
        "multiprocessing": False,
        "disable_fallthroughs": True,
    }
    assert payload["metadata"]["backend_function"] == (
        "pm4py.algo.discovery.inductive.algorithm.apply"
    )
    assert payload["metadata"]["backend_parameters"] == {
        "pm4py:param:activity_key": "activity",
        "pm4py:param:timestamp_key": "timestamp",
        "pm4py:param:case_id_key": "case_id",
        "noise_threshold": 0.2,
        "multiprocessing": False,
        "disable_fallthroughs": True,
    }
    assert payload["metadata"]["converted_to_petri_net"] is True
    assert payload["metadata"]["ignored_parameters"] == {"discovery_timeout_seconds": 1800}
    assert "discovery_timeout_seconds" not in payload["metadata"]["backend_parameters"]


def test_heuristic_classic_parameters_map_to_pm4py_enum_names(monkeypatch) -> None:
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

    captured = _capture_apply(monkeypatch, heuristics_miner, ("net", "im", "fm"))

    payload = discover_heuristics_miner(
        [],
        {
            "variant": "classic",
            "dependency_threshold": 0.7,
            "and_threshold": 0.8,
            "loop_two_threshold": 0.5,
            "dfg_pre_cleaning_noise_thresh": 0.1,
            "min_act_count": 2,
            "min_dfg_occurrences": 3,
        },
    )

    assert captured["variant"] == heuristics_miner.Variants.CLASSIC
    assert payload["metadata"]["backend_parameters"] == {
        "dependency_thresh": 0.7,
        "and_measure_thresh": 0.8,
        "loop_length_two_thresh": 0.5,
        "dfg_pre_cleaning_noise_thresh": 0.1,
        "min_act_count": 2,
        "min_dfg_occurrences": 3,
    }
    assert payload["metadata"]["ignored_parameters"] == {}


def test_heuristic_plusplus_variant_is_executed(monkeypatch) -> None:
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

    captured = _capture_apply(monkeypatch, heuristics_miner, ("net", "im", "fm"))

    payload = discover_heuristics_miner(
        [],
        {
            "variant": "plusplus",
            "dependency_threshold": 0.7,
            "and_threshold": 0.8,
            "loop_two_threshold": 0.5,
            "dfg_pre_cleaning_noise_thresh": 0.1,
            "min_act_count": 2,
            "min_dfg_occurrences": 3,
        },
    )

    assert captured["variant"] == heuristics_miner.Variants.PLUSPLUS
    assert payload["metadata"]["resolved_variant"] == "PLUSPLUS"
    assert payload["metadata"]["ignored_parameters"] == {}


def test_genetic_parameters_pass_through_and_timeout_is_runtime_only(monkeypatch) -> None:
    from pm4py.algo.discovery.genetic import algorithm as genetic_miner

    captured = _capture_apply(monkeypatch, genetic_miner, ("net", "im", "fm"))

    payload = discover_genetic_miner(
        [],
        {
            "population_size": 25,
            "generations": 2,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elitism_rate": 0.2,
            "elitism_min_sample": 3,
            "log_csv": None,
            "discovery_timeout_seconds": 5,
        },
    )

    assert captured["variant"] == genetic_miner.Variants.CLASSIC
    assert captured["parameters"] == {
        genetic_miner.Parameters.POPULATION_SIZE: 25,
        genetic_miner.Parameters.GENERATIONS: 2,
        genetic_miner.Parameters.MUTATION_RATE: 0.1,
        genetic_miner.Parameters.CROSSOVER_RATE: 0.8,
        genetic_miner.Parameters.ELITISM_RATE: 0.2,
        genetic_miner.Parameters.ELITISM_MIN_SAMPLE: 3,
        genetic_miner.Parameters.LOG_CSV: None,
    }
    assert "discovery_timeout_seconds" not in payload["metadata"]["backend_parameters"]
    assert payload["metadata"]["runtime_parameters"] == {"discovery_timeout_seconds": 5}
    assert payload["metadata"]["ignored_parameters"] == {"discovery_timeout_seconds": 5}


def test_ilp_alpha_is_passed_and_unknown_parameters_are_audited(monkeypatch) -> None:
    from pm4py.algo.discovery.ilp import algorithm as ilp_miner

    captured = _capture_apply(monkeypatch, ilp_miner, ("net", "im", "fm"))

    payload = discover_ilp_miner(
        [],
        {
            "alpha": 0.5,
            "custom_param": 123,
        },
    )

    assert captured["variant"] == ilp_miner.Variants.CLASSIC
    assert captured["parameters"] == {ilp_miner.Variants.CLASSIC.value.Parameters.ALPHA: 0.5}
    assert payload["metadata"]["backend_parameters"] == {"alpha": 0.5}
    assert payload["metadata"]["ignored_parameters"] == {"custom_param": 123}
    assert len(payload["metadata"]["warnings"]) == 1


def _capture_discovery_function(monkeypatch, expected_function_name: str) -> dict:
    captured = {}

    def fake_discovery_function(event_log, **kwargs):
        captured["event_log"] = event_log
        captured["kwargs"] = kwargs
        return ("net", "im", "fm")

    def fake_loader(function_name: str, algorithm: str):
        captured["function_name"] = function_name
        captured["algorithm"] = algorithm
        assert function_name == expected_function_name
        return fake_discovery_function

    monkeypatch.setattr(pm4py_backend, "_pm4py_discovery_function", fake_loader)
    monkeypatch.setattr(pm4py_backend, "ensure_pm4py_event_log", lambda event_log: event_log)
    return captured


def _capture_apply(monkeypatch, module: object, return_value: object) -> dict:
    captured = {}

    def fake_apply(event_log, *args, **kwargs):
        captured["event_log"] = event_log
        captured["args"] = args
        captured.update(kwargs)
        return return_value

    monkeypatch.setattr(module, "apply", fake_apply)
    monkeypatch.setattr(pm4py_backend, "ensure_pm4py_event_log", lambda event_log: event_log)
    return captured


def _capture_process_tree_conversion(monkeypatch) -> dict:
    captured = {}

    def fake_convert(process_tree):
        captured["process_tree"] = process_tree
        return ("net", "im", "fm")

    monkeypatch.setattr(pm4py_backend, "_convert_process_tree_to_petri_net", fake_convert)
    return captured
