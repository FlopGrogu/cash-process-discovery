from __future__ import annotations

import sys

from process_discovery_cash.discovery.heuristic import HeuristicsMiner
from process_discovery_cash.experiments.identity import (
    execution_config,
    semantic_algorithm_parameters,
)


def test_heuristics_wrapper_records_recursion_metadata_on_plusplus_success(
    monkeypatch,
) -> None:
    algorithm = HeuristicsMiner()
    captured: dict[str, int] = {}
    original_limit = sys.getrecursionlimit()

    def fake_discover(_train_log, _config):
        captured["during_call_limit"] = sys.getrecursionlimit()
        return {
            "model": ("net", "im", "fm"),
            "model_type": "petri_net",
            "metadata": {"variant": "plusplus", "requested_variant": "plusplus", "warnings": []},
        }

    monkeypatch.setattr(
        "process_discovery_cash.discovery.heuristic.discover_heuristics_miner",
        fake_discover,
    )
    monkeypatch.setattr(
        "process_discovery_cash.discovery.heuristic._pm4py_version",
        lambda: "2.7.22.2",
    )

    result = algorithm.discover([], {"variant": "plusplus", "recursion_limit": 4321})

    assert result.status == "success"
    assert result.metadata["variant"] == "plusplus"
    assert result.metadata["pm4py_version"] == "2.7.22.2"
    assert result.metadata["recursion_limit_used"] == 4321
    assert result.metadata["previous_recursion_limit"] == original_limit
    assert captured["during_call_limit"] == 4321
    assert sys.getrecursionlimit() == original_limit


def test_heuristics_wrapper_records_recursion_metadata_on_classic_success(monkeypatch) -> None:
    algorithm = HeuristicsMiner()
    captured: dict[str, int] = {}
    original_limit = sys.getrecursionlimit()

    def fake_discover(_train_log, _config):
        captured["during_call_limit"] = sys.getrecursionlimit()
        return {
            "model": ("net", "im", "fm"),
            "model_type": "petri_net",
            "metadata": {"variant": "classic", "requested_variant": "classic", "warnings": []},
        }

    monkeypatch.setattr(
        "process_discovery_cash.discovery.heuristic.discover_heuristics_miner",
        fake_discover,
    )

    result = algorithm.discover([], {"variant": "classic", "recursion_limit": 4321})

    assert result.status == "success"
    assert result.metadata["recursion_limit_used"] == 4321
    assert result.metadata["previous_recursion_limit"] == original_limit
    assert captured["during_call_limit"] == 4321
    assert sys.getrecursionlimit() == original_limit


def test_heuristics_wrapper_returns_failed_result_for_recursion_error(monkeypatch) -> None:
    algorithm = HeuristicsMiner()
    original_limit = sys.getrecursionlimit()

    def fake_discover(_train_log, _config):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(
        "process_discovery_cash.discovery.heuristic.discover_heuristics_miner",
        fake_discover,
    )
    monkeypatch.setattr(
        "process_discovery_cash.discovery.heuristic._pm4py_version",
        lambda: "2.7.22.2",
    )

    result = algorithm.discover(
        [],
        {
            "variant": "plusplus",
            "dependency_threshold": 0.5,
            "recursion_limit": 3456,
            "log_id": "road_traffic_fines",
            "input_log_path": "data/raw/road_traffic_fines.xes.gz",
        },
    )

    assert result.status == "failed"
    assert result.error_message == "RecursionError: maximum recursion depth exceeded"
    assert result.metadata["error_type"] == "RecursionError"
    assert result.metadata["log_id"] == "road_traffic_fines"
    assert result.metadata["input_log_path"] == "data/raw/road_traffic_fines.xes.gz"
    assert result.metadata["pm4py_version"] == "2.7.22.2"
    assert result.metadata["recursion_limit_used"] == 3456
    assert result.metadata["previous_recursion_limit"] == original_limit
    assert result.metadata["hyperparameters"]["dependency_threshold"] == 0.5
    assert "RecursionError" in result.metadata["traceback"]
    assert sys.getrecursionlimit() == original_limit


def test_heuristics_execution_identity_excludes_recursion_limit() -> None:
    params = {"variant": "plusplus", "dependency_threshold": 0.2, "recursion_limit": 10000}

    assert semantic_algorithm_parameters(params) == {
        "variant": "plusplus",
        "dependency_threshold": 0.2,
    }
    assert execution_config(params) == {"recursion_limit": 10000}


def test_heuristics_wrapper_rejects_non_positive_recursion_limit() -> None:
    algorithm = HeuristicsMiner()

    result = algorithm.discover([], {"variant": "classic", "recursion_limit": 0})

    assert result.status == "failed"
    assert result.metadata["error_type"] == "ValueError"
    assert "recursion_limit must be a positive integer" in result.error_message
