from __future__ import annotations

import sys

from process_discovery_cash.discovery.inductive import InductiveMiner
from process_discovery_cash.experiments.identity import (
    execution_config,
    semantic_algorithm_parameters,
)


def test_inductive_wrapper_records_recursion_metadata_on_success(monkeypatch) -> None:
    algorithm = InductiveMiner()
    captured: dict[str, int] = {}
    original_limit = sys.getrecursionlimit()

    def fake_discover(_train_log, _config):
        captured["during_call_limit"] = sys.getrecursionlimit()
        return {
            "model": ("net", "im", "fm"),
            "model_type": "petri_net",
            "metadata": {"variant": "im", "requested_variant": "im", "warnings": []},
        }

    monkeypatch.setattr(
        "process_discovery_cash.discovery.inductive.discover_inductive_miner",
        fake_discover,
    )
    monkeypatch.setattr(
        "process_discovery_cash.discovery.inductive._pm4py_version",
        lambda: "2.7.22.2",
    )

    result = algorithm.discover([], {"variant": "im", "recursion_limit": 4321})

    assert result.status == "success"
    assert result.metadata["variant"] == "im"
    assert result.metadata["pm4py_version"] == "2.7.22.2"
    assert result.metadata["recursion_limit_used"] == 4321
    assert result.metadata["previous_recursion_limit"] == original_limit
    assert captured["during_call_limit"] == 4321
    assert sys.getrecursionlimit() == original_limit


def test_inductive_wrapper_returns_failed_result_for_recursion_error(monkeypatch) -> None:
    algorithm = InductiveMiner()
    original_limit = sys.getrecursionlimit()

    def fake_discover(_train_log, _config):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(
        "process_discovery_cash.discovery.inductive.discover_inductive_miner",
        fake_discover,
    )
    monkeypatch.setattr(
        "process_discovery_cash.discovery.inductive._pm4py_version",
        lambda: "2.7.22.2",
    )

    result = algorithm.discover(
        [],
        {
            "variant": "imf",
            "noise_threshold": 0.2,
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
    assert result.metadata["hyperparameters"]["noise_threshold"] == 0.2
    assert "RecursionError" in result.metadata["traceback"]
    assert sys.getrecursionlimit() == original_limit


def test_inductive_wrapper_returns_failed_result_for_generic_exception(monkeypatch) -> None:
    algorithm = InductiveMiner()
    original_limit = sys.getrecursionlimit()

    def fake_discover(_train_log, _config):
        raise RuntimeError("pm4py exploded")

    monkeypatch.setattr(
        "process_discovery_cash.discovery.inductive.discover_inductive_miner",
        fake_discover,
    )

    result = algorithm.discover([], {"variant": "imd", "recursion_limit": 2222})

    assert result.status == "failed"
    assert result.error_message == "RuntimeError: pm4py exploded"
    assert result.metadata["error_type"] == "RuntimeError"
    assert result.metadata["recursion_limit_used"] == 2222
    assert result.metadata["previous_recursion_limit"] == original_limit
    assert sys.getrecursionlimit() == original_limit


def test_execution_identity_excludes_recursion_limit() -> None:
    params = {"variant": "imf", "noise_threshold": 0.2, "recursion_limit": 10000}

    assert semantic_algorithm_parameters(params) == {
        "variant": "imf",
        "noise_threshold": 0.2,
    }
    assert execution_config(params) == {"recursion_limit": 10000}


def test_inductive_wrapper_rejects_non_positive_recursion_limit() -> None:
    algorithm = InductiveMiner()

    result = algorithm.discover([], {"variant": "im", "recursion_limit": 0})

    assert result.status == "failed"
    assert result.metadata["error_type"] == "ValueError"
    assert "recursion_limit must be a positive integer" in result.error_message
