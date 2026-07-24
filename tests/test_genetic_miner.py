from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from process_discovery_cash.discovery import genetic as genetic_module
from process_discovery_cash.discovery.genetic import (
    GeneticMiner,
    GeneticMinerTimeout,
    extract_genetic_miner_params,
    extract_timeout_seconds,
    prepare_genetic_miner_log,
)
from process_discovery_cash.discovery.pm4py_backend import UnsupportedAlgorithmError
from process_discovery_cash.discovery.registry import registered_algorithm_names
from process_discovery_cash.experiments import runner as runner_module
from process_discovery_cash.experiments.runner import run_manifest_index


def test_genetic_miner_is_registered() -> None:
    assert "genetic_miner" in registered_algorithm_names()


def test_genetic_miner_returns_unsupported_when_backend_is_unavailable(monkeypatch) -> None:
    def unavailable_backend(_event_log, _params):
        raise UnsupportedAlgorithmError(
            "Genetic Miner is not available in this pm4py installation: import failed"
        )

    monkeypatch.setattr(genetic_module, "discover_genetic_miner", unavailable_backend)

    result = GeneticMiner().discover(_tiny_list_log(), {"population_size": 25})

    assert result.status == "unsupported"
    assert "not available in this pm4py installation" in (result.error_message or "")


def test_genetic_miner_config_merges_dict_fields_without_dictionary_update_error(
    monkeypatch,
) -> None:
    captured_params = {}

    def successful_backend(event_log, params):
        assert hasattr(event_log, "columns")
        assert pd.api.types.is_datetime64_any_dtype(event_log["time:timestamp"])
        captured_params.update(params)
        return {"model": (object(), object(), object()), "model_type": "petri_net"}

    monkeypatch.setattr(genetic_module, "discover_genetic_miner", successful_backend)
    config = {
        "algorithm": "genetic_miner",
        "backend": "pm4py",
        "default_params": {"population_size": 25, "generations": 50},
        "hyperparameters": {"generations": 100, "mutation_rate": 0.1},
        "params": {"crossover_rate": 0.8},
    }

    result = GeneticMiner().discover(_tiny_list_log(), config)

    assert result.status == "success"
    assert captured_params == {
        "population_size": 25,
        "generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
    }
    assert "dictionary update sequence" not in (result.error_message or "")


def test_prepare_genetic_miner_log_converts_timestamp_to_pandas_datetime() -> None:
    log = [
        [
            {
                "concept:name": "A",
                "time:timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "concept:name": "B",
                "time:timestamp": "2026-01-01T00:01:00+00:00",
            },
        ]
    ]

    dataframe = prepare_genetic_miner_log(log)

    assert pd.api.types.is_datetime64_any_dtype(dataframe["time:timestamp"])
    assert {"case:concept:name", "concept:name", "time:timestamp"}.issubset(dataframe.columns)


def test_genetic_miner_malformed_config_returns_structured_failure() -> None:
    result = GeneticMiner().discover([], {"hyperparameters": "genetic_miner"})

    assert result.status == "failed"
    assert (
        result.error_message
        == "Invalid Genetic Miner config: expected 'hyperparameters' to be a dict, got str"
    )


def test_extract_genetic_miner_params_rejects_non_dict_config() -> None:
    try:
        extract_genetic_miner_params("genetic_miner")
    except TypeError as exc:
        assert "expected config to be a dict, got str" in str(exc)
    else:
        raise AssertionError("Expected non-dict config to fail")


def test_genetic_miner_timeout_returns_structured_timeout(monkeypatch) -> None:
    def timeout_backend(_event_log, _params):
        raise GeneticMinerTimeout("Genetic Miner timed out after 1 seconds")

    monkeypatch.setattr(genetic_module, "discover_genetic_miner", timeout_backend)

    result = GeneticMiner().discover(_tiny_list_log(), {"discovery_timeout_seconds": 1})

    assert result.status == "timeout"
    assert result.error_message == "Genetic Miner timed out after 1 seconds"


def test_genetic_miner_timeout_config_is_not_passed_to_pm4py_params() -> None:
    assert extract_timeout_seconds({"discovery_timeout_seconds": "5"}) == 5
    assert extract_genetic_miner_params(
        {"population_size": 25, "discovery_timeout_seconds": 5}
    ) == {"population_size": 25}


def test_genetic_miner_rejects_removed_timeout_alias() -> None:
    with pytest.raises(TypeError, match="use 'discovery_timeout_seconds'"):
        extract_timeout_seconds({"timeout_seconds": "5"})


def test_runner_writes_unsupported_genetic_result_without_metric_evaluation(
    tmp_path,
    monkeypatch,
) -> None:
    def unavailable_backend(_event_log, _params):
        raise UnsupportedAlgorithmError(
            "Genetic Miner is not available in this pm4py installation: import failed"
        )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Metrics should not be evaluated for unsupported discovery")

    monkeypatch.setattr(genetic_module, "discover_genetic_miner", unavailable_backend)
    monkeypatch.setattr(runner_module, "evaluate_discovery_result", fail_if_called)

    output_path = tmp_path / "genetic_result.json"
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "log_id",
                "log_path",
                "seed",
                "algorithm",
                "backend",
                "params_json",
                "config_hash",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "test",
                "log_id": "tiny",
                "log_path": "data/example/tiny_log.xes",
                "seed": "0",
                "algorithm": "genetic_miner",
                "backend": "pm4py",
                "params_json": json.dumps({"population_size": 25}),
                "config_hash": "abc123",
                "output_path": output_path.as_posix(),
            }
        )

    written_path = run_manifest_index(manifest_path, 0)
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert payload["status"] == "unsupported"
    assert payload["metrics"] == {
        "fitness": None,
        "precision": None,
        "generalization": None,
        "simplicity": None,
    }
    assert all(
        metric_status["status"] == "not_computed"
        for metric_status in payload["metric_statuses"].values()
    )


def _tiny_list_log() -> list[list[dict[str, str]]]:
    return [
        [
            {
                "concept:name": "A",
                "time:timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "concept:name": "B",
                "time:timestamp": "2026-01-01T00:01:00+00:00",
            },
        ]
    ]
