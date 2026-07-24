from __future__ import annotations

import csv
import json

from process_discovery_cash.experiments.aggregation import aggregate_results


def test_aggregate_results_flattens_timing_metadata(tmp_path) -> None:
    result_path = tmp_path / "results" / "row.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp",
                "log_id": "log",
                "log_path": "data/raw/log.xes",
                "test_log_path": "data/raw/log.xes",
                "seed": 0,
                "algorithm_name": "alpha_miner",
                "backend": "pm4py",
                "hyperparameters": {"variant": "classic"},
                "discovered_model_type": "petri_net",
                "model_path": None,
                "metrics": {"fitness": 1.0},
                "metric_statuses": {"fitness": {"status": "success", "value": 1.0, "error": None}},
                "runtime_seconds": 1.0,
                "status": "success",
                "error_message": None,
                "warnings": [],
                "metadata": {
                    "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    "config_hash": "abc",
                    "timings": {
                        "load_train_log_seconds": 0.1,
                        "metrics": {"metric_seconds": {"fitness": 0.2}},
                    },
                    "memory": {
                        "peak_rss_bytes": 123456789,
                        "peak_rss_mb": 117.74,
                    },
                    "slurm": {
                        "job_id": "12345",
                        "requested_memory": "32G",
                        "requested_memory_bytes": 34359738368,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = aggregate_results(tmp_path / "results", tmp_path / "aggregated.csv")

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert rows[0]["status"] == "success"
    assert rows[0]["runtime_seconds"] == "1.0"
    assert rows[0]["error_message"] == ""
    assert rows[0]["timing_load_train_log_seconds"] == "0.1"
    assert rows[0]["timing_metrics_metric_seconds_fitness"] == "0.2"
    assert rows[0]["memory_peak_rss_bytes"] == "123456789"
    assert rows[0]["memory_peak_rss_mb"] == "117.74"
    assert rows[0]["slurm_job_id"] == "12345"
    assert rows[0]["slurm_requested_memory"] == "32G"
    assert rows[0]["slurm_requested_memory_bytes"] == "34359738368"


def test_aggregate_results_handles_saved_model_metric_outputs(tmp_path) -> None:
    result_path = tmp_path / "metrics" / "row.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "status": "success_missing",
                "discovery_status": "success",
                "metric_runtime_seconds": 0.25,
                "discovery_runtime_seconds": 1.5,
                "source_result_path": "results/source.json",
                "source_config_hash": "abc",
                "config_hash": "abc",
                "experiment_id": "exp",
                "log_id": "log",
                "log_path": "data/raw/log.xes",
                "seed": 0,
                "algorithm_name": "alpha_miner",
                "backend": "pm4py",
                "hyperparameters": {"variant": "classic"},
                "model_path": "results/source/discovered_model.pnml",
                "discovered_model_type": "petri_net",
                "test_log_path": "data/raw/log.xes",
                "metric_profile": "alignment",
                "metrics": {"fitness": 1.0},
                "metric_statuses": {"fitness": {"status": "success", "value": 1.0, "error": None}},
                "metadata": {
                    "timings": {
                        "load_model_seconds": 0.1,
                        "metrics": {"profile": "alignment"},
                    }
                },
                "source_metadata": {"timestamp_utc": "2026-01-01T00:00:00+00:00"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = aggregate_results(tmp_path / "metrics", tmp_path / "aggregated.csv")

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert rows[0]["status"] == "success"
    assert rows[0]["metric_run_status"] == "success_missing"
    assert rows[0]["runtime_seconds"] == "1.5"
    assert rows[0]["metric_runtime_seconds"] == "0.25"
    assert rows[0]["config_hash"] == "abc"
    assert rows[0]["metric_profile"] == "alignment"
    assert rows[0]["log_path"] == "data/raw/log.xes"
    assert rows[0]["seed"] == "0"
    assert rows[0]["backend"] == "pm4py"
    assert rows[0]["param_variant"] == "classic"
    assert rows[0]["source_result_path"] == "results/source.json"
    assert rows[0]["source_config_hash"] == "abc"
    assert rows[0]["discovered_model_type"] == "petri_net"
    assert rows[0]["metric_fitness"] == "1.0"
    assert rows[0]["timing_load_model_seconds"] == "0.1"
    assert rows[0]["source_timestamp_utc"] == "2026-01-01T00:00:00+00:00"
