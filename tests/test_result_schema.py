from process_discovery_cash.experiments.result_schema import (
    ExperimentResult,
    MetricRecord,
    MetricResult,
)


def _base_result(status: str) -> dict:
    return {
        "experiment_id": "exp",
        "log_id": "log",
        "log_path": "data/example/tiny_log.xes",
        "test_log_path": "data/example/tiny_log.xes",
        "seed": 0,
        "algorithm_name": "alpha_miner",
        "backend": "pm4py",
        "hyperparameters": {},
        "discovered_model_type": "petri_net",
        "metrics": {},
        "metric_statuses": {},
        "runtime_seconds": 0.1,
        "status": status,
    }


def test_result_schema_validates_success_failed_unsupported_and_partial_metrics() -> None:
    success = ExperimentResult(
        **{
            **_base_result("success"),
            "metrics": {"fitness": 1.0, "precision": None},
            "metric_statuses": {
                "fitness": MetricRecord(value=1.0, status="success"),
                "precision": MetricRecord(status="unsupported_model_type", error="No converter"),
            },
            "metadata": {"timestamp_utc": "2026-05-30T00:00:00+00:00"},
        }
    )
    failed = ExperimentResult(**{**_base_result("failed"), "error_message": "backend error"})
    unsupported = ExperimentResult(
        **{**_base_result("unsupported"), "error_message": "missing backend"}
    )

    assert success.metric_statuses["fitness"].status == "success"
    record = success.to_json_record()
    assert "split_id" not in record
    assert "log_features" not in record["metadata"]
    assert "split_features" not in record["metadata"]
    assert failed.status == "failed"
    assert unsupported.status == "unsupported"


def test_metric_result_schema_accepts_success_missing() -> None:
    result = MetricResult(
        experiment_id="exp",
        log_id="log",
        log_path="data/example/tiny_log.xes",
        test_log_path="data/example/tiny_log.xes",
        seed=0,
        algorithm_name="alpha_miner",
        backend="pm4py",
        hyperparameters={},
        discovered_model_type="petri_net",
        metric_profile="token",
        metric_names=["fitness"],
        metrics={"fitness": 0.0},
        metric_statuses={"fitness": MetricRecord(value=0.0, status="missing_model")},
        metric_runtime_seconds=0.2,
        discovery_runtime_seconds=0.1,
        status="success_missing",
        discovery_status="failed",
        config_hash="abc123",
    )

    assert result.status == "success_missing"


def test_metric_result_schema_keeps_discovery_and_metric_context_separate() -> None:
    result = MetricResult(
        experiment_id="exp",
        log_id="log",
        log_path="data/example/tiny_log.xes",
        test_log_path="data/example/tiny_log.xes",
        seed=0,
        algorithm_name="alpha_miner",
        backend="pm4py",
        hyperparameters={"variant": "classic"},
        discovered_model_type="petri_net",
        metric_profile="token",
        metric_names=["fitness"],
        metrics={"fitness": 1.0},
        metric_statuses={"fitness": MetricRecord(value=1.0, status="success")},
        metric_runtime_seconds=0.2,
        discovery_runtime_seconds=0.1,
        status="success",
        discovery_status="success",
        config_hash="abc123",
        metadata={"timings": {"metrics_eval_call_seconds": 0.2}},
        source_metadata={"config_hash": "abc123"},
    )

    record = result.to_json_record()
    assert record["metric_runtime_seconds"] == 0.2
    assert record["discovery_runtime_seconds"] == 0.1
    assert record["config_hash"] == "abc123"
    assert record["source_metadata"]["config_hash"] == "abc123"
