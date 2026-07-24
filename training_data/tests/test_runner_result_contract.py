import json
import time
from pathlib import Path
from typing import Any

from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.discovery.inductive import InductiveMiner
from process_discovery_cash.experiments import runner


class _CapturingAlgorithm:
    def __init__(self, captured: dict[str, Any]) -> None:
        self._captured = captured

    def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
        self._captured["train_log"] = train_log
        self._captured["config"] = dict(config)
        return DiscoveryResult(
            algorithm_name="fake_algorithm",
            backend_name="fake_backend",
            hyperparameters=config,
            runtime_seconds=0.01,
            status="success",
            model_type="unknown",
        )


class _SleepingAlgorithm:
    algorithm_name = "fake_algorithm"
    backend_name = "fake_backend"
    default_model_type = "unknown"

    def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
        time.sleep(5)
        raise AssertionError("timeout should terminate discovery before completion")


class _FastAlgorithm:
    algorithm_name = "fake_algorithm"
    backend_name = "fake_backend"
    default_model_type = "unknown"

    def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
        return DiscoveryResult(
            algorithm_name=self.algorithm_name,
            backend_name=self.backend_name,
            hyperparameters=config,
            runtime_seconds=0.001,
            status="success",
            model_type="unknown",
        )


def test_runner_uses_full_configured_train_and_test_logs(monkeypatch, tmp_path) -> None:
    captured: dict[str, Any] = {"cache_keys": []}
    row = _manifest_row(tmp_path)

    def fake_load_event_log(path: str, **kwargs) -> str:
        captured["cache_keys"].append(kwargs.get("cache_key"))
        return f"loaded:{path}"

    def fake_evaluate(
        discovery_result,
        test_log,
        *,
        metric_names,
        metric_profile,
        include_timings,
    ):
        captured["test_log"] = test_log
        assert metric_names is not None
        assert metric_profile == "pm4py_default"
        assert include_timings is True
        return (
            {"fitness": 1.0, "precision": 0.75, "generalization": 0.5, "simplicity": 0.25},
            {
                "fitness": {"status": "success", "value": 1.0, "error": None},
                "precision": {"status": "success", "value": 0.75, "error": None},
                "generalization": {"status": "success", "value": 0.5, "error": None},
                "simplicity": {"status": "success", "value": 0.25, "error": None},
            },
            {},
        )

    monkeypatch.setattr(runner, "load_event_log", fake_load_event_log)
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _CapturingAlgorithm(captured))
    monkeypatch.setattr(runner, "evaluate_discovery_result", fake_evaluate)

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert captured["train_log"] == "loaded:data/train.xes"
    assert captured["test_log"] == "loaded:data/test.xes"
    assert captured["cache_keys"] == ["tiny_log", "tiny_log_test"]
    assert captured["config"]["input_log_path"] == "data/train.xes"
    assert payload["log_path"] == "data/train.xes"
    assert payload["test_log_path"] == "data/test.xes"
    timings = payload["metadata"]["timings"]
    assert timings["load_train_log_seconds"] >= 0
    assert timings["load_test_log_seconds"] >= 0
    assert timings["discovery_call_seconds"] >= 0
    assert timings["metrics_eval_call_seconds"] >= 0
    assert timings["total_row_seconds_before_write"] >= 0
    assert timings["discovery_reported_runtime_seconds"] == 0.01
    memory = payload["metadata"]["memory"]
    assert memory["peak_rss_bytes"] >= 0
    assert memory["peak_rss_mb"] >= 0
    assert memory["source"] == "resource.getrusage"
    assert memory["scope"] == "current_process_and_waited_children"


def test_legacy_manifest_falls_back_to_xes_when_artifacts_are_missing(tmp_path) -> None:
    row = _manifest_row(tmp_path)
    row.update(
        {
            "source_log_path": "data/train.xes",
            "discovery_log_path": (tmp_path / "missing-train.parquet").as_posix(),
            "test_discovery_log_path": (tmp_path / "missing-test.parquet").as_posix(),
        }
    )

    assert runner._train_log_path(row) == "data/train.xes"
    assert runner._test_log_path(row) == "data/test.xes"


def test_legacy_manifest_uses_existing_optional_artifacts(tmp_path) -> None:
    row = _manifest_row(tmp_path)
    train_artifact = tmp_path / "train.parquet"
    test_artifact = tmp_path / "test.parquet"
    train_artifact.touch()
    test_artifact.touch()
    row.update(
        {
            "discovery_log_path": train_artifact.as_posix(),
            "test_discovery_log_path": test_artifact.as_posix(),
        }
    )

    assert runner._train_log_path(row) == train_artifact.as_posix()
    assert runner._test_log_path(row) == test_artifact.as_posix()


def test_runner_result_json_excludes_features_and_composite_score(monkeypatch, tmp_path) -> None:
    row = _manifest_row(tmp_path)

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _CapturingAlgorithm({}))
    monkeypatch.setattr(
        runner,
        "evaluate_discovery_result",
        lambda *_args, **_kwargs: (
            {"fitness": 1.0, "precision": 0.75, "generalization": 0.5, "simplicity": 0.25},
            {
                "fitness": {"status": "success", "value": 1.0, "error": None},
                "precision": {"status": "success", "value": 0.75, "error": None},
                "generalization": {"status": "success", "value": 0.5, "error": None},
                "simplicity": {"status": "success", "value": 0.25, "error": None},
            },
            {},
        ),
    )

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert set(payload["metrics"]) == {"fitness", "precision", "generalization", "simplicity"}
    assert "composite_score" not in payload["metrics"]
    assert "composite_score" not in payload["metric_statuses"]
    assert "log_features" not in payload["metadata"]
    assert "split_features" not in payload["metadata"]
    assert "split_id" not in payload


def test_runner_records_slurm_requested_memory(monkeypatch, tmp_path) -> None:
    row = _manifest_row(tmp_path)
    row["metrics_json"] = json.dumps({"enabled": False, "export_model": False})

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "minor")
    monkeypatch.setenv("SLURM_JOB_NAME", "ilp_miner")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("PDCASH_SLURM_REQUESTED_MEMORY", "32G")
    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _CapturingAlgorithm({}))

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["metadata"]["slurm"] == {
        "job_id": "12345",
        "array_job_id": "12345",
        "array_task_id": "7",
        "job_partition": "minor",
        "job_name": "ilp_miner",
        "requested_memory": "32G",
        "requested_memory_bytes": 32 * 1024**3,
        "requested_cpus_per_task": 2,
    }


def test_runner_records_memory_on_failure(monkeypatch, tmp_path) -> None:
    row = _manifest_row(tmp_path)

    monkeypatch.setattr(
        runner,
        "load_event_log",
        lambda *_args, **_kwargs: _raise("load failed"),
    )

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert payload["metadata"]["memory"]["peak_rss_bytes"] >= 0


def test_runner_can_skip_metric_evaluation(monkeypatch, tmp_path) -> None:
    captured: dict[str, Any] = {"loaded_paths": []}
    row = _manifest_row(tmp_path)
    row["metrics_json"] = json.dumps(
        {
            "enabled": False,
            "profile": "pm4py_default",
            "names": ["fitness", "precision"],
            "export_model": False,
        }
    )

    def fake_load_event_log(path: str, **_kwargs) -> str:
        captured["loaded_paths"].append(path)
        return f"loaded:{path}"

    monkeypatch.setattr(runner, "load_event_log", fake_load_event_log)
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _CapturingAlgorithm(captured))
    monkeypatch.setattr(
        runner,
        "evaluate_discovery_result",
        lambda *_args, **_kwargs: _raise("metrics should be skipped"),
    )

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert captured["loaded_paths"] == ["data/train.xes"]
    assert payload["metrics"] == {"fitness": None, "precision": None}
    assert set(payload["metric_statuses"]) == {"fitness", "precision"}
    assert all(
        metric_status["status"] == "not_computed"
        for metric_status in payload["metric_statuses"].values()
    )
    assert payload["metadata"]["metrics_config"]["enabled"] is False
    assert payload["metadata"]["timings"]["load_test_log_seconds"] == 0.0
    assert payload["metadata"]["timings"]["metrics_eval_call_seconds"] == 0.0


def test_runner_skips_train_log_loading_for_split_miner_without_metrics(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, Any] = {}
    row = _manifest_row(tmp_path)
    row["algorithm"] = "split_miner"
    row["algorithm_id"] = "split_miner"
    row["metrics_json"] = json.dumps({"enabled": False, "export_model": False})

    monkeypatch.setattr(
        runner,
        "load_event_log",
        lambda _path: _raise("path-based Split Miner should not load the log in PM4Py"),
    )
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _CapturingAlgorithm(captured))

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert captured["train_log"] is None
    assert captured["config"]["input_log_path"] == "data/train.xes"
    assert payload["status"] == "success"
    assert payload["metadata"]["timings"]["load_train_log_seconds"] == 0.0
    assert payload["metadata"]["timings"]["train_log_load_skipped"] is True


def test_atomic_result_write_preserves_existing_file_on_serialization_failure(
    tmp_path,
) -> None:
    output_path = tmp_path / "result.json"
    output_path.write_text('{"status": "success"}\n', encoding="utf-8")

    try:
        runner._write_json_atomically(output_path, {"bad": object()})
    except TypeError:
        pass
    else:
        raise AssertionError("non-serializable payload should fail")

    assert output_path.read_text(encoding="utf-8") == '{"status": "success"}\n'
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_runner_discovery_timeout_writes_normal_result_and_skips_metrics(
    monkeypatch,
    tmp_path,
) -> None:
    row = _manifest_row(tmp_path)
    row["params_json"] = json.dumps({"discovery_timeout_seconds": 0.05})

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _SleepingAlgorithm())
    monkeypatch.setattr(
        runner,
        "evaluate_discovery_result",
        lambda *_args, **_kwargs: _raise("metrics should be skipped after discovery timeout"),
    )
    monkeypatch.setattr(
        runner,
        "_export_discovered_model",
        lambda *_args, **_kwargs: _raise("model export should be skipped after timeout"),
    )

    started = time.perf_counter()
    output_path = runner.run_manifest_row(row, command_args=["test"])
    elapsed = time.perf_counter() - started
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert elapsed < 2
    assert payload["status"] == "timeout"
    assert payload["error_message"] == "TimeoutError: discovery exceeded 0.05 seconds."
    assert payload["model_path"] is None
    assert payload["runtime_seconds"] >= 0.04
    assert payload["metadata"]["discovery"]["timeout_seconds"] == 0.05
    assert payload["metadata"]["execution_config"] == {
        "discovery_timeout_seconds": 0.05,
        "recursion_limit": 10000,
    }
    assert payload["metadata"]["memory"]["peak_rss_bytes"] >= 0
    assert payload["metadata"]["timings"]["discovery_reported_runtime_seconds"] >= 0.04
    assert payload["metadata"]["timings"]["model_export_seconds"] == 0.0
    assert payload["metadata"]["timings"]["metrics_eval_call_seconds"] == 0.0
    assert all(value is None for value in payload["metrics"].values())
    assert all(record["status"] == "not_computed" for record in payload["metric_statuses"].values())


def test_runner_discovery_timeout_allows_fast_success(monkeypatch, tmp_path) -> None:
    row = _manifest_row(tmp_path)
    row["params_json"] = json.dumps({"discovery_timeout_seconds": 5})
    row["metrics_json"] = json.dumps({"enabled": False, "export_model": False})

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _FastAlgorithm())

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["status"] == "success"
    assert payload["runtime_seconds"] == 0.001
    assert payload["metadata"]["discovery"]["timeout_seconds"] == 5
    assert payload["metadata"]["execution_config"] == {
        "discovery_timeout_seconds": 5,
        "recursion_limit": 10000,
    }


def test_runner_records_algorithm_wrapper_timeout_execution_config(monkeypatch, tmp_path) -> None:
    row = _manifest_row(tmp_path)
    row["params_json"] = json.dumps({"timeout_seconds": 90})
    row["metrics_json"] = json.dumps({"enabled": False, "export_model": False})

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _FastAlgorithm())

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["status"] == "success"
    assert payload["metadata"]["execution_config"] == {
        "timeout_seconds": 90,
        "recursion_limit": 10000,
    }


def test_runner_persists_failed_inductive_result_with_recursion_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    row = _manifest_row(tmp_path)
    row["algorithm"] = "inductive_miner"
    row["algorithm_id"] = "inductive_miner"
    row["params_json"] = json.dumps(
        {"variant": "imf", "noise_threshold": 0.2, "recursion_limit": 4321},
        sort_keys=True,
    )
    row["algorithm_params_json"] = row["params_json"]
    row["metrics_json"] = json.dumps({"enabled": True, "export_model": False})

    def fake_discover(_train_log, _config):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: InductiveMiner())
    monkeypatch.setattr(
        "process_discovery_cash.discovery.inductive.discover_inductive_miner",
        fake_discover,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_discovery_result",
        lambda *_args, **_kwargs: _raise("metrics should be skipped after failed discovery"),
    )

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert payload["error_message"] == "RecursionError: maximum recursion depth exceeded"
    assert payload["hyperparameters"]["recursion_limit"] == 4321
    assert payload["metadata"]["execution_config"] == {"recursion_limit": 4321}
    assert payload["metadata"]["discovery"]["error_type"] == "RecursionError"
    assert payload["metadata"]["discovery"]["log_id"] == "tiny_log"
    assert payload["metadata"]["discovery"]["input_log_path"] == "data/train.xes"
    assert payload["metadata"]["discovery"]["recursion_limit_used"] == 4321
    assert all(record["status"] == "not_computed" for record in payload["metric_statuses"].values())


def test_timeout_worker_honors_configured_recursion_limit_for_pickling(monkeypatch) -> None:
    import multiprocessing
    import pickle
    import sys

    from process_discovery_cash.discovery.base import DiscoveryResult
    from process_discovery_cash.experiments.discovery_timeout import discover_with_timeout

    class _DeepResultAlgorithm:
        algorithm_name = "heuristics_miner_plusplus"
        backend_name = "pm4py"
        default_model_type = "petri_net"

        def discover(self, _train_log, _config):
            return DiscoveryResult(
                algorithm_name=self.algorithm_name,
                backend_name=self.backend_name,
                hyperparameters=dict(_config),
                runtime_seconds=0.01,
                status="success",
                model_type="petri_net",
            )

    original_dumps = pickle.dumps

    def fake_dumps(obj, protocol=None):
        if (
            isinstance(obj, DiscoveryResult)
            and obj.status == "success"
            and obj.hyperparameters.get("recursion_limit") == 1000
        ):
            assert sys.getrecursionlimit() == 1000
            raise RecursionError("maximum recursion depth exceeded while pickling an object")
        if (
            isinstance(obj, DiscoveryResult)
            and obj.status == "success"
            and obj.hyperparameters.get("recursion_limit") == 10000
        ):
            assert sys.getrecursionlimit() == 10000
        return original_dumps(obj, protocol=protocol)

    monkeypatch.setattr(
        "process_discovery_cash.experiments.discovery_timeout.pickle.dumps", fake_dumps
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.discovery_timeout._multiprocessing_context",
        lambda: multiprocessing.get_context("fork"),
    )

    failed = discover_with_timeout(
        _DeepResultAlgorithm(),
        [],
        {"variant": "plusplus", "recursion_limit": 1000},
        5,
    )
    succeeded = discover_with_timeout(
        _DeepResultAlgorithm(),
        [],
        {"variant": "plusplus", "recursion_limit": 10000},
        5,
    )

    assert failed.status == "failed"
    assert (
        failed.error_message
        == "RecursionError: maximum recursion depth exceeded while pickling an object"
    )
    assert succeeded.status == "success"


def test_timeout_worker_defaults_recursion_limit_for_all_algorithms(monkeypatch) -> None:
    import multiprocessing
    import sys

    from process_discovery_cash.discovery.base import DiscoveryResult
    from process_discovery_cash.experiments.discovery_timeout import discover_with_timeout

    class _DeepResultAlgorithm:
        algorithm_name = "alpha_miner"
        backend_name = "pm4py"
        default_model_type = "petri_net"

        def discover(self, _train_log, _config):
            return DiscoveryResult(
                algorithm_name=self.algorithm_name,
                backend_name=self.backend_name,
                hyperparameters=dict(_config),
                runtime_seconds=0.01,
                status="success",
                model_type="petri_net",
                metadata={"during_recursion_limit": sys.getrecursionlimit()},
            )

    monkeypatch.setattr(
        "process_discovery_cash.experiments.discovery_timeout._multiprocessing_context",
        lambda: multiprocessing.get_context("fork"),
    )

    result = discover_with_timeout(
        _DeepResultAlgorithm(),
        [],
        {},
        5,
    )

    assert result.status == "success"
    assert result.metadata["during_recursion_limit"] == 10000


def test_runner_passes_metric_profile_to_evaluator(monkeypatch, tmp_path) -> None:
    captured: dict[str, Any] = {}
    row = _manifest_row(tmp_path)
    row["metrics_json"] = json.dumps(
        {
            "enabled": True,
            "profile": "token",
            "names": ["fitness", "precision"],
            "export_model": False,
        }
    )

    def fake_evaluate(
        _discovery_result,
        _test_log,
        metric_names=None,
        metric_profile=None,
        include_timings=False,
    ):
        captured["metric_names"] = metric_names
        captured["metric_profile"] = metric_profile
        assert include_timings is True
        return (
            {"fitness": 1.0, "precision": 0.75},
            {
                "fitness": {"status": "success", "value": 1.0, "error": None},
                "precision": {"status": "success", "value": 0.75, "error": None},
            },
            {"profile": metric_profile},
        )

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _CapturingAlgorithm({}))
    monkeypatch.setattr(runner, "evaluate_discovery_result", fake_evaluate)

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert captured["metric_names"] == ["fitness", "precision"]
    assert captured["metric_profile"] == "token"
    assert payload["metadata"]["metrics_config"]["profile"] == "token"
    assert payload["metadata"]["timings"]["metrics"]["profile"] == "token"


def test_runner_exports_model_artifact_when_requested(monkeypatch, tmp_path) -> None:
    row = _manifest_row(tmp_path)
    row["metrics_json"] = json.dumps(
        {
            "enabled": False,
            "profile": "pm4py_default",
            "names": ["fitness"],
            "export_model": True,
        }
    )

    class ModelAlgorithm(_CapturingAlgorithm):
        def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
            super().discover(train_log, config)
            return DiscoveryResult(
                algorithm_name="fake_algorithm",
                backend_name="fake_backend",
                hyperparameters=config,
                runtime_seconds=0.01,
                status="success",
                model_type="petri_net",
                discovered_model=(object(), object(), object()),
            )

    def fake_export(discovery_result: DiscoveryResult, output_path: Path) -> None:
        artifact_path = output_path.with_suffix("") / "discovered_model.pnml"
        metric_model_path = output_path.with_suffix("") / "discovered_model.metric.joblib"
        discovery_result.model_path = artifact_path.as_posix()
        discovery_result.metadata["model_artifact_path"] = artifact_path.as_posix()
        discovery_result.metadata["model_artifact_type"] = "pnml"
        discovery_result.metadata["metric_model_artifact_path"] = metric_model_path.as_posix()
        discovery_result.metadata["metric_model_artifact_type"] = "petri_net_joblib"
        discovery_result.metadata["metric_model_cache_version"] = 1

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: ModelAlgorithm({}))
    monkeypatch.setattr(runner, "_export_discovered_model", fake_export)

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["model_path"].endswith("/discovered_model.pnml")
    assert payload["metadata"]["discovery"]["model_artifact_type"] == "pnml"
    assert payload["metadata"]["discovery"]["metric_model_artifact_type"] == "petri_net_joblib"
    assert payload["metadata"]["discovery"]["metric_model_cache_version"] == 1
    assert payload["metadata"]["timings"]["model_export_seconds"] >= 0


def test_runner_keeps_and_reuses_split_miner_artifact_when_export_requested(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, Any] = {}
    row = _manifest_row(tmp_path)
    row["algorithm"] = "split_miner"
    row["algorithm_id"] = "split_miner"
    row["metrics_json"] = json.dumps(
        {
            "enabled": False,
            "profile": "pm4py_default",
            "names": ["fitness"],
            "export_model": True,
        }
    )

    class SplitLikeAlgorithm(_CapturingAlgorithm):
        def discover(self, train_log: Any, config: dict[str, Any]) -> DiscoveryResult:
            super().discover(train_log, config)
            artifact_path = Path(config["output_dir"]) / "split_miner_model.bpmn"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("<definitions />\n", encoding="utf-8")
            return DiscoveryResult(
                algorithm_name="split_miner",
                backend_name="external",
                hyperparameters=config,
                runtime_seconds=0.01,
                status="success",
                model_type="bpmn",
                model_path=artifact_path.as_posix(),
                discovered_model=object(),
            )

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: SplitLikeAlgorithm(captured))

    output_path = runner.run_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert captured["config"]["keep_output_files"] is True
    assert payload["model_path"].endswith("/split_miner_model.bpmn")
    assert Path(payload["model_path"]).exists()
    assert payload["metadata"]["discovery"]["model_artifact_type"] == "bpmn"
    assert not any(warning.startswith("Model export skipped") for warning in payload["warnings"])


def test_runner_skips_existing_successful_output(monkeypatch, tmp_path) -> None:
    row = _manifest_row(tmp_path)
    output_path = Path(row["output_path"])
    existing_payload = _complete_success_payload(row)
    output_path.write_text(json.dumps(existing_payload) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "load_event_log",
        lambda _path: _raise("successful output should be skipped"),
    )

    returned_path = runner.run_manifest_row(row, command_args=["test"])

    assert returned_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == existing_payload


def test_runner_does_not_reuse_result_from_another_manifest_row(tmp_path) -> None:
    row = _manifest_row(tmp_path)
    row["experiment_id"] = "large_manifest"
    row["config_hash"] = "stablehash"
    row["config_id"] = "stablehash"
    output_path = tmp_path / "000008_tiny_log_0_fake_algorithm_stablehash.json"
    row["output_path"] = output_path.as_posix()
    other_path = tmp_path / "000000_tiny_log_0_fake_algorithm_stablehash.json"
    other_path.write_text(json.dumps(_complete_success_payload(row)) + "\n", encoding="utf-8")

    assert runner.is_successfully_completed(row, output_path) is False
    assert not output_path.exists()


def test_result_file_states_only_treat_success_as_reusable(tmp_path) -> None:
    row = _manifest_row(tmp_path)
    output_path = Path(row["output_path"])

    assert runner.inspect_result_file(row, output_path).state is runner.ResultFileState.MISSING

    output_path.write_text("not json\n", encoding="utf-8")
    assert runner.inspect_result_file(row, output_path).state is runner.ResultFileState.CORRUPT

    for status, expected_state in [
        (None, runner.ResultFileState.INCOMPLETE),
        ("failed", runner.ResultFileState.FAILED),
        ("timeout", runner.ResultFileState.TIMEOUT),
        ("unsupported", runner.ResultFileState.UNSUPPORTED),
        ("success", runner.ResultFileState.INCOMPLETE),
    ]:
        payload = {} if status is None else {"status": status}
        output_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        assert runner.inspect_result_file(row, output_path).state is expected_state

    output_path.write_text(
        json.dumps(_complete_success_payload(row)) + "\n",
        encoding="utf-8",
    )
    assert runner.inspect_result_file(row, output_path).state is runner.ResultFileState.SUCCESS

    mismatched_payload = _complete_success_payload(row)
    mismatched_payload["metadata"]["config_hash"] = "different_hash"
    output_path.write_text(json.dumps(mismatched_payload) + "\n", encoding="utf-8")
    assert (
        runner.inspect_result_file(row, output_path).state
        is runner.ResultFileState.IDENTITY_MISMATCH
    )


def test_runner_force_recomputes_existing_successful_output(monkeypatch, tmp_path) -> None:
    row = _manifest_row(tmp_path)
    output_path = Path(row["output_path"])
    output_path.write_text(
        json.dumps({"status": "success", "metadata": {"config_hash": row["config_hash"]}}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "load_event_log", lambda path, **_kwargs: f"loaded:{path}")
    monkeypatch.setattr(runner, "get_algorithm", lambda _name: _CapturingAlgorithm({}))
    monkeypatch.setattr(
        runner,
        "evaluate_discovery_result",
        lambda *_args, **_kwargs: (
            {"fitness": 1.0, "precision": 1.0, "generalization": 1.0, "simplicity": 1.0},
            {
                "fitness": {"status": "success", "value": 1.0, "error": None},
                "precision": {"status": "success", "value": 1.0, "error": None},
                "generalization": {"status": "success", "value": 1.0, "error": None},
                "simplicity": {"status": "success", "value": 1.0, "error": None},
            },
            {},
        ),
    )

    runner.run_manifest_row(row, command_args=["test"], force=True)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["algorithm_name"] == "fake_algorithm"


def _manifest_row(tmp_path) -> dict[str, str]:
    return {
        "experiment_id": "test_experiment",
        "log_id": "tiny_log",
        "log_path": "data/train.xes",
        "test_log_path": "data/test.xes",
        "seed": "0",
        "algorithm": "fake_algorithm",
        "backend": "fake_backend",
        "params_json": "{}",
        "config_hash": "test_hash",
        "output_path": str(tmp_path / "result.json"),
    }


def _complete_success_payload(row: dict[str, str]) -> dict[str, Any]:
    return {
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "log_path": row["log_path"],
        "test_log_path": row["test_log_path"],
        "seed": int(row["seed"]),
        "algorithm_name": row["algorithm"],
        "backend": row["backend"],
        "hyperparameters": json.loads(row["params_json"]),
        "discovered_model_type": "unknown",
        "metrics": {},
        "metric_statuses": {},
        "status": "success",
        "metadata": {
            "config_hash": row["config_hash"],
            "metrics_config": {
                "enabled": True,
                "profile": "pm4py_default",
                "names": ["fitness", "precision", "generalization", "simplicity"],
                "export_model": False,
            },
        },
    }


def _raise(message: str) -> None:
    raise AssertionError(message)
