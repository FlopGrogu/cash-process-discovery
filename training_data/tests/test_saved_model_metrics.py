from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.experiments import saved_model_metrics


def test_evaluate_saved_result_model_uses_exported_artifact(monkeypatch, tmp_path) -> None:
    result_path = _write_source_result(tmp_path / "source.json", tmp_path / "model.pnml")
    captured: dict[str, Any] = {}

    def fake_load_model_artifact(model_path: Path, *, metric_model_path: Path | None = None):
        captured["model_path"] = model_path
        captured["metric_model_path"] = metric_model_path
        return "pnml", "petri_net", (object(), object(), object()), {"load_model_seconds": 0.02}

    def fake_evaluate(
        discovery_result: DiscoveryResult,
        test_log: Any,
        metric_names=None,
        metric_profile=None,
        include_timings=False,
    ):
        captured["test_log"] = test_log
        captured["metric_names"] = metric_names
        captured["metric_profile"] = metric_profile
        captured["include_timings"] = include_timings
        captured["discovery_model_path"] = discovery_result.model_path
        return (
            {"fitness": 1.0},
            {"fitness": {"status": "success", "value": 1.0, "error": None}},
            {"profile": metric_profile, "total_seconds": 0.01},
        )

    monkeypatch.setattr(
        saved_model_metrics,
        "_load_model_artifact_with_fallback",
        fake_load_model_artifact,
    )
    monkeypatch.setattr(
        saved_model_metrics,
        "load_event_log",
        lambda path: f"loaded:{path}",
    )
    monkeypatch.setattr(saved_model_metrics, "evaluate_discovery_result", fake_evaluate)

    output_path = saved_model_metrics.evaluate_saved_result_model(
        result_path,
        metric_profile="alignment",
        metric_names=["fitness"],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert captured["model_path"] == tmp_path / "model.pnml"
    assert captured["metric_model_path"] is None
    assert captured["test_log"] == "loaded:data/test.xes"
    assert captured["metric_names"] == ["fitness"]
    assert captured["metric_profile"] == "alignment"
    assert captured["include_timings"] is True
    assert captured["discovery_model_path"] == (tmp_path / "model.pnml").as_posix()
    assert payload["metric_profile"] == "alignment"
    assert payload["metrics"] == {"fitness": 1.0}
    assert payload["status"] == "success"
    assert payload["source_result_path"] == result_path.as_posix()
    assert payload["config_hash"] == "abc123"
    assert payload["log_path"] == "data/train.xes"
    assert payload["seed"] == 0
    assert payload["backend"] == "pm4py"
    assert payload["hyperparameters"] == {"variant": "classic"}
    assert payload["metric_runtime_seconds"] >= 0
    assert payload["discovery_runtime_seconds"] == 0.1
    assert payload["discovery_status"] == "success"
    assert payload["source_metadata"]["config_hash"] == "abc123"
    assert payload["metadata"]["timings"]["model_load_backend"] == "pnml"


def test_evaluate_saved_result_model_prefers_metric_artifact(monkeypatch, tmp_path) -> None:
    metric_model_path = tmp_path / "model.metric.joblib"
    result_path = _write_source_result(
        tmp_path / "source.json",
        tmp_path / "model.pnml",
        metric_model_path=metric_model_path,
    )
    captured: dict[str, Any] = {}

    def fake_load_model_artifact(model_path: Path, *, metric_model_path: Path | None = None):
        captured["model_path"] = model_path
        captured["metric_model_path"] = metric_model_path
        return (
            "joblib",
            "petri_net",
            (object(), object(), object()),
            {"load_metric_model_seconds": 0.01},
        )

    monkeypatch.setattr(
        saved_model_metrics,
        "_load_model_artifact_with_fallback",
        fake_load_model_artifact,
    )
    monkeypatch.setattr(saved_model_metrics, "load_event_log", lambda path: f"loaded:{path}")
    monkeypatch.setattr(
        saved_model_metrics,
        "evaluate_discovery_result",
        lambda *_args, **_kwargs: (
            {"fitness": 1.0},
            {"fitness": {"status": "success", "value": 1.0, "error": None}},
            {},
        ),
    )

    output_path = saved_model_metrics.evaluate_saved_result_model(result_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert captured["model_path"] == tmp_path / "model.pnml"
    assert captured["metric_model_path"] == metric_model_path
    assert payload["metadata"]["timings"]["model_load_backend"] == "joblib"
    assert payload["metadata"]["timings"]["load_metric_model_seconds"] == 0.01


def test_evaluate_saved_result_tree_skips_results_without_exported_models(
    monkeypatch,
    tmp_path,
) -> None:
    root = tmp_path / "results"
    output_dir = tmp_path / "metrics"
    _write_source_result(root / "good.json", tmp_path / "model.pnml")
    (root / "failed.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "experiment_id": "failed-exp",
                "log_id": "failed-log",
                "algorithm_name": "alpha_miner",
                "backend": "pm4py",
                "log_path": "data/train_failed.xes",
                "test_log_path": "data/test_failed.xes",
                "model_path": None,
                "metadata": {"config_hash": "failed123"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "already_metrics.json").write_text(
        json.dumps(
            {
                "status": "success",
                "source_result_path": "results/source.json",
                "metric_profile": "token",
                "model_path": (tmp_path / "model.pnml").as_posix(),
                "test_log_path": "data/test.xes",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        saved_model_metrics,
        "_load_model_artifact_with_fallback",
        lambda _path, *, metric_model_path=None: (
            "pnml",
            "petri_net",
            (object(), object(), object()),
            {"load_model_seconds": 0.02},
        ),
    )
    monkeypatch.setattr(saved_model_metrics, "load_event_log", lambda path: f"loaded:{path}")
    monkeypatch.setattr(
        saved_model_metrics,
        "evaluate_discovery_result",
        lambda *_args, **_kwargs: (
            {"fitness": 1.0},
            {"fitness": {"status": "success", "value": 1.0, "error": None}},
            {},
        ),
    )

    written_paths = saved_model_metrics.evaluate_saved_result_tree(
        root,
        output_dir,
        metric_profile="token",
        metric_names=["fitness"],
    )

    assert written_paths == [output_dir / "failed.json", output_dir / "good.json"]
    assert (
        json.loads((output_dir / "good.json").read_text(encoding="utf-8"))["metric_profile"]
        == "token"
    )
    failed_payload = json.loads((output_dir / "failed.json").read_text(encoding="utf-8"))
    assert failed_payload["status"] == "success_missing"
    assert failed_payload["metrics"] == {"fitness": 0.0}
    assert failed_payload["metric_statuses"]["fitness"]["status"] == "missing_model"


def test_evaluate_saved_result_model_defaults_missing_model_metrics_to_zero(tmp_path) -> None:
    result_path = _write_source_result(
        tmp_path / "source_missing_model.json",
        tmp_path / "missing_model.pnml",
        create_model_file=False,
    )

    output_path = saved_model_metrics.evaluate_saved_result_model(
        result_path,
        metric_profile="token",
        metric_names=["fitness", "precision"],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["status"] == "success_missing"
    assert payload["metrics"] == {"fitness": 0.0, "precision": 0.0}
    assert payload["metric_statuses"]["fitness"]["status"] == "missing_model"
    assert payload["metric_statuses"]["fitness"]["value"] == 0.0
    assert payload["metadata"]["timings"]["model_load_backend"] == "fallback_zero"
    assert payload["metadata"]["timings"]["load_test_log_seconds"] == 0.0


def test_evaluate_saved_result_model_defaults_failed_discovery_metrics_to_zero(tmp_path) -> None:
    result_path = _write_source_result(tmp_path / "source_failed.json", tmp_path / "model.pnml")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    output_path = saved_model_metrics.evaluate_saved_result_model(
        result_path, metric_names=["fitness"]
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written["status"] == "success_missing"
    assert written["discovery_status"] == "failed"
    assert written["metrics"] == {"fitness": 0.0}
    assert written["metric_statuses"]["fitness"]["status"] == "missing_model"


def test_saved_model_success_missing_output_is_reused_without_force(monkeypatch, tmp_path) -> None:
    result_path = _write_source_result(
        tmp_path / "source_missing_model.json",
        tmp_path / "missing_model.pnml",
        create_model_file=False,
    )
    output_path = saved_model_metrics.evaluate_saved_result_model(
        result_path, metric_names=["fitness"]
    )
    before = output_path.read_text(encoding="utf-8")

    original_load_json = saved_model_metrics._load_json

    def guarded_load_json(path: Path):
        if path == result_path:
            raise AssertionError(f"should not reread {path}")
        return original_load_json(path)

    monkeypatch.setattr(saved_model_metrics, "_load_json", guarded_load_json)

    reused_path = saved_model_metrics.evaluate_saved_result_model(
        result_path,
        metric_names=["fitness"],
        output_path=output_path,
        force=False,
    )

    assert reused_path == output_path
    assert output_path.read_text(encoding="utf-8") == before


def test_load_model_artifact_with_fallback_uses_xml_when_metric_artifact_is_corrupt(
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "model.pnml"
    metric_model_path = tmp_path / "model.metric.joblib"
    model_path.touch()
    metric_model_path.write_text("bad", encoding="utf-8")

    monkeypatch.setattr(
        saved_model_metrics,
        "_load_model_artifact",
        lambda path: ("petri_net", f"xml:{path.name}"),
    )

    backend, model_type, discovered_model, timings = (
        saved_model_metrics._load_model_artifact_with_fallback(
            model_path,
            metric_model_path=metric_model_path,
        )
    )

    assert backend == "pnml"
    assert model_type == "petri_net"
    assert discovered_model == "xml:model.pnml"
    assert "load_model_seconds" in timings


def _write_source_result(
    result_path: Path,
    model_path: Path,
    *,
    metric_model_path: Path | None = None,
    create_model_file: bool = True,
) -> Path:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if create_model_file:
        model_path.touch()
    if metric_model_path is not None:
        metric_model_path.touch()
    result_path.write_text(
        json.dumps(
            {
                "status": "success",
                "experiment_id": "experiment",
                "log_id": "log",
                "algorithm_name": "alpha_miner",
                "backend": "pm4py",
                "hyperparameters": {"variant": "classic"},
                "runtime_seconds": 0.1,
                "log_path": "data/train.xes",
                "model_path": model_path.as_posix(),
                "test_log_path": "data/test.xes",
                "seed": 0,
                "warnings": ["source-warning"],
                "discovered_model_type": "petri_net",
                "metadata": {
                    "config_hash": "abc123",
                    "discovery": (
                        {
                            "metric_model_artifact_path": metric_model_path.as_posix(),
                        }
                        if metric_model_path is not None
                        else {}
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return result_path
