from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import process_discovery_cash.experiments.metric_manifest as metric_manifest
from process_discovery_cash.cli import generate_metric_manifest as generate_metric_manifest_cli
from process_discovery_cash.discovery.base import DiscoveryResult
from process_discovery_cash.experiments.aggregation import aggregate_results
from process_discovery_cash.experiments.metric_manifest import (
    METRIC_MANIFEST_COLUMNS,
    _metric_output_is_stale_success_missing,
    generate_metric_manifest_rows_from_source_manifest,
    metric_log_dir_from_source_manifest_rows,
    metric_output_path_from_source_result_path,
    metric_output_source_root_from_source_manifest_rows,
    run_metric_manifest_row,
    source_model_is_available,
    write_metric_manifest,
)


def test_result_tree_generation_path_is_removed() -> None:
    # The result-tree-based generator (scan result JSONs and skip rows) has been
    # deleted; only the source-manifest workflow remains.
    assert not hasattr(metric_manifest, "generate_metric_manifest_rows")
    assert not hasattr(metric_manifest, "generate_metric_manifest")
    assert not hasattr(metric_manifest, "default_metric_output_root")
    assert not hasattr(metric_manifest, "default_metric_log_dir")


def test_generate_metric_manifest_preserves_every_source_row(tmp_path) -> None:
    # Discovery outputs deliberately differ, but generation must not inspect them.
    output_root = tmp_path / "metric_outputs"
    model_path = tmp_path / "model.pnml"
    model_path.touch()

    good_path = tmp_path / "discovery" / "log_good" / "good.json"
    failed_path = tmp_path / "discovery" / "log_failed" / "failed.json"
    malformed_path = tmp_path / "discovery" / "log_malformed" / "malformed.json"
    missing_model_path = tmp_path / "discovery" / "log_missing_model" / "missing_model.json"
    missing_result_path = tmp_path / "discovery" / "log_missing_result" / "missing_result.json"

    _write_source_result(good_path, model_path)
    _write_source_result(
        missing_model_path,
        tmp_path / "absent.pnml",
        create_model_file=False,
    )
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text(
        json.dumps(
            {
                "status": "timeout",
                "experiment_id": "failed_experiment",
                "log_id": "failed_log",
                "algorithm_name": "alpha_miner",
                "log_path": "data/train_failed.xes",
                "test_log_path": "data/test_failed.xes",
                "model_path": (tmp_path / "never_exported.pnml").as_posix(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("{bad json", encoding="utf-8")

    source_paths = [
        good_path,
        failed_path,
        malformed_path,
        missing_model_path,
        missing_result_path,
    ]
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(source_manifest_path, source_paths)

    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="alignment",
        metric_names=["fitness"],
        output_root=output_root,
    )

    # Every source row is preserved: row count == source row count, nothing dropped.
    assert len(result.rows) == len(source_paths)
    assert result.stats.total == len(source_paths)
    assert result.stats.full == len(source_paths)
    assert result.stats.fallback_failed_discovery == 0
    assert result.stats.fallback_malformed_json == 0
    assert result.stats.fallback_missing_model == 0
    assert result.stats.fallback_missing_result == 0

    emitted = [row["source_result_path"] for row in result.rows]
    assert emitted == [path.as_posix() for path in source_paths]
    assert "model_path" not in METRIC_MANIFEST_COLUMNS
    assert "metric_model_path" not in METRIC_MANIFEST_COLUMNS
    assert result.rows[0]["metric_profile"] == "alignment"
    assert json.loads(result.rows[0]["metric_names_json"]) == ["fitness"]


def test_metric_manifest_preserves_xes_path_over_legacy_artifact_path(tmp_path) -> None:
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(
        source_manifest_path,
        [tmp_path / "discovery.json"],
        extra_row_values={
            "log_id": "tiny",
            "log_path": "data/example/tiny_log.xes",
            "test_log_path": "data/example/tiny_log.xes",
            "test_discovery_log_path": "data/processed/tiny.parquet",
        },
    )

    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="token",
        metric_names=["fitness"],
        output_root=tmp_path / "metrics",
    )

    assert result.rows[0]["test_log_path"] == "data/example/tiny_log.xes"


def test_generate_metric_manifest_defaults_metric_timeout_column(tmp_path) -> None:
    output_root = tmp_path / "metric_outputs"
    model_path = tmp_path / "model.pnml"
    model_path.touch()
    good_path = _write_source_result(tmp_path / "discovery" / "log" / "good.json", model_path)
    missing_path = tmp_path / "discovery" / "log_missing" / "missing.json"
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(source_manifest_path, [good_path, missing_path])

    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="token",
        metric_names=["fitness"],
        output_root=output_root,
    )

    assert "metric_timeout_seconds" in METRIC_MANIFEST_COLUMNS
    # Default applies to every source row, irrespective of discovery state.
    assert [row["metric_timeout_seconds"] for row in result.rows] == ["1800", "1800"]


def test_generate_metric_manifest_disables_metric_timeout_when_none(tmp_path) -> None:
    output_root = tmp_path / "metric_outputs"
    model_path = tmp_path / "model.pnml"
    model_path.touch()
    good_path = _write_source_result(tmp_path / "discovery" / "log" / "good.json", model_path)
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(source_manifest_path, [good_path])

    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="token",
        metric_names=["fitness"],
        output_root=output_root,
        metric_timeout_seconds=None,
    )

    assert result.rows[0]["metric_timeout_seconds"] == ""


def test_generate_metric_manifest_rows_from_source_manifest_uses_only_manifest_rows(
    tmp_path,
) -> None:
    output_root = tmp_path / "metric_outputs"
    model_path = tmp_path / "model.pnml"
    model_path.touch()
    good_path = tmp_path / "discovery" / "log_a" / "good.json"
    failed_path = tmp_path / "discovery" / "log_b" / "failed.json"
    missing_result_path = tmp_path / "discovery" / "log_c" / "missing_result.json"
    unreferenced_path = tmp_path / "discovery" / "log_z" / "unreferenced.json"
    _write_source_result(good_path, model_path)
    _write_source_result(unreferenced_path, model_path)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text(
        json.dumps(
            {
                "status": "timeout",
                "experiment_id": "failed_experiment",
                "log_id": "failed_log",
                "algorithm_name": "alpha_miner",
                "log_path": "data/train_failed.xes",
                "test_log_path": "data/test_failed.xes",
                "model_path": model_path.as_posix(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(
        source_manifest_path,
        [good_path, failed_path, missing_result_path],
    )

    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="token",
        metric_names=["fitness"],
        output_root=output_root,
    )

    assert len(result.rows) == 3
    assert result.stats.total == 3
    assert result.stats.full == 3
    assert result.stats.fallback_failed_discovery == 0
    assert result.stats.fallback_missing_result == 0
    assert result.rows[0]["log_dir"] == (
        "logs/slurm/v6/baseline/alpha_classic/v1/metrics/token"
    )
    assert result.rows[0]["source_result_path"] == good_path.as_posix()
    assert result.rows[0]["output_path"] == (output_root / "log_a" / "good.json").as_posix()
    assert result.rows[1]["source_result_path"] == failed_path.as_posix()
    assert result.rows[1]["output_path"] == (output_root / "log_b" / "failed.json").as_posix()
    assert result.rows[1]["test_log_path"] == ""
    assert result.rows[2]["source_result_path"] == missing_result_path.as_posix()
    assert (
        result.rows[2]["output_path"] == (output_root / "log_c" / "missing_result.json").as_posix()
    )


def test_generate_metric_manifest_rows_from_source_manifest_preserves_flat_layout(tmp_path) -> None:
    output_root = tmp_path / "metric_outputs"
    model_path = tmp_path / "model.pnml"
    model_path.touch()
    good_path = tmp_path / "discovery" / "good.json"
    _write_source_result(good_path, model_path)
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(source_manifest_path, [good_path])

    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="token",
        metric_names=["fitness"],
        output_root=output_root,
    )

    assert result.rows[0]["output_path"] == (output_root / "good.json").as_posix()


def test_generate_metric_manifest_uses_source_manifest_hash_over_result_metadata(
    tmp_path,
) -> None:
    output_root = tmp_path / "metric_outputs"
    model_path = tmp_path / "model.pnml"
    good_path = tmp_path / "discovery" / "good.json"
    _write_source_result(good_path, model_path, config_hash="stale_result_hash")
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(
        source_manifest_path,
        [good_path],
        config_hashes=["current_manifest_hash"],
    )

    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="token",
        metric_names=["fitness"],
        output_root=output_root,
    )

    assert result.rows[0]["source_config_hash"] == "current_manifest_hash"


def test_generate_metric_manifest_fallback_rows_use_source_manifest_hash(tmp_path) -> None:
    output_root = tmp_path / "metric_outputs"
    missing_result_path = tmp_path / "discovery" / "missing.json"
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(
        source_manifest_path,
        [missing_result_path],
        config_hashes=["current_manifest_hash"],
    )

    result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="token",
        metric_names=["fitness"],
        output_root=output_root,
    )

    assert result.rows[0]["source_config_hash"] == "current_manifest_hash"
    assert result.stats.full == 1


def test_generate_metric_manifest_requires_source_config_hash_column(tmp_path) -> None:
    source_manifest_path = tmp_path / "source_manifest.csv"
    source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with source_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "log_dir", "output_path"])
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "experiment",
                "log_dir": "logs/slurm/v6/baseline/alpha_classic/v1",
                "output_path": (tmp_path / "missing.json").as_posix(),
            }
        )

    with pytest.raises(ValueError, match="config_hash"):
        generate_metric_manifest_rows_from_source_manifest(
            source_manifest_path,
            metric_profile="token",
            metric_names=["fitness"],
            output_root=tmp_path / "metric_outputs",
        )


def test_source_manifest_cli_defaults_match_v6_grouped_layout(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    model_path = tmp_path / "model.pnml"
    model_path.touch()
    algorithm_root = tmp_path / "results" / "cluster" / "v6" / "model" / "baseline" / "alpha_plus"
    result_paths = [
        algorithm_root / "bpi2012" / "row.json",
        algorithm_root / "bpi2013" / "row.json",
    ]
    for result_path in result_paths:
        _write_source_result(result_path, model_path)
    source_manifest_path = (
        tmp_path
        / "experiments"
        / "manifests"
        / "v6"
        / "model"
        / "baseline"
        / "alpha_plus"
        / "v1.csv"
    )
    _write_source_manifest(
        source_manifest_path,
        result_paths,
        log_dir="logs/slurm/v6/model/baseline/alpha_plus/v1",
    )

    generate_metric_manifest_cli.main(["--source-manifest", source_manifest_path.as_posix()])

    metric_manifest_path = (
        tmp_path
        / "experiments"
        / "manifests"
        / "v6"
        / "metrics"
        / "baseline"
        / "alpha_plus"
        / "v1"
        / "token_metrics.csv"
    )
    rows = list(csv.DictReader(metric_manifest_path.open(newline="", encoding="utf-8")))

    assert len(rows) == 2
    assert rows[0]["metric_profile"] == "token"
    assert rows[0]["log_dir"] == "logs/slurm/v6/model/baseline/alpha_plus/v1/metrics/token"
    assert rows[0]["output_path"] == (
        "results/cluster/v6/metrics/baseline/alpha_plus/v1/token/bpi2012/row.json"
    )
    assert rows[1]["output_path"] == (
        "results/cluster/v6/metrics/baseline/alpha_plus/v1/token/bpi2013/row.json"
    )


def test_cli_rejects_removed_results_root_option() -> None:
    # The result-tree workflow is gone; users cannot select it any more.
    with pytest.raises(SystemExit):
        generate_metric_manifest_cli.build_parser().parse_args(
            ["--results-root", "results/cluster/whatever"]
        )


def test_cli_requires_source_manifest() -> None:
    with pytest.raises(SystemExit):
        generate_metric_manifest_cli.build_parser().parse_args([])


def test_source_manifest_default_paths_are_derived_from_v6_model_manifest() -> None:
    source_manifest_path = "experiments/manifests/v6/model/baseline/inductive_imf/v1.1.csv"

    assert (
        generate_metric_manifest_cli.default_metric_manifest_path_from_source_manifest(
            source_manifest_path,
            "token",
        ).as_posix()
        == "experiments/manifests/v6/metrics/baseline/inductive_imf/v1.1/token_metrics.csv"
    )
    assert (
        generate_metric_manifest_cli.default_metric_output_root_from_source_manifest(
            source_manifest_path,
            "token",
        ).as_posix()
        == "results/cluster/v6/metrics/baseline/inductive_imf/v1.1/token"
    )


def test_metric_output_source_root_from_source_manifest_rows_ignores_blank_paths() -> None:
    rows = [
        {"output_path": ""},
        {"output_path": "results/cluster/v6/model/baseline/alpha_plus/bpi2012/a.json"},
        {"output_path": "results/cluster/v6/model/baseline/alpha_plus/bpi2013/b.json"},
    ]

    assert metric_output_source_root_from_source_manifest_rows(rows).as_posix() == (
        "results/cluster/v6/model/baseline/alpha_plus"
    )


def test_metric_output_source_root_from_source_manifest_rows_requires_output_paths() -> None:
    try:
        metric_output_source_root_from_source_manifest_rows([{"output_path": ""}])
    except ValueError as exc:
        assert "no output_path values" in str(exc)
    else:
        raise AssertionError("expected missing source output_path values to fail")


def test_metric_output_path_from_source_result_path_raises_for_non_relative_path() -> None:
    try:
        metric_output_path_from_source_result_path(
            "results/cluster/v6/model/baseline/alpha_plus/bpi2012/a.json",
            output_root="results/cluster/v6/metrics/baseline/alpha_plus/v1/token",
            source_results_root="results/cluster/v6/model/baseline/other_algorithm",
        )
    except ValueError as exc:
        assert "is not under" in str(exc)
    else:
        raise AssertionError("expected non-relative source result path to fail")


def test_write_metric_manifest_uses_metric_columns(tmp_path) -> None:
    output_path = tmp_path / "metrics.csv"
    row = {column: "" for column in METRIC_MANIFEST_COLUMNS}
    row.update(
        {
            "source_result_path": "results/source.json",
            "source_config_hash": "abc",
            "metric_profile": "token",
            "metric_names_json": json.dumps(["fitness"]),
            "log_dir": "logs/slurm/metrics/source",
            "output_path": "results/metrics/source.json",
        }
    )

    write_metric_manifest([row], output_path)

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert rows == [row]


def test_metric_log_dir_from_source_manifest_rows_rejects_inconsistent_values() -> None:
    rows = [
        {"log_dir": "logs/slurm/v6/baseline/alpha_classic/v1"},
        {"log_dir": "logs/slurm/v6/baseline/alpha_classic/v2"},
    ]

    try:
        metric_log_dir_from_source_manifest_rows(rows, metric_profile="token")
    except ValueError as exc:
        assert "inconsistent log_dir" in str(exc)
    else:
        raise AssertionError("expected inconsistent source manifest log_dir values to fail")


def test_run_metric_manifest_row_evaluates_one_model(monkeypatch, tmp_path) -> None:
    captured: dict[str, Any] = {}
    row = _metric_row(tmp_path)

    def fake_evaluate(
        discovery_result: DiscoveryResult,
        test_log: Any,
        metric_names=None,
        metric_profile=None,
        include_timings=False,
    ):
        captured["model_path"] = discovery_result.model_path
        captured["test_log"] = test_log
        captured["metric_names"] = metric_names
        captured["metric_profile"] = metric_profile
        captured["include_timings"] = include_timings
        return (
            {"fitness": 1.0},
            {"fitness": {"status": "success", "value": 1.0, "error": None}},
            {"profile": metric_profile},
        )

    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest._load_model_artifact_with_fallback",
        lambda _path, *, metric_model_path=None: (
            "pnml",
            "petri_net",
            (object(), object(), object()),
            {"load_model_seconds": 0.1},
        ),
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.load_event_log",
        lambda path, **_kwargs: f"loaded:{path}",
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.evaluate_discovery_result",
        fake_evaluate,
    )

    output_path = run_metric_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert captured["model_path"] == row["model_path"]
    assert captured["test_log"] == "loaded:data/test.xes"
    assert captured["metric_names"] == ["fitness"]
    assert captured["metric_profile"] == "token"
    assert captured["include_timings"] is True
    assert payload["status"] == "success"
    assert payload["metric_runtime_seconds"] >= 0
    assert payload["discovery_runtime_seconds"] == 1.25
    assert payload["discovery_status"] == "success"
    assert payload["config_hash"] == "abc123"
    assert payload["source_config_hash"] == "abc123"
    assert payload["log_path"] == "data/train.xes"
    assert payload["seed"] == 0
    assert payload["backend"] == "pm4py"
    assert payload["hyperparameters"] == {"variant": "classic"}
    assert payload["metric_profile"] == "token"
    assert payload["metrics"] == {"fitness": 1.0}
    assert payload["metadata"]["timings"]["load_model_seconds"] >= 0
    assert payload["metadata"]["timings"]["model_load_backend"] == "pnml"
    assert payload["metadata"]["timings"]["total_seconds_before_write"] >= 0
    assert payload["metadata"]["timings"]["metrics"]["profile"] == "token"
    assert payload["source_metadata"]["config_hash"] == "abc123"
    memory = payload["metadata"]["memory"]
    assert memory["peak_rss_bytes"] >= 0
    assert memory["peak_rss_mb"] >= 0
    assert memory["source"] == "resource.getrusage"
    assert memory["scope"] == "current_process_and_waited_children"


def test_run_metric_manifest_row_records_timeout(monkeypatch, tmp_path) -> None:
    from process_discovery_cash.experiments.metric_timeout import TimedMetricEvaluation

    row = _metric_row(tmp_path)
    row["metric_timeout_seconds"] = "0.05"
    captured: dict[str, Any] = {}

    def fake_evaluate_with_timeout(
        discovery_result, test_log, metric_names, *, metric_profile, timeout_seconds
    ):
        captured["timeout_seconds"] = timeout_seconds
        return TimedMetricEvaluation(
            metrics={name: None for name in metric_names},
            statuses={
                name: {"status": "timeout", "error": "TimeoutError: ...", "value": None}
                for name in metric_names
            },
            timings={"timed_out": True, "timeout_seconds": timeout_seconds},
            timed_out=True,
        )

    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest._load_model_artifact_with_fallback",
        lambda _path, *, metric_model_path=None: (
            "pnml",
            "petri_net",
            (object(), object(), object()),
            {"load_model_seconds": 0.1},
        ),
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.load_event_log",
        lambda path, **_kwargs: f"loaded:{path}",
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.evaluate_with_timeout",
        fake_evaluate_with_timeout,
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.evaluate_discovery_result",
        lambda *_a, **_k: pytest.fail("evaluate_discovery_result should be bypassed on timeout"),
    )

    output_path = run_metric_manifest_row(row, command_args=["test"])
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert captured["timeout_seconds"] == 0.05
    assert payload["status"] == "timeout"
    assert payload["error_message"] and "exceeded" in payload["error_message"]
    assert payload["metric_statuses"]["fitness"]["status"] == "timeout"
    assert payload["metrics"]["fitness"] is None
    # A timeout result is not a reusable success, so it re-runs next time.
    assert not metric_manifest._metric_output_is_reusable_success(Path(output_path), row)


def test_run_metric_manifest_row_records_slurm_requested_memory(monkeypatch, tmp_path) -> None:
    row = _metric_row(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "minor")
    monkeypatch.setenv("SLURM_JOB_NAME", "metrics_token")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("PDCASH_SLURM_REQUESTED_MEMORY", "4G")
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest._load_model_artifact_with_fallback",
        lambda _path, *, metric_model_path=None: (
            "pnml",
            "petri_net",
            (object(), object(), object()),
            {"load_model_seconds": 0.1},
        ),
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.load_event_log",
        lambda path, **_kwargs: f"loaded:{path}",
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.evaluate_discovery_result",
        lambda *_args, **_kwargs: (
            {"fitness": 1.0},
            {"fitness": {"status": "success", "value": 1.0, "error": None}},
            {},
        ),
    )

    output_path = run_metric_manifest_row(row)
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["metadata"]["slurm"] == {
        "job_id": "12345",
        "array_job_id": "12345",
        "array_task_id": "3",
        "job_partition": "minor",
        "job_name": "metrics_token",
        "requested_memory": "4G",
        "requested_memory_bytes": 4 * 1024**3,
        "requested_cpus_per_task": 1,
    }


def test_run_metric_manifest_row_skips_existing_success(monkeypatch, tmp_path) -> None:
    row = _metric_row(tmp_path)
    output_path = Path(row["output_path"])
    output_path.write_text(
        json.dumps(
            {
                "status": "success",
                "metric_statuses": {"fitness": {"status": "success", "value": 1.0, "error": None}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest._load_model_artifact_with_fallback",
        lambda _path, *, metric_model_path=None: (_ for _ in ()).throw(
            AssertionError("should be skipped")
        ),
    )

    returned_path = run_metric_manifest_row(row)

    assert returned_path == output_path


def test_run_metric_manifest_row_recomputes_partial_success_without_force(
    monkeypatch,
    tmp_path,
) -> None:
    row = _metric_row(tmp_path)
    output_path = Path(row["output_path"])
    output_path.write_text(
        json.dumps(
            {
                "status": "success",
                "metric_statuses": {
                    "fitness": {
                        "status": "backend_error",
                        "value": None,
                        "error": "metric backend failed",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _stub_real_metric_evaluation(monkeypatch)

    returned_path = run_metric_manifest_row(row, force=False)
    payload = json.loads(Path(returned_path).read_text(encoding="utf-8"))

    assert payload["status"] == "success"
    assert payload["metrics"] == {"fitness": 1.0}
    assert payload["metric_statuses"]["fitness"]["status"] == "success"


def test_run_metric_manifest_row_writes_failed_metric_result(monkeypatch, tmp_path) -> None:
    row = _metric_row(tmp_path)
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest._load_model_artifact_with_fallback",
        lambda _path, *, metric_model_path=None: (_ for _ in ()).throw(
            ValueError("unsupported model")
        ),
    )

    output_path = run_metric_manifest_row(row)
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["status"] == "failed"
    assert payload["metric_runtime_seconds"] >= 0
    assert payload["discovery_status"] == "success"
    assert payload["metrics"] == {"fitness": None}
    assert payload["metric_statuses"]["fitness"]["status"] == "not_computed"
    assert "unsupported model" in payload["error_message"]
    assert payload["metadata"]["memory"]["peak_rss_bytes"] >= 0


def test_run_metric_manifest_row_defaults_missing_model_metrics_to_zero(tmp_path) -> None:
    source_result_path = _write_source_result(
        tmp_path / "source_missing_model.json",
        tmp_path / "missing_model.pnml",
        create_model_file=False,
    )
    row = {
        "source_result_path": source_result_path.as_posix(),
        "source_config_hash": "abc123",
        "experiment_id": "experiment",
        "log_id": "log",
        "algorithm_name": "alpha_miner",
        "model_path": "",
        "metric_model_path": "",
        "test_log_path": "data/test.xes",
        "log_cache_key": "log",
        "metric_profile": "token",
        "metric_names_json": json.dumps(["fitness", "precision"]),
        "log_dir": "logs/slurm/metrics/test/token",
        "output_path": (tmp_path / "metric_missing_model.json").as_posix(),
    }

    output_path = run_metric_manifest_row(row)
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["status"] == "success_missing"
    assert payload["metrics"] == {"fitness": 0.0, "precision": 0.0}
    assert payload["metric_statuses"]["fitness"]["status"] == "missing_model"
    assert payload["metric_statuses"]["fitness"]["value"] == 0.0
    assert "defaulted metrics to 0" in payload["metric_statuses"]["fitness"]["error"]
    assert payload["metadata"]["timings"]["model_load_backend"] == "fallback_zero"


def test_run_metric_manifest_row_defaults_failed_discovery_metrics_to_zero(tmp_path) -> None:
    source_result_path = _write_source_result(
        tmp_path / "source_failed.json",
        tmp_path / "model.pnml",
    )
    payload = json.loads(source_result_path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    source_result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    row = _metric_row(tmp_path)
    row["source_result_path"] = source_result_path.as_posix()
    row["output_path"] = (tmp_path / "metric_failed_source.json").as_posix()

    output_path = run_metric_manifest_row(row)
    result_payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert result_payload["status"] == "success_missing"
    assert result_payload["discovery_status"] == "failed"
    assert result_payload["metrics"] == {"fitness": 0.0}
    assert result_payload["metric_statuses"]["fitness"]["status"] == "missing_model"
    assert "defaulted metrics to 0" in result_payload["metric_statuses"]["fitness"]["error"]


def test_run_metric_manifest_row_defaults_missing_source_result_to_zero(tmp_path) -> None:
    row = {
        "source_result_path": (tmp_path / "missing_source.json").as_posix(),
        "source_config_hash": "abc123",
        "experiment_id": "experiment",
        "log_id": "log",
        "algorithm_name": "alpha_miner",
        "model_path": "",
        "metric_model_path": "",
        "test_log_path": "data/test.xes",
        "log_cache_key": "log",
        "metric_profile": "token",
        "metric_names_json": json.dumps(["fitness"]),
        "log_dir": "logs/slurm/metrics/test/token",
        "output_path": (tmp_path / "metric_missing_source.json").as_posix(),
    }

    output_path = run_metric_manifest_row(row)
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert payload["status"] == "success_missing"
    assert payload["discovery_status"] == "failed"
    assert payload["metrics"] == {"fitness": 0.0}
    assert payload["metric_statuses"]["fitness"]["status"] == "missing_model"
    assert "no model is available" in payload["metric_statuses"]["fitness"]["error"]


def _stub_real_metric_evaluation(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_load(_path, *, metric_model_path=None):
        captured["model_path"] = Path(_path)
        captured["metric_model_path"] = metric_model_path
        return ("pnml", "petri_net", (object(), object(), object()), {"load_model_seconds": 0.1})

    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest._load_model_artifact_with_fallback",
        fake_load,
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.load_event_log",
        lambda path, **_kwargs: f"loaded:{path}",
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.evaluate_discovery_result",
        lambda *_args, **_kwargs: (
            {"fitness": 1.0},
            {"fitness": {"status": "success", "value": 1.0, "error": None}},
            {"profile": "token"},
        ),
    )
    return captured


def test_run_metric_manifest_row_recovers_model_from_source_when_row_fields_stale(
    monkeypatch, tmp_path
) -> None:
    # The manifest row was written with an empty model_path (a fallback/stale row),
    # but the source discovery result now exists and points at a real model.
    model_path = tmp_path / "model.pnml"
    source_result_path = _write_source_result(tmp_path / "source.json", model_path)
    row = {
        "source_result_path": source_result_path.as_posix(),
        "source_config_hash": "abc123",
        "experiment_id": "experiment",
        "log_id": "log",
        "algorithm_name": "alpha_miner",
        "model_path": "",
        "metric_model_path": "",
        "test_log_path": "data/test.xes",
        "log_cache_key": "log",
        "metric_profile": "token",
        "metric_names_json": json.dumps(["fitness"]),
        "log_dir": "logs/slurm/metrics/test/token",
        "output_path": (tmp_path / "metric.json").as_posix(),
    }
    captured = _stub_real_metric_evaluation(monkeypatch)

    output_path = run_metric_manifest_row(row)
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert captured["model_path"] == model_path
    assert payload["status"] == "success"
    assert payload["metrics"] == {"fitness": 1.0}


def test_run_metric_manifest_row_recomputes_stale_success_missing_output(
    monkeypatch, tmp_path
) -> None:
    # A stale success_missing placeholder exists, but the source model is now
    # available: it must be recomputed instead of skipped.
    model_path = tmp_path / "model.pnml"
    source_result_path = _write_source_result(tmp_path / "source.json", model_path)
    output_path = tmp_path / "metric.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "success_missing",
                "source_config_hash": "abc123",
                "metric_profile": "token",
                "source_result_path": source_result_path.as_posix(),
                "metrics": {"fitness": 0.0},
                "metric_statuses": {
                    "fitness": {
                        "status": "missing_model",
                        "value": 0.0,
                        "error": "defaulted metrics to 0",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = _metric_row(tmp_path)
    row["source_result_path"] = source_result_path.as_posix()
    row["output_path"] = output_path.as_posix()
    _stub_real_metric_evaluation(monkeypatch)

    returned_path = run_metric_manifest_row(row)
    payload = json.loads(Path(returned_path).read_text(encoding="utf-8"))

    assert payload["status"] == "success"
    assert payload["metrics"] == {"fitness": 1.0}


def test_run_metric_manifest_row_keeps_success_missing_when_no_model_recoverable(
    monkeypatch, tmp_path
) -> None:
    # The source discovery genuinely failed: the existing success_missing output
    # is correct and must not trigger recomputation.
    source_result_path = _write_source_result(
        tmp_path / "source_failed.json",
        tmp_path / "missing.pnml",
        create_model_file=False,
    )
    failed = json.loads(source_result_path.read_text(encoding="utf-8"))
    failed["status"] = "failed"
    source_result_path.write_text(json.dumps(failed) + "\n", encoding="utf-8")

    output_path = tmp_path / "metric.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "success_missing",
                "source_config_hash": "abc123",
                "metric_profile": "token",
                "source_result_path": source_result_path.as_posix(),
                "metrics": {"fitness": 0.0},
                "metric_statuses": {
                    "fitness": {
                        "status": "missing_model",
                        "value": 0.0,
                        "error": "defaulted metrics to 0",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = _metric_row(tmp_path)
    row["source_result_path"] = source_result_path.as_posix()
    row["model_path"] = ""
    row["output_path"] = output_path.as_posix()
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest._load_model_artifact_with_fallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should be skipped")),
    )

    assert source_model_is_available(row) is False
    assert _metric_output_is_stale_success_missing(output_path, row) is False

    returned_path = run_metric_manifest_row(row)
    payload = json.loads(Path(returned_path).read_text(encoding="utf-8"))

    assert payload["status"] == "success_missing"
    assert payload["metrics"] == {"fitness": 0.0}


def test_metric_manifest_smoke_path_aggregates_outputs(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "metric_outputs"
    model_path = tmp_path / "model.pnml"
    model_path.touch()
    row_path = tmp_path / "discovery" / "row.json"
    _write_source_result(row_path, model_path)
    source_manifest_path = tmp_path / "source_manifest.csv"
    _write_source_manifest(source_manifest_path, [row_path])
    # Disable the metric timeout so the in-process (monkeypatched) evaluation
    # path is exercised directly; the timeout subprocess is covered elsewhere.
    manifest_result = generate_metric_manifest_rows_from_source_manifest(
        source_manifest_path,
        metric_profile="token",
        metric_names=["fitness"],
        output_root=output_root,
        metric_timeout_seconds=None,
    )

    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest._load_model_artifact_with_fallback",
        lambda _path, *, metric_model_path=None: (
            "pnml",
            "petri_net",
            (object(), object(), object()),
            {"load_model_seconds": 0.1},
        ),
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.load_event_log",
        lambda path, **_kwargs: f"loaded:{path}",
    )
    monkeypatch.setattr(
        "process_discovery_cash.experiments.metric_manifest.evaluate_discovery_result",
        lambda *_args, **_kwargs: (
            {"fitness": 1.0},
            {"fitness": {"status": "success", "value": 1.0, "error": None}},
            {},
        ),
    )
    run_metric_manifest_row(manifest_result.rows[0])

    aggregate_path = aggregate_results(output_root, tmp_path / "aggregated.csv")
    rows = list(csv.DictReader(aggregate_path.open(newline="", encoding="utf-8")))

    assert rows[0]["config_hash"] == "abc123"
    assert rows[0]["log_path"] == "data/train.xes"
    assert rows[0]["backend"] == "pm4py"
    assert rows[0]["param_variant"] == "classic"
    assert rows[0]["runtime_seconds"] == "1.25"
    assert rows[0]["metric_runtime_seconds"] != ""
    assert rows[0]["metric_run_status"] == "success"
    assert rows[0]["source_config_hash"] == "abc123"
    assert rows[0]["metric_profile"] == "token"
    assert rows[0]["metric_fitness"] == "1.0"


def _metric_row(tmp_path: Path) -> dict[str, str]:
    model_path = tmp_path / "model.pnml"
    model_path.touch()
    source_result_path = _write_source_result(tmp_path / "source.json", model_path)
    return {
        "source_result_path": source_result_path.as_posix(),
        "source_config_hash": "abc123",
        "experiment_id": "experiment",
        "log_id": "log",
        "algorithm_name": "alpha_miner",
        "model_path": model_path.as_posix(),
        "metric_model_path": "",
        "test_log_path": "data/test.xes",
        "log_cache_key": "log",
        "metric_profile": "token",
        "metric_names_json": json.dumps(["fitness"]),
        "log_dir": "logs/slurm/metrics/test/token",
        "output_path": (tmp_path / "metric.json").as_posix(),
    }


def _write_source_result(
    result_path: Path,
    model_path: Path,
    *,
    metric_model_path: Path | None = None,
    create_model_file: bool = True,
    config_hash: str = "abc123",
    metrics_config: dict[str, Any] | None = None,
) -> Path:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if create_model_file:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()
    result_path.write_text(
        json.dumps(
            {
                "status": "success",
                "experiment_id": "experiment",
                "log_id": "log",
                "log_path": "data/train.xes",
                "algorithm_name": "alpha_miner",
                "backend": "pm4py",
                "hyperparameters": {"variant": "classic"},
                "runtime_seconds": 1.25,
                "model_path": model_path.as_posix(),
                "test_log_path": "data/test.xes",
                "seed": 0,
                "warnings": ["source-warning"],
                "discovered_model_type": "petri_net",
                "metrics": {"fitness": None},
                "metric_statuses": {
                    "fitness": {"status": "not_computed", "error": None, "value": None}
                },
                "metadata": {
                    "config_hash": config_hash,
                    **({"metrics_config": metrics_config} if metrics_config is not None else {}),
                    "discovery": (
                        {"metric_model_artifact_path": metric_model_path.as_posix()}
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


def _write_source_manifest(
    manifest_path: Path,
    output_paths: list[Path],
    *,
    log_dir: str = "logs/slurm/v6/baseline/alpha_classic/v1",
    config_hashes: list[str] | None = None,
    extra_row_values: dict[str, str] | None = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config_hashes = config_hashes or ["abc123"] * len(output_paths)
    extra_row_values = extra_row_values or {}
    fieldnames = [
        "experiment_id",
        *[field for field in extra_row_values if field not in {"experiment_id"}],
        "config_hash",
        "log_dir",
        "output_path",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, output_path in enumerate(output_paths):
            row = {
                "experiment_id": f"experiment_{index}",
                **extra_row_values,
                "config_hash": config_hashes[index],
                "log_dir": log_dir,
                "output_path": output_path.as_posix(),
            }
            writer.writerow(row)
