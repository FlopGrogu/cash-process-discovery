from __future__ import annotations

import csv

import pytest

from process_discovery_cash.cli.preprocess_logs import collect_log_inputs


def test_collect_log_inputs_deduplicates_manifest_rows(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["log_id", "log_path", "test_log_path"])
        writer.writeheader()
        writer.writerow(
            {"log_id": "tiny", "log_path": "data/tiny.xes", "test_log_path": "data/tiny.xes"}
        )
        writer.writerow(
            {"log_id": "tiny", "log_path": "data/tiny.xes", "test_log_path": "data/tiny.xes"}
        )

    inputs = collect_log_inputs(config_path=None, manifest_path=str(manifest), log_paths=[])

    assert [(item.log_id, item.path) for item in inputs] == [("tiny", "data/tiny.xes")]


def test_collect_log_inputs_rejects_log_id_path_collisions(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["log_id", "log_path"])
        writer.writeheader()
        writer.writerow({"log_id": "same", "log_path": "data/a.xes"})
        writer.writerow({"log_id": "same", "log_path": "data/b.xes"})

    with pytest.raises(ValueError, match="multiple paths"):
        collect_log_inputs(config_path=None, manifest_path=str(manifest), log_paths=[])


def test_metric_manifest_test_log_uses_plain_log_id_cache_key(tmp_path) -> None:
    manifest = tmp_path / "metrics.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["log_id", "test_log_path"])
        writer.writeheader()
        writer.writerow({"log_id": "tiny", "test_log_path": "data/test.xes"})

    inputs = collect_log_inputs(config_path=None, manifest_path=str(manifest), log_paths=[])

    assert [(item.log_id, item.path) for item in inputs] == [("tiny", "data/test.xes")]


def test_metric_manifest_honors_explicit_test_cache_key(tmp_path) -> None:
    manifest = tmp_path / "metrics.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["log_id", "test_log_path", "log_cache_key"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "log_id": "tiny",
                "test_log_path": "data/test.xes",
                "log_cache_key": "tiny_test",
            }
        )

    inputs = collect_log_inputs(config_path=None, manifest_path=str(manifest), log_paths=[])

    assert [(item.log_id, item.path) for item in inputs] == [("tiny_test", "data/test.xes")]
