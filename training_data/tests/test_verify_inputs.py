from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from process_discovery_cash.cli.verify_inputs import (
    verify_dataset_inputs,
    verify_split_miner_jar,
)
from process_discovery_cash.data.preprocessing.catalog import load_dataset_catalog


def _catalog(tmp_path: Path, source: Path, *, sha256: str, size: int) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "tiny": {
                        "display_name": "Tiny",
                        "source_path": source.as_posix(),
                        "sha256": sha256,
                        "size_bytes": size,
                        "landing_url": "https://example.invalid/tiny",
                        "license": "test-only",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    load_dataset_catalog.cache_clear()
    return path


def test_verify_inputs_accepts_exact_size_and_checksum(tmp_path: Path) -> None:
    source = tmp_path / "tiny.xes"
    source.write_bytes(b"exact input\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    records = verify_dataset_inputs(
        ["tiny"],
        catalog_path=_catalog(tmp_path, source, sha256=digest, size=source.stat().st_size),
    )

    assert records[0]["status"] == "ok"
    assert records[0]["actual_sha256"] == digest


def test_verify_inputs_reports_missing_and_mismatched_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.xes"
    missing_records = verify_dataset_inputs(
        ["tiny"],
        catalog_path=_catalog(tmp_path, missing, sha256="0" * 64, size=1),
    )
    assert missing_records[0]["status"] == "missing"

    source = tmp_path / "wrong.xes"
    source.write_bytes(b"wrong")
    mismatch_records = verify_dataset_inputs(
        ["tiny"],
        catalog_path=_catalog(tmp_path, source, sha256="0" * 64, size=5),
    )
    assert mismatch_records[0]["status"] == "mismatch"


def test_verify_split_miner_jar_checks_presence_without_checksum(tmp_path: Path) -> None:
    jar = tmp_path / "split-miner-1.7.1-all.jar"

    missing = verify_split_miner_jar(jar)
    assert missing["status"] == "missing"

    jar.write_bytes(b"runtime decides whether this is a working JAR")
    present = verify_split_miner_jar(jar)
    assert present["status"] == "ok"
    assert present["actual_size_bytes"] == jar.stat().st_size
    assert "actual_sha256" not in present
    assert "expected_sha256" not in present
