from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from process_discovery_cash.data.inventory import (
    GeneratedInventoryError,
    verify_generated_inventory,
)


def test_generated_inventory_validates_counts_paths_and_checksums(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    augmented = data_root / "augmented/logs/aug_1.xes.gz"
    synthetic = data_root / "synthetic/gedi/logs/syn_1.xes.gz"
    augmented.parent.mkdir(parents=True)
    synthetic.parent.mkdir(parents=True)
    augmented.write_bytes(b"augmented")
    synthetic.write_bytes(b"synthetic")
    _write_csv(
        data_root / "augmented/manifest.csv",
        [
            {
                "child_log_id": "aug_1",
                "status": "accepted",
                "output_path": "data/augmented/logs/aug_1.xes.gz",
                "artifact_sha256": _sha256(augmented),
            }
        ],
    )
    _write_csv(
        data_root / "synthetic/gedi/manifest.csv",
        [
            {
                "log_id": "syn_1",
                "status": "accepted",
                "output_path": "data/synthetic/gedi/logs/syn_1.xes.gz",
                "artifact_sha256": _sha256(synthetic),
            }
        ],
    )
    _write_csv(
        data_root / "synthetic/gedi/targets.csv",
        [{"target_id": "t0000"}, {"target_id": "t0001"}],
    )

    receipt = verify_generated_inventory(
        data_root=data_root,
        expected_inventory={
            "real_logs": 1,
            "accepted_augmented_logs": 1,
            "gedi_targets": 2,
            "accepted_synthetic_logs": 1,
            "hpo_logs": 3,
        },
    )

    assert receipt["status"] == "ok"
    assert receipt["total_event_logs"] == 3
    assert "hpo_logs" not in receipt
    assert receipt["generated_artifact_count"] == 2
    assert len(receipt["generated_artifact_receipt_sha256"]) == 64


def test_generated_inventory_rejects_bad_checksum_and_nonportable_path(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    artifact = data_root / "augmented/logs/aug_1.xes.gz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"wrong")
    _write_csv(
        data_root / "augmented/manifest.csv",
        [
            {
                "child_log_id": "aug_1",
                "status": "accepted",
                "output_path": artifact.as_posix(),
                "artifact_sha256": "0" * 64,
            }
        ],
    )
    _write_csv(data_root / "synthetic/gedi/manifest.csv", [])
    _write_csv(data_root / "synthetic/gedi/targets.csv", [])

    with pytest.raises(GeneratedInventoryError, match="portable data/"):
        verify_generated_inventory(
            data_root=data_root,
            expected_inventory={
                "real_logs": 0,
                "accepted_augmented_logs": 1,
                "gedi_targets": 0,
                "accepted_synthetic_logs": 0,
                "total_event_logs": 1,
            },
        )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
