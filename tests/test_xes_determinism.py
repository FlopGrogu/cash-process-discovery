from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from process_discovery_cash.data.augmentation import (
    AugmentationSpec,
    augment_parent_log,
)
from process_discovery_cash.data.xes import write_canonical_xes


def _frame() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    for case in range(20):
        activities = ["a", "b", "c"] if case % 2 else ["a", "c", "b"]
        for position, activity in enumerate(activities):
            rows.append(
                {
                    "case:concept:name": f"c{case:02d}",
                    "concept:name": activity,
                    "time:timestamp": base + pd.Timedelta(minutes=position),
                }
            )
    return pd.DataFrame(rows)


def test_canonical_xes_has_fixed_gzip_header_and_bytes(tmp_path: Path) -> None:
    first = write_canonical_xes(_frame(), tmp_path / "first.xes.gz")
    second = write_canonical_xes(
        _frame().sample(frac=1, random_state=5), tmp_path / "second.xes.gz"
    )

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\0\0\0\0"


def test_two_clean_python_processes_emit_identical_xes_and_features(tmp_path: Path) -> None:
    script = """
import hashlib, json, sys
import pandas as pd
from process_discovery_cash.data.features import extract_features_from_xes
from process_discovery_cash.data.xes import write_canonical_xes
base = pd.Timestamp("2024-01-01T00:00:00Z")
rows = []
for case in range(20):
    acts = ["a", "b", "c"] if case % 2 else ["a", "c", "b"]
    for pos, act in enumerate(acts):
        rows.append({"case:concept:name": f"c{case:02d}", "concept:name": act,
                     "time:timestamp": base + pd.Timedelta(minutes=pos)})
path = write_canonical_xes(pd.DataFrame(rows), sys.argv[1])
payload = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "features": extract_features_from_xes(str(path))}
print(json.dumps(payload, sort_keys=True, allow_nan=True))
"""
    environment = dict(os.environ, PYTHONHASHSEED="0")
    first = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "a.xes.gz")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "b.xes.gz")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout

    assert json.loads(first) == json.loads(second)
    assert (tmp_path / "a.xes.gz").read_bytes() == (tmp_path / "b.xes.gz").read_bytes()


def test_augmentation_child_ids_records_and_checksums_are_deterministic(
    tmp_path: Path,
) -> None:
    spec = AugmentationSpec("subsample", {"fraction": 0.5})
    first = augment_parent_log(
        _frame(),
        "parent",
        [spec],
        output_dir=tmp_path / "first",
        base_seed=1001,
        parent_sha256="parent-sha",
    )[0]
    second = augment_parent_log(
        _frame(),
        "parent",
        [spec],
        output_dir=tmp_path / "second",
        base_seed=1001,
        parent_sha256="parent-sha",
    )[0]

    assert first.child_log_id == second.child_log_id
    assert first.artifact_sha256 == second.artifact_sha256
    assert hashlib.sha256(Path(first.output_path).read_bytes()).hexdigest() == first.artifact_sha256
    assert Path(first.output_path).read_bytes() == Path(second.output_path).read_bytes()
