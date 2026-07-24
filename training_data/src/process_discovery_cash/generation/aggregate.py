"""Aggregate per-target GEDI result JSONs into the batch manifest.

Cross-candidate near-duplicate deduplication happens here (deterministically,
ordered by target id) because parallel per-target execution can only validate
against the real anchor.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_discovery_cash.generation.feature_space import (
    NEAR_DUPLICATE_DISTANCE,
    RealAnchor,
    to_target_space,
)
from process_discovery_cash.generation.pipeline import (
    MANIFEST_COLUMNS,
    MANIFEST_FILENAME,
    REJECTED_DIRNAME,
    CandidateRecord,
    candidate_record_from_json,
)
from process_discovery_cash.generation.results import load_target_result


def aggregate_results(
    results_dir: str | Path,
    real_features: pd.DataFrame,
    *,
    output_root: str | Path,
    known_target_ids: set[str] | None = None,
) -> tuple[list[CandidateRecord], dict[str, Any]]:
    """Rebuild records from result JSONs, dedup accepted logs, write manifest.

    Returns the final records plus an info dict (missing/unknown target ids).
    """
    results_dir = Path(results_dir)
    output_root = Path(output_root)
    rejected_dir = output_root / REJECTED_DIRNAME

    payloads = []
    for path in sorted(results_dir.glob("*.json")):
        payload = load_target_result(path)
        if payload is not None:
            payloads.append(payload)
    payloads.sort(key=lambda payload: str(payload.get("target_id")))

    anchor = RealAnchor.from_features(real_features)
    real_points = to_target_space(real_features)
    kept_points: list[np.ndarray] = []

    records: list[CandidateRecord] = []
    seen_target_ids: set[str] = set()
    for payload in payloads:
        seen_target_ids.add(str(payload.get("target_id")))
        target_records = sorted(
            (candidate_record_from_json(item) for item in payload.get("records", [])),
            key=lambda record: record.attempt,
        )
        for record in target_records:
            if record.status != "accepted":
                records.append(record)
                continue
            point = to_target_space(record.achieved_values)
            reference = np.vstack([real_points, *kept_points])
            distance = float(anchor.nearest_distance_to(point, reference)[0])
            record.duplicate_distance = distance
            if distance < NEAR_DUPLICATE_DISTANCE:
                record.status = "rejected"
                record.rejection_reason = (
                    "near-duplicate of a real or earlier accepted log "
                    f"(aggregate dedup, standardized distance {distance:.3f})"
                )
                record.output_path = _quarantine_log(record, rejected_dir)
            else:
                kept_points.append(point)
            records.append(record)

    manifest_path = output_root / MANIFEST_FILENAME
    frame = pd.DataFrame([record.to_row() for record in records], columns=MANIFEST_COLUMNS)
    frame = frame.sort_values(["target_id", "attempt"], kind="stable")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)

    info: dict[str, Any] = {"n_result_files": len(payloads)}
    if known_target_ids is not None:
        info["missing_target_ids"] = sorted(known_target_ids - seen_target_ids)
        info["unknown_target_ids"] = sorted(seen_target_ids - known_target_ids)
    return records, info


def _quarantine_log(record: CandidateRecord, rejected_dir: Path) -> str | None:
    if not record.output_path:
        return record.output_path
    source = Path(record.output_path)
    if not source.exists():
        return record.output_path
    rejected_dir.mkdir(parents=True, exist_ok=True)
    destination = rejected_dir / f"{record.log_id}_dedup.xes.gz"
    shutil.move(str(source), destination)
    return str(destination)
