"""Per-target result JSON files for parallel (cluster) GEDI execution.

One JSON per target under a results directory is the terminal marker of that
target's execution: a Slurm rerun skips targets with a terminal result and
regenerates everything else, mirroring the discovery pipeline's result-file
semantics.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from process_discovery_cash.generation.pipeline import CandidateRecord

RESULT_SCHEMA_VERSION = 1


def result_path_for(results_dir: str | Path, target_id: str) -> Path:
    return Path(results_dir) / f"{target_id}.json"


def build_target_result(
    target_id: str,
    records: list[CandidateRecord],
    *,
    row_index: int,
    base_seed: int,
    n_trials: int,
    max_attempts: int,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "target_id": target_id,
        "row_index": row_index,
        "base_seed": base_seed,
        "n_trials": n_trials,
        "max_attempts": max_attempts,
        "terminal_status": records[-1].status if records else "generation_failed",
        "records": [record.to_json_dict() for record in records],
    }


def write_target_result(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return path


def load_target_result(path: str | Path) -> dict[str, Any] | None:
    """Return the result payload, or None when missing/corrupt (=> rerun)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or "terminal_status" not in payload:
        return None
    return payload
