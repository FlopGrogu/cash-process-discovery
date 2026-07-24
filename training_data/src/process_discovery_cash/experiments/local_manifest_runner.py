from __future__ import annotations

import csv
import json
import traceback
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from process_discovery_cash.experiments.identity import (
    semantic_run_identity,
)
from process_discovery_cash.experiments.runner import (
    is_successfully_completed,
    load_manifest_rows,
)
from process_discovery_cash.experiments.runner import (
    run_manifest_row as _run_manifest_row,
)
from process_discovery_cash.utils.hashing import stable_hash

STATUS_COLUMNS = [
    "row_index",
    "run_id",
    "log_id",
    "seed",
    "algorithm_id",
    "algorithm_variant",
    "status",
    "output_path",
    "started_at",
    "finished_at",
    "duration_seconds",
    "error",
]


@dataclass(frozen=True)
class IndexedManifestRow:
    row_index: int
    row: dict[str, str]


@dataclass(frozen=True)
class LocalRunResult:
    row_index: int
    run_id: str
    log_id: str
    seed: str
    algorithm_id: str
    algorithm_variant: str
    status: str
    output_path: str
    started_at: str
    finished_at: str
    duration_seconds: float
    error: str = ""

    def to_csv_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestFilters:
    row_indices: set[int] | None = None
    algorithm: str | None = None
    variant: str | None = None
    log_id: str | None = None
    seed: str | None = None
    max_rows: int | None = None


def default_status_path(manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    return manifest.with_name(f"{manifest.stem}.local_status.csv")


def load_indexed_manifest_rows(
    manifest_path: str | Path,
    output_root: str | Path | None = None,
) -> list[IndexedManifestRow]:
    rows = load_manifest_rows(manifest_path)
    if not rows:
        return []
    _validate_manifest_schema(rows[0])
    return [
        IndexedManifestRow(
            row_index=index,
            row=normalize_manifest_row(row, index, output_root=output_root),
        )
        for index, row in enumerate(rows)
    ]


def select_manifest_rows(
    rows: Iterable[IndexedManifestRow],
    filters: ManifestFilters,
) -> list[IndexedManifestRow]:
    selected = [indexed for indexed in rows if _row_matches(indexed, filters)]
    if filters.max_rows is not None:
        selected = selected[: filters.max_rows]
    return selected


def dry_run_rows(rows: Iterable[IndexedManifestRow]) -> list[dict[str, str]]:
    return [_dry_run_record(indexed) for indexed in rows]


def run_local_manifest(
    rows: list[IndexedManifestRow],
    *,
    status_path: str | Path,
    force: bool = False,
    strict: bool = False,
    workers: int = 1,
    command_args: list[str] | None = None,
) -> list[LocalRunResult]:
    if workers < 1:
        raise ValueError("--workers must be greater than or equal to 1")

    status_file = Path(status_path)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    results: list[LocalRunResult] = []
    with status_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        if workers == 1:
            for indexed in rows:
                result = run_local_manifest_row(
                    indexed,
                    force=force,
                    command_args=command_args,
                )
                writer.writerow(result.to_csv_record())
                handle.flush()
                results.append(result)
                if strict and result.status == "failed":
                    break
            return results

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_local_manifest_row_worker,
                    indexed,
                    force,
                    command_args,
                ): indexed
                for indexed in rows
            }
            for future in as_completed(futures):
                indexed = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed_at = _utc_now()
                    error = f"{type(exc).__name__}: {exc}"
                    _write_error_file(
                        Path(indexed.row["output_path"]),
                        error,
                        traceback.format_exc(),
                    )
                    result = _result_from_row(
                        indexed,
                        status="failed",
                        started_at=_format_time(failed_at),
                        finished_at=_format_time(failed_at),
                        duration_seconds=0.0,
                        error=error,
                    )
                writer.writerow(result.to_csv_record())
                handle.flush()
                results.append(result)
                if strict and result.status == "failed":
                    for pending in futures:
                        pending.cancel()
                    break
    return results


def run_local_manifest_row(
    indexed: IndexedManifestRow,
    *,
    force: bool = False,
    command_args: list[str] | None = None,
) -> LocalRunResult:
    row = indexed.row
    started = _utc_now()
    started_at = _format_time(started)
    output_path = Path(row["output_path"])

    if not force and is_successfully_completed(row, output_path):
        finished = _utc_now()
        return _result_from_row(
            indexed,
            status="skipped",
            started_at=started_at,
            finished_at=_format_time(finished),
            duration_seconds=_duration_seconds(started, finished),
        )

    try:
        written_path = _run_manifest_row(row, command_args=command_args, force=force)
        result_payload = _load_result_payload(written_path)
        row_status = str(result_payload.get("status") or "failed")
        status = row_status if row_status in {"success", "skipped"} else "failed"
        error = (
            "" if status in {"success", "skipped"} else _result_error(result_payload, row_status)
        )
        if status == "failed":
            _write_error_file(output_path, error, "")
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        _write_error_file(output_path, error, traceback.format_exc())

    finished = _utc_now()
    return _result_from_row(
        indexed,
        status=status,
        started_at=started_at,
        finished_at=_format_time(finished),
        duration_seconds=_duration_seconds(started, finished),
        error=error,
    )


def summarize_results(results: Iterable[LocalRunResult]) -> dict[str, int]:
    summary = {"success": 0, "skipped": 0, "failed": 0}
    for result in results:
        if result.status not in summary:
            summary[result.status] = 0
        summary[result.status] += 1
    return summary


def _run_local_manifest_row_worker(
    indexed: IndexedManifestRow,
    force: bool,
    command_args: list[str] | None,
) -> LocalRunResult:
    return run_local_manifest_row(indexed, force=force, command_args=command_args)


def _validate_manifest_schema(row: Mapping[str, str]) -> None:
    missing: list[str] = []
    for column in ["log_id", "log_path", "seed"]:
        if column not in row:
            missing.append(column)
    if "algorithm_id" not in row and "algorithm" not in row:
        missing.append("algorithm_id or algorithm")
    if "output_path" not in row and "output_dir" not in row:
        missing.append("output_path or output_dir")
    if missing:
        raise ValueError(f"Manifest is missing required column(s): {', '.join(missing)}")


def normalize_manifest_row(
    row: Mapping[str, str],
    row_index: int,
    *,
    output_root: str | Path | None,
) -> dict[str, str]:
    normalized = {key: "" if value is None else str(value) for key, value in row.items()}
    algorithm_id = normalized.get("algorithm_id") or normalized.get("algorithm") or ""
    if not algorithm_id:
        raise ValueError(f"Manifest row {row_index} is missing an algorithm_id/algorithm value")
    normalized["algorithm_id"] = algorithm_id
    normalized["algorithm"] = normalized.get("algorithm") or algorithm_id
    normalized["backend"] = normalized.get("backend") or "unknown"
    normalized["experiment_id"] = normalized.get("experiment_id") or "manifest"

    params = _parse_params(normalized)
    params_json = json.dumps(params, sort_keys=True)
    normalized["params_json"] = params_json
    normalized["algorithm_params_json"] = params_json

    variant = (
        normalized.get("algorithm_variant")
        or normalized.get("variant")
        or str(params.get("variant", ""))
    )
    normalized["algorithm_variant"] = variant

    run_id = (
        normalized.get("config_id")
        or normalized.get("config_hash")
        or normalized.get("run_id")
        or stable_hash(_semantic_identity_for_unhashed_row(normalized, params))
    )
    normalized["config_id"] = run_id
    normalized["config_hash"] = normalized.get("config_hash") or run_id
    normalized["run_id"] = normalized.get("run_id") or run_id

    output_path = _normalize_output_path(normalized, output_root)
    normalized["output_path"] = output_path.as_posix()
    return normalized


def _semantic_identity_for_unhashed_row(
    row: Mapping[str, str],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    metrics_json = row.get("metrics_json", "")
    metrics: Mapping[str, Any] | str = {}
    if metrics_json:
        try:
            parsed_metrics = json.loads(metrics_json)
            metrics = parsed_metrics if isinstance(parsed_metrics, Mapping) else metrics_json
        except json.JSONDecodeError:
            metrics = metrics_json
    if not isinstance(metrics, Mapping):
        metrics = {"raw": metrics}
    return semantic_run_identity(
        log_id=row.get("log_id"),
        log_path=row.get("train_log_path") or row.get("log_path"),
        test_log_path=(
            row.get("test_log_path")
            or row.get("evaluation_log_path")
            or row.get("train_log_path")
            or row.get("log_path")
        ),
        seed=row.get("seed", 0),
        algorithm_id=row.get("algorithm_id") or row.get("algorithm"),
        backend=row.get("backend"),
        params=params,
        metrics=metrics,
    )


def _parse_params(row: Mapping[str, str]) -> dict[str, Any]:
    raw_params = (
        row.get("algorithm_params_json")
        or row.get("params_json")
        or row.get("algorithm_params")
        or row.get("params")
        or "{}"
    )
    if not raw_params:
        return {}
    try:
        parsed = json.loads(raw_params)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(raw_params)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        parsed_type = type(parsed).__name__
        raise ValueError(f"Manifest parameters must parse to a mapping, got {parsed_type}")
    return parsed


def _normalize_output_path(row: Mapping[str, str], output_root: str | Path | None) -> Path:
    if row.get("output_path"):
        output_path = Path(str(row["output_path"]))
    elif row.get("output_dir"):
        output_path = Path(str(row["output_dir"])) / "result.json"
    else:
        raise ValueError("Manifest row is missing an output_path/output_dir value")

    if output_root is not None and not output_path.is_absolute():
        output_path = Path(output_root) / output_path
    return output_path


def _row_matches(indexed: IndexedManifestRow, filters: ManifestFilters) -> bool:
    row = indexed.row
    if filters.row_indices is not None and indexed.row_index not in filters.row_indices:
        return False
    if filters.algorithm is not None and row.get("algorithm_id") != filters.algorithm:
        return False
    if filters.variant is not None and row.get("algorithm_variant") != filters.variant:
        return False
    if filters.log_id is not None and row.get("log_id") != filters.log_id:
        return False
    if filters.seed is not None and str(row.get("seed")) != filters.seed:
        return False
    return True


def _dry_run_record(indexed: IndexedManifestRow) -> dict[str, str]:
    row = indexed.row
    return {
        "row_index": str(indexed.row_index),
        "run_id": row["config_id"],
        "log_id": row["log_id"],
        "seed": row["seed"],
        "algorithm_id": row["algorithm_id"],
        "algorithm_variant": row.get("algorithm_variant", ""),
        "output_path": row["output_path"],
    }


def _load_result_payload(output_path: str | Path) -> dict[str, Any]:
    with Path(output_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Result file must contain a JSON object: {output_path}")
    return payload


def _result_error(payload: Mapping[str, Any], row_status: str) -> str:
    error = payload.get("error_message") or payload.get("error") or ""
    if error:
        return str(error)
    return f"Discovery row completed with status {row_status!r}"


def _write_error_file(output_path: Path, error: str, traceback_text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_suffix(".error.txt")
    content = error
    if traceback_text:
        content = f"{content}\n\n{traceback_text}"
    error_path.write_text(f"{content}\n", encoding="utf-8")


def _result_from_row(
    indexed: IndexedManifestRow,
    *,
    status: str,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    error: str = "",
) -> LocalRunResult:
    row = indexed.row
    return LocalRunResult(
        row_index=indexed.row_index,
        run_id=row["config_id"],
        log_id=row["log_id"],
        seed=row["seed"],
        algorithm_id=row["algorithm_id"],
        algorithm_variant=row.get("algorithm_variant", ""),
        status=status,
        output_path=row["output_path"],
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        error=error,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_time(value: datetime) -> str:
    return value.isoformat()


def _duration_seconds(started: datetime, finished: datetime) -> float:
    return round((finished - started).total_seconds(), 6)
