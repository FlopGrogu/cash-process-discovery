from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from process_discovery_cash.data.preprocessing.catalog import (
    DEFAULT_DATASET_CATALOG,
    load_dataset_catalog,
)
from process_discovery_cash.data.preprocessing.metadata import sha256_file
from process_discovery_cash.utils.paths import portable_project_path, resolve_portable_path

SPLIT_MINER_SHA256 = "472c006623d99a6e440aa93a58e29b867cc331cec2b12b3d7fb61fb2a5de8328"
DEFAULT_SPLIT_MINER_JAR = "data/external/split-miner-1.7.1-all.jar"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify manually acquired v6 datasets and the Split Miner JAR."
    )
    parser.add_argument("--catalog", default=str(DEFAULT_DATASET_CATALOG))
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--all", action="store_true", help="Verify all 21 v6 datasets.")
    parser.add_argument(
        "--split-miner-jar",
        default=os.getenv("SPLIT_MINER_JAR", DEFAULT_SPLIT_MINER_JAR),
    )
    parser.add_argument("--no-split-miner", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def verify_dataset_inputs(
    dataset_ids: list[str],
    *,
    catalog_path: str | Path = DEFAULT_DATASET_CATALOG,
) -> list[dict[str, Any]]:
    catalog = load_dataset_catalog(catalog_path)
    records: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        try:
            dataset = catalog.datasets[dataset_id]
        except KeyError:
            records.append(
                {
                    "kind": "dataset",
                    "id": dataset_id,
                    "status": "unknown",
                    "error": "dataset is not present in the v6 catalog",
                }
            )
            continue
        path = resolve_portable_path(dataset.source_path)
        records.append(
            _verify_file(
                kind="dataset",
                identifier=dataset_id,
                path=path,
                expected_sha256=dataset.sha256,
                expected_size=dataset.size_bytes,
                landing_url=dataset.landing_url,
            )
        )
    return records


def verify_split_miner_jar(path: str | Path) -> dict[str, Any]:
    resolved = resolve_portable_path(path)
    return _verify_file(
        kind="external_tool",
        identifier="split-miner-1.7.1",
        path=resolved,
        expected_sha256=SPLIT_MINER_SHA256,
        expected_size=None,
        landing_url=None,
    )


def _verify_file(
    *,
    kind: str,
    identifier: str,
    path: Path,
    expected_sha256: str | None,
    expected_size: int | None,
    landing_url: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "id": identifier,
        "path": portable_project_path(path),
        "expected_sha256": expected_sha256,
        "expected_size_bytes": expected_size,
        "landing_url": landing_url,
    }
    if not expected_sha256 or expected_size is None and kind == "dataset":
        record.update(status="invalid_catalog", error="checksum or size is not pinned")
        return record
    if not path.is_file():
        record.update(status="missing", error="file does not exist")
        return record
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    record.update(actual_size_bytes=actual_size, actual_sha256=actual_sha256)
    if expected_size is not None and actual_size != expected_size:
        record.update(status="mismatch", error="file size does not match the catalog")
    elif actual_sha256 != expected_sha256:
        record.update(status="mismatch", error="SHA-256 does not match the catalog")
    else:
        record["status"] = "ok"
    return record


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog = load_dataset_catalog(args.catalog)
    dataset_ids = sorted(set(args.datasets or []))
    if args.all:
        dataset_ids = sorted(set(dataset_ids) | set(catalog.datasets))
    if not dataset_ids:
        parser.error("Provide --dataset or --all.")

    records = verify_dataset_inputs(dataset_ids, catalog_path=args.catalog)
    if not args.no_split_miner:
        records.append(verify_split_miner_jar(args.split_miner_jar))
    summary = {
        "ok": all(record["status"] == "ok" for record in records),
        "checked": len(records),
        "records": records,
    }
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for record in records:
            print(f"[{record['status']}] {record['id']}: {record.get('path', '')}")
            if record.get("error"):
                print(f"  {record['error']}")
        print(f"Verified {len(records)} inputs; ok={summary['ok']}.")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
