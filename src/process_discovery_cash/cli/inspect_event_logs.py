from __future__ import annotations

import argparse
import json

from process_discovery_cash.data.loading import load_event_log_with_info
from process_discovery_cash.data.preprocessing.artifacts import (
    canonicalize_dataframe,
    inspect_dataframe,
)
from process_discovery_cash.data.preprocessing.catalog import (
    get_dataset,
    load_dataset_catalog,
)
from process_discovery_cash.data.preprocessing.lifecycle import analyze_lifecycle
from process_discovery_cash.data.preprocessing.metadata import inspect_dataset_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect event-log packages without modifying their source files."
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--catalog", default="configs/datasets/processmining_org.yaml")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Parse every event and compute exact statistics; potentially expensive.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    catalog = load_dataset_catalog(args.catalog)
    dataset_ids = args.datasets or list(catalog.datasets)
    report = []
    for dataset_id in dataset_ids:
        dataset = get_dataset(dataset_id, args.catalog)
        package = inspect_dataset_package(dataset)
        record = package.model_dump(mode="json")
        record["resolved_schema"] = dataset.event_schema.model_dump(mode="json")
        if args.full:
            loaded = load_event_log_with_info(dataset.source_path, use_cache=False)
            canonical, validation = canonicalize_dataframe(loaded.log, dataset)
            lifecycle = analyze_lifecycle(
                canonical,
                semantics=dataset.event_schema.lifecycle_semantics,
                case_column="case:concept:name",
                activity_column="concept:name",
                timestamp_column="time:timestamp",
                lifecycle_column=dataset.event_schema.lifecycle,
                start_timestamp_column=dataset.event_schema.start_timestamp,
            )
            record["validation"] = validation
            record["inspection"] = inspect_dataframe(canonical, source_order=True)
            record["lifecycle_analysis"] = lifecycle.model_dump(mode="json")
        report.append(record)

    text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
