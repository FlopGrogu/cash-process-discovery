from __future__ import annotations

import argparse

from process_discovery_cash.data.augmentation import (
    AUGMENTED_LOG_DIRNAME,
    DEFAULT_OUTPUT_ROOT,
    MANIFEST_FILENAME,
    ChildLogRecord,
    augment_parent_log,
    canonicalize_event_log,
    compute_log_stats,
    default_augmentation_plan,
    write_augmentation_manifest,
)
from process_discovery_cash.data.loading import load_event_log
from process_discovery_cash.data.preprocessing.catalog import (
    DEFAULT_DATASET_CATALOG,
    get_dataset,
    load_dataset_catalog,
)
from process_discovery_cash.utils.paths import resolve_portable_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate augmented child event logs from real parent logs. "
            "Children inherit the parent's source/fold and must never be "
            "used as independent test logs."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset id from the catalog. Repeatable.",
    )
    parser.add_argument("--all", action="store_true", help="Augment every catalog dataset.")
    parser.add_argument("--catalog", default=str(DEFAULT_DATASET_CATALOG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--base-seed", type=int, default=1001)
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Also generate stress children (coverage 0.5, noise 0.2, top-activity 0.8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate children whose output files already exist.",
    )
    parser.add_argument(
        "--large-log-traces",
        type=int,
        default=10_000,
        help="Trace count at which a 25%% subsample child is added.",
    )
    parser.add_argument(
        "--long-trace-mean-length",
        type=float,
        default=40.0,
        help="Mean trace length at which a truncation child is added.",
    )
    parser.add_argument(
        "--truncate-length",
        type=int,
        default=50,
        help="Events kept per case by the truncation child.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    catalog = load_dataset_catalog(args.catalog)
    dataset_ids = sorted(set(args.datasets or []))
    if args.all:
        dataset_ids = sorted(set(dataset_ids) | set(catalog.datasets))
    if not dataset_ids:
        parser.error("Provide --dataset or --all.")

    output_root = resolve_portable_path(args.output_root)
    logs_dir = output_root / AUGMENTED_LOG_DIRNAME
    manifest_path = output_root / MANIFEST_FILENAME

    all_records: list[ChildLogRecord] = []
    for dataset_id in dataset_ids:
        dataset = get_dataset(dataset_id, args.catalog)
        parent_log = canonicalize_event_log(
            load_event_log(dataset.source_path, cache_key=dataset_id)
        )
        parent_stats = compute_log_stats(parent_log)
        plan = default_augmentation_plan(
            parent_stats,
            include_stress=args.include_stress,
            large_log_traces=args.large_log_traces,
            long_trace_mean_length=args.long_trace_mean_length,
            truncate_length=args.truncate_length,
        )
        records = augment_parent_log(
            parent_log,
            dataset_id,
            plan,
            output_dir=logs_dir,
            base_seed=args.base_seed,
            parent_path=dataset.source_path,
            parent_sha256=dataset.sha256 or "",
            overwrite=args.overwrite,
        )
        all_records.extend(records)
        write_augmentation_manifest(all_records, manifest_path)
        _print_dataset_summary(dataset_id, parent_stats, records)

    accepted = [record for record in all_records if record.status == "accepted"]
    rejected = [record for record in all_records if record.status == "rejected"]
    skipped = [record for record in all_records if record.status == "skipped_existing"]
    print(
        f"\n{len(dataset_ids)} parent logs -> {len(all_records)} children "
        f"({len(accepted)} accepted, {len(rejected)} rejected, {len(skipped)} skipped)."
    )
    print(f"Child logs: {logs_dir}")
    print(f"Manifest:   {manifest_path}")
    if accepted:
        print("Generated children are ready for the v6 feature-anchor and preprocessing stages.")


def _print_dataset_summary(
    dataset_id: str, parent_stats: dict, records: list[ChildLogRecord]
) -> None:
    print(
        f"{dataset_id}: {parent_stats['n_traces']} traces, "
        f"{parent_stats['n_events']} events, {parent_stats['n_variants']} variants"
    )
    for record in records:
        if record.status == "accepted":
            detail = (
                f"traces={record.n_traces} events={record.n_events} "
                f"activities={record.n_activities} variants={record.n_variants}"
            )
        else:
            detail = record.rejection_reason or ""
        print(f"  [{record.status}] {record.child_log_id}: {detail}")


if __name__ == "__main__":
    main()
