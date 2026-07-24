from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from process_discovery_cash.experiments.local_manifest_runner import (
    ManifestFilters,
    default_status_path,
    dry_run_rows,
    load_indexed_manifest_rows,
    run_local_manifest,
    select_manifest_rows,
    summarize_results,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run rows from a generated experiment manifest on the local machine. "
            "By default, all matching rows are attempted and failures are reported "
            "at the end with a nonzero exit code."
        )
    )
    parser.add_argument("--manifest", required=True, help="Path to a generated manifest CSV.")
    parser.add_argument("--max-rows", type=int, help="Run at most N matching rows.")
    parser.add_argument(
        "--row-index",
        type=int,
        help="Run exactly one zero-based manifest row index.",
    )
    parser.add_argument(
        "--row-indices",
        type=_parse_row_indices,
        help="Comma-separated zero-based row indices to run, for example 1,2,5.",
    )
    parser.add_argument("--algorithm", help="Run only rows for this algorithm_id/algorithm.")
    parser.add_argument(
        "--variant",
        help="Run only rows for this algorithm variant, if present in the manifest.",
    )
    parser.add_argument("--log-id", help="Run only rows for this log_id.")
    parser.add_argument("--seed", help="Run only rows for this seed.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of local worker processes. Default: 1.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run rows even when their success output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without running discovery.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first failed row.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after failed rows. This is the default behavior.",
    )
    parser.add_argument(
        "--status-path",
        help="Status CSV path. Defaults to <manifest stem>.local_status.csv.",
    )
    parser.add_argument(
        "--output-root",
        help="Prefix relative row output paths with this directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable more detailed logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    manifest_path = Path(args.manifest)
    status_path = Path(args.status_path) if args.status_path else default_status_path(manifest_path)
    command_args = sys.argv if argv is None else ["pdcash-run-manifest-local", *argv]

    rows = load_indexed_manifest_rows(manifest_path, output_root=args.output_root)
    requested_indices = _selected_row_indices(args)
    _validate_requested_indices(parser, requested_indices, rows)
    filters = ManifestFilters(
        row_indices=requested_indices,
        algorithm=args.algorithm,
        variant=args.variant,
        log_id=args.log_id,
        seed=str(args.seed) if args.seed is not None else None,
        max_rows=args.max_rows,
    )
    selected = select_manifest_rows(rows, filters)
    LOGGER.info("Selected %d of %d manifest rows", len(selected), len(rows))

    if args.dry_run:
        _print_dry_run(selected)
        return

    results = run_local_manifest(
        selected,
        status_path=status_path,
        force=args.force,
        strict=args.strict,
        workers=args.workers,
        command_args=command_args,
    )
    summary = summarize_results(results)
    print(f"Selected rows: {len(selected)}")
    print(f"Succeeded: {summary.get('success', 0)}")
    print(f"Skipped: {summary.get('skipped', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Status path: {status_path}")

    if summary.get("failed", 0):
        raise SystemExit(1)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.strict and args.continue_on_error:
        parser.error("--strict and --continue-on-error cannot be used together")
    if args.row_index is not None and args.row_indices is not None:
        parser.error("--row-index and --row-indices cannot be used together")
    if args.max_rows is not None and args.max_rows < 0:
        parser.error("--max-rows must be greater than or equal to 0")
    if args.workers < 1:
        parser.error("--workers must be greater than or equal to 1")
    if args.strict and args.workers != 1:
        parser.error("--strict requires --workers 1 for deterministic stop-on-first-failure")


def _selected_row_indices(args: argparse.Namespace) -> set[int] | None:
    if args.row_index is not None:
        return {args.row_index}
    return args.row_indices


def _validate_requested_indices(
    parser: argparse.ArgumentParser,
    requested_indices: set[int] | None,
    rows: list,
) -> None:
    if requested_indices is None:
        return
    available_indices = {row.row_index for row in rows}
    missing = sorted(requested_indices - available_indices)
    if missing:
        parser.error(f"Manifest row index/indices outside loaded CSV: {missing}")


def _parse_row_indices(value: str) -> set[int]:
    indices: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid row index {part!r}; expected integers separated by commas"
            ) from exc
        if index < 0:
            raise argparse.ArgumentTypeError("Row indices must be greater than or equal to 0")
        indices.add(index)
    if not indices:
        raise argparse.ArgumentTypeError("At least one row index is required")
    return indices


def _print_dry_run(rows: list) -> None:
    records = dry_run_rows(rows)
    for record in records:
        print(
            "row_index={row_index} run_id={run_id} log_id={log_id} seed={seed} "
            "algorithm={algorithm_id} variant={algorithm_variant} "
            "output_path={output_path}".format(**record)
        )
    print(f"Matching rows: {len(records)}")


if __name__ == "__main__":
    main()
