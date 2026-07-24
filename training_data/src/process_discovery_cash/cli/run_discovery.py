from __future__ import annotations

import argparse
import sys

from process_discovery_cash.experiments.runner import (
    load_manifest_rows,
    run_manifest_index,
    run_manifest_row,
    run_slurm_array_task,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one discovery experiment from a manifest.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV path.")
    parser.add_argument("--row-index", type=int, help="Zero-based manifest row index.")
    parser.add_argument(
        "--slurm-array-task-id",
        action="store_true",
        help="Use SLURM_ARRAY_TASK_ID as the manifest row index.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all manifest rows sequentially for local debugging.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run rows even when their success output already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command_args = sys.argv if argv is None else ["pdcash-run-discovery", *argv]

    if args.all:
        rows = load_manifest_rows(args.manifest)
        for row in rows:
            output_path = run_manifest_row(row, command_args=command_args, force=args.force)
            print(f"Result path: {output_path}")
        return

    if args.slurm_array_task_id:
        output_path = run_slurm_array_task(
            args.manifest,
            command_args=command_args,
            force=args.force,
        )
    else:
        row_index = 0 if args.row_index is None else args.row_index
        output_path = run_manifest_index(
            args.manifest,
            row_index,
            command_args=command_args,
            force=args.force,
        )
    print(f"Result path: {output_path}")


if __name__ == "__main__":
    main()
