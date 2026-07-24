from __future__ import annotations

import argparse
import sys

from process_discovery_cash.experiments.metric_manifest import (
    run_metric_manifest_index,
    run_metric_slurm_array_task,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one metric job from a metric manifest.")
    parser.add_argument("--manifest", required=True, help="Metric manifest CSV path.")
    parser.add_argument("--row-index", type=int, help="Zero-based metric manifest row index.")
    parser.add_argument(
        "--slurm-array-task-id",
        action="store_true",
        help="Use SLURM_ARRAY_TASK_ID plus METRIC_MANIFEST_ROW_OFFSET as the row index.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run rows even when their successful metric output already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command_args = sys.argv if argv is None else ["pdcash-run-metric", *argv]

    if args.slurm_array_task_id:
        output_path = run_metric_slurm_array_task(
            args.manifest,
            command_args=command_args,
            force=args.force,
        )
    else:
        row_index = 0 if args.row_index is None else args.row_index
        output_path = run_metric_manifest_index(
            args.manifest,
            row_index,
            command_args=command_args,
            force=args.force,
        )
    print(f"Metric result path: {output_path}")


if __name__ == "__main__":
    main()
