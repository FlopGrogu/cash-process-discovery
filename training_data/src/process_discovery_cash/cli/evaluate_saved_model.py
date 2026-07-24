from __future__ import annotations

import argparse
from pathlib import Path

from process_discovery_cash.evaluation.quality_metrics import DEFAULT_METRICS
from process_discovery_cash.experiments.saved_model_metrics import (
    evaluate_saved_result_model,
    evaluate_saved_result_tree,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate metrics for a model artifact exported by a discovery result."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result", help="Single discovery result JSON.")
    source.add_argument("--results-root", help="Directory containing discovery result JSON files.")
    parser.add_argument(
        "--output",
        help="Output metrics JSON path for --result. Defaults next to the source result.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory for --results-root. Defaults to "
            "results/metrics/<results-root-name>_<profile>."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["pm4py_default", "token", "alignment"],
        default="alignment",
        help="Metric profile to compute. Default: alignment.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric names to compute. Default: all quality metrics.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics even when the target metrics JSON already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.result:
        output_path = evaluate_saved_result_model(
            args.result,
            metric_profile=args.profile,
            metric_names=args.metrics,
            output_path=args.output,
            force=args.force,
        )
        print(f"Wrote saved-model metrics to {output_path}")
        return

    results_root = Path(args.results_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/metrics") / f"{results_root.name}_{args.profile}"
    )
    written_paths = evaluate_saved_result_tree(
        results_root,
        output_dir,
        metric_profile=args.profile,
        metric_names=args.metrics,
        force=args.force,
    )
    print(f"Wrote {len(written_paths)} saved-model metric files to {output_dir}")


if __name__ == "__main__":
    main()
