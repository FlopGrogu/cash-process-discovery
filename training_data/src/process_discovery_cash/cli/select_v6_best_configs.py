from __future__ import annotations

import argparse

from process_discovery_cash.experiments.v6 import (
    DEFAULT_V6_OBJECTIVE_METRICS,
    select_best_v6_configs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select best v6 hyperparameter configurations per log/algorithm pair."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more aggregated or joined CSV result files.",
    )
    parser.add_argument(
        "--output",
        default="results/aggregated/v6/best_configs.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--objective-metrics",
        nargs="+",
        default=list(DEFAULT_V6_OBJECTIVE_METRICS),
        help="Metric columns to average for the balanced objective.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = select_best_v6_configs(
        args.inputs,
        args.output,
        objective_metrics=args.objective_metrics,
    )
    print(f"Wrote v6 best configs to {output_path}")


if __name__ == "__main__":
    main()
