from __future__ import annotations

import argparse

from process_discovery_cash.experiments.aggregation import aggregate_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate individual JSON result files.")
    parser.add_argument(
        "--results-root",
        "--input",
        dest="results_root",
        required=True,
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--output",
        default="results/aggregated/results.csv",
        help="Output CSV path.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = aggregate_results(args.results_root, args.output)
    print(f"Wrote aggregated results to {output_path}")


if __name__ == "__main__":
    main()
