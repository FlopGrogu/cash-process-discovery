from __future__ import annotations

import argparse
from pathlib import Path

from process_discovery_cash.evaluation.quality_metrics import DEFAULT_METRICS
from process_discovery_cash.experiments.metric_manifest import (
    generate_metric_manifest_from_source_manifest,
)
from process_discovery_cash.experiments.metric_timeout import DEFAULT_METRIC_TIMEOUT_SECONDS

DEFAULT_SOURCE_MANIFEST_PROFILE = "token"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a metric manifest from a discovery source manifest. Every row "
            "of the source manifest is preserved: one metric row is emitted per "
            "source row. Rows whose discovery result is missing, unreadable, failed, "
            "timed out, or has no exported model are still included and evaluate to "
            "zero-valued metrics at run time rather than being dropped."
        )
    )
    parser.add_argument(
        "--source-manifest",
        required=True,
        help=(
            "Discovery experiment manifest CSV whose output_path rows identify the "
            "discovery result JSON files. This is the only supported input; the "
            "metric manifest always mirrors it one row per row."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["pm4py_default", "token", "alignment"],
        default=None,
        help=f"Metric profile to calculate. Default: {DEFAULT_SOURCE_MANIFEST_PROFILE}.",
    )
    parser.add_argument(
        "--output",
        help=("Output metric manifest CSV path. Defaults to the corresponding v6 metrics path."),
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Metric result output root. Defaults to the corresponding "
            "results/cluster/v6/metrics path."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric names to compute. Default: all quality metrics.",
    )
    parser.add_argument(
        "--metric-timeout-seconds",
        type=float,
        default=DEFAULT_METRIC_TIMEOUT_SECONDS,
        help=(
            "Per-row metric evaluation timeout in seconds written to the "
            "metric_timeout_seconds column. Pass 0 to disable the timeout. "
            f"Default: {DEFAULT_METRIC_TIMEOUT_SECONDS}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    metric_profile = args.profile or DEFAULT_SOURCE_MANIFEST_PROFILE
    try:
        output_path = args.output or default_metric_manifest_path_from_source_manifest(
            args.source_manifest,
            metric_profile,
        )
        output_root = args.output_root or default_metric_output_root_from_source_manifest(
            args.source_manifest,
            metric_profile,
        )
    except ValueError as exc:
        parser.error(str(exc))
    metric_timeout_seconds = (
        args.metric_timeout_seconds if args.metric_timeout_seconds > 0 else None
    )
    output_path, stats = generate_metric_manifest_from_source_manifest(
        args.source_manifest,
        metric_profile=metric_profile,
        output_path=output_path,
        metric_names=args.metrics,
        output_root=output_root,
        metric_timeout_seconds=metric_timeout_seconds,
    )
    print(f"Wrote {stats.total} metric manifest rows to {output_path}")
    print(f"Rows with an exported model: {stats.full}")
    print(f"Fallback (zero-metric) rows: {stats.total - stats.full}")
    print(f"  missing discovery result: {stats.fallback_missing_result}")
    print(f"  malformed discovery JSON: {stats.fallback_malformed_json}")
    print(f"  failed/timed-out discovery: {stats.fallback_failed_discovery}")
    print(f"  missing exported model: {stats.fallback_missing_model}")


def default_metric_manifest_path_from_source_manifest(
    source_manifest_path: str | Path,
    metric_profile: str,
) -> Path:
    path = Path(source_manifest_path)
    v6_relative_parts = _relative_parts_after_marker(
        path,
        ("experiments", "manifests", "v6", "model"),
    )
    if v6_relative_parts is not None:
        if len(v6_relative_parts) < 3 or path.suffix != ".csv":
            raise ValueError(
                "source manifest defaults require a path like "
                "experiments/manifests/v6/model/<scope>/<algorithm>/v1.1.csv"
            )
        return Path("experiments/manifests/v6/metrics").joinpath(
            *v6_relative_parts[:-1],
            path.stem,
            f"{metric_profile}_metrics.csv",
        )

    raise ValueError(
        "source manifest defaults require a path like "
        "experiments/manifests/v6/model/<scope>/<algorithm>/v1.1.csv"
    )


def default_metric_output_root_from_source_manifest(
    source_manifest_path: str | Path,
    metric_profile: str,
) -> Path:
    path = Path(source_manifest_path)
    v6_relative_parts = _relative_parts_after_marker(
        path,
        ("experiments", "manifests", "v6", "model"),
    )
    if v6_relative_parts is not None:
        if len(v6_relative_parts) < 3 or path.suffix != ".csv":
            raise ValueError(
                "source manifest defaults require a path like "
                "experiments/manifests/v6/model/<scope>/<algorithm>/v1.1.csv"
            )
        return Path("results/cluster/v6/metrics").joinpath(
            *v6_relative_parts[:-1],
            path.stem,
            metric_profile,
        )

    raise ValueError(
        "source manifest defaults require a path like "
        "experiments/manifests/v6/model/<scope>/<algorithm>/v1.1.csv"
    )


def _relative_parts_after_marker(
    path: Path,
    marker: tuple[str, ...],
) -> tuple[str, ...] | None:
    parts = path.parts
    marker_length = len(marker)
    for index in range(len(parts) - marker_length + 1):
        if parts[index : index + marker_length] == marker:
            return tuple(parts[index + marker_length :])
    return None


if __name__ == "__main__":
    main()
