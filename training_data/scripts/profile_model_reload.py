from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from process_discovery_cash.experiments.saved_model_metrics import (
    _load_json,
    _load_model_artifact,
    _load_model_artifact_with_fallback,
    _resolve_metric_model_path,
    _resolve_model_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile XML model reload versus metric sidecar reload for one result JSON."
    )
    parser.add_argument(
        "--result", required=True, help="Path to a successful discovery result JSON."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of load iterations per backend. Default: 5.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result_path = Path(args.result)
    payload = _load_json(result_path)
    model_path = _resolve_model_path(payload, result_path)
    metric_model_path = _resolve_metric_model_path(payload, result_path)

    xml_times = _profile(args.iterations, lambda: _load_model_artifact(model_path))
    report = {
        "result_path": result_path.as_posix(),
        "model_path": model_path.as_posix(),
        "xml_backend": model_path.suffix.lower().lstrip("."),
        "xml_seconds": _summary(xml_times),
    }

    if metric_model_path is not None and metric_model_path.exists():
        fast_times = _profile(
            args.iterations,
            lambda: _load_model_artifact_with_fallback(
                model_path,
                metric_model_path=metric_model_path,
            ),
        )
        report["metric_model_path"] = metric_model_path.as_posix()
        report["fast_seconds"] = _summary(fast_times)
    else:
        report["metric_model_path"] = None
        report["fast_seconds"] = None

    print(json.dumps(report, indent=2, sort_keys=True))


def _profile(iterations: int, loader) -> list[float]:
    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        loader()
        timings.append(time.perf_counter() - started)
    return timings


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


if __name__ == "__main__":
    main()
