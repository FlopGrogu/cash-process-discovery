from __future__ import annotations

import argparse
from pathlib import Path

from process_discovery_cash.generation.anchor import (
    ANCHOR_FILENAME,
    load_anchor_features,
)
from process_discovery_cash.generation.feature_space import TARGET_FEATURES
from process_discovery_cash.generation.gedi_backend import GediBackend
from process_discovery_cash.generation.pipeline import run_single_target
from process_discovery_cash.generation.results import (
    build_target_result,
    load_target_result,
    result_path_for,
    write_target_result,
)
from process_discovery_cash.generation.targets import targets_from_frame

DEFAULT_OUTPUT_ROOT = "data/synthetic/gedi"
DEFAULT_RESULTS_DIR = "results/gedi"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run GEDI log synthesis for exactly one target row of a targets.csv. "
            "One Slurm array task executes one row; the per-target result JSON is "
            "the terminal marker for reruns."
        )
    )
    parser.add_argument("--targets", required=True, help="targets.csv from --mode design.")
    parser.add_argument("--row-index", type=int, required=True, help="0-based data row.")
    parser.add_argument(
        "--anchor-features",
        default=None,
        help=f"Anchor CSV (default: {ANCHOR_FILENAME} next to the targets file).",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--base-seed", type=int, default=2024)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--gedi-python", default=None)
    parser.add_argument("--workdir", default=None, help="Scratch dir (default <output-root>/work).")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    backend = GediBackend(
        python_bin=args.gedi_python,
        n_trials=args.n_trials,
        timeout_seconds=args.timeout_seconds,
    )
    run_target_row(args, parser=parser, backend=backend)


def run_target_row(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    backend,
) -> None:
    import pandas as pd

    targets_path = Path(args.targets)
    if not targets_path.exists():
        parser.error(f"Targets file not found: {targets_path}")
    frame = pd.read_csv(targets_path)
    if not 0 <= args.row_index < len(frame):
        parser.error(
            f"--row-index {args.row_index} out of range for {targets_path} "
            f"({len(frame)} data rows)."
        )
    target = targets_from_frame(frame.iloc[[args.row_index]])[0]

    result_path = result_path_for(args.results_dir, target.target_id)
    existing = load_target_result(result_path)
    if existing is not None and not args.overwrite:
        print(
            f"SKIP target_id={target.target_id} row_index={args.row_index} "
            f"terminal_status={existing.get('terminal_status')} ({result_path})"
        )
        return

    anchor_path = (
        Path(args.anchor_features)
        if args.anchor_features
        else targets_path.parent / ANCHOR_FILENAME
    )
    real_features = load_anchor_features(anchor_path)
    missing = [feature for feature in TARGET_FEATURES if feature not in real_features.columns]
    if real_features.empty or missing:
        parser.error(
            f"Anchor features at {anchor_path} are missing or incomplete "
            f"(missing columns: {', '.join(missing) or 'all'}). "
            "Generate them with --mode design."
        )

    unavailable = backend.available() if hasattr(backend, "available") else None
    if unavailable:
        parser.error(unavailable)

    output_root = Path(args.output_root)
    workdir = Path(args.workdir) if args.workdir else output_root / "work"
    records = run_single_target(
        target,
        real_features,
        backend,
        output_root=output_root,
        base_seed=args.base_seed,
        workdir=workdir,
        max_attempts=args.max_attempts,
        overwrite=args.overwrite,
    )
    payload = build_target_result(
        target.target_id,
        records,
        row_index=args.row_index,
        base_seed=args.base_seed,
        n_trials=args.n_trials,
        max_attempts=args.max_attempts,
    )
    write_target_result(result_path, payload)
    last = records[-1]
    print(
        f"DONE target_id={target.target_id} row_index={args.row_index} "
        f"status={last.status} attempts={len(records)} "
        f"result={result_path} output={last.output_path}"
    )


if __name__ == "__main__":
    main()
