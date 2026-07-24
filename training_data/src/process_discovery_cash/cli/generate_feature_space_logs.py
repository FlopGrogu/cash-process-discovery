from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from process_discovery_cash.data.preprocessing.catalog import DEFAULT_DATASET_CATALOG
from process_discovery_cash.generation.aggregate import aggregate_results
from process_discovery_cash.generation.anchor import (
    ANCHOR_FILENAME,
    build_anchor_features,
)
from process_discovery_cash.generation.feature_space import design_bounds
from process_discovery_cash.generation.gedi_backend import GediBackend
from process_discovery_cash.generation.pipeline import (
    COVERAGE_FILENAME,
    MANIFEST_FILENAME,
    TARGETS_FILENAME,
    coverage_report,
    revalidate_existing,
    run_generation,
    summarize_records,
)
from process_discovery_cash.generation.targets import design_targets, targets_to_frame
from process_discovery_cash.utils.paths import resolve_portable_path

DEFAULT_OUTPUT_ROOT = "data/synthetic/gedi"
DEFAULT_RESULTS_DIR = "results/gedi"
PILOT_N_TARGETS = 36
MAIN_N_TARGETS = 200


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feature-space extension with GEDI. Modes: design (targets + anchor, "
            "for cluster batches), pilot/main (sequential local generation), "
            "aggregate (merge per-target cluster results), validate (revalidate "
            "accepted logs). Generated logs are training/validation data only, "
            "never final test logs."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "main", "design", "aggregate", "validate"],
        default="pilot",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=None,
        help=f"Target count (default {PILOT_N_TARGETS} pilot, {MAIN_N_TARGETS} design/main).",
    )
    parser.add_argument("--catalog", default=str(DEFAULT_DATASET_CATALOG))
    parser.add_argument(
        "--anchor-features",
        default=None,
        help=f"Anchor CSV (default: <output-root>/{ANCHOR_FILENAME}).",
    )
    parser.add_argument(
        "--compute-anchor",
        action="store_true",
        help=(
            "Compute missing anchor rows by parsing the raw logs (heavy; never on a "
            "cluster login node). Without it, missing rows are an error."
        ),
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Per-target result JSON directory (aggregate mode).",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--gedi-python", default=None, help="Sidecar GEDI interpreter.")
    parser.add_argument("--n-trials", type=int, default=50, help="SMAC trials per target.")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Per GEDI call.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Scratch directory for GEDI attempts (default <output-root>/work).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = resolve_portable_path(args.output_root)
    anchor_path = (
        resolve_portable_path(args.anchor_features)
        if args.anchor_features
        else output_root / ANCHOR_FILENAME
    )

    real_features = build_anchor_features(
        args.catalog, anchor_path, compute_missing=args.compute_anchor
    )
    print(f"Anchor: {len(real_features)} real logs from {anchor_path}")

    if args.mode == "validate":
        _run_validate(args, output_root, real_features)
        return
    if args.mode == "aggregate":
        _run_aggregate(args, output_root, real_features)
        return

    n_targets = args.n_targets or (PILOT_N_TARGETS if args.mode == "pilot" else MAIN_N_TARGETS)
    targets = design_targets(real_features, n_targets=n_targets, seed=args.seed)
    output_root.mkdir(parents=True, exist_ok=True)
    targets_frame = targets_to_frame(targets)
    targets_path = output_root / TARGETS_FILENAME
    targets_frame.to_csv(targets_path, index=False)
    feasible = int(targets_frame["feasible"].sum())
    print(
        f"Designed {len(targets)} targets ({feasible} feasible) "
        f"[{dict(targets_frame['band'].value_counts())}] -> {targets_path}"
    )
    if args.mode == "design":
        print(
            "Submit to the cluster with:\n"
            f"  bash scripts/submit_gedi_targets_slurm.sh --targets {targets_path} --all-rows\n"
            "then aggregate with:\n"
            f"  python scripts/generate_feature_space_logs.py --mode aggregate "
            f"--output-root {output_root}"
        )
        return

    backend = GediBackend(
        python_bin=args.gedi_python,
        n_trials=args.n_trials,
        timeout_seconds=args.timeout_seconds,
    )
    unavailable = backend.available()
    if unavailable:
        parser.error(unavailable)

    workdir = resolve_portable_path(args.workdir) if args.workdir else output_root / "work"
    records = run_generation(
        targets,
        real_features,
        backend,
        output_root=output_root,
        base_seed=args.seed,
        workdir=workdir,
        max_attempts=args.max_attempts,
        overwrite=args.overwrite,
    )
    _report(records, real_features, output_root)


def _run_validate(args, output_root: Path, real_features: pd.DataFrame) -> None:
    manifest_path = output_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise SystemExit(f"No manifest to validate at {manifest_path}")
    records = revalidate_existing(manifest_path, real_features)
    summary = summarize_records(records)
    print(f"Revalidated {len(records)} accepted logs: {summary.get('by_status')}")
    for record in records:
        if record.status != "accepted":
            print(f"  [now {record.status}] {record.log_id}: {record.rejection_reason}")


def _run_aggregate(args, output_root: Path, real_features: pd.DataFrame) -> None:
    targets_path = output_root / TARGETS_FILENAME
    known_target_ids = None
    if targets_path.exists():
        known_target_ids = set(pd.read_csv(targets_path)["target_id"].astype(str))
    records, info = aggregate_results(
        resolve_portable_path(args.results_dir),
        real_features,
        output_root=output_root,
        known_target_ids=known_target_ids,
    )
    print(f"Aggregated {info['n_result_files']} per-target result files.")
    if info.get("missing_target_ids"):
        print(
            f"WARNING: {len(info['missing_target_ids'])} targets have no result yet: "
            f"{', '.join(info['missing_target_ids'][:10])}"
            f"{'...' if len(info['missing_target_ids']) > 10 else ''}"
        )
    if info.get("unknown_target_ids"):
        print(
            "WARNING: results for target ids not in targets.csv "
            f"(stale batch?): {', '.join(info['unknown_target_ids'][:10])}"
        )
    _report(records, real_features, output_root)


def _report(records, real_features: pd.DataFrame, output_root: Path) -> None:
    summary = summarize_records(records)
    accepted = [record for record in records if record.status == "accepted"]
    bounds = design_bounds(real_features)
    coverage = coverage_report(
        real_features, [record.achieved_values for record in accepted], bounds
    )
    coverage_path = output_root / COVERAGE_FILENAME
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== Generation summary ===")
    print(f"Status counts: {summary.get('by_status')}")
    if "rejection_reasons" in summary:
        print(f"Rejection reasons: {summary['rejection_reasons']}")
    if accepted:
        print(
            f"Mean attainment error (accepted, <=1 required): "
            f"{summary['mean_attainment_error']:.3f}"
        )
        print(f"Achieved bands: {summary.get('band_achieved')}")
    real_cov = coverage["real_only"]["coverage"]
    combined_cov = coverage["real_plus_generated"]["coverage"]
    print(
        f"Pairwise binned coverage: real-only {real_cov:.3f} -> "
        f"real+generated {combined_cov:.3f} "
        f"({coverage['real_plus_generated']['occupied_cells']}/"
        f"{coverage['real_plus_generated']['total_cells']} cells)"
    )
    print("Feature-space holes (real vs generated):")
    for name, counts in coverage["holes"].items():
        print(f"  {name}: real={counts['real']} generated={counts['generated']}")
    print(f"Logs:     {output_root / 'logs'}")
    print(f"Manifest: {output_root / MANIFEST_FILENAME}")
    print(f"Coverage: {coverage_path}")


if __name__ == "__main__":
    main()
