from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from process_discovery_cash.experiments.v6 import (
    DEFAULT_V6_AUGMENTATION_MANIFEST,
    DEFAULT_V6_AUGMENTED_EXPLORE_CONFIG_ROOT,
    DEFAULT_V6_BASELINE_CONFIG_GLOB,
    DEFAULT_V6_DEFAULT_RUN_SURVEY_CONFIG_GLOB,
    DEFAULT_V6_EXPLORE_CONFIG_GLOB,
    DEFAULT_V6_SYNTHETIC_EXPLORE_CONFIG_ROOT,
    DEFAULT_V6_SYNTHETIC_MANIFEST,
    generate_all_v6_ordinary_manifests,
    generate_v6_augmented_explore_manifests,
    generate_v6_default_run_survey_manifests,
    generate_v6_manifests,
    generate_v6_primary_manifests,
    generate_v6_synthetic_explore_manifests,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate v6 real-log experiment manifests.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all",
        action="store_true",
        help="Generate all 40 canonical ordinary v6 manifests.",
    )
    mode.add_argument(
        "--primary",
        action="store_true",
        help=(
            "Generate the 30 primary manifests: baseline, augmented explore, "
            "and synthetic explore."
        ),
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Write below this directory, preserving the path below "
            "experiments/manifests/v6. Recommended for reviews and tests."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate all ordinary manifests and verify the committed receipt ledger.",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help=(
            "Deprecated compatibility flag; ignored. Manifests reference source "
            "XES files and use optional runtime-local caches."
        ),
    )
    parser.add_argument(
        "--config-glob",
        default=DEFAULT_V6_BASELINE_CONFIG_GLOB,
        help=(
            "Glob for v6 per-algorithm experiment YAMLs. Default-run survey and "
            "explore modes automatically select their own default globs."
        ),
    )
    mode.add_argument(
        "--default-run-survey",
        action="store_true",
        help=(
            "Generate the v6 discovery-only runtime and failure survey: all 21 real "
            "logs with one default configuration for each algorithm variant."
        ),
    )
    mode.add_argument(
        "--augmented-explore",
        action="store_true",
        help=(
            "Generate manifests for accepted augmented logs using the same per-algorithm "
            "configs as v6/explore."
        ),
    )
    parser.add_argument(
        "--augmentation-manifest",
        default=str(DEFAULT_V6_AUGMENTATION_MANIFEST),
        help="Augmentation manifest CSV containing accepted child logs.",
    )
    parser.add_argument(
        "--augmented-config-root",
        default=str(DEFAULT_V6_AUGMENTED_EXPLORE_CONFIG_ROOT),
        help="Directory where derived augmented v6/explore YAML configs are written.",
    )
    parser.add_argument(
        "--exclude-stress",
        action="store_true",
        help="Exclude accepted augmented rows whose stress column is truthy.",
    )
    mode.add_argument(
        "--synthetic-explore",
        action="store_true",
        help=(
            "Generate manifests for accepted GEDI synthetic logs using the same "
            "per-algorithm configs as v6/explore."
        ),
    )
    parser.add_argument(
        "--synthetic-manifest",
        default=str(DEFAULT_V6_SYNTHETIC_MANIFEST),
        help="GEDI synthesis manifest CSV containing accepted synthetic logs.",
    )
    parser.add_argument(
        "--synthetic-config-root",
        default=str(DEFAULT_V6_SYNTHETIC_EXPLORE_CONFIG_ROOT),
        help="Directory where derived synthetic v6/explore YAML configs are written.",
    )
    parser.add_argument(
        "--require-log-files",
        action="store_true",
        help="Require every accepted child log file to exist before writing manifests.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check:
        from process_discovery_cash.experiments.receipts import verify_v6_receipts

        if not args.all:
            parser.error("--check requires --all")
        if args.default_run_survey or args.augmented_explore or args.synthetic_explore:
            parser.error("--check cannot be combined with a mode flag")
        if args.output_root:
            root = Path(args.output_root)
            generate_all_v6_ordinary_manifests(
                require_artifacts=args.require_artifacts,
                output_root=root,
            )
            verify_v6_receipts(root, kinds={"ordinary"})
        else:
            with tempfile.TemporaryDirectory(prefix="pdcash-v6-manifests-") as temporary:
                root = Path(temporary)
                generate_all_v6_ordinary_manifests(
                    require_artifacts=args.require_artifacts,
                    output_root=root,
                )
                verify_v6_receipts(root, kinds={"ordinary"})
        print("All v6 manifest receipts match.")
        return

    if args.primary:
        written = generate_v6_primary_manifests(
            require_artifacts=args.require_artifacts,
            output_root=args.output_root,
        )
    elif args.default_run_survey:
        config_glob = (
            DEFAULT_V6_DEFAULT_RUN_SURVEY_CONFIG_GLOB
            if args.config_glob == DEFAULT_V6_BASELINE_CONFIG_GLOB
            else args.config_glob
        )
        written = generate_v6_default_run_survey_manifests(
            require_artifacts=args.require_artifacts,
            config_glob=config_glob,
            output_root=args.output_root,
        )
    elif args.synthetic_explore:
        config_glob = (
            DEFAULT_V6_EXPLORE_CONFIG_GLOB
            if args.config_glob == DEFAULT_V6_BASELINE_CONFIG_GLOB
            else args.config_glob
        )
        written = generate_v6_synthetic_explore_manifests(
            synthetic_manifest=args.synthetic_manifest,
            require_artifacts=args.require_artifacts,
            source_config_glob=config_glob,
            output_config_root=args.synthetic_config_root,
            require_log_files=args.require_log_files,
        )
    elif args.augmented_explore:
        config_glob = (
            DEFAULT_V6_EXPLORE_CONFIG_GLOB
            if args.config_glob == DEFAULT_V6_BASELINE_CONFIG_GLOB
            else args.config_glob
        )
        written = generate_v6_augmented_explore_manifests(
            augmentation_manifest=args.augmentation_manifest,
            require_artifacts=args.require_artifacts,
            source_config_glob=config_glob,
            output_config_root=args.augmented_config_root,
            include_stress=not args.exclude_stress,
            require_log_files=args.require_log_files,
        )
    elif args.all:
        written = generate_all_v6_ordinary_manifests(
            require_artifacts=args.require_artifacts,
            output_root=args.output_root,
        )
    else:
        written = generate_v6_manifests(
            require_artifacts=args.require_artifacts,
            config_glob=args.config_glob,
            output_root=args.output_root,
        )
    for algorithm_id, path in written.items():
        print(f"Wrote v6 {algorithm_id} manifest to {path}")


if __name__ == "__main__":
    main()
