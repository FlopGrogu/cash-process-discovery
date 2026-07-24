from __future__ import annotations

import argparse

from process_discovery_cash.hpo.export_manifest import export_hpo_discovery_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a discovery-style manifest from an HPO experiment's trial result "
            "files (one row per evaluated configuration). The export feeds the "
            "standard metric workflow: pdcash-generate-metric-manifest "
            "--source-manifest <export>."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="HPO experiment config YAML.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Manifest CSV path. Defaults to the experiment's manifest_path "
            "(e.g. experiments/manifests/v6/model/hpo/<algorithm>/v1.csv)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest_path, stats = export_hpo_discovery_manifest(args.config, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {stats.exported} manifest rows to {manifest_path}")
    if stats.skipped_other_algorithm:
        print(f"Skipped {stats.skipped_other_algorithm} result(s) from other algorithms")
    if stats.skipped_unreadable:
        print(f"Skipped {stats.skipped_unreadable} unreadable result file(s)")
    if stats.skipped_hash_mismatch:
        print(
            f"Skipped {stats.skipped_hash_mismatch} result(s) with stale config hashes "
            "(see log warnings)"
        )


if __name__ == "__main__":
    main()
