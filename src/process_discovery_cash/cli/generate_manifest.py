from __future__ import annotations

import argparse

from process_discovery_cash.experiments.manifest import generate_manifest, generate_manifest_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deterministic experiment manifest.")
    parser.add_argument(
        "--experiment-config",
        "--config",
        dest="experiment_config",
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument("--output", help="Output manifest CSV path.")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help=(
            "Deprecated compatibility flag; ignored. Manifests reference source "
            "XES files and use optional runtime-local caches."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = generate_manifest_rows(
        args.experiment_config,
        require_artifacts=args.require_artifacts,
    )
    output_path = generate_manifest(
        args.experiment_config,
        args.output,
        require_artifacts=args.require_artifacts,
    )
    print(f"Wrote {len(rows)} manifest rows to {output_path}")


if __name__ == "__main__":
    main()
