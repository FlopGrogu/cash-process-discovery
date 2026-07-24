from __future__ import annotations

import argparse
from pathlib import Path

from process_discovery_cash.hpo.study_manifest import (
    default_study_manifest_path,
    generate_hpo_study_rows,
    write_study_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an HPO study manifest (one row per log x algorithm study) "
            "from experiment configs with an 'hpo' block."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Experiment config YAML with an 'hpo' block. Repeatable.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Study manifest CSV path. Defaults to "
            "experiments/manifests/hpo/<experiment_id>/studies.csv (single config only)."
        ),
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Write each study manifest below this directory, preserving its path "
            "below experiments/manifests/v6. Supports multiple --config values."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.output and args.output_root:
        parser.error("--output and --output-root are mutually exclusive")
    if args.output_root:
        total = 0
        for config in args.config:
            rows = generate_hpo_study_rows([config])
            configured = default_study_manifest_path(config)
            try:
                relative = configured.relative_to("experiments/manifests/v6")
            except ValueError:
                parser.error(f"Not a canonical v6 config: {config}")
            manifest_path = write_study_manifest(
                rows,
                Path(args.output_root) / relative,
            )
            total += len(rows)
            print(f"Wrote {len(rows)} study rows to {manifest_path}")
        print(f"Wrote {total} total HPO study rows.")
        return

    output = args.output
    if output is None:
        if len(args.config) != 1:
            parser.error("--output is required when passing multiple --config files")
        output = default_study_manifest_path(args.config[0])

    rows = generate_hpo_study_rows(args.config)
    manifest_path = write_study_manifest(rows, output)
    print(f"Wrote {len(rows)} study rows to {manifest_path}")


if __name__ == "__main__":
    main()
