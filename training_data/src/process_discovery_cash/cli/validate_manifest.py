from __future__ import annotations

import argparse
from pathlib import Path

from process_discovery_cash.experiments.manifest_validation import validate_manifest_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an experiment manifest CSV.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV path.")
    parser.add_argument(
        "--project-root",
        help="Repository root used to validate absolute paths and output parents.",
    )
    parser.add_argument(
        "--check-output-parents",
        action="store_true",
        help="Create/check output parent directories referenced by manifest rows.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = validate_manifest_file(
        args.manifest,
        project_root=args.project_root,
        check_output_parents=args.check_output_parents,
    )
    if result.ok:
        print(f"Manifest OK: {Path(args.manifest)} ({result.row_count} rows)")
        return
    for issue in result.issues:
        print(f"ERROR: {issue.format()}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
