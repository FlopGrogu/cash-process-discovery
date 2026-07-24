from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from process_discovery_cash.experiments.receipts import (
    RECEIPT_SCOPES,
    verify_v6_receipts,
    write_v6_receipt_ledger,
)
from process_discovery_cash.experiments.v6 import (
    generate_all_v6_ordinary_manifests,
    generate_v6_default_run_survey_manifests,
    generate_v6_primary_manifests,
)
from process_discovery_cash.hpo.study_manifest import (
    generate_hpo_study_rows,
    write_study_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify the v6 manifest receipt ledger.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="Update the committed ledger.")
    action.add_argument("--check", action="store_true", help="Verify the committed ledger.")
    parser.add_argument(
        "--manifest-root",
        help="Existing generated root below experiments/manifests/v6.",
    )
    parser.add_argument(
        "--scope",
        choices=sorted(RECEIPT_SCOPES),
        help=(
            "Receipt scope. Checks default to primary; ledger writes default to all."
        ),
    )
    return parser


def _generate_hpo(root: Path) -> None:
    for config in sorted(Path.cwd().glob("configs/experiments/v6/hpo/*/*.yaml")):
        relative = Path("hpo") / config.parent.name / config.stem / "studies.csv"
        write_study_manifest(generate_hpo_study_rows([config]), root / relative)


def _generate_scope(root: Path, scope: str) -> None:
    if scope == "primary":
        generate_v6_primary_manifests(output_root=root)
    elif scope == "survey":
        generate_v6_default_run_survey_manifests(output_root=root)
    elif scope == "hpo":
        _generate_hpo(root)
    elif scope == "all":
        generate_all_v6_ordinary_manifests(output_root=root)
        _generate_hpo(root)
    else:  # pragma: no cover - argparse and public validation guard this
        raise ValueError(f"Unknown receipt scope: {scope}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    scope = args.scope or ("all" if args.write else "primary")
    scopes = {scope}
    if args.write and scope != "all":
        parser.error("--write requires --scope all")
    if args.manifest_root:
        root = Path(args.manifest_root)
        if args.write:
            print(f"Wrote receipt ledger to {write_v6_receipt_ledger(root)}")
        else:
            verify_v6_receipts(root, scopes=scopes)
            print(f"All v6 manifest receipts match for scope {scope}.")
        return

    with tempfile.TemporaryDirectory(prefix="pdcash-v6-receipts-") as temporary:
        root = Path(temporary)
        _generate_scope(root, scope)
        if args.write:
            print(f"Wrote receipt ledger to {write_v6_receipt_ledger(root)}")
        else:
            verify_v6_receipts(root, scopes=scopes)
            print(f"All v6 manifest receipts match for scope {scope}.")


if __name__ == "__main__":
    main()
