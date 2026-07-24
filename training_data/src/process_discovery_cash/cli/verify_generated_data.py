from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from process_discovery_cash.data.inventory import (
    GeneratedInventoryError,
    verify_generated_inventory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the complete generated v6 data inventory and checksums."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override DATA_ROOT for this validation.",
    )
    parser.add_argument("--json", action="store_true", help="Print the compact receipt as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = verify_generated_inventory(data_root=args.data_root)
    except GeneratedInventoryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(
            "Generated v6 data verified: "
            f"real={receipt['real_logs']} "
            f"augmented={receipt['accepted_augmented_logs']} "
            f"targets={receipt['gedi_targets']} "
            f"synthetic={receipt['accepted_synthetic_logs']} "
            f"total={receipt['total_event_logs']} "
            f"artifacts={receipt['generated_artifact_count']}"
        )
        print(
            "Generated artifact receipt SHA-256: "
            f"{receipt['generated_artifact_receipt_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
