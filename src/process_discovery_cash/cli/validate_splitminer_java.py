from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree

from process_discovery_cash.data.preprocessing.catalog import get_dataset
from process_discovery_cash.data.preprocessing.selection import select_discovery_artifact
from process_discovery_cash.discovery.split import SplitMiner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Split Miner Java and a preprocessed XES artifact."
    )
    parser.add_argument("--dataset", default="bpi2012")
    parser.add_argument("--catalog", default="configs/datasets/processmining_org.yaml")
    parser.add_argument("--jar", default=os.getenv("SPLIT_MINER_JAR"))
    parser.add_argument("--java-bin", default=os.getenv("JAVA_BIN", "java"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run Split Miner v1 after static validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = get_dataset(args.dataset, args.catalog)
    selection = select_discovery_artifact(
        log_id=dataset.dataset_id,
        dataset_id=dataset.dataset_id,
        path=dataset.source_path,
        algorithm_id="split_miner",
        catalog_path=args.catalog,
    )
    if not selection.discovery_log_path:
        raise SystemExit("Split Miner v1 artifact is missing; run preprocess_event_logs.py first.")

    counts = _lifecycle_counts(Path(selection.discovery_log_path))
    non_complete = {
        transition: count for transition, count in counts.items() if transition != "complete"
    }
    if not counts.get("complete") or non_complete:
        raise SystemExit(f"Artifact is not a minimal complete-event XES log: {counts}")

    version = subprocess.run(
        [args.java_bin, "-version"],
        text=True,
        capture_output=True,
        check=False,
    )
    result = {
        "dataset_id": args.dataset,
        "artifact": selection.discovery_log_path,
        "lifecycle_counts": counts,
        "java_return_code": version.returncode,
        "java_version": (version.stderr or version.stdout).strip(),
    }
    if args.execute:
        if not args.jar:
            raise SystemExit("--jar or SPLIT_MINER_JAR is required with --execute.")
        discovery = SplitMiner().discover(
            None,
            {
                "jar_path": args.jar,
                "java_bin": args.java_bin,
                "input_log_path": selection.discovery_log_path,
                "input_artifact_kind": "splitminer_v1_xes",
                "keep_output_files": False,
            },
        )
        result["discovery"] = discovery.to_json_record()
        if discovery.status != "success":
            print(json.dumps(result, indent=2, default=str))
            raise SystemExit(1)
    print(json.dumps(result, indent=2, default=str))


def _lifecycle_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    opener = gzip.open if path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rb") as handle:
        for _event, element in ElementTree.iterparse(handle, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] != "event":
                continue
            transition = next(
                (
                    child.attrib.get("value", "").lower()
                    for child in element
                    if child.attrib.get("key") == "lifecycle:transition"
                ),
                "<missing>",
            )
            counts[transition] = counts.get(transition, 0) + 1
            element.clear()
    return counts


if __name__ == "__main__":
    main()
