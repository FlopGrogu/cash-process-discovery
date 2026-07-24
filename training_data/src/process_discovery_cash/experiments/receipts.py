"""Compact, deterministic receipts for generated v6 manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.experiments.v6 import discover_v6_ordinary_configs
from process_discovery_cash.hpo.study_manifest import default_study_manifest_path
from process_discovery_cash.utils.paths import project_root

DEFAULT_RECEIPT_LEDGER = Path("release/v6-manifest-receipts.json")
HPO_CONFIG_GLOB = "configs/experiments/v6/hpo/*/*.yaml"
RECEIPT_SCOPES = frozenset({"primary", "survey", "hpo", "all"})
GENERATOR_SOURCES = (
    "src/process_discovery_cash/config/load.py",
    "src/process_discovery_cash/config/schema.py",
    "src/process_discovery_cash/experiments/identity.py",
    "src/process_discovery_cash/experiments/manifest.py",
    "src/process_discovery_cash/experiments/v6.py",
    "src/process_discovery_cash/hpo/study_manifest.py",
)


def canonical_config_paths() -> list[Path]:
    return [
        *discover_v6_ordinary_configs(),
        *sorted(project_root().glob(HPO_CONFIG_GLOB)),
    ]


def manifest_path_for_config(config_path: str | Path) -> tuple[str, Path]:
    path = Path(config_path)
    experiment = load_experiment_config(path)
    if experiment.hpo is not None:
        return "hpo-study", default_study_manifest_path(path)
    if not experiment.manifest_path:
        raise ValueError(f"v6 config has no manifest_path: {path}")
    return "ordinary", Path(experiment.manifest_path)


def receipt_scope_for_config(config_path: str | Path) -> str:
    path = Path(config_path)
    relative = path.relative_to(project_root() / "configs/experiments/v6")
    family = relative.parts[0]
    if family in {"baseline", "explore", "explore_synthetic"}:
        return "primary"
    if family == "default_run_survey":
        return "survey"
    if family == "hpo":
        return "hpo"
    raise ValueError(f"Unsupported v6 receipt family: {family}")


def manifest_path_below_root(expected_path: str | Path, root: str | Path) -> Path:
    expected = Path(expected_path)
    marker = ("experiments", "manifests", "v6")
    for index in range(len(expected.parts) - len(marker) + 1):
        if expected.parts[index : index + len(marker)] == marker:
            return Path(root).joinpath(*expected.parts[index + len(marker) :])
    raise ValueError(f"Expected a v6 manifest path, got: {expected}")


def build_v6_receipt_ledger(manifest_root: str | Path) -> dict[str, Any]:
    root = project_root()
    generator_sha256 = _combined_hash(root / path for path in GENERATOR_SOURCES)
    receipts = []
    for config_path in canonical_config_paths():
        kind, expected_path = manifest_path_for_config(config_path)
        scope = receipt_scope_for_config(config_path)
        generated_path = manifest_path_below_root(expected_path, manifest_root)
        if not generated_path.is_file():
            raise FileNotFoundError(f"Generated manifest is missing: {generated_path}")
        receipts.append(
            {
                "config_path": config_path.relative_to(root).as_posix(),
                "config_sha256": _sha256(config_path),
                "generator_sha256": generator_sha256,
                "kind": kind,
                "scope": scope,
                "expected_manifest_path": expected_path.as_posix(),
                "row_count": _csv_row_count(generated_path),
                "sha256": _sha256(generated_path),
            }
        )
    return {
        "schema_version": 2,
        "canonical_config_count": len(receipts),
        "ordinary_manifest_count": sum(row["kind"] == "ordinary" for row in receipts),
        "primary_manifest_count": sum(row["scope"] == "primary" for row in receipts),
        "survey_manifest_count": sum(row["scope"] == "survey" for row in receipts),
        "hpo_study_manifest_count": sum(row["kind"] == "hpo-study" for row in receipts),
        "receipts": receipts,
    }


def write_v6_receipt_ledger(
    manifest_root: str | Path,
    output_path: str | Path = DEFAULT_RECEIPT_LEDGER,
) -> Path:
    destination = project_root() / output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_v6_receipt_ledger(manifest_root)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def verify_v6_receipts(
    manifest_root: str | Path,
    ledger_path: str | Path = DEFAULT_RECEIPT_LEDGER,
    *,
    kinds: set[str] | None = None,
    scopes: set[str] | None = None,
) -> None:
    if scopes is not None:
        invalid_scopes = scopes - RECEIPT_SCOPES
        if invalid_scopes:
            raise ValueError(f"Unknown receipt scopes: {', '.join(sorted(invalid_scopes))}")
        if "all" in scopes:
            scopes = None
    ledger = json.loads((project_root() / ledger_path).read_text(encoding="utf-8"))
    root = project_root()
    expected_generator_hash = _combined_hash(root / path for path in GENERATOR_SOURCES)
    errors: list[str] = []
    checked = 0
    for receipt in ledger["receipts"]:
        if kinds is not None and receipt["kind"] not in kinds:
            continue
        if scopes is not None and receipt.get("scope") not in scopes:
            continue
        checked += 1
        config_path = root / receipt["config_path"]
        manifest_path = manifest_path_below_root(
            receipt["expected_manifest_path"],
            manifest_root,
        )
        checks = {
            "config_sha256": _sha256(config_path) if config_path.is_file() else "missing",
            "generator_sha256": expected_generator_hash,
            "row_count": _csv_row_count(manifest_path) if manifest_path.is_file() else -1,
            "sha256": _sha256(manifest_path) if manifest_path.is_file() else "missing",
        }
        for field, actual in checks.items():
            if actual != receipt[field]:
                errors.append(
                    f"{receipt['expected_manifest_path']}: {field} "
                    f"expected {receipt[field]!r}, got {actual!r}"
                )
    if not checked:
        raise ValueError("No receipt rows selected")
    if errors:
        raise ValueError("v6 manifest receipt mismatch:\n" + "\n".join(errors))


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_hash(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(project_root()).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()
