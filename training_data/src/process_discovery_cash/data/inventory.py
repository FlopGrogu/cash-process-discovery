"""Independent inventory and checksum validation for generated v6 data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from process_discovery_cash.utils.paths import data_root as configured_data_root
from process_discovery_cash.utils.paths import project_root

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_RELEASE_METADATA = Path("release/v6.json")


class GeneratedInventoryError(ValueError):
    """Raised when generated data does not match the v6 release contract."""


def verify_generated_inventory(
    *,
    data_root: str | Path | None = None,
    expected_inventory: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate generated v6 manifests, counts, portable paths, and file hashes."""
    root = (
        Path(data_root).expanduser().resolve(strict=False)
        if data_root is not None
        else configured_data_root()
    )
    expected = _normalize_expected_inventory(expected_inventory or _release_inventory())
    errors: list[str] = []

    augmentation_manifest = root / "augmented/manifest.csv"
    target_design = root / "synthetic/gedi/targets.csv"
    synthetic_manifest = root / "synthetic/gedi/manifest.csv"

    augmentation_rows = _read_csv(augmentation_manifest, errors)
    target_rows = _read_csv(target_design, errors)
    synthetic_rows = _read_csv(synthetic_manifest, errors)

    augmented_receipts = _accepted_artifact_receipts(
        augmentation_rows,
        id_field="child_log_id",
        kind="augmented",
        root=root,
        errors=errors,
    )
    synthetic_receipts = _accepted_artifact_receipts(
        synthetic_rows,
        id_field="log_id",
        kind="synthetic",
        root=root,
        errors=errors,
    )
    target_ids = _unique_nonempty_values(target_rows, "target_id", "GEDI target", errors)

    counts = {
        "real_logs": int(expected["real_logs"]),
        "accepted_augmented_logs": len(augmented_receipts),
        "gedi_targets": len(target_ids),
        "accepted_synthetic_logs": len(synthetic_receipts),
    }
    counts["total_event_logs"] = (
        counts["real_logs"]
        + counts["accepted_augmented_logs"]
        + counts["accepted_synthetic_logs"]
    )
    for field, actual in counts.items():
        wanted = int(expected[field])
        if actual != wanted:
            errors.append(f"{field}: expected {wanted}, got {actual}")

    if errors:
        raise GeneratedInventoryError(
            "Generated v6 inventory validation failed:\n" + "\n".join(sorted(errors))
        )

    manifest_hashes = {
        "augmentation_manifest_sha256": _sha256(augmentation_manifest),
        "gedi_manifest_sha256": _sha256(synthetic_manifest),
        "target_design_sha256": _sha256(target_design),
    }
    artifact_receipts = sorted([*augmented_receipts, *synthetic_receipts])
    return {
        "status": "ok",
        **counts,
        **manifest_hashes,
        "generated_artifact_count": len(artifact_receipts),
        "generated_artifact_receipt_sha256": _combined_receipt_hash(artifact_receipts),
    }


def _release_inventory() -> dict[str, int]:
    metadata_path = project_root() / DEFAULT_RELEASE_METADATA
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    inventory = payload.get("canonical_inventory")
    if not isinstance(inventory, dict):
        raise GeneratedInventoryError(
            f"Release metadata has no canonical_inventory mapping: {metadata_path}"
        )
    required = {
        "real_logs",
        "accepted_augmented_logs",
        "gedi_targets",
        "accepted_synthetic_logs",
        "total_event_logs",
    }
    missing = sorted(required - inventory.keys())
    if missing:
        raise GeneratedInventoryError(
            f"Release metadata is missing inventory fields: {', '.join(missing)}"
        )
    return {field: int(inventory[field]) for field in required}


def _normalize_expected_inventory(expected: dict[str, int]) -> dict[str, int]:
    normalized = dict(expected)
    if "total_event_logs" not in normalized and "hpo_logs" in normalized:
        normalized["total_event_logs"] = normalized.pop("hpo_logs")
    return normalized


def _read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        errors.append(f"missing deterministic manifest: {path}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"could not read {path}: {exc}")
        return []


def _accepted_artifact_receipts(
    rows: list[dict[str, str]],
    *,
    id_field: str,
    kind: str,
    root: Path,
    errors: list[str],
) -> list[tuple[str, str]]:
    accepted = [row for row in rows if (row.get("status") or "").strip() == "accepted"]
    _unique_nonempty_values(accepted, id_field, f"accepted {kind} log", errors)
    receipts: list[tuple[str, str]] = []
    for row in accepted:
        log_id = (row.get(id_field) or "<missing-id>").strip()
        portable_path = (row.get("output_path") or "").strip()
        expected_sha256 = (row.get("artifact_sha256") or "").strip().lower()
        if not portable_path.startswith("data/") or Path(portable_path).is_absolute():
            errors.append(
                f"{kind} {log_id}: output_path must use the portable data/... namespace"
            )
            continue
        if not SHA256_PATTERN.fullmatch(expected_sha256):
            errors.append(f"{kind} {log_id}: artifact_sha256 is missing or malformed")
            continue
        artifact = root.joinpath(*Path(portable_path).parts[1:])
        if not artifact.is_file():
            errors.append(f"{kind} {log_id}: artifact is missing: {portable_path}")
            continue
        actual_sha256 = _sha256(artifact)
        if actual_sha256 != expected_sha256:
            errors.append(
                f"{kind} {log_id}: SHA-256 mismatch for {portable_path}; "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
            continue
        receipts.append((portable_path, actual_sha256))
    return receipts


def _unique_nonempty_values(
    rows: list[dict[str, str]],
    field: str,
    label: str,
    errors: list[str],
) -> set[str]:
    values: list[str] = []
    for index, row in enumerate(rows):
        value = (row.get(field) or "").strip()
        if not value:
            errors.append(f"{label} row {index}: missing {field}")
        else:
            values.append(value)
    duplicates = sorted(value for value in set(values) if values.count(value) > 1)
    if duplicates:
        errors.append(f"{label}: duplicate {field} values: {', '.join(duplicates)}")
    return set(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_receipt_hash(receipts: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for portable_path, artifact_sha256 in receipts:
        digest.update(portable_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(artifact_sha256))
    return digest.hexdigest()
