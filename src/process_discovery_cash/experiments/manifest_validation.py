from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = [
    "experiment_id",
    "log_id",
    "log_path",
    "seed",
    "algorithm_id",
    "output_path",
]

PATH_COLUMNS = [
    "log_path",
    "test_log_path",
    "source_log_path",
    "discovery_log_path",
    "test_discovery_log_path",
    "preprocessing_metadata_path",
    "log_dir",
    "output_path",
]

MACHINE_LOCAL_PREFIXES = (
    "/" + "Users/",
    "/home/stud/",
)


@dataclass(frozen=True)
class ManifestValidationIssue:
    row_number: int | None
    column: str | None
    message: str

    def format(self) -> str:
        location = "manifest"
        if self.row_number is not None:
            location = f"row {self.row_number}"
        if self.column:
            location = f"{location}, column {self.column}"
        return f"{location}: {self.message}"


@dataclass(frozen=True)
class ManifestValidationResult:
    row_count: int
    issues: list[ManifestValidationIssue]

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_manifest_file(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    check_output_parents: bool = False,
) -> ManifestValidationResult:
    manifest = Path(manifest_path)
    issues: list[ManifestValidationIssue] = []
    if not manifest.exists():
        return ManifestValidationResult(
            row_count=0,
            issues=[
                ManifestValidationIssue(
                    None,
                    None,
                    f"manifest does not exist: {manifest}",
                )
            ],
        )

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        issues.extend(_schema_issues(fieldnames))
        rows = list(reader)

    if not rows:
        issues.append(ManifestValidationIssue(None, None, "manifest has no data rows"))

    root = Path(project_root).resolve(strict=False) if project_root else None
    header_values = list(fieldnames)
    output_parent_cache: set[Path] = set()
    for index, row in enumerate(rows, start=2):
        if [row.get(column, "") for column in fieldnames] == header_values:
            issues.append(
                ManifestValidationIssue(
                    index,
                    None,
                    "duplicate header row found in manifest body",
                )
            )
        issues.extend(_path_issues(row, index, root))
        if check_output_parents:
            issue = _ensure_output_parent(row, index, root, output_parent_cache)
            if issue:
                issues.append(issue)
            issue = _ensure_log_directory(row, index, root, output_parent_cache)
            if issue:
                issues.append(issue)

    return ManifestValidationResult(row_count=len(rows), issues=issues)


def raise_for_manifest_validation(result: ManifestValidationResult) -> None:
    if result.ok:
        return
    formatted = "\n".join(f"- {issue.format()}" for issue in result.issues)
    raise ValueError(f"Manifest validation failed:\n{formatted}")


def _schema_issues(fieldnames: list[str]) -> list[ManifestValidationIssue]:
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if not missing:
        return []
    return [
        ManifestValidationIssue(
            None,
            None,
            f"missing required column(s): {', '.join(missing)}",
        )
    ]


def _path_issues(
    row: dict[str, str],
    row_number: int,
    project_root: Path | None,
) -> list[ManifestValidationIssue]:
    issues: list[ManifestValidationIssue] = []
    for column in PATH_COLUMNS:
        value = row.get(column, "")
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            continue
        if _has_machine_local_prefix(value):
            issues.append(
                ManifestValidationIssue(
                    row_number,
                    column,
                    "uses a machine-local absolute path; use a repo-relative path instead",
                )
            )
            continue
        if project_root and not _is_relative_to(path, project_root):
            issues.append(
                ManifestValidationIssue(
                    row_number,
                    column,
                    f"absolute path is outside PROJECT_ROOT ({project_root})",
                )
            )
    return issues


def _ensure_output_parent(
    row: dict[str, str],
    row_number: int,
    project_root: Path | None,
    output_parent_cache: set[Path],
) -> ManifestValidationIssue | None:
    output_value = row.get("output_path", "")
    if not output_value:
        return None
    output_path = Path(output_value)
    if not output_path.is_absolute() and project_root:
        output_path = project_root / output_path
    parent = output_path.parent
    if parent in output_parent_cache:
        return None
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ManifestValidationIssue(
            row_number,
            "output_path",
            f"could not create output parent {parent}: {type(exc).__name__}: {exc}",
        )
    output_parent_cache.add(parent)
    return None


def _ensure_log_directory(
    row: dict[str, str],
    row_number: int,
    project_root: Path | None,
    output_parent_cache: set[Path],
) -> ManifestValidationIssue | None:
    log_dir_value = row.get("log_dir", "")
    if not log_dir_value:
        return None
    log_dir = Path(log_dir_value)
    if not log_dir.is_absolute() and project_root:
        log_dir = project_root / log_dir
    if log_dir in output_parent_cache:
        return None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ManifestValidationIssue(
            row_number,
            "log_dir",
            f"could not create log directory {log_dir}: {type(exc).__name__}: {exc}",
        )
    output_parent_cache.add(log_dir)
    return None


def _has_machine_local_prefix(value: str) -> bool:
    return any(value.startswith(prefix) for prefix in MACHINE_LOCAL_PREFIXES)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True
