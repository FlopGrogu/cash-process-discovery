#!/usr/bin/env python3
"""Verify the source-only submission archive and its SHA-256 sidecar."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path, PurePosixPath

PREFIX = PurePosixPath("process-discovery-cash-v6")
REQUIRED = {
    PREFIX / ".python-version",
    PREFIX / "CITATION.cff",
    PREFIX / "LICENSE",
    PREFIX / "Makefile",
    PREFIX / "README.md",
    PREFIX / "THIRD_PARTY_NOTICES.md",
    PREFIX / "docs/cluster.md",
    PREFIX / "environments/gedi/pyproject.toml",
    PREFIX / "environments/gedi/requirements.txt",
    PREFIX / "pyproject.toml",
    PREFIX / "release/v6-manifest-receipts.json",
    PREFIX / "release/v6.json",
    PREFIX / "scripts/audit_submission.py",
    PREFIX / "scripts/verify_submission_archive.py",
    PREFIX / "requirements.txt",
}
MODEL_SUFFIXES = {
    ".bpmn",
    ".joblib",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pnml",
    ".pt",
    ".pth",
}
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")
GENERATED_SUFFIXES = {".journal", ".sha256"}
GENERATED_DIRECTORIES = {
    ("build",),
    ("figures",),
    ("logs",),
    ("models",),
    ("outputs",),
    ("runs",),
    ("submission-dist",),
    ("tmp",),
}
SECRET_FILENAMES = {
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checksum", type=Path)
    return parser


def _allowed_data_path(relative: PurePosixPath) -> bool:
    parts = relative.parts
    if relative == PurePosixPath("data/README.md"):
        return True
    if len(parts) >= 2 and parts[:2] == ("data", "example"):
        return True
    return relative.name == ".gitkeep"


def _allowed_results_path(relative: PurePosixPath) -> bool:
    return relative == PurePosixPath("results/README.md") or relative.name == ".gitkeep"


def verify_archive(archive: Path, checksum: Path) -> str:
    archive = archive.resolve()
    checksum = checksum.resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum_fields = checksum.read_text(encoding="utf-8").strip().split()
    if checksum_fields != [digest, archive.name]:
        raise ValueError(f"checksum sidecar does not match {archive.name}")

    errors: list[str] = []
    with tarfile.open(archive, mode="r:gz") as handle:
        members = {
            PurePosixPath(member.name)
            for member in handle.getmembers()
            if member.isfile() or member.issym()
        }
    missing = REQUIRED - members
    errors.extend(f"required archive member is missing: {path}" for path in sorted(missing))

    for member in sorted(members):
        if not member.parts or member.parts[0] != PREFIX.name:
            errors.append(f"archive member is outside the release prefix: {member}")
            continue
        relative = PurePosixPath(*member.parts[1:])
        if ".git" in relative.parts:
            errors.append(f"Git metadata is present: {member}")
        if relative.suffix.lower() == ".csv":
            errors.append(f"generated CSV is present: {member}")
        if (
            relative.name == ".env"
            or relative.name.startswith(".env.")
            and relative.name != ".env.example"
            or relative.name in SECRET_FILENAMES
            or relative.suffix.lower() in {".key", ".pem"}
        ):
            errors.append(f"secret-bearing file is present: {member}")
        if relative.suffix.lower() in MODEL_SUFFIXES:
            errors.append(f"generated model artifact is present: {member}")
        if relative.suffix.lower() in GENERATED_SUFFIXES:
            errors.append(f"generated artifact is present: {member}")
        if relative.name.lower().endswith(ARCHIVE_SUFFIXES):
            errors.append(f"generated archive is present: {member}")
        if relative.parts[:2] == ("experiments", "manifests") and relative.name != ".gitkeep":
            errors.append(f"generated experiment manifest is present: {member}")
        if relative.parts[:2] == ("experiments", "generated") and relative.name != ".gitkeep":
            errors.append(f"generated experiment artifact is present: {member}")
        if relative.parts[:1] == ("data",) and not _allowed_data_path(relative):
            errors.append(f"non-source data artifact is present: {member}")
        if relative.parts[:1] == ("results",) and not _allowed_results_path(relative):
            errors.append(f"generated result artifact is present: {member}")
        if relative.parts[:1] in GENERATED_DIRECTORIES:
            errors.append(f"generated artifact directory is present: {member}")
    if errors:
        raise ValueError("\n".join(errors))
    return digest


def main() -> None:
    args = build_parser().parse_args()
    checksum = args.checksum or args.archive.with_suffix(args.archive.suffix + ".sha256")
    digest = verify_archive(args.archive, checksum)
    print(f"Submission archive verified: {args.archive} ({digest})")


if __name__ == "__main__":
    main()
