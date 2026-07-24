#!/usr/bin/env python3
"""Fail when deprecated, private, inconsistent, or non-portable material leaks."""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPOSITORY = "https://github.com/FlopGrogu/cash-process-discovery"
RELEASE_VERSION = "6.0.0"
PYTHON_VERSION = "3.11.15"
TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".slurm",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_TEXT = (
    "/Users/",
    "/home/pascal",
    "/nfs/data8/",
    "<repository-url>",
    "<repo-url>",
    "github.com/pascalmad/process-mining-cash",
    "github.com/pascaldinh/process-mining-cash",
    "configs/cluster/slurm_defaults.yaml",
    "configs/experiments/v1/",
    "configs/experiments/v2/",
    "configs/experiments/v3/",
    "configs/experiments/v4/",
    "configs/experiments/v5/",
    "experiments/manifests/cash/",
    "experiments/manifests/v5/",
    "cashv1",
    "cashv2",
    "cashv3",
    "split_miner_2",
    "pdcash-train-model",
    "/benchmark/",
    "v6_benchmark",
    "--benchmark",
    "final_benchmark_rows",
    "_benchmark_",
)
REQUIRED_FILES = (
    ".python-version",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "environments/gedi/requirements.txt",
    "pyproject.toml",
    "release/v6-manifest-receipts.json",
    "release/v6.json",
    "requirements-hpo.txt",
    "requirements.txt",
)
OBSOLETE_PATHS = (
    ".dockerignore",
    ".github/workflows/ci.yml",
    "Apptainer.def",
    "Dockerfile",
    "container",
    "environments/gedi/pyproject.toml",
    "environments/gedi/uv.lock",
    "uv.lock",
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
GENERATED_DIRECTORIES = {
    "build",
    "figures",
    "logs",
    "models",
    "outputs",
    "runs",
    "submission-dist",
    "tmp",
}
GENERATED_SUFFIXES = {
    ".bpmn",
    ".joblib",
    ".journal",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pnml",
    ".pt",
    ".pth",
    ".sha256",
}
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")
SECRET_FILENAMES = {
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
}


def submission_files() -> list[Path]:
    git_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ).resolve()
    project_pathspec = ROOT.resolve().relative_to(git_root).as_posix()
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            project_pathspec,
        ],
        cwd=git_root,
        check=True,
        capture_output=True,
    )
    return [
        git_root / item.decode()
        for item in result.stdout.split(b"\0")
        if item and (git_root / item.decode()).is_file()
    ]


def _metadata_errors() -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"required release file is missing: {relative}")
    for forbidden in (
        "configs/cluster/slurm_defaults.yaml",
        "configs/experiments/v6/benchmark",
        *OBSOLETE_PATHS,
    ):
        if (ROOT / forbidden).exists():
            errors.append(f"obsolete dependency/configuration file exists: {forbidden}")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    if project.get("version") != RELEASE_VERSION:
        errors.append(f"pyproject version must be {RELEASE_VERSION}")
    if project.get("urls", {}).get("Repository") != CANONICAL_REPOSITORY:
        errors.append("pyproject Repository URL is not canonical")

    release = json.loads((ROOT / "release/v6.json").read_text(encoding="utf-8"))
    if release.get("release") != RELEASE_VERSION:
        errors.append(f"release/v6.json version must be {RELEASE_VERSION}")
    if release.get("repository") != CANONICAL_REPOSITORY:
        errors.append("release/v6.json Repository URL is not canonical")
    if release.get("python") != PYTHON_VERSION:
        errors.append(f"release/v6.json Python must be {PYTHON_VERSION}")
    inventory = release.get("canonical_inventory", {})
    if inventory.get("total_event_logs") != 215:
        errors.append("release/v6.json total_event_logs must be 215")
    if inventory.get("primary_v6_configs") != 30:
        errors.append("release/v6.json primary_v6_configs must be 30")
    if inventory.get("default_run_survey_rows") != 210:
        errors.append("release/v6.json default_run_survey_rows must be 210")

    if (ROOT / ".python-version").read_text(encoding="utf-8").strip() != PYTHON_VERSION:
        errors.append(f".python-version must contain {PYTHON_VERSION}")
    return errors


def _markdown_link_errors(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().strip("<>")
        target_without_anchor = target.split("#", 1)[0]
        if (
            not target_without_anchor
            or "://" in target_without_anchor
            or target_without_anchor.startswith(("mailto:", "data:"))
        ):
            continue
        candidate = (path.parent / target_without_anchor).resolve()
        if not candidate.exists():
            relative = path.relative_to(ROOT).as_posix()
            errors.append(f"{relative}: local Markdown link does not exist: {target}")
    return errors


def _artifact_errors(path: Path, relative: str) -> list[str]:
    relative_path = Path(relative)
    name = relative_path.name
    suffix = relative_path.suffix.lower()
    errors: list[str] = []
    if relative_path.parts and relative_path.parts[0] in GENERATED_DIRECTORIES:
        errors.append(f"generated artifact directory is tracked: {relative}")
    if relative_path.parts[:2] == ("experiments", "generated") and name != ".gitkeep":
        errors.append(f"generated experiment artifact is tracked: {relative}")
    if suffix == ".csv":
        errors.append(f"generated CSV is tracked: {relative}")
    if suffix in GENERATED_SUFFIXES:
        errors.append(f"generated artifact is tracked: {relative}")
    if name.lower().endswith(ARCHIVE_SUFFIXES):
        errors.append(f"generated archive is tracked: {relative}")
    if (
        name == ".env"
        or name.startswith(".env.")
        and name != ".env.example"
        or name in SECRET_FILENAMES
        or suffix in {".key", ".pem"}
    ):
        errors.append(f"secret-bearing file is tracked: {relative}")
    return errors


def main() -> None:
    errors = _metadata_errors()
    for path in submission_files():
        relative = path.relative_to(ROOT).as_posix()
        if path.resolve() == Path(__file__).resolve():
            continue
        errors.extend(_artifact_errors(path, relative))
        if relative.startswith("configs/experiments/") and not relative.startswith(
            "configs/experiments/v6/"
        ):
            errors.append(f"pre-v6 experiment config is tracked: {relative}")
        if relative.startswith("experiments/manifests/") and relative != (
            "experiments/manifests/.gitkeep"
        ):
            errors.append(f"generated manifest is tracked: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Makefile":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in FORBIDDEN_TEXT:
            if needle in text:
                errors.append(f"{relative}: contains forbidden reference {needle!r}")
        if path.suffix.lower() == ".md":
            errors.extend(_markdown_link_errors(path, text))
    if errors:
        raise SystemExit("Submission audit failed:\n" + "\n".join(sorted(set(errors))))
    print(
        "Submission audit passed: canonical v6 source, metadata, links, "
        "privacy, and dependency authorities are consistent."
    )


if __name__ == "__main__":
    main()
