#!/usr/bin/env python3
"""Create a deterministic source-only archive from the cleaned Git HEAD."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "submission-dist" / "process-discovery-cash-v6.tar.gz"


def main() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise SystemExit(
            "Refusing to archive a dirty tree. Commit the cleaned submission source first."
        )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_submission.py")],
        cwd=ROOT,
        check=True,
    )
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar.gz",
            "--prefix=process-discovery-cash-v6/",
            f"--output={DESTINATION}",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )
    digest = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
    checksum_path = DESTINATION.with_suffix(DESTINATION.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {DESTINATION.name}\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_submission_archive.py"),
            str(DESTINATION),
            "--checksum",
            str(checksum_path),
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"Wrote {DESTINATION}")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()
