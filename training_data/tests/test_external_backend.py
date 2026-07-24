from __future__ import annotations

import gzip
from pathlib import Path

from process_discovery_cash.discovery.external_backend import prepare_xes_input


def test_prepare_xes_input_decompresses_gzipped_xes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "log.xes.gz"
    source_text = "<log />\n"
    with gzip.open(source, "wb") as handle:
        handle.write(source_text.encode("utf-8"))

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    prepared = prepare_xes_input([], source.as_posix(), work_dir)

    assert prepared == work_dir / "log.xes"
    assert prepared.read_text(encoding="utf-8") == source_text
