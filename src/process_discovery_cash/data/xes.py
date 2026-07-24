"""Deterministic XES serialization for generated event logs."""

from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path
from typing import TextIO
from xml.sax.saxutils import quoteattr

import pandas as pd

from process_discovery_cash.utils.paths import resolve_portable_path

CASE_ID_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
TIMESTAMP_COLUMN = "time:timestamp"
LIFECYCLE_COLUMN = "lifecycle:transition"


def write_canonical_xes(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    *,
    force_complete: bool = True,
) -> Path:
    """Write the canonical three-attribute XES projection byte-for-byte deterministically.

    Trace and event order, XML element order, timestamp representation, gzip
    metadata, and line endings are all fixed. Generated logs intentionally
    contain only the case id, activity, timestamp, and lifecycle transition
    required by the v6 pipeline.
    """
    missing = [
        column
        for column in (CASE_ID_COLUMN, ACTIVITY_COLUMN, TIMESTAMP_COLUMN)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Cannot write XES; missing column(s): {', '.join(missing)}")

    output = resolve_portable_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = dataframe.copy()
    frame["@@canonical_order"] = range(len(frame))
    frame[CASE_ID_COLUMN] = frame[CASE_ID_COLUMN].astype(str)
    frame[ACTIVITY_COLUMN] = frame[ACTIVITY_COLUMN].astype(str)
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN],
        errors="raise",
        utc=True,
    )
    frame = frame.sort_values(
        [CASE_ID_COLUMN, TIMESTAMP_COLUMN, "@@canonical_order"],
        kind="stable",
    )

    suffix = ".xes.gz" if output.name.lower().endswith(".gz") else ".xes"
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=suffix,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        if suffix.endswith(".gz"):
            with temporary.open("wb") as raw:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    compresslevel=9,
                    mtime=0,
                ) as compressed:
                    with _text_writer(compressed) as text:
                        _write_xes_document(frame, text, force_complete=force_complete)
                raw.flush()
                os.fsync(raw.fileno())
        else:
            with temporary.open("w", encoding="utf-8", newline="\n") as text:
                _write_xes_document(frame, text, force_complete=force_complete)
                text.flush()
                os.fsync(text.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _text_writer(binary_handle) -> TextIO:
    import io

    return io.TextIOWrapper(binary_handle, encoding="utf-8", newline="\n")


def _write_xes_document(
    frame: pd.DataFrame,
    handle: TextIO,
    *,
    force_complete: bool,
) -> None:
    handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
    handle.write('<log xes.version="1.0" xes.features="nested-attributes">\n')
    handle.write(
        '  <extension name="Concept" prefix="concept" '
        'uri="http://www.xes-standard.org/concept.xesext"/>\n'
    )
    handle.write(
        '  <extension name="Lifecycle" prefix="lifecycle" '
        'uri="http://www.xes-standard.org/lifecycle.xesext"/>\n'
    )
    handle.write(
        '  <extension name="Time" prefix="time" uri="http://www.xes-standard.org/time.xesext"/>\n'
    )
    handle.write('  <classifier name="Event Name" keys="concept:name"/>\n')
    for case_id, events in frame.groupby(CASE_ID_COLUMN, sort=False):
        handle.write("  <trace>\n")
        handle.write(f'    <string key="concept:name" value={quoteattr(str(case_id))}/>\n')
        for _, row in events.iterrows():
            timestamp = pd.Timestamp(row[TIMESTAMP_COLUMN]).isoformat().replace("+00:00", "Z")
            transition = (
                "complete"
                if force_complete
                else str(row.get(LIFECYCLE_COLUMN, "complete")).strip().lower()
            )
            handle.write("    <event>\n")
            handle.write(
                f'      <string key="concept:name" value={quoteattr(str(row[ACTIVITY_COLUMN]))}/>\n'
            )
            handle.write(
                f'      <string key="lifecycle:transition" value={quoteattr(transition)}/>\n'
            )
            handle.write(f'      <date key="time:timestamp" value={quoteattr(timestamp)}/>\n')
            handle.write("    </event>\n")
        handle.write("  </trace>\n")
    handle.write("</log>\n")
