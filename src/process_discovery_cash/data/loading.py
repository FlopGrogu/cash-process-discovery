from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from process_discovery_cash.utils.paths import resolve_portable_path

SUPPORTED_LOG_FORMATS = (".xes", ".xes.gz", ".csv", ".parquet")
CASE_ID_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
TIMESTAMP_COLUMN = "time:timestamp"
DEFAULT_LOG_CACHE_DIR = Path("data/processed/log_cache")
LOG_CACHE_SCHEMA_VERSION = 2


class LogCacheError(RuntimeError):
    """Raised when an existing log cache is incomplete or unreadable."""


@dataclass(frozen=True)
class LoadedEventLog:
    log: Any
    backend: str
    source_path: Path
    cache_path: Path | None = None
    cache_hit: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cache_hit": self.cache_hit,
            "cache_path": str(self.cache_path) if self.cache_path is not None else None,
            "source_path": str(self.source_path),
        }


@dataclass(frozen=True)
class LogCacheResult:
    source_path: Path
    cache_path: Path
    metadata_path: Path
    created: bool
    row_count: int


def load_event_log(
    path: str | Path,
    *,
    cache_key: str | None = None,
    cache_dir: str | Path = DEFAULT_LOG_CACHE_DIR,
    use_cache: bool = True,
) -> Any:
    """Load an event log, preferring a valid Parquet cache for XES inputs."""
    return load_event_log_with_info(
        path,
        cache_key=cache_key,
        cache_dir=cache_dir,
        use_cache=use_cache,
    ).log


def load_event_log_with_info(
    path: str | Path,
    *,
    cache_key: str | None = None,
    cache_dir: str | Path = DEFAULT_LOG_CACHE_DIR,
    use_cache: bool = True,
) -> LoadedEventLog:
    path = resolve_portable_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Event log not found: {path}")

    log_format = detect_log_format(path)
    if log_format == "xes":
        cache_path, metadata_path = log_cache_paths(path, cache_key=cache_key, cache_dir=cache_dir)
        if use_cache:
            cache_state = _cache_state(path, cache_path, metadata_path)
            if cache_state == "valid":
                return LoadedEventLog(
                    log=_read_parquet_cache(cache_path),
                    backend="parquet",
                    source_path=path,
                    cache_path=cache_path,
                    cache_hit=True,
                )
            if cache_state == "corrupt":
                raise LogCacheError(
                    f"Log cache is incomplete or invalid for {path}: "
                    f"{cache_path} and {metadata_path}"
                )

        log, backend = _load_xes_dataframe(path)
        return LoadedEventLog(
            log=log,
            backend=backend,
            source_path=path,
            cache_path=cache_path,
            cache_hit=False,
        )

    if log_format == "csv":
        return LoadedEventLog(
            log=_load_csv(path),
            backend="pandas_csv",
            source_path=path,
        )
    if log_format == "parquet":
        return LoadedEventLog(
            log=_read_parquet_cache(path),
            backend="parquet_artifact",
            source_path=path,
            cache_path=path,
            cache_hit=True,
        )

    raise _unsupported_format_error(path)


def preprocess_event_log(
    path: str | Path,
    *,
    cache_key: str | None = None,
    cache_dir: str | Path = DEFAULT_LOG_CACHE_DIR,
    force: bool = False,
) -> LogCacheResult:
    """Convert one XES log to a validated Zstandard-compressed Parquet cache."""
    source_path = resolve_portable_path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Event log not found: {source_path}")
    if detect_log_format(source_path) != "xes":
        raise ValueError(f"Only XES logs can be preprocessed: {source_path}")

    cache_path, metadata_path = log_cache_paths(
        source_path,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )
    cache_state = _cache_state(source_path, cache_path, metadata_path)
    if not force and cache_state == "valid":
        metadata = _read_cache_metadata(metadata_path)
        if metadata.get("source_sha256") == _sha256(source_path):
            return LogCacheResult(
                source_path=source_path,
                cache_path=cache_path,
                metadata_path=metadata_path,
                created=False,
                row_count=int(metadata["row_count"]),
            )

    dataframe = _load_xes_with_rust(source_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source_path.stat()
    metadata = {
        "cache_schema_version": LOG_CACHE_SCHEMA_VERSION,
        "columns": [str(column) for column in dataframe.columns],
        "parser": "rustxes",
        "pm4py_version": _package_version("pm4py"),
        "row_count": len(dataframe),
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_size": source_stat.st_size,
    }

    _write_parquet_atomically(dataframe, cache_path)
    _write_json_atomically(metadata, metadata_path)
    return LogCacheResult(
        source_path=source_path,
        cache_path=cache_path,
        metadata_path=metadata_path,
        created=True,
        row_count=len(dataframe),
    )


def log_cache_paths(
    source_path: str | Path,
    *,
    cache_key: str | None = None,
    cache_dir: str | Path = DEFAULT_LOG_CACHE_DIR,
) -> tuple[Path, Path]:
    key = _safe_cache_key(cache_key or _infer_cache_key(Path(source_path)))
    cache_path = resolve_portable_path(cache_dir) / f"{key}.parquet"
    return cache_path, cache_path.with_suffix(".parquet.json")


def detect_log_format(path: str | Path) -> str:
    """Return the supported event-log format for a path."""
    path = Path(path)
    lower_name = path.name.lower()
    if lower_name.endswith(".xes.gz") or lower_name.endswith(".xes"):
        return "xes"
    if lower_name.endswith(".csv"):
        return "csv"
    if lower_name.endswith(".parquet"):
        return "parquet"
    raise _unsupported_format_error(path)


def ensure_pm4py_event_log(log: Any) -> Any:
    """Convert supported log-like objects to a pm4py EventLog."""
    try:
        from pm4py.objects.conversion.log import converter as log_converter
        from pm4py.objects.log.obj import Event, EventLog, EventStream, Trace
    except Exception as exc:
        raise TypeError(f"pm4py EventLog conversion is unavailable: {exc}") from exc

    if isinstance(log, EventLog):
        return log
    if isinstance(log, EventStream):
        return _convert_with_pm4py(log, log_converter, EventLog)
    if _looks_like_dataframe(log):
        return _convert_with_pm4py(normalize_pm4py_dataframe(log), log_converter, EventLog)
    if isinstance(log, list):
        return _list_to_event_log(log, EventLog, Trace, Event)
    return _convert_with_pm4py(log, log_converter, EventLog)


def prepare_pm4py_discovery_log(log: Any, *, allow_dataframe: bool) -> Any:
    if allow_dataframe and _looks_like_dataframe(log):
        return normalize_pm4py_dataframe(log)
    return ensure_pm4py_event_log(log)


def normalize_pm4py_dataframe(dataframe: Any) -> Any:
    rename_map = {}
    if CASE_ID_COLUMN not in dataframe.columns and "case_id" in dataframe.columns:
        rename_map["case_id"] = CASE_ID_COLUMN
    if ACTIVITY_COLUMN not in dataframe.columns and "activity" in dataframe.columns:
        rename_map["activity"] = ACTIVITY_COLUMN
    if TIMESTAMP_COLUMN not in dataframe.columns and "timestamp" in dataframe.columns:
        rename_map["timestamp"] = TIMESTAMP_COLUMN

    normalized = dataframe.rename(columns=rename_map).copy() if rename_map else dataframe
    required = [CASE_ID_COLUMN, ACTIVITY_COLUMN, TIMESTAMP_COLUMN]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise TypeError(
            "DataFrame event log is missing required pm4py columns: "
            f"{', '.join(missing)}. Required columns are: {', '.join(required)}."
        )
    return normalized


def _load_xes_dataframe(path: Path) -> tuple[Any, str]:
    try:
        import pm4py
    except ImportError:
        return _load_xes_minimal(path), "minimal_xml"

    # This reader feeds discovery and artifact generation, so its behavior must
    # not depend on the optional native extensions available on a workstation.
    # The Rust parser remains pinned and is used explicitly by
    # `preprocess_event_log`, where its output is captured in a checksummed
    # Parquet artifact.
    return (
        _canonicalize_dataframe(
            pm4py.read_xes(
                str(path),
                variant="chunk_regex",
                return_legacy_log_object=False,
            )
        ),
        "chunk_regex",
    )


def _load_xes_with_rust(path: Path) -> Any:
    if not _rustxes_dependencies_available():
        raise RuntimeError(
            "The rustxes preprocessing backend requires r4pm, polars, and pyarrow. "
            "Install the project dependencies before preprocessing logs."
        )
    import pm4py

    return _canonicalize_dataframe(
        pm4py.read_xes(
            str(path),
            variant="rustxes",
            return_legacy_log_object=False,
        )
    )


def _rustxes_dependencies_available() -> bool:
    if any(importlib.util.find_spec(name) is None for name in ("r4pm", "polars", "pyarrow")):
        return False
    try:
        import r4pm.df  # noqa: F401
    except ImportError:
        return False
    return True


def _cache_state(source_path: Path, cache_path: Path, metadata_path: Path) -> str:
    cache_exists = cache_path.exists()
    metadata_exists = metadata_path.exists()
    if not cache_exists and not metadata_exists:
        return "missing"
    if cache_exists != metadata_exists:
        return "corrupt"
    try:
        metadata = _read_cache_metadata(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return "corrupt"
    if metadata.get("cache_schema_version") != LOG_CACHE_SCHEMA_VERSION:
        return "stale"
    source_stat = source_path.stat()
    if metadata.get("source_size") != source_stat.st_size:
        return "stale"
    if metadata.get("source_mtime_ns") != source_stat.st_mtime_ns:
        return "stale"
    if metadata.get("source_path") != str(source_path):
        return "stale"
    if metadata.get("source_sha256") != _sha256(source_path):
        return "stale"
    return "valid"


def _read_cache_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {
        "cache_schema_version",
        "row_count",
        "source_mtime_ns",
        "source_path",
        "source_sha256",
        "source_size",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Cache metadata is missing fields: {', '.join(sorted(missing))}")
    return metadata


def _read_parquet_cache(cache_path: Path) -> Any:
    try:
        import pandas as pd

        return pd.read_parquet(cache_path, engine="pyarrow")
    except Exception as exc:
        raise LogCacheError(f"Could not read Parquet log cache {cache_path}: {exc}") from exc


def _write_parquet_atomically(dataframe: Any, cache_path: Path) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp.parquet",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        dataframe.to_parquet(
            temp_path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        os.replace(temp_path, cache_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _write_json_atomically(payload: dict[str, Any], output_path: Path) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _load_xes_minimal(path: Path) -> list[list[dict[str, Any]]]:
    if path.name.lower().endswith(".xes.gz"):
        with gzip.open(path, "rb") as handle:
            tree = ElementTree.parse(handle)
    else:
        tree = ElementTree.parse(path)
    root = tree.getroot()
    traces: list[list[dict[str, Any]]] = []
    for trace_element in root.findall(".//{*}trace"):
        trace: list[dict[str, Any]] = []
        for event_element in trace_element.findall("{*}event"):
            event: dict[str, Any] = {}
            for child in list(event_element):
                key = child.attrib.get("key")
                value = _parse_xes_attribute_value(child)
                if key:
                    event[key] = value
            trace.append(event)
        traces.append(trace)
    return traces


def _load_csv(path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError:
        return _load_csv_minimal(path)
    return pd.read_csv(path)


def _load_csv_minimal(path: Path) -> list[list[dict[str, Any]]]:
    traces: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = row.get(CASE_ID_COLUMN) or row.get("case_id") or "case_0"
            traces.setdefault(case_id, []).append(dict(row))
    return list(traces.values())


def _convert_with_pm4py(log: Any, log_converter: Any, event_log_class: type) -> Any:
    try:
        converted = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    except Exception as exc:
        raise TypeError(f"Could not convert {type(log).__name__} to pm4py EventLog: {exc}") from exc
    if not isinstance(converted, event_log_class):
        raise TypeError(
            "pm4py conversion did not return EventLog. "
            f"Input type was {type(log).__name__}; output type was {type(converted).__name__}."
        )
    return converted


def _list_to_event_log(
    log: list[Any],
    event_log_class: type,
    trace_class: type,
    event_class: type,
) -> Any:
    if not log:
        return event_log_class()
    if all(isinstance(item, dict) for item in log):
        return _event_dicts_to_event_log(log, event_log_class, trace_class, event_class)

    event_log = event_log_class()
    for index, trace in enumerate(log):
        trace_attributes = dict(getattr(trace, "attributes", {}))
        trace_attributes.setdefault(ACTIVITY_COLUMN, f"trace_{index}")
        converted_trace = trace_class(attributes=trace_attributes)
        for event in trace:
            if isinstance(event, event_class):
                converted_trace.append(event)
            elif isinstance(event, dict):
                converted_trace.append(event_class(dict(event)))
            else:
                raise TypeError(
                    "List event log contains unsupported event type "
                    f"{type(event).__name__}; expected dict or pm4py Event."
                )
        event_log.append(converted_trace)
    return event_log


def _event_dicts_to_event_log(
    events: list[dict[str, Any]],
    event_log_class: type,
    trace_class: type,
    event_class: type,
) -> Any:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, event in enumerate(events):
        case_id = event.get(CASE_ID_COLUMN) or event.get("case_id") or f"case_{index}"
        grouped.setdefault(str(case_id), []).append(event)

    event_log = event_log_class()
    for case_id, case_events in grouped.items():
        trace = trace_class(attributes={ACTIVITY_COLUMN: case_id})
        for event in case_events:
            trace.append(event_class(dict(event)))
        event_log.append(trace)
    return event_log


def _looks_like_dataframe(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "groupby") and hasattr(value, "rename")


def _canonicalize_dataframe(dataframe: Any) -> Any:
    import pandas as pd

    normalized = normalize_pm4py_dataframe(dataframe)
    required = [CASE_ID_COLUMN, ACTIVITY_COLUMN, TIMESTAMP_COLUMN]
    remaining = sorted(column for column in normalized.columns if column not in required)
    canonical = normalized.loc[:, [*required, *remaining]].copy()
    canonical[TIMESTAMP_COLUMN] = pd.to_datetime(
        canonical[TIMESTAMP_COLUMN],
        utc=True,
    ).astype("datetime64[ns, UTC]")
    return canonical


def _parse_xes_attribute_value(element: ElementTree.Element) -> Any:
    value = element.attrib.get("value")
    tag = element.tag.rsplit("}", maxsplit=1)[-1]
    if value is None:
        return None
    if tag == "date":
        return _parse_xes_datetime(value)
    if tag == "int":
        return int(value)
    if tag == "float":
        return float(value)
    if tag == "boolean":
        return value.lower() == "true"
    return value


def _parse_xes_datetime(value: str) -> datetime | str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value


def _infer_cache_key(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".xes.gz"):
        return name[: -len(".xes.gz")]
    return path.stem


def _safe_cache_key(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not normalized:
        raise ValueError(f"Could not derive a cache key from {value!r}")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _unsupported_format_error(path: Path) -> ValueError:
    supported = ", ".join(SUPPORTED_LOG_FORMATS)
    return ValueError(
        f"Unsupported log format for file '{path.name}'. Supported formats are: {supported}."
    )
