from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from process_discovery_cash.data.loading import load_event_log_with_info
from process_discovery_cash.data.preprocessing.catalog import get_dataset
from process_discovery_cash.data.preprocessing.lifecycle import analyze_lifecycle
from process_discovery_cash.data.preprocessing.metadata import (
    inspect_dataset_package,
    sha256_file,
)
from process_discovery_cash.data.preprocessing.models import DatasetSpec
from process_discovery_cash.data.xes import write_canonical_xes
from process_discovery_cash.utils.hashing import stable_hash
from process_discovery_cash.utils.paths import resolve_portable_path

PREPROCESSING_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("data/processed/event_logs")
CASE_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
TIMESTAMP_COLUMN = "time:timestamp"
SOURCE_INDEX_COLUMN = "@@source_event_index"
_INVALID_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class PreprocessingOptions(BaseModel):
    invalid_row_policy: str = "fail"
    optional_attributes: list[str] = Field(default_factory=list)
    compact_labels: bool = False


@dataclass(frozen=True)
class ArtifactSet:
    dataset_id: str
    fingerprint: str
    output_dir: Path
    pm4py_path: Path
    splitminer_v1_path: Path
    metadata_path: Path
    label_mapping_path: Path | None = None


def preprocessing_fingerprint(
    dataset: DatasetSpec,
    *,
    options: PreprocessingOptions | None = None,
) -> str:
    options = options or PreprocessingOptions()
    source_hash = dataset.sha256 or sha256_file(resolve_portable_path(dataset.source_path))
    semantic_companions = []
    if dataset.metadata_affects_resolution:
        for companion in dataset.companions:
            if companion.role == "processmining_metadata":
                semantic_companions.append(
                    {
                        "role": companion.role,
                        "sha256": sha256_file(resolve_portable_path(companion.path)),
                    }
                )
    payload = {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "source_sha256": source_hash,
        "schema": dataset.event_schema.model_dump(mode="json"),
        "options": options.model_dump(mode="json"),
        "semantic_companions": semantic_companions,
    }
    return stable_hash(payload)[:24]


def artifact_paths(
    dataset: DatasetSpec,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    options: PreprocessingOptions | None = None,
    fingerprint_override: str | None = None,
) -> ArtifactSet:
    # A caller that already knows the fingerprint (resolved where the raw source
    # was available) can skip hashing the source — required on hosts that hold
    # only the preprocessed artifacts, not the raw logs.
    fingerprint = fingerprint_override or preprocessing_fingerprint(dataset, options=options)
    output_dir = resolve_portable_path(output_root) / dataset.dataset_id / fingerprint
    return ArtifactSet(
        dataset_id=dataset.dataset_id,
        fingerprint=fingerprint,
        output_dir=output_dir,
        pm4py_path=output_dir / "pm4py.parquet",
        splitminer_v1_path=output_dir / "splitminer-v1.xes",
        metadata_path=output_dir / "metadata.json",
        label_mapping_path=(
            output_dir / "label-mapping.json"
            if (options or PreprocessingOptions()).compact_labels
            else None
        ),
    )


def preprocess_dataset(
    dataset_id: str,
    *,
    catalog_path: str | Path = "configs/datasets/processmining_org.yaml",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    options: PreprocessingOptions | None = None,
    force: bool = False,
) -> ArtifactSet:
    dataset = get_dataset(dataset_id, catalog_path)
    return preprocess_dataset_spec(
        dataset,
        output_root=output_root,
        options=options,
        force=force,
    )


def preprocess_dataset_spec(
    dataset: DatasetSpec,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    options: PreprocessingOptions | None = None,
    force: bool = False,
) -> ArtifactSet:
    options = options or PreprocessingOptions()
    paths = artifact_paths(dataset, output_root=output_root, options=options)
    if not force and _artifacts_complete(paths):
        return paths

    package = inspect_dataset_package(dataset)
    loaded = load_event_log_with_info(dataset.source_path, use_cache=False)
    if not isinstance(loaded.log, pd.DataFrame):
        raise TypeError(
            f"Dataset preprocessing requires a DataFrame parser, got {type(loaded.log).__name__}"
        )
    canonical, validation = canonicalize_dataframe(loaded.log, dataset, options=options)
    label_mapping: dict[str, str] | None = None
    if options.compact_labels:
        labels = sorted(canonical[ACTIVITY_COLUMN].unique())
        label_mapping = {str(label): f"A{index:04d}" for index, label in enumerate(labels, start=1)}
        canonical[ACTIVITY_COLUMN] = canonical[ACTIVITY_COLUMN].map(label_mapping)
    lifecycle = analyze_lifecycle(
        canonical,
        semantics=dataset.event_schema.lifecycle_semantics,
        case_column=CASE_COLUMN,
        activity_column=ACTIVITY_COLUMN,
        timestamp_column=TIMESTAMP_COLUMN,
        lifecycle_column=dataset.event_schema.lifecycle,
        start_timestamp_column=dataset.event_schema.start_timestamp,
    )
    projection = discovery_projection(canonical, dataset)
    inspection = inspect_dataframe(canonical, source_order=True)

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomically(projection, paths.pm4py_path)
    _write_minimal_xes(projection, paths.splitminer_v1_path, force_complete=True)

    artifacts: dict[str, dict[str, Any]] = {
        "pm4py_parquet": _artifact_metadata(paths.pm4py_path, len(projection)),
        "splitminer_v1_xes": _artifact_metadata(paths.splitminer_v1_path, len(projection)),
    }
    if label_mapping is not None and paths.label_mapping_path is not None:
        _write_json_atomically(
            {
                "original_to_compact": label_mapping,
                "compact_to_original": {
                    compact: original for original, compact in label_mapping.items()
                },
            },
            paths.label_mapping_path,
        )
        artifacts["label_mapping_json"] = _artifact_metadata(
            paths.label_mapping_path, len(label_mapping)
        )
    metadata = {
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        "preprocessing_fingerprint": paths.fingerprint,
        "dataset": dataset.model_dump(mode="json", by_alias=True),
        "package": package.model_dump(mode="json"),
        "parser": loaded.metadata(),
        "resolved_schema": dataset.event_schema.model_dump(mode="json"),
        "options": options.model_dump(mode="json"),
        "validation": validation,
        "inspection": inspection,
        "projection": {
            "input_events": len(canonical),
            "discovery_events": len(projection),
            "dropped_events": len(canonical) - len(projection),
            "rule": _projection_rule(dataset),
        },
        "lifecycle_analysis": lifecycle.model_dump(mode="json"),
        "artifacts": artifacts,
    }
    if package.processmining_metadata:
        expected = package.processmining_metadata
        if expected.expected_events is not None and expected.expected_events != len(canonical):
            metadata["package"]["discrepancies"].append(
                f"Expected {expected.expected_events} events, observed {len(canonical)}."
            )
        cases = canonical[CASE_COLUMN].nunique()
        if expected.expected_traces is not None and expected.expected_traces != cases:
            metadata["package"]["discrepancies"].append(
                f"Expected {expected.expected_traces} traces, observed {cases}."
            )
    _write_json_atomically(metadata, paths.metadata_path)
    return paths


def canonicalize_dataframe(
    dataframe: pd.DataFrame,
    dataset: DatasetSpec,
    *,
    options: PreprocessingOptions | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    options = options or PreprocessingOptions()
    source_case = _source_column(dataset.event_schema.case_id)
    rename = {
        source_case: CASE_COLUMN,
        dataset.event_schema.activity: ACTIVITY_COLUMN,
        dataset.event_schema.complete_timestamp: TIMESTAMP_COLUMN,
    }
    missing_columns = [column for column in rename if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required source columns: {', '.join(missing_columns)}")

    canonical = dataframe.rename(columns=rename).copy()
    canonical[SOURCE_INDEX_COLUMN] = range(len(canonical))
    canonical[TIMESTAMP_COLUMN], timestamp_failures = _parse_timestamp_column(
        canonical[TIMESTAMP_COLUMN]
    )
    if (
        dataset.event_schema.start_timestamp
        and dataset.event_schema.start_timestamp in canonical.columns
    ):
        canonical[dataset.event_schema.start_timestamp], start_failures = _parse_timestamp_column(
            canonical[dataset.event_schema.start_timestamp]
        )
    else:
        start_failures = 0

    invalid = {
        "source_columns": [str(column) for column in dataframe.columns],
        "missing_case_id": int(canonical[CASE_COLUMN].isna().sum()),
        "missing_activity": int(canonical[ACTIVITY_COLUMN].isna().sum()),
        "missing_complete_timestamp": int(canonical[TIMESTAMP_COLUMN].isna().sum()),
        "timestamp_parse_failures": timestamp_failures,
        "start_timestamp_parse_failures": start_failures,
    }
    row_count = len(canonical)
    invalid.update(
        {
            "missing_case_id_percent": _percent(invalid["missing_case_id"], row_count),
            "missing_activity_percent": _percent(invalid["missing_activity"], row_count),
            "missing_complete_timestamp_percent": _percent(
                invalid["missing_complete_timestamp"], row_count
            ),
            "xml_invalid_case_labels": _xml_invalid_count(canonical[CASE_COLUMN]),
            "xml_invalid_activity_labels": _xml_invalid_count(canonical[ACTIVITY_COLUMN]),
            "suspicious_activity_labels": _suspicious_activity_labels(canonical[ACTIVITY_COLUMN]),
        }
    )
    invalid_mask = (
        canonical[CASE_COLUMN].isna()
        | canonical[ACTIVITY_COLUMN].isna()
        | canonical[TIMESTAMP_COLUMN].isna()
    )
    invalid_count = int(invalid_mask.sum())
    invalid["invalid_rows"] = invalid_count
    if invalid_count:
        if options.invalid_row_policy != "drop":
            raise ValueError(
                f"Dataset contains {invalid_count} invalid discovery events: {invalid}"
            )
        canonical = canonical.loc[~invalid_mask].copy()

    for column in (CASE_COLUMN, ACTIVITY_COLUMN):
        canonical[column] = canonical[column].astype(str).map(sanitize_xml_text)

    selected = [
        CASE_COLUMN,
        ACTIVITY_COLUMN,
        TIMESTAMP_COLUMN,
        SOURCE_INDEX_COLUMN,
    ]
    for column in [
        dataset.event_schema.lifecycle,
        dataset.event_schema.start_timestamp,
        *dataset.event_schema.optional_attributes,
        *options.optional_attributes,
    ]:
        if column and column in canonical.columns and column not in selected:
            selected.append(column)
    return canonical.loc[:, selected], invalid


def discovery_projection(dataframe: pd.DataFrame, dataset: DatasetSpec) -> pd.DataFrame:
    projected = dataframe
    if dataset.event_schema.lifecycle_semantics in {"standard", "extended_standard"}:
        lifecycle = dataset.event_schema.lifecycle
        if not lifecycle or lifecycle not in dataframe.columns:
            raise ValueError("Complete-event projection requires a lifecycle column")
        projected = dataframe.loc[
            dataframe[lifecycle].fillna("").astype(str).str.lower() == "complete"
        ].copy()
    return projected.sort_values(
        [CASE_COLUMN, TIMESTAMP_COLUMN, SOURCE_INDEX_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)


def inspect_dataframe(dataframe: pd.DataFrame, *, source_order: bool) -> dict[str, Any]:
    source_grouped = dataframe.groupby(CASE_COLUMN, sort=False)
    lengths = source_grouped.size()
    duplicate_timestamps = int(
        dataframe.duplicated([CASE_COLUMN, TIMESTAMP_COLUMN], keep=False).sum()
    )
    non_monotonic = 0
    if source_order:
        non_monotonic = int(
            source_grouped[TIMESTAMP_COLUMN]
            .apply(lambda values: not values.is_monotonic_increasing)
            .sum()
        )
    ordered = dataframe.sort_values(
        [CASE_COLUMN, TIMESTAMP_COLUMN, SOURCE_INDEX_COLUMN],
        kind="mergesort",
    )
    grouped = ordered.groupby(CASE_COLUMN, sort=False)
    variants = grouped[ACTIVITY_COLUMN].agg(tuple)
    dfg = set()
    for activities in variants:
        dfg.update(zip(activities, activities[1:], strict=False))
    return {
        "events": len(dataframe),
        "cases": int(dataframe[CASE_COLUMN].nunique()),
        "activities": int(dataframe[ACTIVITY_COLUMN].nunique()),
        "high_cardinality_activity_labels": bool(dataframe[ACTIVITY_COLUMN].nunique() > 500),
        "variants": int(variants.nunique()),
        "dfg_edges": len(dfg),
        "columns": [str(column) for column in dataframe.columns],
        "duplicate_timestamp_events": duplicate_timestamps,
        "non_monotonic_cases": non_monotonic,
        "single_event_cases": int((lengths == 1).sum()),
        "max_trace_length": int(lengths.max()) if len(lengths) else 0,
        "timezone": _timezone_description(dataframe[TIMESTAMP_COLUMN]),
        "lifecycle_values": (
            dataframe.get("lifecycle:transition", pd.Series(dtype=object))
            .fillna("<missing>")
            .astype(str)
            .value_counts()
            .to_dict()
        ),
    }


def sanitize_xml_text(value: str) -> str:
    return _INVALID_XML.sub("\ufffd", value)


def _xml_invalid_count(series: pd.Series) -> int:
    return int(series.dropna().astype(str).str.contains(_INVALID_XML).sum())


def _suspicious_activity_labels(series: pd.Series) -> list[str]:
    labels = series.dropna().astype(str).drop_duplicates()
    suspicious = labels[
        labels.str.len().gt(256)
        | labels.str.contains(r"[\r\n\t]", regex=True)
        | labels.str.match(r"^\s|\s$")
    ]
    return suspicious.head(20).tolist()


def _percent(count: int, total: int) -> float:
    return (100.0 * count / total) if total else 0.0


def _source_column(configured: str) -> str:
    if configured.startswith("trace:"):
        return f"case:{configured.removeprefix('trace:')}"
    return configured


def _parse_timestamp_column(series: pd.Series) -> tuple[pd.Series, int]:
    original_non_null = int(series.notna().sum())
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    elif pd.api.types.is_datetime64_dtype(series.dtype):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        values = series.dropna().astype(str)
        has_zone = values.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all()
        parsed = pd.to_datetime(series, errors="coerce", utc=bool(has_zone))
    failures = original_non_null - int(parsed.notna().sum())
    return parsed, failures


def _projection_rule(dataset: DatasetSpec) -> str:
    if dataset.event_schema.lifecycle_semantics in {"standard", "extended_standard"}:
        return "lifecycle_complete_events"
    return "all_events"


def _timezone_description(series: pd.Series) -> str:
    timezone_value = getattr(series.dt, "tz", None)
    return str(timezone_value) if timezone_value is not None else "naive"


def _artifact_metadata(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "events": rows,
    }


def _artifacts_complete(paths: ArtifactSet) -> bool:
    required = [paths.pm4py_path, paths.splitminer_v1_path, paths.metadata_path]
    if paths.label_mapping_path is not None:
        required.append(paths.label_mapping_path)
    return all(path.exists() for path in required)


def _write_parquet_atomically(dataframe: pd.DataFrame, path: Path) -> None:
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".parquet", delete=False
        ) as handle:
            temp = Path(handle.name)
        dataframe.to_parquet(temp, engine="pyarrow", compression="zstd", index=False)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _write_minimal_xes(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    force_complete: bool,
) -> None:
    write_canonical_xes(dataframe, path, force_complete=force_complete)


def _write_json_atomically(payload: dict[str, Any], path: Path) -> None:
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".json",
            delete=False,
        ) as handle:
            temp = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
