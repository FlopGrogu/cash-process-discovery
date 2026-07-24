"""Real-log anchor features on top of the canonical 48-feature extractor.

The GEDI feature-space code targets six axes; their values (and the stats used
for validation) are derived from the vendored 48-feature implementation in
``data/features.py`` so real anchor logs and generated candidates are always
measured by the exact same extractor. The anchor CSV lives inside the batch
output root (next to ``targets.csv``) and carries all 48 raw features plus the
derived axis columns.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from process_discovery_cash.data.features import extract_features_from_xes
from process_discovery_cash.generation.feature_space import TARGET_FEATURES

# GEDI target axis -> vendored feature name.
AXIS_MAP = {
    "num_traces": "rs4pd_total_traces",
    "avg_trace_length": "rs4pd_trace_length_avg",
    "num_activities": "rs4pd_distinct_events",
    "variant_ratio": "ratio_unique_traces_per_trace",
    "dfg_density": "rs4pd_flow_density",
    "repetition_prevalence": "relative_number_of_traces_with_repetition",
}
# Validation statistics -> vendored feature name.
STAT_MAP = {
    "num_events": "rs4pd_total_events",
    "num_variants": "rs4pd_distinct_traces",
    "min_trace_length": "trace_len_min",
    "max_trace_length": "trace_len_max",
}

ANCHOR_FILENAME = "anchor_features.csv"


def compute_log_feature_row(xes_path: str | Path, log_id: str | None = None) -> dict[str, Any]:
    """All 48 features for one XES file, plus the derived axis/stat columns."""
    xes_path = Path(xes_path)
    features = extract_features_from_xes(str(xes_path))
    row: dict[str, Any] = {
        "log_id": log_id or _infer_log_id(xes_path),
        "log_path": str(xes_path),
    }
    row.update(features)
    for axis, source in AXIS_MAP.items():
        row[axis] = _as_float(features.get(source))
    for stat, source in STAT_MAP.items():
        row[stat] = _as_float(features.get(source))
    return row


def load_anchor_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_anchor_features(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    leading = [column for column in ("log_id", "log_path") if column in frame.columns]
    ordered = leading + [column for column in frame.columns if column not in leading]
    frame.reindex(columns=ordered).to_csv(path, index=False)
    return path


def build_anchor_features(
    catalog_path: str | Path,
    anchor_path: str | Path,
    *,
    compute_missing: bool = False,
) -> pd.DataFrame:
    """Anchor rows for every catalog dataset, extracting missing ones on demand.

    With ``compute_missing=False`` (login-node safe) missing rows raise instead
    of parsing multi-hundred-MB raw XES files.
    """
    from process_discovery_cash.data.preprocessing.catalog import load_dataset_catalog

    catalog = load_dataset_catalog(catalog_path)
    cache = load_anchor_features(anchor_path)
    rows = cache.to_dict("records") if not cache.empty else []
    complete = {
        str(row["log_id"])
        for row in rows
        if all(_is_present(row.get(axis)) for axis in TARGET_FEATURES)
    }
    missing = [dataset_id for dataset_id in sorted(catalog.datasets) if dataset_id not in complete]
    if missing and not compute_missing:
        raise RuntimeError(
            f"Anchor cache {anchor_path} is missing feature rows for: {', '.join(missing)}. "
            "Rerun with --compute-anchor on a machine that may parse the raw logs "
            "(not a cluster login node), or copy an existing anchor_features.csv "
            "next to the targets file."
        )
    for dataset_id in missing:
        dataset = catalog.datasets[dataset_id]
        print(f"Computing anchor features for {dataset_id} ({dataset.source_path})...")
        rows = [row for row in rows if str(row.get("log_id")) != dataset_id]
        rows.append(compute_log_feature_row(dataset.source_path, log_id=dataset_id))
    if missing:
        write_anchor_features(rows, anchor_path)
    frame = pd.DataFrame(rows)
    frame = frame[frame["log_id"].astype(str).isin(set(catalog.datasets))]
    return frame.reset_index(drop=True)


def _infer_log_id(path: Path) -> str:
    name = path.name
    for suffix in (".xes.gz", ".xes"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def _is_present(value: Any) -> bool:
    try:
        return value is not None and not math.isnan(float(value))
    except (TypeError, ValueError):
        return False
