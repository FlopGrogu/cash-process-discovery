"""Real-log augmentation: derive child event logs from real parent logs.

Children densify the real-log feature space for discovery experiments and HPO.
They inherit their parent's source/fold via ``parent_log_id`` in the augmentation
manifest and must never be used as independent test logs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_discovery_cash.data.loading import (
    ACTIVITY_COLUMN,
    CASE_ID_COLUMN,
    TIMESTAMP_COLUMN,
)
from process_discovery_cash.data.preprocessing.metadata import sha256_file
from process_discovery_cash.data.xes import write_canonical_xes
from process_discovery_cash.utils.hashing import stable_hash, stable_json_dumps
from process_discovery_cash.utils.paths import (
    portable_project_path,
    resolve_portable_path,
)

REQUIRED_COLUMNS = (CASE_ID_COLUMN, ACTIVITY_COLUMN, TIMESTAMP_COLUMN)
DEFAULT_OUTPUT_ROOT = Path("data/augmented")
AUGMENTED_LOG_DIRNAME = "logs"
MANIFEST_FILENAME = "manifest.csv"
CHILD_LOG_ID_PREFIX = "aug_"

MIN_CHILD_TRACES = 10
MIN_CHILD_ACTIVITIES = 3
MIN_CHILD_VARIANTS = 2

MANIFEST_COLUMNS = [
    "child_log_id",
    "parent_log_id",
    "parent_path",
    "parent_sha256",
    "augmentation",
    "parameters",
    "seed",
    "stress",
    "status",
    "rejection_reason",
    "output_path",
    "artifact_sha256",
    "n_traces",
    "n_events",
    "n_activities",
    "n_variants",
]


@dataclass(frozen=True)
class AugmentationSpec:
    """One augmentation operator application with fixed parameters."""

    operator: str
    parameters: dict[str, Any]
    stress: bool = False

    def token(self) -> str:
        """Short filename-safe identifier, e.g. ``cov080`` or ``noise005``."""
        if self.operator == "variant_coverage":
            return f"cov{_percent_token(self.parameters['coverage'])}"
        if self.operator == "subsample":
            return f"sub{_percent_token(self.parameters['fraction'])}"
        if self.operator == "noise":
            return f"noise{_percent_token(self.parameters['probability'])}"
        if self.operator == "truncate":
            return f"trunc{int(self.parameters['max_events'])}"
        if self.operator == "top_activities":
            return f"topact{_percent_token(self.parameters['coverage'])}"
        raise ValueError(f"Unknown augmentation operator: {self.operator}")


@dataclass(frozen=True)
class ChildLogRecord:
    """Manifest row describing one generated (or rejected/skipped) child log."""

    child_log_id: str
    parent_log_id: str
    parent_path: str
    parent_sha256: str
    augmentation: str
    parameters: dict[str, Any]
    seed: int
    stress: bool
    status: str
    rejection_reason: str | None
    output_path: str | None
    artifact_sha256: str | None
    n_traces: int | None
    n_events: int | None
    n_activities: int | None
    n_variants: int | None

    def to_row(self) -> dict[str, Any]:
        row = {column: getattr(self, column) for column in MANIFEST_COLUMNS}
        row["parameters"] = stable_json_dumps(self.parameters)
        return row


def canonicalize_event_log(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Project to the canonical case/activity/timestamp schema and sort.

    Rows with missing required values are dropped; events are stably sorted by
    timestamp within each case so positional operators see the true order.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Event log is missing required columns: {', '.join(missing)}.")
    log = dataframe.loc[:, list(REQUIRED_COLUMNS)].copy()
    log[CASE_ID_COLUMN] = log[CASE_ID_COLUMN].astype(str)
    log[ACTIVITY_COLUMN] = log[ACTIVITY_COLUMN].astype(str)
    if not pd.api.types.is_datetime64_any_dtype(log[TIMESTAMP_COLUMN]):
        log[TIMESTAMP_COLUMN] = pd.to_datetime(
            log[TIMESTAMP_COLUMN], errors="coerce", utc=True, format="mixed"
        )
    log = log.dropna(subset=list(REQUIRED_COLUMNS))
    log = log.sort_values([CASE_ID_COLUMN, TIMESTAMP_COLUMN], kind="stable")
    return log.reset_index(drop=True)


def compute_variants(log: pd.DataFrame) -> pd.Series:
    """Map each case id to its variant (the ordered tuple of activities)."""
    return log.groupby(CASE_ID_COLUMN, sort=False)[ACTIVITY_COLUMN].agg(tuple)


def compute_log_stats(log: pd.DataFrame) -> dict[str, Any]:
    case_variants = compute_variants(log)
    n_traces = int(case_variants.size)
    return {
        "n_traces": n_traces,
        "n_events": int(len(log)),
        "n_activities": int(log[ACTIVITY_COLUMN].nunique()),
        "n_variants": int(case_variants.nunique()),
        "mean_trace_length": float(len(log) / n_traces) if n_traces else 0.0,
    }


def subsample_variants(
    log: pd.DataFrame, fraction: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample complete cases, preserving the parent's variant distribution.

    Per-variant quotas use largest-remainder rounding so the child hits the
    target case count while staying proportional to the parent's variant mix.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    case_variants = compute_variants(log)
    n_cases = case_variants.size
    target = max(1, round(n_cases * fraction))

    variant_cases: dict[tuple, list[str]] = {}
    for case_id, variant in case_variants.items():
        variant_cases.setdefault(variant, []).append(case_id)
    # Deterministic variant order: frequency descending, then lexicographic.
    ordered_variants = sorted(
        variant_cases, key=lambda variant: (-len(variant_cases[variant]), variant)
    )

    exact = {v: len(variant_cases[v]) * fraction for v in ordered_variants}
    quotas = {v: int(exact[v]) for v in ordered_variants}
    remainder = target - sum(quotas.values())
    by_fraction = sorted(
        ordered_variants,
        key=lambda v: (-(exact[v] - quotas[v]), -len(variant_cases[v]), v),
    )
    index = 0
    while remainder > 0 and index < len(by_fraction):
        variant = by_fraction[index]
        if quotas[variant] < len(variant_cases[variant]):
            quotas[variant] += 1
            remainder -= 1
        index += 1

    selected: list[str] = []
    for variant in ordered_variants:
        quota = quotas[variant]
        if quota <= 0:
            continue
        cases = sorted(variant_cases[variant])
        chosen = rng.choice(len(cases), size=min(quota, len(cases)), replace=False)
        selected.extend(cases[position] for position in chosen)

    return log[log[CASE_ID_COLUMN].isin(set(selected))].reset_index(drop=True)


def filter_variant_coverage(log: pd.DataFrame, coverage: float) -> pd.DataFrame:
    """Keep the most frequent variants until case coverage reaches the threshold."""
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    case_variants = compute_variants(log)
    counts = case_variants.value_counts()
    ordered = sorted(counts.index, key=lambda variant: (-counts[variant], variant))
    total = case_variants.size
    kept: set[tuple] = set()
    covered = 0
    for variant in ordered:
        kept.add(variant)
        covered += counts[variant]
        if covered / total >= coverage:
            break
    kept_cases = set(case_variants[case_variants.isin(kept)].index)
    return log[log[CASE_ID_COLUMN].isin(kept_cases)].reset_index(drop=True)


def inject_event_noise(
    log: pd.DataFrame, probability: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Perturb events: with probability p, skip/duplicate/insert/swap per event.

    Swaps exchange activity labels between neighbouring events of the same case
    so timestamps stay monotone; inserted/duplicated events receive a timestamp
    between their neighbours. Cases are never emptied (a skip that would remove
    a case's last remaining event becomes a no-op).
    """
    if not 0 <= probability <= 1:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if probability == 0:
        return log.copy()

    activities = sorted(log[ACTIVITY_COLUMN].unique())
    output_rows: list[dict[str, Any]] = []
    for case_id, group in log.groupby(CASE_ID_COLUMN, sort=False):
        case_activities = group[ACTIVITY_COLUMN].to_list()
        case_timestamps = group[TIMESTAMP_COLUMN].to_list()
        n_events = len(case_activities)
        perturb = rng.random(n_events) < probability
        operations = rng.integers(0, 4, size=n_events)
        case_rows: list[dict[str, Any]] = []
        position = 0
        while position < n_events:
            activity = case_activities[position]
            timestamp = case_timestamps[position]
            if not perturb[position]:
                case_rows.append(_event_row(case_id, activity, timestamp))
                position += 1
                continue
            operation = operations[position]
            if operation == 0:  # skip
                position += 1
            elif operation == 1:  # duplicate
                case_rows.append(_event_row(case_id, activity, timestamp))
                inserted_at = _between_timestamp(case_timestamps, position)
                case_rows.append(_event_row(case_id, activity, inserted_at))
                position += 1
            elif operation == 2:  # insert a random activity after this event
                case_rows.append(_event_row(case_id, activity, timestamp))
                inserted = activities[int(rng.integers(0, len(activities)))]
                inserted_at = _between_timestamp(case_timestamps, position)
                case_rows.append(_event_row(case_id, inserted, inserted_at))
                position += 1
            else:  # swap activity labels with the next event in the same case
                if position + 1 < n_events:
                    case_rows.append(_event_row(case_id, case_activities[position + 1], timestamp))
                    case_rows.append(_event_row(case_id, activity, case_timestamps[position + 1]))
                    position += 2
                else:
                    case_rows.append(_event_row(case_id, activity, timestamp))
                    position += 1
        if not case_rows:  # never emit an empty case
            case_rows.append(_event_row(case_id, case_activities[0], case_timestamps[0]))
        output_rows.extend(case_rows)

    noisy = pd.DataFrame(output_rows, columns=list(REQUIRED_COLUMNS))
    noisy[TIMESTAMP_COLUMN] = pd.to_datetime(noisy[TIMESTAMP_COLUMN], utc=True)
    return noisy


def truncate_traces(log: pd.DataFrame, max_events: int) -> pd.DataFrame:
    """Keep only the first ``max_events`` events of every case."""
    if max_events < 1:
        raise ValueError(f"max_events must be >= 1, got {max_events}")
    truncated = log.groupby(CASE_ID_COLUMN, sort=False).head(max_events)
    return truncated.reset_index(drop=True)


def filter_top_activities(log: pd.DataFrame, coverage: float) -> pd.DataFrame:
    """Keep the most frequent activities covering the given event share.

    Cases that lose all their events disappear. This is a structural stress
    operator: removing mid-trace events creates artificial directly-follows
    behaviour.
    """
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    counts = log[ACTIVITY_COLUMN].value_counts()
    ordered = sorted(counts.index, key=lambda activity: (-counts[activity], activity))
    total = len(log)
    kept: set[str] = set()
    covered = 0
    for activity in ordered:
        kept.add(activity)
        covered += counts[activity]
        if covered / total >= coverage:
            break
    return log[log[ACTIVITY_COLUMN].isin(kept)].reset_index(drop=True)


def validate_child_log(log: pd.DataFrame) -> str | None:
    """Return a rejection reason, or None if the child log is acceptable."""
    missing = [column for column in REQUIRED_COLUMNS if column not in log.columns]
    if missing:
        return f"missing required columns: {', '.join(missing)}"
    if not pd.api.types.is_datetime64_any_dtype(log[TIMESTAMP_COLUMN]):
        parsed = pd.to_datetime(log[TIMESTAMP_COLUMN], errors="coerce", utc=True)
        if parsed.isna().any():
            return "timestamps could not be parsed"
    elif log[TIMESTAMP_COLUMN].isna().any():
        return "timestamps could not be parsed"
    if log[[CASE_ID_COLUMN, ACTIVITY_COLUMN]].isna().any().any():
        return "log contains events with missing case id or activity"
    stats = compute_log_stats(log)
    if stats["n_traces"] < MIN_CHILD_TRACES:
        return f"too few traces: {stats['n_traces']} < {MIN_CHILD_TRACES}"
    if stats["n_activities"] < MIN_CHILD_ACTIVITIES:
        return f"too few activities: {stats['n_activities']} < {MIN_CHILD_ACTIVITIES}"
    if stats["n_variants"] < MIN_CHILD_VARIANTS:
        return f"too few variants: {stats['n_variants']} < {MIN_CHILD_VARIANTS}"
    return None


def default_augmentation_plan(
    parent_stats: dict[str, Any],
    *,
    include_stress: bool = False,
    large_log_traces: int = 10_000,
    long_trace_mean_length: float = 40.0,
    truncate_length: int = 50,
) -> list[AugmentationSpec]:
    """Default set of ~3 children per parent, plus size/shape-triggered extras."""
    plan = [
        AugmentationSpec("variant_coverage", {"coverage": 0.8}),
        AugmentationSpec("subsample", {"fraction": 0.5}),
        AugmentationSpec("noise", {"probability": 0.05}),
    ]
    if parent_stats["n_traces"] >= large_log_traces:
        plan.append(AugmentationSpec("subsample", {"fraction": 0.25}))
    if parent_stats["mean_trace_length"] >= long_trace_mean_length:
        plan.append(AugmentationSpec("truncate", {"max_events": truncate_length}))
    if include_stress:
        plan.extend(
            [
                AugmentationSpec("variant_coverage", {"coverage": 0.5}, stress=True),
                AugmentationSpec("noise", {"probability": 0.2}, stress=True),
                AugmentationSpec("top_activities", {"coverage": 0.8}, stress=True),
            ]
        )
    return plan


def child_seed(base_seed: int, parent_log_id: str, spec: AugmentationSpec) -> int:
    """Deterministic per-child seed derived from base seed, parent, and spec."""
    payload = {
        "base_seed": base_seed,
        "parent_log_id": parent_log_id,
        "operator": spec.operator,
        "parameters": spec.parameters,
    }
    return int(stable_hash(payload, length=8), 16)


def child_log_id(parent_log_id: str, spec: AugmentationSpec, seed: int) -> str:
    return f"{CHILD_LOG_ID_PREFIX}{parent_log_id}__{spec.token()}__s{seed}"


def apply_augmentation(
    log: pd.DataFrame, spec: AugmentationSpec, rng: np.random.Generator
) -> pd.DataFrame:
    operators: dict[str, Callable[[], pd.DataFrame]] = {
        "variant_coverage": lambda: filter_variant_coverage(log, spec.parameters["coverage"]),
        "subsample": lambda: subsample_variants(log, spec.parameters["fraction"], rng),
        "noise": lambda: inject_event_noise(log, spec.parameters["probability"], rng),
        "truncate": lambda: truncate_traces(log, spec.parameters["max_events"]),
        "top_activities": lambda: filter_top_activities(log, spec.parameters["coverage"]),
    }
    if spec.operator not in operators:
        raise ValueError(f"Unknown augmentation operator: {spec.operator}")
    return operators[spec.operator]()


def augment_parent_log(
    log: pd.DataFrame,
    parent_log_id: str,
    specs: list[AugmentationSpec],
    *,
    output_dir: str | Path,
    base_seed: int,
    parent_path: str = "",
    parent_sha256: str = "",
    overwrite: bool = False,
) -> list[ChildLogRecord]:
    """Generate, validate, and export child logs for one parent log."""
    output_dir = resolve_portable_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = canonicalize_event_log(log)

    records: list[ChildLogRecord] = []
    for spec in specs:
        seed = child_seed(base_seed, parent_log_id, spec)
        child_id = child_log_id(parent_log_id, spec, seed)
        output_path = output_dir / f"{child_id}.xes.gz"
        record = ChildLogRecord(
            child_log_id=child_id,
            parent_log_id=parent_log_id,
            parent_path=portable_project_path(parent_path) if parent_path else "",
            parent_sha256=parent_sha256,
            augmentation=spec.operator,
            parameters=spec.parameters,
            seed=seed,
            stress=spec.stress,
            status="accepted",
            rejection_reason=None,
            output_path=portable_project_path(output_path),
            artifact_sha256=None,
            n_traces=None,
            n_events=None,
            n_activities=None,
            n_variants=None,
        )
        if output_path.exists() and not overwrite:
            records.append(
                replace(
                    record,
                    status="skipped_existing",
                    rejection_reason="output file exists; rerun with --overwrite",
                )
            )
            continue
        rng = np.random.default_rng(seed)
        child = apply_augmentation(canonical, spec, rng)
        reason = validate_child_log(child)
        if reason is not None:
            records.append(
                replace(record, status="rejected", rejection_reason=reason, output_path=None)
            )
            continue
        _write_xes_gz(child, output_path)
        stats = compute_log_stats(child)
        records.append(
            replace(
                record,
                n_traces=stats["n_traces"],
                n_events=stats["n_events"],
                n_activities=stats["n_activities"],
                n_variants=stats["n_variants"],
                artifact_sha256=sha256_file(output_path),
            )
        )
    return records


def write_augmentation_manifest(records: list[ChildLogRecord], manifest_path: str | Path) -> Path:
    """Merge records into the augmentation manifest CSV (new rows win)."""
    manifest_path = resolve_portable_path(manifest_path)
    existing_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        existing_rows = existing.to_dict("records")

    replaced_ids = {
        record.child_log_id for record in records if record.status != "skipped_existing"
    }
    existing_ids = {str(row.get("child_log_id")) for row in existing_rows}
    rows = [row for row in existing_rows if str(row.get("child_log_id")) not in replaced_ids]
    rows.extend(
        record.to_row()
        for record in records
        if record.status != "skipped_existing" or record.child_log_id not in existing_ids
    )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest = manifest.sort_values(["parent_log_id", "child_log_id"], kind="stable")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    return manifest_path


def _write_xes_gz(log: pd.DataFrame, output_path: Path) -> None:
    write_canonical_xes(log, output_path)


def _event_row(case_id: str, activity: str, timestamp: Any) -> dict[str, Any]:
    return {
        CASE_ID_COLUMN: case_id,
        ACTIVITY_COLUMN: activity,
        TIMESTAMP_COLUMN: timestamp,
    }


def _between_timestamp(timestamps: list[Any], position: int) -> Any:
    current = timestamps[position]
    if position + 1 < len(timestamps):
        return current + (timestamps[position + 1] - current) / 2
    return current + pd.Timedelta(seconds=1)


def _percent_token(value: float) -> str:
    return f"{round(value * 100):03d}"
