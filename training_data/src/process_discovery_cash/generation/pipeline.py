"""Generate, validate, and record feature-targeted synthetic logs.

Accepted logs are written like other synthetic logs (``syn_*`` ids) and are
training/validation data only — never final test logs. Rejected candidates are
quarantined with metadata so unreachable targets stay visible as evidence
about the feasible feature frontier.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_discovery_cash.data.augmentation import (
    canonicalize_event_log,
    inject_event_noise,
)
from process_discovery_cash.data.features import FEATURE_NAMES
from process_discovery_cash.data.preprocessing.metadata import sha256_file
from process_discovery_cash.data.xes import write_canonical_xes
from process_discovery_cash.generation.anchor import compute_log_feature_row
from process_discovery_cash.generation.feature_space import (
    NEAR_DUPLICATE_DISTANCE,
    TARGET_FEATURES,
    RealAnchor,
    assign_band,
    pairwise_binned_coverage,
    to_target_space,
)
from process_discovery_cash.generation.gedi_backend import (
    GEDI_FEATURE_MAP,
    GediBackend,
)
from process_discovery_cash.generation.targets import TargetSpec
from process_discovery_cash.utils.hashing import stable_hash, stable_json_dumps
from process_discovery_cash.utils.paths import portable_project_path, resolve_portable_path

LOG_ID_PREFIX = "syn_gedi_"
LOGS_DIRNAME = "logs"
REJECTED_DIRNAME = "rejected"
MANIFEST_FILENAME = "manifest.csv"
TARGETS_FILENAME = "targets.csv"
COVERAGE_FILENAME = "coverage.json"

MIN_TRACES = 10
MIN_ACTIVITIES = 3
MIN_VARIANTS = 2
MAX_TRACE_LENGTH = 5_000

# Tolerances in transformed target space; enforced only for the four axes GEDI
# optimizes directly (log10 axes: ~factor 2; variant ratio: absolute).
AXIS_TOLERANCES = {
    "num_traces": 0.30,
    "avg_trace_length": 0.30,
    "num_activities": 0.30,
    "variant_ratio": 0.25,
    "dfg_density": np.inf,
    "repetition_prevalence": np.inf,
}
ENFORCED_AXES = tuple(GEDI_FEATURE_MAP)

MANIFEST_COLUMNS = [
    "target_id",
    "log_id",
    "attempt",
    "seed",
    "status",
    "rejection_reason",
    "band_intended",
    "band_achieved",
    "band_reassigned",
    "output_path",
    "artifact_sha256",
    "concurrency",
    "noise_level",
    *[f"target_{feature}" for feature in TARGET_FEATURES],
    *[f"achieved_{feature}" for feature in TARGET_FEATURES],
    "achieved_features_json",
    "attainment_error",
    "axis_errors",
    "n_traces",
    "n_events",
    "n_activities",
    "n_variants",
    "duplicate_distance",
    "nearest_real_distance",
    "gedi_objectives",
    "gedi_incumbent",
]


@dataclass
class CandidateRecord:
    target_id: str
    log_id: str
    attempt: int
    seed: int
    status: str
    rejection_reason: str | None = None
    band_intended: str = ""
    band_achieved: str | None = None
    band_reassigned: bool = False
    output_path: str | None = None
    artifact_sha256: str | None = None
    concurrency: str = ""
    noise_level: float = 0.0
    target_values: dict[str, float] = field(default_factory=dict)
    achieved_values: dict[str, float] = field(default_factory=dict)
    achieved_features: dict[str, Any] = field(default_factory=dict)
    attainment_error: float | None = None
    axis_errors: dict[str, float] = field(default_factory=dict)
    n_traces: int | None = None
    n_events: int | None = None
    n_activities: int | None = None
    n_variants: int | None = None
    duplicate_distance: float | None = None
    nearest_real_distance: float | None = None
    gedi_objectives: dict[str, Any] = field(default_factory=dict)
    gedi_incumbent: dict[str, Any] | None = None

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "target_id": self.target_id,
            "log_id": self.log_id,
            "attempt": self.attempt,
            "seed": self.seed,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "band_intended": self.band_intended,
            "band_achieved": self.band_achieved,
            "band_reassigned": self.band_reassigned,
            "output_path": self.output_path,
            "artifact_sha256": self.artifact_sha256,
            "concurrency": self.concurrency,
            "noise_level": self.noise_level,
            "achieved_features_json": stable_json_dumps(self.achieved_features),
            "attainment_error": self.attainment_error,
            "axis_errors": stable_json_dumps(self.axis_errors),
            "n_traces": self.n_traces,
            "n_events": self.n_events,
            "n_activities": self.n_activities,
            "n_variants": self.n_variants,
            "duplicate_distance": self.duplicate_distance,
            "nearest_real_distance": self.nearest_real_distance,
            "gedi_objectives": stable_json_dumps(self.gedi_objectives),
            "gedi_incumbent": stable_json_dumps(self.gedi_incumbent or {}),
        }
        for feature in TARGET_FEATURES:
            row[f"target_{feature}"] = self.target_values.get(feature)
            row[f"achieved_{feature}"] = self.achieved_values.get(feature)
        return row

    def to_json_dict(self) -> dict[str, Any]:
        """Plain-dict form for per-target result JSON files."""
        return {
            "target_id": self.target_id,
            "log_id": self.log_id,
            "attempt": self.attempt,
            "seed": self.seed,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "band_intended": self.band_intended,
            "band_achieved": self.band_achieved,
            "band_reassigned": self.band_reassigned,
            "output_path": self.output_path,
            "artifact_sha256": self.artifact_sha256,
            "concurrency": self.concurrency,
            "noise_level": self.noise_level,
            "target_values": dict(self.target_values),
            "achieved_values": dict(self.achieved_values),
            "achieved_features": dict(self.achieved_features),
            "attainment_error": self.attainment_error,
            "axis_errors": dict(self.axis_errors),
            "n_traces": self.n_traces,
            "n_events": self.n_events,
            "n_activities": self.n_activities,
            "n_variants": self.n_variants,
            "duplicate_distance": self.duplicate_distance,
            "nearest_real_distance": self.nearest_real_distance,
            "gedi_objectives": dict(self.gedi_objectives),
            "gedi_incumbent": dict(self.gedi_incumbent) if self.gedi_incumbent else None,
        }


def candidate_record_from_json(payload: dict[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        target_id=str(payload["target_id"]),
        log_id=str(payload["log_id"]),
        attempt=int(payload["attempt"]),
        seed=int(payload["seed"]),
        status=str(payload["status"]),
        rejection_reason=payload.get("rejection_reason"),
        band_intended=str(payload.get("band_intended", "")),
        band_achieved=payload.get("band_achieved"),
        band_reassigned=bool(payload.get("band_reassigned", False)),
        output_path=payload.get("output_path"),
        artifact_sha256=payload.get("artifact_sha256"),
        concurrency=str(payload.get("concurrency", "")),
        noise_level=float(payload.get("noise_level", 0.0)),
        target_values=dict(payload.get("target_values") or {}),
        achieved_values=dict(payload.get("achieved_values") or {}),
        achieved_features=dict(payload.get("achieved_features") or {}),
        attainment_error=payload.get("attainment_error"),
        axis_errors=dict(payload.get("axis_errors") or {}),
        n_traces=payload.get("n_traces"),
        n_events=payload.get("n_events"),
        n_activities=payload.get("n_activities"),
        n_variants=payload.get("n_variants"),
        duplicate_distance=payload.get("duplicate_distance"),
        nearest_real_distance=payload.get("nearest_real_distance"),
        gedi_objectives=dict(payload.get("gedi_objectives") or {}),
        gedi_incumbent=payload.get("gedi_incumbent"),
    )


def run_single_target(
    target: TargetSpec,
    real_features: pd.DataFrame,
    backend: GediBackend,
    *,
    output_root: str | Path,
    base_seed: int,
    workdir: str | Path,
    max_attempts: int = 3,
    max_expected_events: float = 200_000.0,
    overwrite: bool = False,
    extra_reference_points: list[np.ndarray] | None = None,
) -> list[CandidateRecord]:
    """Run GEDI for one target and validate every attempt.

    Duplicate checking covers the real anchor plus any ``extra_reference_points``
    the caller threads in (the sequential runner passes previously accepted
    logs; parallel per-target execution passes none and defers cross-candidate
    dedup to aggregation).
    """
    output_root = resolve_portable_path(output_root)
    logs_dir = output_root / LOGS_DIRNAME
    rejected_dir = output_root / REJECTED_DIRNAME
    logs_dir.mkdir(parents=True, exist_ok=True)

    anchor = RealAnchor.from_features(real_features)
    real_points = to_target_space(real_features)
    reference_points = [real_points, *(extra_reference_points or [])]

    log_id = f"{LOG_ID_PREFIX}{target.target_id}"
    final_path = logs_dir / f"{log_id}.xes.gz"
    base_record = CandidateRecord(
        target_id=target.target_id,
        log_id=log_id,
        attempt=0,
        seed=0,
        status="",
        band_intended=target.band,
        concurrency=target.concurrency,
        noise_level=target.noise_level,
        target_values=dict(target.values),
    )
    if not target.feasible:
        base_record.status = "target_infeasible"
        base_record.rejection_reason = target.infeasible_reason
        return [base_record]
    if final_path.exists() and not overwrite:
        base_record.status = "skipped_existing"
        base_record.rejection_reason = "output file exists; rerun with --overwrite"
        base_record.output_path = portable_project_path(final_path)
        base_record.artifact_sha256 = sha256_file(final_path)
        return [base_record]

    records: list[CandidateRecord] = []
    for attempt in range(1, max_attempts + 1):
        seed = _candidate_seed(base_seed, target.target_id, attempt)
        record = CandidateRecord(
            target_id=target.target_id,
            log_id=log_id,
            attempt=attempt,
            seed=seed,
            status="",
            band_intended=target.band,
            concurrency=target.concurrency,
            noise_level=target.noise_level,
            target_values=dict(target.values),
        )
        attempt_dir = resolve_portable_path(workdir) / target.target_id / f"attempt_{attempt}"
        outcome = backend.generate(target, seed=seed, workdir=attempt_dir)
        record.gedi_objectives = outcome.spec.get("objectives", {})
        record.gedi_incumbent = outcome.incumbent
        if outcome.status != "success":
            record.status = "generation_failed"
            record.rejection_reason = (outcome.error or "unknown GEDI failure")[-500:]
            records.append(record)
            continue

        reason = _finalize_candidate(
            record,
            xes_path=outcome.xes_path,
            target=target,
            anchor=anchor,
            reference_points=reference_points,
            candidate_dir=attempt_dir,
            final_path=final_path,
            rejected_dir=rejected_dir,
            max_expected_events=max_expected_events,
        )
        records.append(record)
        if reason is None:
            break
    return records


def run_generation(
    targets: list[TargetSpec],
    real_features: pd.DataFrame,
    backend: GediBackend,
    *,
    output_root: str | Path,
    base_seed: int,
    workdir: str | Path,
    max_attempts: int = 3,
    max_expected_events: float = 200_000.0,
    overwrite: bool = False,
) -> list[CandidateRecord]:
    """Sequentially run every target, threading accepted logs into dedup."""
    output_root = resolve_portable_path(output_root)
    manifest_path = output_root / MANIFEST_FILENAME
    existing_rows = load_manifest(manifest_path).to_dict("records")

    accepted_points: list[np.ndarray] = []
    records: list[CandidateRecord] = []
    for target in targets:
        target_records = run_single_target(
            target,
            real_features,
            backend,
            output_root=output_root,
            base_seed=base_seed,
            workdir=workdir,
            max_attempts=max_attempts,
            max_expected_events=max_expected_events,
            overwrite=overwrite,
            extra_reference_points=accepted_points,
        )
        records.extend(target_records)
        _write_manifest(records, manifest_path, existing_rows)
        last = target_records[-1]
        if last.status == "accepted":
            accepted_points.append(to_target_space(last.achieved_values))
    return records


def _finalize_candidate(
    record: CandidateRecord,
    *,
    xes_path: Path,
    target: TargetSpec,
    anchor: RealAnchor,
    reference_points: list[np.ndarray],
    candidate_dir: Path,
    final_path: Path,
    rejected_dir: Path,
    max_expected_events: float,
) -> str | None:
    """Validate one generated candidate; move it to its final location if OK.

    The canonical extractor is file-based, so the (noise-injected) candidate is
    written to the attempt directory first, measured from disk, and only then
    moved to ``logs/`` or quarantined under ``rejected/``.
    """
    try:
        import pm4py

        # Do not let pm4py choose a parser based on which optional packages
        # happen to be installed. The generated candidate is small enough for
        # the stable pure-Python parser and this keeps validation identical
        # across native and emulated container hosts.
        frame = pm4py.read_xes(
            str(xes_path),
            variant="chunk_regex",
            return_legacy_log_object=False,
        )
        log = canonicalize_event_log(frame)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a rejection
        record.status = "rejected"
        record.rejection_reason = f"could not parse generated log: {exc}"[:500]
        return record.rejection_reason

    if record.noise_level > 0:
        rng = np.random.default_rng(record.seed)
        log = inject_event_noise(log, record.noise_level, rng)

    candidate_path = candidate_dir / f"{record.log_id}_a{record.attempt}.xes.gz"
    _write_xes_gz(log, candidate_path)
    try:
        stats = _measure_candidate(
            record,
            candidate_path,
            target=target,
            anchor=anchor,
            reference_points=reference_points,
        )
    except Exception as exc:  # noqa: BLE001 - unmeasurable candidates are rejections
        record.status = "rejected"
        record.rejection_reason = f"could not measure generated log: {exc}"[:500]
        candidate_path.unlink(missing_ok=True)
        return record.rejection_reason

    reason = _rejection_reason(record, stats, max_expected_events)
    if reason is not None:
        record.status = "rejected"
        record.rejection_reason = reason
        rejected_dir.mkdir(parents=True, exist_ok=True)
        quarantine = rejected_dir / candidate_path.name
        shutil.move(str(candidate_path), quarantine)
        record.output_path = portable_project_path(quarantine)
        record.artifact_sha256 = sha256_file(quarantine)
        return reason

    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidate_path), final_path)
    record.status = "accepted"
    record.output_path = portable_project_path(final_path)
    record.artifact_sha256 = sha256_file(final_path)
    return None


def _measure_candidate(
    record: CandidateRecord,
    xes_path: Path,
    *,
    target: TargetSpec | None,
    anchor: RealAnchor,
    reference_points: list[np.ndarray],
) -> dict[str, float]:
    """Measure a candidate file with the canonical extractor and fill the record.

    Returns the validation stats (min/max trace length) for `_rejection_reason`.
    Geometry (errors, bands, distances) is only computed when every target axis
    is finite; otherwise `_rejection_reason` rejects the log first.
    """
    row = compute_log_feature_row(xes_path, log_id=record.log_id)
    record.achieved_features = {name: row.get(name) for name in FEATURE_NAMES}
    record.achieved_values = {
        feature: float(row.get(feature, float("nan"))) for feature in TARGET_FEATURES
    }
    record.n_traces = _safe_count(row.get("num_traces"))
    record.n_events = _safe_count(row.get("num_events"))
    record.n_activities = _safe_count(row.get("num_activities"))
    record.n_variants = _safe_count(row.get("num_variants"))

    if all(math.isfinite(value) for value in record.achieved_values.values()):
        target_values = target.values if target is not None else record.target_values
        target_point = to_target_space(target_values)[0]
        achieved_point = to_target_space(record.achieved_values)[0]
        record.axis_errors = {
            feature: float(abs(achieved_point[index] - target_point[index]))
            for index, feature in enumerate(TARGET_FEATURES)
        }
        record.attainment_error = max(
            record.axis_errors[feature] / AXIS_TOLERANCES[feature] for feature in ENFORCED_AXES
        )
        record.nearest_real_distance = float(
            anchor.nearest_real_distance(np.atleast_2d(achieved_point))[0]
        )
        record.band_achieved = assign_band(record.nearest_real_distance)
        record.band_reassigned = record.band_achieved != record.band_intended
        reference = np.vstack(reference_points)
        record.duplicate_distance = float(
            anchor.nearest_distance_to(np.atleast_2d(achieved_point), reference)[0]
        )
    return {
        "min_trace_length": float(row.get("min_trace_length", float("nan"))),
        "max_trace_length": float(row.get("max_trace_length", float("nan"))),
    }


def _rejection_reason(
    record: CandidateRecord,
    stats: dict[str, float],
    max_expected_events: float,
) -> str | None:
    if record.n_traces < MIN_TRACES:
        return f"too few traces: {record.n_traces} < {MIN_TRACES}"
    if record.n_activities < MIN_ACTIVITIES:
        return f"too few activities: {record.n_activities} < {MIN_ACTIVITIES}"
    if record.n_variants < MIN_VARIANTS:
        return f"too few variants: {record.n_variants} < {MIN_VARIANTS}"
    min_trace_length = stats.get("min_trace_length", float("nan"))
    if not math.isfinite(min_trace_length) or min_trace_length < 1:
        return "log contains empty or unmeasurable traces"
    if not all(math.isfinite(value) for value in record.achieved_values.values()):
        missing = [
            feature for feature, value in record.achieved_values.items() if not math.isfinite(value)
        ]
        return f"target axis features not computable: {', '.join(missing)}"
    if record.n_events > 1.5 * max_expected_events:
        return f"pathological size: {record.n_events} events exceeds compute cap"
    max_trace_length = stats.get("max_trace_length", float("nan"))
    if math.isfinite(max_trace_length) and max_trace_length > MAX_TRACE_LENGTH:
        return f"pathological trace length: {max_trace_length:.0f} > {MAX_TRACE_LENGTH}"
    if record.attainment_error is not None and record.attainment_error > 1.0:
        worst = max(
            ((f, record.axis_errors[f] / AXIS_TOLERANCES[f]) for f in ENFORCED_AXES),
            key=lambda item: item[1],
        )
        return f"target attainment too poor: {worst[0]} off by {record.axis_errors[worst[0]]:.3f}"
    duplicate_distance = record.duplicate_distance
    if duplicate_distance is not None and duplicate_distance < NEAR_DUPLICATE_DISTANCE:
        return (
            "near-duplicate of a real or accepted log "
            f"(standardized distance {record.duplicate_distance:.3f})"
        )
    return None


def revalidate_existing(
    manifest_path: str | Path,
    real_features: pd.DataFrame,
    *,
    max_expected_events: float = 200_000.0,
) -> list[CandidateRecord]:
    """Recompute features and re-run validation for accepted logs on disk."""
    manifest_path = Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    anchor = RealAnchor.from_features(real_features)
    real_points = to_target_space(real_features)

    records: list[CandidateRecord] = []
    accepted = manifest[manifest["status"] == "accepted"]
    for _, row in accepted.iterrows():
        record = CandidateRecord(
            target_id=str(row["target_id"]),
            log_id=str(row["log_id"]),
            attempt=int(row["attempt"]),
            seed=int(row["seed"]),
            status="accepted",
            band_intended=str(row["band_intended"]),
            concurrency=str(row["concurrency"]),
            noise_level=float(row["noise_level"]),
            target_values={feature: float(row[f"target_{feature}"]) for feature in TARGET_FEATURES},
            output_path=str(row["output_path"]),
        )
        path = Path(record.output_path)
        if not path.exists():
            record.status = "rejected"
            record.rejection_reason = f"accepted log missing on disk: {path}"
            records.append(record)
            continue
        try:
            stats = _measure_candidate(
                record,
                path,
                target=None,
                anchor=anchor,
                reference_points=[real_points],
            )
        except Exception as exc:  # noqa: BLE001
            record.status = "rejected"
            record.rejection_reason = f"could not measure stored log: {exc}"[:500]
            records.append(record)
            continue
        reason = _rejection_reason(record, stats, max_expected_events)
        if reason is not None and "near-duplicate" not in reason:
            record.status = "rejected"
            record.rejection_reason = reason
        records.append(record)
    return records


def coverage_report(
    real_features: pd.DataFrame,
    accepted_features: list[dict[str, float]],
    bounds: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Pairwise binned coverage before/after plus named feature-space holes."""
    real_points = to_target_space(real_features)
    report: dict[str, Any] = {
        "real_only": pairwise_binned_coverage(real_points, bounds),
    }
    if accepted_features:
        accepted_points = np.vstack([to_target_space(row) for row in accepted_features])
        combined = np.vstack([real_points, accepted_points])
        report["real_plus_generated"] = pairwise_binned_coverage(combined, bounds)
    else:
        report["real_plus_generated"] = report["real_only"]

    holes = {
        "mid_activity_vocabulary (20<=activities<=120)": lambda row: (
            20 <= row["num_activities"] <= 120
        ),
        "medium_variant_ratio (0.25<=vr<=0.6)": lambda row: 0.25 <= row["variant_ratio"] <= 0.6,
        "large_with_diversity (traces>5000, activities>20)": lambda row: (
            row["num_traces"] > 5000 and row["num_activities"] > 20
        ),
        "long_nontrivial (len>30, vr>0.2)": lambda row: (
            row["avg_trace_length"] > 30 and row["variant_ratio"] > 0.2
        ),
        "dense_dfg (density>0.3)": lambda row: row["dfg_density"] > 0.3,
    }
    real_rows = real_features[TARGET_FEATURES].to_dict("records")
    report["holes"] = {
        name: {
            "real": sum(1 for row in real_rows if predicate(row)),
            "generated": sum(1 for row in accepted_features if predicate(row)),
        }
        for name, predicate in holes.items()
    }
    return report


def _write_manifest(
    records: list[CandidateRecord],
    manifest_path: Path,
    existing_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Merge this run's records into the manifest; processed targets win."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    processed = {record.target_id for record in records if record.status != "skipped_existing"}
    existing_targets = {str(row.get("target_id")) for row in existing_rows or []}
    rows = [row for row in existing_rows or [] if str(row.get("target_id")) not in processed]
    rows.extend(
        record.to_row()
        for record in records
        if record.status != "skipped_existing" or record.target_id not in existing_targets
    )
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    frame.to_csv(manifest_path, index=False)


def _candidate_seed(base_seed: int, target_id: str, attempt: int) -> int:
    payload = {"base_seed": base_seed, "target_id": target_id, "attempt": attempt}
    return int(stable_hash(payload, length=8), 16)


def _safe_count(value: Any) -> int:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(value):
        return 0
    return int(value)


def _write_xes_gz(log: pd.DataFrame, output_path: Path) -> None:
    write_canonical_xes(log, output_path)


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = resolve_portable_path(manifest_path)
    if not manifest_path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(manifest_path)


def summarize_records(records: list[CandidateRecord]) -> dict[str, Any]:
    statuses = pd.Series([record.status for record in records])
    reasons = pd.Series(
        [record.rejection_reason or "" for record in records if record.status == "rejected"]
    )
    summary: dict[str, Any] = {"by_status": statuses.value_counts().to_dict()}
    if not reasons.empty:
        summary["rejection_reasons"] = reasons.str.split(":").str[0].value_counts().to_dict()
    accepted = [record for record in records if record.status == "accepted"]
    if accepted:
        summary["mean_attainment_error"] = float(
            np.mean([record.attainment_error for record in accepted])
        )
        summary["band_achieved"] = (
            pd.Series([record.band_achieved for record in accepted]).value_counts().to_dict()
        )
    return summary
