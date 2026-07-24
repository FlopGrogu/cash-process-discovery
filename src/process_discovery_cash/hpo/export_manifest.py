"""Export a discovery-style manifest from completed HPO trials.

Metric manifests are generated only from a discovery source manifest
(``pdcash-generate-metric-manifest --source-manifest ...``). HPO trials do not
come from a pre-generated manifest, so this module reconstructs one after the
fact: every trial result file becomes one manifest row with exactly the
columns ``generate_manifest_rows`` would have produced (``build_trial_row``
mirrors it, and the config hash is recomputed from the recorded
hyperparameters and must match). Writing the export to
``experiments/manifests/v6/model/hpo/<algorithm>/v1.csv`` slots HPO results
into the standard v6 metric workflow: metric manifest and
``results/cluster/v6/metrics/hpo/...`` output root are derived from that path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.experiments.manifest import (
    _experiment_output_dir,
    _normalize_algorithm_ref,
    write_manifest,
)
from process_discovery_cash.hpo.trial_runner import (
    StudyContext,
    build_trial_row,
    trial_config_hash,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class HpoManifestExportStats:
    exported: int = 0
    skipped_other_algorithm: int = 0
    skipped_unreadable: int = 0
    skipped_hash_mismatch: int = 0
    skipped_paths: list[str] = field(default_factory=list)


def export_hpo_discovery_manifest(
    experiment_config_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, HpoManifestExportStats]:
    """Write one manifest row per evaluated trial of every study in the experiment.

    Rows are rebuilt from the ``hyperparameters`` recorded in each trial's
    result JSON; a row is exported only when its recomputed config hash matches
    the result file, so stale results from older code versions are skipped
    (and reported) instead of silently mislabeled. Failed/timeout trials are
    exported too — the metric pass mirrors every discovery row and evaluates
    rows without a model to zero metrics, exactly as for grid manifests.
    """
    experiment_config_path = Path(experiment_config_path)
    experiment = load_experiment_config(experiment_config_path)
    if experiment.hpo is None:
        raise ValueError(
            f"Experiment '{experiment.experiment_id}' ({experiment_config_path}) has no "
            "'hpo' block; this exporter only handles HPO experiments."
        )
    if output_path is None:
        output_path = experiment.manifest_path
    if not output_path:
        raise ValueError(
            f"Experiment '{experiment.experiment_id}' does not set manifest_path; "
            "pass --output or set manifest_path (e.g. "
            "experiments/manifests/v6/model/hpo/<algorithm>/v1.csv)."
        )

    stats = HpoManifestExportStats()
    rows: list[dict[str, str]] = []
    results_dir = _experiment_output_dir(experiment)
    for log_ref in experiment.logs:
        log_results_dir = results_dir / log_ref.log_id
        result_paths = sorted(log_results_dir.glob("*.json"))
        if not result_paths:
            continue
        for entry in experiment.algorithms:
            algorithm_ref = _normalize_algorithm_ref(entry)
            ctx = StudyContext.from_experiment(
                experiment_config_path, log_ref.log_id, algorithm_ref.name
            )
            rows.extend(_rows_for_study(ctx, result_paths, stats))

    rows.sort(key=lambda row: (row["log_id"], row["algorithm_id"], row["config_hash"]))
    manifest_path = write_manifest(rows, output_path)
    stats.exported = len(rows)
    return manifest_path, stats


def _rows_for_study(
    ctx: StudyContext,
    result_paths: list[Path],
    stats: HpoManifestExportStats,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result_path in result_paths:
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            stats.skipped_unreadable += 1
            stats.skipped_paths.append(result_path.as_posix())
            continue
        if not isinstance(payload, dict):
            stats.skipped_unreadable += 1
            stats.skipped_paths.append(result_path.as_posix())
            continue
        # Results of multiple algorithms may share one log directory when they
        # share an output template; keep only this study's algorithm.
        if payload.get("algorithm_name") != ctx.algorithm_config.algorithm_id:
            stats.skipped_other_algorithm += 1
            continue
        params = payload.get("hyperparameters")
        if not isinstance(params, dict):
            stats.skipped_unreadable += 1
            stats.skipped_paths.append(result_path.as_posix())
            continue
        config_hash = trial_config_hash(ctx, params)
        recorded_hash = _recorded_config_hash(payload, result_path)
        if recorded_hash != config_hash:
            stats.skipped_hash_mismatch += 1
            stats.skipped_paths.append(result_path.as_posix())
            LOGGER.warning(
                "Skipping %s: recomputed config hash %s does not match recorded %s "
                "(result predates a hash-affecting change?)",
                result_path,
                config_hash,
                recorded_hash,
            )
            continue
        rows.append(build_trial_row(ctx, params, config_hash))
    return rows


def _recorded_config_hash(payload: dict, result_path: Path) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("config_hash"):
        return str(metadata["config_hash"])
    return result_path.stem
