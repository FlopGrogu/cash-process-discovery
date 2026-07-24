"""Study manifest: one CSV row per (log, algorithm) HPO study.

The Slurm array wrapper maps ``SLURM_ARRAY_TASK_ID`` to a row of this file,
the same way row manifests drive the discovery arrays.
"""

from __future__ import annotations

import csv
from pathlib import Path

from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.experiments.manifest import (
    _experiment_output_dir,
    _normalize_algorithm_ref,
)
from process_discovery_cash.hpo.study import (
    journal_path_for,
    study_name_for,
    summary_path_for,
)
from process_discovery_cash.utils.paths import portable_project_path, resolve_portable_path

STUDY_MANIFEST_COLUMNS = [
    "study_index",
    "study_name",
    "experiment_id",
    "experiment_config_path",
    "log_id",
    "algorithm_name",
    "algorithm_id",
    "journal_path",
    "results_dir",
    "summary_path",
    "log_dir",
]


def generate_hpo_study_rows(
    experiment_config_paths: list[str | Path],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for config_path in experiment_config_paths:
        config_path = Path(config_path)
        experiment = load_experiment_config(config_path)
        if experiment.hpo is None:
            raise ValueError(
                f"Experiment '{experiment.experiment_id}' ({config_path}) has no 'hpo' "
                "block; study manifests are only generated for HPO experiments."
            )
        results_dir = _experiment_output_dir(experiment)
        for log_ref in experiment.logs:
            for entry in experiment.algorithms:
                algorithm_ref = _normalize_algorithm_ref(entry)
                study_name = study_name_for(
                    experiment.experiment_id, log_ref.log_id, algorithm_ref.name
                )
                journal_path = journal_path_for(
                    experiment.hpo.storage_root, experiment.experiment_id, study_name
                )
                summary_path = summary_path_for(
                    experiment.hpo.storage_root,
                    experiment.experiment_id,
                    experiment.hpo.summary_dirname,
                    study_name,
                )
                rows.append(
                    {
                        "study_index": str(len(rows)),
                        "study_name": study_name,
                        "experiment_id": experiment.experiment_id,
                        "experiment_config_path": portable_project_path(config_path),
                        "log_id": log_ref.log_id,
                        "algorithm_name": algorithm_ref.name,
                        "algorithm_id": str(algorithm_ref.algorithm_id or algorithm_ref.name),
                        "journal_path": journal_path.as_posix(),
                        "results_dir": results_dir.as_posix(),
                        "summary_path": summary_path.as_posix(),
                        "log_dir": experiment.output.log_dir,
                    }
                )
    return rows


def default_study_manifest_path(experiment_config_path: str | Path) -> Path:
    """Mirror a canonical v6 HPO config below the v6 manifest namespace."""
    config_path = Path(experiment_config_path)
    parts = config_path.parts
    marker = ("configs", "experiments")
    for index in range(len(parts) - len(marker) + 1):
        if parts[index : index + len(marker)] != marker:
            continue
        suffix = list(parts[index + len(marker) :])
        if len(suffix) >= 4 and suffix[:2] == ["v6", "hpo"]:
            suffix[-1] = Path(suffix[-1]).stem
            return Path("experiments/manifests").joinpath(*suffix, "studies.csv")
        break
    raise ValueError(
        "HPO study manifests are supported only for canonical configs below "
        "configs/experiments/v6/hpo/<algorithm>/<version>.yaml; "
        f"got {config_path.as_posix()}"
    )


def write_study_manifest(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    output_path = resolve_portable_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STUDY_MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def load_study_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
