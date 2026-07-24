from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import yaml

from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.experiments.manifest import generate_manifest
from process_discovery_cash.utils.paths import project_root

DEFAULT_V6_BASELINE_CONFIG_GLOB = "configs/experiments/v6/baseline/*/*.yaml"
DEFAULT_V6_DEFAULT_RUN_SURVEY_CONFIG_GLOB = (
    "configs/experiments/v6/default_run_survey/*/*.yaml"
)
DEFAULT_V6_EXPLORE_CONFIG_GLOB = "configs/experiments/v6/explore/*/*.yaml"
DEFAULT_V6_SYNTHETIC_EXPLORE_CONFIG_GLOB = "configs/experiments/v6/explore_synthetic/*/*.yaml"
DEFAULT_V6_AUGMENTATION_MANIFEST = Path("data/augmented/manifest.csv")
DEFAULT_V6_AUGMENTED_EXPLORE_CONFIG_ROOT = Path("configs/experiments/v6/explore_augmented")
DEFAULT_V6_AUGMENTED_EXPERIMENT_SLUG = "explore_augmented"
DEFAULT_V6_SYNTHETIC_MANIFEST = Path("data/synthetic/gedi/manifest.csv")
DEFAULT_V6_SYNTHETIC_LOGS_DIR = Path("data/synthetic/gedi/logs")
DEFAULT_V6_SYNTHETIC_EXPLORE_CONFIG_ROOT = Path("configs/experiments/v6/explore_synthetic")
DEFAULT_V6_SYNTHETIC_EXPERIMENT_SLUG = "explore_synthetic"
DEFAULT_V6_OBJECTIVE_METRICS = ["fitness", "precision", "generalization", "simplicity"]
V6_DEFAULT_RUN_SURVEY_ALGORITHMS = {
    "alpha_classic",
    "alpha_plus",
    "genetic",
    "heuristic_classic",
    "heuristic_plusplus",
    "ilp",
    "inductive_im",
    "inductive_imd",
    "inductive_imf",
    "split",
}
V6_BASELINE_ALGORITHMS = V6_DEFAULT_RUN_SURVEY_ALGORITHMS


def discover_v6_baseline_configs(
    config_glob: str = DEFAULT_V6_BASELINE_CONFIG_GLOB,
) -> list[Path]:
    latest_by_directory: dict[Path, Path] = {}
    for path in sorted(Path.cwd().glob(config_glob)):
        if (
            config_glob == DEFAULT_V6_BASELINE_CONFIG_GLOB
            and path.parent.name not in V6_BASELINE_ALGORITHMS
        ):
            continue
        current = latest_by_directory.get(path.parent)
        if current is None or _v6_config_version_key(path) > _v6_config_version_key(current):
            latest_by_directory[path.parent] = path
    return sorted(latest_by_directory.values())


def generate_v6_manifests(
    *,
    require_artifacts: bool = False,
    config_glob: str = DEFAULT_V6_BASELINE_CONFIG_GLOB,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    config_paths = discover_v6_baseline_configs(config_glob)
    written: dict[str, Path] = {}
    for config_path in config_paths:
        manifest_path = generate_manifest(
            config_path,
            _v6_manifest_destination(config_path, output_root),
            require_artifacts=require_artifacts,
        )
        written[config_path.parent.name] = manifest_path
    return written


def discover_v6_default_run_survey_configs(
    config_glob: str = DEFAULT_V6_DEFAULT_RUN_SURVEY_CONFIG_GLOB,
) -> list[Path]:
    config_paths = discover_v6_baseline_configs(config_glob)
    if config_glob == DEFAULT_V6_DEFAULT_RUN_SURVEY_CONFIG_GLOB:
        discovered = {path.parent.name for path in config_paths}
        missing = V6_DEFAULT_RUN_SURVEY_ALGORITHMS - discovered
        unexpected = discovered - V6_DEFAULT_RUN_SURVEY_ALGORITHMS
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if unexpected:
                details.append(f"unexpected: {', '.join(sorted(unexpected))}")
            raise ValueError(
                f"Invalid v6 default-run survey algorithm set ({'; '.join(details)})"
            )
    return config_paths


def generate_v6_default_run_survey_manifests(
    *,
    require_artifacts: bool = False,
    config_glob: str = DEFAULT_V6_DEFAULT_RUN_SURVEY_CONFIG_GLOB,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for config_path in discover_v6_default_run_survey_configs(config_glob):
        written[config_path.parent.name] = generate_manifest(
            config_path,
            _v6_manifest_destination(config_path, output_root),
            require_artifacts=require_artifacts,
        )
    return written


def discover_v6_ordinary_configs() -> list[Path]:
    """Return the 40 canonical non-HPO v6 configurations."""
    return sorted(
        [
            *discover_v6_primary_configs(),
            *discover_v6_baseline_configs(DEFAULT_V6_DEFAULT_RUN_SURVEY_CONFIG_GLOB),
        ]
    )


def discover_v6_primary_configs() -> list[Path]:
    """Return the 30 baseline and explore configurations in the primary workflow."""
    paths: list[Path] = []
    for pattern in (
        DEFAULT_V6_BASELINE_CONFIG_GLOB,
        DEFAULT_V6_EXPLORE_CONFIG_GLOB,
        DEFAULT_V6_SYNTHETIC_EXPLORE_CONFIG_GLOB,
    ):
        paths.extend(discover_v6_baseline_configs(pattern))
    return sorted(paths)


def generate_v6_primary_manifests(
    *,
    require_artifacts: bool = False,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Generate the 30 baseline and explore manifests in the primary workflow."""
    return _generate_v6_config_manifests(
        discover_v6_primary_configs(),
        require_artifacts=require_artifacts,
        output_root=output_root,
    )


def generate_all_v6_ordinary_manifests(
    *,
    require_artifacts: bool = False,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Generate all 40 ordinary v6 manifests without touching tracked files."""
    return _generate_v6_config_manifests(
        discover_v6_ordinary_configs(),
        require_artifacts=require_artifacts,
        output_root=output_root,
    )


def _generate_v6_config_manifests(
    config_paths: list[Path],
    *,
    require_artifacts: bool,
    output_root: str | Path | None,
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for config_path in config_paths:
        key = (
            config_path.relative_to(project_root() / "configs/experiments/v6")
            .with_suffix("")
            .as_posix()
        )
        written[key] = generate_manifest(
            config_path,
            _v6_manifest_destination(config_path, output_root),
            require_artifacts=require_artifacts,
        )
    return written


def load_v6_augmented_log_refs(
    manifest_path: str | Path = DEFAULT_V6_AUGMENTATION_MANIFEST,
    *,
    include_stress: bool = True,
    require_log_files: bool = False,
) -> list[dict[str, str]]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    logs: list[dict[str, str]] = []
    for row in rows:
        if row.get("status") != "accepted":
            continue
        if not include_stress and _truthy(row.get("stress")):
            continue
        log_id = row.get("child_log_id", "")
        output_path = row.get("output_path", "")
        if not log_id or not output_path:
            continue
        if require_log_files and not Path(output_path).exists():
            raise FileNotFoundError(f"Augmented log file is missing: {output_path}")
        logs.append(
            {
                "log_id": log_id,
                "dataset_id": log_id,
                "path": output_path,
            }
        )

    if not logs:
        raise ValueError(f"No accepted augmented logs found in {path}")
    return sorted(logs, key=lambda row: row["log_id"])


def prepare_v6_augmented_explore_configs(
    *,
    augmentation_manifest: str | Path = DEFAULT_V6_AUGMENTATION_MANIFEST,
    source_config_glob: str = DEFAULT_V6_EXPLORE_CONFIG_GLOB,
    output_root: str | Path = DEFAULT_V6_AUGMENTED_EXPLORE_CONFIG_ROOT,
    include_stress: bool = True,
    require_log_files: bool = False,
) -> dict[str, Path]:
    logs = load_v6_augmented_log_refs(
        augmentation_manifest,
        include_stress=include_stress,
        require_log_files=require_log_files,
    )
    output_root = Path(output_root)
    written: dict[str, Path] = {}
    for source_path in discover_v6_baseline_configs(source_config_glob):
        algorithm_slug = source_path.parent.name
        version = source_path.stem
        payload = _load_yaml(source_path)
        payload["experiment_id"] = (
            f"v6_{DEFAULT_V6_AUGMENTED_EXPERIMENT_SLUG}_{algorithm_slug}_{version}"
        )
        payload["logs"] = logs
        output = dict(payload.get("output") or {})
        output["results_dir"] = (
            f"results/cluster/v6/model/{DEFAULT_V6_AUGMENTED_EXPERIMENT_SLUG}/{algorithm_slug}"
        )
        output["log_dir"] = (
            f"logs/slurm/v6/model/{DEFAULT_V6_AUGMENTED_EXPERIMENT_SLUG}/{algorithm_slug}/{version}"
        )
        payload["output"] = output
        payload["manifest_path"] = (
            f"experiments/manifests/v6/model/{DEFAULT_V6_AUGMENTED_EXPERIMENT_SLUG}/"
            f"{algorithm_slug}/{version}.csv"
        )

        destination = output_root / algorithm_slug / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        written[algorithm_slug] = destination
    return written


def generate_v6_augmented_explore_manifests(
    *,
    augmentation_manifest: str | Path = DEFAULT_V6_AUGMENTATION_MANIFEST,
    require_artifacts: bool = False,
    source_config_glob: str = DEFAULT_V6_EXPLORE_CONFIG_GLOB,
    output_config_root: str | Path = DEFAULT_V6_AUGMENTED_EXPLORE_CONFIG_ROOT,
    include_stress: bool = True,
    require_log_files: bool = False,
) -> dict[str, Path]:
    config_paths = prepare_v6_augmented_explore_configs(
        augmentation_manifest=augmentation_manifest,
        source_config_glob=source_config_glob,
        output_root=output_config_root,
        include_stress=include_stress,
        require_log_files=require_log_files,
    )
    written: dict[str, Path] = {}
    for algorithm_slug, config_path in config_paths.items():
        written[algorithm_slug] = generate_manifest(
            config_path,
            require_artifacts=require_artifacts,
        )
    return written


def load_v6_synthetic_log_refs(
    manifest_path: str | Path = DEFAULT_V6_SYNTHETIC_MANIFEST,
    *,
    logs_dir: str | Path = DEFAULT_V6_SYNTHETIC_LOGS_DIR,
    require_log_files: bool = False,
) -> list[dict[str, str]]:
    """Accepted GEDI synthetic logs as v6 log references.

    Every accepted manifest row becomes one log ref. The log file path is
    rebuilt from ``log_id`` under ``logs_dir`` (portable), not read from the
    manifest ``output_path`` column, because rows produced on the cluster carry
    absolute cluster paths that do not resolve on other machines.
    """
    path = Path(manifest_path)
    logs_dir = Path(logs_dir)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    logs: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("status") != "accepted":
            continue
        log_id = row.get("log_id", "")
        if not log_id or log_id in seen:
            continue
        seen.add(log_id)
        log_path = logs_dir / f"{log_id}.xes.gz"
        if require_log_files and not log_path.exists():
            raise FileNotFoundError(f"Synthetic log file is missing: {log_path}")
        logs.append(
            {
                "log_id": log_id,
                "dataset_id": log_id,
                "path": log_path.as_posix(),
            }
        )

    if not logs:
        raise ValueError(f"No accepted synthetic logs found in {path}")
    return sorted(logs, key=lambda row: row["log_id"])


def prepare_v6_synthetic_explore_configs(
    *,
    synthetic_manifest: str | Path = DEFAULT_V6_SYNTHETIC_MANIFEST,
    logs_dir: str | Path = DEFAULT_V6_SYNTHETIC_LOGS_DIR,
    source_config_glob: str = DEFAULT_V6_EXPLORE_CONFIG_GLOB,
    output_root: str | Path = DEFAULT_V6_SYNTHETIC_EXPLORE_CONFIG_ROOT,
    require_log_files: bool = False,
) -> dict[str, Path]:
    logs = load_v6_synthetic_log_refs(
        synthetic_manifest,
        logs_dir=logs_dir,
        require_log_files=require_log_files,
    )
    output_root = Path(output_root)
    written: dict[str, Path] = {}
    for source_path in discover_v6_baseline_configs(source_config_glob):
        algorithm_slug = source_path.parent.name
        version = source_path.stem
        payload = _load_yaml(source_path)
        payload["experiment_id"] = (
            f"v6_{DEFAULT_V6_SYNTHETIC_EXPERIMENT_SLUG}_{algorithm_slug}_{version}"
        )
        payload["logs"] = logs
        output = dict(payload.get("output") or {})
        output["results_dir"] = (
            f"results/cluster/v6/model/{DEFAULT_V6_SYNTHETIC_EXPERIMENT_SLUG}/"
            f"{algorithm_slug}/{version}"
        )
        output["log_dir"] = (
            f"logs/slurm/v6/model/{DEFAULT_V6_SYNTHETIC_EXPERIMENT_SLUG}/{algorithm_slug}/{version}"
        )
        payload["output"] = output
        payload["manifest_path"] = (
            f"experiments/manifests/v6/model/{DEFAULT_V6_SYNTHETIC_EXPERIMENT_SLUG}/"
            f"{algorithm_slug}/{version}.csv"
        )

        destination = output_root / algorithm_slug / source_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        written[algorithm_slug] = destination
    return written


def generate_v6_synthetic_explore_manifests(
    *,
    synthetic_manifest: str | Path = DEFAULT_V6_SYNTHETIC_MANIFEST,
    logs_dir: str | Path = DEFAULT_V6_SYNTHETIC_LOGS_DIR,
    require_artifacts: bool = False,
    source_config_glob: str = DEFAULT_V6_EXPLORE_CONFIG_GLOB,
    output_config_root: str | Path = DEFAULT_V6_SYNTHETIC_EXPLORE_CONFIG_ROOT,
    require_log_files: bool = False,
) -> dict[str, Path]:
    config_paths = prepare_v6_synthetic_explore_configs(
        synthetic_manifest=synthetic_manifest,
        logs_dir=logs_dir,
        source_config_glob=source_config_glob,
        output_root=output_config_root,
        require_log_files=require_log_files,
    )
    written: dict[str, Path] = {}
    for algorithm_slug, config_path in config_paths.items():
        written[algorithm_slug] = generate_manifest(
            config_path,
            require_artifacts=require_artifacts,
        )
    return written


_VERSION_RE = re.compile(r"^v(\d+(?:\.\d+)*)$")


def _v6_config_version_key(path: Path) -> tuple[int, ...] | tuple[int, str]:
    match = _VERSION_RE.match(path.stem)
    if match is None:
        return (-1, path.stem)
    return tuple(int(part) for part in match.group(1).split("."))


def select_best_v6_configs(
    input_paths: list[str | Path],
    output_path: str | Path,
    *,
    objective_metrics: list[str] = DEFAULT_V6_OBJECTIVE_METRICS,
) -> Path:
    candidates = []
    for input_path in input_paths:
        candidates.extend(_load_best_config_candidates(input_path, objective_metrics))

    best_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["log_id"], candidate["algorithm_name"])
        previous = best_by_pair.get(key)
        if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
            best_by_pair[key] = candidate

    rows = [_best_config_output_row(row, objective_metrics) for row in best_by_pair.values()]
    rows.sort(key=lambda row: (row["log_id"], row["algorithm_name"], row["config_hash"]))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _best_config_fieldnames(rows, objective_metrics)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _load_best_config_candidates(
    input_path: str | Path,
    objective_metrics: list[str],
) -> list[dict[str, Any]]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate = _candidate_from_row(row, objective_metrics)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _candidate_from_row(
    row: dict[str, str],
    objective_metrics: list[str],
) -> dict[str, Any] | None:
    status = _first_value(row, ["status_metrics", "status"])
    if status and status != "success":
        return None
    metric_values: dict[str, float] = {}
    for metric in objective_metrics:
        value = _parse_float(_first_value(row, [f"metric_{metric}_metrics", f"metric_{metric}"]))
        if value is None:
            return None
        status_value = _first_value(
            row,
            [f"metric_status_{metric}_metrics", f"metric_status_{metric}"],
        )
        if status_value and status_value != "success":
            return None
        metric_values[metric] = value

    log_id = _first_value(row, ["log_id_metrics", "log_id_discovery", "log_id"])
    algorithm_name = _first_value(
        row,
        ["algorithm_name_metrics", "algorithm_name_discovery", "algorithm_name", "algorithm_id"],
    )
    config_hash = _first_value(row, ["source_config_hash", "config_hash"])
    if not log_id or not algorithm_name or not config_hash:
        return None

    runtime_seconds = _parse_float(
        _first_value(row, ["runtime_seconds_discovery", "runtime_seconds"])
    )
    score = sum(metric_values.values()) / len(metric_values)
    return {
        "raw": dict(row),
        "log_id": log_id,
        "algorithm_name": algorithm_name,
        "config_hash": config_hash,
        "objective_score": score,
        "objective_metrics": metric_values,
        "runtime_seconds": runtime_seconds,
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, str]:
    runtime_seconds = candidate.get("runtime_seconds")
    if runtime_seconds is None or math.isnan(float(runtime_seconds)):
        runtime_seconds = math.inf
    return (
        -float(candidate["objective_score"]),
        float(runtime_seconds),
        str(candidate["config_hash"]),
    )


def _best_config_output_row(
    candidate: dict[str, Any],
    objective_metrics: list[str],
) -> dict[str, Any]:
    raw = candidate["raw"]
    row = {
        "log_id": candidate["log_id"],
        "algorithm_name": candidate["algorithm_name"],
        "config_hash": candidate["config_hash"],
        "objective_score": f"{candidate['objective_score']:.12g}",
        "runtime_seconds": (
            "" if candidate["runtime_seconds"] is None else candidate["runtime_seconds"]
        ),
        "result_path": _first_value(raw, ["source_result_path", "result_path"]),
        "experiment_id": _first_value(raw, ["experiment_id_discovery", "experiment_id"]),
    }
    for metric in objective_metrics:
        row[f"metric_{metric}"] = candidate["objective_metrics"][metric]
    for key, value in raw.items():
        if key.startswith("param_"):
            row[key] = value
    return row


def _best_config_fieldnames(
    rows: list[dict[str, Any]],
    objective_metrics: list[str],
) -> list[str]:
    leading = [
        "log_id",
        "algorithm_name",
        "config_hash",
        "objective_score",
        *[f"metric_{metric}" for metric in objective_metrics],
        "runtime_seconds",
        "result_path",
        "experiment_id",
    ]
    param_fields = sorted({key for row in rows for key in row if key.startswith("param_")})
    return leading + [key for key in param_fields if key not in leading]


def _first_value(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def _v6_manifest_destination(
    config_path: str | Path,
    output_root: str | Path | None,
) -> Path | None:
    if output_root is None:
        return None
    experiment = load_experiment_config(config_path)
    if not experiment.manifest_path:
        raise ValueError(f"v6 config has no manifest_path: {config_path}")
    configured = Path(experiment.manifest_path)
    marker = ("experiments", "manifests", "v6")
    parts = configured.parts
    for index in range(len(parts) - len(marker) + 1):
        if parts[index : index + len(marker)] == marker:
            return Path(output_root).joinpath(*parts[index + len(marker) :])
    raise ValueError(f"v6 manifest_path must be under experiments/manifests/v6: {configured}")
