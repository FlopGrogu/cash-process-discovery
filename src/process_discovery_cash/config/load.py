from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from process_discovery_cash.config.schema import AlgorithmConfig, ExperimentConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return loaded


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    payload = load_yaml(path)
    logs_from = payload.pop("logs_from", None)
    if logs_from is not None:
        if "logs" in payload:
            raise ValueError("Experiment config cannot define both logs and logs_from")
        logs_path = Path(logs_from)
        if not logs_path.is_absolute():
            logs_path = path.parent / logs_path
        source_payload = load_yaml(logs_path)
        if "logs" not in source_payload:
            raise ValueError(f"Referenced config does not define logs: {logs_path}")
        payload["logs"] = source_payload["logs"]
    experiment = ExperimentConfig(**payload)
    from process_discovery_cash.data.preprocessing.catalog import match_dataset

    for log in experiment.logs:
        if not Path(experiment.dataset_catalog_path).exists():
            if log.dataset_id is not None:
                raise FileNotFoundError(
                    f"Dataset catalog not found: {experiment.dataset_catalog_path}"
                )
            continue
        dataset = match_dataset(
            dataset_id=log.dataset_id,
            log_id=log.log_id,
            path=log.path,
            catalog_path=experiment.dataset_catalog_path,
        )
        if dataset is None:
            continue
        log.dataset_id = dataset.dataset_id
        log.source_path = dataset.source_path
        if log.path is None:
            log.path = dataset.source_path
        if log.train_path is None:
            log.train_path = log.path
        if log.test_path is None:
            log.test_path = log.path
    return experiment


def load_algorithm_config(path: str | Path) -> AlgorithmConfig:
    return AlgorithmConfig(**load_yaml(path))


def resolve_algorithm_config_path(
    algorithm_name: str,
    algorithm_config_dir: str | Path = "configs/algorithms",
) -> Path:
    return Path(algorithm_config_dir) / f"{algorithm_name}.yaml"


def merge_dicts(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged
