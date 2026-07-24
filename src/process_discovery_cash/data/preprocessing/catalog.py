from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from process_discovery_cash.data.preprocessing.models import DatasetCatalog, DatasetSpec

DEFAULT_DATASET_CATALOG = Path("configs/datasets/processmining_org.yaml")


@lru_cache(maxsize=8)
def load_dataset_catalog(
    path: str | Path = DEFAULT_DATASET_CATALOG,
) -> DatasetCatalog:
    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    raw_datasets = payload.get("datasets") or {}
    datasets = {
        dataset_id: DatasetSpec(dataset_id=dataset_id, **spec)
        for dataset_id, spec in raw_datasets.items()
    }
    return DatasetCatalog(datasets=datasets)


def get_dataset(
    dataset_id: str,
    catalog_path: str | Path = DEFAULT_DATASET_CATALOG,
) -> DatasetSpec:
    catalog = load_dataset_catalog(catalog_path)
    try:
        return catalog.datasets[dataset_id]
    except KeyError as exc:
        known = ", ".join(sorted(catalog.datasets))
        raise KeyError(f"Unknown dataset_id {dataset_id!r}. Known datasets: {known}") from exc


def match_dataset(
    *,
    dataset_id: str | None = None,
    log_id: str | None = None,
    path: str | Path | None = None,
    catalog_path: str | Path = DEFAULT_DATASET_CATALOG,
) -> DatasetSpec | None:
    catalog = load_dataset_catalog(catalog_path)
    if dataset_id:
        if dataset_id in catalog.datasets:
            return catalog.datasets[dataset_id]
        synthetic = synthetic_dataset_spec(log_id=log_id or dataset_id, path=path)
        if synthetic is not None:
            return synthetic
        return get_dataset(dataset_id, catalog_path)
    if log_id and log_id in catalog.datasets:
        return catalog.datasets[log_id]
    if path:
        candidate = Path(path).as_posix()
        candidate_name = Path(path).name
        for dataset in catalog.datasets.values():
            if candidate == dataset.source_path or candidate_name == Path(dataset.source_path).name:
                return dataset
    return synthetic_dataset_spec(log_id=log_id, path=path)


def synthetic_dataset_spec(
    *,
    log_id: str | None,
    path: str | Path | None,
) -> DatasetSpec | None:
    if path is None or not _is_synthetic_reference(log_id=log_id, path=path):
        return None
    source_path = Path(path).as_posix()
    resolved_log_id = log_id or Path(path).name.removesuffix(".xes.gz").removesuffix(".xes")
    return DatasetSpec(
        dataset_id=resolved_log_id,
        display_name=resolved_log_id,
        source_path=source_path,
        schema={"lifecycle_semantics": "absent"},
        notes=["Transient dataset specification for a generated event log."],
    )


def _is_synthetic_reference(*, log_id: str | None, path: str | Path) -> bool:
    if log_id and log_id.startswith(("syn_", "aug_")):
        return True
    candidate = Path(path).as_posix()
    return any(
        candidate.startswith(prefix) or f"/{prefix}" in candidate
        for prefix in ("data/synthetic/", "data/augmented/")
    )
