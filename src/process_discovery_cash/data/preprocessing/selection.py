from __future__ import annotations

from pathlib import Path

from process_discovery_cash.data.preprocessing.artifacts import (
    DEFAULT_OUTPUT_ROOT,
    PreprocessingOptions,
    artifact_paths,
)
from process_discovery_cash.data.preprocessing.catalog import match_dataset
from process_discovery_cash.data.preprocessing.metadata import sha256_file
from process_discovery_cash.data.preprocessing.models import ArtifactSelection


def select_discovery_artifact(
    *,
    log_id: str | None,
    path: str | Path,
    algorithm_id: str,
    params: dict[str, object] | None = None,
    dataset_id: str | None = None,
    catalog_path: str | Path = "configs/datasets/processmining_org.yaml",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    options: PreprocessingOptions | None = None,
    require_existing: bool = True,
    preprocessing_fingerprint: str | None = None,
) -> ArtifactSelection:
    dataset = match_dataset(
        dataset_id=dataset_id,
        log_id=log_id,
        path=path,
        catalog_path=catalog_path,
    )
    if dataset is None:
        return ArtifactSelection(source_log_path=Path(path).as_posix())

    paths = artifact_paths(
        dataset,
        output_root=output_root,
        options=options,
        fingerprint_override=preprocessing_fingerprint,
    )
    if algorithm_id == "split_miner":
        artifact = paths.splitminer_v1_path
        kind = "splitminer_v1_xes"
    else:
        artifact = paths.pm4py_path
        kind = "pm4py_parquet"

    artifact_exists = bool(artifact and artifact.exists())
    artifact_hash = sha256_file(artifact) if artifact_exists else None
    return ArtifactSelection(
        dataset_id=dataset.dataset_id,
        source_log_path=Path(dataset.source_path).as_posix(),
        discovery_log_path=(
            artifact.as_posix() if artifact and (artifact_exists or not require_existing) else None
        ),
        artifact_kind=kind if artifact else None,
        artifact_sha256=artifact_hash,
        preprocessing_fingerprint=paths.fingerprint,
        preprocessing_metadata_path=paths.metadata_path.as_posix(),
    )
