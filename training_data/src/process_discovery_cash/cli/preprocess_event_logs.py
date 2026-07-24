from __future__ import annotations

import argparse

from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.data.preprocessing.artifacts import (
    DEFAULT_OUTPUT_ROOT,
    ArtifactSet,
    PreprocessingOptions,
    preprocess_dataset,
    preprocess_dataset_spec,
)
from process_discovery_cash.data.preprocessing.catalog import (
    load_dataset_catalog,
    match_dataset,
)
from process_discovery_cash.data.preprocessing.models import DatasetSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate algorithm-specific canonical event-log artifacts."
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--catalog", default="configs/datasets/processmining_org.yaml")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--drop-invalid-rows", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog = load_dataset_catalog(args.catalog)
    dataset_ids = set(args.datasets or [])
    transient_datasets: dict[tuple[str, str], DatasetSpec] = {}
    for config_path in args.configs or []:
        experiment = load_experiment_config(config_path)
        for log in experiment.logs:
            references = [log.source_path or log.path]
            if log.test_path and log.test_path != log.path:
                references.append(log.test_path)
            for path in references:
                if path is None:
                    continue
                dataset = match_dataset(
                    dataset_id=log.dataset_id,
                    log_id=log.log_id,
                    path=path,
                    catalog_path=args.catalog,
                )
                if dataset is None:
                    continue
                if dataset.dataset_id in catalog.datasets:
                    dataset_ids.add(dataset.dataset_id)
                else:
                    transient_datasets[(dataset.dataset_id, dataset.source_path)] = dataset
    if args.all:
        dataset_ids.update(catalog.datasets)
    if not dataset_ids and not transient_datasets:
        parser.error("Provide --dataset, --config, or --all.")

    options = PreprocessingOptions(invalid_row_policy="drop" if args.drop_invalid_rows else "fail")
    for dataset_id in sorted(dataset_ids):
        artifacts = preprocess_dataset(
            dataset_id,
            catalog_path=args.catalog,
            output_root=args.output_root,
            options=options,
            force=args.force,
        )
        _print_artifacts(dataset_id, artifacts)
    for dataset in sorted(
        transient_datasets.values(),
        key=lambda item: (item.dataset_id, item.source_path),
    ):
        artifacts = preprocess_dataset_spec(
            dataset,
            output_root=args.output_root,
            options=options,
            force=args.force,
        )
        _print_artifacts(dataset.dataset_id, artifacts)


def _print_artifacts(dataset_id: str, artifacts: ArtifactSet) -> None:
    print(
        f"{dataset_id}: fingerprint={artifacts.fingerprint} "
        f"pm4py={artifacts.pm4py_path} splitminer_v1={artifacts.splitminer_v1_path}"
    )


if __name__ == "__main__":
    main()
