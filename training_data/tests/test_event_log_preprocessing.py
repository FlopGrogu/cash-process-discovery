from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from process_discovery_cash.cli import preprocess_event_logs as preprocess_cli
from process_discovery_cash.data.preprocessing.artifacts import (
    ArtifactSet,
    PreprocessingOptions,
    _write_minimal_xes,
    artifact_paths,
    canonicalize_dataframe,
    discovery_projection,
    preprocess_dataset_spec,
    preprocessing_fingerprint,
    sanitize_xml_text,
)
from process_discovery_cash.data.preprocessing.catalog import (
    load_dataset_catalog,
    match_dataset,
    synthetic_dataset_spec,
)
from process_discovery_cash.data.preprocessing.lifecycle import analyze_lifecycle
from process_discovery_cash.data.preprocessing.metadata import (
    inspect_dataset_package,
    read_processmining_metadata,
    read_xes_header_metadata,
    sha256_file,
)
from process_discovery_cash.data.preprocessing.models import DatasetSpec
from process_discovery_cash.data.preprocessing.selection import select_discovery_artifact


def _dataset(**overrides) -> DatasetSpec:
    payload = {
        "dataset_id": "test",
        "display_name": "Test",
        "source_path": "test.xes",
        "schema": {
            "lifecycle": "lifecycle:transition",
            "lifecycle_semantics": "standard",
        },
    }
    payload.update(overrides)
    return DatasetSpec(**payload)


def test_canonical_sorting_and_complete_projection_is_stable() -> None:
    dataframe = pd.DataFrame(
        {
            "case:concept:name": ["2", "1", "1", "1"],
            "concept:name": ["C", "B", "A", "A"],
            "time:timestamp": [
                "2020-01-02T00:00:00Z",
                "2020-01-01T00:00:01Z",
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:00:00Z",
            ],
            "lifecycle:transition": ["complete", "complete", "start", "complete"],
        }
    )
    canonical, validation = canonicalize_dataframe(dataframe, _dataset())
    projected = discovery_projection(canonical, _dataset())

    assert validation["invalid_rows"] == 0
    assert projected["concept:name"].tolist() == ["A", "B", "C"]
    assert projected["@@source_event_index"].tolist() == [3, 1, 0]


def test_invalid_rows_fail_or_drop() -> None:
    dataframe = pd.DataFrame(
        {
            "case:concept:name": ["1", None],
            "concept:name": ["A", "B"],
            "time:timestamp": ["2020-01-01T00:00:00Z", "bad"],
            "lifecycle:transition": ["complete", "complete"],
        }
    )
    with pytest.raises(ValueError, match="invalid discovery events"):
        canonicalize_dataframe(dataframe, _dataset())
    canonical, validation = canonicalize_dataframe(
        dataframe,
        _dataset(),
        options=PreprocessingOptions(invalid_row_policy="drop"),
    )
    assert len(canonical) == 1
    assert validation["invalid_rows"] == 1
    assert validation["timestamp_parse_failures"] == 1


def test_xml_control_characters_are_sanitized() -> None:
    assert sanitize_xml_text("A\x00B\x1fC") == "A\ufffdB\ufffdC"


def test_lifecycle_pairing_green_and_extended_yellow() -> None:
    dataframe = pd.DataFrame(
        {
            "case:concept:name": ["1", "1"],
            "concept:name": ["A", "A"],
            "time:timestamp": pd.to_datetime(
                ["2020-01-01T00:00:00Z", "2020-01-01T00:01:00Z"], utc=True
            ),
            "lifecycle:transition": ["start", "complete"],
        }
    )
    standard = analyze_lifecycle(
        dataframe,
        semantics="standard",
        case_column="case:concept:name",
        activity_column="concept:name",
        timestamp_column="time:timestamp",
        lifecycle_column="lifecycle:transition",
    )
    extended = analyze_lifecycle(
        dataframe,
        semantics="extended_standard",
        case_column="case:concept:name",
        activity_column="concept:name",
        timestamp_column="time:timestamp",
        lifecycle_column="lifecycle:transition",
    )
    assert standard.interval_quality == "green"
    assert standard.paired == 1
    assert extended.interval_quality == "yellow"


def test_processmining_metadata_and_duplicate_log_alias_detection(tmp_path: Path) -> None:
    xes = tmp_path / "source.xes"
    alias = tmp_path / "DATA1.xml"
    xes.write_text(
        '<?xml version="1.0"?><log><classifier name="Event Name" '
        'keys="concept:name"/><trace/></log>',
        encoding="utf-8",
    )
    alias.write_bytes(xes.read_bytes())
    metadata = tmp_path / "DATA.xml"
    metadata.write_text(
        "<metadata><name>Example</name><number_of_traces>0</number_of_traces>"
        "<number_of_events>0</number_of_events></metadata>",
        encoding="utf-8",
    )
    parsed = read_processmining_metadata(metadata)
    assert parsed.expected_events == 0
    with pytest.raises(ValueError, match="Expected processmining.org metadata"):
        read_processmining_metadata(alias)
    dataset = _dataset(
        source_path=xes.as_posix(),
        companions=[
            {"path": alias.as_posix(), "role": "duplicate_log_alias"},
            {"path": metadata.as_posix(), "role": "processmining_metadata"},
        ],
    )
    inspected = inspect_dataset_package(dataset)
    assert inspected.companions[0].detected_role == "duplicate_log_alias"
    assert inspected.processmining_metadata is not None


def test_xes_header_extracts_first_trace_and_event_attribute_keys(tmp_path: Path) -> None:
    xes = tmp_path / "source.xes"
    xes.write_text(
        '<log><global scope="event"><string key="concept:name" '
        'value="UNKNOWN"/></global><trace><string key="concept:name" '
        'value="case-1"/><event><string key="concept:name" value="A"/>'
        '<date key="time:timestamp" value="2020-01-01T00:00:00Z"/>'
        "</event></trace></log>",
        encoding="utf-8",
    )
    header = read_xes_header_metadata(xes)
    assert header.trace_attribute_keys == ["concept:name"]
    assert header.event_attribute_keys == ["concept:name", "time:timestamp"]
    assert header.globals[0].attributes[0].key == "concept:name"


def test_companions_only_change_fingerprint_when_they_affect_resolution(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.xes"
    source.write_text("<log/>", encoding="utf-8")
    metadata = tmp_path / "DATA.xml"
    metadata.write_text("<metadata/>", encoding="utf-8")
    readme = tmp_path / "README.txt"
    readme.write_text("one", encoding="utf-8")
    dataset = _dataset(
        source_path=source.as_posix(),
        companions=[
            {"path": metadata.as_posix(), "role": "processmining_metadata"},
            {"path": readme.as_posix(), "role": "documentation"},
        ],
    )
    first = preprocessing_fingerprint(dataset)
    readme.write_text("two", encoding="utf-8")
    assert preprocessing_fingerprint(dataset) == first
    metadata.write_text("<metadata><name>changed</name></metadata>", encoding="utf-8")
    assert preprocessing_fingerprint(dataset) == first
    resolving_dataset = dataset.model_copy(update={"metadata_affects_resolution": True})
    resolving_first = preprocessing_fingerprint(resolving_dataset)
    metadata.write_text("<metadata><name>changed again</name></metadata>", encoding="utf-8")
    assert preprocessing_fingerprint(resolving_dataset) != resolving_first


def test_fingerprint_is_stable_when_package_is_relocated(tmp_path: Path) -> None:
    first_source = tmp_path / "one.xes"
    second_source = tmp_path / "nested" / "two.xes"
    second_source.parent.mkdir()
    first_source.write_text("<log/>", encoding="utf-8")
    second_source.write_bytes(first_source.read_bytes())
    first = _dataset(source_path=first_source.as_posix())
    second = _dataset(source_path=second_source.as_posix())
    assert preprocessing_fingerprint(first) == preprocessing_fingerprint(second)


def test_minimal_xes_hash_is_stable(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "case:concept:name": ["1"],
            "concept:name": ["A"],
            "time:timestamp": pd.to_datetime(["2020-01-01T00:00:00Z"], utc=True),
            "@@source_event_index": [0],
        }
    )
    first = tmp_path / "first.xes"
    second = tmp_path / "second.xes"
    _write_minimal_xes(dataframe, first, force_complete=True)
    _write_minimal_xes(dataframe, second, force_complete=True)
    assert sha256_file(first) == sha256_file(second)


def test_bpic15_date_fields_are_not_configured_as_start_timestamps() -> None:
    load_dataset_catalog.cache_clear()
    catalog = load_dataset_catalog()
    for dataset_id in [f"bpic15_{index}" for index in range(1, 6)]:
        schema = catalog.datasets[dataset_id].event_schema
        assert schema.start_timestamp is None
        assert "dateFinished" not in schema.optional_attributes
        assert "dateStop" not in schema.optional_attributes
    assert catalog.datasets["bpi2018"].event_schema.activity == "concept:name"


def test_processmining_catalog_retains_lifecycle_semantics_without_v2_policy() -> None:
    load_dataset_catalog.cache_clear()
    catalog = load_dataset_catalog()
    assert len(catalog.datasets) == 21
    assert catalog.datasets["bpi2012"].event_schema.lifecycle_semantics == "standard"
    assert catalog.datasets["bpi2017"].event_schema.lifecycle_semantics == "extended_standard"
    assert not hasattr(catalog.datasets["bpi2012"], "splitminer_v2_eligibility")


def test_artifact_paths_never_include_split_miner_v2(tmp_path: Path) -> None:
    source = tmp_path / "source.xes"
    source.write_text("<log/>", encoding="utf-8")
    paths = artifact_paths(_dataset(source_path=source.as_posix()), output_root=tmp_path)

    assert paths.pm4py_path.name == "pm4py.parquet"
    assert paths.splitminer_v1_path.name == "splitminer-v1.xes"
    assert not hasattr(paths, "splitminer_v2_path")


def test_synthetic_dataset_spec_is_transient_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "generated.xes"
    source.write_text("<log/>", encoding="utf-8")

    first = synthetic_dataset_spec(log_id="syn_generated", path=source)
    second = synthetic_dataset_spec(log_id="syn_generated", path=source)

    assert first is not None
    assert second is not None
    assert first == second
    assert first.dataset_id == "syn_generated"
    assert first.display_name == "syn_generated"
    assert first.source_path == source.as_posix()
    assert first.event_schema.lifecycle_semantics == "absent"


def test_synthetic_path_resolves_without_catalog_entry(tmp_path: Path) -> None:
    source = tmp_path / "data" / "synthetic" / "logs" / "generated.xes"
    source.parent.mkdir(parents=True)
    source.write_text("<log/>", encoding="utf-8")
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text("datasets: {}\n", encoding="utf-8")
    load_dataset_catalog.cache_clear()

    resolved = match_dataset(
        log_id="generated",
        path=source,
        catalog_path=catalog_path,
    )
    ordinary = match_dataset(
        log_id="ordinary",
        path=tmp_path / "ordinary.xes",
        catalog_path=catalog_path,
    )

    assert resolved is not None
    assert resolved.dataset_id == "generated"
    assert ordinary is None


def test_synthetic_source_content_changes_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.xes"
    source.write_text("<log/>", encoding="utf-8")
    dataset = synthetic_dataset_spec(log_id="syn_test", path=source)
    assert dataset is not None

    first = preprocessing_fingerprint(dataset)
    source.write_text("<log><trace/></log>", encoding="utf-8")

    assert preprocessing_fingerprint(dataset) != first


def test_preprocess_dataset_spec_writes_standard_artifact_set(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic.xes"
    source.write_text(
        '<log><trace><string key="concept:name" value="1"/></trace></log>',
        encoding="utf-8",
    )
    dataset = synthetic_dataset_spec(log_id="syn_test", path=source)
    assert dataset is not None
    dataframe = pd.DataFrame(
        {
            "case:concept:name": ["1"],
            "concept:name": ["A"],
            "time:timestamp": ["2024-01-01T00:00:00Z"],
        }
    )

    class Loaded:
        log = dataframe

        @staticmethod
        def metadata() -> dict[str, str]:
            return {"backend": "test"}

    monkeypatch.setattr(
        "process_discovery_cash.data.preprocessing.artifacts.load_event_log_with_info",
        lambda *_args, **_kwargs: Loaded(),
    )

    artifacts = preprocess_dataset_spec(dataset, output_root=tmp_path / "processed")
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert artifacts.pm4py_path.exists()
    assert artifacts.splitminer_v1_path.exists()
    assert metadata["dataset"]["dataset_id"] == "syn_test"
    assert metadata["resolved_schema"]["lifecycle_semantics"] == "absent"
    assert set(metadata["artifacts"]) == {"pm4py_parquet", "splitminer_v1_xes"}


def test_preprocess_config_discovers_synthetic_train_and_test_logs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.xes"
    test = tmp_path / "test.xes"
    train.write_text("<log/>", encoding="utf-8")
    test.write_text("<log/>", encoding="utf-8")
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text("datasets: {}\n", encoding="utf-8")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "synthetic_train_test",
                "dataset_catalog_path": catalog_path.as_posix(),
                "logs": [
                    {
                        "log_id": "syn_train_test",
                        "path": train.as_posix(),
                        "test_path": test.as_posix(),
                    }
                ],
                "algorithms": ["alpha_miner"],
            }
        ),
        encoding="utf-8",
    )
    load_dataset_catalog.cache_clear()
    captured: list[DatasetSpec] = []

    def fake_preprocess(dataset: DatasetSpec, **_kwargs) -> ArtifactSet:
        captured.append(dataset)
        output_dir = tmp_path / "processed" / dataset.dataset_id / str(len(captured))
        return ArtifactSet(
            dataset_id=dataset.dataset_id,
            fingerprint=str(len(captured)),
            output_dir=output_dir,
            pm4py_path=output_dir / "pm4py.parquet",
            splitminer_v1_path=output_dir / "splitminer-v1.xes",
            metadata_path=output_dir / "metadata.json",
        )

    monkeypatch.setattr(preprocess_cli, "preprocess_dataset_spec", fake_preprocess)
    monkeypatch.setattr(
        preprocess_cli,
        "preprocess_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    preprocess_cli.main(
        [
            "--config",
            config_path.as_posix(),
            "--catalog",
            catalog_path.as_posix(),
            "--output-root",
            (tmp_path / "processed").as_posix(),
        ]
    )

    assert [dataset.source_path for dataset in captured] == [
        test.as_posix(),
        train.as_posix(),
    ]


def test_split_miner_artifact_selection_is_v1_only(tmp_path: Path) -> None:
    source = tmp_path / "source.xes"
    source.write_text("<log/>", encoding="utf-8")
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "test": {
                        "display_name": "Test",
                        "source_path": source.as_posix(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    load_dataset_catalog.cache_clear()

    selection = select_discovery_artifact(
        log_id="test",
        dataset_id="test",
        path=source,
        algorithm_id="split_miner",
        catalog_path=catalog_path,
        output_root=tmp_path / "processed",
        require_existing=False,
    )

    assert selection.artifact_kind == "splitminer_v1_xes"
    assert selection.discovery_log_path is not None
    assert selection.discovery_log_path.endswith("/splitminer-v1.xes")


def test_manifest_selection_does_not_pass_sidecars_to_discovery(tmp_path: Path) -> None:
    source = tmp_path / "source.xes"
    source.write_text("<log/>", encoding="utf-8")
    metadata = tmp_path / "DATA.xml"
    metadata.write_text("<metadata/>", encoding="utf-8")
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "test": {
                        "display_name": "Test",
                        "source_path": source.as_posix(),
                        "companions": [
                            {
                                "path": metadata.as_posix(),
                                "role": "processmining_metadata",
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    load_dataset_catalog.cache_clear()
    selection = select_discovery_artifact(
        log_id="test",
        dataset_id="test",
        path=source,
        algorithm_id="inductive_miner",
        catalog_path=catalog_path,
        output_root=tmp_path / "processed",
        require_existing=False,
    )
    assert selection.source_log_path == source.as_posix()
    assert selection.discovery_log_path is not None
    assert selection.discovery_log_path.endswith("/pm4py.parquet")
    assert metadata.as_posix() not in json.dumps(selection.model_dump())

    dataset = load_dataset_catalog(catalog_path).datasets["test"]
    paths = artifact_paths(dataset, output_root=tmp_path / "processed")
    paths.output_dir.mkdir(parents=True)
    paths.pm4py_path.write_bytes(b"parquet placeholder")
    existing = select_discovery_artifact(
        log_id="test",
        dataset_id="test",
        path=source,
        algorithm_id="inductive_miner",
        catalog_path=catalog_path,
        output_root=tmp_path / "processed",
        require_existing=True,
    )
    assert existing.discovery_log_path == paths.pm4py_path.as_posix()
    assert existing.artifact_sha256 == sha256_file(paths.pm4py_path)


def test_selection_reports_artifact_hash_when_artifact_exists_even_without_require_existing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.xes"
    source.write_text("<log/>", encoding="utf-8")
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "test": {
                        "display_name": "Test",
                        "source_path": source.as_posix(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    load_dataset_catalog.cache_clear()
    dataset = load_dataset_catalog(catalog_path).datasets["test"]
    paths = artifact_paths(dataset, output_root=tmp_path / "processed")
    paths.output_dir.mkdir(parents=True)
    paths.pm4py_path.write_bytes(b"parquet placeholder")

    selection = select_discovery_artifact(
        log_id="test",
        dataset_id="test",
        path=source,
        algorithm_id="inductive_miner",
        catalog_path=catalog_path,
        output_root=tmp_path / "processed",
        require_existing=False,
    )

    assert selection.discovery_log_path == paths.pm4py_path.as_posix()
    assert selection.artifact_sha256 == sha256_file(paths.pm4py_path)
