from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest
from fake_gedi import FakeGediBackend

import process_discovery_cash.utils.paths as paths
from process_discovery_cash.data.augmentation import (
    AugmentationSpec,
    augment_parent_log,
)
from process_discovery_cash.data.features import extract_features_from_xes
from process_discovery_cash.experiments.manifest import generate_manifest
from process_discovery_cash.experiments.metric_manifest import (
    generate_metric_manifest_from_source_manifest,
    run_metric_manifest_row,
)
from process_discovery_cash.experiments.runner import run_manifest_row
from process_discovery_cash.generation.feature_space import BAND_IN_DISTRIBUTION
from process_discovery_cash.generation.pipeline import run_generation
from process_discovery_cash.generation.targets import TargetSpec

pytestmark = pytest.mark.integration


def _log() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    rows = []
    for case in range(20):
        activities = ["a", "b", "c"] if case % 2 else ["a", "c", "b"]
        for position, activity in enumerate(activities):
            rows.append(
                {
                    "case:concept:name": f"c{case:02d}",
                    "concept:name": activity,
                    "time:timestamp": base + pd.Timedelta(minutes=position),
                }
            )
    return pd.DataFrame(rows)


def test_tiny_augmentation_gedi_xes_discovery_metric_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    monkeypatch.setenv("DATA_ROOT", data_root.as_posix())
    monkeypatch.setenv("RESULTS_ROOT", results_root.as_posix())
    monkeypatch.setenv("LOG_ROOT", (tmp_path / "logs").as_posix())
    monkeypatch.setattr(paths, "_DOTENV_LOADED", True)
    monkeypatch.setattr(paths, "_DOTENV_VALUES", {})

    augmented = augment_parent_log(
        _log(),
        "tiny",
        [AugmentationSpec("subsample", {"fraction": 0.5})],
        output_dir=data_root / "augmented/logs",
        base_seed=1001,
    )[0]
    assert augmented.status == "accepted"
    features = extract_features_from_xes(str(augmented.output_path))
    assert len(features) == 48

    anchor = pd.DataFrame(
        [
            {
                "log_id": "anchor",
                "num_traces": 1000,
                "avg_trace_length": 10,
                "num_activities": 20,
                "variant_ratio": 0.5,
                "dfg_density": 0.2,
                "repetition_prevalence": 0.4,
            }
        ]
    )
    target = TargetSpec(
        target_id="t0000",
        band=BAND_IN_DISTRIBUTION,
        values={
            "num_traces": 60,
            "avg_trace_length": 5,
            "num_activities": 8,
            "variant_ratio": 0.5,
            "dfg_density": 0.3,
            "repetition_prevalence": 0.5,
        },
        concurrency="low",
        noise_level=0,
        nearest_real_distance=1,
    )
    records = run_generation(
        [target],
        anchor,
        FakeGediBackend(),
        output_root=data_root / "synthetic/gedi",
        base_seed=2024,
        workdir=tmp_path / "gedi-work",
        max_attempts=1,
    )
    assert len(records) == 1

    recorded_augmented = Path(augmented.output_path)
    relative_augmented = (
        recorded_augmented.relative_to(data_root)
        if recorded_augmented.is_absolute()
        else recorded_augmented.relative_to("data")
    )
    config = tmp_path / "integration.yaml"
    config.write_text(
        f"""
experiment_id: v6_tiny_integration
logs:
  - log_id: aug_tiny
    path: data/{relative_augmented.as_posix()}
seeds: [42]
output:
  results_dir: results/integration
  log_dir: logs/slurm/integration
  output_path_template: "{{results_dir}}/{{log_id}}/{{config_hash}}.json"
metrics:
  enabled: false
  profile: token
  names: [fitness, precision, generalization, simplicity]
  export_model: true
algorithms:
  - name: alpha_miner_classic
    algorithm_id: alpha_miner_classic
    backend: pm4py
    model_type: petri_net
    artifact_algorithm_id: alpha_miner
    runtime_params: [discovery_timeout_seconds]
    default_params: {{variant: classic}}
    configs: [{{}}]
""",
        encoding="utf-8",
    )
    source_manifest = generate_manifest(config, tmp_path / "source.csv")
    with source_manifest.open(encoding="utf-8", newline="") as handle:
        discovery_row = next(csv.DictReader(handle))
    assert discovery_row["log_path"].endswith(".xes.gz")
    assert discovery_row["discovery_log_path"] == ""
    assert discovery_row["artifact_sha256"] == ""
    result_path = run_manifest_row(discovery_row, command_args=["integration"], force=True)
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "success"

    metric_manifest, stats = generate_metric_manifest_from_source_manifest(
        source_manifest,
        metric_profile="token",
        output_path=tmp_path / "metrics.csv",
        output_root="results/metrics/integration",
    )
    assert stats.total == 1
    with metric_manifest.open(encoding="utf-8", newline="") as handle:
        metric_row = next(csv.DictReader(handle))
    metric_path = run_metric_manifest_row(metric_row, command_args=["integration"], force=True)
    metric_payload = json.loads(metric_path.read_text(encoding="utf-8"))
    assert metric_payload["status"] in {"success", "success_missing"}
