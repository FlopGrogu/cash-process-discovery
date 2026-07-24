import csv
import json
from pathlib import Path

import pytest

from process_discovery_cash.experiments.manifest import MANIFEST_COLUMNS
from process_discovery_cash.experiments.metric_manifest import (
    generate_metric_manifest_rows_from_source_manifest,
)
from process_discovery_cash.hpo.export_manifest import export_hpo_discovery_manifest
from process_discovery_cash.hpo.study_manifest import default_study_manifest_path
from process_discovery_cash.hpo.trial_runner import (
    StudyContext,
    build_trial_row,
    trial_config_hash,
)

pytestmark = pytest.mark.legacy_hpo

_CONFIG_TEMPLATE = """
experiment_id: hpo_export_test
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
seeds: [42]
output:
  results_dir: {results_dir}
  output_path_template: '{{results_dir}}/{{log_id}}/{{config_hash}}.json'
manifest_path: {manifest_path}
metrics:
  enabled: true
  profile: token
  export_model: false
algorithms:
  - name: heuristics_miner
    algorithm_id: heuristics_miner
    backend: pm4py
    model_type: petri_net
    runtime_params:
      - discovery_timeout_seconds
    default_params:
      variant: classic
      discovery_timeout_seconds: 240
    search_space_override:
      dependency_threshold:
        min: 0.0
        max: 1.0
        type: float
hpo:
  n_trials: 8
  n_startup_trials: 2
  sampler_seed: 42
"""


@pytest.fixture
def export_setup(tmp_path: Path) -> tuple[Path, StudyContext, Path]:
    results_dir = tmp_path / "results" / "cluster" / "v6" / "model" / "hpo" / "heuristics" / "v1"
    manifest_path = (
        tmp_path / "experiments" / "manifests" / "v6" / "model" / "hpo" / "heuristics" / "v1.csv"
    )
    config_path = tmp_path / "v1.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            results_dir=results_dir.as_posix(),
            manifest_path=manifest_path.as_posix(),
        ),
        encoding="utf-8",
    )
    ctx = StudyContext.from_experiment(config_path, "tiny", "heuristics_miner")
    return config_path, ctx, manifest_path


def _write_result(ctx: StudyContext, dependency_threshold: float, status: str) -> dict[str, str]:
    params = ctx.finalize_trial_params(
        dict(ctx.default_params, dependency_threshold=dependency_threshold)
    )
    config_hash = trial_config_hash(ctx, params)
    row = build_trial_row(ctx, params, config_hash)
    payload = {
        "status": status,
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "log_path": row["log_path"],
        "test_log_path": row["test_log_path"],
        "seed": int(row["seed"]),
        "algorithm_name": row["algorithm_id"],
        "backend": row["backend"],
        "discovered_model_type": "petri_net",
        "hyperparameters": params,
        "metrics": {},
        "metric_statuses": {},
        "metadata": {"config_hash": config_hash},
    }
    result_path = Path(row["output_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return row


def test_export_rebuilds_manifest_rows_from_results(export_setup) -> None:
    config_path, ctx, manifest_path = export_setup
    expected_rows = [
        _write_result(ctx, 0.25, "success"),
        _write_result(ctx, 0.5, "success"),
        _write_result(ctx, 0.75, "timeout"),
    ]

    exported_path, stats = export_hpo_discovery_manifest(config_path)

    assert exported_path == manifest_path
    assert stats.exported == 3
    assert stats.skipped_hash_mismatch == 0
    with exported_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == MANIFEST_COLUMNS
    from process_discovery_cash.experiments.runner import load_manifest_rows

    rows = load_manifest_rows(exported_path)
    assert sorted(row["config_hash"] for row in rows) == sorted(
        row["config_hash"] for row in expected_rows
    )
    assert rows == sorted(
        rows, key=lambda row: (row["log_id"], row["algorithm_id"], row["config_hash"])
    )


def test_export_skips_foreign_and_stale_results(export_setup) -> None:
    config_path, ctx, _manifest_path = export_setup
    row = _write_result(ctx, 0.5, "success")
    results_dir = Path(row["output_path"]).parent

    (results_dir / "other_algo.json").write_text(
        json.dumps({"algorithm_name": "alpha_miner", "hyperparameters": {}}),
        encoding="utf-8",
    )
    stale = json.loads(Path(row["output_path"]).read_text(encoding="utf-8"))
    stale["metadata"]["config_hash"] = "deadbeef00000000"
    (results_dir / "deadbeef00000000.json").write_text(json.dumps(stale), encoding="utf-8")
    (results_dir / "corrupt.json").write_text("{not json", encoding="utf-8")

    _path, stats = export_hpo_discovery_manifest(config_path)

    assert stats.exported == 1
    assert stats.skipped_other_algorithm == 1
    assert stats.skipped_hash_mismatch == 1
    assert stats.skipped_unreadable == 1


def test_export_feeds_metric_manifest_generation(export_setup, tmp_path: Path) -> None:
    config_path, ctx, _manifest_path = export_setup
    _write_result(ctx, 0.25, "success")
    _write_result(ctx, 0.75, "timeout")

    exported_path, _stats = export_hpo_discovery_manifest(config_path)

    generation = generate_metric_manifest_rows_from_source_manifest(
        exported_path,
        metric_profile="token",
        output_root=tmp_path / "metric_results",
    )
    metric_rows = generation.rows

    # One metric row per exported discovery row, same order.
    assert generation.stats.total == 2
    with exported_path.open("r", encoding="utf-8", newline="") as handle:
        source_hashes = [row["config_hash"] for row in csv.DictReader(handle)]
    assert [row["source_config_hash"] for row in metric_rows] == source_hashes


def test_export_requires_manifest_path(tmp_path: Path) -> None:
    config_path = tmp_path / "no_manifest_path.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            results_dir=(tmp_path / "results").as_posix(),
            manifest_path="null",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest_path"):
        export_hpo_discovery_manifest(config_path)


def test_default_study_manifest_path_mirrors_config_tree() -> None:
    assert default_study_manifest_path(
        "configs/experiments/v6/hpo/heuristic_plusplus/v1.yaml"
    ) == Path("experiments/manifests/v6/hpo/heuristic_plusplus/v1/studies.csv")
    with pytest.raises(ValueError, match="only for canonical configs"):
        default_study_manifest_path("/abs/prefix/configs/experiments/hpo/local_smoke.yaml")
