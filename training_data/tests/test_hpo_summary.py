import json
from pathlib import Path

import pytest

from process_discovery_cash.hpo.study_manifest import (
    STUDY_MANIFEST_COLUMNS,
    generate_hpo_study_rows,
    load_study_manifest_rows,
    write_study_manifest,
)
from process_discovery_cash.hpo.summary import build_study_summary, write_study_summary
from process_discovery_cash.hpo.trial_runner import StudyContext

pytestmark = pytest.mark.legacy_hpo

_CONFIG_TEMPLATE = """
experiment_id: hpo_summary_test
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
  - log_id: tiny2
    path: data/example/tiny_log.xes
seeds: [42]
output:
  results_dir: {results_dir}
  output_path_template: '{{results_dir}}/{{log_id}}/{{config_hash}}.json'
metrics:
  enabled: true
  profile: token
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
  n_trials: 4
  n_startup_trials: 2
  sampler_seed: 42
  storage_root: {storage_root}
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        _CONFIG_TEMPLATE.format(
            results_dir=(tmp_path / "results").as_posix(),
            storage_root=(tmp_path / "runs" / "hpo").as_posix(),
        ),
        encoding="utf-8",
    )
    return path


def test_generate_study_rows_one_per_log_algorithm(config_path: Path, tmp_path: Path) -> None:
    rows = generate_hpo_study_rows([config_path])

    assert len(rows) == 2
    assert [row["log_id"] for row in rows] == ["tiny", "tiny2"]
    assert all(set(row) == set(STUDY_MANIFEST_COLUMNS) for row in rows)
    assert rows[0]["study_index"] == "0"
    assert rows[1]["study_index"] == "1"
    assert rows[0]["study_name"] == "hpo_summary_test__tiny__heuristics_miner"
    assert rows[0]["journal_path"].endswith(
        "hpo_summary_test/hpo_summary_test__tiny__heuristics_miner.journal"
    )
    assert rows[0]["summary_path"].endswith(
        "hpo_summary_test/hpo_summaries/hpo_summary_test__tiny__heuristics_miner.json"
    )
    assert rows[0]["summary_path"].startswith(rows[0]["journal_path"].rsplit("/", 2)[0])

    manifest_path = write_study_manifest(rows, tmp_path / "studies.csv")
    assert load_study_manifest_rows(manifest_path) == rows


def test_generate_study_rows_requires_hpo_block(tmp_path: Path) -> None:
    config = tmp_path / "plain.yaml"
    config.write_text(
        """
experiment_id: plain
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: heuristics_miner
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hpo"):
        generate_hpo_study_rows([config])


def test_build_and_write_summary(config_path: Path, tmp_path: Path) -> None:
    import optuna
    from optuna.trial import TrialState

    ctx = StudyContext.from_experiment(config_path, "tiny", "heuristics_miner")
    study = optuna.create_study(direction="maximize")

    for number, value in enumerate([0.4, 0.9, 0.6]):
        trial = study.ask()
        trial.suggest_float("dependency_threshold", 0.0, 1.0)
        trial.set_user_attr("config_hash", f"hash{number}")
        trial.set_user_attr("run_status", "success")
        trial.set_user_attr("cached", number == 0)
        trial.set_user_attr("result_path", f"results/{number}.json")
        trial.set_user_attr("metric_fitness", value)
        study.tell(trial, value)
    failed = study.ask()
    failed.set_user_attr("run_status", "timeout")
    study.tell(failed, state=TrialState.FAIL)

    summary = build_study_summary(study, ctx)

    assert summary["experiment_id"] == "hpo_summary_test"
    assert summary["log_id"] == "tiny"
    assert summary["n_trials_target"] == 4
    assert summary["trials_by_state"] == {"COMPLETE": 3, "FAIL": 1}
    assert summary["trials_by_run_status"] == {"success": 3, "timeout": 1}
    assert summary["cached_trials"] == 1
    assert summary["best_trial"]["objective_value"] == pytest.approx(0.9)
    assert summary["best_trial"]["config_hash"] == "hash1"
    assert summary["best_trial"]["metrics"] == {"fitness": 0.9}
    assert "dependency_threshold" in summary["best_trial"]["params"]
    assert summary["study_wall_time_seconds"] is not None
    assert summary["sampler"]["n_startup_trials"] == 2

    written = write_study_summary(study, ctx)
    assert written.exists()
    assert written.parent.name == "hpo_summaries"
    assert Path(ctx.hpo.storage_root) in written.parents
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["best_trial"]["config_hash"] == "hash1"


def test_summary_with_empty_study(config_path: Path) -> None:
    import optuna

    ctx = StudyContext.from_experiment(config_path, "tiny", "heuristics_miner")
    study = optuna.create_study(direction="maximize")

    summary = build_study_summary(study, ctx)

    assert summary["best_trial"] is None
    assert summary["trials_by_state"] == {}
    assert summary["study_wall_time_seconds"] is None
