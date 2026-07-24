import json
from pathlib import Path

import pytest

from process_discovery_cash.experiments.manifest import MANIFEST_COLUMNS, generate_manifest_rows
from process_discovery_cash.hpo import trial_runner
from process_discovery_cash.hpo.trial_runner import (
    StudyContext,
    build_trial_row,
    run_trial,
    trial_config_hash,
)

pytestmark = pytest.mark.legacy_hpo

_BASE_EXPERIMENT = """
experiment_id: {experiment_id}
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
seeds: [42]
output:
  results_dir: {results_dir}
  output_path_template: '{{results_dir}}/{{log_id}}/{{config_hash}}.json'
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
{search_space}
{extra}
"""

_HPO_BLOCK = """
hpo:
  n_trials: 8
  n_startup_trials: 2
  sampler_seed: 42
"""


def _write_config(
    tmp_path: Path,
    *,
    experiment_id: str,
    search_space: str,
    extra: str = "",
) -> Path:
    config_path = tmp_path / f"{experiment_id}.yaml"
    config_path.write_text(
        _BASE_EXPERIMENT.format(
            experiment_id=experiment_id,
            results_dir=(tmp_path / "results").as_posix(),
            search_space=search_space,
            extra=extra,
        ),
        encoding="utf-8",
    )
    return config_path


def _hpo_context(tmp_path: Path) -> StudyContext:
    config_path = _write_config(
        tmp_path,
        experiment_id="hpo_parity",
        search_space=(
            "      dependency_threshold:\n        min: 0.0\n        max: 1.0\n        type: float\n"
        ),
        extra=_HPO_BLOCK,
    )
    return StudyContext.from_experiment(config_path, "tiny", "heuristics_miner")


def test_trial_config_hash_matches_manifest_generation(tmp_path: Path) -> None:
    grid_config = _write_config(
        tmp_path,
        experiment_id="grid_parity",
        search_space=("      dependency_threshold:\n        values: [0.5]\n"),
    )
    manifest_rows = generate_manifest_rows(grid_config)
    assert len(manifest_rows) == 1
    manifest_row = manifest_rows[0]
    params = json.loads(manifest_row["params_json"])

    ctx = _hpo_context(tmp_path)

    assert trial_config_hash(ctx, params) == manifest_row["config_hash"]


def test_build_trial_row_matches_manifest_row(tmp_path: Path) -> None:
    grid_config = _write_config(
        tmp_path,
        experiment_id="hpo_parity",
        search_space=("      dependency_threshold:\n        values: [0.5]\n"),
    )
    manifest_row = generate_manifest_rows(grid_config)[0]
    params = json.loads(manifest_row["params_json"])

    ctx = _hpo_context(tmp_path)
    row = build_trial_row(ctx, params, trial_config_hash(ctx, params))

    assert set(row) == set(MANIFEST_COLUMNS)
    assert row == manifest_row


def test_context_requires_hpo_block(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        experiment_id="no_hpo",
        search_space=("      dependency_threshold:\n        values: [0.5]\n"),
    )

    with pytest.raises(ValueError, match="hpo"):
        StudyContext.from_experiment(config_path, "tiny", "heuristics_miner")


def test_context_rejects_row_index_template(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_template.yaml"
    config_path.write_text(
        _BASE_EXPERIMENT.format(
            experiment_id="bad_template",
            results_dir=(tmp_path / "results").as_posix(),
            search_space=(
                "      dependency_threshold:\n"
                "        min: 0.0\n"
                "        max: 1.0\n"
                "        type: float\n"
            ),
            extra=_HPO_BLOCK,
        ).replace(
            "'{results_dir}/{log_id}/{config_hash}.json'",
            "'{results_dir}/{row_index}_{config_hash}.json'",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row_index"):
        StudyContext.from_experiment(config_path, "tiny", "heuristics_miner")


def test_context_rejects_sampling_block(tmp_path: Path) -> None:
    config_path = tmp_path / "hpo_with_sampling.yaml"
    config_path.write_text(
        """
experiment_id: hpo_with_sampling
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
seeds: [42]
output:
  output_path_template: '{results_dir}/{log_id}/{config_hash}.json'
algorithms:
  - name: heuristics_miner
    search_space_override:
      dependency_threshold:
        min: 0.0
        max: 1.0
        type: float
    sampling:
      n_samples: 4
      seed: 1
hpo:
  n_trials: 8
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sampling"):
        StudyContext.from_experiment(config_path, "tiny", "heuristics_miner")


def _success_payload(row: dict[str, str], metrics: dict[str, float]) -> dict:
    return {
        "status": "success",
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "log_path": row["log_path"],
        "test_log_path": row["test_log_path"],
        "seed": int(row["seed"]),
        "algorithm_name": row["algorithm_id"],
        "backend": row["backend"],
        "discovered_model_type": "petri_net",
        "hyperparameters": json.loads(row["params_json"]),
        "metrics": metrics,
        "metric_statuses": {
            name: {"status": "success", "value": None, "error": None} for name in metrics
        },
        "metadata": {"config_hash": row["config_hash"]},
    }


def test_run_trial_uses_cached_success_result(tmp_path: Path, monkeypatch) -> None:
    ctx = _hpo_context(tmp_path)
    params = ctx.finalize_trial_params(dict(ctx.default_params, dependency_threshold=0.5))
    config_hash = trial_config_hash(ctx, params)
    row = build_trial_row(ctx, params, config_hash)
    result_path = Path(row["output_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {"fitness": 1.0, "precision": 0.5, "generalization": 0.5, "simplicity": 1.0}
    result_path.write_text(json.dumps(_success_payload(row, metrics)), encoding="utf-8")

    def _explode(*args, **kwargs):
        raise AssertionError("cached trial must not execute")

    monkeypatch.setattr(trial_runner, "run_row_in_subprocess", _explode)
    monkeypatch.setattr(trial_runner, "run_manifest_row", _explode)

    outcome = run_trial(ctx, params)

    assert outcome.cached is True
    assert outcome.config_hash == config_hash
    assert outcome.objective.value == pytest.approx(0.75)
    assert outcome.objective.run_status == "success"


def test_run_trial_uses_cached_failed_result(tmp_path: Path, monkeypatch) -> None:
    ctx = _hpo_context(tmp_path)
    params = ctx.finalize_trial_params(dict(ctx.default_params, dependency_threshold=0.25))
    config_hash = trial_config_hash(ctx, params)
    row = build_trial_row(ctx, params, config_hash)
    result_path = Path(row["output_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"status": "timeout"}), encoding="utf-8")

    def _explode(*args, **kwargs):
        raise AssertionError("cached trial must not execute")

    monkeypatch.setattr(trial_runner, "run_row_in_subprocess", _explode)

    outcome = run_trial(ctx, params)

    assert outcome.cached is True
    assert outcome.objective.value == 0.0
    assert outcome.objective.run_status == "timeout"


def test_run_trial_executes_and_reads_result(tmp_path: Path, monkeypatch) -> None:
    ctx = _hpo_context(tmp_path)
    params = ctx.finalize_trial_params(dict(ctx.default_params, dependency_threshold=0.75))
    metrics = {"fitness": 0.8, "precision": 0.8, "generalization": 0.8, "simplicity": 0.8}
    calls = {"count": 0}

    def _fake_run(row, command_args=None, force=False):
        calls["count"] += 1
        result_path = Path(row["output_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(_success_payload(row, metrics)), encoding="utf-8")
        return result_path

    monkeypatch.setattr(trial_runner, "run_manifest_row", _fake_run)

    outcome = run_trial(ctx, params, isolate=False)

    assert calls["count"] == 1
    assert outcome.cached is False
    assert outcome.objective.value == pytest.approx(0.8)


def test_run_trial_child_death_without_result_is_crashed(tmp_path: Path, monkeypatch) -> None:
    from process_discovery_cash.experiments.run_isolation import RowExecutionOutcome

    ctx = _hpo_context(tmp_path)
    params = ctx.finalize_trial_params(dict(ctx.default_params, dependency_threshold=0.9))

    def _fake_subprocess(row, **kwargs):
        return RowExecutionOutcome(
            exit_code=-9,
            signal_name="SIGKILL",
            killed_by_parent=False,
            oom_suspected=True,
            child_peak_rss_bytes=None,
            duration_seconds=1.0,
        )

    monkeypatch.setattr(trial_runner, "run_row_in_subprocess", _fake_subprocess)

    outcome = run_trial(ctx, params)

    assert outcome.cached is False
    assert outcome.objective.run_status == "crashed"
    assert outcome.objective.value == 0.0
    assert outcome.killed_at_study_deadline is False


def test_run_trial_study_deadline_kill_is_flagged(tmp_path: Path, monkeypatch) -> None:
    from process_discovery_cash.experiments.run_isolation import RowExecutionOutcome

    ctx = _hpo_context(tmp_path)
    params = ctx.finalize_trial_params(dict(ctx.default_params, dependency_threshold=0.1))

    def _fake_subprocess(row, **kwargs):
        return RowExecutionOutcome(
            exit_code=None,
            signal_name="SIGTERM",
            killed_by_parent=True,
            oom_suspected=False,
            child_peak_rss_bytes=None,
            duration_seconds=5.0,
        )

    monkeypatch.setattr(trial_runner, "run_row_in_subprocess", _fake_subprocess)

    outcome = run_trial(ctx, params, deadline_monotonic=100.0, deadline_is_study_deadline=True)

    assert outcome.killed_at_study_deadline is True
    assert outcome.objective.run_status == "study_walltime"

    outcome = run_trial(ctx, params, deadline_monotonic=100.0, deadline_is_study_deadline=False)

    assert outcome.killed_at_study_deadline is False
    assert outcome.objective.run_status == "trial_walltime_exceeded"
