from pathlib import Path

import pytest

from process_discovery_cash.config.load import load_algorithm_config, load_experiment_config


def test_config_loading_works() -> None:
    experiment = load_experiment_config("configs/experiments/v6/baseline/alpha_classic/v1.yaml")
    algorithm = load_algorithm_config("configs/algorithms/alpha.yaml")

    assert experiment.experiment_id == "v6_baseline_alpha_classic_v1"
    assert experiment.logs[0].log_id == "bpi2012"
    assert algorithm.algorithm_id == "alpha_miner"
    assert algorithm.algorithm == "alpha_miner"
    assert algorithm.backend == "pm4py"
    assert algorithm.supported is True
    assert algorithm.default_params["discovery_timeout_seconds"] == 86400


@pytest.mark.parametrize(
    ("config_name", "timeout_field"),
    [
        ("alpha", "discovery_timeout_seconds"),
        ("genetic", "discovery_timeout_seconds"),
        ("heuristic", "discovery_timeout_seconds"),
        ("ilp", "discovery_timeout_seconds"),
        ("inductive", "discovery_timeout_seconds"),
        ("split", "timeout_seconds"),
    ],
)
def test_algorithm_runtime_defaults_are_uniform(
    config_name: str,
    timeout_field: str,
) -> None:
    algorithm = load_algorithm_config(f"configs/algorithms/{config_name}.yaml")

    assert algorithm.default_params[timeout_field] == 86400


def test_inductive_config_exposes_recursion_limit_runtime_control() -> None:
    algorithm = load_algorithm_config("configs/algorithms/inductive.yaml")

    assert algorithm.default_params["recursion_limit"] == 10000
    assert algorithm.default_params["discovery_timeout_seconds"] == 86400
    assert algorithm.runtime_params == ["discovery_timeout_seconds", "recursion_limit"]


def test_heuristics_config_exposes_recursion_limit_runtime_control() -> None:
    algorithm = load_algorithm_config("configs/algorithms/heuristic.yaml")

    assert algorithm.default_params["recursion_limit"] == 10000
    assert algorithm.default_params["discovery_timeout_seconds"] == 86400
    assert algorithm.runtime_params == ["discovery_timeout_seconds", "recursion_limit"]


def test_algorithm_reference_accepts_sampling_block(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
experiment_id: lhs_sampling
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: inductive_miner
    sampling:
      n_samples: 10
      seed: 42
""",
        encoding="utf-8",
    )

    experiment = load_experiment_config(config)

    sampling = experiment.algorithms[0].sampling
    assert sampling is not None
    assert sampling.n_samples == 10
    assert sampling.seed == 42
    assert sampling.strategy == "latin_hypercube"


def test_algorithm_reference_rejects_sampling_with_configs(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
experiment_id: lhs_sampling_conflict
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: inductive_miner
    configs:
      - variant: im
    sampling:
      n_samples: 10
      seed: 42
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sampling"):
        load_experiment_config(config)


def test_algorithm_reference_rejects_non_positive_n_samples(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
experiment_id: lhs_sampling_invalid_n
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: inductive_miner
    sampling:
      n_samples: 0
      seed: 42
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="n_samples"):
        load_experiment_config(config)


def test_experiment_config_accepts_hpo_block(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
experiment_id: hpo_experiment
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: inductive_miner
hpo:
  n_trials: 50
  n_startup_trials: 5
  sampler_seed: 7
  per_trial_walltime_seconds: 300
  objective:
    weights:
      fitness: 2.0
      precision: 1.0
""",
        encoding="utf-8",
    )

    experiment = load_experiment_config(config)

    hpo = experiment.hpo
    assert hpo is not None
    assert hpo.n_trials == 50
    assert hpo.n_startup_trials == 5
    assert hpo.sampler_seed == 7
    assert hpo.per_trial_walltime_seconds == 300
    assert hpo.objective.weights == {"fitness": 2.0, "precision": 1.0}
    assert hpo.objective.failed_trial_value == 0.0
    assert hpo.constant_liar is True


def test_hpo_block_rejects_unknown_objective_metric(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
experiment_id: hpo_bad_weights
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: inductive_miner
hpo:
  n_trials: 10
  objective:
    weights:
      not_a_metric: 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not_a_metric"):
        load_experiment_config(config)


def test_hpo_block_rejects_disabled_metrics(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
experiment_id: hpo_metrics_disabled
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: inductive_miner
metrics:
  enabled: false
hpo:
  n_trials: 10
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metrics.enabled"):
        load_experiment_config(config)


def test_hpo_block_rejects_zero_weights(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
experiment_id: hpo_zero_weights
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: inductive_miner
hpo:
  n_trials: 10
  objective:
    weights:
      fitness: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive"):
        load_experiment_config(config)


def test_algorithm_reference_requires_seed_in_sampling(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """
experiment_id: lhs_sampling_missing_seed
logs:
  - log_id: tiny
    path: data/example/tiny_log.xes
algorithms:
  - name: inductive_miner
    sampling:
      n_samples: 10
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seed"):
        load_experiment_config(config)
