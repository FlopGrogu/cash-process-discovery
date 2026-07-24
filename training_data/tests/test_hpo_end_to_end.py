import json
from pathlib import Path

from process_discovery_cash.cli.hpo_summary import main as hpo_summary_main
from process_discovery_cash.cli.run_hpo_study import main as run_hpo_study_main

_CONFIG_TEMPLATE = """
experiment_id: hpo_e2e
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
      dependency_threshold:
        min: 0.0
        max: 1.0
        type: float
hpo:
  n_trials: 3
  n_startup_trials: 3
  sampler_seed: 42
  per_trial_walltime_seconds: 240
  storage_root: {storage_root}
"""


def test_hpo_study_end_to_end(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    config_path = tmp_path / "hpo_e2e.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            results_dir=results_dir.as_posix(),
            storage_root=(tmp_path / "runs" / "hpo").as_posix(),
        ),
        encoding="utf-8",
    )
    argv = [
        "--config",
        str(config_path),
        "--log-id",
        "tiny",
        "--algorithm",
        "heuristics_miner",
        "--no-isolate-runs",
        "--worker-walltime-seconds",
        "600",
        "--safety-margin-seconds",
        "0",
    ]

    run_hpo_study_main(argv)

    result_files = sorted((results_dir / "tiny").glob("*.json"))
    assert len(result_files) >= 1
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    assert all(payload["status"] == "success" for payload in payloads)
    assert all("fitness" in payload["metrics"] for payload in payloads)

    summary_path = (
        tmp_path
        / "runs"
        / "hpo"
        / "hpo_e2e"
        / "hpo_summaries"
        / "hpo_e2e__tiny__heuristics_miner.json"
    )
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["trials_by_state"].get("COMPLETE") == 3
    best = summary["best_trial"]
    assert best is not None
    assert 0.0 < best["objective_value"] <= 1.0
    assert "dependency_threshold" in best["params"]
    best_result = results_dir / "tiny" / f"{best['config_hash']}.json"
    assert best_result.exists()

    # Second invocation resumes the journal: the trial budget is already met,
    # so no new result files appear and the summary regenerates cleanly.
    run_hpo_study_main(argv)
    assert sorted((results_dir / "tiny").glob("*.json")) == result_files

    summary_path.unlink()
    hpo_summary_main(
        [
            "--config",
            str(config_path),
            "--log-id",
            "tiny",
            "--algorithm",
            "heuristics_miner",
        ]
    )
    assert summary_path.exists()
