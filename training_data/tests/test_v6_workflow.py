from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from process_discovery_cash.cli.generate_v6_manifests import (
    build_parser as build_v6_manifest_parser,
)
from process_discovery_cash.cli.generate_v6_manifests import (
    main as generate_v6_manifests_main,
)
from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.experiments.dynamic_worker import (
    load_dynamic_manifest_entries,
    run_dynamic_worker,
)
from process_discovery_cash.experiments.manifest import generate_manifest_rows
from process_discovery_cash.experiments.v6 import (
    discover_v6_baseline_configs,
    discover_v6_default_run_survey_configs,
    generate_v6_default_run_survey_manifests,
    generate_v6_manifests,
    prepare_v6_augmented_explore_configs,
    prepare_v6_synthetic_explore_configs,
    select_best_v6_configs,
)

V6_BASELINE_LOG_COUNT = 21
V6_EXPLORE_MERGED_LOG_COUNT = 98


def test_v6_default_run_survey_runs_every_algorithm_variant_at_its_defaults() -> None:
    expected = {
        "alpha_classic": (
            "alpha_miner_classic",
            {"variant": "classic", "discovery_timeout_seconds": 86400},
        ),
        "alpha_plus": (
            "alpha_miner_plus",
            {"variant": "plus", "discovery_timeout_seconds": 86400},
        ),
        "genetic": (
            "genetic_miner",
            {
                "population_size": 500,
                "elitism_rate": 0.01,
                "crossover_rate": 1.0,
                "mutation_rate": 0.01,
                "generations": 100,
                "elitism_min_sample": 5,
                "log_csv": None,
                "discovery_timeout_seconds": 86400,
            },
        ),
        "heuristic_classic": (
            "heuristics_miner",
            {
                "variant": "classic",
                "dependency_threshold": 0.5,
                "and_threshold": 0.65,
                "loop_two_threshold": 0.5,
                "min_act_count": 1,
                "min_dfg_occurrences": 1,
                "dfg_pre_cleaning_noise_thresh": 0.0,
                "recursion_limit": 10000,
                "discovery_timeout_seconds": 86400,
            },
        ),
        "heuristic_plusplus": (
            "heuristics_miner_plusplus",
            {
                "variant": "plusplus",
                "dependency_threshold": 0.5,
                "and_threshold": 0.65,
                "loop_two_threshold": 0.5,
                "min_act_count": 1,
                "min_dfg_occurrences": 1,
                "dfg_pre_cleaning_noise_thresh": 0.0,
                "recursion_limit": 10000,
                "discovery_timeout_seconds": 86400,
            },
        ),
        "ilp": (
            "ilp_miner",
            {"alpha": 1.0, "discovery_timeout_seconds": 86400},
        ),
        "inductive_im": (
            "inductive_miner_im",
            {
                "variant": "im",
                "disable_fallthroughs": False,
                "multi_processing": False,
                "recursion_limit": 10000,
                "discovery_timeout_seconds": 86400,
            },
        ),
        "inductive_imd": (
            "inductive_miner_imd",
            {
                "variant": "imd",
                "disable_fallthroughs": False,
                "multi_processing": False,
                "recursion_limit": 10000,
                "discovery_timeout_seconds": 86400,
            },
        ),
        "inductive_imf": (
            "inductive_miner_imf",
            {
                "variant": "imf",
                "noise_threshold": 0.0,
                "disable_fallthroughs": False,
                "multi_processing": False,
                "recursion_limit": 10000,
                "discovery_timeout_seconds": 86400,
            },
        ),
        "split": (
            "split_miner",
            {
                "jar_path": "data/external/split-miner-1.7.1-all.jar",
                "jar_sha256": (
                    "472c006623d99a6e440aa93a58e29b867cc331cec2b12b3d7fb61fb2a5de8328"
                ),
                "jar_env_var": "SPLIT_MINER_JAR",
                "java_bin": None,
                "java_options": [
                    "-Xms64m",
                    "-Xmx3g",
                    "-XX:MaxMetaspaceSize=256m",
                    "-Xss512k",
                    "-Djava.awt.headless=true",
                ],
                "timeout_seconds": 86400,
                "epsilon": 0.1,
                "eta": 0.2,
                "parallelismFirst": False,
                "removeLoopActivityMarkers": False,
                "replaceIORs": False,
                "diagram": False,
            },
        ),
    }

    config_paths = discover_v6_default_run_survey_configs()

    assert len(config_paths) == len(expected)
    assert {path.parent.name for path in config_paths} == set(expected)
    assert {path.name for path in config_paths} == {"v1.yaml"}
    for path in config_paths:
        algorithm_id, default_params = expected[path.parent.name]
        rows = generate_manifest_rows(path)

        assert len(rows) == V6_BASELINE_LOG_COUNT
        assert len({row["log_id"] for row in rows}) == V6_BASELINE_LOG_COUNT
        assert {row["algorithm_id"] for row in rows} == {algorithm_id}
        assert all(json.loads(row["params_json"]) == default_params for row in rows)
        assert all(
            row["output_path"].startswith(
                f"results/cluster/v6/model/default_run_survey/{path.parent.name}/"
            )
            for row in rows
        )
        assert all(json.loads(row["metrics_json"])["enabled"] is False for row in rows)
        assert all(json.loads(row["metrics_json"])["export_model"] is True for row in rows)


def test_v6_default_run_survey_generates_discovery_manifests_only(tmp_path: Path) -> None:
    written = generate_v6_default_run_survey_manifests(output_root=tmp_path)
    assert len(written) == 10
    assert sum(len(_read_manifest(path)) for path in written.values()) == 210
    assert not list(tmp_path.rglob("*metrics.csv"))


def test_v6_default_run_survey_cli_has_no_legacy_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generate_v6_manifests_main(
        ["--default-run-survey", "--output-root", str(tmp_path)]
    )
    assert len(list(tmp_path.glob("model/default_run_survey/*/v1.csv"))) == 10
    assert "Wrote v6 alpha_classic manifest" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        legacy_flag = "--" + "bench" + "mark"
        build_v6_manifest_parser().parse_args([legacy_flag])


def test_v6_baseline_yaml_configs_are_discoverable() -> None:
    config_paths = discover_v6_baseline_configs()

    expected = {
        "alpha_classic": (
            "v1.yaml",
            V6_BASELINE_LOG_COUNT,
            ("discovery_timeout_seconds", 86400),
        ),
        "alpha_plus": (
            "v1.yaml",
            V6_BASELINE_LOG_COUNT,
            ("discovery_timeout_seconds", 86400),
        ),
        "genetic": ("v2.yaml", V6_BASELINE_LOG_COUNT, ("discovery_timeout_seconds", 240)),
        "heuristic_classic": (
            "v2.yaml",
            V6_BASELINE_LOG_COUNT,
            ("discovery_timeout_seconds", 240),
        ),
        "heuristic_plusplus": (
            "v2.yaml",
            V6_BASELINE_LOG_COUNT,
            ("discovery_timeout_seconds", 240),
        ),
        "ilp": ("v2.yaml", V6_BASELINE_LOG_COUNT, ("discovery_timeout_seconds", 240)),
        "inductive_im": (
            "v1.1.yaml",
            V6_BASELINE_LOG_COUNT,
            ("discovery_timeout_seconds", 86400),
        ),
        "inductive_imd": (
            "v1.1.yaml",
            V6_BASELINE_LOG_COUNT,
            ("discovery_timeout_seconds", 86400),
        ),
        "inductive_imf": (
            "v2.yaml",
            V6_BASELINE_LOG_COUNT,
            ("discovery_timeout_seconds", 240),
        ),
        "split": ("v2.yaml", V6_BASELINE_LOG_COUNT, ("timeout_seconds", 240)),
    }
    assert len(config_paths) == 10
    assert {path.parent.name for path in config_paths} == set(expected)

    for path in config_paths:
        version, row_count, timeout = expected[path.parent.name]
        assert path.name == version
        assert "/hpo/" not in path.read_text(encoding="utf-8")
        rows = generate_manifest_rows(path)
        assert len(rows) == row_count
        assert len({row["log_id"] for row in rows}) == row_count
        timeout_key, timeout_seconds = timeout
        assert all(
            json.loads(row["params_json"])[timeout_key] == timeout_seconds for row in rows
        )


def test_v6_explore_yaml_configs_cover_all_v6_algorithms() -> None:
    config_paths = sorted(Path("configs/experiments/v6/explore").glob("*/*.yaml"))

    assert len(config_paths) == 10
    assert {path.name for path in config_paths} == {"v1.yaml"}
    assert {path.parent.name for path in config_paths} == {
        "alpha_classic",
        "alpha_plus",
        "genetic",
        "heuristic_classic",
        "heuristic_plusplus",
        "inductive_im",
        "inductive_imd",
        "ilp",
        "inductive_imf",
        "split",
    }


def test_v6_inline_algorithm_config_loads_without_algorithm_yaml() -> None:
    experiment = load_experiment_config("configs/experiments/v6/baseline/alpha_plus/v1.yaml")
    algorithm = experiment.algorithms[0]

    assert algorithm.name == "alpha_miner_plus"
    assert algorithm.algorithm_id == "alpha_miner_plus"
    assert algorithm.backend == "pm4py"
    assert algorithm.model_type == "petri_net"
    assert algorithm.runtime_params == ["discovery_timeout_seconds"]


def test_v6_inline_algorithm_requires_backend_and_runtime_params(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text(
        """
experiment_id: broken_v6_inline
logs:
  - {log_id: bpi2012, dataset_id: bpi2012}
algorithms:
  - name: alpha_miner_plus
    algorithm_id: alpha_miner_plus
    model_type: petri_net
    configs:
      - variant: plus
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="backend"):
        load_experiment_config(config_path)


def test_v6_manifest_generation_uses_alias_ids_and_explicit_configs_only() -> None:
    alpha_rows = generate_manifest_rows("configs/experiments/v6/baseline/alpha_plus/v1.yaml")
    im_rows = generate_manifest_rows("configs/experiments/v6/baseline/inductive_im/v1.1.yaml")
    imd_rows = generate_manifest_rows("configs/experiments/v6/baseline/inductive_imd/v1.1.yaml")
    inductive_rows = generate_manifest_rows("configs/experiments/v6/baseline/inductive_imf/v2.yaml")

    assert len(alpha_rows) == V6_BASELINE_LOG_COUNT
    assert {row["algorithm_id"] for row in alpha_rows} == {"alpha_miner_plus"}
    assert all(
        row["output_path"].startswith("results/cluster/v6/model/baseline/alpha_plus/")
        for row in alpha_rows
    )
    assert json.loads(alpha_rows[0]["params_json"]) == {
        "variant": "plus",
        "discovery_timeout_seconds": 86400,
    }

    assert len(im_rows) == V6_BASELINE_LOG_COUNT
    assert {row["algorithm_id"] for row in im_rows} == {"inductive_miner_im"}
    assert json.loads(im_rows[0]["params_json"]) == {
        "variant": "im",
        "disable_fallthroughs": False,
        "multi_processing": False,
        "recursion_limit": 10000,
        "discovery_timeout_seconds": 86400,
    }

    assert len(imd_rows) == V6_BASELINE_LOG_COUNT
    assert {row["algorithm_id"] for row in imd_rows} == {"inductive_miner_imd"}
    assert json.loads(imd_rows[0]["params_json"]) == {
        "variant": "imd",
        "disable_fallthroughs": False,
        "multi_processing": False,
        "recursion_limit": 10000,
        "discovery_timeout_seconds": 86400,
    }

    assert len(inductive_rows) == V6_BASELINE_LOG_COUNT
    assert {row["algorithm_id"] for row in inductive_rows} == {"inductive_miner_imf"}
    assert json.loads(inductive_rows[0]["params_json"]) == {
        "variant": "imf",
        "noise_threshold": 0.0,
        "disable_fallthroughs": False,
        "multi_processing": False,
        "recursion_limit": 10000,
        "discovery_timeout_seconds": 240,
    }


def test_v6_manifests_are_xes_first_for_every_algorithm() -> None:
    alpha_rows = generate_manifest_rows("configs/experiments/v6/baseline/alpha_plus/v1.yaml")
    split_rows = generate_manifest_rows("configs/experiments/v6/baseline/split/v2.yaml")

    for row in [*alpha_rows, *split_rows]:
        assert row["log_path"].lower().endswith((".xes", ".xes.gz"))
        assert row["test_log_path"].lower().endswith((".xes", ".xes.gz"))
        assert row["discovery_log_path"] == ""
        assert row["test_discovery_log_path"] == ""
        assert row["artifact_kind"] == ""
        assert row["artifact_sha256"] == ""
        assert row["preprocessing_fingerprint"] == ""
        assert row["preprocessing_metadata_path"] == ""


def test_v6_convenience_wrapper_generates_one_manifest_per_alias(tmp_path: Path) -> None:
    written = generate_v6_manifests(output_root=tmp_path)

    assert len(written) == 10
    assert set(written) == {
        "alpha_classic",
        "alpha_plus",
        "genetic",
        "heuristic_classic",
        "heuristic_plusplus",
        "ilp",
        "inductive_im",
        "inductive_imd",
        "inductive_imf",
        "split",
    }
    total_rows = 0
    for path in written.values():
        rows = _read_manifest(path)
        assert len(rows) == V6_BASELINE_LOG_COUNT
        total_rows += len(rows)
    assert total_rows == V6_BASELINE_LOG_COUNT * 10


def test_v6_best_config_selection_uses_balanced_composite_and_tie_breakers(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "joined.csv"
    rows = [
        _metric_row("cfg_b", fitness=0.9, precision=0.7, runtime=20),
        _metric_row("cfg_a", fitness=0.8, precision=0.8, runtime=10),
        _metric_row("cfg_c", fitness=0.8, precision=0.8, runtime=5),
        _metric_row("other", log_id="log_2", algorithm_name="alpha_miner_plus", fitness=1.0),
    ]
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_path = select_best_v6_configs([input_path], tmp_path / "best.csv")
    selected = _read_manifest(output_path)

    assert [row["config_hash"] for row in selected] == ["cfg_c", "other"]
    assert selected[0]["log_id"] == "log_1"
    assert selected[0]["objective_score"] == "0.8"


def test_v6_hash_addressed_result_is_skipped_by_dynamic_worker(tmp_path: Path) -> None:
    written = generate_v6_manifests(output_root=tmp_path / "manifests")
    row = _read_manifest(written["genetic"])[0]
    output_path = tmp_path / "results" / "success.json"
    row["output_path"] = output_path.as_posix()
    _write_success(output_path, row)

    isolated_manifest = tmp_path / "genetic.csv"
    with isolated_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    entries = load_dynamic_manifest_entries(isolated_manifest)
    stats = run_dynamic_worker(
        entries,
        state_dir=tmp_path / "state",
        worker_walltime_seconds=100,
        safety_margin_seconds=10,
        run_row=lambda *_args, **_kwargs: pytest.fail("existing v6 success should be skipped"),
    )

    assert stats.skipped_success == 1
    assert stats.claimed == 0


def test_v6_explore_manifest_generation_uses_curated_lhs_dimensions_and_runtime_caps() -> None:
    alpha_rows = generate_manifest_rows("configs/experiments/v6/explore/alpha_classic/v1.yaml")
    alpha_plus_rows = generate_manifest_rows("configs/experiments/v6/explore/alpha_plus/v1.yaml")
    heuristic_rows = generate_manifest_rows(
        "configs/experiments/v6/explore/heuristic_classic/v1.yaml"
    )
    heuristic_plus_rows = generate_manifest_rows(
        "configs/experiments/v6/explore/heuristic_plusplus/v1.yaml"
    )
    ilp_rows = generate_manifest_rows("configs/experiments/v6/explore/ilp/v1.yaml")
    genetic_rows = generate_manifest_rows("configs/experiments/v6/explore/genetic/v1.yaml")
    im_rows = generate_manifest_rows("configs/experiments/v6/explore/inductive_im/v1.yaml")
    imd_rows = generate_manifest_rows("configs/experiments/v6/explore/inductive_imd/v1.yaml")
    inductive_rows = generate_manifest_rows("configs/experiments/v6/explore/inductive_imf/v1.yaml")
    split_rows = generate_manifest_rows("configs/experiments/v6/explore/split/v1.yaml")

    assert len({row["log_id"] for row in alpha_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(alpha_rows) == V6_EXPLORE_MERGED_LOG_COUNT
    assert {row["algorithm_id"] for row in alpha_rows} == {"alpha_miner_classic"}
    assert json.loads(alpha_rows[0]["params_json"]) == {
        "variant": "classic",
        "discovery_timeout_seconds": 240,
    }

    assert len({row["log_id"] for row in alpha_plus_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(alpha_plus_rows) == V6_EXPLORE_MERGED_LOG_COUNT
    assert {row["algorithm_id"] for row in alpha_plus_rows} == {"alpha_miner_plus"}
    assert json.loads(alpha_plus_rows[0]["params_json"]) == {
        "variant": "plus",
        "discovery_timeout_seconds": 240,
    }

    assert len({row["log_id"] for row in heuristic_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(heuristic_rows) == V6_EXPLORE_MERGED_LOG_COUNT * 64
    heuristic_params = [json.loads(row["params_json"]) for row in heuristic_rows]
    assert {params["variant"] for params in heuristic_params} == {"classic"}
    assert {params["discovery_timeout_seconds"] for params in heuristic_params} == {240}
    assert all(0.0 <= params["dependency_threshold"] <= 1.0 for params in heuristic_params)
    assert all(0.0 <= params["and_threshold"] <= 1.0 for params in heuristic_params)
    assert all(0.0 <= params["loop_two_threshold"] <= 1.0 for params in heuristic_params)
    assert all(0.0 <= params["dfg_pre_cleaning_noise_thresh"] <= 0.6 for params in heuristic_params)
    assert all(1 <= params["min_act_count"] <= 100 for params in heuristic_params)
    assert all(1 <= params["min_dfg_occurrences"] <= 100 for params in heuristic_params)

    assert len({row["log_id"] for row in heuristic_plus_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(heuristic_plus_rows) == V6_EXPLORE_MERGED_LOG_COUNT * 64
    heuristic_plus_params = [json.loads(row["params_json"]) for row in heuristic_plus_rows]
    assert {params["variant"] for params in heuristic_plus_params} == {"plusplus"}
    assert {params["discovery_timeout_seconds"] for params in heuristic_plus_params} == {240}
    assert all(0.0 <= params["dependency_threshold"] <= 1.0 for params in heuristic_plus_params)
    assert all(0.0 <= params["and_threshold"] <= 1.0 for params in heuristic_plus_params)
    assert all(0.0 <= params["loop_two_threshold"] <= 1.0 for params in heuristic_plus_params)
    assert all(
        0.0 <= params["dfg_pre_cleaning_noise_thresh"] <= 0.6 for params in heuristic_plus_params
    )
    assert all(1 <= params["min_act_count"] <= 100 for params in heuristic_plus_params)
    assert all(1 <= params["min_dfg_occurrences"] <= 100 for params in heuristic_plus_params)

    assert len({row["log_id"] for row in ilp_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(ilp_rows) == V6_EXPLORE_MERGED_LOG_COUNT * 8
    ilp_params = [json.loads(row["params_json"]) for row in ilp_rows]
    assert {params["discovery_timeout_seconds"] for params in ilp_params} == {240}
    assert all(0.0 <= params["alpha"] <= 1.0 for params in ilp_params)

    assert len({row["log_id"] for row in genetic_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(genetic_rows) == V6_EXPLORE_MERGED_LOG_COUNT * 16
    genetic_params = [json.loads(row["params_json"]) for row in genetic_rows]
    assert {params["discovery_timeout_seconds"] for params in genetic_params} == {240}
    assert {params["log_csv"] for params in genetic_params} == {None}
    assert all(10 <= params["population_size"] <= 750 for params in genetic_params)
    assert all(5 <= params["generations"] <= 200 for params in genetic_params)
    assert all(0.001 <= params["mutation_rate"] <= 0.2 for params in genetic_params)
    assert all(0.4 <= params["crossover_rate"] <= 1.0 for params in genetic_params)
    assert all(0.0 <= params["elitism_rate"] <= 0.3 for params in genetic_params)
    assert all(1 <= params["elitism_min_sample"] <= 50 for params in genetic_params)

    assert len({row["log_id"] for row in im_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(im_rows) == V6_EXPLORE_MERGED_LOG_COUNT * 2
    assert {row["algorithm_id"] for row in im_rows} == {"inductive_miner_im"}
    im_params = [json.loads(row["params_json"]) for row in im_rows]
    assert {params["variant"] for params in im_params} == {"im"}
    assert {params["discovery_timeout_seconds"] for params in im_params} == {240}
    assert {params["disable_fallthroughs"] for params in im_params} == {False, True}
    assert {params["multi_processing"] for params in im_params} == {False}
    assert {params["recursion_limit"] for params in im_params} == {10000}

    assert len({row["log_id"] for row in imd_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(imd_rows) == V6_EXPLORE_MERGED_LOG_COUNT * 2
    assert {row["algorithm_id"] for row in imd_rows} == {"inductive_miner_imd"}
    imd_params = [json.loads(row["params_json"]) for row in imd_rows]
    assert {params["variant"] for params in imd_params} == {"imd"}
    assert {params["discovery_timeout_seconds"] for params in imd_params} == {240}
    assert {params["disable_fallthroughs"] for params in imd_params} == {False, True}
    assert {params["multi_processing"] for params in imd_params} == {False}
    assert {params["recursion_limit"] for params in imd_params} == {10000}

    assert len({row["log_id"] for row in inductive_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(inductive_rows) == V6_EXPLORE_MERGED_LOG_COUNT * 16
    inductive_params = [json.loads(row["params_json"]) for row in inductive_rows]
    assert {params["variant"] for params in inductive_params} == {"imf"}
    assert {params["disable_fallthroughs"] for params in inductive_params} == {False, True}
    assert {params["discovery_timeout_seconds"] for params in inductive_params} == {240}
    assert all(0.0 <= params["noise_threshold"] <= 1.0 for params in inductive_params)

    assert len({row["log_id"] for row in split_rows}) == V6_EXPLORE_MERGED_LOG_COUNT
    assert len(split_rows) == V6_EXPLORE_MERGED_LOG_COUNT * 16
    split_params = [json.loads(row["params_json"]) for row in split_rows]
    assert {params["timeout_seconds"] for params in split_params} == {240}
    assert {params["parallelismFirst"] for params in split_params} == {False, True}
    assert {params["removeLoopActivityMarkers"] for params in split_params} == {False, True}
    assert {params["replaceIORs"] for params in split_params} == {False, True}
    assert all(0.0 <= params["epsilon"] <= 1.0 for params in split_params)
    assert all(0.0 <= params["eta"] <= 1.0 for params in split_params)


def test_v6_augmented_explore_configs_reuse_real_explore_parameter_sets(
    tmp_path: Path,
) -> None:
    augmentation_manifest = tmp_path / "augmentation_manifest.csv"
    logs_dir = tmp_path / "data" / "augmented" / "logs"
    logs_dir.mkdir(parents=True)
    child_a_path = logs_dir / "aug_parent_a__sub050__s1.xes.gz"
    child_b_path = logs_dir / "aug_parent_b__noise005__s2.xes.gz"
    child_a_path.write_bytes(b"child-a")
    child_b_path.write_bytes(b"child-b")
    rows = [
        {
            "child_log_id": "aug_parent_a__sub050__s1",
            "parent_log_id": "parent_a",
            "parent_path": "data/raw/parent_a.xes.gz",
            "augmentation": "subsample",
            "parameters": "{}",
            "seed": "1",
            "stress": "False",
            "status": "accepted",
            "rejection_reason": "",
            "output_path": child_a_path.as_posix(),
            "n_traces": "10",
            "n_events": "30",
            "n_activities": "3",
            "n_variants": "2",
            "created_at_utc": "2026-07-05T00:00:00+00:00",
        },
        {
            "child_log_id": "aug_parent_b__noise005__s2",
            "parent_log_id": "parent_b",
            "parent_path": "data/raw/parent_b.xes.gz",
            "augmentation": "noise",
            "parameters": "{}",
            "seed": "2",
            "stress": "False",
            "status": "accepted",
            "rejection_reason": "",
            "output_path": child_b_path.as_posix(),
            "n_traces": "10",
            "n_events": "30",
            "n_activities": "3",
            "n_variants": "2",
            "created_at_utc": "2026-07-05T00:00:00+00:00",
        },
    ]
    with augmentation_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    written = prepare_v6_augmented_explore_configs(
        augmentation_manifest=augmentation_manifest,
        output_root=tmp_path / "configs",
    )

    assert set(written) == {
        "alpha_classic",
        "alpha_plus",
        "genetic",
        "heuristic_classic",
        "heuristic_plusplus",
        "inductive_im",
        "inductive_imd",
        "ilp",
        "inductive_imf",
        "split",
    }
    for algorithm_slug, augmented_config_path in written.items():
        real_config_path = Path("configs/experiments/v6/explore") / algorithm_slug / "v1.yaml"
        real_rows = generate_manifest_rows(real_config_path)
        augmented_rows = generate_manifest_rows(augmented_config_path)

        real_params = _params_for_first_log(real_rows)
        augmented_params = _params_for_first_log(augmented_rows)
        assert augmented_params == real_params
        assert len(augmented_rows) == len(rows) * len(real_params)
        assert all(row["log_id"].startswith("aug_") for row in augmented_rows)
        assert all(
            row["output_path"].startswith(
                f"results/cluster/v6/model/explore_augmented/{algorithm_slug}/"
            )
            for row in augmented_rows
        )


def test_v6_synthetic_explore_configs_reuse_real_explore_parameter_sets(
    tmp_path: Path,
) -> None:
    synthetic_manifest = tmp_path / "gedi_manifest.csv"
    logs_dir = tmp_path / "data" / "synthetic" / "gedi" / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "syn_gedi_t0001.xes.gz").write_bytes(b"syn-a")
    (logs_dir / "syn_gedi_t0002.xes.gz").write_bytes(b"syn-b")
    fieldnames = ["target_id", "log_id", "status", "output_path"]
    rows = [
        # accepted rows carry an absolute external output_path that must be ignored
        # in favour of a portable path rebuilt from log_id under logs_dir.
        {
            "target_id": "t0001",
            "log_id": "syn_gedi_t0001",
            "status": "accepted",
            "output_path": "/mnt/external/logs/syn_gedi_t0001.xes.gz",
        },
        {
            "target_id": "t0001",
            "log_id": "syn_gedi_t0001",
            "status": "rejected",
            "output_path": "",
        },
        {
            "target_id": "t0002",
            "log_id": "syn_gedi_t0002",
            "status": "accepted",
            "output_path": "/mnt/external/logs/syn_gedi_t0002.xes.gz",
        },
        {
            "target_id": "t0003",
            "log_id": "syn_gedi_t0003",
            "status": "generation_failed",
            "output_path": "",
        },
    ]
    with synthetic_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    written = prepare_v6_synthetic_explore_configs(
        synthetic_manifest=synthetic_manifest,
        logs_dir=logs_dir,
        output_root=tmp_path / "configs",
        require_log_files=True,
    )

    assert set(written) == {
        "alpha_classic",
        "alpha_plus",
        "genetic",
        "heuristic_classic",
        "heuristic_plusplus",
        "inductive_im",
        "inductive_imd",
        "ilp",
        "inductive_imf",
        "split",
    }
    for algorithm_slug, synthetic_config_path in written.items():
        real_config_path = Path("configs/experiments/v6/explore") / algorithm_slug / "v1.yaml"
        real_rows = generate_manifest_rows(real_config_path)
        synthetic_rows = generate_manifest_rows(synthetic_config_path)

        real_params = _params_for_first_log(real_rows)
        synthetic_params = _params_for_first_log(synthetic_rows)
        assert synthetic_params == real_params
        # two accepted logs (deduped across attempts) times the parameter grid
        assert len(synthetic_rows) == 2 * len(real_params)
        assert all(row["log_id"].startswith("syn_gedi_") for row in synthetic_rows)
        assert all(
            row["log_path"] == f"{logs_dir.as_posix()}/{row['log_id']}.xes.gz"
            for row in synthetic_rows
        )
        assert all(
            row["output_path"].startswith(
                f"results/cluster/v6/model/explore_synthetic/{algorithm_slug}/"
            )
            for row in synthetic_rows
        )


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _params_for_first_log(rows: list[dict[str, str]]) -> list[dict]:
    first_log_id = rows[0]["log_id"]
    return [json.loads(row["params_json"]) for row in rows if row["log_id"] == first_log_id]


def _metric_row(
    config_hash: str,
    *,
    log_id: str = "log_1",
    algorithm_name: str = "heuristics_miner",
    fitness: float = 0.8,
    precision: float = 0.8,
    runtime: float = 10.0,
) -> dict[str, str]:
    return {
        "log_id": log_id,
        "algorithm_name": algorithm_name,
        "source_config_hash": config_hash,
        "status": "success",
        "runtime_seconds_discovery": str(runtime),
        "source_result_path": f"results/{config_hash}.json",
        "experiment_id": "v6_baseline_v1",
        "metric_fitness": str(fitness),
        "metric_precision": str(precision),
        "metric_generalization": "0.8",
        "metric_simplicity": "0.8",
        "metric_status_fitness": "success",
        "metric_status_precision": "success",
        "metric_status_generalization": "success",
        "metric_status_simplicity": "success",
        "param_variant": "classic",
    }


def _write_success(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": row["experiment_id"],
        "log_id": row["log_id"],
        "log_path": row["log_path"],
        "test_log_path": row["test_log_path"],
        "seed": int(row["seed"]),
        "algorithm_name": row["algorithm_id"],
        "backend": row["backend"],
        "hyperparameters": json.loads(row["params_json"]),
        "discovered_model_type": "petri_net",
        "metrics": {},
        "metric_statuses": {},
        "status": "success",
        "metadata": {
            "config_hash": row["config_hash"],
            "metrics_config": json.loads(row["metrics_json"]),
        },
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
