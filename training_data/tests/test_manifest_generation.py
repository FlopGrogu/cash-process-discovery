from __future__ import annotations

import csv
from pathlib import Path

import pytest

from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.experiments.manifest import (
    generate_manifest,
    generate_manifest_rows,
)
from process_discovery_cash.experiments.receipts import verify_v6_receipts
from process_discovery_cash.experiments.v6 import (
    discover_v6_ordinary_configs,
    discover_v6_primary_configs,
    generate_all_v6_ordinary_manifests,
    generate_v6_primary_manifests,
)


def test_repository_contains_only_the_46_canonical_v6_configs() -> None:
    all_configs = sorted(Path("configs/experiments").glob("**/*.yaml"))
    ordinary = discover_v6_ordinary_configs()
    hpo = sorted(Path("configs/experiments/v6/hpo").glob("*/*.yaml"))

    assert len(all_configs) == 46
    assert len(ordinary) == 40
    assert len(hpo) == 6
    assert all(path.as_posix().startswith("configs/experiments/v6/") for path in all_configs)


def test_baseline_manifest_is_deterministic_and_portable(tmp_path: Path) -> None:
    config = "configs/experiments/v6/baseline/alpha_classic/v1.yaml"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    generate_manifest(config, first)
    generate_manifest(config, second)

    assert first.read_bytes() == second.read_bytes()
    with first.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 21
    assert all(row["log_path"].startswith("data/") for row in rows)
    assert all(row["output_path"].startswith("results/") for row in rows)
    assert all(row["log_dir"].startswith("logs/slurm/") for row in rows)


def test_v6_output_root_regenerates_all_ordinary_manifests_in_tmp_path(
    tmp_path: Path,
) -> None:
    written = generate_all_v6_ordinary_manifests(output_root=tmp_path)

    assert len(written) == 40
    assert len(list(tmp_path.glob("**/*.csv"))) == 40
    assert all(path.is_relative_to(tmp_path) for path in written.values())


def test_primary_generation_contains_only_baseline_and_explore(
    tmp_path: Path,
) -> None:
    configs = discover_v6_primary_configs()
    written = generate_v6_primary_manifests(output_root=tmp_path)

    assert len(configs) == len(written) == 30
    assert {path.parents[1].name for path in configs} == {
        "baseline",
        "explore",
        "explore_synthetic",
    }
    assert len(list(tmp_path.glob("**/*.csv"))) == 30
    assert not list(tmp_path.glob("model/default_run_survey/**/*.csv"))
    assert not list(tmp_path.glob("hpo/**/*.csv"))
    verify_v6_receipts(tmp_path, scopes={"primary"})


@pytest.mark.parametrize(
    "config",
    sorted(Path("configs/experiments/v6/hpo").glob("*/*.yaml")),
)
def test_generic_manifest_generator_routes_hpo_to_study_generator(config: Path) -> None:
    assert len(load_experiment_config(config).logs) == 215
    with pytest.raises(ValueError, match="pdcash-generate-hpo-studies"):
        generate_manifest_rows(config)


def test_manifest_config_hash_changes_with_metric_profile(tmp_path: Path) -> None:
    source = Path("configs/experiments/v6/baseline/alpha_classic/v1.yaml").read_text(
        encoding="utf-8"
    )
    default_config = tmp_path / "default.yaml"
    alignment_config = tmp_path / "alignment.yaml"
    default_config.write_text(source.replace("profile: token", "profile: pm4py_default"))
    alignment_config.write_text(source.replace("profile: token", "profile: alignment"))

    default_rows = generate_manifest_rows(default_config)
    alignment_rows = generate_manifest_rows(alignment_config)

    assert default_rows[0]["config_hash"] != alignment_rows[0]["config_hash"]


def test_deprecated_artifact_flag_does_not_change_xes_manifest() -> None:
    config = "configs/experiments/v6/default_run_survey/alpha_classic/v1.yaml"

    ordinary_rows = generate_manifest_rows(config)
    with pytest.deprecated_call(match="XES-first"):
        compatibility_rows = generate_manifest_rows(config, require_artifacts=True)

    assert compatibility_rows == ordinary_rows
