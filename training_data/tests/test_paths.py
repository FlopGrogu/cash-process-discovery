from __future__ import annotations

from pathlib import Path

import process_discovery_cash.utils.paths as paths


def test_portable_path_namespaces_use_configured_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    data_root = tmp_path / "datasets"
    results_root = tmp_path / "outputs"
    log_root = tmp_path / "scheduler"

    monkeypatch.setenv("PROJECT_ROOT", project_root.as_posix())
    monkeypatch.setenv("DATA_ROOT", data_root.as_posix())
    monkeypatch.setenv("RESULTS_ROOT", results_root.as_posix())
    monkeypatch.setenv("LOG_ROOT", log_root.as_posix())

    assert paths.project_root() == project_root.resolve()
    assert paths.resolve_portable_path("data/raw/log.xes") == data_root / "raw/log.xes"
    assert paths.resolve_portable_path("results/run.json") == results_root / "run.json"
    assert paths.resolve_portable_path("logs/slurm/job.out") == log_root / "job.out"
    assert paths.portable_project_path(data_root / "raw/log.xes") == "data/raw/log.xes"
    assert paths.portable_project_path(results_root / "run.json") == "results/run.json"
    assert paths.portable_project_path(log_root / "job.out") == "logs/slurm/job.out"


def test_load_dotenv_does_not_override_existing_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "PROJECT_ROOT=/from/dotenv\nDATA_ROOT=/from/dotenv/data\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PROJECT_ROOT", "/from/environment")
    monkeypatch.delenv("DATA_ROOT", raising=False)
    monkeypatch.setattr(paths, "_DOTENV_LOADED", False)
    monkeypatch.setattr(paths, "_DOTENV_VALUES", {})

    paths.load_dotenv_if_present(dotenv_path)

    assert paths.project_root() == Path("/from/environment")
    assert paths.data_root() == Path("/from/dotenv/data")
