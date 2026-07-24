from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tomllib
from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

import process_discovery_cash.cli.generate_feature_space_logs as generation_cli
from process_discovery_cash.generation.gedi_backend import DEFAULT_GEDI_PYTHON

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPOSITORY = "https://github.com/FlopGrogu/cash-process-discovery"
ARCHIVE_VERIFIER_SPEC = spec_from_file_location(
    "verify_submission_archive",
    PROJECT_ROOT / "scripts/verify_submission_archive.py",
)
assert ARCHIVE_VERIFIER_SPEC is not None and ARCHIVE_VERIFIER_SPEC.loader is not None
ARCHIVE_VERIFIER = module_from_spec(ARCHIVE_VERIFIER_SPEC)
ARCHIVE_VERIFIER_SPEC.loader.exec_module(ARCHIVE_VERIFIER)
verify_archive = ARCHIVE_VERIFIER.verify_archive
AUDIT_SPEC = spec_from_file_location(
    "audit_submission",
    PROJECT_ROOT / "scripts/audit_submission.py",
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
AUDIT = module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def test_release_metadata_and_dependency_authorities_are_consistent() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    release = json.loads((PROJECT_ROOT / "release/v6.json").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == release["release"] == "6.0.0"
    assert pyproject["project"]["urls"]["Repository"] == CANONICAL_REPOSITORY
    assert release["repository"] == CANONICAL_REPOSITORY
    assert release["canonical_inventory"]["default_run_survey_rows"] == 210
    assert release["canonical_inventory"]["total_event_logs"] == 215
    assert release["canonical_inventory"]["primary_v6_configs"] == 30
    assert "hpo_logs" not in release["canonical_inventory"]
    assert "final_" + "bench" + "mark_rows" not in release["canonical_inventory"]
    assert (PROJECT_ROOT / "requirements.txt").is_file()
    assert (PROJECT_ROOT / "environments/gedi/requirements.txt").is_file()
    assert not list(PROJECT_ROOT.glob("*.csv"))
    assert len(
        list(
            PROJECT_ROOT.glob(
                "configs/experiments/v6/default_run_survey/*/v1.yaml"
            )
        )
    ) == 10
    receipt_ledger = json.loads(
        (PROJECT_ROOT / "release/v6-manifest-receipts.json").read_text(encoding="utf-8")
    )
    assert receipt_ledger["schema_version"] == 2
    assert receipt_ledger["primary_manifest_count"] == 30
    assert receipt_ledger["survey_manifest_count"] == 10
    assert receipt_ledger["hpo_study_manifest_count"] == 6


def test_quick_start_uses_canonical_clone_and_nested_project_directory() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "git clone https://github.com/FlopGrogu/cash-process-discovery.git" in readme
    assert "cd cash-process-discovery/training_data" in readme


def test_submission_inventory_is_scoped_to_nested_project() -> None:
    files = AUDIT.submission_files()

    assert PROJECT_ROOT / "README.md" in files
    assert files
    assert all(path.is_relative_to(PROJECT_ROOT) for path in files)


def test_make_workflows_use_plain_python_and_xes_sources() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "$(PYTHON) -m venv .venv" in makefile
    assert "$(PYTHON) -m venv .venv-gedi" in makefile
    assert "$(CORE_PYTHON) scripts/augment_logs.py --all" in makefile
    manifests_recipe = makefile.split("manifests:", 1)[1].split("manifests-survey:", 1)[0]
    assert "generate_v6_manifests.py --primary" in manifests_recipe
    assert "generate_hpo_studies.py" not in manifests_recipe
    assert "manifest_receipts.py --check --scope primary" in manifests_recipe
    assert "manifests-hpo:" in makefile
    assert "uv " not in makefile
    assert "pdcash-preprocess-event-logs" not in makefile


def test_primary_documentation_has_no_uv_or_container_requirement() -> None:
    primary_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs/setup.md",
        PROJECT_ROOT / "docs/reproducibility.md",
        PROJECT_ROOT / "docs/data.md",
        PROJECT_ROOT / "docs/experiments.md",
        PROJECT_ROOT / "docs/metrics.md",
    ]

    for path in primary_paths:
        content = path.read_text(encoding="utf-8")
        assert "uv run" not in content
        assert "uv sync" not in content
        assert "docker run" not in content.lower()
        assert "apptainer build" not in content.lower()

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    primary_workflow = readme.split("## XES-first workflow", 1)[1].split(
        "## Input data and generated inventory",
        1,
    )[0]
    assert "generate_hpo_studies.py" not in primary_workflow
    assert "run_hpo_study.py" not in primary_workflow


def test_gedi_default_matches_frozen_sidecar_environment() -> None:
    expected = ".venv-gedi/bin/python"
    submitter = (PROJECT_ROOT / "scripts/submit_gedi_targets_slurm.sh").read_text(
        encoding="utf-8"
    )
    payload = (PROJECT_ROOT / "slurm/run_gedi_target.sh").read_text(encoding="utf-8")

    assert DEFAULT_GEDI_PYTHON == expected
    assert f"${{PROJECT_ROOT}}/{expected}" in submitter
    assert f"${{PROJECT_ROOT}}/{expected}" in payload
    assert "pdcash_resolve_data_path" in submitter
    assert "pdcash_resolve_data_path" in payload


def test_generation_cli_resolves_default_output_below_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "external-data"
    anchor_calls: list[Path] = []

    monkeypatch.setenv("DATA_ROOT", data_root.as_posix())
    monkeypatch.setattr(
        generation_cli,
        "build_anchor_features",
        lambda _catalog, path, compute_missing: (
            anchor_calls.append(Path(path))
            or pd.DataFrame(
                [
                    {
                        "log_id": "real",
                        "num_traces": 10,
                        "avg_trace_length": 2.0,
                        "num_activities": 3,
                        "variant_ratio": 0.1,
                        "dfg_density": 0.1,
                        "repetition_prevalence": 0.1,
                    }
                ]
            )
        ),
    )
    monkeypatch.setattr(generation_cli, "design_targets", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        generation_cli,
        "targets_to_frame",
        lambda _targets: pd.DataFrame({"feasible": [], "band": []}),
    )

    generation_cli.main(["--mode", "design", "--n-targets", "1"])

    expected_root = data_root / "synthetic/gedi"
    assert anchor_calls == [expected_root / "anchor_features.csv"]
    assert (expected_root / "targets.csv").is_file()


def test_aggregate_resolves_results_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "external-results"
    captured: list[Path] = []
    monkeypatch.setenv("RESULTS_ROOT", results_root.as_posix())
    monkeypatch.setattr(
        generation_cli,
        "aggregate_results",
        lambda results_dir, *_args, **_kwargs: (
            captured.append(Path(results_dir)) or ([], {"n_result_files": 0})
        ),
    )
    monkeypatch.setattr(generation_cli, "_report", lambda *_args, **_kwargs: None)

    generation_cli._run_aggregate(
        Namespace(results_dir="results/gedi"),
        tmp_path / "data/synthetic/gedi",
        pd.DataFrame(),
    )

    assert captured == [results_root / "gedi"]


def test_submission_audit_passes_for_working_source() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_submission.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Submission audit passed" in result.stdout


def test_generated_submission_artifacts_are_ignored() -> None:
    ignored = {
        ".env.production",
        "credentials.json",
        "data/raw/private-log.xes",
        "experiments/manifests/generated.csv",
        "figures/workflow.pdf",
        "logs/slurm/job.err",
        "models/discovered.pnml",
        "results/local/result.json",
        "runs/hpo/study.journal",
        "submission-dist/process-discovery-cash-v6.tar.gz",
    }
    retained = {
        ".env.example",
        "data/example/tiny_log.xes",
        "data/raw/.gitkeep",
        "experiments/manifests/.gitkeep",
        "results/local/.gitkeep",
        "results/README.md",
    }
    candidates = sorted(ignored | retained)
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=PROJECT_ROOT,
        input="\n".join(candidates) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == ignored


def test_archive_verifier_accepts_source_only_and_rejects_data(
    tmp_path: Path,
) -> None:
    allowed = {
        "process-discovery-cash-v6/.python-version": "3.11.15\n",
        "process-discovery-cash-v6/Apptainer.def": "Bootstrap: docker\n",
        "process-discovery-cash-v6/Dockerfile": "FROM scratch\n",
        "process-discovery-cash-v6/LICENSE": "license\n",
        "process-discovery-cash-v6/Makefile": "check:\n",
        "process-discovery-cash-v6/README.md": "readme\n",
        "process-discovery-cash-v6/THIRD_PARTY_NOTICES.md": "notices\n",
        "process-discovery-cash-v6/docs/cluster.md": "cluster\n",
        "process-discovery-cash-v6/environments/gedi/pyproject.toml": "[project]\n",
        "process-discovery-cash-v6/environments/gedi/requirements.txt": "gedi==1.0.8\n",
        "process-discovery-cash-v6/pyproject.toml": "[project]\n",
        "process-discovery-cash-v6/requirements.txt": "pm4py==2.7.22.2\n",
        "process-discovery-cash-v6/release/v6-manifest-receipts.json": "{}\n",
        "process-discovery-cash-v6/release/v6.json": "{}\n",
        "process-discovery-cash-v6/scripts/audit_submission.py": "pass\n",
        "process-discovery-cash-v6/scripts/verify_submission_archive.py": "pass\n",
        "process-discovery-cash-v6/data/example/tiny_log.xes": "<log />\n",
        "process-discovery-cash-v6/data/raw/.gitkeep": "",
        "process-discovery-cash-v6/results/README.md": "results\n",
        "process-discovery-cash-v6/experiments/manifests/.gitkeep": "",
    }
    archive = _write_archive(tmp_path / "source.tar.gz", allowed)
    checksum = _write_checksum(archive)

    assert verify_archive(archive, checksum) == hashlib.sha256(archive.read_bytes()).hexdigest()

    forbidden = dict(allowed)
    forbidden["process-discovery-cash-v6/data/raw/private-log.xes"] = "<log />\n"
    archive = _write_archive(tmp_path / "with-data.tar.gz", forbidden)
    checksum = _write_checksum(archive)

    with pytest.raises(ValueError, match="non-source data artifact"):
        verify_archive(archive, checksum)

    forbidden = dict(allowed)
    forbidden["process-discovery-cash-v6/.env.production"] = "TOKEN=secret\n"
    archive = _write_archive(tmp_path / "with-secret.tar.gz", forbidden)
    checksum = _write_checksum(archive)

    with pytest.raises(ValueError, match="secret-bearing file"):
        verify_archive(archive, checksum)

    forbidden = dict(allowed)
    forbidden["process-discovery-cash-v6/models/discovered.pnml"] = "<pnml />\n"
    archive = _write_archive(tmp_path / "with-model.tar.gz", forbidden)
    checksum = _write_checksum(archive)

    with pytest.raises(ValueError, match="generated model artifact"):
        verify_archive(archive, checksum)

    forbidden = dict(allowed)
    forbidden["process-discovery-cash-v6/figures/workflow.pdf"] = "generated figure\n"
    archive = _write_archive(tmp_path / "with-figure.tar.gz", forbidden)
    checksum = _write_checksum(archive)

    with pytest.raises(ValueError, match="generated artifact directory"):
        verify_archive(archive, checksum)

    forbidden = dict(allowed)
    forbidden["process-discovery-cash-v6/experiments/generated/run.json"] = "{}\n"
    archive = _write_archive(tmp_path / "with-generated-experiment.tar.gz", forbidden)
    checksum = _write_checksum(archive)

    with pytest.raises(ValueError, match="generated experiment artifact"):
        verify_archive(archive, checksum)

    forbidden = dict(allowed)
    forbidden["process-discovery-cash-v6/release/debug-output.zip"] = "generated archive\n"
    archive = _write_archive(tmp_path / "with-generated-archive.tar.gz", forbidden)
    checksum = _write_checksum(archive)

    with pytest.raises(ValueError, match="generated archive"):
        verify_archive(archive, checksum)


def _write_archive(path: Path, members: dict[str, str]) -> Path:
    with tarfile.open(path, mode="w:gz") as handle:
        for name, content in members.items():
            payload = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            handle.addfile(info, io.BytesIO(payload))
    return path


def _write_checksum(archive: Path) -> Path:
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return checksum
