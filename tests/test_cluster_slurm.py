from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from process_discovery_cash.experiments.manifest_validation import validate_manifest_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_manifest_row_has_portable_paths_and_array_validation() -> None:
    text = (PROJECT_ROOT / "slurm/run_manifest_row.sh").read_text(encoding="utf-8")

    for forbidden in [
        "/private/" + "cluster/account",
        "/home/" + "example-user",
        "/" + "Users/example-user",
    ]:
        assert forbidden not in text
    assert "#SBATCH --partition=" not in text
    assert "#SBATCH --partition=all" not in text
    assert "sbatch -p all" not in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"' not in text
    assert 'SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_PROJECT_ROOT}}"' in text
    assert "PROJECT_ROOT does not look like the process-mining-cash repository" in text
    assert 'echo "PWD=$(pwd)"' in text
    assert 'echo "MANIFEST_PATH=${MANIFEST_PATH}"' in text
    assert '[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "SLURM_ARRAY_TASK_ID is not set"' in text
    assert "ACTUAL_ARRAY_TASK_ID=$((MANIFEST_ROW_OFFSET + SLURM_ARRAY_TASK_ID))" in text
    assert 'echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"' in text
    assert 'echo "MANIFEST_ROW_OFFSET=${MANIFEST_ROW_OFFSET}"' in text
    assert 'echo "ACTUAL_MANIFEST_ROW_INDEX=${ACTUAL_ARRAY_TASK_ID}"' in text
    assert '--row-index "${ACTUAL_ARRAY_TASK_ID}"' in text
    assert "SELECTED_ROW_INDEX" in text


def test_dynamic_manifest_slurm_uses_worker_pool_and_portable_paths() -> None:
    text = (PROJECT_ROOT / "slurm/run_dynamic_manifest.slurm").read_text(encoding="utf-8")

    for forbidden in [
        "/private/" + "cluster/account",
        "/home/" + "example-user",
        "/" + "Users/example-user",
    ]:
        assert forbidden not in text
    assert "#SBATCH --partition=" not in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --mem=16G" in text
    assert "#SBATCH --output=logs/slurm/dynamic_workers/bootstrap_%j.out" in text
    assert "#SBATCH --error=logs/slurm/dynamic_workers/bootstrap_%j.err" in text
    assert "bash slurm/run_dynamic_manifest.slurm [sbatch options] <manifest.csv>" in text
    assert "0-N%M" not in text
    assert 'SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_PROJECT_ROOT}}"' in text
    assert "resolve_manifest_log_dir" in text
    assert "resolve_worker_log_dir" in text
    assert 'STATE_DIR="${DYNAMIC_STATE_DIR:-}"' in text
    assert '"--output=${worker_log_dir_abs}/%A_%a.out"' in text
    assert '"--error=${worker_log_dir_abs}/%A_%a.err"' in text
    assert "scripts/run_dynamic_worker.py" in text
    assert "--worker-walltime-seconds" in text
    assert "--safety-margin-seconds" in text
    assert "--retry-failed" in text


def test_slurm_readme_lists_only_v6_entry_points() -> None:
    text = (PROJECT_ROOT / "slurm/README.md").read_text(encoding="utf-8")

    assert "Only the v6 workflow is supported." in text
    assert "run_dynamic_manifest.slurm" in text
    assert "run_dynamic_metric_manifest.slurm" in text
    assert "run_hpo_study.slurm" in text


def test_direct_discovery_array_template_uses_uniform_runtime_resources() -> None:
    text = (PROJECT_ROOT / "slurm/templates/discovery_array.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --mem=16G" in text
    assert "#SBATCH --cpus-per-task=1" in text


def test_dynamic_manifest_wrapper_submits_into_manifest_log_directory(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=1,
        algorithm="alpha_miner",
        log_dir="logs/slurm/v6/baseline/alpha_classic/v1",
    )

    result = _run_dynamic_wrapper(
        tmp_path,
        ["--array=0-4", "--job-name=alpha-workers", manifest_path.as_posix()],
        env_overrides={"SBATCH_BIN": "echo"},
    )

    assert result.returncode == 0
    expected = "logs/slurm/v6/baseline/alpha_classic/v1"
    physical = tmp_path / "runtime/logs/v6/baseline/alpha_classic/v1/dynamic_workers"
    assert f"Manifest log directory={expected}" in result.stdout
    assert f"Dynamic worker log directory={expected}/dynamic_workers" in result.stdout
    assert f"--output={physical}/%A_%a.out" in result.stdout
    assert f"--error={physical}/%A_%a.err" in result.stdout
    assert "--time=24:00:00" in result.stdout
    assert "--mem=16G" in result.stdout
    assert physical.is_dir()


def test_dynamic_manifest_wrapper_accepts_manifest_without_sbatch_options(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=1,
        algorithm="alpha_miner",
        log_dir="logs/slurm/v6/default_run_survey/alpha_classic/v1",
    )

    result = _run_dynamic_wrapper(
        tmp_path,
        [manifest_path.as_posix()],
        env_overrides={"SBATCH_BIN": "echo"},
    )

    assert result.returncode == 0
    assert "--time=24:00:00" in result.stdout
    assert "--mem=16G" in result.stdout


def test_dynamic_manifest_wrapper_preserves_explicit_runtime_resources(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=1,
        algorithm="alpha_miner",
        log_dir="logs/slurm/v6/default_run_survey/alpha_classic/v1",
    )

    result = _run_dynamic_wrapper(
        tmp_path,
        [
            "--time=02:00:00",
            "--mem=7G",
            "--array=0-0",
            manifest_path.as_posix(),
        ],
        env_overrides={"SBATCH_BIN": "echo"},
    )

    assert result.returncode == 0
    submitting = next(
        line for line in result.stdout.splitlines() if line.startswith("Submitting:")
    )
    assert submitting.count("--time=02:00:00") == 1
    assert submitting.count("--mem=7G") == 1
    assert "--time=24:00:00" not in submitting
    assert "--mem=16G" not in submitting


def test_dynamic_manifest_wrapper_mirrors_v6_manifest_path(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path
        / "experiments"
        / "manifests"
        / "v6"
        / "default_run_survey"
        / "alpha_classic"
        / "v1.csv",
        row_count=1,
        algorithm="alpha_miner",
        log_dir="logs/slurm/v6/source/alpha",
    )

    result = _run_dynamic_wrapper(
        tmp_path,
        ["--array=0-4", "--job-name=alpha-workers", manifest_path.as_posix()],
        env_overrides={"SBATCH_BIN": "echo"},
    )

    assert result.returncode == 0
    assert "Manifest log directory=logs/slurm/v6/source/alpha" in result.stdout
    assert (
        "Dynamic worker log directory="
        "logs/slurm/v6/default_run_survey/alpha_classic/v1/dynamic_workers"
        in result.stdout
    )
    physical = (
        tmp_path
        / "runtime/logs/v6/default_run_survey/alpha_classic/v1/dynamic_workers"
    )
    assert f"--output={physical}/%A_%a.out" in result.stdout
    assert f"--error={physical}/%A_%a.err" in result.stdout
    assert physical.is_dir()


def test_dynamic_manifest_wrapper_rejects_inconsistent_log_directories(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    fieldnames = [
        "experiment_id",
        "log_id",
        "log_path",
        "seed",
        "algorithm_id",
        "log_dir",
        "output_path",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "test_01",
                "log_id": "tiny_log",
                "log_path": "data/example/tiny_log.xes",
                "seed": "0",
                "algorithm_id": "alpha_miner",
                "log_dir": "logs/slurm/v6/baseline/alpha_classic/v1",
                "output_path": "results/cluster/test_submit/result_0.json",
            }
        )
        writer.writerow(
            {
                "experiment_id": "test_01",
                "log_id": "tiny_log",
                "log_path": "data/example/tiny_log.xes",
                "seed": "0",
                "algorithm_id": "alpha_miner",
                "log_dir": "logs/slurm/v6/baseline/alpha_classic/v2",
                "output_path": "results/cluster/test_submit/result_1.json",
            }
        )

    result = _run_dynamic_wrapper(
        tmp_path,
        ["--array=0-1", manifest_path.as_posix()],
        env_overrides={"SBATCH_BIN": "echo"},
    )

    assert result.returncode == 1
    assert "inconsistent log_dir" in result.stderr


def test_dynamic_metric_manifest_slurm_uses_worker_pool_and_portable_paths() -> None:
    text = (PROJECT_ROOT / "slurm/run_dynamic_metric_manifest.slurm").read_text(encoding="utf-8")

    for forbidden in [
        "/private/" + "cluster/account",
        "/home/" + "example-user",
        "/" + "Users/example-user",
    ]:
        assert forbidden not in text
    assert "#SBATCH --partition=" not in text
    assert "#SBATCH --time=" not in text
    assert "#SBATCH --mem=" not in text
    assert "#SBATCH --output=logs/slurm/dynamic_metric_workers/bootstrap_%j.out" in text
    assert "#SBATCH --error=logs/slurm/dynamic_metric_workers/bootstrap_%j.err" in text
    assert (
        "bash slurm/run_dynamic_metric_manifest.slurm [sbatch options] <metric-manifest.csv>"
        in text
    )
    assert "0-N%M" not in text
    assert 'SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_PROJECT_ROOT}}"' in text
    assert 'STATE_DIR="${DYNAMIC_METRIC_STATE_DIR:-}"' in text
    assert "resolve_manifest_log_dir" in text
    assert "resolve_worker_log_dir" in text
    assert '"--output=${worker_log_dir_abs}/%A_%a.out"' in text
    assert '"--error=${worker_log_dir_abs}/%A_%a.err"' in text
    assert "scripts/run_dynamic_metric_worker.py" in text
    assert "--worker-walltime-seconds" in text
    assert "--safety-margin-seconds" in text
    assert "--retry-failed" in text


def test_dynamic_metric_wrapper_submits_into_metric_manifest_log_directory(tmp_path: Path) -> None:
    manifest_path = _write_metric_manifest(tmp_path / "metrics.csv", row_count=1, profile="token")

    result = _run_dynamic_metric_wrapper(
        tmp_path,
        ["--array=0-4", "--job-name=token-workers", manifest_path.as_posix()],
        env_overrides={"SBATCH_BIN": "echo"},
    )

    assert result.returncode == 0
    assert "Metric manifest log directory=logs/slurm/metrics/test_submit/token" in result.stdout
    assert (
        "Dynamic metric worker log directory=logs/slurm/metrics/test_submit/token/dynamic_workers"
        in result.stdout
    )
    physical = tmp_path / "runtime/logs/metrics/test_submit/token/dynamic_workers"
    assert f"--output={physical}/%A_%a.out" in result.stdout
    assert f"--error={physical}/%A_%a.err" in result.stdout
    assert physical.is_dir()


def test_dynamic_metric_wrapper_mirrors_metric_manifest_path(tmp_path: Path) -> None:
    manifest_path = _write_metric_manifest(
        tmp_path
        / "experiments"
        / "manifests"
        / "v6"
        / "metrics"
        / "baseline"
        / "alpha_plus"
        / "v1"
        / "token_metrics.csv",
        row_count=1,
        profile="token",
    )

    result = _run_dynamic_metric_wrapper(
        tmp_path,
        ["--array=0-4", "--job-name=token-workers", manifest_path.as_posix()],
        env_overrides={"SBATCH_BIN": "echo"},
    )

    assert result.returncode == 0
    assert "Metric manifest log directory=logs/slurm/metrics/test_submit/token" in result.stdout
    assert (
        "Dynamic metric worker log directory="
        "logs/slurm/v6/model/baseline/alpha_plus/v1/metrics/token/dynamic_workers" in result.stdout
    )
    physical = (
        tmp_path
        / "runtime/logs/v6/model/baseline/alpha_plus/v1/metrics/token/dynamic_workers"
    )
    assert f"--output={physical}/%A_%a.out" in result.stdout
    assert f"--error={physical}/%A_%a.err" in result.stdout
    assert physical.is_dir()


def test_dynamic_metric_wrapper_does_not_override_manifest_timeout_by_default(
    tmp_path: Path,
) -> None:
    manifest_path = _write_metric_manifest(tmp_path / "metrics.csv", row_count=1)

    result = _run_dynamic_metric_wrapper(
        tmp_path,
        [manifest_path.as_posix()],
        env_overrides={
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_CPUS_PER_TASK": "1",
            "METRIC_TIMEOUT_SECONDS": "",
            "PYTHON": "/bin/echo",
        },
    )

    assert result.returncode == 0
    assert "--metric-timeout-seconds" not in result.stdout


def test_dynamic_metric_wrapper_forwards_explicit_metric_timeout_override(
    tmp_path: Path,
) -> None:
    manifest_path = _write_metric_manifest(tmp_path / "metrics.csv", row_count=1)

    result = _run_dynamic_metric_wrapper(
        tmp_path,
        [manifest_path.as_posix()],
        env_overrides={
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_CPUS_PER_TASK": "1",
            "METRIC_TIMEOUT_SECONDS": "7200",
            "PYTHON": "/bin/echo",
        },
    )

    assert result.returncode == 0
    assert "--metric-timeout-seconds 7200" in result.stdout


def test_run_metric_row_has_portable_paths_and_array_validation() -> None:
    text = (PROJECT_ROOT / "slurm/run_metric_row.sh").read_text(encoding="utf-8")

    for forbidden in [
        "/private/" + "cluster/account",
        "/home/" + "example-user",
        "/" + "Users/example-user",
    ]:
        assert forbidden not in text
    assert "#SBATCH --partition=" not in text
    assert "#SBATCH --partition=all" not in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"' not in text
    assert 'SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_PROJECT_ROOT}}"' in text
    assert "PROJECT_ROOT does not look like the process-mining-cash repository" in text
    assert '[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "SLURM_ARRAY_TASK_ID is not set"' in text
    assert "ACTUAL_ARRAY_TASK_ID=$((METRIC_MANIFEST_ROW_OFFSET + SLURM_ARRAY_TASK_ID))" in text
    assert 'echo "METRIC_MANIFEST_ROW_OFFSET=${METRIC_MANIFEST_ROW_OFFSET}"' in text
    assert 'echo "ACTUAL_METRIC_MANIFEST_ROW_INDEX=${ACTUAL_ARRAY_TASK_ID}"' in text
    assert "--slurm-array-task-id" in text
    assert "SELECTED_ROW_INDEX" in text
    assert '"model_path"' not in text


def test_submit_manifest_slurm_exports_roots_and_chdir() -> None:
    text = (PROJECT_ROOT / "scripts/submit_manifest_slurm.sh").read_text(encoding="utf-8")

    assert '--chdir "${PROJECT_ROOT}"' in text
    assert (
        '--export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},'
        "RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${RESOLVED_LOG_ROOT},"
        "DISCOVERY_ALGORITHM=${ALGORITHM},"
        "PDCASH_SLURM_REQUESTED_MEMORY=${MEMORY},"
        'MANIFEST_ROW_OFFSET=${row_offset}"'
    ) in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"' not in text


def test_submit_metric_manifest_slurm_exports_roots_and_chdir() -> None:
    text = (PROJECT_ROOT / "scripts/submit_metric_manifest_slurm.sh").read_text(encoding="utf-8")

    assert '--chdir "${PROJECT_ROOT}"' in text
    assert (
        '--export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},'
        "RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},"
        'METRIC_PROFILE=${PROFILE},METRIC_MANIFEST_ROW_OFFSET=${row_offset}"'
    ) in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"' not in text
    assert '"model_path"' not in text


def test_shell_dotenv_loader_loads_values_without_overriding(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "# ignored",
                "PDCASH_TEST_FOO=from-dotenv",
                "PDCASH_TEST_QUOTED='quoted value'",
                "export PDCASH_TEST_EXPORTED=exported",
                "PDCASH_TEST_EXISTING=from-dotenv",
                "1INVALID=ignored",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source scripts/lib/env.sh; "
                "export PDCASH_TEST_EXISTING=from-env; "
                'pdcash_load_dotenv "$1"; '
                'printf "%s|%s|%s|%s|%s" '
                '"${PDCASH_TEST_FOO}" '
                '"${PDCASH_TEST_QUOTED}" '
                '"${PDCASH_TEST_EXPORTED}" '
                '"${PDCASH_TEST_EXISTING}" '
                '"$(env | grep -q "^1INVALID=" && printf set || printf unset)"'
            ),
            "_",
            dotenv_path.as_posix(),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "from-dotenv|quoted value|exported|from-env|unset"


def test_submit_manifest_slurm_rejects_partition_all(tmp_path: Path) -> None:
    result = _run_submit(
        tmp_path,
        ["--manifest", "missing.csv", "--partition", "all"],
    )

    assert result.returncode == 1
    assert "partition 'all' is not allowed for student accounts" in result.stderr


def test_submit_metric_manifest_slurm_rejects_partition_all(tmp_path: Path) -> None:
    result = _run_submit_metric(
        tmp_path,
        ["--manifest", "missing.csv", "--partition", "all"],
    )

    assert result.returncode == 1
    assert "partition 'all' is not allowed for student accounts" in result.stderr


def test_submit_manifest_slurm_reads_dotenv_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in [
        "DISCOVERY_PARTITION",
        "DISCOVERY_QOS",
        "DISCOVERY_TIME",
        "DISCOVERY_MEM",
        "ALPHA_MINER_PARTITION",
        "ALPHA_MINER_QOS",
        "ALPHA_MINER_TIME",
        "ALPHA_MINER_MEM",
    ]:
        monkeypatch.delenv(name, raising=False)

    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=1)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "DISCOVERY_PARTITION=minor",
                "DISCOVERY_QOS=minor_student_prio",
                "DISCOVERY_TIME=00:20:00",
                "DISCOVERY_MEM=5G",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_submit(
        tmp_path,
        ["--manifest", manifest_path.as_posix(), "--dry-run"],
        env_overrides={"PDCASH_DOTENV_PATH": dotenv_path.as_posix()},
    )

    assert result.returncode == 0
    assert "Partition=minor" in result.stdout
    assert "QOS=minor_student_prio" in result.stdout
    assert "Time=00:20:00" in result.stdout
    assert "Memory=5G" in result.stdout


def test_slurm_spool_copy_uses_project_root_to_load_dotenv_support(tmp_path: Path) -> None:
    spool_script = tmp_path / "slurm_script"
    shutil.copy(PROJECT_ROOT / "slurm/run_manifest_row.sh", spool_script)

    result = subprocess.run(
        ["bash", spool_script.as_posix(), "missing.csv"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PROJECT_ROOT": PROJECT_ROOT.as_posix(),
            "SLURM_ARRAY_TASK_ID": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "scripts/lib/env.sh: No such file or directory" not in result.stderr
    assert "manifest does not exist: missing.csv" in result.stderr


def test_slurm_spool_copy_uses_submit_dir_to_load_dotenv_support(tmp_path: Path) -> None:
    spool_script = tmp_path / "slurm_script"
    shutil.copy(PROJECT_ROOT / "slurm/run_manifest_row.sh", spool_script)

    result = subprocess.run(
        ["bash", spool_script.as_posix(), "missing.csv"],
        cwd=tmp_path,
        env={
            **os.environ,
            "SLURM_SUBMIT_DIR": PROJECT_ROOT.as_posix(),
            "SLURM_ARRAY_TASK_ID": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "scripts/lib/env.sh: No such file or directory" not in result.stderr
    assert "manifest does not exist: missing.csv" in result.stderr


def test_submit_manifest_slurm_accepts_configurable_partitions(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=1)

    for partition in ["cpu", "minor", "major"]:
        result = _run_submit(
            tmp_path,
            [
                "--manifest",
                manifest_path.as_posix(),
                "--partition",
                partition,
                "--time",
                "00:30:00",
                "--mem",
                "6G",
                "--dry-run",
            ],
        )

        assert result.returncode == 0
        assert f"Partition={partition}" in result.stdout
        assert f"sbatch --partition={partition}" in result.stdout
        assert "--time=00:30:00" in result.stdout
        assert "--mem=6G" in result.stdout
        assert f"--chdir {PROJECT_ROOT}" in result.stdout
        assert "--export=ALL,PROJECT_ROOT=" in result.stdout
        assert "DATA_ROOT=" in result.stdout
        assert "RESULTS_ROOT=" in result.stdout
        assert "LOG_ROOT=" in result.stdout
        assert "DISCOVERY_ALGORITHM=alpha_miner" in result.stdout
        assert "PDCASH_SLURM_REQUESTED_MEMORY=6G" in result.stdout
        assert "--array=0-0" in result.stdout
        if partition == "cpu":
            # The CPU partition requires an explicit student QOS by default.
            assert "QOS=minor_student" in result.stdout
            assert "--qos=minor_student" in result.stdout
        else:
            assert "QOS=\n" in result.stdout
            assert "--qos=" not in result.stdout
        assert "slurm/run_manifest_row.sh" in result.stdout


def test_submit_metric_manifest_slurm_alignment_defaults(tmp_path: Path) -> None:
    manifest_path = _write_metric_manifest(
        tmp_path / "metric_manifest.csv",
        row_count=1,
        profile="alignment",
    )

    result = _run_submit_metric(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "Metric profile=alignment" in result.stdout
    assert "Partition=CPU" in result.stdout
    assert "QOS=minor_student" in result.stdout
    assert "--qos=minor_student" in result.stdout
    assert "Time=24:00:00" in result.stdout
    assert "Memory=32G" in result.stdout
    assert "--job-name=metrics_alignment" in result.stdout
    assert f"--output={tmp_path}/runtime/logs/metrics_alignment_%A_%a.out" in result.stdout
    assert "--export=ALL,PROJECT_ROOT=" in result.stdout
    assert "METRIC_PROFILE=alignment" in result.stdout
    assert "METRIC_MANIFEST_ROW_OFFSET=0" in result.stdout
    assert "slurm/run_metric_row.sh" in result.stdout


def test_submit_metric_manifest_slurm_token_defaults(tmp_path: Path) -> None:
    manifest_path = _write_metric_manifest(
        tmp_path / "metric_manifest.csv",
        row_count=1,
        profile="token",
    )

    result = _run_submit_metric(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "Metric profile=token" in result.stdout
    assert "Partition=CPU" in result.stdout
    assert "Time=06:00:00" in result.stdout
    assert "Memory=16G" in result.stdout
    assert "QOS=minor_student" in result.stdout
    assert "--qos=minor_student" in result.stdout


def test_submit_manifest_slurm_computes_full_array_range(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=3)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--partition",
            "minor",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "--array=0-2" in result.stdout


def test_submit_manifest_slurm_computes_max_rows_array_range(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=3)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--partition",
            "minor",
            "--max-rows",
            "2",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "--array=0-1" in result.stdout


def test_submit_manifest_slurm_computes_single_row_array(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=2)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--partition",
            "minor",
            "--row",
            "1",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "--array=0 " in result.stdout
    assert "MANIFEST_ROW_OFFSET=1" in result.stdout
    assert "Manifest row range=1-1" in result.stdout


def test_submit_manifest_slurm_accepts_explicit_array_range_and_exclude(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=1000,
        algorithm="inductive_miner",
    )

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--algorithm",
            "inductive_miner",
            "--array=0-999",
            "--exclude",
            "worker-minor-6",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "Algorithm=inductive_miner" in result.stdout
    assert "Partition=CPU" in result.stdout
    assert "Time=24:00:00" in result.stdout
    assert "Memory=16G" in result.stdout
    assert "--array=0-999" in result.stdout
    assert "--exclude=worker-minor-6" in result.stdout
    assert f"--output={tmp_path}/runtime/logs/inductive_miner_%A_%a.out" in result.stdout
    assert f"--error={tmp_path}/runtime/logs/inductive_miner_%A_%a.err" in result.stdout
    assert "--qos=minor_student" in result.stdout


def test_submit_manifest_slurm_does_not_exclude_inductive_workers_by_default(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=1,
        algorithm="inductive_miner",
    )

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--algorithm",
            "inductive_miner",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "Exclude=\n" in result.stdout
    assert "--exclude=" not in result.stdout
    assert "worker-minor-6" not in result.stdout


def test_submit_manifest_slurm_uses_and_creates_experiment_log_subdirectory(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=1,
        algorithm="inductive_miner",
    )
    log_subdir = "v6_inductive_timeout_test"

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--log-subdir",
            log_subdir,
            "--time",
            "00:40:00",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "Time=00:40:00" in result.stdout
    assert f"Slurm log directory=logs/slurm/{log_subdir}" in result.stdout
    physical = tmp_path / "runtime/logs" / log_subdir
    assert f"--output={physical}/inductive_miner_%A_%a.out" in result.stdout
    assert f"--error={physical}/inductive_miner_%A_%a.err" in result.stdout
    assert physical.is_dir()


def test_submit_manifest_slurm_uses_manifest_log_directory(tmp_path: Path) -> None:
    log_dir = "logs/slurm/v6/baseline/alpha_classic/v1"
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=1,
        algorithm="alpha_miner",
        log_dir=log_dir,
    )

    result = _run_submit(
        tmp_path,
        ["--manifest", manifest_path.as_posix(), "--dry-run"],
    )

    assert result.returncode == 0
    assert f"Manifest log directory={log_dir}" in result.stdout
    assert f"Slurm log directory={log_dir}" in result.stdout
    physical = tmp_path / "runtime/logs/v6/baseline/alpha_classic/v1"
    assert f"--output={physical}/alpha_miner_%A_%a.out" in result.stdout
    assert f"--error={physical}/alpha_miner_%A_%a.err" in result.stdout
    assert f"LOG_ROOT={physical}" in result.stdout
    assert physical.is_dir()


def test_submit_manifest_slurm_algorithm_defaults_are_uniform(tmp_path: Path) -> None:
    algorithms = [
        "alpha_miner",
        "alpha_miner_classic",
        "alpha_miner_plus",
        "heuristics_miner",
        "heuristics_miner_plusplus",
        "inductive_miner",
        "inductive_miner_im",
        "inductive_miner_imd",
        "inductive_miner_imf",
        "ilp_miner",
        "genetic_miner",
        "split_miner",
    ]

    for algorithm in algorithms:
        manifest_path = _write_manifest(
            tmp_path / f"{algorithm}.csv",
            row_count=1,
            algorithm=algorithm,
        )
        result = _run_submit(
            tmp_path,
            [
                "--manifest",
                manifest_path.as_posix(),
                "--algorithm",
                algorithm,
                "--dry-run",
            ],
        )

        assert result.returncode == 0
        assert "Time=24:00:00" in result.stdout
        assert "--time=24:00:00" in result.stdout
        assert "Memory=16G" in result.stdout
        assert "--mem=16G" in result.stdout


def test_submit_manifest_slurm_offsets_manual_array_range(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=2000,
        algorithm="heuristics_miner",
    )

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--algorithm",
            "heuristics_miner",
            "--array",
            "0-999",
            "--row-offset",
            "1000",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "--array=0-999" in result.stdout
    assert "MANIFEST_ROW_OFFSET=1000" in result.stdout
    assert "Manifest row range=1000-1999" in result.stdout
    assert "--export=ALL,PROJECT_ROOT=" in result.stdout
    assert "MANIFEST_ROW_OFFSET=1000" in result.stdout
    assert "--job-name=heuristics_miner_o1000" in result.stdout
    assert f"--output={tmp_path}/runtime/logs/heuristics_miner_o1000_%A_%a.out" in result.stdout
    assert f"--error={tmp_path}/runtime/logs/heuristics_miner_o1000_%A_%a.err" in result.stdout


def test_submit_manifest_slurm_rejects_offset_array_past_manifest_end(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=2000)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--array",
            "0-999",
            "--row-offset",
            "1001",
            "--dry-run",
        ],
    )

    assert result.returncode == 1
    assert "outside manifest range" in result.stderr


def test_submit_manifest_slurm_rejects_raw_array_above_default_task_limit(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=2000)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--array",
            "1000-1999",
            "--dry-run",
        ],
    )

    assert result.returncode == 1
    assert "exceeds --max-array-tasks 1000" in result.stderr


def test_submit_manifest_slurm_all_rows_dry_run_chunks_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=2501,
        algorithm="heuristics_miner",
    )

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--algorithm",
            "heuristics_miner",
            "--all-rows",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert result.stdout.count("sbatch --partition=CPU") == 3
    assert "--array=0-999" in result.stdout
    assert "--array=0-500" in result.stdout
    assert "MANIFEST_ROW_OFFSET=0" in result.stdout
    assert "MANIFEST_ROW_OFFSET=1000" in result.stdout
    assert "MANIFEST_ROW_OFFSET=2000" in result.stdout
    assert "Manifest row range=0-999" in result.stdout
    assert "Manifest row range=1000-1999" in result.stdout
    assert "Manifest row range=2000-2500" in result.stdout
    assert "--job-name=heuristics_miner_o0" in result.stdout
    assert "--job-name=heuristics_miner_o1000" in result.stdout
    assert "--job-name=heuristics_miner_o2000" in result.stdout
    assert f"--output={tmp_path}/runtime/logs/heuristics_miner_o1000_%A_%a.out" in result.stdout


def test_submit_metric_manifest_slurm_all_rows_dry_run_chunks_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _write_metric_manifest(
        tmp_path / "metric_manifest.csv",
        row_count=2501,
        profile="alignment",
    )

    result = _run_submit_metric(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--all-rows",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert result.stdout.count("sbatch --partition=CPU") == 3
    assert "--array=0-999" in result.stdout
    assert "--array=0-500" in result.stdout
    assert "METRIC_MANIFEST_ROW_OFFSET=0" in result.stdout
    assert "METRIC_MANIFEST_ROW_OFFSET=1000" in result.stdout
    assert "METRIC_MANIFEST_ROW_OFFSET=2000" in result.stdout
    assert "Metric manifest row range=0-999" in result.stdout
    assert "Metric manifest row range=1000-1999" in result.stdout
    assert "Metric manifest row range=2000-2500" in result.stdout
    assert "--job-name=metrics_alignment_o0" in result.stdout
    assert "--job-name=metrics_alignment_o1000" in result.stdout
    assert "--job-name=metrics_alignment_o2000" in result.stdout


def test_submit_manifest_slurm_all_rows_respects_smaller_chunk_size(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=1201)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--all-rows",
            "--max-array-tasks",
            "500",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert result.stdout.count("sbatch --partition=CPU") == 3
    assert "--array=0-499" in result.stdout
    assert "--array=0-200" in result.stdout
    assert "MANIFEST row range" not in result.stdout
    assert "Manifest row range=1000-1200" in result.stdout


def test_submit_metric_manifest_slurm_offsets_manual_array_range(tmp_path: Path) -> None:
    manifest_path = _write_metric_manifest(
        tmp_path / "metric_manifest.csv",
        row_count=2000,
        profile="token",
    )

    result = _run_submit_metric(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--array",
            "0-999",
            "--row-offset",
            "1000",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "--array=0-999" in result.stdout
    assert "METRIC_MANIFEST_ROW_OFFSET=1000" in result.stdout
    assert "Metric manifest row range=1000-1999" in result.stdout
    assert "--job-name=metrics_token_o1000" in result.stdout
    assert f"--output={tmp_path}/runtime/logs/metrics_token_o1000_%A_%a.out" in result.stdout


def test_submit_manifest_slurm_all_rows_rejects_conflicting_options(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=10)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--all-rows",
            "--row-offset",
            "1000",
            "--dry-run",
        ],
    )

    assert result.returncode == 1
    assert "--all-rows cannot be combined with --row-offset" in result.stderr


def test_submit_manifest_slurm_adds_explicit_qos(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=1000)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--algorithm",
            "alpha_miner",
            "--partition",
            "major",
            "--qos=major_student",
            "--array",
            "0-999",
            "--dry-run",
        ],
    )

    assert result.returncode == 0
    assert "QOS=major_student" in result.stdout
    assert "--qos=major_student" in result.stdout


def test_submit_manifest_slurm_defaults_all_algorithms_to_cpu_with_student_qos(
    tmp_path: Path,
) -> None:
    for algorithm in ["ilp_miner", "genetic_miner"]:
        manifest_path = _write_manifest(
            tmp_path / f"{algorithm}.csv",
            row_count=1,
            algorithm=algorithm,
        )

        result = _run_submit(
            tmp_path,
            [
                "--manifest",
                manifest_path.as_posix(),
                "--algorithm",
                algorithm,
                "--dry-run",
            ],
        )

        assert result.returncode == 0
        assert "Partition=CPU" in result.stdout
        assert "QOS=minor_student" in result.stdout
        assert "--qos=minor_student" in result.stdout


def test_submit_manifest_slurm_uses_algorithm_qos_env_default(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        row_count=1,
        algorithm="inductive_miner",
    )

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--algorithm",
            "inductive_miner",
            "--dry-run",
        ],
        env_overrides={"INDUCTIVE_MINER_QOS": "minor_student"},
    )

    assert result.returncode == 0
    assert "QOS=minor_student" in result.stdout
    assert "--qos=minor_student" in result.stdout


def test_submit_manifest_slurm_rejects_empty_qos(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=1)

    result = _run_submit(
        tmp_path,
        [
            "--manifest",
            manifest_path.as_posix(),
            "--qos=",
            "--dry-run",
        ],
    )

    assert result.returncode == 1
    assert "--qos requires a non-empty name" in result.stderr


def test_manifest_validation_detects_duplicate_header_rows(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.csv", row_count=1)
    header = manifest_path.read_text(encoding="utf-8").splitlines()[0]
    with manifest_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(f"{header}\n")

    result = validate_manifest_file(manifest_path, project_root=PROJECT_ROOT)

    assert not result.ok
    assert any("duplicate header" in issue.message for issue in result.issues)


def test_manifest_validation_rejects_missing_required_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "log_id"])
        writer.writeheader()
        writer.writerow({"experiment_id": "test", "log_id": "tiny_log"})

    result = validate_manifest_file(manifest_path, project_root=PROJECT_ROOT)

    assert not result.ok
    assert any("missing required column" in issue.message for issue in result.issues)


def _run_submit(
    tmp_path: Path,
    args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["DATA_ROOT"] = (tmp_path / "runtime/data").as_posix()
    env["RESULTS_ROOT"] = (tmp_path / "runtime/results").as_posix()
    env["LOG_ROOT"] = (tmp_path / "runtime/logs").as_posix()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", "scripts/submit_manifest_slurm.sh", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_dynamic_wrapper(
    tmp_path: Path,
    args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["DATA_ROOT"] = (tmp_path / "runtime/data").as_posix()
    env["RESULTS_ROOT"] = (tmp_path / "runtime/results").as_posix()
    env["LOG_ROOT"] = (tmp_path / "runtime/logs").as_posix()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", "slurm/run_dynamic_manifest.slurm", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_submit_metric(
    tmp_path: Path,
    args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["DATA_ROOT"] = (tmp_path / "runtime/data").as_posix()
    env["RESULTS_ROOT"] = (tmp_path / "runtime/results").as_posix()
    env["LOG_ROOT"] = (tmp_path / "runtime/logs").as_posix()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", "scripts/submit_metric_manifest_slurm.sh", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_dynamic_metric_wrapper(
    tmp_path: Path,
    args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["DATA_ROOT"] = (tmp_path / "runtime/data").as_posix()
    env["RESULTS_ROOT"] = (tmp_path / "runtime/results").as_posix()
    env["LOG_ROOT"] = (tmp_path / "runtime/logs").as_posix()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", "slurm/run_dynamic_metric_manifest.slurm", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_manifest(
    path: Path,
    *,
    row_count: int,
    algorithm: str = "alpha_miner",
    log_dir: str = "",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "log_id",
        "log_path",
        "seed",
        "algorithm_id",
        "log_dir",
        "output_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(
                {
                    "experiment_id": "test_01",
                    "log_id": "tiny_log",
                    "log_path": "data/example/tiny_log.xes",
                    "seed": "0",
                    "algorithm_id": algorithm,
                    "log_dir": log_dir,
                    "output_path": f"results/cluster/test_submit/result_{index}.json",
                }
            )
    return path


def _write_metric_manifest(
    path: Path,
    *,
    row_count: int,
    profile: str = "alignment",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_result_path",
        "source_config_hash",
        "experiment_id",
        "log_id",
        "algorithm_name",
        "test_log_path",
        "metric_profile",
        "metric_names_json",
        "log_dir",
        "output_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(row_count):
            writer.writerow(
                {
                    "source_result_path": f"results/cluster/source/result_{index}.json",
                    "source_config_hash": f"hash_{index}",
                    "experiment_id": "test_01",
                    "log_id": "tiny_log",
                    "algorithm_name": "alpha_miner",
                    "test_log_path": "data/example/tiny_log.xes",
                    "metric_profile": profile,
                    "metric_names_json": '["fitness"]',
                    "log_dir": f"logs/slurm/metrics/test_submit/{profile}",
                    "output_path": f"results/metrics/test_submit/result_{index}.json",
                }
            )
    return path


def _write_gedi_targets(path: Path, row_count: int = 3) -> Path:
    columns = [
        "target_id",
        "band",
        "concurrency",
        "noise_level",
        "nearest_real_distance",
        "feasible",
        "infeasible_reason",
        "repairs",
        "target_num_traces",
        "target_avg_trace_length",
        "target_num_activities",
        "target_variant_ratio",
        "target_dfg_density",
        "target_repetition_prevalence",
    ]
    lines = [",".join(columns)]
    for index in range(row_count):
        lines.append(f"t{index:04d},in_distribution,low,0.0,0.5,True,,,100,8.0,10,0.4,0.3,0.5")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path.parent / "anchor_features.csv").write_text(
        "log_id,num_traces,avg_trace_length,num_activities,"
        "variant_ratio,dfg_density,repetition_prevalence\n"
        "real_0,1000,10.0,12,0.4,0.2,0.5\n",
        encoding="utf-8",
    )
    return path


def _run_submit_gedi(
    tmp_path: Path,
    args: list[str],
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["DATA_ROOT"] = (tmp_path / "runtime/data").as_posix()
    env["RESULTS_ROOT"] = (tmp_path / "runtime/results").as_posix()
    env["LOG_ROOT"] = (tmp_path / "runtime/logs").as_posix()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["bash", "scripts/submit_gedi_targets_slurm.sh", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_gedi_target_has_portable_paths_and_array_validation() -> None:
    text = (PROJECT_ROOT / "slurm/run_gedi_target.sh").read_text(encoding="utf-8")

    for forbidden in [
        "/private/" + "cluster/account",
        "/home/" + "example-user",
        "/" + "Users/example-user",
    ]:
        assert forbidden not in text
    assert "#SBATCH --partition=" not in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"' not in text
    assert 'SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_PROJECT_ROOT}}"' in text
    assert "PROJECT_ROOT does not look like the process-mining-cash repository" in text
    assert '[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "SLURM_ARRAY_TASK_ID is not set"' in text
    assert "ACTUAL_ROW=$((GEDI_ROW_OFFSET + SLURM_ARRAY_TASK_ID))" in text
    assert 'echo "GEDI_ROW_OFFSET=${GEDI_ROW_OFFSET}"' in text
    assert 'echo "ACTUAL_GEDI_ROW_INDEX=${ACTUAL_ROW}"' in text
    assert "madeira" in text
    assert "GEDI_PYTHON" in text
    assert "scripts/run_gedi_target.py" in text
    assert '--row-index "${ACTUAL_ROW}"' in text
    assert "SELECTED_ROW_INDEX" in text


def test_submit_gedi_targets_slurm_exports_roots_and_chdir() -> None:
    text = (PROJECT_ROOT / "scripts/submit_gedi_targets_slurm.sh").read_text(encoding="utf-8")

    assert '--chdir "${PROJECT_ROOT}"' in text
    assert (
        '--export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},'
        "RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},"
        "GEDI_PYTHON=${GEDI_PYTHON},GEDI_ROW_OFFSET=${row_offset},"
        "GEDI_BASE_SEED=${BASE_SEED},GEDI_N_TRIALS=${N_TRIALS},"
        "GEDI_MAX_ATTEMPTS=${MAX_ATTEMPTS},"
        'GEDI_TIMEOUT_SECONDS=${TIMEOUT_SECONDS}"'
    ) in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"' not in text
    assert "slurm/run_gedi_target.sh" in text


def test_submit_gedi_targets_slurm_rejects_partition_all(tmp_path: Path) -> None:
    targets = _write_gedi_targets(tmp_path / "targets.csv")
    result = _run_submit_gedi(
        tmp_path, ["--targets", str(targets), "--partition", "all", "--dry-run"]
    )

    assert result.returncode == 1
    assert "partition 'all' is not allowed for student accounts" in result.stderr


def test_submit_gedi_targets_slurm_defaults_and_row_selection(tmp_path: Path) -> None:
    targets = _write_gedi_targets(tmp_path / "targets.csv", row_count=3)

    result = _run_submit_gedi(tmp_path, ["--targets", str(targets), "--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "Partition=CPU" in result.stdout
    assert "QOS=minor_student" in result.stdout
    assert "--qos=minor_student" in result.stdout
    assert "Time=03:00:00" in result.stdout
    assert "Memory=8G" in result.stdout
    assert "--array=0-2" in result.stdout
    assert "slurm/run_gedi_target.sh" in result.stdout
    assert "GEDI_N_TRIALS=50" in result.stdout
    assert "GEDI_MAX_ATTEMPTS=3" in result.stdout

    result = _run_submit_gedi(tmp_path, ["--targets", str(targets), "--row", "1", "--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "--array=0" in result.stdout
    assert "GEDI_ROW_OFFSET=1" in result.stdout


def test_submit_gedi_targets_resolves_portable_data_root(tmp_path: Path) -> None:
    targets_path = tmp_path / "runtime/data/synthetic/gedi/targets.csv"
    targets_path.parent.mkdir(parents=True)
    targets = _write_gedi_targets(targets_path, row_count=1)

    result = _run_submit_gedi(
        tmp_path,
        ["--targets", "data/synthetic/gedi/targets.csv", "--dry-run"],
    )

    assert result.returncode == 0, result.stderr
    assert f"Targets={targets}" in result.stdout
    assert targets.as_posix() in result.stdout


def test_submit_gedi_targets_slurm_requires_anchor_features(tmp_path: Path) -> None:
    targets = _write_gedi_targets(tmp_path / "targets.csv")
    (tmp_path / "anchor_features.csv").unlink()

    result = _run_submit_gedi(tmp_path, ["--targets", str(targets), "--dry-run"])

    assert result.returncode == 1
    assert "anchor_features.csv not found" in result.stderr
