from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from process_discovery_cash.discovery.split import SplitMiner
from process_discovery_cash.generation.gedi_backend import GediBackend
from process_discovery_cash.generation.targets import TargetSpec

pytestmark = pytest.mark.external

SPLIT_MINER_SHA256 = "472c006623d99a6e440aa93a58e29b867cc331cec2b12b3d7fb61fb2a5de8328"


def test_real_gedi_environment_generates_xes(tmp_path: Path) -> None:
    backend = GediBackend(
        python_bin=".venv-gedi/bin/python",
        n_trials=2,
        timeout_seconds=600,
    )
    unavailable = backend.available()
    if unavailable:
        pytest.fail(unavailable)

    target = TargetSpec(
        target_id="external_smoke",
        band="in_distribution",
        values={
            "num_traces": 50.0,
            "avg_trace_length": 6.0,
            "num_activities": 5.0,
            "variant_ratio": 0.2,
            "dfg_density": 0.3,
            "repetition_prevalence": 0.3,
        },
        concurrency="low",
        noise_level=0.0,
        nearest_real_distance=0.5,
    )

    result = backend.generate(target, seed=2024, workdir=tmp_path / "gedi")

    assert result.status == "success", result.error
    assert result.xes_path is not None
    assert result.xes_path.is_file()
    assert result.xes_path.stat().st_size > 0


def test_split_miner_171_with_java8_discovers_tiny_log(tmp_path: Path) -> None:
    java_bin = os.getenv("JAVA8_BIN", "java")
    version = subprocess.run(
        [java_bin, "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    version_text = f"{version.stdout}\n{version.stderr}"
    if version.returncode != 0 or 'version "1.8.' not in version_text:
        pytest.fail("External Split Miner check requires Java 8; set JAVA8_BIN.")

    result = SplitMiner().discover(
        [],
        {
            "jar_path": "data/external/split-miner-1.7.1-all.jar",
            "jar_sha256": SPLIT_MINER_SHA256,
            "java_bin": java_bin,
            "input_log_path": "data/example/tiny_log.xes",
            "output_dir": (tmp_path / "split").as_posix(),
            "keep_output_files": True,
            "timeout_seconds": 120,
            "epsilon": 0.1,
            "eta": 0.2,
            "parallelismFirst": False,
            "removeLoopActivityMarkers": False,
            "replaceIORs": False,
            "diagram": False,
        },
    )

    assert result.status == "success", result.error_message
    assert result.model_path is not None
    assert Path(result.model_path).is_file()
