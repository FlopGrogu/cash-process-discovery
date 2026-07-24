"""Subprocess wrapper around GEDI running in its dedicated sidecar venv."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from process_discovery_cash.generation.targets import TargetSpec
from process_discovery_cash.utils.paths import project_root

# Repo feature name -> feeed/GEDI objective name. Only these four axes can be
# optimized directly by GEDI; dfg_density and repetition_prevalence are steered
# indirectly (loops/operator probabilities) and measured after generation.
GEDI_FEATURE_MAP = {
    "num_traces": "n_traces",
    "avg_trace_length": "trace_len_mean",
    "num_activities": "n_unique_activities",
    "variant_ratio": "ratio_variants_per_number_of_traces",
}

CONCURRENCY_PARALLEL_BOUNDS = {
    "low": (0.01, 0.15),
    "medium": (0.1, 0.4),
    "high": (0.3, 0.8),
}

DEFAULT_GEDI_PYTHON = ".venv-gedi/bin/python"
GEDI_PYTHON_ENV_VAR = "GEDI_PYTHON"


@dataclass(frozen=True)
class GediResult:
    status: str  # "success" | "failed"
    xes_path: Path | None
    error: str | None
    gedi_achieved: dict[str, Any] | None
    incumbent: dict[str, Any] | None
    spec: dict[str, Any]


class GediBackend:
    """Runs one GEDI generation per call in the sidecar interpreter."""

    def __init__(
        self,
        *,
        python_bin: str | Path | None = None,
        n_trials: int = 30,
        timeout_seconds: int = 1800,
    ) -> None:
        resolved = python_bin or os.getenv(GEDI_PYTHON_ENV_VAR) or DEFAULT_GEDI_PYTHON
        self.python_bin = Path(resolved)
        if not self.python_bin.is_absolute():
            self.python_bin = project_root() / self.python_bin
        self.runner_script = Path(__file__).with_name("_gedi_runner.py")
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds

    def available(self) -> str | None:
        """Return None if the backend is usable, else a description of why not."""
        if not self.python_bin.exists():
            return (
                f"GEDI interpreter not found at {self.python_bin}. Create the sidecar venv "
                f"(see docs/feature_space_generation.md) or set ${GEDI_PYTHON_ENV_VAR}."
            )
        probe = subprocess.run(
            [str(self.python_bin), "-c", "import gedi, smac"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if probe.returncode != 0:
            return f"GEDI import failed in {self.python_bin}: {probe.stderr.strip()[-500:]}"
        return None

    def generate(self, target: TargetSpec, seed: int, workdir: Path) -> GediResult:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        spec = self.build_spec(target, seed=seed, workdir=workdir)
        spec_path = workdir / "spec.json"
        result_path = workdir / "result.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        try:
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONHASHSEED": str(seed),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            process = subprocess.run(
                [str(self.python_bin), str(self.runner_script), str(spec_path), str(result_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return GediResult(
                status="failed",
                xes_path=None,
                error=f"GEDI generation timed out after {self.timeout_seconds}s",
                gedi_achieved=None,
                incumbent=None,
                spec=spec,
            )
        if not result_path.exists():
            tail = (process.stderr or process.stdout or "").strip()[-800:]
            return GediResult(
                status="failed",
                xes_path=None,
                error=f"GEDI runner produced no result file (exit {process.returncode}): {tail}",
                gedi_achieved=None,
                incumbent=None,
                spec=spec,
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        xes_path = Path(payload["xes_path"]) if payload.get("xes_path") else None
        if payload.get("status") == "success" and (xes_path is None or not xes_path.exists()):
            payload["status"] = "failed"
            payload["error"] = f"GEDI reported success but XES is missing: {xes_path}"
        return GediResult(
            status=payload.get("status", "failed"),
            xes_path=xes_path,
            error=payload.get("error"),
            gedi_achieved=payload.get("gedi_achieved"),
            incumbent=payload.get("incumbent"),
            spec=spec,
        )

    def build_spec(self, target: TargetSpec, *, seed: int, workdir: Path) -> dict[str, Any]:
        values = target.values
        objectives = {
            GEDI_FEATURE_MAP["num_traces"]: int(round(values["num_traces"])),
            GEDI_FEATURE_MAP["avg_trace_length"]: float(values["avg_trace_length"]),
            GEDI_FEATURE_MAP["num_activities"]: int(round(values["num_activities"])),
            GEDI_FEATURE_MAP["variant_ratio"]: float(values["variant_ratio"]),
        }
        n_traces = int(round(values["num_traces"]))
        n_activities = int(round(values["num_activities"]))
        parallel_low, parallel_high = CONCURRENCY_PARALLEL_BOUNDS[target.concurrency]
        config_space: dict[str, Any] = {
            "mode": [
                max(3, math.floor(0.85 * n_activities)),
                max(4, math.ceil(1.15 * n_activities)),
            ],
            "sequence": [0.01, 1.0],
            "choice": [0.01, 1.0],
            "parallel": [parallel_low, parallel_high],
            "loop": [0.01, 1.0],
            # Tiny alphabets lose visible activities to silent transitions.
            "silent": [0.01, 0.3] if n_activities <= 5 else [0.01, 1.0],
            "lt_dependency": [0.01, 1.0],
            "num_traces": [max(10, math.floor(0.9 * n_traces)), math.ceil(1.1 * n_traces) + 1],
            "or": [0],
            # Duplicate labels only when strong repetition is requested.
            "duplicate": [0.0, 0.3] if values["repetition_prevalence"] >= 0.5 else [0],
        }
        return {
            "objectives": objectives,
            "config_space": config_space,
            "n_trials": self.n_trials,
            "seed": int(seed),
            "workdir": str(workdir),
            "target_id": target.target_id,
        }
