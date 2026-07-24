"""Deterministic stand-in for the GEDI backend used across generation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FakeGediBackend:
    """Writes a small synthetic XES shaped after the target instead of running GEDI."""

    def __init__(self, fail_target_ids: set[str] | None = None) -> None:
        self.fail_target_ids = fail_target_ids or set()
        self.calls = 0

    def available(self) -> str | None:
        return None

    def generate(self, target, seed, workdir):
        from process_discovery_cash.generation.gedi_backend import GediResult

        self.calls += 1
        if target.target_id in self.fail_target_ids:
            return GediResult(
                status="failed",
                xes_path=None,
                error="synthetic failure",
                gedi_achieved=None,
                incumbent=None,
                spec={"objectives": {}},
            )
        import pm4py

        rng = np.random.default_rng(seed)
        rows = []
        base = pd.Timestamp("2024-01-01T00:00:00Z")
        n_traces = int(round(target.values["num_traces"]))
        n_acts = int(round(target.values["num_activities"]))
        length = max(2, int(round(target.values["avg_trace_length"])))
        for case in range(n_traces):
            sequence = ["a0", *[f"a{int(rng.integers(1, n_acts))}" for _ in range(length - 1)]]
            for position, activity in enumerate(sequence):
                rows.append(
                    {
                        "case:concept:name": f"c{case}",
                        "concept:name": activity,
                        "time:timestamp": base + pd.Timedelta(minutes=position),
                    }
                )
        frame = pd.DataFrame(rows)
        workdir = workdir / "x"
        workdir.mkdir(parents=True, exist_ok=True)
        xes_path = workdir / "generated.xes"
        pm4py.write_xes(frame, str(xes_path), case_id_key="case:concept:name")
        return GediResult(
            status="success",
            xes_path=xes_path,
            error=None,
            gedi_achieved={},
            incumbent={"mode": n_acts},
            spec={"objectives": {"n_traces": n_traces}},
        )
