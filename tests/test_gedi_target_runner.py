from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fake_gedi import FakeGediBackend

from process_discovery_cash.cli.run_gedi_target import build_parser, run_target_row
from process_discovery_cash.generation.aggregate import aggregate_results
from process_discovery_cash.generation.anchor import write_anchor_features
from process_discovery_cash.generation.feature_space import BAND_IN_DISTRIBUTION
from process_discovery_cash.generation.pipeline import run_single_target
from process_discovery_cash.generation.results import (
    load_target_result,
    result_path_for,
)
from process_discovery_cash.generation.targets import (
    TargetSpec,
    targets_from_frame,
    targets_to_frame,
)

pytest.importorskip("pm4py")


def _real_features_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for index in range(21):
        rows.append(
            {
                "log_id": f"real_{index}",
                "num_traces": int(rng.integers(800, 150_000)),
                "avg_trace_length": float(rng.uniform(3, 60)),
                "num_activities": int(rng.integers(4, 400)),
                "variant_ratio": float(rng.uniform(0.01, 0.9)),
                "dfg_density": float(rng.uniform(0.02, 0.5)),
                "repetition_prevalence": float(rng.uniform(0.0, 0.9)),
            }
        )
    return pd.DataFrame(rows)


def _target(target_id: str = "t0000", **overrides) -> TargetSpec:
    values = {
        "num_traces": 60.0,
        "avg_trace_length": 8.0,
        "num_activities": 10.0,
        "variant_ratio": 0.5,
        "dfg_density": 0.3,
        "repetition_prevalence": 0.5,
    }
    values.update(overrides.pop("values", {}))
    defaults = dict(
        target_id=target_id,
        band=BAND_IN_DISTRIBUTION,
        values=values,
        concurrency="low",
        noise_level=0.0,
        nearest_real_distance=0.4,
    )
    defaults.update(overrides)
    return TargetSpec(**defaults)


def _write_batch(tmp_path, targets: list[TargetSpec]):
    output_root = tmp_path / "gedi"
    output_root.mkdir(parents=True, exist_ok=True)
    targets_path = output_root / "targets.csv"
    targets_to_frame(targets).to_csv(targets_path, index=False)
    real = _real_features_frame()
    write_anchor_features(real.to_dict("records"), output_root / "anchor_features.csv")
    return output_root, targets_path


def _run_row(tmp_path, targets_path, row_index: int, backend, **extra_args) -> None:
    parser = build_parser()
    argv = [
        "--targets",
        str(targets_path),
        "--row-index",
        str(row_index),
        "--output-root",
        str(targets_path.parent),
        "--results-dir",
        str(tmp_path / "results"),
        "--n-trials",
        "5",
        "--max-attempts",
        "2",
    ]
    for key, value in extra_args.items():
        argv.append(f"--{key.replace('_', '-')}")
        if value is not None:
            argv.append(str(value))
    args = parser.parse_args(argv)
    run_target_row(args, parser=parser, backend=backend)


def test_targets_round_trip_preserves_specs() -> None:
    targets = [
        _target("t0000"),
        _target("t0001", feasible=False, infeasible_reason="because"),
    ]
    targets[0].repairs = ["variant_ratio: 0.9 -> 0.5 (expressibility)"]

    rebuilt = targets_from_frame(targets_to_frame(targets))

    assert [t.values for t in rebuilt] == [t.values for t in targets]
    assert [t.band for t in rebuilt] == [t.band for t in targets]
    assert rebuilt[0].repairs == targets[0].repairs
    assert rebuilt[1].feasible is False
    assert rebuilt[1].infeasible_reason == "because"


def test_run_single_target_accepts_with_fake_backend(tmp_path) -> None:
    real = _real_features_frame()
    records = run_single_target(
        _target(),
        real,
        FakeGediBackend(),
        output_root=tmp_path / "out",
        base_seed=99,
        workdir=tmp_path / "work",
        max_attempts=2,
    )

    assert records[-1].status in {"accepted", "rejected"}
    accepted = [record for record in records if record.status == "accepted"]
    if accepted:
        record = accepted[0]
        assert (tmp_path / "out" / "logs" / f"{record.log_id}.xes.gz").exists()
        # Achieved features are recomputed with the canonical 48-feature extractor.
        assert record.achieved_features.get("rs4pd_total_traces") == record.n_traces


def test_cli_writes_result_json_and_skips_on_rerun(tmp_path) -> None:
    output_root, targets_path = _write_batch(tmp_path, [_target()])
    backend = FakeGediBackend()

    _run_row(tmp_path, targets_path, 0, backend)
    result_path = result_path_for(tmp_path / "results", "t0000")
    payload = load_target_result(result_path)

    assert payload is not None
    assert payload["target_id"] == "t0000"
    assert payload["row_index"] == 0
    assert payload["terminal_status"] in {"accepted", "rejected"}
    assert payload["records"]
    calls_before = backend.calls

    _run_row(tmp_path, targets_path, 0, backend)  # second run must skip
    assert backend.calls == calls_before

    _run_row(tmp_path, targets_path, 0, backend, overwrite=None)
    assert backend.calls > calls_before


def test_cli_seed_determinism_across_runs(tmp_path) -> None:
    output_root, targets_path = _write_batch(tmp_path, [_target()])

    _run_row(tmp_path, targets_path, 0, FakeGediBackend())
    first = load_target_result(result_path_for(tmp_path / "results", "t0000"))
    _run_row(tmp_path, targets_path, 0, FakeGediBackend(), overwrite=None)
    second = load_target_result(result_path_for(tmp_path / "results", "t0000"))

    assert [r["seed"] for r in first["records"]] == [r["seed"] for r in second["records"]]


def test_cli_records_infeasible_target_without_backend_call(tmp_path) -> None:
    target = _target("t0000", feasible=False, infeasible_reason="impossible corner")
    output_root, targets_path = _write_batch(tmp_path, [target])
    backend = FakeGediBackend()

    _run_row(tmp_path, targets_path, 0, backend)
    payload = load_target_result(result_path_for(tmp_path / "results", "t0000"))

    assert payload["terminal_status"] == "target_infeasible"
    assert backend.calls == 0


def test_cli_rejects_out_of_range_row_index(tmp_path) -> None:
    output_root, targets_path = _write_batch(tmp_path, [_target()])

    with pytest.raises(SystemExit):
        _run_row(tmp_path, targets_path, 5, FakeGediBackend())


def test_aggregate_dedups_near_duplicates_deterministically(tmp_path) -> None:
    real = _real_features_frame()
    output_root = tmp_path / "gedi"
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    achieved = {
        "num_traces": 60.0,
        "avg_trace_length": 8.0,
        "num_activities": 10.0,
        "variant_ratio": 0.5,
        "dfg_density": 0.3,
        "repetition_prevalence": 0.5,
    }
    for index, target_id in enumerate(["t0000", "t0001"]):
        log_path = logs_dir / f"syn_gedi_{target_id}.xes.gz"
        log_path.write_bytes(b"placeholder")
        payload = {
            "schema_version": 1,
            "target_id": target_id,
            "row_index": index,
            "base_seed": 2024,
            "n_trials": 5,
            "max_attempts": 2,
            "terminal_status": "accepted",
            "records": [
                {
                    "target_id": target_id,
                    "log_id": f"syn_gedi_{target_id}",
                    "attempt": 1,
                    "seed": 1,
                    "status": "accepted",
                    "band_intended": BAND_IN_DISTRIBUTION,
                    "band_achieved": BAND_IN_DISTRIBUTION,
                    "output_path": str(log_path),
                    "target_values": dict(achieved),
                    "achieved_values": dict(achieved),
                }
            ],
        }
        (results_dir / f"{target_id}.json").write_text(json.dumps(payload))

    records, info = aggregate_results(
        results_dir,
        real,
        output_root=output_root,
        known_target_ids={"t0000", "t0001", "t0002"},
    )

    by_target = {record.target_id: record for record in records}
    assert by_target["t0000"].status == "accepted"
    assert by_target["t0001"].status == "rejected"
    assert "near-duplicate" in by_target["t0001"].rejection_reason
    assert (output_root / "rejected" / "syn_gedi_t0001_dedup.xes.gz").exists()
    assert not (logs_dir / "syn_gedi_t0001.xes.gz").exists()
    assert (output_root / "manifest.csv").exists()
    assert info["missing_target_ids"] == ["t0002"]

    manifest = pd.read_csv(output_root / "manifest.csv")
    assert set(manifest["target_id"]) == {"t0000", "t0001"}
