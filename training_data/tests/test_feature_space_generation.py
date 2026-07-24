from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fake_gedi import FakeGediBackend

from process_discovery_cash.generation.feature_space import (
    BAND_EXPANSION,
    BAND_IN_DISTRIBUTION,
    BAND_STRESS,
    TARGET_FEATURES,
    assign_band,
    design_bounds,
    from_target_space,
    pairwise_binned_coverage,
    to_target_space,
)
from process_discovery_cash.generation.gedi_backend import GediBackend
from process_discovery_cash.generation.pipeline import (
    CandidateRecord,
    _rejection_reason,
    run_generation,
    summarize_records,
)
from process_discovery_cash.generation.targets import (
    TargetSpec,
    _parse_bool,
    design_targets,
    repair_target,
    targets_to_frame,
)


@pytest.mark.parametrize("value", ["truthy", "2", 2, -1, object()])
def test_target_feasibility_rejects_malformed_booleans(value) -> None:
    with pytest.raises(ValueError, match="Invalid boolean value"):
        _parse_bool(value)


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


def test_to_target_space_roundtrip() -> None:
    raw = {
        "num_traces": 1000.0,
        "avg_trace_length": 10.0,
        "num_activities": 30.0,
        "variant_ratio": 0.4,
        "dfg_density": 0.2,
        "repetition_prevalence": 0.6,
    }
    point = to_target_space(raw)[0]

    assert point[0] == pytest.approx(3.0)  # log10(1000)
    assert point[3] == pytest.approx(0.4)
    back = from_target_space(point)
    for feature in TARGET_FEATURES:
        assert back[feature] == pytest.approx(raw[feature], rel=1e-6)


def test_band_assignment_thresholds() -> None:
    assert assign_band(0.5) == BAND_IN_DISTRIBUTION
    assert assign_band(2.0) == BAND_EXPANSION
    assert assign_band(4.0) == BAND_STRESS


def test_design_targets_deterministic_and_banded() -> None:
    real = _real_features_frame()

    first = design_targets(real, n_targets=40, seed=7)
    second = design_targets(real, n_targets=40, seed=7)

    assert len(first) == 40
    assert [t.values for t in first] == [t.values for t in second]
    assert [t.band for t in first] == [t.band for t in second]
    frame = targets_to_frame(first)
    bands = frame["band"].value_counts().to_dict()
    assert bands.get(BAND_IN_DISTRIBUTION, 0) >= bands.get(BAND_EXPANSION, 0)
    # All designed values respect the caps after repair.
    assert (frame["target_num_traces"] >= 10).all()
    assert (frame["target_num_activities"] >= 3).all()
    assert (frame["target_avg_trace_length"] >= 2).all()


def test_repair_target_fixes_impossible_variant_ratio() -> None:
    spec = TargetSpec(
        target_id="t0000",
        band=BAND_IN_DISTRIBUTION,
        values={
            "num_traces": 5000.0,
            "avg_trace_length": 2.0,
            "num_activities": 3.0,
            "variant_ratio": 0.95,
            "dfg_density": 0.3,
            "repetition_prevalence": 0.1,
        },
        concurrency="low",
        noise_level=0.0,
        nearest_real_distance=0.5,
    )

    repair_target(
        spec,
        caps_low={"num_traces": 10, "avg_trace_length": 2, "num_activities": 3},
        caps_high={"num_traces": 10_000, "avg_trace_length": 100, "num_activities": 200},
    )

    # 3 activities x length 2 can express at most 9 variants; 0.95*5000 is absurd.
    assert spec.values["variant_ratio"] < 0.01 or not spec.feasible


def test_repair_target_caps_expected_events() -> None:
    spec = TargetSpec(
        target_id="t0001",
        band=BAND_STRESS,
        values={
            "num_traces": 10_000.0,
            "avg_trace_length": 100.0,
            "num_activities": 50.0,
            "variant_ratio": 0.2,
            "dfg_density": 0.2,
            "repetition_prevalence": 0.5,
        },
        concurrency="medium",
        noise_level=0.0,
        nearest_real_distance=3.5,
    )

    repair_target(
        spec,
        caps_low={"num_traces": 10, "avg_trace_length": 2, "num_activities": 3},
        caps_high={"num_traces": 10_000, "avg_trace_length": 100, "num_activities": 200},
        max_expected_events=200_000,
    )

    assert spec.values["num_traces"] * spec.values["avg_trace_length"] <= 200_000 * 1.01
    assert any("events cap" in repair for repair in spec.repairs)


def test_gedi_backend_spec_maps_features_and_concurrency(tmp_path) -> None:
    backend = GediBackend(python_bin="/nonexistent/python", n_trials=30)
    spec = TargetSpec(
        target_id="t0002",
        band=BAND_IN_DISTRIBUTION,
        values={
            "num_traces": 500.0,
            "avg_trace_length": 12.0,
            "num_activities": 20.0,
            "variant_ratio": 0.3,
            "dfg_density": 0.2,
            "repetition_prevalence": 0.7,
        },
        concurrency="high",
        noise_level=0.05,
        nearest_real_distance=0.4,
    )

    payload = backend.build_spec(spec, seed=11, workdir=tmp_path)

    assert payload["objectives"]["n_traces"] == 500
    assert payload["objectives"]["trace_len_mean"] == pytest.approx(12.0)
    assert payload["objectives"]["n_unique_activities"] == 20
    assert payload["objectives"]["ratio_variants_per_number_of_traces"] == pytest.approx(0.3)
    assert payload["config_space"]["parallel"] == [0.3, 0.8]
    assert payload["config_space"]["duplicate"] == [0.0, 0.3]  # repetition >= 0.5
    assert payload["n_trials"] == 30
    assert payload["seed"] == 11


def test_gedi_backend_never_scales_down_the_configured_trial_count(tmp_path) -> None:
    backend = GediBackend(python_bin="/nonexistent/python", n_trials=50)
    expensive = TargetSpec(
        target_id="t_expensive",
        band=BAND_STRESS,
        values={
            "num_traces": 10_000.0,
            "avg_trace_length": 20.0,
            "num_activities": 50.0,
            "variant_ratio": 0.2,
            "dfg_density": 0.2,
            "repetition_prevalence": 0.5,
        },
        concurrency="medium",
        noise_level=0.0,
        nearest_real_distance=3.0,
    )

    assert backend.build_spec(expensive, seed=12, workdir=tmp_path)["n_trials"] == 50


def test_pairwise_binned_coverage_improves_with_new_points() -> None:
    real = _real_features_frame()
    bounds = design_bounds(real)
    real_points = to_target_space(real)

    base = pairwise_binned_coverage(real_points, bounds)
    extra = np.vstack([real_points, to_target_space(_far_corner_features())])
    extended = pairwise_binned_coverage(extra, bounds)

    assert 0 < base["coverage"] <= 1
    assert extended["occupied_cells"] >= base["occupied_cells"]


def _far_corner_features() -> dict[str, float]:
    return {
        "num_traces": 12.0,
        "avg_trace_length": 2.1,
        "num_activities": 3.0,
        "variant_ratio": 0.99,
        "dfg_density": 0.99,
        "repetition_prevalence": 0.99,
    }


def test_rejection_reason_rejects_small_and_duplicate_logs() -> None:
    record = CandidateRecord(
        target_id="t0003",
        log_id="syn_gedi_t0003",
        attempt=1,
        seed=1,
        status="",
        n_traces=5,
        n_activities=8,
        n_variants=4,
        n_events=40,
    )
    features = {"min_trace_length": 2, "max_trace_length": 10}
    assert "too few traces" in _rejection_reason(record, features, 200_000)

    record.n_traces = 100
    record.attainment_error = 0.5
    record.axis_errors = {f: 0.0 for f in TARGET_FEATURES}
    record.duplicate_distance = 0.01
    assert "near-duplicate" in _rejection_reason(record, features, 200_000)

    record.duplicate_distance = 0.8
    assert _rejection_reason(record, features, 200_000) is None


def test_run_generation_with_fake_backend(tmp_path) -> None:
    pytest.importorskip("pm4py")
    real = _real_features_frame()
    targets = [
        TargetSpec(
            target_id="t0000",
            band=BAND_IN_DISTRIBUTION,
            values={
                "num_traces": 60.0,
                "avg_trace_length": 8.0,
                "num_activities": 10.0,
                "variant_ratio": 0.5,
                "dfg_density": 0.3,
                "repetition_prevalence": 0.5,
            },
            concurrency="low",
            noise_level=0.0,
            nearest_real_distance=0.4,
        ),
        TargetSpec(
            target_id="t0001",
            band=BAND_IN_DISTRIBUTION,
            values={
                "num_traces": 80.0,
                "avg_trace_length": 6.0,
                "num_activities": 8.0,
                "variant_ratio": 0.4,
                "dfg_density": 0.3,
                "repetition_prevalence": 0.4,
            },
            concurrency="low",
            noise_level=0.0,
            nearest_real_distance=0.4,
            feasible=False,
            infeasible_reason="marked infeasible for test",
        ),
        TargetSpec(
            target_id="t0002",
            band=BAND_IN_DISTRIBUTION,
            values={
                "num_traces": 70.0,
                "avg_trace_length": 7.0,
                "num_activities": 9.0,
                "variant_ratio": 0.4,
                "dfg_density": 0.3,
                "repetition_prevalence": 0.4,
            },
            concurrency="low",
            noise_level=0.0,
            nearest_real_distance=0.4,
        ),
    ]
    backend = FakeGediBackend(fail_target_ids={"t0002"})

    records = run_generation(
        targets,
        real,
        backend,  # type: ignore[arg-type]
        output_root=tmp_path / "out",
        base_seed=99,
        workdir=tmp_path / "work",
        max_attempts=2,
    )

    by_target = {}
    for record in records:
        by_target.setdefault(record.target_id, []).append(record)
    assert by_target["t0000"][-1].status in {"accepted", "rejected"}
    assert by_target["t0001"][0].status == "target_infeasible"
    assert [r.status for r in by_target["t0002"]] == ["generation_failed", "generation_failed"]

    manifest = pd.read_csv(tmp_path / "out" / "manifest.csv")
    assert set(manifest["target_id"]) == {"t0000", "t0001", "t0002"}
    accepted = manifest[manifest["status"] == "accepted"]
    for _, row in accepted.iterrows():
        assert row["achieved_num_traces"] > 0  # recomputed, never copied
        assert (tmp_path / "out" / "logs" / f"{row['log_id']}.xes.gz").exists()

    summary = summarize_records(records)
    assert summary["by_status"]["generation_failed"] == 2
