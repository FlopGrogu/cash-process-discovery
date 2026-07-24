from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_discovery_cash.data.augmentation import (
    AugmentationSpec,
    augment_parent_log,
    canonicalize_event_log,
    child_seed,
    compute_log_stats,
    compute_variants,
    default_augmentation_plan,
    filter_top_activities,
    filter_variant_coverage,
    inject_event_noise,
    subsample_variants,
    truncate_traces,
    validate_child_log,
)
from process_discovery_cash.data.loading import (
    ACTIVITY_COLUMN,
    CASE_ID_COLUMN,
    TIMESTAMP_COLUMN,
)

REQUIRED = [CASE_ID_COLUMN, ACTIVITY_COLUMN, TIMESTAMP_COLUMN]


def _make_log(case_traces: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    for case_id, activities in case_traces.items():
        for position, activity in enumerate(activities):
            rows.append(
                {
                    CASE_ID_COLUMN: case_id,
                    ACTIVITY_COLUMN: activity,
                    TIMESTAMP_COLUMN: base + pd.Timedelta(minutes=position),
                }
            )
    return pd.DataFrame(rows)


def _variant_mix_log() -> pd.DataFrame:
    traces: dict[str, list[str]] = {}
    for index in range(8):
        traces[f"ab_{index}"] = ["a", "b"]
    for index in range(4):
        traces[f"abc_{index}"] = ["a", "b", "c"]
    for index in range(4):
        traces[f"ac_{index}"] = ["a", "c"]
    return _make_log(traces)


def test_compute_variants_returns_ordered_activity_tuples() -> None:
    log = canonicalize_event_log(_make_log({"c1": ["a", "b", "c"], "c2": ["a", "c"]}))

    variants = compute_variants(log)

    assert variants["c1"] == ("a", "b", "c")
    assert variants["c2"] == ("a", "c")


def test_subsample_variants_keeps_complete_cases_and_variant_mix() -> None:
    log = canonicalize_event_log(_variant_mix_log())
    rng = np.random.default_rng(7)

    child = subsample_variants(log, 0.5, rng)

    child_variants = compute_variants(child)
    assert child_variants.size == 8
    counts = child_variants.value_counts()
    assert counts[("a", "b")] == 4
    assert counts[("a", "b", "c")] == 2
    assert counts[("a", "c")] == 2
    # Complete cases: every sampled case keeps its full trace.
    parent_lengths = log.groupby(CASE_ID_COLUMN)[ACTIVITY_COLUMN].size()
    child_lengths = child.groupby(CASE_ID_COLUMN)[ACTIVITY_COLUMN].size()
    for case_id, length in child_lengths.items():
        assert length == parent_lengths[case_id]


def test_subsample_variants_is_deterministic_for_fixed_seed() -> None:
    log = canonicalize_event_log(_variant_mix_log())

    first = subsample_variants(log, 0.5, np.random.default_rng(3))
    second = subsample_variants(log, 0.5, np.random.default_rng(3))

    assert sorted(first[CASE_ID_COLUMN].unique()) == sorted(second[CASE_ID_COLUMN].unique())


def test_filter_variant_coverage_keeps_most_frequent_variants() -> None:
    log = canonicalize_event_log(_variant_mix_log())

    child = filter_variant_coverage(log, 0.5)

    assert set(compute_variants(child).unique()) == {("a", "b")}


def test_inject_event_noise_preserves_columns_and_cases() -> None:
    log = canonicalize_event_log(
        _make_log({f"c{index}": ["a", "b", "c", "d"] for index in range(10)})
    )

    child = inject_event_noise(log, 0.2, np.random.default_rng(11))

    assert list(child.columns) == REQUIRED
    assert set(child[CASE_ID_COLUMN].unique()) == set(log[CASE_ID_COLUMN].unique())
    assert not child[REQUIRED].isna().any().any()
    # Timestamps stay sorted within each case.
    for _, group in child.groupby(CASE_ID_COLUMN):
        assert group[TIMESTAMP_COLUMN].is_monotonic_increasing


def test_inject_event_noise_zero_probability_is_identity() -> None:
    log = canonicalize_event_log(_make_log({"c1": ["a", "b"], "c2": ["a", "c"]}))

    child = inject_event_noise(log, 0.0, np.random.default_rng(1))

    pd.testing.assert_frame_equal(child, log)


def test_truncate_traces_limits_trace_length() -> None:
    log = canonicalize_event_log(_make_log({"long": list("abcdefgh"), "short": ["a", "b"]}))

    child = truncate_traces(log, 3)

    lengths = child.groupby(CASE_ID_COLUMN)[ACTIVITY_COLUMN].size()
    assert lengths["long"] == 3
    assert lengths["short"] == 2
    assert compute_variants(child)["long"] == ("a", "b", "c")


def test_filter_top_activities_drops_rare_activities_and_empty_cases() -> None:
    traces = {f"main_{index}": ["a", "b", "a", "b"] for index in range(5)}
    traces["rare"] = ["z"]
    log = canonicalize_event_log(_make_log(traces))

    child = filter_top_activities(log, 0.8)

    assert set(child[ACTIVITY_COLUMN].unique()) == {"a", "b"}
    assert "rare" not in set(child[CASE_ID_COLUMN].unique())


def test_validate_child_log_rejects_too_small_logs() -> None:
    too_few_traces = canonicalize_event_log(
        _make_log({"c1": ["a", "b", "c"], "c2": ["a", "c", "b"]})
    )
    assert "too few traces" in validate_child_log(too_few_traces)

    two_activities = canonicalize_event_log(
        _make_log({f"c{index}": ["a", "b"] if index % 2 else ["b", "a"] for index in range(12)})
    )
    assert "too few activities" in validate_child_log(two_activities)

    one_variant = canonicalize_event_log(
        _make_log({f"c{index}": ["a", "b", "c"] for index in range(12)})
    )
    assert "too few variants" in validate_child_log(one_variant)


def test_validate_child_log_accepts_valid_log() -> None:
    traces = {f"c{index}": ["a", "b", "c"] if index % 2 else ["a", "c", "b"] for index in range(12)}
    log = canonicalize_event_log(_make_log(traces))

    assert validate_child_log(log) is None


def test_validate_child_log_rejects_missing_columns() -> None:
    log = pd.DataFrame({CASE_ID_COLUMN: ["c1"], ACTIVITY_COLUMN: ["a"]})

    assert "missing required columns" in validate_child_log(log)


def test_default_plan_has_three_children_plus_triggers() -> None:
    small = {"n_traces": 100, "n_events": 500, "mean_trace_length": 5.0}
    assert len(default_augmentation_plan(small)) == 3

    large_long = {"n_traces": 50_000, "n_events": 5_000_000, "mean_trace_length": 100.0}
    plan = default_augmentation_plan(large_long)
    operators = [spec.operator for spec in plan]
    assert operators.count("subsample") == 2
    assert "truncate" in operators

    stress = default_augmentation_plan(small, include_stress=True)
    assert sum(spec.stress for spec in stress) == 3


def test_child_seed_is_deterministic_and_spec_dependent() -> None:
    spec_a = AugmentationSpec("subsample", {"fraction": 0.5})
    spec_b = AugmentationSpec("subsample", {"fraction": 0.25})

    assert child_seed(1001, "sepsis", spec_a) == child_seed(1001, "sepsis", spec_a)
    assert child_seed(1001, "sepsis", spec_a) != child_seed(1001, "sepsis", spec_b)
    assert child_seed(1001, "sepsis", spec_a) != child_seed(1002, "sepsis", spec_a)


def test_augment_parent_log_roundtrip_and_manifest_fields(tmp_path) -> None:
    pytest.importorskip("pm4py")
    traces = {
        f"c{index}": ["a", "b", "c", "d"] if index % 2 else ["a", "c", "b", "d"]
        for index in range(40)
    }
    log = _make_log(traces)

    records = augment_parent_log(
        log,
        "tiny_parent",
        [AugmentationSpec("subsample", {"fraction": 0.5})],
        output_dir=tmp_path,
        base_seed=1001,
        parent_path="data/example/tiny_log.xes",
    )

    assert len(records) == 1
    record = records[0]
    assert record.status == "accepted"
    assert record.parent_log_id == "tiny_parent"
    assert record.child_log_id.startswith("aug_tiny_parent__sub050__s")

    import pm4py

    reloaded = pm4py.read_xes(record.output_path, return_legacy_log_object=False)
    reloaded = canonicalize_event_log(reloaded)
    stats = compute_log_stats(reloaded)
    assert stats["n_traces"] == record.n_traces == 20
    assert stats["n_events"] == record.n_events
    assert stats["n_activities"] == record.n_activities
    assert stats["n_variants"] == record.n_variants


def test_augment_parent_log_skips_existing_output_without_overwrite(tmp_path) -> None:
    pytest.importorskip("pm4py")
    traces = {f"c{index}": ["a", "b", "c"] if index % 2 else ["a", "c", "b"] for index in range(40)}
    log = _make_log(traces)
    spec = AugmentationSpec("subsample", {"fraction": 0.5})

    first = augment_parent_log(log, "tiny_parent", [spec], output_dir=tmp_path, base_seed=1001)
    second = augment_parent_log(log, "tiny_parent", [spec], output_dir=tmp_path, base_seed=1001)

    assert first[0].status == "accepted"
    assert second[0].status == "skipped_existing"


def test_augment_parent_log_rejects_too_small_children(tmp_path) -> None:
    log = _make_log({"c1": ["a", "b", "c"], "c2": ["a", "c", "b"]})

    records = augment_parent_log(
        log,
        "tiny_parent",
        [AugmentationSpec("subsample", {"fraction": 0.5})],
        output_dir=tmp_path,
        base_seed=1001,
    )

    assert records[0].status == "rejected"
    assert "too few traces" in records[0].rejection_reason
    assert records[0].output_path is None
    assert list(tmp_path.glob("*.xes.gz")) == []
