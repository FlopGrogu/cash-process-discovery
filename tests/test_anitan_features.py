from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from process_discovery_cash.data.features import FEATURE_NAMES, extract_features_from_xes
from process_discovery_cash.generation.anchor import (
    AXIS_MAP,
    STAT_MAP,
    compute_log_feature_row,
)
from process_discovery_cash.generation.feature_space import TARGET_FEATURES


def _write_tiny_xes(path) -> None:
    pytest.importorskip("pm4py")
    import pm4py

    rows = []
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    traces = {
        "c1": ["a", "b", "c"],
        "c2": ["a", "b", "c"],
        "c3": ["a", "c", "b"],
        "c4": ["a", "b", "b"],
    }
    for case_id, activities in traces.items():
        for position, activity in enumerate(activities):
            rows.append(
                {
                    "case:concept:name": case_id,
                    "concept:name": activity,
                    "time:timestamp": base + pd.Timedelta(minutes=position),
                }
            )
    pm4py.write_xes(pd.DataFrame(rows), str(path), case_id_key="case:concept:name")


def test_extractor_counts_match_hand_computed_values(tmp_path) -> None:
    xes_path = tmp_path / "tiny.xes.gz"
    _write_tiny_xes(xes_path)

    row = extract_features_from_xes(str(xes_path))

    assert row["rs4pd_total_traces"] == 4
    assert row["rs4pd_total_events"] == 12
    assert row["rs4pd_distinct_events"] == 3
    assert row["rs4pd_trace_length_avg"] == pytest.approx(3.0)
    # 3 distinct variants over 4 traces.
    assert row["ratio_unique_traces_per_trace"] == pytest.approx(0.75)
    # DF pairs: ab, bc, ac, cb, bb -> 5 of 9 possible ordered pairs.
    assert row["rs4pd_flow_density"] == pytest.approx(5 / 9)


def test_extractor_returns_all_48_features(tmp_path) -> None:
    xes_path = tmp_path / "tiny.xes.gz"
    _write_tiny_xes(xes_path)

    row = extract_features_from_xes(str(xes_path))

    assert len(FEATURE_NAMES) == 48
    assert set(FEATURE_NAMES) <= set(row.keys())


def test_committed_48_feature_golden_vector() -> None:
    golden = json.loads(Path("tests/golden/anitan_tiny_features.json").read_text(encoding="utf-8"))
    actual = extract_features_from_xes("data/example/tiny_log.xes")

    assert len(golden) == len(FEATURE_NAMES) == 48
    assert actual == pytest.approx(golden)


def test_axis_map_covers_all_target_axes_and_maps_into_feature_row(tmp_path) -> None:
    assert set(AXIS_MAP) == set(TARGET_FEATURES)

    xes_path = tmp_path / "tiny.xes.gz"
    _write_tiny_xes(xes_path)
    row = compute_log_feature_row(xes_path, log_id="tiny")

    assert row["log_id"] == "tiny"
    assert row["num_traces"] == 4
    assert row["avg_trace_length"] == pytest.approx(3.0)
    assert row["num_activities"] == 3
    assert row["variant_ratio"] == pytest.approx(0.75)
    for stat in STAT_MAP:
        assert stat in row
    assert row["num_variants"] == 3
    assert row["min_trace_length"] == 3
    assert row["max_trace_length"] == 3
    # The 48 raw features remain available for downstream analysis.
    assert set(FEATURE_NAMES) <= set(row.keys())
