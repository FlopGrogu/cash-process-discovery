from __future__ import annotations

import pytest

from process_discovery_cash.config.load import load_algorithm_config
from process_discovery_cash.experiments.sampling import sample_parameter_lhs

HEURISTIC_CONFIG = load_algorithm_config("configs/algorithms/heuristic.yaml")
INDUCTIVE_CONFIG = load_algorithm_config("configs/algorithms/inductive.yaml")


def test_sample_parameter_lhs_returns_requested_row_count() -> None:
    rows = sample_parameter_lhs(
        HEURISTIC_CONFIG.default_params,
        HEURISTIC_CONFIG.search_space,
        None,
        n_samples=25,
        seed=0,
    )

    assert len(rows) == 25


def test_sample_parameter_lhs_is_deterministic_for_fixed_seed() -> None:
    kwargs = dict(
        default_params=HEURISTIC_CONFIG.default_params,
        search_space=HEURISTIC_CONFIG.search_space,
        conditional_search_space=None,
        n_samples=15,
        seed=42,
    )

    first = sample_parameter_lhs(**kwargs)
    second = sample_parameter_lhs(**kwargs)

    assert first == second


def test_sample_parameter_lhs_differs_across_seeds() -> None:
    rows_a = sample_parameter_lhs(
        HEURISTIC_CONFIG.default_params,
        HEURISTIC_CONFIG.search_space,
        None,
        n_samples=15,
        seed=1,
    )
    rows_b = sample_parameter_lhs(
        HEURISTIC_CONFIG.default_params,
        HEURISTIC_CONFIG.search_space,
        None,
        n_samples=15,
        seed=2,
    )

    assert rows_a != rows_b


def test_sample_parameter_lhs_only_draws_declared_values() -> None:
    rows = sample_parameter_lhs(
        HEURISTIC_CONFIG.default_params,
        HEURISTIC_CONFIG.search_space,
        None,
        n_samples=30,
        seed=3,
    )

    for key, spec in HEURISTIC_CONFIG.search_space.items():
        declared_values = set(spec["values"])
        assert {row[key] for row in rows} <= declared_values


def test_sample_parameter_lhs_merges_default_params() -> None:
    rows = sample_parameter_lhs(
        INDUCTIVE_CONFIG.default_params,
        INDUCTIVE_CONFIG.search_space,
        None,
        n_samples=5,
        seed=4,
    )

    fixed_keys = set(INDUCTIVE_CONFIG.default_params) - set(INDUCTIVE_CONFIG.search_space)
    assert fixed_keys
    for row in rows:
        for key in fixed_keys:
            assert row[key] == INDUCTIVE_CONFIG.default_params[key]


def test_sample_parameter_lhs_respects_conditional_search_space() -> None:
    rows = sample_parameter_lhs(
        INDUCTIVE_CONFIG.default_params,
        INDUCTIVE_CONFIG.search_space,
        INDUCTIVE_CONFIG.conditional_search_space,
        n_samples=40,
        seed=5,
    )

    assert len(rows) == 40
    assert all("noise_threshold" not in row for row in rows if row["variant"] != "imf")
    imf_rows = [row for row in rows if row["variant"] == "imf"]
    assert imf_rows
    declared_noise_values = set(
        INDUCTIVE_CONFIG.conditional_search_space[0]["params"]["noise_threshold"]["values"]
    )
    assert all(row["noise_threshold"] in declared_noise_values for row in imf_rows)


def test_sample_parameter_lhs_conditional_branch_is_stratified() -> None:
    # Force every base draw into the imf branch by fixing variant to a
    # singleton, so the whole n_samples budget lands in one conditional
    # sub-design and its stratification quality can be checked directly.
    search_space = {
        "variant": {"values": ["imf"]},
        "disable_fallthroughs": {"values": [False]},
        "multi_processing": {"values": [False]},
    }
    rows = sample_parameter_lhs(
        INDUCTIVE_CONFIG.default_params,
        search_space,
        INDUCTIVE_CONFIG.conditional_search_space,
        n_samples=14,
        seed=6,
    )

    declared_noise_values = INDUCTIVE_CONFIG.conditional_search_space[0]["params"][
        "noise_threshold"
    ]["values"]
    observed = {row["noise_threshold"] for row in rows}
    # 14 samples over 7 declared values should spread across most/all of them,
    # not collapse onto one or two values as independent 1-D draws might.
    assert len(observed) >= len(declared_noise_values) - 1


def test_sample_parameter_lhs_allows_degenerate_base_with_applicable_conditional() -> None:
    search_space = {
        "variant": {"values": ["imf"]},
        "disable_fallthroughs": {"values": [False]},
        "multi_processing": {"values": [False]},
    }

    rows = sample_parameter_lhs(
        INDUCTIVE_CONFIG.default_params,
        search_space,
        INDUCTIVE_CONFIG.conditional_search_space,
        n_samples=9,
        seed=7,
    )

    assert len(rows) == 9
    assert all(row["variant"] == "imf" for row in rows)
    assert len({row["noise_threshold"] for row in rows}) > 1


def test_sample_parameter_lhs_raises_on_zero_free_dimensions() -> None:
    search_space = {
        "variant": {"values": ["im"]},
        "disable_fallthroughs": {"values": [False]},
        "multi_processing": {"values": [False]},
    }

    with pytest.raises(ValueError, match="no free dimensions"):
        sample_parameter_lhs(
            INDUCTIVE_CONFIG.default_params,
            search_space,
            None,
            n_samples=5,
            seed=8,
        )


def test_sample_parameter_lhs_supports_float_ranges() -> None:
    rows = sample_parameter_lhs(
        default_params={"alpha": 1.0},
        search_space={"alpha": {"min": 0.05, "max": 1.0, "type": "float"}},
        conditional_search_space=None,
        n_samples=20,
        seed=9,
    )

    values = [row["alpha"] for row in rows]
    assert len(rows) == 20
    assert all(0.05 <= value <= 1.0 for value in values)
    assert len(set(values)) > 10


def test_sample_parameter_lhs_supports_integer_ranges() -> None:
    rows = sample_parameter_lhs(
        default_params={"population_size": 500},
        search_space={
            "population_size": {
                "min": 100,
                "max": 1000,
                "type": "integer",
            }
        },
        conditional_search_space=None,
        n_samples=20,
        seed=10,
    )

    values = [row["population_size"] for row in rows]
    assert len(rows) == 20
    assert all(100 <= value <= 1000 for value in values)
    assert all(isinstance(value, int) for value in values)
    assert len(set(values)) > 10


def test_sample_parameter_lhs_defaults_positive_effort_ranges_to_log_scale() -> None:
    rows = sample_parameter_lhs(
        default_params={"population_size": 500},
        search_space={
            "population_size": {
                "min": 100,
                "max": 1000,
                "type": "integer",
            }
        },
        conditional_search_space=None,
        n_samples=5,
        seed=11,
    )

    linear_rows = sample_parameter_lhs(
        default_params={"population_size": 500},
        search_space={
            "population_size": {
                "min": 100,
                "max": 1000,
                "type": "integer",
                "scale": "linear",
            }
        },
        conditional_search_space=None,
        n_samples=5,
        seed=11,
    )

    assert rows != linear_rows


def test_sample_parameter_lhs_supports_ranged_conditional_dimension() -> None:
    rows = sample_parameter_lhs(
        default_params=INDUCTIVE_CONFIG.default_params,
        search_space={
            "variant": {"values": ["imf"]},
            "disable_fallthroughs": {"values": [False]},
            "multi_processing": {"values": [False]},
        },
        conditional_search_space=[
            {
                "when": {"variant": "imf"},
                "params": {
                    "noise_threshold": {"min": 0.0, "max": 0.6, "type": "float"},
                },
            }
        ],
        n_samples=12,
        seed=12,
    )

    values = [row["noise_threshold"] for row in rows]
    assert all(0.0 <= value <= 0.6 for value in values)
    assert len(set(values)) > 6
