from __future__ import annotations

import inspect

import pytest

from process_discovery_cash.config.load import load_algorithm_config
from process_discovery_cash.discovery import pm4py_backend
from process_discovery_cash.discovery.pm4py_backend import (
    UnsupportedAlgorithmError,
    _resolve_variant,
    discover_alpha_miner,
    discover_genetic_miner,
    discover_heuristics_miner,
    discover_ilp_miner,
    discover_inductive_miner,
)


@pytest.mark.parametrize(
    ("algorithm_label", "module_path", "requested", "candidates", "expected"),
    [
        (
            "Alpha Miner",
            "pm4py.algo.discovery.alpha.algorithm",
            "classic",
            {"classic": ["ALPHA_VERSION_CLASSIC"], "plus": ["ALPHA_VERSION_PLUS"]},
            "ALPHA_VERSION_CLASSIC",
        ),
        (
            "Alpha Miner",
            "pm4py.algo.discovery.alpha.algorithm",
            "plus",
            {"classic": ["ALPHA_VERSION_CLASSIC"], "plus": ["ALPHA_VERSION_PLUS"]},
            "ALPHA_VERSION_PLUS",
        ),
        (
            "Inductive Miner",
            "pm4py.algo.discovery.inductive.algorithm",
            "im",
            {"im": ["IM"], "imf": ["IMf"], "imd": ["IMd"]},
            "IM",
        ),
        (
            "Inductive Miner",
            "pm4py.algo.discovery.inductive.algorithm",
            "imf",
            {"im": ["IM"], "imf": ["IMf"], "imd": ["IMd"]},
            "IMf",
        ),
        (
            "Inductive Miner",
            "pm4py.algo.discovery.inductive.algorithm",
            "imd",
            {"im": ["IM"], "imf": ["IMf"], "imd": ["IMd"]},
            "IMd",
        ),
        (
            "Heuristics Miner",
            "pm4py.algo.discovery.heuristics.algorithm",
            "classic",
            {"classic": ["CLASSIC"], "plusplus": ["PLUSPLUS"]},
            "CLASSIC",
        ),
        (
            "Heuristics Miner",
            "pm4py.algo.discovery.heuristics.algorithm",
            "plusplus",
            {"classic": ["CLASSIC"], "plusplus": ["PLUSPLUS"]},
            "PLUSPLUS",
        ),
    ],
)
def test_yaml_variants_map_to_installed_pm4py_enums(
    algorithm_label: str,
    module_path: str,
    requested: str,
    candidates: dict[str, list[str]],
    expected: str,
) -> None:
    module = __import__(module_path, fromlist=["algorithm"])

    resolved_name, resolved_variant = _resolve_variant(
        algorithm_label,
        module.Variants,
        requested,
        next(iter(candidates)),
        candidates,
    )

    assert resolved_name == expected
    assert resolved_variant == getattr(module.Variants, expected)


def test_public_pm4py_functions_and_genetic_signature_are_installed() -> None:
    import pm4py.discovery as discovery

    for function_name in [
        "discover_petri_net_alpha",
        "discover_petri_net_alpha_plus",
        "discover_petri_net_ilp",
        "discover_petri_net_genetic",
        "discover_petri_net_inductive",
        "discover_petri_net_heuristics",
        "discover_heuristics_net",
    ]:
        assert hasattr(discovery, function_name)

    genetic_signature = inspect.signature(discovery.discover_petri_net_genetic)
    assert {
        "population_size",
        "elitism_rate",
        "crossover_rate",
        "mutation_rate",
        "generations",
        "elitism_min_sample",
        "log_csv",
    } <= set(genetic_signature.parameters)
    assert "deprecated" in (discovery.discover_petri_net_alpha_plus.__doc__ or "").lower()


def test_configured_variants_are_present_in_installed_pm4py() -> None:
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner

    inductive_config = load_algorithm_config("configs/algorithms/inductive.yaml")
    heuristic_config = load_algorithm_config("configs/algorithms/heuristic.yaml")

    assert set(inductive_config.search_space["variant"]["values"]) == {"im", "imf", "imd"}
    assert inductive_config.default_params["recursion_limit"] == 10000
    assert inductive_config.default_params["discovery_timeout_seconds"] == 86400
    assert inductive_config.runtime_params == ["discovery_timeout_seconds", "recursion_limit"]
    assert hasattr(inductive_miner.Variants, "IM")
    assert hasattr(inductive_miner.Variants, "IMf")
    assert hasattr(inductive_miner.Variants, "IMd")
    assert set(heuristic_config.search_space["variant"]["values"]) == {
        "classic",
        "plusplus",
    }
    assert heuristic_config.default_params["recursion_limit"] == 10000
    assert heuristic_config.default_params["discovery_timeout_seconds"] == 86400
    assert heuristic_config.runtime_params == ["discovery_timeout_seconds", "recursion_limit"]
    assert hasattr(heuristics_miner.Variants, "CLASSIC")
    assert hasattr(heuristics_miner.Variants, "PLUSPLUS")


@pytest.mark.parametrize(
    ("wrapper", "params", "expected_backend_function"),
    [
        (
            discover_alpha_miner,
            {"variant": "classic"},
            "pm4py.algo.discovery.alpha.algorithm.apply",
        ),
        (
            discover_alpha_miner,
            {"variant": "plus"},
            "pm4py.algo.discovery.alpha.algorithm.apply",
        ),
        (
            discover_inductive_miner,
            {"variant": "imd"},
            "pm4py.algo.discovery.inductive.algorithm.apply",
        ),
        (
            discover_ilp_miner,
            {"alpha": 1.0},
            "pm4py.algo.discovery.ilp.algorithm.apply",
        ),
        (
            discover_heuristics_miner,
            {"variant": "plusplus"},
            "pm4py.algo.discovery.heuristics.algorithm.apply",
        ),
        (
            discover_genetic_miner,
            {"population_size": 100},
            "pm4py.algo.discovery.genetic.algorithm.apply",
        ),
    ],
)
def test_backend_wrappers_call_expected_pm4py_api(
    monkeypatch,
    wrapper,
    params,
    expected_backend_function,
) -> None:
    _capture_public_discovery_function(monkeypatch)
    _capture_lower_level_apply(monkeypatch)

    payload = wrapper([], params)

    assert payload["metadata"]["backend_function"] == expected_backend_function
    assert payload["model"] == ("net", "im", "fm")


@pytest.mark.parametrize(
    ("wrapper", "params", "message"),
    [
        (discover_alpha_miner, {"variant": "unknown"}, "Alpha Miner variant"),
        (discover_inductive_miner, {"variant": "unknown"}, "Inductive Miner variant"),
        (discover_ilp_miner, {"variant": "unknown"}, "ILP Miner variant"),
        (discover_heuristics_miner, {"variant": "unknown"}, "Heuristics Miner variant"),
        (discover_genetic_miner, {"variant": "unknown"}, "Genetic Miner variant"),
    ],
)
def test_backend_wrappers_reject_unsupported_variants(
    monkeypatch,
    wrapper,
    params,
    message,
) -> None:
    _capture_public_discovery_function(monkeypatch)
    _capture_lower_level_apply(monkeypatch)

    with pytest.raises(UnsupportedAlgorithmError, match=message):
        wrapper([], params)


def test_alpha_plus_metadata_marks_optional_deprecated_variant(monkeypatch) -> None:
    _capture_public_discovery_function(monkeypatch)
    _capture_lower_level_apply(monkeypatch)

    payload = discover_alpha_miner([], {"variant": "plus"})

    assert payload["metadata"]["requested_variant"] == "plus"
    assert payload["metadata"]["resolved_variant"] == "ALPHA_VERSION_PLUS"
    assert any("deprecated" in warning for warning in payload["metadata"]["warnings"])


def _capture_public_discovery_function(monkeypatch) -> None:
    def fake_discovery_function(_event_log, **_kwargs):
        return ("net", "im", "fm")

    monkeypatch.setattr(
        pm4py_backend,
        "_pm4py_discovery_function",
        lambda _function_name, _algorithm: fake_discovery_function,
    )
    monkeypatch.setattr(pm4py_backend, "ensure_pm4py_event_log", lambda event_log: event_log)


def _capture_lower_level_apply(monkeypatch) -> None:
    def fake_process_tree_to_petri_net(_process_tree):
        return ("net", "im", "fm")

    monkeypatch.setattr(
        pm4py_backend,
        "_convert_process_tree_to_petri_net",
        fake_process_tree_to_petri_net,
    )

    for module_path in [
        "pm4py.algo.discovery.alpha.algorithm",
        "pm4py.algo.discovery.inductive.algorithm",
        "pm4py.algo.discovery.ilp.algorithm",
        "pm4py.algo.discovery.heuristics.algorithm",
        "pm4py.algo.discovery.genetic.algorithm",
    ]:
        module = __import__(module_path, fromlist=["algorithm"])
        monkeypatch.setattr(module, "apply", lambda *_args, **_kwargs: ("net", "im", "fm"))
