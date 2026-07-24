from process_discovery_cash.discovery.registry import get_algorithm, registered_algorithm_names


def test_algorithms_and_v6_variant_aliases_are_registered() -> None:
    assert registered_algorithm_names() == [
        "alpha_miner",
        "alpha_miner_classic",
        "alpha_miner_plus",
        "genetic_miner",
        "heuristic_miner",
        "heuristic_miner_plusplus",
        "heuristics_miner",
        "heuristics_miner_plusplus",
        "ilp_miner",
        "inductive_miner",
        "inductive_miner_im",
        "inductive_miner_imd",
        "inductive_miner_imf",
        "split_miner",
    ]


def test_registry_returns_algorithm_instances() -> None:
    algorithm = get_algorithm("alpha_miner")
    assert algorithm.algorithm_name == "alpha_miner"


def test_registry_returns_v6_alias_instances_with_alias_names() -> None:
    assert get_algorithm("alpha_miner_plus").algorithm_name == "alpha_miner_plus"
    assert get_algorithm("heuristics_miner_plusplus").algorithm_name == "heuristics_miner_plusplus"
    assert get_algorithm("heuristic_miner_plusplus").algorithm_name == "heuristics_miner_plusplus"
    assert get_algorithm("inductive_miner_imf").algorithm_name == "inductive_miner_imf"
