from __future__ import annotations

from process_discovery_cash.discovery.alpha import AlphaMiner
from process_discovery_cash.discovery.base import DiscoveryAlgorithm
from process_discovery_cash.discovery.genetic import GeneticMiner
from process_discovery_cash.discovery.heuristic import HeuristicsMiner
from process_discovery_cash.discovery.ilp import ILPMiner
from process_discovery_cash.discovery.inductive import InductiveMiner
from process_discovery_cash.discovery.split import SplitMiner


class AlphaMinerClassic(AlphaMiner):
    algorithm_name = "alpha_miner_classic"


class AlphaMinerPlus(AlphaMiner):
    algorithm_name = "alpha_miner_plus"


class InductiveMinerIM(InductiveMiner):
    algorithm_name = "inductive_miner_im"


class InductiveMinerIMf(InductiveMiner):
    algorithm_name = "inductive_miner_imf"


class InductiveMinerIMd(InductiveMiner):
    algorithm_name = "inductive_miner_imd"


class HeuristicsMinerPlusPlus(HeuristicsMiner):
    algorithm_name = "heuristics_miner_plusplus"


ALGORITHM_REGISTRY: dict[str, type[DiscoveryAlgorithm]] = {
    "alpha_miner": AlphaMiner,
    "alpha_miner_classic": AlphaMinerClassic,
    "alpha_miner_plus": AlphaMinerPlus,
    "inductive_miner": InductiveMiner,
    "inductive_miner_im": InductiveMinerIM,
    "inductive_miner_imf": InductiveMinerIMf,
    "inductive_miner_imd": InductiveMinerIMd,
    "ilp_miner": ILPMiner,
    "heuristics_miner": HeuristicsMiner,
    "heuristics_miner_plusplus": HeuristicsMinerPlusPlus,
    "heuristic_miner": HeuristicsMiner,
    "heuristic_miner_plusplus": HeuristicsMinerPlusPlus,
    "genetic_miner": GeneticMiner,
    "split_miner": SplitMiner,
}


def get_algorithm(name: str) -> DiscoveryAlgorithm:
    try:
        return ALGORITHM_REGISTRY[name]()
    except KeyError as exc:
        known = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise KeyError(f"Unknown discovery algorithm '{name}'. Known algorithms: {known}") from exc


def registered_algorithm_names() -> list[str]:
    return sorted(ALGORITHM_REGISTRY)
