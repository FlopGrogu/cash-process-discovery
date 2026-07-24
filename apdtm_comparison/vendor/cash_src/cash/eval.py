"""
Local refinement: run real process discovery experiments around the BO best config.

After the RF surrogate BO narrows the search space, this module runs a small number
of actual algorithm executions to find the best config with real measured scores.
"""

from __future__ import annotations

import time

import numpy as np

from cash.bo import SEARCH_SPACE
from cash.model import TARGET, WEIGHTS


def generate_variants(algorithm: str, hyperparams: dict, n_variants: int, seed: int = 42) -> list:
    """Generate n_variants perturbed configs around the best BO config."""
    rng = np.random.default_rng(seed)
    configs = [{"algorithm": algorithm, "hyperparams": dict(hyperparams)}]

    space = SEARCH_SPACE.get(algorithm, {})
    if not space:
        return configs

    for _ in range(n_variants - 1):
        params = {}
        for name, spec in space.items():
            kind, lo, hi = spec
            center = hyperparams.get(name, (lo + hi) / 2)
            if kind == "float":
                val = float(np.clip(rng.normal(center, (hi - lo) * 0.15), lo, hi))
            else:
                val = int(np.clip(round(rng.normal(center, max(1, (hi - lo) * 0.15))), lo, hi))
            params[name] = val
        configs.append({"algorithm": algorithm, "hyperparams": params})

    return configs


def run_algorithm(df, algorithm: str, hyperparams: dict):
    """Run a pm4py discovery algorithm on a DataFrame log. Returns (net, im, fm)."""
    import pm4py

    if algorithm == "inductive_miner":
        return pm4py.discover_petri_net_inductive(df, noise_threshold=hyperparams.get("noise_threshold", 0.0))
    elif algorithm == "heuristic_miner":
        return pm4py.discover_petri_net_heuristics(
            df,
            dependency_threshold=hyperparams.get("dependency_threshold", 0.5),
            and_threshold=hyperparams.get("and_threshold", 0.65),
            loop_two_threshold=hyperparams.get("loop_two_threshold", 0.5),
        )
    elif algorithm == "alpha_miner":
        return pm4py.discover_petri_net_alpha(df)
    elif algorithm == "alpha_plus_miner":
        return pm4py.discover_petri_net_alpha_plus(df)
    elif algorithm == "ilp_miner":
        return pm4py.discover_petri_net_ilp(df)
    elif algorithm == "genetic_miner":
        return pm4py.discover_petri_net_genetic(
            df,
            generations=hyperparams.get("generations", 10),
            population_size=hyperparams.get("population_size", 50),
            mutation_rate=hyperparams.get("mutation_rate", 0.3),
            crossover_rate=hyperparams.get("crossover_rate", 0.8),
            elitism_rate=hyperparams.get("elitism_rate", 0.2),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def compute_metrics(df, net, im, fm) -> dict:
    """Compute the 4 quality metrics and composite_score."""
    import pm4py

    metrics = {}
    try:
        res = pm4py.fitness_token_based_replay(df, net, im, fm)
        metrics["fitness"] = res.get("log_fitness")
    except Exception:
        metrics["fitness"] = None

    try:
        metrics["precision"] = pm4py.precision_token_based_replay(df, net, im, fm)
    except Exception:
        metrics["precision"] = None

    try:
        metrics["simplicity"] = pm4py.simplicity_petri_net(net, im, fm)
    except Exception:
        metrics["simplicity"] = None

    try:
        metrics["generalization"] = pm4py.generalization_tbr(df, net, im, fm)
    except Exception:
        metrics["generalization"] = None

    if all(v is not None for v in metrics.values()):
        metrics[TARGET] = sum(metrics[k] * w for k, w in WEIGHTS.items())
    else:
        metrics[TARGET] = None

    return metrics


def run_local_refinement(df, algorithm: str, hyperparams: dict, n_trials: int = 10) -> dict:
    """
    Run n_trials real experiments around the best BO config.
    Returns the config with the highest real composite_score.
    """
    configs = generate_variants(algorithm, hyperparams, n_trials)
    best_result = None

    for i, cfg in enumerate(configs, 1):
        algo = cfg["algorithm"]
        params = cfg["hyperparams"]
        t0 = time.time()
        try:
            net, im, fm = run_algorithm(df, algo, params)
            metrics = compute_metrics(df, net, im, fm)
            elapsed = time.time() - t0
            score = metrics.get(TARGET)
            fmt_params = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in params.items()
            )
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"  Trial {i:2d}: {fmt_params} → score={score_str} ({elapsed:.1f}s)")
            if score is not None and (best_result is None or score > best_result["real_score"]):
                best_result = {
                    "algorithm": algo,
                    "hyperparameters": params,
                    "real_score": round(score, 4),
                    "metrics": {
                        k: round(v, 4) if v is not None else None
                        for k, v in metrics.items()
                        if k != TARGET
                    },
                }
        except Exception as e:
            print(f"  Trial {i:2d}: ERROR: {e}")

    return best_result
