"""
Bayesian Optimisation over the RF surrogate.

The objective function asks the RF to predict the composite_score for a given
(log_features, algorithm, hyperparams) triple. Optuna maximises this prediction.

Warm-start configs (from similar logs) are enqueued before the main BO loop so
the search starts from promising regions of the space.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from cash.model import ALGORITHMS, predict

optuna.logging.set_verbosity(optuna.logging.WARNING)


# Hyperparameter search bounds per algorithm
SEARCH_SPACE: dict[str, dict] = {
    "inductive_miner": {
        "noise_threshold": ("float", 0.0, 0.5),
    },
    "heuristic_miner": {
        "dependency_threshold":          ("float", 0.0, 1.0),
        "and_threshold":                 ("float", 0.0, 1.0),
        "loop_two_threshold":            ("float", 0.0, 1.0),
        "dfg_pre_cleaning_noise_thresh": ("float", 0.0, 0.5),
        "min_act_count":                 ("int",   1,   10),
        "min_dfg_occurrences":           ("int",   1,   10),
    },
    "genetic_miner": {
        "generations":     ("int",   5,   50),
        "population_size": ("int",   10,  100),
        "mutation_rate":   ("float", 0.0, 0.5),
        "crossover_rate":  ("float", 0.5, 1.0),
        "elitism_rate":    ("float", 0.0, 0.3),
    },
    "alpha_miner":      {},
    "alpha_plus_miner": {},
    "ilp_miner":        {},
}


def _suggest_config(trial: optuna.Trial) -> tuple[str, dict]:
    algorithm = trial.suggest_categorical("algorithm", ALGORITHMS)
    params: dict = {}
    for name, spec in SEARCH_SPACE.get(algorithm, {}).items():
        kind, lo, hi = spec
        if kind == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        else:
            params[name] = trial.suggest_int(name, lo, hi)
    return algorithm, params


def run_bo(
    models: dict,
    le: LabelEncoder,
    log_features: dict,
    n_trials: int = 30,
    warm_start_configs=None,
    weights: dict | None = None,
) -> dict:
    """
    Run Bayesian Optimisation and return the best configuration found.

    models: {measure: Pipeline} returned by model.train().
    weights: per-measure importance for the composite objective (default equal).
    warm_start_configs: list of dicts with keys 'algorithm' and 'hyperparams',
                        typically retrieved from the k nearest training logs.

    Returns a dict with:
        algorithm, hyperparameters, predicted_score, prediction_std, confidence
    """
    known_algos = set(le.classes_)

    def objective(trial: optuna.Trial) -> float:
        algorithm, hyperparams = _suggest_config(trial)
        if algorithm not in known_algos:
            return 0.0
        score, _ = predict(models, le, log_features, algorithm, hyperparams, weights)
        return score

    study = optuna.create_study(direction="maximize")

    # Enqueue warm-start configurations from similar logs
    if warm_start_configs:
        for cfg in warm_start_configs:
            algo = cfg.get("algorithm", "")
            hparams = cfg.get("hyperparams", {})
            params = {"algorithm": algo}
            for name, spec in SEARCH_SPACE.get(algo, {}).items():
                if name in hparams:
                    params[name] = hparams[name]
            try:
                study.enqueue_trial(params)
            except Exception:
                pass

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    best_algorithm = best.params["algorithm"]
    best_hyperparams = {k: v for k, v in best.params.items() if k != "algorithm"}

    score, std = predict(models, le, log_features, best_algorithm, best_hyperparams, weights)

    confidence = "high" if std < 0.05 else "medium" if std < 0.15 else "low"

    return {
        "algorithm": best_algorithm,
        "hyperparameters": best_hyperparams,
        "predicted_score": round(score, 4),
        "prediction_std": round(std, 4),
        "confidence": confidence,
    }
