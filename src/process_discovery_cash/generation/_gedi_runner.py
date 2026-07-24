"""Standalone GEDI runner executed with the GEDI sidecar interpreter.

This script must not import process_discovery_cash: it runs inside the
dedicated GEDI virtualenv (see docs/feature_space_generation.md), which only
has gedi and its pinned dependencies installed. It reads a JSON spec, runs
GEDI's SMAC optimization for one target, writes the best generated log as
XES, and always writes a JSON result (also on failure).

GEDI's own ``GenerateEventLogs`` entry point wraps SMAC in a multiprocessing
pool with ``n_workers=-1``, which deadlocks on macOS and ignores seeds
(``RANDOM_SEED`` is hardcoded). We therefore drive GEDI's own methods
(``gen_log``/``eval_log``/``generate_optimized_log``) with a single-worker,
explicitly seeded SMAC scenario.
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def main() -> None:
    # Absolute paths only: _generate() chdirs into the workdir, so any relative
    # path would silently resolve to a nested location afterwards.
    spec_path = os.path.abspath(sys.argv[1])
    result_path = os.path.abspath(sys.argv[2])
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    spec["workdir"] = os.path.abspath(spec["workdir"])
    result = {
        "status": "failed",
        "error": None,
        "xes_path": None,
        "gedi_achieved": None,
        "incumbent": None,
        "n_incumbents": None,
    }
    try:
        result.update(_generate(spec))
    except Exception:  # noqa: BLE001 - report every failure through the result file
        result["error"] = traceback.format_exc(limit=25)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)


def _generate(spec: dict) -> dict:
    workdir = spec["workdir"]
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    import random

    import gedi.generator as gedi_generator
    import numpy as np

    seed = int(spec["seed"])
    gedi_generator.RANDOM_SEED = seed
    random.seed(seed)
    np.random.seed(seed)

    from ConfigSpace import ConfigurationSpace
    from gedi.generator import GenerateEventLogs
    from pm4py import write_xes
    from smac import HyperparameterOptimizationFacade, Scenario

    instance = object.__new__(GenerateEventLogs)
    instance.objectives = dict(spec["objectives"])

    space: dict = {}
    for key, value in spec["config_space"].items():
        if isinstance(value, list):
            space[key] = tuple(value) if len(value) > 1 else value[0]
        else:
            space[key] = value
    config_space = ConfigurationSpace(space)
    config_space.seed(seed)

    # GEDI's eval_log returns raw absolute errors, so SMAC would optimize
    # whichever objective has the largest scale (n_traces) and ignore the
    # ratio-valued ones. Normalize every error by the target magnitude
    # (floored so near-zero targets stay finite).
    floors = {
        "n_traces": 10.0,
        "trace_len_mean": 2.0,
        "n_unique_activities": 3.0,
        "ratio_variants_per_number_of_traces": 0.05,
    }

    def normalized_errors(errors: dict) -> dict:
        return {
            name: float(error) / max(abs(float(instance.objectives[name])), floors.get(name, 1.0))
            for name, error in errors.items()
        }

    def scaled_gen_log(config, seed: int = 0):
        return normalized_errors(instance.gen_log(config, seed))

    scenario = Scenario(
        config_space,
        deterministic=True,
        n_trials=int(spec["n_trials"]),
        objectives=[*instance.objectives.keys()],
        n_workers=1,
        seed=seed,
        output_directory=os.path.join(workdir, "smac_output"),
    )
    multi_objective = HyperparameterOptimizationFacade.get_multi_objective_algorithm(
        scenario, objective_weights=[1] * len(instance.objectives)
    )
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=scaled_gen_log,
        multi_objective_algorithm=multi_objective,
        overwrite=True,
    )
    incumbents = smac.optimize()
    if not isinstance(incumbents, list):
        incumbents = [incumbents]
    if not incumbents:
        raise RuntimeError("SMAC returned no incumbent configuration")

    best_config, best_cost = None, None
    for config in incumbents:
        cost = sum(scaled_gen_log(config, seed=seed).values())
        if best_cost is None or cost < best_cost:
            best_config, best_cost = config, cost

    generated = instance.generate_optimized_log(best_config)
    xes_path = os.path.join(workdir, "generated.xes")
    write_xes(generated["log"], xes_path)

    return {
        "status": "success",
        "xes_path": xes_path,
        "gedi_achieved": {k: _plain(v) for k, v in generated["metafeatures"].items()},
        "incumbent": {k: _plain(v) for k, v in dict(best_config).items()},
        "n_incumbents": len(incumbents),
    }


def _plain(value):
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
