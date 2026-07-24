from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import qmc

from process_discovery_cash.experiments.manifest import (
    _condition_matches,
    _is_range_parameter_spec,
    _validate_range_parameter_spec,
)

_LOG_SCALE_PARAMETER_NAMES = {
    "generations",
    "min_act_count",
    "min_dfg_occurrences",
    "population_size",
}


@dataclass(frozen=True)
class SamplingDimension:
    key: str
    kind: Literal["categorical", "float_range", "integer_range"]
    values: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    scale: Literal["linear", "log"] = "linear"

    @property
    def is_free(self) -> bool:
        if self.kind == "categorical":
            return len(self.values) > 1
        assert self.minimum is not None and self.maximum is not None
        return self.minimum < self.maximum

    @property
    def fixed_value(self) -> Any:
        if self.kind == "categorical":
            return self.values[0]
        assert self.minimum is not None
        if self.kind == "integer_range":
            return int(round(self.minimum))
        return round(self.minimum, 6)

    def sample(self, unit_draw: float) -> Any:
        if self.kind == "categorical":
            arity = len(self.values)
            index = min(int(unit_draw * arity), arity - 1)
            return self.values[index]

        assert self.minimum is not None and self.maximum is not None
        lower = self.minimum
        upper = self.maximum
        if self.scale == "log":
            lower = float(np.log(lower))
            upper = float(np.log(upper))

        sampled = lower + (upper - lower) * unit_draw
        if self.scale == "log":
            sampled = float(np.exp(sampled))

        if self.kind == "integer_range":
            return int(
                min(
                    max(round(sampled), int(round(self.minimum))),
                    int(round(self.maximum)),
                )
            )
        return round(sampled, 6)


def sample_parameter_lhs(
    default_params: dict[str, Any],
    search_space: dict[str, Any] | None,
    conditional_search_space: list[dict[str, Any]] | None,
    *,
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Draw ``n_samples`` hyperparameter configurations via Latin Hypercube sampling.

    Supports both discrete ``values`` dimensions and ranged specs declared with
    ``min``/``max`` for sampling-aware experiment configs. Ranged dimensions use
    linear scale by default, except for a small set of positive effort/count
    parameters that default to log scaling.
    """
    search_space = search_space or {}
    conditional_search_space = conditional_search_space or []

    base_keys = sorted(search_space)
    base_dimensions = [_resolve_sampling_dimension(key, search_space[key]) for key in base_keys]

    if not _has_free_dimensions(
        base_dimensions,
        default_params,
        conditional_search_space,
    ):
        raise ValueError(
            f"Latin Hypercube sampling requested (n_samples={n_samples}) but the "
            "search space has no free dimensions to sample; remove 'sampling' or "
            "free at least one search-space dimension."
        )

    base_combinations = _draw_lhs_combinations(
        default_params,
        base_dimensions,
        n_samples=n_samples,
        rng=np.random.default_rng(seed),
    )

    for branch_index, conditional_spec in enumerate(conditional_search_space):
        condition = conditional_spec.get("when", {})
        params_spec = conditional_spec.get("params") or {}
        cond_keys = sorted(params_spec)
        if not cond_keys:
            continue
        conditional_dimensions = [
            _resolve_sampling_dimension(key, params_spec[key]) for key in cond_keys
        ]

        matching_positions = [
            i for i, params in enumerate(base_combinations) if _condition_matches(params, condition)
        ]
        if not matching_positions:
            continue

        branch_rng = np.random.default_rng([seed, branch_index + 1])
        branch_combinations = _draw_lhs_combinations(
            {},
            conditional_dimensions,
            n_samples=len(matching_positions),
            rng=branch_rng,
        )
        for local_index, global_index in enumerate(matching_positions):
            base_combinations[global_index].update(branch_combinations[local_index])

    return base_combinations


def _draw_lhs_combinations(
    default_params: dict[str, Any],
    dimensions: list[SamplingDimension],
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if not dimensions:
        return [dict(default_params) for _ in range(n_samples)]

    draws = qmc.LatinHypercube(d=len(dimensions), rng=rng).random(n=n_samples)

    combinations = []
    for row in draws:
        params = dict(default_params)
        for dimension, unit_draw in zip(dimensions, row, strict=True):
            params[dimension.key] = dimension.sample(float(unit_draw))
        combinations.append(params)
    return combinations


def _has_free_dimensions(
    base_dimensions: list[SamplingDimension],
    default_params: dict[str, Any],
    conditional_search_space: list[dict[str, Any]],
) -> bool:
    if any(dimension.is_free for dimension in base_dimensions):
        return True
    if not conditional_search_space:
        return False

    candidate_params = dict(default_params)
    for dimension in base_dimensions:
        candidate_params[dimension.key] = dimension.fixed_value

    for conditional_spec in conditional_search_space:
        if not _condition_matches(candidate_params, conditional_spec.get("when", {})):
            continue
        params_spec = conditional_spec.get("params") or {}
        conditional_dimensions = [
            _resolve_sampling_dimension(key, spec) for key, spec in params_spec.items()
        ]
        if any(dimension.is_free for dimension in conditional_dimensions):
            return True
    return False


def _resolve_sampling_dimension(key: str, spec: Any) -> SamplingDimension:
    if _is_range_parameter_spec(spec):
        if not isinstance(spec, dict):
            raise TypeError(f"Range spec for '{key}' must be a dict, got {type(spec).__name__}.")
        _validate_range_parameter_spec(key, spec)
        parameter_type = spec["type"]
        scale = spec.get("scale") or _default_sampling_scale(key, spec)
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        if parameter_type == "integer":
            return SamplingDimension(
                key=key,
                kind="integer_range",
                minimum=minimum,
                maximum=maximum,
                scale=scale,
            )
        return SamplingDimension(
            key=key,
            kind="float_range",
            minimum=minimum,
            maximum=maximum,
            scale=scale,
        )

    values = _resolve_discrete_values(spec)
    return SamplingDimension(
        key=key,
        kind="categorical",
        values=tuple(values),
    )


def _resolve_discrete_values(spec: Any) -> list[Any]:
    if isinstance(spec, dict) and "values" in spec:
        return list(spec["values"])
    if isinstance(spec, list):
        return list(spec)
    return [spec]


def _default_sampling_scale(key: str, spec: dict[str, Any]) -> Literal["linear", "log"]:
    if spec.get("type") == "integer" and key in _LOG_SCALE_PARAMETER_NAMES:
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        if minimum > 0 and maximum > 0:
            return "log"
    return "linear"
