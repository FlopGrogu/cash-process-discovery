"""Translate algorithm search spaces into Optuna trial suggestions.

Reuses the sampling-dimension parsing shared with Latin Hypercube sampling
(``experiments/sampling.py``): discrete ``{type, values}`` specs become
categorical suggestions, ``{min, max, type, scale}`` range specs become
``suggest_float``/``suggest_int`` (with ``log=True`` for log-scaled ranges).
Conditional branches (``when: {param: value}``) are only suggested when the
drawn parent parameters match, which maps onto TPE's tree-structured space
(use ``group=True`` in the sampler).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from process_discovery_cash.experiments.manifest import _condition_matches
from process_discovery_cash.experiments.sampling import (
    SamplingDimension,
    _resolve_sampling_dimension,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import optuna

_CATEGORICAL_VALUE_TYPES = (type(None), bool, int, float, str)


@dataclass(frozen=True)
class HpoSearchSpace:
    base: tuple[SamplingDimension, ...] = ()
    conditional: tuple[tuple[dict[str, Any], tuple[SamplingDimension, ...]], ...] = field(
        default_factory=tuple
    )


def build_hpo_search_space(
    search_space: dict[str, Any] | None,
    conditional_search_space: list[dict[str, Any]] | None,
) -> HpoSearchSpace:
    search_space = search_space or {}
    conditional_search_space = conditional_search_space or []

    base = tuple(
        _validated_dimension(_resolve_sampling_dimension(key, search_space[key]))
        for key in sorted(search_space)
    )
    conditional = []
    for conditional_spec in conditional_search_space:
        params_spec = conditional_spec.get("params") or {}
        dimensions = tuple(
            _validated_dimension(_resolve_sampling_dimension(key, params_spec[key]))
            for key in sorted(params_spec)
        )
        if dimensions:
            conditional.append((dict(conditional_spec.get("when", {})), dimensions))

    space = HpoSearchSpace(base=base, conditional=tuple(conditional))
    if not _has_free_dimension(space):
        raise ValueError(
            "HPO requested but the search space has no free dimensions to optimize; "
            "free at least one search-space dimension or run a plain manifest instead."
        )
    return space


def suggest_trial_params(
    trial: optuna.Trial,
    default_params: dict[str, Any],
    space: HpoSearchSpace,
) -> dict[str, Any]:
    """Draw one configuration; defaults first, suggested values win."""
    params = dict(default_params)
    for dimension in space.base:
        params[dimension.key] = _suggest_dimension(trial, dimension)
    for condition, dimensions in space.conditional:
        if not _condition_matches(params, condition):
            continue
        for dimension in dimensions:
            params[dimension.key] = _suggest_dimension(trial, dimension)
    return params


def _suggest_dimension(trial: optuna.Trial, dimension: SamplingDimension) -> Any:
    if not dimension.is_free:
        return dimension.fixed_value
    if dimension.kind == "categorical":
        return trial.suggest_categorical(dimension.key, list(dimension.values))
    assert dimension.minimum is not None and dimension.maximum is not None
    log = dimension.scale == "log"
    if dimension.kind == "integer_range":
        return trial.suggest_int(
            dimension.key,
            int(round(dimension.minimum)),
            int(round(dimension.maximum)),
            log=log,
        )
    return trial.suggest_float(dimension.key, dimension.minimum, dimension.maximum, log=log)


def _validated_dimension(dimension: SamplingDimension) -> SamplingDimension:
    if dimension.kind == "categorical":
        invalid = [
            value for value in dimension.values if not isinstance(value, _CATEGORICAL_VALUE_TYPES)
        ]
        if invalid:
            raise ValueError(
                f"Search-space parameter '{dimension.key}' has categorical values "
                f"unsupported by Optuna (need None/bool/int/float/str): {invalid!r}"
            )
    elif dimension.scale == "log":
        assert dimension.minimum is not None
        if dimension.minimum <= 0:
            raise ValueError(
                f"Search-space parameter '{dimension.key}' uses log scale but its "
                f"minimum ({dimension.minimum}) is not positive."
            )
    return dimension


def _has_free_dimension(space: HpoSearchSpace) -> bool:
    if any(dimension.is_free for dimension in space.base):
        return True
    return any(dimension.is_free for _, dimensions in space.conditional for dimension in dimensions)
