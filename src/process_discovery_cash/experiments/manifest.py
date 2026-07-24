from __future__ import annotations

import csv
import json
import logging
import warnings
from dataclasses import dataclass
from itertools import product
from numbers import Real
from pathlib import Path
from typing import Any

from process_discovery_cash.config.load import (
    load_algorithm_config,
    load_experiment_config,
)
from process_discovery_cash.config.schema import AlgorithmReference
from process_discovery_cash.experiments.discovery_timeout import (
    DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    DISCOVERY_TIMEOUT_FIELD,
)
from process_discovery_cash.experiments.identity import semantic_run_identity
from process_discovery_cash.utils.hashing import stable_hash
from process_discovery_cash.utils.paths import portable_project_path, project_root

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedAlgorithmConfig:
    algorithm_id: str
    backend: str
    model_type: str
    default_params: dict[str, Any]
    search_space: dict[str, Any]
    conditional_search_space: list[dict[str, Any]]
    runtime_params: list[str]
    supported: bool
    artifact_algorithm_id: str | None = None
    inline_definition: bool = False


MANIFEST_COLUMNS = [
    "experiment_id",
    "log_id",
    "dataset_id",
    "log_path",
    "test_log_path",
    "source_log_path",
    "discovery_log_path",
    "test_discovery_log_path",
    "artifact_kind",
    "artifact_sha256",
    "preprocessing_fingerprint",
    "preprocessing_metadata_path",
    "seed",
    "algorithm_id",
    "algorithm_variant",
    "algorithm",
    "backend",
    "algorithm_params_json",
    "params_json",
    "metrics_json",
    "config_id",
    "config_hash",
    "log_dir",
    "output_path",
]


def generate_manifest_rows(
    experiment_config_path: str | Path,
    *,
    require_artifacts: bool = False,
) -> list[dict[str, str]]:
    if require_artifacts:
        warnings.warn(
            "--require-artifacts is deprecated and ignored; manifests are XES-first "
            "and optimized caches are resolved locally at runtime.",
            DeprecationWarning,
            stacklevel=2,
        )
    experiment_config_path = Path(experiment_config_path)
    root = project_root()
    experiment = load_experiment_config(experiment_config_path)
    if experiment.hpo is not None:
        raise ValueError(
            f"Experiment '{experiment.experiment_id}' is an HPO experiment. "
            "Generate its study manifest with pdcash-generate-hpo-studies; export "
            "completed trials with pdcash-export-hpo-manifest."
        )
    algorithm_configs = _load_algorithm_configs(experiment.algorithm_config_dir)
    rows: list[dict[str, str]] = []

    seed = _experiment_seed(experiment)
    metrics_config = _experiment_metrics_config(experiment)

    for log_ref, algorithm_ref in product(
        experiment.logs,
        experiment.algorithms,
    ):
        normalized_algorithm_ref = _normalize_algorithm_ref(algorithm_ref)
        _algorithm_config_path, algorithm_config = _resolve_algorithm_config(
            normalized_algorithm_ref,
            algorithm_configs,
            experiment,
        )
        _validate_algorithm_support(normalized_algorithm_ref, algorithm_config, experiment)
        for params in _algorithm_parameter_sets(
            normalized_algorithm_ref,
            algorithm_config,
            strict=experiment.strict_parameter_validation,
        ):
            algorithm_variant = _algorithm_variant(algorithm_config, params)
            source_log_path = log_ref.source_path or log_ref.path
            config_id = stable_hash(
                semantic_run_identity(
                    log_id=log_ref.log_id,
                    log_path=log_ref.path,
                    test_log_path=log_ref.test_path,
                    seed=seed,
                    algorithm_id=algorithm_config.algorithm_id,
                    backend=algorithm_config.backend,
                    params=params,
                    metrics=metrics_config,
                    runtime_fields=algorithm_config.runtime_params,
                )
            )
            row_index = len(rows)
            output_path = _result_output_path(
                experiment,
                log_id=log_ref.log_id,
                seed=seed,
                algorithm_id=algorithm_config.algorithm_id,
                config_hash=config_id,
                row_index=row_index,
            )
            params_json = json.dumps(params, sort_keys=True)
            metrics_json = json.dumps(metrics_config, sort_keys=True)
            rows.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "log_id": log_ref.log_id,
                    "dataset_id": log_ref.dataset_id or "",
                    "log_path": _portable_manifest_path(log_ref.path, root),
                    "test_log_path": _portable_manifest_path(log_ref.test_path, root),
                    "source_log_path": _portable_manifest_path(source_log_path, root),
                    "discovery_log_path": "",
                    "test_discovery_log_path": "",
                    "artifact_kind": "",
                    "artifact_sha256": "",
                    "preprocessing_fingerprint": "",
                    "preprocessing_metadata_path": "",
                    "seed": str(seed),
                    "algorithm_id": algorithm_config.algorithm_id,
                    "algorithm_variant": algorithm_variant,
                    "algorithm": algorithm_config.algorithm_id,
                    "backend": algorithm_config.backend,
                    "algorithm_params_json": params_json,
                    "params_json": params_json,
                    "metrics_json": metrics_json,
                    "config_id": config_id,
                    "config_hash": config_id,
                    "log_dir": _portable_manifest_path(
                        experiment.output.log_dir,
                        root,
                    ),
                    "output_path": _portable_manifest_path(output_path, root),
                }
            )

    return rows


def _experiment_metrics_config(experiment: Any) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "profile": "pm4py_default",
        "names": ["fitness", "precision", "generalization", "simplicity"],
        "export_model": False,
    }
    metrics = getattr(experiment, "metrics", None)
    if metrics is None:
        return defaults
    if hasattr(metrics, "model_dump"):
        configured = metrics.model_dump()
    elif isinstance(metrics, dict):
        configured = dict(metrics)
    else:
        configured = {key: getattr(metrics, key) for key in defaults if hasattr(metrics, key)}
    merged = dict(defaults)
    merged.update({key: value for key, value in configured.items() if value is not None})
    return merged


def _algorithm_variant(algorithm_config: Any, params: dict[str, Any]) -> str:
    if "variant" in params:
        return str(params.get("variant", ""))
    if algorithm_config.algorithm_id == "split_miner":
        return "v1"
    variant_enum = getattr(algorithm_config, "pm4py_api", {}).get("variant_enum")
    if isinstance(variant_enum, dict) and len(variant_enum) == 1:
        return next(iter(variant_enum))
    return ""


def _experiment_seed(experiment: Any) -> int:
    seeds = list(getattr(experiment, "seeds", None) or [0])
    if len(seeds) > 1:
        LOGGER.warning(
            "Experiment '%s' defines multiple seeds (%s); full train/test experiments "
            "execute each log/configuration once, using the first seed %s.",
            experiment.experiment_id,
            seeds,
            seeds[0],
        )
    return int(seeds[0])


def _experiment_output_dir(experiment: Any) -> Path:
    results_dir = getattr(getattr(experiment, "output", None), "results_dir", None)
    if results_dir:
        return Path(results_dir)
    return Path(experiment.output_root) / experiment.experiment_id


def _result_output_path(
    experiment: Any,
    *,
    log_id: str,
    seed: int,
    algorithm_id: str,
    config_hash: str,
    row_index: int,
) -> Path:
    output = getattr(experiment, "output", None)
    output_path_template = getattr(output, "output_path_template", None)
    if output_path_template:
        results_dir = _experiment_output_dir(experiment)
        rendered = output_path_template.format(
            experiment_id=experiment.experiment_id,
            results_dir=results_dir.as_posix(),
            log_id=log_id,
            seed=seed,
            algorithm_id=algorithm_id,
            config_hash=config_hash,
            row_index=row_index,
        )
        return Path(rendered)
    return (
        _experiment_output_dir(experiment) / f"{row_index:06d}_{log_id}_{seed}_"
        f"{algorithm_id}_{config_hash}.json"
    )


def _portable_manifest_path(value: str | Path, project_root: Path) -> str:
    return portable_project_path(value)


def _algorithm_parameter_sets(
    algorithm_ref: AlgorithmReference,
    algorithm_config: Any,
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    shared_params = _algorithm_ref_shared_params(algorithm_ref)
    _validate_parameter_names(algorithm_config, shared_params)
    _validate_parameter_values(
        algorithm_config,
        _merge_params(algorithm_config.default_params, shared_params),
        strict,
    )
    if algorithm_ref.configs:
        parameter_sets = []
        for config_params in algorithm_ref.configs:
            _validate_parameter_names(algorithm_config, config_params)
            merged_params = _merge_params(
                algorithm_config.default_params,
                config_params,
                shared_params,
            )
            _validate_parameter_values(algorithm_config, merged_params, strict)
            parameter_sets.append(merged_params)
        return parameter_sets

    search_space = (
        algorithm_ref.search_space_override
        if algorithm_ref.search_space_override is not None
        else algorithm_config.search_space
    )
    search_space = _constrain_search_space(search_space, shared_params)
    conditional_search_space = (
        []
        if algorithm_ref.search_space_override is not None
        else algorithm_config.conditional_search_space
    )
    _validate_search_space_override(
        algorithm_config,
        search_space,
        strict,
        sampling_enabled=algorithm_ref.sampling is not None,
    )
    if algorithm_ref.sampling is not None:
        # Imported lazily: sampling.py imports parameter-spec helpers back
        # from this module, so a top-level import here would be circular.
        from process_discovery_cash.experiments.sampling import sample_parameter_lhs

        raw_parameter_sets = sample_parameter_lhs(
            algorithm_config.default_params,
            search_space,
            conditional_search_space,
            n_samples=algorithm_ref.sampling.n_samples,
            seed=algorithm_ref.sampling.seed,
        )
    else:
        raw_parameter_sets = expand_parameter_grid(
            algorithm_config.default_params,
            search_space,
            conditional_search_space,
        )

    parameter_sets = []
    for params in raw_parameter_sets:
        merged_params = _merge_params(params, shared_params)
        _validate_parameter_values(algorithm_config, merged_params, strict)
        parameter_sets.append(merged_params)
    return parameter_sets


def _constrain_search_space(
    search_space: dict[str, Any] | None,
    shared_params: dict[str, Any],
) -> dict[str, Any] | None:
    if not search_space:
        return search_space
    constrained = dict(search_space)
    for parameter_name, value in shared_params.items():
        if parameter_name in constrained:
            constrained[parameter_name] = {"values": [value]}
    return constrained


def _algorithm_ref_shared_params(algorithm_ref: AlgorithmReference) -> dict[str, Any]:
    shared_params = dict(algorithm_ref.params)
    for key, value in _model_extra(algorithm_ref).items():
        if key not in {
            "name",
            "algorithm_id",
            "config",
            "backend",
            "model_type",
            "runtime_params",
            "artifact_algorithm_id",
            "default_params",
            "supported",
            "params",
            "configs",
            "search_space_override",
            "sampling",
        }:
            shared_params[key] = value
    return shared_params


def _merge_params(*sources: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        merged.update(dict(source))
    return merged


def _model_extra(model: Any) -> dict[str, Any]:
    extra = getattr(model, "model_extra", None)
    if isinstance(extra, dict):
        return extra
    return getattr(model, "__pydantic_extra__", None) or {}


def write_manifest(rows: list[dict[str, str]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def generate_manifest(
    experiment_config_path: str | Path,
    output_path: str | Path | None = None,
    *,
    require_artifacts: bool = False,
) -> Path:
    rows = generate_manifest_rows(
        experiment_config_path,
        require_artifacts=require_artifacts,
    )
    if output_path is None:
        experiment = load_experiment_config(experiment_config_path)
        output_path = experiment.manifest_path or (
            Path("experiments/manifests") / f"{experiment.experiment_id}_manifest.csv"
        )
    return write_manifest(rows, output_path)


def expand_parameter_grid(
    default_params: dict[str, Any],
    search_space: dict[str, Any] | None,
    conditional_search_space: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not search_space:
        base_combinations = [dict(default_params)]
    else:
        base_combinations = _expand_flat_parameter_grid(default_params, search_space)

    if not conditional_search_space:
        return base_combinations

    combinations: list[dict[str, Any]] = []
    for base_params in base_combinations:
        matching_conditional_specs = [
            conditional_spec
            for conditional_spec in conditional_search_space
            if _condition_matches(base_params, conditional_spec.get("when", {}))
        ]
        if not matching_conditional_specs:
            combinations.append(base_params)
            continue

        conditional_combinations = [dict(base_params)]
        for conditional_spec in matching_conditional_specs:
            params_spec = conditional_spec.get("params") or {}
            expanded_conditional_params = _expand_flat_parameter_grid({}, params_spec)
            conditional_combinations = [
                _merge_params(existing, conditional_params)
                for existing, conditional_params in product(
                    conditional_combinations,
                    expanded_conditional_params,
                )
            ]
        combinations.extend(conditional_combinations)

    return combinations


def _resolve_parameter_values(spec: Any) -> list[Any]:
    if isinstance(spec, dict) and "values" in spec:
        return list(spec["values"])
    elif _is_range_parameter_spec(spec):
        raise ValueError(
            "Range-based search-space specs require sampling and cannot be expanded "
            "as a discrete grid."
        )
    elif isinstance(spec, list):
        return spec
    else:
        return [spec]


def _expand_flat_parameter_grid(
    default_params: dict[str, Any],
    search_space: dict[str, Any],
) -> list[dict[str, Any]]:
    keys = sorted(search_space)
    values: list[list[Any]] = [_resolve_parameter_values(search_space[key]) for key in keys]

    combinations: list[dict[str, Any]] = []
    for combo in product(*values):
        params = dict(default_params)
        params.update(dict(zip(keys, combo, strict=True)))
        combinations.append(params)
    return combinations


def _condition_matches(params: dict[str, Any], condition: dict[str, Any]) -> bool:
    return all(params.get(key) == value for key, value in condition.items())


def _load_algorithm_configs(config_dir: str | Path) -> dict[str, tuple[Path, Any]]:
    config_root = Path(config_dir)
    configs: dict[str, tuple[Path, Any]] = {}
    for path in sorted(config_root.glob("*.yaml")):
        config = load_algorithm_config(path)
        algorithm_id = config.algorithm_id
        if algorithm_id in configs:
            previous_path = configs[algorithm_id][0]
            raise ValueError(
                f"Duplicate algorithm_id '{algorithm_id}' in {previous_path} and {path}"
            )
        configs[algorithm_id] = (path, config)
    return configs


def _resolve_algorithm_config(
    algorithm_ref: AlgorithmReference,
    algorithm_configs: dict[str, tuple[Path, Any]],
    experiment: Any,
) -> tuple[Path | None, Any]:
    if _is_inline_algorithm_definition(algorithm_ref):
        return None, _inline_algorithm_config(algorithm_ref)
    if algorithm_ref.config:
        path = Path(algorithm_ref.config)
        config = load_algorithm_config(path)
        if config.algorithm_id != algorithm_ref.name:
            raise ValueError(
                f"Experiment references algorithm '{algorithm_ref.name}' with config "
                f"{path}, but that config declares algorithm_id '{config.algorithm_id}'."
            )
        return path, config

    try:
        return algorithm_configs[algorithm_ref.name]
    except KeyError as exc:
        known = ", ".join(sorted(algorithm_configs)) or "none"
        raise ValueError(
            f"Unknown algorithm '{algorithm_ref.name}' in experiment "
            f"'{experiment.experiment_id}'. Known algorithms: {known}."
        ) from exc


def _validate_algorithm_support(
    algorithm_ref: AlgorithmReference,
    algorithm_config: Any,
    experiment: Any,
) -> None:
    if algorithm_config.supported or experiment.allow_unsupported:
        return
    raise ValueError(
        f"Algorithm '{algorithm_ref.name}' is marked unsupported in its algorithm config. "
        "Set allow_unsupported: true in the experiment only for explicit unsupported-backend "
        "checks."
    )


def _validate_search_space_override(
    algorithm_config: Any,
    search_space: dict[str, Any] | None,
    strict: bool,
    *,
    sampling_enabled: bool = False,
) -> None:
    if not search_space:
        return
    _validate_parameter_names(algorithm_config, search_space)
    for parameter_name, spec in search_space.items():
        if _is_range_parameter_spec(spec):
            if not sampling_enabled:
                raise ValueError(
                    f"Algorithm '{algorithm_config.algorithm_id}' parameter "
                    f"'{parameter_name}' uses a range-based search-space spec that "
                    "requires sampling."
                )
            _validate_range_parameter_spec(parameter_name, spec)
            continue
        for value in _search_space_values(spec):
            _validate_parameter_value(algorithm_config, parameter_name, value, strict)


def _validate_parameter_names(algorithm_config: Any, params: dict[str, Any]) -> None:
    if getattr(algorithm_config, "inline_definition", False):
        return
    allowed = _declared_parameter_names(algorithm_config)
    unknown = sorted(set(params) - allowed)
    if not unknown:
        return
    raise ValueError(
        f"Algorithm '{algorithm_config.algorithm_id}' received undeclared parameter(s): "
        f"{', '.join(unknown)}. Declare parameters in default_params or search_space."
    )


def _validate_parameter_values(
    algorithm_config: Any,
    params: dict[str, Any],
    strict: bool,
) -> None:
    for parameter_name, value in params.items():
        _validate_parameter_value(
            algorithm_config,
            parameter_name,
            value,
            strict,
            params=params,
        )


def _validate_parameter_value(
    algorithm_config: Any,
    parameter_name: str,
    value: Any,
    strict: bool,
    params: dict[str, Any] | None = None,
) -> None:
    allowed_specs = _allowed_specs_for_parameter(algorithm_config, parameter_name, params)
    if allowed_specs is None:
        return
    if any(_parameter_value_allowed_by_spec(value, spec) for spec in allowed_specs):
        return
    message = (
        f"Algorithm '{algorithm_config.algorithm_id}' parameter '{parameter_name}' has "
        f"value {value!r}, which is outside the declared search_space "
        f"{_format_parameter_specs(allowed_specs)}."
    )
    if strict:
        raise ValueError(message)
    LOGGER.warning("%s Allowing it because strict_parameter_validation is false.", message)


def _declared_parameter_names(algorithm_config: Any) -> set[str]:
    return (
        set(algorithm_config.default_params)
        | set(algorithm_config.search_space)
        | set(algorithm_config.runtime_params)
        | _conditional_parameter_names(algorithm_config)
    )


def _conditional_parameter_names(algorithm_config: Any) -> set[str]:
    parameter_names: set[str] = set()
    for conditional_spec in algorithm_config.conditional_search_space:
        params_spec = conditional_spec.get("params") or {}
        parameter_names.update(params_spec)
    return parameter_names


def _allowed_specs_for_parameter(
    algorithm_config: Any,
    parameter_name: str,
    params: dict[str, Any] | None = None,
) -> list[Any] | None:
    allowed_specs: list[Any] = []
    if parameter_name in algorithm_config.search_space:
        allowed_specs.append(algorithm_config.search_space[parameter_name])

    matching_conditional_specs: list[Any] = []
    conditional_parameter_declared = False
    for conditional_spec in algorithm_config.conditional_search_space:
        params_spec = conditional_spec.get("params") or {}
        if parameter_name not in params_spec:
            continue
        conditional_parameter_declared = True
        if params is None or _condition_matches(params, conditional_spec.get("when", {})):
            matching_conditional_specs.append(params_spec[parameter_name])

    if matching_conditional_specs:
        allowed_specs.extend(matching_conditional_specs)
    elif conditional_parameter_declared and parameter_name not in algorithm_config.search_space:
        return []

    if not allowed_specs:
        return None
    return allowed_specs


def _search_space_values(spec: Any) -> list[Any]:
    if _is_range_parameter_spec(spec):
        raise ValueError(
            "Range-based search-space specs cannot be interpreted as explicit discrete values."
        )
    if isinstance(spec, dict) and "values" in spec:
        return list(spec["values"])
    if isinstance(spec, list):
        return list(spec)
    return [spec]


def _is_range_parameter_spec(spec: Any) -> bool:
    return isinstance(spec, dict) and "min" in spec and "max" in spec


def _validate_range_parameter_spec(parameter_name: str, spec: dict[str, Any]) -> None:
    parameter_type = spec.get("type")
    if parameter_type not in {"float", "integer"}:
        raise ValueError(
            f"Range-based search-space spec for '{parameter_name}' must declare "
            "type 'float' or 'integer'."
        )

    minimum = spec.get("min")
    maximum = spec.get("max")
    if not isinstance(minimum, Real) or not isinstance(maximum, Real):
        raise ValueError(
            f"Range-based search-space spec for '{parameter_name}' must use numeric "
            "'min' and 'max' bounds."
        )
    if minimum > maximum:
        raise ValueError(
            f"Range-based search-space spec for '{parameter_name}' has min={minimum!r} "
            f"greater than max={maximum!r}."
        )

    if parameter_type == "integer":
        if not float(minimum).is_integer() or not float(maximum).is_integer():
            raise ValueError(
                f"Range-based search-space spec for '{parameter_name}' declares "
                "type 'integer' but uses non-integer bounds."
            )

    scale = spec.get("scale")
    if scale is not None and scale not in {"linear", "log"}:
        raise ValueError(
            f"Range-based search-space spec for '{parameter_name}' has unsupported "
            f"scale {scale!r}; use 'linear' or 'log'."
        )
    if scale == "log" and (minimum <= 0 or maximum <= 0):
        raise ValueError(
            f"Range-based search-space spec for '{parameter_name}' uses log scale "
            "but has non-positive bounds."
        )


def _parameter_value_allowed_by_spec(value: Any, spec: Any) -> bool:
    if _is_range_parameter_spec(spec):
        parameter_type = spec.get("type")
        minimum = spec["min"]
        maximum = spec["max"]
        if not isinstance(value, Real):
            return False
        if value < minimum or value > maximum:
            return False
        if parameter_type == "integer":
            return float(value).is_integer()
        return True
    return value in _search_space_values(spec)


def _format_parameter_specs(specs: list[Any]) -> str:
    formatted: list[str] = []
    for spec in specs:
        if _is_range_parameter_spec(spec):
            scale = spec.get("scale") or "default"
            formatted.append(
                f"[{spec['min']!r}, {spec['max']!r}] (type={spec.get('type')}, scale={scale})"
            )
        else:
            formatted.append(repr(_search_space_values(spec)))
    return ", ".join(formatted)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _normalize_algorithm_ref(value: str | AlgorithmReference) -> AlgorithmReference:
    if isinstance(value, AlgorithmReference):
        return value
    return AlgorithmReference(name=value)


def _is_inline_algorithm_definition(algorithm_ref: AlgorithmReference) -> bool:
    return algorithm_ref.backend is not None


def _inline_algorithm_config(algorithm_ref: AlgorithmReference) -> ResolvedAlgorithmConfig:
    runtime_params = list(algorithm_ref.runtime_params)
    default_params = dict(algorithm_ref.default_params)
    if DISCOVERY_TIMEOUT_FIELD in runtime_params:
        default_params.setdefault(
            DISCOVERY_TIMEOUT_FIELD,
            DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
        )
    if "timeout_seconds" in runtime_params:
        default_params.setdefault(
            "timeout_seconds",
            DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
        )
    return ResolvedAlgorithmConfig(
        algorithm_id=str(algorithm_ref.algorithm_id or algorithm_ref.name),
        backend=str(algorithm_ref.backend),
        model_type=str(algorithm_ref.model_type),
        default_params=default_params,
        search_space={},
        conditional_search_space=[],
        runtime_params=runtime_params,
        supported=bool(algorithm_ref.supported),
        artifact_algorithm_id=algorithm_ref.artifact_algorithm_id,
        inline_definition=True,
    )


def _artifact_algorithm_id(algorithm_config: Any) -> str:
    return str(
        getattr(algorithm_config, "artifact_algorithm_id", None) or algorithm_config.algorithm_id
    )
