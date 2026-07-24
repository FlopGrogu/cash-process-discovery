from __future__ import annotations

from typing import Any

from process_discovery_cash.data.loading import (
    ensure_pm4py_event_log,  # noqa: F401 - retained for backend monkeypatch compatibility
    prepare_pm4py_discovery_log,
)

RUNTIME_CONFIG_FIELDS = {"input_log_path", "output_dir", "log_id"}
WRAPPER_ONLY_FIELDS = {"discovery_timeout_seconds", "recursion_limit"}
INTERNAL_CONFIG_FIELDS = RUNTIME_CONFIG_FIELDS | WRAPPER_ONLY_FIELDS

ACTIVITY_KEY = "pm4py:param:activity_key"
TIMESTAMP_KEY = "pm4py:param:timestamp_key"
START_TIMESTAMP_KEY = "pm4py:param:start_timestamp_key"
CASE_ID_KEY = "pm4py:param:case_id_key"


class UnsupportedAlgorithmError(RuntimeError):
    """Raised when a requested pm4py discovery algorithm is unavailable."""


def _pm4py_import_error(exc: Exception, algorithm: str) -> UnsupportedAlgorithmError:
    return UnsupportedAlgorithmError(
        f"{algorithm} is not available in the installed pm4py version: {exc}"
    )


def _pm4py_discovery_function(function_name: str, algorithm: str) -> Any:
    try:
        from pm4py import discovery as pm4py_discovery
    except Exception as exc:
        raise _pm4py_import_error(exc, algorithm) from exc

    function = getattr(pm4py_discovery, function_name, None)
    if function is None:
        raise UnsupportedAlgorithmError(
            f"{algorithm} is not available in the installed pm4py version: "
            f"pm4py.discovery.{function_name} is missing."
        )
    return function


def _pm4py_algorithm_module(module_path: str, algorithm: str) -> Any:
    try:
        return __import__(module_path, fromlist=["algorithm"])
    except Exception as exc:
        raise _pm4py_import_error(exc, algorithm) from exc


def discover_alpha_miner(event_log: Any, params: dict[str, Any]) -> dict[str, Any]:
    alpha_miner = _pm4py_algorithm_module(
        "pm4py.algo.discovery.alpha.algorithm",
        "Alpha Miner",
    )
    requested_variant = _requested_variant(params, default="classic")
    resolved_variant_name, resolved_variant = _resolve_variant(
        "Alpha Miner",
        getattr(alpha_miner, "Variants", None),
        requested_variant,
        "classic",
        {
            "classic": ["ALPHA_VERSION_CLASSIC", "CLASSIC"],
            "plus": ["ALPHA_VERSION_PLUS", "PLUS"],
        },
    )
    requested_parameters = dict(params)
    parameters, backend_parameters, ignored_parameters, warnings = _build_backend_parameters(
        "Alpha Miner",
        requested_parameters,
        _alpha_parameter_mapping(alpha_miner),
    )
    if requested_variant == "plus":
        warnings.append(
            "Alpha+ is exposed by PM4Py but deprecated and should be treated as optional."
        )

    pm4py_log = prepare_pm4py_discovery_log(
        event_log,
        allow_dataframe=requested_variant == "classic",
    )
    model = alpha_miner.apply(
        pm4py_log,
        parameters=backend_parameters or None,
        variant=resolved_variant,
    )
    _assert_petri_net_tuple("Alpha Miner", "apply", model)
    return {
        "model": model,
        "model_type": "petri_net",
        "metadata": _discovery_metadata(
            alpha_miner,
            "apply",
            requested_variant,
            resolved_variant_name,
            parameters=parameters,
            backend_parameters=backend_parameters,
            requested_parameters=requested_parameters,
            ignored_parameters=ignored_parameters,
            warnings=warnings,
            backend_function="pm4py.algo.discovery.alpha.algorithm.apply",
        ),
    }


def discover_inductive_miner(event_log: Any, params: dict[str, Any]) -> dict[str, Any]:
    inductive_miner = _pm4py_algorithm_module(
        "pm4py.algo.discovery.inductive.algorithm",
        "Inductive Miner",
    )
    requested_variant = _requested_variant(params, default="im")
    resolved_variant_name, resolved_variant = _resolve_variant(
        "Inductive Miner",
        getattr(inductive_miner, "Variants", None),
        requested_variant,
        "im",
        {
            "im": ["IM"],
            "imf": ["IMf", "IMF"],
            "imd": ["IMd", "IMD"],
        },
    )
    requested_parameters = dict(params)
    parameters, backend_parameters, ignored_parameters, warnings = _build_backend_parameters(
        "Inductive Miner",
        requested_parameters,
        _inductive_parameter_mapping(inductive_miner),
    )

    pm4py_log = prepare_pm4py_discovery_log(event_log, allow_dataframe=True)
    process_tree = inductive_miner.apply(
        pm4py_log,
        parameters=backend_parameters or None,
        variant=resolved_variant,
    )
    model = _convert_process_tree_to_petri_net(process_tree)
    return {
        "model": model,
        "model_type": "petri_net",
        "metadata": _discovery_metadata(
            inductive_miner,
            "apply",
            requested_variant,
            resolved_variant_name,
            parameters=parameters,
            backend_parameters=backend_parameters,
            requested_parameters=requested_parameters,
            ignored_parameters=ignored_parameters,
            warnings=warnings,
            backend_function="pm4py.algo.discovery.inductive.algorithm.apply",
            extra={"converted_to_petri_net": True},
        ),
    }


def discover_ilp_miner(event_log: Any, params: dict[str, Any]) -> dict[str, Any]:
    ilp_miner = _pm4py_algorithm_module(
        "pm4py.algo.discovery.ilp.algorithm",
        "ILP Miner",
    )
    requested_variant = _requested_variant(params, default="classic")
    resolved_variant_name, resolved_variant = _resolve_variant(
        "ILP Miner",
        getattr(ilp_miner, "Variants", None),
        requested_variant,
        "classic",
        {"classic": ["CLASSIC"]},
    )
    requested_parameters = dict(params)
    parameters, backend_parameters, ignored_parameters, warnings = _build_backend_parameters(
        "ILP Miner",
        requested_parameters,
        _ilp_parameter_mapping(resolved_variant),
    )

    pm4py_log = prepare_pm4py_discovery_log(event_log, allow_dataframe=False)
    model = ilp_miner.apply(
        pm4py_log,
        variant=resolved_variant,
        parameters=backend_parameters or None,
    )
    _assert_petri_net_tuple("ILP Miner", "apply", model)
    return {
        "model": model,
        "model_type": "petri_net",
        "metadata": _discovery_metadata(
            ilp_miner,
            "apply",
            requested_variant,
            resolved_variant_name,
            parameters=parameters,
            backend_parameters=backend_parameters,
            requested_parameters=requested_parameters,
            ignored_parameters=ignored_parameters,
            warnings=warnings,
            backend_function="pm4py.algo.discovery.ilp.algorithm.apply",
        ),
    }


def discover_heuristics_miner(event_log: Any, params: dict[str, Any]) -> dict[str, Any]:
    heuristics_miner = _pm4py_algorithm_module(
        "pm4py.algo.discovery.heuristics.algorithm",
        "Heuristics Miner",
    )
    requested_variant = _requested_variant(params, default="classic")
    resolved_variant_name, resolved_variant = _resolve_variant(
        "Heuristics Miner",
        getattr(heuristics_miner, "Variants", None),
        requested_variant,
        "classic",
        {
            "classic": ["CLASSIC"],
            "plusplus": ["PLUSPLUS"],
            "++": ["PLUSPLUS"],
        },
    )
    requested_parameters = dict(params)
    parameters, backend_parameters, ignored_parameters, warnings = _build_backend_parameters(
        "Heuristics Miner",
        requested_parameters,
        _heuristic_parameter_mapping(resolved_variant),
    )

    pm4py_log = prepare_pm4py_discovery_log(event_log, allow_dataframe=True)
    model = heuristics_miner.apply(
        pm4py_log,
        parameters=backend_parameters or None,
        variant=resolved_variant,
    )
    _assert_petri_net_tuple("Heuristics Miner", "apply", model)
    return {
        "model": model,
        "model_type": "petri_net",
        "metadata": _discovery_metadata(
            heuristics_miner,
            "apply",
            requested_variant,
            resolved_variant_name,
            parameters=parameters,
            backend_parameters=backend_parameters,
            requested_parameters=requested_parameters,
            ignored_parameters=ignored_parameters,
            warnings=warnings,
            backend_function="pm4py.algo.discovery.heuristics.algorithm.apply",
        ),
    }


def discover_genetic_miner(event_log: Any, params: dict[str, Any]) -> dict[str, Any]:
    genetic_miner = _pm4py_algorithm_module(
        "pm4py.algo.discovery.genetic.algorithm",
        "Genetic Miner",
    )
    requested_variant = _requested_variant(params, default="classic")
    resolved_variant_name, resolved_variant = _resolve_variant(
        "Genetic Miner",
        getattr(genetic_miner, "Variants", None),
        requested_variant,
        "classic",
        {"classic": ["CLASSIC"]},
    )
    requested_parameters = dict(params)
    parameters, backend_parameters, ignored_parameters, warnings = _build_backend_parameters(
        "Genetic Miner",
        requested_parameters,
        _genetic_parameter_mapping(genetic_miner),
        allow_none_keys={"log_csv"},
    )
    runtime_parameters = {}
    if "discovery_timeout_seconds" in requested_parameters:
        runtime_parameters["discovery_timeout_seconds"] = requested_parameters[
            "discovery_timeout_seconds"
        ]
    model = genetic_miner.apply(
        event_log,
        variant=resolved_variant,
        parameters=backend_parameters or None,
    )
    _assert_petri_net_tuple("Genetic Miner", "apply", model)
    return {
        "model": model,
        "model_type": "petri_net",
        "metadata": _discovery_metadata(
            genetic_miner,
            "apply",
            requested_variant,
            resolved_variant_name,
            parameters=parameters,
            backend_parameters=backend_parameters,
            requested_parameters=requested_parameters,
            ignored_parameters=ignored_parameters,
            warnings=warnings,
            runtime_parameters=runtime_parameters,
            backend_function="pm4py.algo.discovery.genetic.algorithm.apply",
        ),
    }


def _available_variant_names(variants: Any) -> list[str]:
    if variants is None:
        return []
    members = getattr(variants, "__members__", None)
    if isinstance(members, dict):
        return list(members)
    return [name for name in dir(variants) if not name.startswith("_") and name[:1].isupper()]


def _resolve_variant(
    algorithm_label: str,
    variants_enum: Any,
    requested_variant: str | None,
    default_variant: str,
    candidates_by_config_name: dict[str, list[str]],
) -> tuple[str, Any]:
    normalized = _normalize_variant_name(requested_variant, default_variant)
    available_variants = _available_variant_names(variants_enum)

    for candidate in candidates_by_config_name.get(normalized, []):
        if variants_enum is not None and hasattr(variants_enum, candidate):
            return candidate, getattr(variants_enum, candidate)

    supported = ", ".join(sorted(candidates_by_config_name)) or "none"
    available = ", ".join(available_variants) if available_variants else "none"
    raise UnsupportedAlgorithmError(
        f"Unsupported {algorithm_label} variant '{normalized}'. "
        f"Supported configured variants are: {supported}. "
        f"Available pm4py variants are: {available}."
    )


def _build_backend_parameters(
    algorithm_label: str,
    requested_params: dict[str, Any],
    supported_param_mapping: dict[str, Any],
    ignored_keys: set[str] | None = None,
    allow_none_keys: set[str] | None = None,
) -> tuple[dict[str, Any], dict[Any, Any], dict[str, Any], list[str]]:
    ignored_keys = {"variant", *INTERNAL_CONFIG_FIELDS} if ignored_keys is None else ignored_keys
    allow_none_keys = set() if allow_none_keys is None else allow_none_keys
    parameters: dict[str, Any] = {}
    backend_parameters: dict[Any, Any] = {}
    ignored_parameters: dict[str, Any] = {}
    warnings: list[str] = []

    for key, value in requested_params.items():
        if key == "variant" or key in RUNTIME_CONFIG_FIELDS:
            continue
        if key in ignored_keys:
            ignored_parameters[key] = value
            warnings.append(
                f"{algorithm_label} parameter '{key}' is handled by the wrapper/runtime "
                "and was not passed to pm4py."
            )
            continue

        backend_key = supported_param_mapping.get(key)
        if backend_key is None:
            ignored_parameters[key] = value
            warnings.append(
                f"{algorithm_label} parameter '{key}' is not supported by the pm4py "
                "backend and was not passed to pm4py."
            )
            continue

        if value is None and key not in allow_none_keys:
            ignored_parameters[key] = value
            warnings.append(
                f"{algorithm_label} parameter '{key}' is None and was not passed to pm4py."
            )
            continue

        parameters[key] = value
        backend_parameters[backend_key] = value

    return parameters, backend_parameters, ignored_parameters, warnings


def _simplified_parameter_mapping(*parameter_names: str) -> dict[str, str]:
    return {parameter_name: parameter_name for parameter_name in parameter_names}


def _requested_variant(params: dict[str, Any], default: str) -> str:
    return _normalize_variant_name(params.get("variant"), default)


def _normalize_variant_name(value: Any, default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    return normalized or default


def _alpha_parameter_mapping(alpha_miner: Any) -> dict[str, Any]:
    parameter_enum = getattr(alpha_miner, "Parameters", None)
    return {
        "activity_key": _parameter_key(parameter_enum, "ACTIVITY_KEY", ACTIVITY_KEY),
        "timestamp_key": _parameter_key(parameter_enum, "TIMESTAMP_KEY", TIMESTAMP_KEY),
        "start_timestamp_key": _parameter_key(
            parameter_enum,
            "START_TIMESTAMP_KEY",
            START_TIMESTAMP_KEY,
        ),
        "case_id_key": _parameter_key(parameter_enum, "CASE_ID_KEY", CASE_ID_KEY),
    }


def _inductive_parameter_mapping(inductive_miner: Any) -> dict[str, Any]:
    parameter_enum = getattr(inductive_miner, "Parameters", None)
    return {
        "activity_key": _parameter_key(parameter_enum, "ACTIVITY_KEY", ACTIVITY_KEY),
        "timestamp_key": _parameter_key(parameter_enum, "TIMESTAMP_KEY", TIMESTAMP_KEY),
        "case_id_key": _parameter_key(parameter_enum, "CASE_ID_KEY", CASE_ID_KEY),
        "noise_threshold": "noise_threshold",
        "multi_processing": "multiprocessing",
        "disable_fallthroughs": "disable_fallthroughs",
    }


def _inductive_parameters(
    inductive_miner: Any,
    requested_parameters: dict[str, Any],
    resolved_variant_name: str,
) -> tuple[dict[str, Any], dict[Any, Any], dict[str, Any], list[str]]:
    parameter_enum = getattr(inductive_miner, "Parameters", None)
    parameter_mapping = {
        "activity_key": _parameter_key(parameter_enum, "ACTIVITY_KEY", ACTIVITY_KEY),
        "timestamp_key": _parameter_key(parameter_enum, "TIMESTAMP_KEY", TIMESTAMP_KEY),
        "case_id_key": _parameter_key(parameter_enum, "CASE_ID_KEY", CASE_ID_KEY),
        "multi_processing": "multiprocessing",
    }
    if resolved_variant_name == "IMf":
        parameter_mapping["noise_threshold"] = "noise_threshold"

    build_parameters = dict(requested_parameters)
    variant_limited_parameters: dict[str, Any] = {}
    if resolved_variant_name != "IMf" and "noise_threshold" in build_parameters:
        variant_limited_parameters["noise_threshold"] = build_parameters.pop("noise_threshold")
    if "disable_fallthroughs" in build_parameters:
        variant_limited_parameters["disable_fallthroughs"] = build_parameters.pop(
            "disable_fallthroughs"
        )

    parameters, backend_parameters, ignored_parameters, warnings = _build_backend_parameters(
        "Inductive Miner",
        build_parameters,
        parameter_mapping,
    )
    if "noise_threshold" in variant_limited_parameters:
        ignored_parameters["noise_threshold"] = variant_limited_parameters["noise_threshold"]
        warnings.append(
            "Inductive Miner parameter 'noise_threshold' is only passed to pm4py "
            "for the IMf variant in the low-level variant-specific backend and "
            "was not passed to pm4py."
        )
    if "disable_fallthroughs" in variant_limited_parameters:
        ignored_parameters["disable_fallthroughs"] = variant_limited_parameters[
            "disable_fallthroughs"
        ]
        warnings.append(
            "Inductive Miner parameter 'disable_fallthroughs' is exposed by the "
            "high-level pm4py helper, but the low-level variant-specific apply "
            "path in pm4py 2.7.22.2 does not read it; it was not passed to pm4py."
        )
    return parameters, backend_parameters, ignored_parameters, warnings


def _ilp_parameter_mapping(resolved_variant: Any) -> dict[str, Any]:
    parameter_enum = _variant_parameter_enum(resolved_variant)
    return {
        "activity_key": _parameter_key(parameter_enum, "ACTIVITY_KEY", ACTIVITY_KEY),
        "alpha": _parameter_key(parameter_enum, "ALPHA", "alpha"),
    }


def _heuristic_parameter_mapping(resolved_variant: Any) -> dict[str, Any]:
    parameter_enum = _variant_parameter_enum(resolved_variant)
    return {
        "activity_key": _parameter_key(parameter_enum, "ACTIVITY_KEY", ACTIVITY_KEY),
        "timestamp_key": _parameter_key(parameter_enum, "TIMESTAMP_KEY", TIMESTAMP_KEY),
        "start_timestamp_key": _parameter_key(
            parameter_enum,
            "START_TIMESTAMP_KEY",
            START_TIMESTAMP_KEY,
        ),
        "case_id_key": _parameter_key(parameter_enum, "CASE_ID_KEY", CASE_ID_KEY),
        "dependency_threshold": _parameter_key(
            parameter_enum,
            "DEPENDENCY_THRESH",
            "dependency_thresh",
        ),
        "dependency_thresh": _parameter_key(
            parameter_enum,
            "DEPENDENCY_THRESH",
            "dependency_thresh",
        ),
        "and_threshold": _parameter_key(
            parameter_enum,
            "AND_MEASURE_THRESH",
            "and_measure_thresh",
        ),
        "and_measure_thresh": _parameter_key(
            parameter_enum,
            "AND_MEASURE_THRESH",
            "and_measure_thresh",
        ),
        "min_act_count": _parameter_key(parameter_enum, "MIN_ACT_COUNT", "min_act_count"),
        "min_dfg_occurrences": _parameter_key(
            parameter_enum,
            "MIN_DFG_OCCURRENCES",
            "min_dfg_occurrences",
        ),
        "dfg_pre_cleaning_noise_threshold": _parameter_key(
            parameter_enum,
            "DFG_PRE_CLEANING_NOISE_THRESH",
            "dfg_pre_cleaning_noise_thresh",
        ),
        "dfg_pre_cleaning_noise_thresh": _parameter_key(
            parameter_enum,
            "DFG_PRE_CLEANING_NOISE_THRESH",
            "dfg_pre_cleaning_noise_thresh",
        ),
        "loop_two_threshold": _parameter_key(
            parameter_enum,
            "LOOP_LENGTH_TWO_THRESH",
            "loop_length_two_thresh",
        ),
        "loop_length_two_thresh": _parameter_key(
            parameter_enum,
            "LOOP_LENGTH_TWO_THRESH",
            "loop_length_two_thresh",
        ),
    }


def _genetic_parameter_mapping(genetic_miner: Any) -> dict[str, Any]:
    parameter_enum = getattr(genetic_miner, "Parameters", None)
    return {
        "activity_key": _parameter_key(parameter_enum, "ACTIVITY_KEY", ACTIVITY_KEY),
        "timestamp_key": _parameter_key(parameter_enum, "TIMESTAMP_KEY", TIMESTAMP_KEY),
        "case_id_key": _parameter_key(parameter_enum, "CASE_ID_KEY", CASE_ID_KEY),
        "population_size": _parameter_key(
            parameter_enum,
            "POPULATION_SIZE",
            "population_size",
        ),
        "elitism_rate": _parameter_key(parameter_enum, "ELITISM_RATE", "elitism_rate"),
        "crossover_rate": _parameter_key(
            parameter_enum,
            "CROSSOVER_RATE",
            "crossover_rate",
        ),
        "mutation_rate": _parameter_key(parameter_enum, "MUTATION_RATE", "mutation_rate"),
        "generations": _parameter_key(parameter_enum, "GENERATIONS", "generations"),
        "elitism_min_sample": _parameter_key(
            parameter_enum,
            "ELITISM_MIN_SAMPLE",
            "elitism_min_sample",
        ),
        "log_csv": _parameter_key(parameter_enum, "LOG_CSV", "log_csv"),
    }


def _parameter_key(parameter_enum: Any, enum_name: str, fallback: Any) -> Any:
    if parameter_enum is not None and hasattr(parameter_enum, enum_name):
        return getattr(parameter_enum, enum_name)
    return fallback


def _variant_parameter_enum(resolved_variant: Any) -> Any:
    return getattr(getattr(resolved_variant, "value", None), "Parameters", None)


def _convert_process_tree_to_petri_net(process_tree: Any) -> tuple[Any, Any, Any]:
    if isinstance(process_tree, tuple) and len(process_tree) == 3:
        return process_tree
    try:
        from pm4py.objects.conversion.process_tree import converter as pt_converter

        model = pt_converter.apply(process_tree)
    except Exception as exc:
        raise RuntimeError(
            f"Could not convert Inductive Miner process tree to Petri net: {exc}"
        ) from exc
    _assert_petri_net_tuple("Inductive Miner", "process tree conversion", model)
    return model


def _discovery_metadata(
    backend_module: Any,
    function_name: str,
    requested_variant: str,
    resolved_variant: str,
    parameters: dict[str, Any],
    backend_parameters: dict[Any, Any],
    requested_parameters: dict[str, Any],
    ignored_parameters: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    runtime_parameters: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    backend_function: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "variant": requested_variant,
        "requested_variant": requested_variant,
        "resolved_variant": resolved_variant,
        "parameters": dict(parameters),
        "requested_parameters": dict(requested_parameters),
        "backend_parameters": _json_safe_parameter_dict(backend_parameters),
        "ignored_parameters": dict(ignored_parameters or {}),
        "warnings": list(warnings or []),
        "backend_function": backend_function
        if backend_function is not None
        else _backend_function_name(backend_module, function_name),
    }
    if runtime_parameters is not None:
        metadata["runtime_parameters"] = dict(runtime_parameters)
    if extra:
        metadata.update(extra)
    return metadata


def _json_safe_parameter_dict(parameters: dict[Any, Any]) -> dict[str, Any]:
    return {_parameter_metadata_key(key): value for key, value in parameters.items()}


def _parameter_metadata_key(key: Any) -> str:
    value = getattr(key, "value", None)
    if isinstance(value, str):
        return value
    name = getattr(key, "name", None)
    if name:
        return str(name)
    if value is not None:
        return str(value)
    return str(key)


def _backend_function_name(backend_module: Any, function_name: str) -> str:
    module_name = getattr(backend_module, "__name__", backend_module.__class__.__name__)
    return f"{module_name}.{function_name}"


def _assert_petri_net_tuple(algorithm_label: str, function_name: str, model: Any) -> None:
    if isinstance(model, tuple) and len(model) == 3:
        return
    raise RuntimeError(
        f"{algorithm_label} {function_name} returned {type(model).__name__}; "
        "expected a Petri net tuple of (net, initial_marking, final_marking)."
    )
