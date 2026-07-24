"""Execute one HPO trial as a standard manifest-row discovery job.

A trial synthesizes a manifest-row dict (same columns and identity hashing as
``experiments/manifest.py``) and executes it through the existing runner —
isolated in a child process by default so a hard deadline bounds discovery
*and* metric computation. Result JSONs land at the same
``{results_dir}/{log_id}/{config_hash}.json`` paths a manifest run would use,
which makes trials idempotent: a config the sampler proposes twice (or a study
restart) is answered from the existing result file without re-running.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from process_discovery_cash.config.load import load_experiment_config
from process_discovery_cash.config.schema import ExperimentConfig, HpoConfig
from process_discovery_cash.data.preprocessing.models import ArtifactSelection
from process_discovery_cash.experiments.identity import semantic_run_identity
from process_discovery_cash.experiments.manifest import (
    _algorithm_ref_shared_params,
    _algorithm_variant,
    _constrain_search_space,
    _experiment_metrics_config,
    _experiment_seed,
    _load_algorithm_configs,
    _merge_params,
    _normalize_algorithm_ref,
    _portable_manifest_path,
    _resolve_algorithm_config,
    _result_output_path,
    _validate_algorithm_support,
    _validate_parameter_names,
    _validate_parameter_values,
    _validate_search_space_override,
)
from process_discovery_cash.experiments.run_isolation import (
    ROW_KIND_DISCOVERY,
    run_row_in_subprocess,
)
from process_discovery_cash.experiments.runner import (
    ResultFileState,
    inspect_result_file,
    run_manifest_row,
)
from process_discovery_cash.hpo.objective import ObjectiveOutcome, objective_from_result_payload
from process_discovery_cash.hpo.search_space import HpoSearchSpace, build_hpo_search_space
from process_discovery_cash.utils.hashing import stable_hash
from process_discovery_cash.utils.paths import project_root

_TERMINAL_FAILURE_STATES = {
    ResultFileState.FAILED,
    ResultFileState.TIMEOUT,
    ResultFileState.UNSUPPORTED,
}


@dataclass(frozen=True)
class StudyContext:
    """Everything resolvable once per (log, algorithm) study."""

    experiment: ExperimentConfig
    hpo: HpoConfig
    experiment_config_path: str
    log_ref: Any
    algorithm_ref: Any
    algorithm_config: Any
    space: HpoSearchSpace
    default_params: dict[str, Any]
    shared_params: dict[str, Any]
    metrics_config: dict[str, Any]
    seed: int
    artifact: Any
    test_artifact: Any
    strict: bool

    @classmethod
    def from_experiment(
        cls,
        experiment_config_path: str | Path,
        log_id: str,
        algorithm_name: str,
        *,
        require_artifacts: bool = False,
    ) -> StudyContext:
        if require_artifacts:
            warnings.warn(
                "--require-artifacts is deprecated and ignored; HPO rows use XES "
                "sources and optional runtime-local caches.",
                DeprecationWarning,
                stacklevel=2,
            )
        experiment_config_path = Path(experiment_config_path)
        experiment = load_experiment_config(experiment_config_path)
        if experiment.hpo is None:
            raise ValueError(
                f"Experiment '{experiment.experiment_id}' has no 'hpo' block; "
                "HPO studies require one."
            )
        _validate_hpo_output_template(experiment)

        log_ref = _find_log_ref(experiment, log_id)
        algorithm_ref = _find_algorithm_ref(experiment, algorithm_name)
        if algorithm_ref.sampling is not None or algorithm_ref.configs:
            raise ValueError(
                f"Algorithm '{algorithm_ref.name}' in HPO experiment "
                f"'{experiment.experiment_id}' must not declare 'sampling' or 'configs'; "
                "the optimizer proposes configurations itself."
            )

        algorithm_configs = _load_algorithm_configs(experiment.algorithm_config_dir)
        _config_path, algorithm_config = _resolve_algorithm_config(
            algorithm_ref, algorithm_configs, experiment
        )
        _validate_algorithm_support(algorithm_ref, algorithm_config, experiment)

        shared_params = _algorithm_ref_shared_params(algorithm_ref)
        _validate_parameter_names(algorithm_config, shared_params)
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
        strict = experiment.strict_parameter_validation
        # HPO consumes range specs the same way sampling does.
        _validate_search_space_override(
            algorithm_config, search_space, strict, sampling_enabled=True
        )
        space = build_hpo_search_space(search_space, conditional_search_space)

        metrics_config = _experiment_metrics_config(experiment)
        if experiment.hpo.export_model is not None:
            metrics_config["export_model"] = experiment.hpo.export_model

        artifact = ArtifactSelection(
            dataset_id=log_ref.dataset_id,
            source_log_path=log_ref.source_path or log_ref.path,
        )
        test_artifact = artifact
        if log_ref.test_path and log_ref.test_path != log_ref.path:
            test_artifact = ArtifactSelection(
                dataset_id=log_ref.dataset_id,
                source_log_path=log_ref.test_path,
            )

        return cls(
            experiment=experiment,
            hpo=experiment.hpo,
            experiment_config_path=str(experiment_config_path),
            log_ref=log_ref,
            algorithm_ref=algorithm_ref,
            algorithm_config=algorithm_config,
            space=space,
            default_params=dict(algorithm_config.default_params),
            shared_params=shared_params,
            metrics_config=metrics_config,
            seed=_experiment_seed(experiment),
            artifact=artifact,
            test_artifact=test_artifact,
            strict=strict,
        )

    def finalize_trial_params(self, suggested_params: dict[str, Any]) -> dict[str, Any]:
        """Apply shared ref params on top of suggestions and validate values."""
        params = _merge_params(suggested_params, self.shared_params)
        _validate_parameter_values(self.algorithm_config, params, self.strict)
        return params


@dataclass(frozen=True)
class TrialOutcome:
    config_hash: str
    objective: ObjectiveOutcome
    cached: bool
    result_path: Path
    killed_at_study_deadline: bool = False


def trial_config_hash(ctx: StudyContext, params: dict[str, Any]) -> str:
    """Identity hash for one trial config — must match ``generate_manifest_rows``."""
    return stable_hash(
        semantic_run_identity(
            log_id=ctx.log_ref.log_id,
            log_path=ctx.log_ref.path,
            test_log_path=ctx.log_ref.test_path,
            seed=ctx.seed,
            algorithm_id=ctx.algorithm_config.algorithm_id,
            backend=ctx.algorithm_config.backend,
            params=params,
            metrics=ctx.metrics_config,
            runtime_fields=ctx.algorithm_config.runtime_params,
        )
    )


def build_trial_row(
    ctx: StudyContext,
    params: dict[str, Any],
    config_hash: str,
) -> dict[str, str]:
    """Manifest-row dict for one trial, mirroring ``generate_manifest_rows``."""
    root = project_root()
    experiment = ctx.experiment
    log_ref = ctx.log_ref
    artifact = ctx.artifact
    test_artifact = ctx.test_artifact
    output_path = _result_output_path(
        experiment,
        log_id=log_ref.log_id,
        seed=ctx.seed,
        algorithm_id=ctx.algorithm_config.algorithm_id,
        config_hash=config_hash,
        row_index=0,
    )
    params_json = json.dumps(params, sort_keys=True)
    metrics_json = json.dumps(ctx.metrics_config, sort_keys=True)
    return {
        "experiment_id": experiment.experiment_id,
        "log_id": log_ref.log_id,
        "dataset_id": artifact.dataset_id or "",
        "log_path": _portable_manifest_path(log_ref.path, root),
        "test_log_path": _portable_manifest_path(log_ref.test_path, root),
        "source_log_path": _portable_manifest_path(artifact.source_log_path, root),
        "discovery_log_path": _portable_manifest_path(artifact.discovery_log_path, root)
        if artifact.discovery_log_path
        else "",
        "test_discovery_log_path": _portable_manifest_path(test_artifact.discovery_log_path, root)
        if test_artifact.discovery_log_path
        else "",
        "artifact_kind": artifact.artifact_kind or "",
        "artifact_sha256": artifact.artifact_sha256 or "",
        "preprocessing_fingerprint": artifact.preprocessing_fingerprint or "",
        "preprocessing_metadata_path": _portable_manifest_path(
            artifact.preprocessing_metadata_path, root
        )
        if artifact.preprocessing_metadata_path
        else "",
        "seed": str(ctx.seed),
        "algorithm_id": ctx.algorithm_config.algorithm_id,
        "algorithm_variant": _algorithm_variant(ctx.algorithm_config, params),
        "algorithm": ctx.algorithm_config.algorithm_id,
        "backend": ctx.algorithm_config.backend,
        "algorithm_params_json": params_json,
        "params_json": params_json,
        "metrics_json": metrics_json,
        "config_id": config_hash,
        "config_hash": config_hash,
        "log_dir": _portable_manifest_path(experiment.output.log_dir, root),
        "output_path": _portable_manifest_path(output_path, root),
    }


def run_trial(
    ctx: StudyContext,
    params: dict[str, Any],
    *,
    deadline_monotonic: float | None = None,
    deadline_is_study_deadline: bool = False,
    memory_limit_mb: float | None = None,
    isolate: bool = True,
    scratch_dir: str | Path | None = None,
    command_args: list[str] | None = None,
    on_tick: Callable[[], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> TrialOutcome:
    """Evaluate one configuration, reusing an existing result file when possible."""
    config_hash = trial_config_hash(ctx, params)
    row = build_trial_row(ctx, params, config_hash)
    output_path = Path(row["output_path"])
    weights = ctx.hpo.objective.weights
    failed_value = ctx.hpo.objective.failed_trial_value

    inspection = inspect_result_file(row, output_path)
    if inspection.state is ResultFileState.SUCCESS or (
        inspection.state in _TERMINAL_FAILURE_STATES and inspection.payload is not None
    ):
        objective = objective_from_result_payload(
            inspection.payload or {}, weights, failed_value=failed_value
        )
        return TrialOutcome(
            config_hash=config_hash,
            objective=objective,
            cached=True,
            result_path=output_path,
        )

    killed_by_parent = False
    if isolate:
        outcome = run_row_in_subprocess(
            row,
            kind=ROW_KIND_DISCOVERY,
            command_args=command_args,
            deadline_monotonic=deadline_monotonic,
            memory_limit_mb=memory_limit_mb,
            scratch_dir=scratch_dir,
            on_tick=on_tick,
            monotonic=monotonic,
        )
        killed_by_parent = outcome.killed_by_parent
    else:
        run_manifest_row(row, command_args=command_args)

    inspection = inspect_result_file(row, output_path)
    if inspection.payload is not None and inspection.state is not ResultFileState.CORRUPT:
        objective = objective_from_result_payload(
            inspection.payload, weights, failed_value=failed_value
        )
        return TrialOutcome(
            config_hash=config_hash,
            objective=objective,
            cached=False,
            result_path=output_path,
        )

    # The child died without writing a usable result (deadline kill, OOM, crash).
    if killed_by_parent and deadline_is_study_deadline:
        return TrialOutcome(
            config_hash=config_hash,
            objective=ObjectiveOutcome(value=failed_value, run_status="study_walltime"),
            cached=False,
            result_path=output_path,
            killed_at_study_deadline=True,
        )
    run_status = "trial_walltime_exceeded" if killed_by_parent else "crashed"
    return TrialOutcome(
        config_hash=config_hash,
        objective=ObjectiveOutcome(value=failed_value, run_status=run_status),
        cached=False,
        result_path=output_path,
    )


def _validate_hpo_output_template(experiment: ExperimentConfig) -> None:
    template = experiment.output.output_path_template
    if not template:
        raise ValueError(
            f"HPO experiment '{experiment.experiment_id}' must set "
            "output.output_path_template (e.g. '{results_dir}/{log_id}/{config_hash}.json') "
            "so result paths are stable across trials and restarts."
        )
    if "{row_index" in template:
        raise ValueError(
            f"HPO experiment '{experiment.experiment_id}' output_path_template must not "
            "use {row_index}; trials have no stable row order."
        )
    if "{config_hash" not in template:
        raise ValueError(
            f"HPO experiment '{experiment.experiment_id}' output_path_template must "
            "include {config_hash} so distinct configurations get distinct result files."
        )


def _find_log_ref(experiment: ExperimentConfig, log_id: str) -> Any:
    for log_ref in experiment.logs:
        if log_ref.log_id == log_id:
            return log_ref
    known = ", ".join(ref.log_id for ref in experiment.logs) or "none"
    raise ValueError(
        f"Unknown log_id '{log_id}' in experiment '{experiment.experiment_id}'. "
        f"Known logs: {known}."
    )


def _find_algorithm_ref(experiment: ExperimentConfig, algorithm_name: str) -> Any:
    for entry in experiment.algorithms:
        normalized = _normalize_algorithm_ref(entry)
        if normalized.name == algorithm_name:
            return normalized
    known = (
        ", ".join(_normalize_algorithm_ref(entry).name for entry in experiment.algorithms) or "none"
    )
    raise ValueError(
        f"Unknown algorithm '{algorithm_name}' in experiment "
        f"'{experiment.experiment_id}'. Known algorithms: {known}."
    )
