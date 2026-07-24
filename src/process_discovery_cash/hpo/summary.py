"""Per-study summary JSON: best configuration plus study health counters."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from process_discovery_cash.experiments.runner import _write_json_atomically
from process_discovery_cash.hpo.study import (
    best_complete_trial,
    count_trials_by_state,
    study_name_for,
    summary_path_for,
)
from process_discovery_cash.hpo.trial_runner import StudyContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    import optuna


def build_study_summary(study: optuna.Study, ctx: StudyContext) -> dict[str, Any]:
    trials = study.get_trials(deepcopy=False)
    run_status_counts: dict[str, int] = {}
    cached_count = 0
    for trial in trials:
        run_status = trial.user_attrs.get("run_status")
        if run_status:
            run_status_counts[run_status] = run_status_counts.get(run_status, 0) + 1
        if trial.user_attrs.get("cached"):
            cached_count += 1

    best = best_complete_trial(study)
    best_summary = None
    if best is not None:
        best_summary = {
            "trial_number": best.number,
            "objective_value": best.value,
            "params": dict(best.params),
            "config_hash": best.user_attrs.get("config_hash"),
            "result_path": best.user_attrs.get("result_path"),
            "metrics": {
                key.removeprefix("metric_"): value
                for key, value in best.user_attrs.items()
                if key.startswith("metric_")
            },
        }

    return {
        "study_name": study.study_name,
        "experiment_id": ctx.experiment.experiment_id,
        "log_id": ctx.log_ref.log_id,
        "algorithm_name": ctx.algorithm_ref.name,
        "algorithm_id": ctx.algorithm_config.algorithm_id,
        "n_trials_target": ctx.hpo.n_trials,
        "trials_by_state": count_trials_by_state(study),
        "trials_by_run_status": run_status_counts,
        "cached_trials": cached_count,
        "best_trial": best_summary,
        "study_wall_time_seconds": _study_wall_time_seconds(trials),
        "objective_weights": dict(ctx.hpo.objective.weights),
        "failed_trial_value": ctx.hpo.objective.failed_trial_value,
        "sampler": {
            "name": "TPESampler",
            "sampler_seed": ctx.hpo.sampler_seed,
            "n_startup_trials": ctx.hpo.n_startup_trials,
            "multivariate": ctx.hpo.multivariate,
            "group": ctx.hpo.group,
            "constant_liar": ctx.hpo.constant_liar,
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }


def default_summary_path(ctx: StudyContext) -> Path:
    study_name = study_name_for(
        ctx.experiment.experiment_id, ctx.log_ref.log_id, ctx.algorithm_ref.name
    )
    return summary_path_for(
        ctx.hpo.storage_root,
        ctx.experiment.experiment_id,
        ctx.hpo.summary_dirname,
        study_name,
    )


def write_study_summary(
    study: optuna.Study,
    ctx: StudyContext,
    output_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path) if output_path else default_summary_path(ctx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(output_path, build_study_summary(study, ctx))
    return output_path


def _study_wall_time_seconds(trials: list[Any]) -> float | None:
    starts = [trial.datetime_start for trial in trials if trial.datetime_start]
    ends = [trial.datetime_complete for trial in trials if trial.datetime_complete]
    if not starts or not ends:
        return None
    return max(0.0, (max(ends) - min(starts)).total_seconds())
