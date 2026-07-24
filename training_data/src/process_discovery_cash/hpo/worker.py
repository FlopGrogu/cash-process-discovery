"""Ask→run→tell worker loop for one HPO study.

Several workers (typically ``SLURM_CPUS_PER_TASK`` per allocation) run this
loop concurrently against the same journal-backed study. Each iteration asks
the sampler for a configuration, evaluates it (or answers from an existing
result file), and tells the objective back. Workers stop when the study has
enough completed trials or when the remaining walltime no longer fits a full
trial (mirroring the dynamic worker's claim deadline).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from process_discovery_cash.hpo.search_space import suggest_trial_params
from process_discovery_cash.hpo.study import completed_trial_count
from process_discovery_cash.hpo.trial_runner import StudyContext, TrialOutcome, run_trial
from process_discovery_cash.utils.paths import portable_project_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    import optuna

_MAX_CONSECUTIVE_ERRORS = 3


@dataclass
class HpoWorkerStats:
    executed: int = 0
    cached: int = 0
    told_complete: int = 0
    told_failed: int = 0
    stopped_reason: str = ""
    run_status_counts: dict[str, int] = field(default_factory=dict)


def run_hpo_worker(
    *,
    ctx: StudyContext,
    study: optuna.Study,
    worker_id: int = 0,
    worker_walltime_seconds: float,
    safety_margin_seconds: float,
    max_trials_per_worker: int | None = None,
    isolate_runs: bool = True,
    child_memory_limit_mb: float | None = None,
    scratch_dir: str | Path | None = None,
    run_trial_fn: Callable[..., TrialOutcome] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> HpoWorkerStats:
    from optuna.trial import TrialState

    started = monotonic()
    claim_deadline = started + worker_walltime_seconds - safety_margin_seconds
    per_trial_walltime = ctx.hpo.per_trial_walltime_seconds
    run_trial_fn = run_trial_fn or run_trial
    stats = HpoWorkerStats()
    consecutive_errors = 0

    while True:
        if completed_trial_count(study) >= ctx.hpo.n_trials:
            stats.stopped_reason = "n_trials_reached"
            break
        attempted = stats.executed + stats.cached
        if max_trials_per_worker is not None and attempted >= max_trials_per_worker:
            stats.stopped_reason = "max_trials_per_worker"
            break
        now = monotonic()
        if now + per_trial_walltime > claim_deadline:
            stats.stopped_reason = "walltime"
            break

        trial = study.ask()
        try:
            suggested = suggest_trial_params(trial, ctx.default_params, ctx.space)
            params = ctx.finalize_trial_params(suggested)
            trial_deadline = now + per_trial_walltime
            outcome = run_trial_fn(
                ctx,
                params,
                deadline_monotonic=min(claim_deadline, trial_deadline),
                deadline_is_study_deadline=claim_deadline <= trial_deadline,
                memory_limit_mb=child_memory_limit_mb,
                isolate=isolate_runs,
                scratch_dir=scratch_dir,
                monotonic=monotonic,
            )
            _set_trial_attrs(trial, outcome, worker_id)
            if outcome.killed_at_study_deadline:
                study.tell(trial, state=TrialState.FAIL)
                stats.told_failed += 1
                stats.stopped_reason = "walltime"
                _progress("study_deadline_kill", trial.number, outcome, stats, started, monotonic)
                break
            study.tell(trial, outcome.objective.value)
            stats.told_complete += 1
            if outcome.cached:
                stats.cached += 1
            else:
                stats.executed += 1
            run_status = outcome.objective.run_status
            stats.run_status_counts[run_status] = stats.run_status_counts.get(run_status, 0) + 1
            consecutive_errors = 0
            _progress("trial_told", trial.number, outcome, stats, started, monotonic)
        except Exception as exc:  # one bad trial must not kill the worker
            try:
                trial.set_user_attr("worker_error", f"{type(exc).__name__}: {exc}")
                study.tell(trial, state=TrialState.FAIL)
            except Exception:
                pass
            stats.told_failed += 1
            consecutive_errors += 1
            _progress(
                "trial_error",
                trial.number,
                None,
                stats,
                started,
                monotonic,
                detail=f"{type(exc).__name__}: {exc}",
            )
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                stats.stopped_reason = "consecutive_errors"
                break

    _progress("worker_stop", None, None, stats, started, monotonic, detail=stats.stopped_reason)
    return stats


def _set_trial_attrs(trial: Any, outcome: TrialOutcome, worker_id: int) -> None:
    trial.set_user_attr("config_hash", outcome.config_hash)
    trial.set_user_attr("run_status", outcome.objective.run_status)
    trial.set_user_attr("cached", outcome.cached)
    trial.set_user_attr("result_path", portable_project_path(outcome.result_path))
    trial.set_user_attr("worker_id", worker_id)
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        trial.set_user_attr("slurm_job_id", slurm_job_id)
    for name, value in outcome.objective.metric_values.items():
        trial.set_user_attr(f"metric_{name}", value)


def _progress(
    event: str,
    trial_number: int | None,
    outcome: TrialOutcome | None,
    stats: HpoWorkerStats,
    started: float,
    monotonic: Callable[[], float],
    *,
    detail: str | None = None,
) -> None:
    fields = [
        f"event={event}",
        f"told_complete={stats.told_complete}",
        f"told_failed={stats.told_failed}",
        f"cached={stats.cached}",
        f"executed={stats.executed}",
        f"elapsed_seconds={max(0.0, monotonic() - started):.3f}",
    ]
    if trial_number is not None:
        fields.insert(1, f"trial={trial_number}")
    if outcome is not None:
        fields.insert(2, f"objective={outcome.objective.value:.6f}")
        fields.insert(3, f"run_status={outcome.objective.run_status}")
    if detail:
        fields.append(f"detail={json.dumps(detail)}")
    print(" ".join(fields), flush=True)
