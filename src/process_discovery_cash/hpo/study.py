"""Optuna study setup shared by all workers of one (log, algorithm) HPO study.

The study lives in a journal file (``JournalStorage`` + ``JournalFileBackend``
with the ``JournalFileOpenLock`` variant, which is the NFS-safe lock), so any
number of worker processes in one Slurm allocation — or across restarts — can
share it without a database server.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import optuna


def _safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "study"


def study_name_for(experiment_id: str, log_id: str, algorithm_name: str) -> str:
    return "__".join(_safe_part(part) for part in (experiment_id, log_id, algorithm_name))


def journal_path_for(storage_root: str | Path, experiment_id: str, study_name: str) -> Path:
    return Path(storage_root) / _safe_part(experiment_id) / f"{study_name}.journal"


def summary_path_for(
    storage_root: str | Path,
    experiment_id: str,
    summary_dirname: str,
    study_name: str,
) -> Path:
    """Summary JSONs live next to the journals, never under the results root —
    ``aggregate_results`` sweeps every ``*.json`` below the results tree and the
    result contract must hold for all of them."""
    return Path(storage_root) / _safe_part(experiment_id) / summary_dirname / f"{study_name}.json"


def worker_sampler_seed(sampler_seed: int, worker_id: int) -> int:
    """Distinct sampler seed per worker so random-startup draws differ."""
    return sampler_seed * 1000 + worker_id


def open_study(
    *,
    study_name: str,
    journal_path: str | Path,
    sampler_seed: int,
    n_startup_trials: int,
    multivariate: bool = True,
    group: bool = True,
    constant_liar: bool = True,
) -> optuna.Study:
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(
        JournalFileBackend(
            str(journal_path),
            lock_obj=JournalFileOpenLock(str(journal_path)),
        )
    )
    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
        n_startup_trials=n_startup_trials,
        multivariate=multivariate,
        group=group,
        constant_liar=constant_liar,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )


def completed_trial_count(study: optuna.Study) -> int:
    from optuna.trial import TrialState

    return sum(
        1 for trial in study.get_trials(deepcopy=False) if trial.state is TrialState.COMPLETE
    )


def count_trials_by_state(study: optuna.Study) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trial in study.get_trials(deepcopy=False):
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return counts


def best_complete_trial(study: optuna.Study) -> Any | None:
    """Best COMPLETE trial or None (``study.best_trial`` raises on empty studies)."""
    from optuna.trial import TrialState

    best = None
    for trial in study.get_trials(deepcopy=False):
        if trial.state is not TrialState.COMPLETE or trial.value is None:
            continue
        if best is None or trial.value > best.value:
            best = trial
    return best
