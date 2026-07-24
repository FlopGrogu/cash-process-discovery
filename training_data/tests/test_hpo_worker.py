from pathlib import Path
from types import SimpleNamespace

import pytest

from process_discovery_cash.config.schema import HpoConfig
from process_discovery_cash.hpo.objective import ObjectiveOutcome
from process_discovery_cash.hpo.search_space import build_hpo_search_space
from process_discovery_cash.hpo.study import (
    best_complete_trial,
    completed_trial_count,
    count_trials_by_state,
    journal_path_for,
    open_study,
    study_name_for,
    worker_sampler_seed,
)
from process_discovery_cash.hpo.trial_runner import TrialOutcome
from process_discovery_cash.hpo.worker import run_hpo_worker

pytestmark = pytest.mark.legacy_hpo


def _fake_ctx(n_trials: int = 5, per_trial_walltime: float = 10.0) -> SimpleNamespace:
    hpo = HpoConfig(
        n_trials=n_trials,
        n_startup_trials=2,
        per_trial_walltime_seconds=per_trial_walltime,
    )
    space = build_hpo_search_space({"alpha": {"min": 0.0, "max": 1.0, "type": "float"}}, [])
    return SimpleNamespace(
        hpo=hpo,
        space=space,
        default_params={"base": 1},
        finalize_trial_params=lambda params: dict(params),
    )


def _open_tmp_study(tmp_path: Path, worker_id: int = 0):
    study_name = study_name_for("exp", "log", "algo")
    journal = journal_path_for(tmp_path / "runs", "exp", study_name)
    return open_study(
        study_name=study_name,
        journal_path=journal,
        sampler_seed=worker_sampler_seed(42, worker_id),
        n_startup_trials=2,
    )


def _stub_run_trial(counter: dict, *, cached: bool = False, value: float = 0.5):
    def _run(ctx, params, **kwargs):
        counter["count"] = counter.get("count", 0) + 1
        return TrialOutcome(
            config_hash=f"hash{counter['count']}",
            objective=ObjectiveOutcome(
                value=value, run_status="success", metric_values={"fitness": value}
            ),
            cached=cached,
            result_path=Path("results/fake.json"),
        )

    return _run


def test_worker_stops_at_n_trials_and_resumes(tmp_path: Path) -> None:
    ctx = _fake_ctx(n_trials=5)
    study = _open_tmp_study(tmp_path)
    counter: dict = {}

    stats = run_hpo_worker(
        ctx=ctx,
        study=study,
        worker_walltime_seconds=10_000,
        safety_margin_seconds=0,
        run_trial_fn=_stub_run_trial(counter),
    )

    assert stats.stopped_reason == "n_trials_reached"
    assert stats.told_complete == 5
    assert completed_trial_count(study) == 5

    resumed = _open_tmp_study(tmp_path, worker_id=1)
    resumed_stats = run_hpo_worker(
        ctx=ctx,
        study=resumed,
        worker_id=1,
        worker_walltime_seconds=10_000,
        safety_margin_seconds=0,
        run_trial_fn=_stub_run_trial({}),
    )

    assert resumed_stats.stopped_reason == "n_trials_reached"
    assert resumed_stats.told_complete == 0
    assert completed_trial_count(resumed) == 5


def test_worker_stops_when_walltime_cannot_fit_a_trial(tmp_path: Path) -> None:
    ctx = _fake_ctx(per_trial_walltime=100.0)
    study = _open_tmp_study(tmp_path)
    counter: dict = {}

    stats = run_hpo_worker(
        ctx=ctx,
        study=study,
        worker_walltime_seconds=50.0,
        safety_margin_seconds=0.0,
        run_trial_fn=_stub_run_trial(counter),
        monotonic=lambda: 0.0,
    )

    assert stats.stopped_reason == "walltime"
    assert counter.get("count", 0) == 0


def test_worker_counts_cached_trials(tmp_path: Path) -> None:
    ctx = _fake_ctx(n_trials=3)
    study = _open_tmp_study(tmp_path)
    counter: dict = {}

    stats = run_hpo_worker(
        ctx=ctx,
        study=study,
        worker_walltime_seconds=10_000,
        safety_margin_seconds=0,
        run_trial_fn=_stub_run_trial(counter, cached=True),
    )

    assert stats.cached == 3
    assert stats.executed == 0
    assert stats.told_complete == 3


def test_worker_tells_fail_on_study_deadline_kill(tmp_path: Path) -> None:
    ctx = _fake_ctx(n_trials=10)
    study = _open_tmp_study(tmp_path)

    def _killed(ctx_arg, params, **kwargs):
        return TrialOutcome(
            config_hash="hash",
            objective=ObjectiveOutcome(value=0.0, run_status="study_walltime"),
            cached=False,
            result_path=Path("results/fake.json"),
            killed_at_study_deadline=True,
        )

    stats = run_hpo_worker(
        ctx=ctx,
        study=study,
        worker_walltime_seconds=10_000,
        safety_margin_seconds=0,
        run_trial_fn=_killed,
    )

    assert stats.stopped_reason == "walltime"
    assert stats.told_failed == 1
    assert stats.told_complete == 0
    assert count_trials_by_state(study) == {"FAIL": 1}


def test_worker_survives_trial_errors_then_stops(tmp_path: Path) -> None:
    ctx = _fake_ctx(n_trials=10)
    study = _open_tmp_study(tmp_path)

    def _boom(ctx_arg, params, **kwargs):
        raise RuntimeError("backend exploded")

    stats = run_hpo_worker(
        ctx=ctx,
        study=study,
        worker_walltime_seconds=10_000,
        safety_margin_seconds=0,
        run_trial_fn=_boom,
    )

    assert stats.stopped_reason == "consecutive_errors"
    assert stats.told_failed == 3
    failed = [t for t in study.get_trials(deepcopy=False)]
    assert all("backend exploded" in t.user_attrs["worker_error"] for t in failed)


def test_worker_records_user_attrs(tmp_path: Path) -> None:
    ctx = _fake_ctx(n_trials=1)
    study = _open_tmp_study(tmp_path)

    stats = run_hpo_worker(
        ctx=ctx,
        study=study,
        worker_id=7,
        worker_walltime_seconds=10_000,
        safety_margin_seconds=0,
        run_trial_fn=_stub_run_trial({}, value=0.9),
    )

    assert stats.told_complete == 1
    trial = study.get_trials(deepcopy=False)[0]
    assert trial.user_attrs["config_hash"] == "hash1"
    assert trial.user_attrs["run_status"] == "success"
    assert trial.user_attrs["cached"] is False
    assert trial.user_attrs["worker_id"] == 7
    assert trial.user_attrs["metric_fitness"] == 0.9


def test_study_helpers(tmp_path: Path) -> None:
    assert study_name_for("exp v1", "log/one", "algo") == "exp_v1__log_one__algo"
    journal = journal_path_for("runs/hpo", "exp v1", "study")
    assert journal == Path("runs/hpo/exp_v1/study.journal")
    assert worker_sampler_seed(42, 3) == 42003

    study = _open_tmp_study(tmp_path)
    assert best_complete_trial(study) is None
    assert completed_trial_count(study) == 0
