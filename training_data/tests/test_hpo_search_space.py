import pytest

from process_discovery_cash.hpo.search_space import (
    build_hpo_search_space,
    suggest_trial_params,
)

pytestmark = pytest.mark.legacy_hpo


def _make_study():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=7),
    )


def test_discrete_and_range_dimensions_translate() -> None:
    space = build_hpo_search_space(
        {
            "variant": {"type": "categorical", "values": ["classic", "plusplus"]},
            "dependency_threshold": {"min": 0.0, "max": 1.0, "type": "float", "scale": "linear"},
            "min_act_count": {"min": 1, "max": 100, "type": "integer", "scale": "log"},
        },
        [],
    )
    study = _make_study()

    trial = study.ask()
    params = suggest_trial_params(trial, {"seedless_default": True}, space)

    assert params["seedless_default"] is True
    assert params["variant"] in {"classic", "plusplus"}
    assert 0.0 <= params["dependency_threshold"] <= 1.0
    assert isinstance(params["min_act_count"], int)
    assert 1 <= params["min_act_count"] <= 100


def test_fixed_dimension_is_not_suggested() -> None:
    space = build_hpo_search_space(
        {
            "variant": {"type": "categorical", "values": ["im"]},
            "noise": {"min": 0.0, "max": 0.5, "type": "float"},
        },
        [],
    )
    study = _make_study()

    trial = study.ask()
    params = suggest_trial_params(trial, {}, space)

    assert params["variant"] == "im"
    assert "variant" not in trial.params
    assert "noise" in trial.params


def test_conditional_dimension_only_suggested_when_parent_matches() -> None:
    space = build_hpo_search_space(
        {"variant": {"type": "categorical", "values": ["im", "imf"]}},
        [
            {
                "when": {"variant": "imf"},
                "params": {"noise_threshold": {"min": 0.0, "max": 0.6, "type": "float"}},
            }
        ],
    )
    study = _make_study()

    seen_variants = set()
    for _ in range(20):
        trial = study.ask()
        params = suggest_trial_params(trial, {}, space)
        seen_variants.add(params["variant"])
        if params["variant"] == "imf":
            assert 0.0 <= params["noise_threshold"] <= 0.6
        else:
            assert "noise_threshold" not in params
    assert seen_variants == {"im", "imf"}


def test_fully_fixed_space_raises() -> None:
    with pytest.raises(ValueError, match="no free dimensions"):
        build_hpo_search_space(
            {"variant": {"type": "categorical", "values": ["im"]}},
            [],
        )


def test_conditional_free_dimension_counts_as_free() -> None:
    space = build_hpo_search_space(
        {"variant": {"type": "categorical", "values": ["imf"]}},
        [
            {
                "when": {"variant": "imf"},
                "params": {"noise_threshold": {"min": 0.0, "max": 0.6, "type": "float"}},
            }
        ],
    )

    assert space.conditional


def test_log_scale_with_non_positive_minimum_raises() -> None:
    with pytest.raises(ValueError, match="log scale"):
        build_hpo_search_space(
            {"count": {"min": 0, "max": 10, "type": "integer", "scale": "log"}},
            [],
        )


def test_unsupported_categorical_value_type_raises() -> None:
    with pytest.raises(ValueError, match="categorical values"):
        build_hpo_search_space(
            {"weights": {"type": "categorical", "values": [[1, 2], [3, 4]]}},
            [],
        )


def test_defaults_do_not_override_suggestions() -> None:
    space = build_hpo_search_space(
        {"alpha": {"min": 0.0, "max": 1.0, "type": "float"}},
        [],
    )
    study = _make_study()

    trial = study.ask()
    params = suggest_trial_params(trial, {"alpha": 99.0, "other": 1}, space)

    assert 0.0 <= params["alpha"] <= 1.0
    assert params["other"] == 1
