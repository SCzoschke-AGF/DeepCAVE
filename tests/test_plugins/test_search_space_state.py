import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.plugins.hyperparameter import search_space_state as state


def _metadata():
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("rate", lower=0.01, upper=1.0, log=True),
            UniformIntegerHyperparameter("depth", lower=1, upper=9),
            CategoricalHyperparameter("kind", choices=["a", "b", "c"]),
        ]
    )
    return state.configspace_metadata(configspace, list(configspace.keys()))


def test_planning_intents_are_versioned_and_isolated_by_run():
    planning_state = state.empty_planning_state()
    planning_state = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-a",
            "mode": "patch",
            "selection": {"depth": {"min": 2, "max": 5}},
        },
    )
    planning_state = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-b",
            "mode": "patch",
            "selection": {"kind": {"choices": ["b"]}},
        },
    )

    assert planning_state["version"] == state.PLANNED_SEARCH_SPACE_STATE_VERSION
    assert planning_state["revision"] == 2
    assert planning_state["runs"]["run-a"] == {"depth": {"min": 2, "max": 5}}
    assert planning_state["runs"]["run-b"] == {"kind": {"choices": ["b"]}}


def test_global_store_uses_reload_scoped_browser_memory():
    store = state.planning_state_store()

    assert store.id == state.PLANNED_SEARCH_SPACE_STORE_ID
    assert store.storage_type == "memory"
    assert store.data == state.empty_planning_state()


def test_reconcile_preserves_valid_extensions_and_repairs_stale_values():
    metadata = _metadata()
    stored = {
        "rate": {"min": 0.001, "max": 2.0},
        "depth": {"min": "invalid", "max": 7},
        "kind": {"choices": ["c", "removed"]},
        "obsolete": {"min": 0, "max": 1},
    }

    selection = state.reconcile_selection(metadata, stored)

    assert selection["rate"] == {"min": pytest.approx(0.001), "max": pytest.approx(2.0)}
    assert selection["depth"] == {"min": 1, "max": 9}
    assert selection["kind"] == {"choices": ["c"]}
    assert "obsolete" not in selection


def test_replace_resets_only_the_target_run():
    planning_state = {
        "version": 1,
        "revision": 2,
        "runs": {
            "run-a": {"depth": {"min": 2, "max": 5}},
            "run-b": {"kind": {"choices": ["b"]}},
        },
    }
    reset = state.initial_selection(_metadata())

    updated = state.apply_planning_intent(
        planning_state,
        {"run_id": "run-a", "mode": "replace", "selection": reset},
    )

    assert updated["runs"]["run-a"] == reset
    assert updated["runs"]["run-b"] == planning_state["runs"]["run-b"]


def test_unknown_state_version_is_not_reused():
    metadata = _metadata()
    stale = {
        "version": 0,
        "runs": {"run-a": {"depth": {"min": 3, "max": 4}}},
    }

    selection = state.selection_from_state(stale, "run-a", metadata)

    assert selection["depth"] == {"min": 1, "max": 9}


def test_identical_and_invalid_intents_are_true_noops():
    planning_state = state.apply_planning_intent(
        state.empty_planning_state(),
        {
            "run_id": "run-a",
            "source": "hpo",
            "mode": "patch",
            "selection": {"depth": {"min": 2, "max": 5}},
        },
    )

    repeated = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-a",
            "source": "hpo",
            "mode": "patch",
            "selection": {"depth": {"min": 2, "max": 5}},
        },
    )
    invalid = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-a",
            "mode": "patch",
            "selection": {"depth": {"min": float("nan"), "max": 5}},
        },
    )

    assert repeated is planning_state
    assert invalid is planning_state
    assert repeated["revision"] == 1


def test_sequences_reject_stale_same_parameter_but_preserve_other_patches():
    planning_state = state.empty_planning_state()
    planning_state = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-a",
            "source": "hpo",
            "sequence": 5,
            "selection": {"depth": {"min": 3, "max": 6}},
        },
    )
    planning_state = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-a",
            "source": "hpo",
            "sequence": 4,
            "selection": {
                "depth": {"min": 1, "max": 2},
                "kind": {"choices": ["b"]},
            },
        },
    )

    assert planning_state["runs"]["run-a"] == {
        "depth": {"min": 3, "max": 6},
        "kind": {"choices": ["b"]},
    }
    assert planning_state["revision"] == 2


def test_reconciliation_rejects_nonfinite_ranges_and_preserves_empty_categories():
    selection = state.reconcile_selection(
        _metadata(),
        {
            "rate": {"min": float("-inf"), "max": 1.0},
            "kind": {"choices": []},
        },
    )

    assert selection["rate"] == {"min": 0.01, "max": 1.0}
    assert selection["kind"] == {"choices": []}
