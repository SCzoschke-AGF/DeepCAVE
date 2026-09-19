import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.plugins.hyperparameter import search_space_state as state
from deepcave.utils.cache import Cache


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
            "objective": 0,
            "mode": "patch",
            "selection": {"depth": {"min": 2, "max": 5}},
        },
    )
    planning_state = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-b",
            "objective": 0,
            "mode": "patch",
            "selection": {"kind": {"choices": ["b"]}},
        },
    )

    assert planning_state["version"] == state.PLANNED_SEARCH_SPACE_STATE_VERSION
    assert planning_state["revision"] == 2
    assert planning_state["runs"]["run-a"]["objective:0"] == {"depth": {"min": 2, "max": 5}}
    assert planning_state["runs"]["run-b"]["objective:0"] == {"kind": {"choices": ["b"]}}


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
        "version": state.PLANNED_SEARCH_SPACE_STATE_VERSION,
        "revision": 2,
        "runs": {
            "run-a": {"objective:0": {"depth": {"min": 2, "max": 5}}},
            "run-b": {"objective:0": {"kind": {"choices": ["b"]}}},
        },
        "event_sequences": {},
        "last_change": None,
    }
    reset = state.initial_selection(_metadata())

    updated = state.apply_planning_intent(
        planning_state,
        {"run_id": "run-a", "objective": 0, "mode": "replace", "selection": reset},
    )

    assert updated["runs"]["run-a"]["objective:0"] == reset
    assert updated["runs"]["run-b"] == planning_state["runs"]["run-b"]


def test_unknown_state_version_is_not_reused():
    metadata = _metadata()
    stale = {
        "version": 0,
        "runs": {"run-a": {"depth": {"min": 3, "max": 4}}},
    }

    selection = state.selection_from_state(stale, "run-a", 0, metadata)

    assert selection["depth"] == {"min": 1, "max": 9}


def test_identical_and_invalid_intents_are_true_noops():
    planning_state = state.apply_planning_intent(
        state.empty_planning_state(),
        {
            "run_id": "run-a",
            "objective": 0,
            "source": "hpo",
            "mode": "patch",
            "selection": {"depth": {"min": 2, "max": 5}},
        },
    )

    repeated = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-a",
            "objective": 0,
            "source": "hpo",
            "mode": "patch",
            "selection": {"depth": {"min": 2, "max": 5}},
        },
    )
    invalid = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-a",
            "objective": 0,
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
            "objective": 0,
            "source": "hpo",
            "sequence": 5,
            "selection": {"depth": {"min": 3, "max": 6}},
        },
    )
    planning_state = state.apply_planning_intent(
        planning_state,
        {
            "run_id": "run-a",
            "objective": 0,
            "source": "hpo",
            "sequence": 4,
            "selection": {
                "depth": {"min": 1, "max": 2},
                "kind": {"choices": ["b"]},
            },
        },
    )

    assert planning_state["runs"]["run-a"]["objective:0"] == {
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


def test_objectives_are_isolated_within_one_run():
    planning_state = state.empty_planning_state()
    for objective, minimum in ((0, 2), (1, 4)):
        planning_state = state.apply_planning_intent(
            planning_state,
            {
                "run_id": "run-a",
                "objective": objective,
                "selection": {"depth": {"min": minimum, "max": 7}},
            },
        )

    assert state.selection_from_state(planning_state, "run-a", 0, _metadata())["depth"] == {
        "min": 2,
        "max": 7,
    }
    assert state.selection_from_state(planning_state, "run-a", 1, _metadata())["depth"] == {
        "min": 4,
        "max": 7,
    }


def test_latest_changed_scope_is_persisted_and_restored(tmp_path):
    cache = Cache(tmp_path / "meta.json")
    planning_state = state.apply_planning_intent(
        state.empty_planning_state(),
        {
            "run_id": "run-a",
            "objective": "user_attr:score",
            "selection": {"depth": {"min": 3, "max": 8}},
        },
    )

    revision = state.persist_planning_state_change(cache, planning_state)
    restored = state.planning_state_from_cache(cache)

    assert revision == 1
    assert restored["runs"] == {"run-a": {"user_attr:score": {"depth": {"min": 3, "max": 8}}}}
    assert restored["last_change"] is None


def test_legacy_manually_saved_space_is_hydrated(tmp_path):
    cache = Cache(tmp_path / "meta.json")
    cache.set(
        state.LEGACY_SAVED_SEARCH_SPACES_KEY,
        "run-a",
        "objective:0",
        value={"depth": [2, 8], "kind": ["b"]},
    )

    restored = state.planning_state_from_cache(cache)
    selection = state.selection_from_state(restored, "run-a", 0, _metadata())

    assert selection["depth"] == {"min": 2, "max": 8}
    assert selection["kind"] == {"choices": ["b"]}


def test_autosave_overwrites_only_the_changed_run_objective_scope(tmp_path):
    cache = Cache(tmp_path / "meta.json")
    planning_state = state.empty_planning_state()
    for run_id, objective, minimum in (
        ("run-a", 0, 2),
        ("run-a", 1, 4),
        ("run-b", 0, 3),
        ("run-a", 0, 5),
    ):
        planning_state = state.apply_planning_intent(
            planning_state,
            {
                "run_id": run_id,
                "objective": objective,
                "selection": {"depth": {"min": minimum, "max": 8}},
            },
        )
        state.persist_planning_state_change(cache, planning_state)

    assert cache.get(state.PLANNED_SEARCH_SPACE_CACHE_KEY) == {
        "run-a": {
            "objective:0": {"depth": {"min": 5, "max": 8}},
            "objective:1": {"depth": {"min": 4, "max": 8}},
        },
        "run-b": {"objective:0": {"depth": {"min": 3, "max": 8}}},
    }
