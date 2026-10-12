from types import SimpleNamespace

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, UniformFloatHyperparameter
from dash import dcc

from deepcave import config
from deepcave.plugins.hyperparameter import configuration_neighborhood as neighborhood
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial


def _data():
    return {
        "run_id": "study",
        "objective_id": 0,
        "objective_name": "Accuracy",
        "optimize": "upper",
        "hyperparameters": ["fixed", "x"],
        "constant_indices": [0],
        "records": [
            {
                "trial_id": 4,
                "config_id": 0,
                "start_time": 0.0,
                "end_time": 1.0,
                "objective": 0.7,
                "objective_rank": 3.0,
                "configuration": {"x": 0.0, "fixed": 1},
            },
            {
                "trial_id": 9,
                "config_id": 1,
                "start_time": 1.0,
                "end_time": 2.0,
                "objective": 0.9,
                "objective_rank": 1.0,
                "configuration": {"x": 0.5, "fixed": 1},
            },
            {
                "trial_id": 12,
                "config_id": 2,
                "start_time": 2.0,
                "end_time": 3.0,
                "objective": 0.8,
                "objective_rank": 2.0,
                "configuration": {"x": 1.0, "fixed": 1},
            },
        ],
        "matrix": [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
    }


def _component_by_id(component, component_id):
    return next(
        candidate
        for candidate in neighborhood._walk_components(component)
        if getattr(candidate, "id", None) == component_id
    )


def test_plugin_is_registered_for_configuration_analyses():
    plugin_ids = [plugin.id for plugin in config.PLUGINS["Configuration Analyses"]]
    assert "configuration_neighborhood" in plugin_ids


def test_both_views_and_separate_state_stores_stay_mounted():
    output = neighborhood.ConfigurationNeighborhood.get_output_layout(
        lambda name, *_args: name
    )
    store_ids = {
        component.id
        for component in neighborhood._walk_components(output)
        if isinstance(component, dcc.Store)
    }
    page = neighborhood._page_layout(_data())
    selection = _component_by_id(page, neighborhood.SELECTION_VIEW_ID)
    similarity = _component_by_id(page, neighborhood.SIMILARITY_VIEW_ID)

    assert neighborhood.SELECTED_TRIAL_STORE_ID in store_ids
    assert neighborhood.VIEW_MODE_STORE_ID in store_ids
    assert neighborhood.DISTANCE_STORE_ID in store_ids
    assert selection.style == {"display": "block"}
    assert similarity.style == {"display": "none"}
    assert neighborhood.view_styles("similarity") == (
        {"display": "block"},
        {"display": "block"},
    )


def test_history_and_similarity_points_carry_true_trial_ids():
    data = _data()
    history = neighborhood.history_figure(data)
    configspace = ConfigurationSpace()
    configspace.add(
        [UniformFloatHyperparameter("x", lower=0.0, upper=1.0), Constant("fixed", 1)]
    )
    payload = neighborhood.distance_payload(data, 9, configspace)
    similarity = neighborhood.similarity_figure(payload)

    assert list(history.data[0].customdata) == [4, 9, 12]
    assert neighborhood._trial_id_from_click(
        {"points": [{"pointIndex": 0, "x": 999, "customdata": 9}]}
    ) == 9
    assert list(similarity.data[0].customdata) == [4, 12]
    assert list(similarity.data[1].customdata) == [9]


def test_quick_selections_are_deterministic_and_direction_aware():
    records = _data()["records"]

    assert neighborhood.quick_selection_ids(records, "upper") == {
        "best": 9,
        "median": 12,
        "worst": 4,
        "latest": 12,
    }
    assert neighborhood.quick_selection_ids(records, "lower")["best"] == 4
    assert [
        record["trial_id"]
        for record in neighborhood.top_five_percent_records(records, "upper")
    ] == [9]


def test_all_selection_paths_share_the_same_state_transition():
    data = _data()
    common = (data, None, "select", False, None)

    history = neighborhood.reduce_selection_state(
        neighborhood.HISTORY_GRAPH_ID,
        *common,
        history_click={"points": [{"customdata": 9}]},
    )
    table = neighborhood.reduce_selection_state(
        neighborhood.TRIAL_TABLE_ID,
        *common,
        active_cell={"row_id": 9},
    )
    quick = neighborhood.reduce_selection_state(
        neighborhood.QUICK_BEST_ID,
        *common,
    )

    assert history[:2] == (9, "similarity")
    assert table[:2] == (9, "similarity")
    assert quick[:2] == (9, "similarity")


def test_select_new_preserves_reference_and_run_change_resets_it():
    data = _data()
    context = {"run_id": "study", "objective_id": 0}
    back = neighborhood.reduce_selection_state(
        neighborhood.SELECT_NEW_TRIAL_ID,
        data,
        9,
        "similarity",
        False,
        context,
    )
    changed_data = dict(data, run_id="other-study")
    reset = neighborhood.reduce_selection_state(
        neighborhood.ANALYSIS_DATA_ID,
        changed_data,
        9,
        "similarity",
        True,
        context,
    )

    assert back[:2] == (9, "select")
    assert reset[:3] == (None, "select", False)


def test_neighbors_include_reference_before_k_other_trials():
    data = _data()
    configspace = ConfigurationSpace()
    configspace.add(
        [UniformFloatHyperparameter("x", lower=0.0, upper=1.0), Constant("fixed", 1)]
    )
    payload = neighborhood.distance_payload(data, 9, configspace)

    nearest = neighborhood.nearest_records(payload, 1)
    assert [record["trial_id"] for record in nearest] == [9, 4]
    assert nearest[0]["distance"] == 0
    figure = neighborhood.parameter_difference_figure(payload, 1)
    assert list(figure.data[0].y) == ["x"]
    assert list(figure.data[0].x) == [0.5]


def test_process_keeps_real_optuna_trial_numbers_and_reuses_normalization(monkeypatch):
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0)})
    objective = Objective("Accuracy", optimize="upper")
    configurations = {0: {"x": 0.1}, 1: {"x": 0.9}}
    run = SimpleNamespace(
        id="study",
        latest_change=1.0,
        prefix="Optuna",
        configspace=configspace,
        configs=configurations,
        history=[
            Trial(0, 0, -1, [0.2], 0, 0, 1, Status.SUCCESS, {"optuna_trial_number": 8}),
            Trial(1, 0, -1, [0.8], 0, 1, 2, Status.SUCCESS, {"optuna_trial_number": 21}),
        ],
        get_config=lambda config_id: Configuration(configspace, values=configurations[config_id]),
        get_objective=lambda _objective_id: objective,
    )
    calls = 0
    original = neighborhood.normalized_configuration_matrix

    def count_normalization(configs, space):
        nonlocal calls
        calls += 1
        return original(configs, space)

    monkeypatch.setattr(neighborhood, "normalized_configuration_matrix", count_normalization)
    neighborhood.clear_configuration_neighborhood_cache()
    first = neighborhood.ConfigurationNeighborhood.process(run, {"objective_id": 0})
    second = neighborhood.ConfigurationNeighborhood.process(run, {"objective_id": 0})

    assert [record["trial_id"] for record in first["records"]] == [8, 21]
    assert np.asarray(first["matrix"]).shape == (2, 1)
    assert second["matrix"] == first["matrix"]
    assert calls == 1


def test_compatibility_is_limited_to_non_empty_optuna_studies():
    compatible = SimpleNamespace(
        prefix="Optuna", empty=lambda: False, configspace={"x": object()}
    )
    group = SimpleNamespace(prefix="group", empty=lambda: False, configspace={"x": object()})

    assert neighborhood.ConfigurationNeighborhood.check_run_compatibility(compatible)
    assert not neighborhood.ConfigurationNeighborhood.check_run_compatibility(group)
