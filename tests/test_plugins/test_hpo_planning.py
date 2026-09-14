from types import SimpleNamespace

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dash import dcc, html

from deepcave import config
from deepcave.plugins.hyperparameter import hpo_planning as planning
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial


def test_plugin_is_registered_directly_after_configuration_cube():
    plugin_ids = [plugin.id for plugin in config.PLUGINS["Hyperparameter Analysis"]]

    configuration_cube_index = plugin_ids.index("ccube")
    assert plugin_ids[configuration_cube_index + 1] == "hpo_planning"
    assert planning.HPOPlanning.name == "HPO Planner"


def test_planner_cache_key_is_schema_versioned():
    planner = object.__new__(planning.HPOPlanning)
    inputs = {"objective_id": {"value": 0}}

    legacy_key = super(planning.HPOPlanning, planner)._dict_as_key(inputs)

    assert planner._dict_as_key(inputs) == f"{legacy_key}-v{planning.CACHE_SCHEMA_VERSION}"


def _walk_components(component):
    """Yield a Dash component and all descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk_components(child)


def test_toolbar_contains_only_local_planning_controls():
    registered = []
    (toolbar,) = planning.HPOPlanning.get_input_layout(
        lambda name, *_args, **_kwargs: registered.append(name) or name
    )

    assert registered == [
        "objective_direction",
        "elite_proportion_1",
        "elite_proportion_2",
        "elite_proportion_3",
    ]
    assert "analysis-toolbar" in toolbar.className
    assert not any(isinstance(component, html.Button) for component in _walk_components(toolbar))
    show_top_n = next(
        component
        for component in _walk_components(toolbar)
        if isinstance(component, dbc.Input) and component.id == planning.SHOW_TOP_N_ID
    )
    assert show_top_n.value == planning.DEFAULT_PARAMETER_TOP_TRIAL_COUNT == 5
    assert show_top_n.min == 1
    assert show_top_n.max == planning.TOP_TRIAL_LIMIT
    assert show_top_n.step == 1
    assert show_top_n.debounce is True
    hidden_direction_inputs = [
        component
        for component in _walk_components(toolbar)
        if getattr(component, "id", None) == "objective_direction"
    ]
    assert len(hidden_direction_inputs) == 1
    assert hidden_direction_inputs[0].style == {"display": "none"}


def test_elite_proportions_are_validated_and_sorted():
    assert planning._validate_elite_proportions([0.01, 0.10, 0.05]) == [0.10, 0.05, 0.01]

    with pytest.raises(ValueError, match="different"):
        planning._validate_elite_proportions([0.10, 0.10, 0.01])
    with pytest.raises(ValueError, match="between"):
        planning._validate_elite_proportions([0.10, 0.05, 0.0001])


def test_numeric_statistics_use_linear_or_logarithmic_observed_ranges():
    all_values = pd.Series([1.0, 10.0, 100.0])
    elite_values = pd.Series([10.0])
    linear = UniformFloatHyperparameter("linear", lower=1.0, upper=100.0)
    logarithmic = UniformFloatHyperparameter("logarithmic", lower=1.0, upper=100.0, log=True)

    linear_statistics = planning._feature_statistics(all_values, elite_values, linear)
    log_statistics = planning._feature_statistics(all_values, elite_values, logarithmic)

    assert linear_statistics["median_normalized"] == pytest.approx(9 / 99)
    assert log_statistics["median_normalized"] == pytest.approx(0.5)


def test_planner_normalization_uses_current_linear_or_logarithmic_space():
    linear = {
        "current_min": 100,
        "current_max": 600,
        "log": False,
        "parameter_type": "integer",
        "step": 50,
    }
    logarithmic = {
        "current_min": 0.001,
        "current_max": 10.0,
        "log": True,
        "parameter_type": "float",
        "step": None,
    }

    assert planning._normalize_value(100, linear) == pytest.approx(0)
    assert planning._normalize_value(600, linear) == pytest.approx(1)
    assert planning._normalize_value(0.0001, logarithmic) == pytest.approx(-0.25)
    assert planning._normalize_value(100, logarithmic) == pytest.approx(1.25)
    assert planning._denormalize_value(1.25, logarithmic) == pytest.approx(100)


def test_top_trial_markers_are_toggled_per_parameter_and_limited_globally():
    numeric = {
        "index": 0,
        "name": "x",
        "parameter_type": "float",
        "current_min": 0.0,
        "current_max": 10.0,
        "log": False,
        "step": None,
        "elite_positions": [],
    }
    categorical = {
        "index": 1,
        "name": "kind",
        "parameter_type": "categorical",
        "choices": ["a", "b", "c"],
        "elite_values": [],
    }
    selection = {
        "x": {"min": 0.0, "max": 10.0},
        "kind": {"choices": ["a", "b", "c"]},
    }
    trials = [
        {
            "rank": 1,
            "trial_number": 11,
            "configuration": {"x": 0.0, "kind": "a"},
        },
        {
            "rank": 2,
            "trial_number": 12,
            "configuration": {"x": 10.0, "kind": "c"},
        },
        {
            "rank": 3,
            "trial_number": 13,
            "configuration": {"x": 5.0, "kind": "b"},
        },
    ]

    numeric_row, categorical_row = planning._planner_rows(
        [numeric, categorical],
        selection,
        trials,
        top_n=2,
        visible_top_trial_indices=[0],
    )
    numeric_components = list(_walk_components(numeric_row))
    categorical_components = list(_walk_components(categorical_row))
    numeric_markers = [
        component
        for component in numeric_components
        if getattr(component, "className", None) == "planner-top-trial-marker"
    ]

    assert [marker.style["left"] for marker in numeric_markers] == [
        "16.666667%",
        "83.333333%",
    ]
    assert [marker.to_plotly_json()["props"]["data-rank"] for marker in numeric_markers] == [
        "1",
        "2",
    ]
    assert not any(
        getattr(component, "className", None) == "planner-top-trial-marker"
        for component in categorical_components
    )
    top_trial_buttons = [
        component
        for component in [*numeric_components, *categorical_components]
        if isinstance(getattr(component, "id", None), dict)
        and component.id.get("type") == planning.TOP_TRIAL_BUTTON_TYPE
    ]
    assert [button.active for button in top_trial_buttons] == [True, False]

    (categorical_row,) = planning._planner_rows(
        [categorical],
        {"kind": selection["kind"]},
        trials,
        top_n=1,
        visible_top_trial_indices=[1],
    )
    categorical_markers = [
        component
        for component in _walk_components(categorical_row)
        if getattr(component, "className", None) == "planner-top-trial-marker"
    ]
    assert [marker.style["left"] for marker in categorical_markers] == ["0.000000%"]
    assert planning._parameter_top_trial_count(None) == 5
    assert planning._parameter_top_trial_count(0) == 1
    assert planning._parameter_top_trial_count(1_000) == planning.TOP_TRIAL_LIMIT
    assert planning._updated_top_trial_visibility([numeric, categorical], [], 0) == [0]
    assert planning._updated_top_trial_visibility([numeric, categorical], [0], 1) == [0, 1]
    assert planning._updated_top_trial_visibility([numeric, categorical], [0, 1], 0) == [1]
    assert planning._updated_top_trial_visibility([numeric, categorical], [1, 99], None) == [1]


def test_integer_step_snapping_and_extensions_are_not_clipped():
    parameter = {
        "index": 0,
        "name": "n_estimators",
        "current_min": 100,
        "current_max": 600,
        "log": False,
        "parameter_type": "integer",
        "step": 50,
    }
    selection = {"n_estimators": {"min": 100, "max": 600}}

    moved = planning._updated_selection(
        [parameter],
        selection,
        {"type": "hpo-planner-range", "index": 0},
        [0.674, 1.5],
    )
    metrics = planning._selection_metrics(parameter, moved["n_estimators"])

    assert moved["n_estimators"] == {"min": 450, "max": 850}
    assert metrics["span_ratio"] == pytest.approx(0.8)
    assert metrics["right_extension"] == pytest.approx(0.5)
    assert metrics["extends"] is True


def test_logarithmic_span_metrics_are_calculated_in_sampling_space():
    parameter = {
        "current_min": 0.001,
        "current_max": 10.0,
        "log": True,
        "parameter_type": "float",
        "step": None,
    }

    metrics = planning._selection_metrics(parameter, {"min": 0.0001, "max": 100.0})

    assert metrics["min_normalized"] == pytest.approx(-0.25)
    assert metrics["max_normalized"] == pytest.approx(1.25)
    assert metrics["span_ratio"] == pytest.approx(1.5)
    assert metrics["left_extension"] == pytest.approx(0.25)
    assert metrics["right_extension"] == pytest.approx(0.25)


def test_nonpositive_log_edit_does_not_create_a_second_control_state():
    parameter = {
        "index": 0,
        "name": "gamma",
        "current_min": 0.001,
        "current_max": 10.0,
        "log": True,
        "parameter_type": "float",
        "step": None,
    }
    selection = {"gamma": {"min": 0.001, "max": 10.0}}
    triggered_id = {"type": "hpo-planner-min", "index": 0}

    updated = planning._updated_selection([parameter], selection, triggered_id, 0)

    assert updated == selection
    assert not planning._control_matches_selection([parameter], selection, triggered_id, 0)


def test_categorical_quantiles_use_observed_categories_in_configspace_order():
    hyperparameter = CategoricalHyperparameter("kind", choices=["a", "b", "c", "d"])

    statistics = planning._feature_statistics(
        pd.Series(["a", "b", "c", "d"]),
        pd.Series(["a", "c", "d"]),
        hyperparameter,
    )

    assert statistics["minimum"] == "a"
    assert statistics["q10"] == "a"
    assert statistics["median"] == "c"
    assert statistics["q90"] == "d"
    assert statistics["maximum"] == "d"
    assert statistics["median_normalized"] == pytest.approx(2 / 3)


def test_ordinal_and_constant_features_are_supported():
    ordinal = OrdinalHyperparameter("size", sequence=["small", "medium", "large"])
    constant = Constant("fixed", "value")

    ordinal_statistics = planning._feature_statistics(
        pd.Series(["small", "medium", "large"]),
        pd.Series(["medium", "large"]),
        ordinal,
    )
    constant_statistics = planning._feature_statistics(
        pd.Series(["value", "value"]),
        pd.Series(["value"]),
        constant,
    )

    assert ordinal_statistics["minimum"] == "medium"
    assert ordinal_statistics["maximum"] == "large"
    assert constant_statistics["minimum_normalized"] == pytest.approx(0.5)
    assert constant_statistics["maximum_normalized"] == pytest.approx(0.5)


def test_global_elite_threshold_and_conditional_feature_activity():
    configspace = ConfigurationSpace(
        space={
            "always": (0.0, 10.0),
            "conditional": ["a", "b", "c"],
        }
    )
    trials = [
        {"objective": 10.0, "configuration": {"always": 1.0}},
        {"objective": 9.0, "configuration": {"always": 2.0, "conditional": "a"}},
        {"objective": 9.0, "configuration": {"always": 3.0, "conditional": "b"}},
        {"objective": 1.0, "configuration": {"always": 4.0, "conditional": "c"}},
    ]

    summary = planning._summarize_features(
        trials,
        configspace,
        optimize="upper",
        elite_proportions=[0.5, 0.25],
    )
    lookup = {(row["feature"], row["proportion"]): row for row in summary}

    assert lookup[("always", 0.5)]["threshold"] == pytest.approx(9.0)
    assert lookup[("always", 0.5)]["n"] == 3
    assert lookup[("conditional", 0.5)]["n"] == 2
    assert lookup[("conditional", 0.25)]["n"] == 0
    assert lookup[("conditional", 0.25)]["median"] is None


def test_minimization_uses_lower_objective_quantile():
    configspace = ConfigurationSpace(space={"x": (0.0, 10.0)})
    trials = [
        {"objective": 1.0, "configuration": {"x": 1.0}},
        {"objective": 2.0, "configuration": {"x": 2.0}},
        {"objective": 3.0, "configuration": {"x": 3.0}},
        {"objective": 4.0, "configuration": {"x": 4.0}},
    ]

    summary = planning._summarize_features(trials, configspace, "lower", [0.5])

    assert summary[0]["threshold"] == pytest.approx(2.5)
    assert summary[0]["n"] == 2
    assert summary[0]["minimum"] == pytest.approx(1.0)
    assert summary[0]["maximum"] == pytest.approx(2.0)


def test_parameter_insight_combines_enrichment_and_objective_on_shared_ranges():
    configspace = ConfigurationSpace()
    configspace.add(CategoricalHyperparameter("booster", choices=["tree", "linear"]))
    trials = [
        {"objective": 1.0, "configuration": {"booster": "tree"}},
        {"objective": 2.0, "configuration": {"booster": "tree"}},
        {"objective": 3.0, "configuration": {"booster": "linear"}},
        {"objective": 4.0, "configuration": {"booster": "linear"}},
    ]

    proportions = [0.5, 0.25, 0.1]
    (insight,) = planning._parameter_insights(trials, configspace, "upper", proportions)
    figure = planning._parameter_insight_figure(insight, "Accuracy", 0.5)

    assert insight["labels"] == ["tree", "linear"]
    assert insight["tick_labels"] == ["tree", "linear"]
    assert insight["axis_title"] == "booster"
    assert insight["elite_groups"][0]["enrichment"] == pytest.approx([0.0, 2.0])
    assert insight["median"] == pytest.approx([1.5, 3.5])
    assert insight["q90"] == pytest.approx([1.9, 3.9])
    assert figure.data[0].type == "bar"
    assert figure.data[0].yaxis is None
    assert figure.data[0].marker.color == planning.ELITE_MARKER_COLORS[0]
    assert figure.data[0].marker.opacity == planning.INSIGHT_ELITE_BAR_OPACITY
    assert figure.data[1].name == "q90"
    assert figure.data[1].yaxis == "y2"
    assert figure.data[2].name == "median"
    assert figure.data[2].yaxis == "y2"
    assert figure.data[3].name == "1.0× baseline"
    assert figure.layout.yaxis.title.text == "Elite enrichment"
    assert figure.layout.yaxis2.title.text == "Accuracy"
    assert figure.layout.yaxis2.overlaying == "y"
    assert figure.layout.xaxis.title.text is None
    assert list(figure.layout.xaxis.ticktext) == ["tree", "linear"]
    assert figure.layout.xaxis.tickangle == 0
    assert figure.layout.hovermode is False
    assert all(trace.hoverinfo == "skip" for trace in figure.data)
    top_trial_markers = [
        annotation
        for annotation in figure.layout.annotations
        if annotation.name and annotation.name.startswith("top_trial_marker_")
    ]
    top_trial_ranks = [
        annotation
        for annotation in figure.layout.annotations
        if annotation.name and annotation.name.startswith("top_trial_rank_")
    ]
    assert [marker.x for marker in top_trial_markers] == pytest.approx(
        [0.7375, 0.7625, 0.2375, 0.2625]
    )
    assert [rank.text for rank in top_trial_ranks] == ["1", "2", "3", "4"]
    assert all(marker.font.size == 21 for marker in top_trial_markers)
    assert all(rank.font.size == 9 for rank in top_trial_ranks)
    assert all(getattr(marker, "hovertext", None) is None for marker in top_trial_markers)
    rug_line = next(shape for shape in figure.layout.shapes if shape.x0 == 0 and shape.x1 == 1)
    stems = [shape for shape in figure.layout.shapes if shape.x0 == shape.x1]
    assert len(stems) == 4
    assert rug_line.y0 == pytest.approx(planning.INSIGHT_TOP_TRIAL_RUG_Y)
    assert all(
        stem.y0 == pytest.approx(planning.INSIGHT_TOP_TRIAL_RUG_Y)
        and stem.y1 == pytest.approx(planning.INSIGHT_TOP_TRIAL_STEM_Y)
        for stem in stems
    )
    assert [stem.x0 for stem in stems] == pytest.approx([marker.x for marker in top_trial_markers])
    x_title = next(
        annotation
        for annotation in figure.layout.annotations
        if annotation.name == "insight_x_title"
    )
    assert x_title.text == "booster"
    assert x_title.y == pytest.approx(planning.INSIGHT_X_TITLE_Y)
    assert figure.layout.margin.b == 110

    for proportion, color in zip(proportions, planning.ELITE_MARKER_COLORS):
        group_figure = planning._parameter_insight_figure(insight, "Accuracy", proportion)
        assert group_figure.data[0].marker.color == color
        assert planning._group_label(proportion) in group_figure.data[0].name


def test_numeric_parameter_insight_uses_fixed_bins_and_includes_upper_bound():
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0)})
    trials = [
        {"objective": 0.1, "configuration": {"x": 0.05}},
        {"objective": 0.9, "configuration": {"x": 1.0}},
    ]

    (insight,) = planning._parameter_insights(trials, configspace, "upper", [0.5, 0.25, 0.1])

    assert len(insight["labels"]) == planning.INSIGHT_BIN_COUNT
    assert insight["tick_labels"][0] == "0\u2060–\u20600.1"
    assert insight["tick_labels"][-1] == "0.9\u2060–\u20601"
    assert insight["axis_title"] == "x"
    assert insight["trial_counts"][0] == 1
    assert insight["trial_counts"][-1] == 1
    assert sum(insight["trial_counts"]) == 2


def test_integer_parameter_insight_uses_discrete_values_without_decimal_bins():
    configspace = ConfigurationSpace()
    configspace.add(UniformIntegerHyperparameter("max_depth", lower=3, upper=12))
    trials = [
        {"objective": 0.1, "configuration": {"max_depth": 3}},
        {"objective": 0.9, "configuration": {"max_depth": 12}},
    ]

    (insight,) = planning._parameter_insights(trials, configspace, "upper", [0.5, 0.25, 0.1])
    figure = planning._parameter_insight_figure(insight, "Accuracy", 0.5)

    expected_labels = [str(value) for value in range(3, 13)]
    assert insight["labels"] == expected_labels
    assert insight["tick_labels"] == expected_labels
    assert insight["trial_counts"] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert [trial["bin_index"] for trial in insight["top_trials"]] == [9, 0]
    assert list(figure.layout.xaxis.ticktext) == expected_labels


def test_wide_integer_parameter_insight_groups_only_whole_number_ranges():
    configspace = ConfigurationSpace()
    configspace.add(UniformIntegerHyperparameter("max_depth", lower=1, upper=25))
    trials = [
        {"objective": float(value), "configuration": {"max_depth": value}} for value in range(1, 26)
    ]

    (insight,) = planning._parameter_insights(trials, configspace, "upper", [0.5, 0.25, 0.1])

    assert insight["labels"] == [
        "1–3",
        "4–6",
        "7–9",
        "10–12",
        "13–15",
        "16–17",
        "18–19",
        "20–21",
        "22–23",
        "24–25",
    ]
    assert sum(insight["trial_counts"]) == 25


def test_numeric_parameter_insight_scales_small_tick_labels_in_axis_title():
    configspace = ConfigurationSpace(space={"learning_rate": (0.00455, 0.01005)})
    trials = [
        {"objective": 0.1, "configuration": {"learning_rate": 0.00455}},
        {"objective": 0.9, "configuration": {"learning_rate": 0.01005}},
    ]

    (insight,) = planning._parameter_insights(trials, configspace, "upper", [0.5, 0.25, 0.1])
    figure = planning._parameter_insight_figure(insight, "Accuracy", 0.5)

    assert insight["tick_labels"][0] == "4.55\u2060–\u20605.10"
    assert insight["axis_title"] == "learning_rate (×10⁻³)"
    assert list(figure.layout.xaxis.ticktext) == insight["tick_labels"]
    assert figure.layout.xaxis.tickangle == 0
    x_title = next(
        annotation
        for annotation in figure.layout.annotations
        if annotation.name == "insight_x_title"
    )
    assert x_title.text == insight["axis_title"]


def test_process_keeps_repeated_trials_and_creates_no_interactions():
    objective = Objective("Accuracy", optimize="upper")
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    history = [
        Trial(0, 0, -1, [0.9], 0.0, 0.0, 1.0, Status.SUCCESS, {}),
        Trial(0, 0, -1, [0.9], 0.0, 1.0, 2.0, Status.SUCCESS, {}),
    ]
    run = SimpleNamespace(
        history=history,
        configspace=configspace,
        get_config=lambda _config_id: {"x": 0.25, "y": 0.75},
        get_objective=lambda _objective_id: objective,
    )

    outputs = planning.HPOPlanning.process(
        run,
        {
            "objective_id": 0,
            "elite_proportion_1": 0.5,
            "elite_proportion_2": 0.25,
            "elite_proportion_3": 0.1,
        },
    )

    assert outputs["features"] == ["x", "y"]
    assert outputs["objective_name"] == "Accuracy"
    assert outputs["optimize"] == "upper"
    assert [trial["rank"] for trial in outputs["trials"]] == [1, 2]
    assert [trial["trial_number"] for trial in outputs["trials"]] == [0, 1]
    assert outputs["total_trials"] == 2
    assert [insight["name"] for insight in outputs["insights"]] == ["x", "y"]
    assert len(outputs["summary"]) == 6
    assert all("×" not in row["feature"] for row in outputs["summary"])
    assert max(row["n"] for row in outputs["summary"] if row["feature"] == "x") == 2


def test_process_ranks_in_objective_direction_and_retains_top_100_candidates():
    objective = Objective("Loss", optimize="lower")
    configspace = ConfigurationSpace(space={"x": (0.0, 40.0)})
    history = [
        Trial(
            index,
            0,
            -1,
            [float(34 - index)],
            0.0,
            float(index),
            float(index + 1),
            Status.SUCCESS,
            {"optuna_trial_number": 100 + index},
        )
        for index in range(35)
    ]
    run = SimpleNamespace(
        history=history,
        configspace=configspace,
        get_config=lambda config_id: {"x": float(config_id)},
        get_objective=lambda _objective_id: objective,
    )

    outputs = planning.HPOPlanning.process(
        run,
        {
            "objective_id": 0,
            "elite_proportion_1": 0.5,
            "elite_proportion_2": 0.25,
            "elite_proportion_3": 0.1,
        },
    )

    assert len(outputs["trials"]) == 35
    assert outputs["total_trials"] == 35
    assert outputs["trials"][0]["objective"] == pytest.approx(0)
    assert outputs["trials"][0]["trial_number"] == 134
    assert outputs["trials"][-1]["rank"] == 35


def test_top_trial_coverage_is_capped_at_100_trials():
    assert planning._coverage_trial_count(1) == 1
    assert planning._coverage_trial_count(35) == 10
    assert planning._coverage_trial_count(1_000) == 10
    assert planning._coverage_trial_count(10_000) == 10
    assert planning._coverage_trial_count(35, 1) == 1
    assert planning._coverage_trial_count(35, 100) == 35
    assert planning._coverage_trial_count(1_000, 5) == 5
    assert planning._coverage_trial_count(1_000, 101) == 100
    coverage_data = {
        "total_trials": 1_000,
        "trials": [{"rank": rank} for rank in range(1, 101)],
    }
    assert len(planning._visible_coverage_trials(coverage_data, 1)) == 1
    assert len(planning._visible_coverage_trials(coverage_data, 100)) == 100


def test_spread_is_enabled_and_selected_with_an_attribute():
    assert planning._spread_control_state(None) == (True, False)
    assert planning._spread_control_state("folds") == (False, True)


def test_dependencies_exclude_time_from_objectives():
    objectives = [Objective("Loss"), Objective("Accuracy", optimize="upper"), Objective("Time")]

    class DependencyRun:
        def get_objectives(self):
            return objectives

        def get_objective_id(self, objective):
            return objectives.index(objective)

    inputs = {"objective_id": {"value": None}}
    result = planning.HPOPlanning.load_dependency_inputs(
        object.__new__(planning.HPOPlanning),
        DependencyRun(),
        {},
        inputs,
    )

    assert [option["label"] for option in result["objective_id"]["options"]] == [
        "Objective · Loss",
        "Objective · Accuracy",
    ]


def test_dependencies_add_only_complete_numeric_scalar_user_attrs():
    objective = Objective("Accuracy", optimize="upper")
    history = [
        Trial(
            index,
            0,
            -1,
            [0.8 + index / 100],
            0.0,
            0.0,
            1.0,
            Status.SUCCESS,
            {
                "user_attrs": {
                    "score": 3 - index,
                    "boolean": True,
                    "folds": [0.7, 0.8],
                    **({"incomplete": 0.5} if index == 0 else {}),
                }
            },
        )
        for index in range(2)
    ]
    run = SimpleNamespace(
        history=history,
        get_objectives=lambda: [objective],
        get_objective_id=lambda _objective: 0,
        get_objective=lambda _objective_id: objective,
    )

    result = planning.HPOPlanning.load_dependency_inputs(
        object.__new__(planning.HPOPlanning),
        run,
        {},
        {
            "objective_id": {"value": None},
            "objective_direction": {"value": "lower"},
        },
    )

    assert result["objective_id"]["value"] == 0
    assert [option["value"] for option in result["objective_id"]["options"]] == [
        0,
        "user_attr:score",
    ]
    assert result["objective_direction"]["value"] == "upper"

    preserved_direction = planning.HPOPlanning.load_dependency_inputs(
        object.__new__(planning.HPOPlanning),
        run,
        {"objective_id": {"value": "user_attr:score"}},
        {
            "objective_id": {"value": "user_attr:score"},
            "objective_direction": {"value": "lower"},
        },
    )
    assert preserved_direction["objective_direction"]["value"] == "lower"


def test_user_attr_main_objective_controls_all_direction_aware_outputs():
    objective = Objective("Accuracy", optimize="upper")
    configspace = ConfigurationSpace(space={"x": (0.0, 2.0)})
    history = [
        Trial(
            index,
            0,
            -1,
            [0.5 + index / 10],
            0.0,
            0.0,
            1.0,
            Status.SUCCESS,
            {
                "optuna_trial_number": 10 + index,
                "user_attrs": {
                    "score": score,
                    "folds": [score - 0.1, score + 0.1],
                },
            },
        )
        for index, score in enumerate((3.0, 1.0, 2.0))
    ]
    run = SimpleNamespace(
        history=history,
        configspace=configspace,
        get_config=lambda config_id: {"x": float(config_id)},
        get_objective=lambda _objective_id: objective,
    )
    inputs = {
        "objective_id": "user_attr:score",
        "objective_direction": "upper",
        "elite_proportion_1": 0.5,
        "elite_proportion_2": 0.25,
        "elite_proportion_3": 0.1,
    }

    maximized = planning.HPOPlanning.process(run, inputs)
    minimized = planning.HPOPlanning.process(run, {**inputs, "objective_direction": "lower"})

    assert maximized["objective_name"] == "score"
    assert maximized["optimize"] == "upper"
    assert maximized["trials"][0]["trial_number"] == 10
    assert maximized["insights"][0]["top_trials"][0]["trial_number"] == 10
    assert minimized["optimize"] == "lower"
    assert minimized["trials"][0]["trial_number"] == 11
    assert minimized["insights"][0]["top_trials"][0]["trial_number"] == 11
    assert minimized["spread_attributes"] == ["folds"]


def test_planner_layout_uses_separate_metadata_and_selection_stores():
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("x", lower=0.0, upper=10.0),
            CategoricalHyperparameter("kind", choices=["a", "b"]),
        ]
    )
    outputs = {
        "elite_proportions": [0.1, 0.05, 0.01],
        "features": ["x", "kind"],
        "objective_name": "Accuracy",
        "optimize": "upper",
        "trials": [
            {
                "trial_number": 7,
                "rank": 1,
                "objective": 0.9,
                "configuration": {"x": 1.5, "kind": "a"},
            },
            {
                "trial_number": 8,
                "rank": 2,
                "objective": 0.8,
                "configuration": {"x": 9.0, "kind": "b"},
            },
        ],
        "summary": [],
    }
    for feature in outputs["features"]:
        for proportion in outputs["elite_proportions"]:
            row = {
                "feature": feature,
                "proportion": proportion,
                "group": planning._group_label(proportion),
                "threshold": 0.5,
                "n": 2,
                "minimum": 1.0,
                "q10": 1.2,
                "median": 1.5,
                "q90": 1.8,
                "maximum": 2.0,
                "minimum_normalized": 0.0,
                "q10_normalized": 0.2,
                "median_normalized": 0.5,
                "q90_normalized": 0.8,
                "maximum_normalized": 1.0,
            }
            if feature == "kind":
                row.update(
                    minimum="a",
                    q10="a",
                    median="b",
                    q90="b",
                    maximum="b",
                )
            outputs["summary"].append(row)
    outputs["insights"] = planning._parameter_insights(
        outputs["trials"], configspace, "upper", [0.1, 0.05, 0.01]
    )

    (layout,) = planning.HPOPlanning.load_outputs(
        SimpleNamespace(configspace=configspace), {}, outputs
    )
    components = list(_walk_components(layout))
    stores = [component for component in components if isinstance(component, dcc.Store)]
    sliders = [component for component in components if isinstance(component, dcc.RangeSlider)]
    dropdowns = [component for component in components if isinstance(component, dcc.Dropdown)]

    assert [store.id for store in stores] == [
        planning.METADATA_STORE_ID,
        {
            "type": planning.PLANNED_SEARCH_SPACE_CONTEXT_TYPE,
            "source": planning.PLANNING_SOURCE,
        },
        planning.SELECTION_STORE_ID,
        planning.TRIALS_STORE_ID,
        planning.INSIGHTS_STORE_ID,
        planning.INSIGHT_SELECTION_STORE_ID,
    ]
    assert stores[-1].data is None
    metadata, selection = stores[0].data, stores[2].data
    assert metadata[0]["current_min"] == pytest.approx(0)
    assert metadata[0]["current_max"] == pytest.approx(10)
    assert planning.ELITE_MARKER_COLORS[2] == "#D95F02"
    # Raw quantiles are positioned against ConfigSpace 0–10, not observed summary 1–2.
    first_elite = metadata[0]["elite_positions"][0]
    assert first_elite["color"] == planning.ELITE_MARKER_COLORS[0]
    assert first_elite["minimum_position"] == pytest.approx(0.1)
    assert first_elite["q10_position"] == pytest.approx(0.12)
    assert first_elite["median_position"] == pytest.approx(0.15)
    assert first_elite["q90_position"] == pytest.approx(0.18)
    assert first_elite["maximum_position"] == pytest.approx(0.2)
    assert first_elite["position"] == first_elite["median_position"]
    assert selection == {
        "x": {"min": 0.0, "max": 10.0},
        "kind": {"choices": ["a", "b"]},
    }

    persisted_state = {
        "version": 1,
        "revision": 1,
        "runs": {
            "study": {
                "x": {"min": -2.0, "max": 4.0},
                "kind": {"choices": ["b"]},
            }
        },
    }
    (persisted_layout,) = planning.HPOPlanning.load_outputs(
        SimpleNamespace(id="study", configspace=configspace),
        {"planned_search_space_state": persisted_state},
        outputs,
    )
    persisted_stores = [
        component
        for component in _walk_components(persisted_layout)
        if isinstance(component, dcc.Store)
    ]
    assert persisted_stores[2].data == {
        "x": {"min": -2.0, "max": 4.0},
        "kind": {"choices": ["b"]},
    }
    assert sliders[0].value == pytest.approx([0, 1])
    assert sliders[0].marks == {0: "0", 1: "10"}
    assert sliders[0].tooltip["transform"] == "hpoPlanning.parameter0"
    assert dropdowns[0].multi is True
    assert dropdowns[0].value == ["a", "b"]
    coverage_graph = next(
        component
        for component in components
        if isinstance(component, dcc.Graph) and component.id == planning.COVERAGE_GRAPH_ID
    )
    assert coverage_graph.id == planning.COVERAGE_GRAPH_ID
    assert list(coverage_graph.figure.data[0].x) == [1, 2]
    assert list(coverage_graph.figure.data[1].x) == []
    assert coverage_graph.figure.layout.showlegend is False
    assert coverage_graph.figure.layout.margin.r == 22
    assert coverage_graph.figure.layout.yaxis.title.text == "Accuracy (higher is better)"
    coverage_badge = next(
        component
        for component in components
        if getattr(component, "id", None) == planning.COVERAGE_BADGE_ID
    )
    assert coverage_badge.children == "Included top trials: 2 / 2"
    spread_dropdown = next(
        component
        for component in components
        if isinstance(component, dcc.Dropdown) and component.id == planning.SPREAD_ATTRIBUTE_ID
    )
    spread_toggle = next(
        component
        for component in components
        if isinstance(component, dbc.Checkbox) and component.id == planning.SPREAD_TOGGLE_ID
    )
    spread_mode = next(
        component
        for component in components
        if isinstance(component, dbc.RadioItems) and component.id == planning.SPREAD_MODE_ID
    )
    spread_scatter = next(
        component
        for component in components
        if isinstance(component, dbc.Checkbox) and component.id == planning.SPREAD_SCATTER_ID
    )
    option_label = next(
        component
        for component in components
        if isinstance(component, dbc.Label) and component.children == "Options"
    )
    top_trial_count = next(
        component
        for component in components
        if isinstance(component, dcc.Slider) and component.id == planning.TOP_TRIAL_COUNT_ID
    )
    assert spread_dropdown.options == []
    assert spread_dropdown.value is None
    assert spread_dropdown.disabled is True
    assert spread_toggle.value is False
    assert spread_toggle.disabled is True
    assert spread_mode.value == "sd"
    assert all(option["disabled"] is True for option in spread_mode.options)
    assert spread_scatter.value is False
    assert spread_scatter.disabled is True
    assert option_label.className == "small text-muted mb-1"
    assert top_trial_count.min == 1
    assert top_trial_count.max == 100
    assert top_trial_count.step == 1
    assert top_trial_count.value == 10
    assert top_trial_count.marks == {1: "1", 10: "10", 25: "25", 50: "50", 100: "100"}
    assert components.index(top_trial_count) < components.index(spread_dropdown)
    insight_buttons = [
        component
        for component in components
        if isinstance(getattr(component, "id", None), dict)
        and component.id.get("type") == planning.INSIGHT_BUTTON_TYPE
    ]
    assert [button.id["index"] for button in insight_buttons] == [0, 1]
    assert all(button.children.className == "fas fa-chart-bar" for button in insight_buttons)
    top_trial_buttons = [
        component
        for component in components
        if isinstance(getattr(component, "id", None), dict)
        and component.id.get("type") == planning.TOP_TRIAL_BUTTON_TYPE
    ]
    assert [button.id["index"] for button in top_trial_buttons] == [0, 1]
    assert all(button.children.className == "fas fa-map-marker-alt" for button in top_trial_buttons)
    assert all(button.active is False for button in top_trial_buttons)
    insight_modal = next(
        component
        for component in components
        if getattr(component, "id", None) == planning.INSIGHT_MODAL_ID
    )
    assert isinstance(insight_modal, dbc.Offcanvas)
    assert insight_modal.is_open is False
    assert insight_modal.placement == "end"
    assert insight_modal.backdrop is False
    assert insight_modal.scrollable is True
    elite_control = next(
        component
        for component in components
        if isinstance(component, dbc.RadioItems)
        and component.id == planning.INSIGHT_ELITE_CONTROL_ID
    )
    assert [option["label"] for option in elite_control.options] == [
        "Top 10%",
        "Top 5%",
        "Top 1%",
    ]
    assert elite_control.value == pytest.approx(0.1)
    insight_graph = next(
        component
        for component in components
        if isinstance(component, dcc.Graph) and component.id == planning.INSIGHT_GRAPH_ID
    )
    assert getattr(insight_graph, "figure", None) is None
    assert not any(
        getattr(component, "className", None) == "planner-tip" for component in components
    )

    table_header = next(
        component
        for component in components
        if isinstance(component, html.Div)
        and getattr(component, "className", None) == "planner-table-header"
    )
    assert [cell.children for cell in table_header.children] == [
        "Parameter",
        "Position in current search space",
        "Min (original)",
        "Max (original)",
    ]
    assert not any(
        isinstance(getattr(component, "id", None), dict)
        and component.id.get("type") == "hpo-planner-metric"
        for component in components
    )

    legend = next(
        component
        for component in components
        if isinstance(component, html.Div)
        and getattr(component, "className", None) == "planner-legend"
    )
    general_legend_row, elite_legend_row = legend.children
    assert general_legend_row.className == "planner-legend-row"
    assert len(general_legend_row.children) == 4
    assert elite_legend_row.className == "planner-legend-row planner-legend-elites"
    assert len(elite_legend_row.children) == 3
    assert all(item.children[-1].startswith("Top ") for item in elite_legend_row.children)

    range_control = next(
        component
        for component in components
        if isinstance(component, html.Div)
        and getattr(component, "className", None) == "planner-range-control"
    )
    range_control_props = range_control.to_plotly_json()["props"]
    assert range_control_props["data-current-min"] == "0.0"
    assert range_control_props["data-current-max"] == "10.0"
    assert range_control_props["data-log"] == "false"
    assert range_control_props["data-step"] == ""
    assert range_control_props["data-parameter-type"] == "float"

    current_reference = next(
        component
        for component in components
        if isinstance(component, html.Div)
        and getattr(component, "className", None) == "planner-current-reference"
    )
    assert getattr(current_reference, "style", None) is None
    assert current_reference.children.className == "planner-current-reference-segment"
    assert current_reference.children.style == {
        "left": "16.666667%",
        "width": "66.666667%",
    }

    elite_tracks = [
        component
        for component in components
        if isinstance(component, html.Div)
        and getattr(component, "className", None) == "planner-elite-track"
    ]
    elite_bands = [
        component
        for component in components
        if isinstance(component, html.Span)
        and getattr(component, "className", None) == "planner-elite-band"
    ]
    elite_markers = [
        component
        for component in components
        if isinstance(component, html.Span)
        and getattr(component, "className", None) == "planner-elite-marker"
    ]
    elite_whiskers = [
        component
        for component in components
        if isinstance(component, html.Span)
        and getattr(component, "className", None) == "planner-elite-whisker"
    ]
    assert len(elite_tracks) == len(elite_whiskers) == len(elite_bands) == len(elite_markers) == 3
    assert all(
        "min–max" in track.title and "Q10–Q90" in track.title and "median" in track.title
        for track in elite_tracks
    )

    unselected_segments = [
        component
        for component in components
        if isinstance(component, html.Span)
        and getattr(component, "className", None) == "planner-unselected-segment"
    ]
    assert [segment.style for segment in unselected_segments] == [
        {"left": "0", "width": "16.666667%"},
        {"left": "83.333333%", "width": "16.666667%"},
    ]


def test_elite_ranges_use_the_planner_logarithmic_transform():
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("gamma", lower=0.001, upper=10.0, log=True))
    proportions = [0.1, 0.05, 0.01]
    summary = [
        {
            "feature": "gamma",
            "proportion": proportion,
            "n": 10,
            "minimum": 0.001,
            "q10": 0.01,
            "median": 0.1,
            "q90": 1.0,
            "maximum": 10.0,
        }
        for proportion in proportions
    ]

    metadata = planning._build_planning_metadata(configspace, ["gamma"], proportions, summary)

    for elite in metadata[0]["elite_positions"]:
        assert elite["minimum_position"] == pytest.approx(0)
        assert elite["q10_position"] == pytest.approx(0.25)
        assert elite["median_position"] == pytest.approx(0.5)
        assert elite["q90_position"] == pytest.approx(0.75)
        assert elite["maximum_position"] == pytest.approx(1)


def test_python_preview_preserves_integer_types_and_categorical_choices():
    metadata = [
        {"name": "n_estimators", "parameter_type": "integer"},
        {"name": "booster", "parameter_type": "categorical"},
    ]
    selection = {
        "n_estimators": {"min": 180, "max": 420},
        "booster": {"choices": ["gbtree", "dart"]},
    }

    preview = planning._python_space_preview(metadata, selection)

    assert "'n_estimators': [180, 420]" in preview
    assert "420.0" not in preview
    assert "'booster': ['gbtree', 'dart']" in preview


def test_top_trial_coverage_reports_all_violated_active_parameters():
    metadata = [
        {"name": "depth", "parameter_type": "integer"},
        {"name": "booster", "parameter_type": "categorical"},
        {"name": "conditional", "parameter_type": "float"},
    ]
    selection = {
        "depth": {"min": 2, "max": 6},
        "booster": {"choices": ["tree"]},
        "conditional": {"min": 0.0, "max": 1.0},
    }
    trials = [
        {
            "trial_number": 4,
            "rank": 1,
            "objective": 0.9,
            "configuration": {"depth": 5, "booster": "tree", "conditional": None},
        },
        {
            "trial_number": 9,
            "rank": 2,
            "objective": 0.8,
            "configuration": {"depth": 8, "booster": "linear", "conditional": 0.5},
        },
    ]

    coverage = planning._coverage_state(trials, metadata, selection)
    figure = planning._coverage_figure(trials, metadata, selection, "Accuracy", optimize="upper")

    assert coverage[0]["included"] is True
    assert coverage[0]["violations"] == []
    assert coverage[1]["included"] is False
    assert coverage[1]["violations"] == ["depth", "booster"]
    assert planning._coverage_badge(coverage) == "Included top trials: 1 / 2"
    assert list(figure.data[0].x) == [1]
    assert figure.data[0].showlegend is False
    assert list(figure.data[1].x) == [2]
    assert figure.data[1].showlegend is True
    assert figure.data[1].marker.color == planning.EXCLUDED_TRIAL_COLOR
    assert figure.layout.showlegend is True
    assert list(figure.data[1].customdata[0]) == [9, 2, "depth, booster"]


def test_complete_list_user_attrs_require_numeric_values_on_every_successful_trial():
    trials = [
        Trial(
            index,
            0,
            -1,
            [0.8],
            0.0,
            0.0,
            1.0,
            Status.SUCCESS,
            {
                "user_attrs": {
                    "folds": [0.7, 0.8 + index / 10],
                    "contains_bool": [0.7, True],
                    "empty": [],
                    **({"incomplete": [0.5, 0.6]} if index == 0 else {}),
                }
            },
        )
        for index in range(2)
    ]

    assert planning._complete_user_attr_names(trials, list_values=True) == ["folds"]


def test_coverage_spread_uses_population_standard_deviation_and_hover_values():
    trials = [
        {
            "trial_number": 1,
            "rank": 1,
            "objective": 0.8,
            "configuration": {},
            "spread_values": {"folds": [0.7, 0.8, 0.9]},
        }
    ]

    hidden = planning._coverage_figure(trials, [], {}, "AP", "upper")
    visible = planning._coverage_figure(
        trials,
        [],
        {},
        "AP",
        "upper",
        spread_attribute="folds",
        show_spread=True,
    )

    assert hidden.data[0].error_y.visible is None
    assert list(visible.data[0].error_y.array) == pytest.approx([float(np.std([0.7, 0.8, 0.9]))])
    assert visible.data[0].customdata[0][3] == pytest.approx(float(np.std([0.7, 0.8, 0.9])))
    assert visible.data[0].customdata[0][4] == "[0.7, 0.8, 0.9]"
    assert "folds SD" in visible.data[0].hovertemplate


def test_coverage_spread_supports_range_and_scatter_values():
    trials = [
        {
            "trial_number": 1,
            "rank": 1,
            "objective": 1.0,
            "configuration": {},
            "spread_values": {"folds": [0.7, 0.8, 0.9]},
        }
    ]

    figure = planning._coverage_figure(
        trials,
        [],
        {},
        "AP",
        "upper",
        spread_attribute="folds",
        show_spread=True,
        spread_mode="range",
        show_scatter=True,
    )

    assert list(figure.data[0].y) == [1.0]
    assert figure.data[0].error_y.visible is None
    assert "folds range" in figure.data[0].hovertemplate
    assert list(figure.data[2].y) == pytest.approx([0.8])
    assert list(figure.data[2].error_y.array) == pytest.approx([0.1])
    assert list(figure.data[3].x) == pytest.approx([0.88, 1.0, 1.12])
    assert list(figure.data[3].y) == pytest.approx([0.7, 0.8, 0.9])


def test_coverage_uses_log_scale_only_for_positive_wide_objective_ranges():
    trials = [
        {"trial_number": 1, "rank": 1, "objective": 1.0, "configuration": {}},
        {"trial_number": 2, "rank": 2, "objective": 0.001, "configuration": {}},
    ]

    figure = planning._coverage_figure(trials, [], {}, "Loss", optimize="lower")

    assert figure.layout.yaxis.type == "log"
    assert figure.layout.yaxis.title.text == "Loss (lower is better)"
    assert list(figure.layout.xaxis.tickvals) == [1, 2]


def test_only_nonempty_optuna_studies_with_features_are_compatible():
    suitable = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    empty = SimpleNamespace(prefix="Optuna", empty=lambda: True, configspace={"x": 1})
    no_features = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={})
    group = SimpleNamespace(
        prefix="group",
        runs=[suitable],
        empty=lambda: False,
        configspace={"x": 1},
    )

    assert planning.HPOPlanning.check_run_compatibility(suitable)
    assert not planning.HPOPlanning.check_run_compatibility(empty)
    assert not planning.HPOPlanning.check_run_compatibility(no_features)
    assert planning.HPOPlanning.check_run_compatibility(group)
