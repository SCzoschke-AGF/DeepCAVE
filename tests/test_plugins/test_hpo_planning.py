from types import SimpleNamespace

import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
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


def test_toolbar_contains_all_controls_in_one_compact_row():
    registered = []
    (toolbar,) = planning.HPOPlanning.get_input_layout(
        lambda name, *_args, **_kwargs: registered.append(name) or name
    )

    assert registered == [
        "objective_id",
        "elite_proportion_1",
        "elite_proportion_2",
        "elite_proportion_3",
    ]
    assert "analysis-toolbar" in toolbar.className
    assert not any(isinstance(component, html.Button) for component in _walk_components(toolbar))


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
    assert len(outputs["summary"]) == 6
    assert all("×" not in row["feature"] for row in outputs["summary"])
    assert max(row["n"] for row in outputs["summary"] if row["feature"] == "x") == 2


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
        "Loss",
        "Accuracy",
    ]


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

    (layout,) = planning.HPOPlanning.load_outputs(
        SimpleNamespace(configspace=configspace), {}, outputs
    )
    components = list(_walk_components(layout))
    stores = [component for component in components if isinstance(component, dcc.Store)]
    sliders = [component for component in components if isinstance(component, dcc.RangeSlider)]
    dropdowns = [component for component in components if isinstance(component, dcc.Dropdown)]

    assert [store.id for store in stores] == [
        planning.METADATA_STORE_ID,
        planning.SELECTION_STORE_ID,
    ]
    metadata, selection = stores[0].data, stores[1].data
    assert metadata[0]["current_min"] == pytest.approx(0)
    assert metadata[0]["current_max"] == pytest.approx(10)
    assert planning.ELITE_MARKER_COLORS[2] == "#D95F02"
    # Raw quantiles are positioned against ConfigSpace 0–10, not observed summary 1–2.
    first_elite = metadata[0]["elite_positions"][0]
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
    assert sliders[0].value == pytest.approx([0, 1])
    assert sliders[0].marks == {0: "0", 1: "10"}
    assert sliders[0].tooltip["transform"] == "hpoPlanning.parameter0"
    assert dropdowns[0].multi is True
    assert dropdowns[0].value == ["a", "b"]
    assert not any(isinstance(component, dcc.Graph) for component in components)
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


def test_only_nonempty_optuna_studies_with_features_are_compatible():
    suitable = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    empty = SimpleNamespace(prefix="Optuna", empty=lambda: True, configspace={"x": 1})
    no_features = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={})
    group = SimpleNamespace(prefix="group", empty=lambda: False, configspace={"x": 1})

    assert planning.HPOPlanning.check_run_compatibility(suitable)
    assert not planning.HPOPlanning.check_run_compatibility(empty)
    assert not planning.HPOPlanning.check_run_compatibility(no_features)
    assert not planning.HPOPlanning.check_run_compatibility(group)
