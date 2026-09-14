from types import SimpleNamespace

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from dash import dcc, html

from deepcave import config
from deepcave.plugins.hyperparameter import analysis_utils
from deepcave.plugins.hyperparameter import two_dimensional_analysis as analysis
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial


def test_plugin_is_registered_directly_after_one_dimensional_analysis():
    plugin_ids = [plugin.id for plugin in config.PLUGINS["Hyperparameter Analysis"]]

    one_dimensional_index = plugin_ids.index("one_dimensional_analysis")
    assert plugin_ids[one_dimensional_index + 1] == "two_dimensional_analysis"


def test_cache_key_is_versioned_for_axis_mask_geometry():
    plugin = object.__new__(analysis.TwoDimensionalAnalysis)
    inputs = {"feature": {"value": "x"}}
    legacy_key = super(analysis.TwoDimensionalAnalysis, plugin)._dict_as_key(inputs)

    assert plugin._dict_as_key(inputs) == f"{legacy_key}-v{analysis.CACHE_SCHEMA_VERSION}"


def _walk_components(component):
    """Yield a Dash component and all of its descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk_components(child)


def test_input_controls_use_a_single_toolbar_row():
    registered = []
    (toolbar,) = analysis.TwoDimensionalAnalysis.get_input_layout(
        lambda input_id, *_args, **_kwargs: registered.append(input_id) or input_id
    )

    assert registered == [
        "n_bins_1",
        "n_bins_2",
        "color_by_rank",
        "hyperparameter_name_1",
        "hyperparameter_name_2",
        "elite_proportion",
    ]
    assert "analysis-toolbar" in toolbar.className
    assert not any(isinstance(component, dbc.Switch) for component in _walk_components(toolbar))


def test_output_layout_uses_independent_columns_and_separate_control_panels():
    registered = []
    (grid,) = analysis.TwoDimensionalAnalysis.get_output_layout(
        lambda name, *_args, **_kwargs: registered.append(name) or name
    )

    assert grid.className == "analysis-grid analysis-grid--two-dimensional"
    assert registered == ["trials_graph", "elite_rate_graph", "objective_graph"]
    assert len(grid.children) == 2
    assert all(column.className == "two-dimensional-analysis-column" for column in grid.children)
    left_column, right_column = grid.children
    assert [card.className for card in left_column.children] == [
        "analysis-card",
        "analysis-card",
    ]
    assert [card.className for card in right_column.children] == [
        "analysis-card",
        "analysis-card",
        "analysis-card",
    ]
    assert [card.children[0].children for card in left_column.children] == [
        "Observed trials",
        "Empirical objective median contour",
    ]
    assert [card.children[0].children for card in right_column.children] == [
        "Elite enrichement (1x = Baseline)",
        "Plot controls",
        "HPO Planner - Search Space",
    ]
    plot_controls = right_column.children[1]
    range_controls = right_column.children[2]
    bin_sliders = [
        component
        for component in _walk_components(plot_controls)
        if isinstance(component, dcc.Slider)
    ]
    assert [component.id for component in bin_sliders] == [
        "two_dimensional_analysis-n_bins_1-input",
        "two_dimensional_analysis-n_bins_2-input",
    ]
    assert all(component.value == analysis.N_BINS for component in bin_sliders)
    assert all(component.min == analysis.MIN_BINS for component in bin_sliders)
    assert all(component.max == analysis.MAX_BINS for component in bin_sliders)
    assert all(component.step == 1 for component in bin_sliders)
    planning_mounts = [
        component
        for component in _walk_components(range_controls)
        if getattr(component, "id", None) == analysis.PLANNING_CONTROLS_ID
    ]
    intent_stores = [
        component
        for component in _walk_components(range_controls)
        if isinstance(component, dcc.Store) and component.id == analysis.PLANNING_INTENT_ID
    ]
    assert len(planning_mounts) == 1
    assert len(intent_stores) == 1
    reset_buttons = [
        component
        for component in _walk_components(range_controls)
        if isinstance(component, dbc.Button) and component.id == analysis.PLANNING_RESET_ID
    ]
    assert len(reset_buttons) == 1
    assert reset_buttons[0].n_clicks == 0
    assert "Reset feature ranges" in str(reset_buttons[0].children)
    assert not any(
        getattr(component, "id", None) == analysis.PLANNING_CONTROLS_ID
        for component in _walk_components(plot_controls)
    )
    assert not any(
        isinstance(component, dcc.Slider) for component in _walk_components(range_controls)
    )
    color_switches = [
        component
        for component in _walk_components(plot_controls)
        if isinstance(component, dbc.Switch)
    ]
    assert len(color_switches) == 1
    assert color_switches[0].id == "two_dimensional_analysis-color_by_rank-input"
    assert color_switches[0].label == "Color observed trials by trial rank"
    assert color_switches[0].value is False
    assert not any(
        isinstance(component, dbc.Switch) for component in _walk_components(range_controls)
    )
    assert "Set the numeric resolution" not in str(plot_controls)


def test_dependencies_exclude_time_and_enforce_distinct_hyperparameters():
    objectives = [Objective("Loss"), Objective("Accuracy", optimize="upper"), Objective("Time")]
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (1, 5), "kind": ["a", "b"]})

    class DependencyRun:
        def get_objectives(self):
            return objectives

        def get_objective_id(self, objective):
            return objectives.index(objective)

    run = DependencyRun()
    run.configspace = configspace
    inputs = {
        "objective_id": {"value": None},
        "hyperparameter_name_1": {"value": "x"},
        "hyperparameter_name_2": {"value": "x"},
        "elite_proportion": {"value": 0.2},
    }

    result = analysis.TwoDimensionalAnalysis.load_dependency_inputs(
        object.__new__(analysis.TwoDimensionalAnalysis),
        run,
        {},
        inputs,
    )

    assert [option["label"] for option in result["objective_id"]["options"]] == [
        "Loss",
        "Accuracy",
    ]
    assert result["hyperparameter_name_1"]["value"] == "x"
    assert result["hyperparameter_name_2"]["value"] != "x"
    assert "x" not in [option["value"] for option in result["hyperparameter_name_2"]["options"]]
    assert result["n_bins_1"]["value"] == analysis.N_BINS
    assert result["n_bins_2"]["value"] == analysis.N_BINS
    assert result["color_by_rank"]["value"] is False


def test_numeric_grouping_uses_four_linear_or_logarithmic_bins():
    linear = UniformFloatHyperparameter("linear", lower=1.0, upper=101.0)
    logarithmic = UniformFloatHyperparameter("logarithmic", lower=1.0, upper=10000.0, log=True)

    linear_groups = analysis_utils._group_parameter_values(
        pd.Series([1.0, 21.0, 41.0, 61.0, 101.0]),
        linear,
        n_bins=4,
        discrete_numeric=True,
    )
    logarithmic_groups = analysis_utils._group_parameter_values(
        pd.Series([1.0, 10.0, 100.0, 1000.0, 10000.0]),
        logarithmic,
        n_bins=4,
        discrete_numeric=True,
    )

    assert len(linear_groups.cat.categories) == 4
    assert len(logarithmic_groups.cat.categories) == 4
    assert 26.0 in linear_groups.cat.categories[0]
    assert 10.0 in logarithmic_groups.cat.categories[0]


def test_logarithmic_group_coordinate_stays_inside_positive_domain():
    hyperparameter = UniformFloatHyperparameter(
        "gamma",
        lower=0.0001,
        upper=20.0,
        log=True,
    )
    # pandas expands the first interval below zero for include_lowest=True.
    group = pd.Interval(-0.0009, 0.00211, closed="right")

    coordinate = analysis._group_coordinate(group, hyperparameter)

    assert coordinate == pytest.approx(np.sqrt(hyperparameter.lower * group.right))
    assert coordinate > 0

    # Previously cached analyses can still contain the invalid zero coordinate.
    assert analysis._contour_coordinates(
        [{"coordinate": 0.0}, {"coordinate": 0.01}],
        hyperparameter,
    ) == pytest.approx([hyperparameter.lower, 0.01])


def test_axis_groups_trim_empty_edges_but_keep_internal_gaps():
    groups = pd.Series(
        pd.Categorical(
            ["used-1", "used-2"],
            categories=["empty-edge", "used-1", "empty-middle", "used-2"],
            ordered=True,
        )
    )

    assert analysis._observed_categories(groups) == [
        "used-1",
        "empty-middle",
        "used-2",
    ]


@pytest.mark.parametrize(
    ("optimize", "objectives", "expected_threshold", "expected_quantiles"),
    [
        ("upper", [10.0, 9.0, 8.0, 7.0], 8.5, [1.0, 0.75, 0.5, 0.25]),
        ("lower", [1.0, 2.0, 3.0, 4.0], 2.5, [1.0, 0.75, 0.5, 0.25]),
    ],
)
def test_pair_summary_honors_objective_direction(
    optimize,
    objectives,
    expected_threshold,
    expected_quantiles,
):
    hp1 = CategoricalHyperparameter("x", choices=["a", "b"])
    hp2 = CategoricalHyperparameter("y", choices=["c", "d"])
    trials = pd.DataFrame(
        {
            "trial_number": range(4),
            "hyperparameter_1": ["a", "a", "b", "b"],
            "hyperparameter_2": ["c", "d", "c", "d"],
            "objective": objectives,
        }
    )

    summarized, x_groups, y_groups, threshold, elite_count, cells = analysis._summarize_pair(
        trials,
        hp1,
        hp2,
        optimize,
        0.5,
    )

    assert threshold == pytest.approx(expected_threshold)
    assert elite_count == 2
    assert summarized["objective_rank"].tolist() == pytest.approx([1, 2, 3, 4])
    assert summarized["rank_quantile"].tolist() == pytest.approx(expected_quantiles)
    assert x_groups == ["a", "b"]
    assert y_groups == ["c", "d"]
    assert sum(cell["elite"] for cell in cells) == 2
    assert [cell["elite_enrichment"] for cell in cells] == pytest.approx([2.0, 2.0, 0.0, 0.0])


def test_elite_threshold_ties_are_all_included_and_joint_gaps_remain_empty():
    hp1 = CategoricalHyperparameter("x", choices=["a", "b"])
    hp2 = CategoricalHyperparameter("y", choices=["c", "d"])
    trials = pd.DataFrame(
        {
            "trial_number": range(4),
            "hyperparameter_1": ["a", "a", "b", "b"],
            "hyperparameter_2": ["c", "d", "c", "c"],
            "objective": [10.0, 9.0, 9.0, 0.0],
        }
    )

    _, x_groups, y_groups, threshold, elite_count, cells = analysis._summarize_pair(
        trials,
        hp1,
        hp2,
        "upper",
        0.5,
    )
    matrix = analysis._cell_matrix(
        pd.DataFrame(cells),
        [str(group) for group in x_groups],
        [str(group) for group in y_groups],
        "elite_rate_pct",
    )

    assert threshold == pytest.approx(9.0)
    assert elite_count == 3
    assert matrix[1][1] is None


def test_process_keeps_repeated_configurations_and_original_trial_numbers():
    objective = Objective("Accuracy", optimize="upper")
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    history = [
        Trial(0, 0, -1, [0.8], 0.0, 0.0, 1.0, Status.SUCCESS, {"optuna_trial_number": 4}),
        Trial(0, 0, -1, [0.9], 0.0, 1.0, 2.0, Status.SUCCESS, {"optuna_trial_number": 9}),
    ]
    run = SimpleNamespace(
        history=history,
        configspace=configspace,
        get_config=lambda _config_id: {"x": 0.5, "y": 0.75},
        get_objective=lambda _objective_id: objective,
    )

    outputs = analysis.TwoDimensionalAnalysis.process(
        run,
        {
            "objective_id": 0,
            "hyperparameter_name_1": "x",
            "hyperparameter_name_2": "y",
            "elite_proportion": 0.5,
        },
    )

    assert [trial["trial_number"] for trial in outputs["trials"]] == [4, 9]


def test_feature_changes_reuse_the_objective_wide_trial_table():
    objective = Objective("Accuracy", optimize="upper")
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)})
    configurations = {
        0: {"x": 0.1, "y": 0.2, "z": 0.3},
        1: {"x": 0.4, "y": 0.5, "z": 0.6},
    }

    class CachedRun:
        id = "cached-run"
        latest_change = 1.0
        configs = configurations
        history = [
            Trial(0, 0, -1, [0.8], 0.0, 0.0, 1.0, Status.SUCCESS, {}),
            Trial(1, 0, -1, [0.9], 0.0, 1.0, 2.0, Status.SUCCESS, {}),
        ]

        def __init__(self):
            self.configspace = configspace

        def get_objective(self, _objective_id):
            return objective

        def get_config(self, _config_id):
            raise AssertionError("stored configurations should be read without reconstruction")

    run = CachedRun()
    analysis_utils._TRIAL_TABLE_CACHE.clear()
    first_table = analysis_utils._get_trial_table(run, 0)
    second_table = analysis_utils._get_trial_table(run, 0)

    assert first_table is second_table
    assert list(first_table.columns) == [
        "trial_number",
        "objective",
        "x",
        "y",
        "z",
        "objective_rank",
        "rank_quantile",
    ]
    assert analysis._collect_trials(run, 0, ["x", "y"])["x"].tolist() == [0.1, 0.4]
    assert analysis._collect_trials(run, 0, ["x", "z"])["z"].tolist() == [0.3, 0.6]

    run.latest_change = 2.0
    refreshed_table = analysis_utils._get_trial_table(run, 0)
    assert refreshed_table is not first_table
    assert all(":2.0:" in key[1] for key in analysis_utils._TRIAL_TABLE_CACHE)


def test_independent_bin_counts_feed_enrichment_and_contour(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Accuracy", optimize="upper")
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    configurations = [{"x": x / 5, "y": y / 5} for y in range(6) for x in range(6)]
    history = [
        Trial(
            index,
            0,
            -1,
            [float(index)],
            0.0,
            0.0,
            float(index),
            Status.SUCCESS,
            {"optuna_trial_number": index},
        )
        for index in range(len(configurations))
    ]
    run = SimpleNamespace(
        history=history,
        configspace=configspace,
        get_config=lambda config_id: configurations[config_id],
        get_objective=lambda _objective_id: objective,
    )
    inputs = {
        "objective_id": 0,
        "hyperparameter_name_1": "x",
        "hyperparameter_name_2": "y",
        "elite_proportion": 0.25,
        "n_bins_1": 3,
        "n_bins_2": 5,
    }

    outputs = analysis.TwoDimensionalAnalysis.process(run, inputs)
    _, enrichment_figure, contour_figure = analysis.TwoDimensionalAnalysis.load_outputs(
        run,
        inputs,
        outputs,
    )

    assert len(outputs["axis_1"]) == 3
    assert len(outputs["axis_2"]) == 5
    assert all(axis["kind"] == "interval" for axis in outputs["axis_1"])
    assert all(axis["lower"] < axis["upper"] for axis in outputs["axis_1"])
    assert np.asarray(enrichment_figure.data[0].z).shape == (5, 3)
    assert np.asarray(contour_figure.data[0].z).shape == (7, 5)
    assert contour_figure.data[0].x[0] == pytest.approx(outputs["axis_1"][0]["lower"])
    assert contour_figure.data[0].x[-1] == pytest.approx(outputs["axis_1"][-1]["upper"])
    assert contour_figure.data[0].y[0] == pytest.approx(outputs["axis_2"][0]["lower"])
    assert contour_figure.data[0].y[-1] == pytest.approx(outputs["axis_2"][-1]["upper"])
    assert len(contour_figure.layout.xaxis.tickvals) == 3
    assert len(contour_figure.layout.yaxis.tickvals) == 5


def test_numeric_outputs_use_contour_and_preserve_missing_cells(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Accuracy", optimize="upper")
    run = SimpleNamespace(
        configspace={
            "x": UniformFloatHyperparameter("x", lower=1.0, upper=100.0, log=True),
            "y": UniformFloatHyperparameter("y", lower=0.0, upper=10.0),
        },
        get_objective=lambda _objective_id: objective,
    )
    inputs = {
        "objective_id": 0,
        "hyperparameter_name_1": "x",
        "hyperparameter_name_2": "y",
        "elite_proportion": 0.5,
    }
    outputs = {
        "trials": [
            {
                "trial_number": 1,
                "hyperparameter_1": 1.0,
                "hyperparameter_2": 0.0,
                "objective": 0.8,
                "objective_rank": 2.0,
                "rank_quantile": 0.5,
                "is_elite": False,
                "elite_rank": None,
            },
            {
                "trial_number": 2,
                "hyperparameter_1": 100.0,
                "hyperparameter_2": 10.0,
                "objective": 0.9,
                "objective_rank": 1.0,
                "rank_quantile": 1.0,
                "is_elite": True,
                "elite_rank": 1.0,
            },
        ],
        "cells": [
            {
                "group_1": "low-x",
                "group_2": "low-y",
                "n": 1,
                "elite": 0,
                "elite_rate_pct": 0.0,
                "elite_median_rank": None,
                "objective_median": 0.8,
            },
            {
                "group_1": "high-x",
                "group_2": "high-y",
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_median_rank": 1.0,
                "objective_median": 0.9,
            },
        ],
        "axis_1": [
            {"label": "low-x", "coordinate": 1.0},
            {"label": "high-x", "coordinate": 100.0},
        ],
        "axis_2": [
            {"label": "low-y", "coordinate": 0.0},
            {"label": "high-y", "coordinate": 10.0},
        ],
        "elite_threshold": 0.85,
        "elite_count": 1,
    }

    trials_figure, elite_rate_figure, objective_figure = (
        analysis.TwoDimensionalAnalysis.load_outputs(
            run,
            inputs,
            outputs,
        )
    )

    assert trials_figure.data[0].type == "scattergl"
    assert trials_figure.layout.xaxis.type == "log"
    assert trials_figure.data[0].marker.reversescale is False
    assert list(trials_figure.data[0].marker.color) == pytest.approx([0.8, 0.9])
    assert elite_rate_figure.data[0].type == "heatmap"
    assert elite_rate_figure.data[0].z[0][1] is None
    assert elite_rate_figure.data[0].z[1][1] == pytest.approx(2.0)
    assert elite_rate_figure.data[0].zmax == pytest.approx(2.0)
    assert "Elite enrichment: %{z:.2f}×" in elite_rate_figure.data[0].hovertemplate
    assert "Elite rate: %{customdata[2]:.1f}%" in elite_rate_figure.data[0].hovertemplate
    assert elite_rate_figure.data[0].showscale is False
    assert objective_figure.data[0].type == "contour"
    assert objective_figure.data[0].connectgaps is False
    assert objective_figure.data[0].z[0][1] is None
    assert objective_figure.layout.xaxis.type == "log"
    assert (
        trials_figure.data[0].marker.colorscale
        == elite_rate_figure.data[0].colorscale
        == objective_figure.data[0].colorscale
    )
    assert trials_figure.data[0].marker.colorbar.orientation == "h"
    assert objective_figure.data[0].colorbar.orientation == "h"
    assert trials_figure.data[0].marker.colorbar.title.text == "Accuracy"
    assert objective_figure.data[0].colorbar.title.text == "Accuracy"
    assert trials_figure.layout.annotations == ()
    assert objective_figure.layout.annotations == ()
    assert all(
        figure.layout.title.text is None
        for figure in (trials_figure, elite_rate_figure, objective_figure)
    )

    rank_inputs = {**inputs, "color_by_rank": True}
    rank_figure, _, _ = analysis.TwoDimensionalAnalysis.load_outputs(
        run,
        rank_inputs,
        outputs,
    )
    assert list(rank_figure.data[0].marker.color) == pytest.approx([2.0, 1.0])
    assert rank_figure.data[0].marker.reversescale is True
    assert rank_figure.data[0].marker.colorbar.title.text == "Trial rank"


def test_enrichment_distinguishes_observed_zero_from_an_empty_bin():
    cells = pd.DataFrame(
        [
            {
                "group_1": "observed-zero",
                "group_2": "row",
                "n": 10,
                "elite": 0,
                "elite_rate_pct": 0.0,
                "elite_median_rank": None,
            },
            {
                "group_1": "enriched",
                "group_2": "row",
                "n": 10,
                "elite": 2,
                "elite_rate_pct": 20.0,
                "elite_median_rank": 1.5,
            },
            {
                "group_1": "empty",
                "group_2": "row",
                "n": 0,
                "elite": 0,
                "elite_rate_pct": 0.0,
                "elite_median_rank": None,
            },
        ]
    )

    figure = analysis._elite_enrichment_figure(
        cells,
        ["observed-zero", "enriched", "empty"],
        ["row"],
        "learning_rate",
        "max_depth",
        elite_threshold=0.5,
        elite_proportion=0.1,
    )

    assert list(figure.data[0].z[0]) == [0.0, 2.0, None]
    annotation_text = [annotation.text for annotation in figure.layout.annotations]
    assert "0×" in annotation_text
    assert "2.00×" in annotation_text
    assert "No data" in annotation_text
    assert annotation_text.count("n = 0") == 1
    assert figure.data[0].colorscale[0][1] == "#440154"
    assert figure.data[0].colorscale[-1][1] == "#fde725"
    assert figure.data[0].showscale is False
    assert figure.layout.xaxis.domain == (0.06, 0.94)
    assert figure.layout.xaxis.scaleanchor is None
    assert all("<br>" not in label for label in figure.layout.xaxis.ticktext)
    assert figure.layout.xaxis.title.standoff == 4
    assert figure.layout.margin.b == analysis.ENRICHMENT_BOTTOM_MARGIN
    assert "Elite enrichment" not in [annotation.text for annotation in figure.layout.annotations]

    figure_with_twenty_percent_elite = analysis._elite_enrichment_figure(
        cells,
        ["observed-zero", "enriched", "empty"],
        ["row"],
        "learning_rate",
        "max_depth",
        elite_threshold=0.5,
        elite_proportion=0.2,
    )
    assert list(figure_with_twenty_percent_elite.data[0].z[0]) == [0.0, 1.0, None]


@pytest.mark.parametrize(
    ("x_bin_count", "y_bin_count", "shows_cell_text"),
    [
        (15, 6, True),
        (16, 6, False),
        (15, 7, False),
    ],
)
def test_enrichment_hides_cell_text_for_dense_grids(
    x_bin_count,
    y_bin_count,
    shows_cell_text,
):
    x_labels = [f"x-{index}" for index in range(x_bin_count)]
    y_labels = [f"y-{index}" for index in range(y_bin_count)]
    cells = pd.DataFrame(
        [
            {
                "group_1": x_labels[0],
                "group_2": y_labels[0],
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_median_rank": 1.0,
            }
        ]
    )

    figure = analysis._elite_enrichment_figure(
        cells,
        x_labels,
        y_labels,
        "feature-1",
        "feature-2",
        elite_threshold=0.5,
        elite_proportion=0.1,
    )

    assert bool(figure.layout.annotations) is shows_cell_text
    assert "Elite enrichment" in figure.data[0].hovertemplate


def test_categorical_axis_uses_objective_heatmap(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Loss", optimize="lower")
    run = SimpleNamespace(
        id="study",
        configspace={
            "kind": CategoricalHyperparameter("kind", choices=["a", "b"]),
            "x": UniformFloatHyperparameter("x", lower=0.0, upper=1.0),
        },
        get_objective=lambda _objective_id: objective,
    )
    inputs = {
        "objective_id": 0,
        "hyperparameter_name_1": "kind",
        "hyperparameter_name_2": "x",
        "elite_proportion": 0.5,
        "planned_search_space_state": {
            "version": 1,
            "revision": 1,
            "runs": {"study": {"kind": {"choices": ["b"]}}},
        },
    }
    outputs = {
        "trials": [
            {
                "trial_number": 0,
                "hyperparameter_1": "a",
                "hyperparameter_2": 0.5,
                "objective": 1.0,
                "objective_rank": 1.0,
                "rank_quantile": 1.0,
                "is_elite": True,
                "elite_rank": 1.0,
            }
        ],
        "cells": [
            {
                "group_1": "a",
                "group_2": "0.5",
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_median_rank": 1.0,
                "objective_median": 1.0,
            }
        ],
        "axis_1": [{"label": "a", "coordinate": "a"}],
        "axis_2": [{"label": "0.5", "coordinate": 0.5}],
        "elite_threshold": 1.0,
        "elite_count": 1,
    }

    _, _, objective_figure = analysis.TwoDimensionalAnalysis.load_outputs(run, inputs, outputs)

    assert objective_figure.data[0].type == "heatmap"
    assert objective_figure.data[0].reversescale is True
    assert len(objective_figure.layout.shapes) == 1
    assert objective_figure.layout.shapes[0].x0 == -0.5
    assert objective_figure.layout.shapes[0].x1 == 0.5


def test_reduced_planning_controls_adapt_to_parameter_type():
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("x", lower=0.01, upper=1.0, log=True),
            CategoricalHyperparameter("kind", choices=["a", "b"]),
        ]
    )
    numeric, categorical = analysis.configspace_metadata(configspace, ["x", "kind"])

    numeric_control = analysis._planning_control(numeric, {"min": 0.1, "max": 2.0}, 0)
    category_control = analysis._planning_control(categorical, {"choices": ["b"]}, 1)
    numeric_components = list(_walk_components(numeric_control))
    category_components = list(_walk_components(category_control))

    (slider,) = [item for item in numeric_components if isinstance(item, dcc.RangeSlider)]
    (dropdown,) = [item for item in category_components if isinstance(item, dcc.Dropdown)]
    assert slider.id == {"type": analysis.PLANNING_RANGE_TYPE, "index": 0}
    assert slider.min < 0
    assert slider.max > 1
    assert dropdown.id == {"type": analysis.PLANNING_CATEGORIES_TYPE, "index": 1}
    assert dropdown.value == ["b"]
    assert any(
        getattr(item, "className", None) == "planner-extension-segment"
        for item in numeric_components
    )


def test_logarithmic_interval_mask_uses_exact_fractional_bin_positions():
    parameter = {
        "name": "x",
        "parameter_type": "float",
        "current_min": 1.0,
        "current_max": 100.0,
        "log": True,
        "step": None,
    }
    axes = [
        {"kind": "interval", "lower": 1.0, "upper": 10.0},
        {"kind": "interval", "lower": 10.0, "upper": 100.0},
    ]

    included, domain = analysis._binned_axis_segments(
        axes,
        parameter,
        {"min": np.sqrt(10.0), "max": np.sqrt(1000.0)},
    )

    assert domain == (-0.5, 1.5)
    assert len(included) == 1
    assert included[0] == pytest.approx((0.0, 1.0))


def test_rectangular_mask_does_not_double_overlay_corners():
    figure = analysis.go.Figure()

    analysis._apply_rectangular_mask(
        figure,
        [(0.25, 0.75)],
        (0.0, 1.0),
        [(0.25, 0.75)],
        (0.0, 1.0),
    )

    shapes = list(figure.layout.shapes)
    assert len(shapes) == 4
    assert all(shape.fillcolor == analysis.MASK_COLOR for shape in shapes)
    vertical = [shape for shape in shapes if shape.y0 == 0.0 and shape.y1 == 1.0]
    horizontal = [shape for shape in shapes if shape.x0 == 0.25 and shape.x1 == 0.75]
    assert len(vertical) == 2
    assert len(horizontal) == 2


def test_only_suitable_optuna_studies_are_compatible():
    suitable = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1, "y": 2})
    one_parameter = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    group = SimpleNamespace(
        prefix="group",
        runs=[suitable],
        empty=lambda: False,
        configspace={"x": 1, "y": 2},
    )

    assert analysis.TwoDimensionalAnalysis.check_run_compatibility(suitable)
    assert not analysis.TwoDimensionalAnalysis.check_run_compatibility(one_parameter)
    assert analysis.TwoDimensionalAnalysis.check_run_compatibility(group)
