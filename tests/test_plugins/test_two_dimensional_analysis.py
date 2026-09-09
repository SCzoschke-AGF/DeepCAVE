from types import SimpleNamespace

import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from dash import html

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


def _walk_components(component):
    """Yield a Dash component and all of its descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children"):
            yield from _walk_components(child)


def test_every_control_shares_the_single_toolbar_row():
    registered = []
    (toolbar,) = analysis.TwoDimensionalAnalysis.get_input_layout(
        lambda input_id, *_args, **_kwargs: registered.append(input_id) or input_id
    )

    assert registered == [
        "objective_id",
        "hyperparameter_name_1",
        "hyperparameter_name_2",
        "elite_proportion",
    ]
    assert "analysis-toolbar" in toolbar.className
    assert not any(isinstance(component, html.Button) for component in _walk_components(toolbar))


def test_output_layout_is_a_two_by_two_grid_with_an_empty_final_cell():
    registered = []
    (grid,) = analysis.TwoDimensionalAnalysis.get_output_layout(
        lambda name, *_args, **_kwargs: registered.append(name) or name
    )

    assert grid.className == "analysis-grid analysis-grid--two-dimensional"
    assert registered == ["trials_graph", "elite_rate_graph", "objective_graph"]
    assert [card.className for card in grid.children] == [
        "analysis-card",
        "analysis-card",
        "analysis-card",
        "analysis-card analysis-card--empty",
    ]
    assert [card.children[0].children for card in grid.children[:3]] == [
        "Observed trials",
        "Elite enrichement (1x = Baseline)",
        "Empirical objective median contour",
    ]


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

    assert trials_figure.data[0].type == "scatter"
    assert trials_figure.layout.xaxis.type == "log"
    assert trials_figure.data[0].marker.reversescale is True
    assert list(trials_figure.data[0].marker.color) == pytest.approx([2.0, 1.0])
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
    assert all(
        figure.layout.title.text is None
        for figure in (trials_figure, elite_rate_figure, objective_figure)
    )


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
    assert "Elite enrichment" not in [
        annotation.text for annotation in figure.layout.annotations
    ]

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


def test_categorical_axis_uses_objective_heatmap(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Loss", optimize="lower")
    run = SimpleNamespace(
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


def test_only_suitable_optuna_studies_are_compatible():
    suitable = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1, "y": 2})
    one_parameter = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    group = SimpleNamespace(prefix="group", empty=lambda: False, configspace={"x": 1, "y": 2})

    assert analysis.TwoDimensionalAnalysis.check_run_compatibility(suitable)
    assert not analysis.TwoDimensionalAnalysis.check_run_compatibility(one_parameter)
    assert not analysis.TwoDimensionalAnalysis.check_run_compatibility(group)
