from types import SimpleNamespace

import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

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

    overview, objective_figure = analysis.TwoDimensionalAnalysis.load_outputs(
        run,
        inputs,
        outputs,
    )

    assert [trace.type for trace in overview.data] == ["scatter", "heatmap"]
    assert overview.layout.xaxis.type == "log"
    assert overview.data[1].z[0][1] is None
    assert objective_figure.data[0].type == "contour"
    assert objective_figure.data[0].connectgaps is False
    assert objective_figure.data[0].z[0][1] is None
    assert objective_figure.layout.xaxis.type == "log"


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

    _, objective_figure = analysis.TwoDimensionalAnalysis.load_outputs(run, inputs, outputs)

    assert objective_figure.data[0].type == "heatmap"


def test_only_suitable_optuna_studies_are_compatible():
    suitable = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1, "y": 2})
    one_parameter = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    group = SimpleNamespace(prefix="group", empty=lambda: False, configspace={"x": 1, "y": 2})

    assert analysis.TwoDimensionalAnalysis.check_run_compatibility(suitable)
    assert not analysis.TwoDimensionalAnalysis.check_run_compatibility(one_parameter)
    assert not analysis.TwoDimensionalAnalysis.check_run_compatibility(group)
