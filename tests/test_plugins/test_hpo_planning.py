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

from deepcave import config
from deepcave.plugins.hyperparameter import hpo_planning as planning
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial


def test_plugin_is_registered_directly_after_configuration_cube():
    plugin_ids = [plugin.id for plugin in config.PLUGINS["Hyperparameter Analysis"]]

    configuration_cube_index = plugin_ids.index("ccube")
    assert plugin_ids[configuration_cube_index + 1] == "hpo_planning"


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


def test_plot_contains_only_single_feature_ranges_and_scales_height(monkeypatch):
    monkeypatch.setattr(planning, "save_image", lambda *_args: None)
    outputs = {
        "elite_proportions": [0.1, 0.05, 0.01],
        "features": ["x", "conditional"],
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
            if feature == "conditional" and proportion == 0.01:
                row.update(
                    {
                        "n": 0,
                        "minimum": None,
                        "q10": None,
                        "median": None,
                        "q90": None,
                        "maximum": None,
                        "minimum_normalized": None,
                        "q10_normalized": None,
                        "median_normalized": None,
                        "q90_normalized": None,
                        "maximum_normalized": None,
                    }
                )
            outputs["summary"].append(row)

    figure, style = planning.HPOPlanning.load_outputs(
        SimpleNamespace(),
        {},
        outputs,
    )

    assert [trace.name for trace in figure.data] == [
        "Minimum–Maximum",
        "Q10 / Q90",
        "Median",
    ]
    assert len(figure.layout.shapes) == 1
    assert any(annotation.text == "No active trials" for annotation in figure.layout.annotations)
    assert all("×" not in annotation.text for annotation in figure.layout.annotations)
    assert figure.layout.height == 500
    assert style == {"height": "500px"}


def test_only_nonempty_optuna_studies_with_features_are_compatible():
    suitable = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    empty = SimpleNamespace(prefix="Optuna", empty=lambda: True, configspace={"x": 1})
    no_features = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={})
    group = SimpleNamespace(prefix="group", empty=lambda: False, configspace={"x": 1})

    assert planning.HPOPlanning.check_run_compatibility(suitable)
    assert not planning.HPOPlanning.check_run_compatibility(empty)
    assert not planning.HPOPlanning.check_run_compatibility(no_features)
    assert not planning.HPOPlanning.check_run_compatibility(group)
