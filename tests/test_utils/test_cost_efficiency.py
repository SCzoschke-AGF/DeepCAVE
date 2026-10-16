"""Independent extraction, elite ranking, screening, and fANOVA target tests."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Constant, EqualsCondition

from deepcave.runs import Status
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective
from deepcave.plugins.hyperparameter.importances import Importances
from deepcave.utils.cost_efficiency import (
    calculate_importances,
    duration_trend,
    elite_summary,
    importance_comparison,
    select_elite_indices,
)
from deepcave.utils.performance_cost import extract_trial_data, extract_trial_duration


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, 3.0),
        ({"start_time": None}, None),
        ({"end_time": None}, None),
        ({"end_time": -1}, None),
        ({"end_time": float("nan")}, None),
        ({"end_time": float("inf")}, None),
        ({"end_time": 0}, 0.0),
        ({"status": Status.PRUNED}, None),
        ({"status": Status.FAILED}, None),
    ],
)
def test_duration(changes, expected):
    trial = SimpleNamespace(status=Status.SUCCESS, start_time=0, end_time=3, costs=[0.003])
    trial.__dict__.update(changes)
    assert extract_trial_duration(trial) == expected


def test_precise_duration_is_retained_and_missing_time_does_not_fall_back():
    trial = SimpleNamespace(status=Status.SUCCESS, start_time=0, end_time=0, costs=[0.003])
    assert extract_trial_duration(trial, 0) == 0.003
    trial.costs = [None]
    assert extract_trial_duration(trial, 0) is None


@pytest.mark.parametrize(
    "values,fraction,maximize,expected",
    [
        ([3, 9, 6, 1], 0.5, True, [1, 2]),
        ([3, 9, 6, 1], 0.5, False, [3, 0]),
        ([-3, -9, -6, -1], 0.5, True, [3, 0]),
        ([3], 0.01, True, [0]),
        ([], 0.05, False, []),
        ([2, 2, 2, 1], 0.5, True, [0, 1]),
        ([1, 1, 1, 2], 0.5, False, [0, 1]),
        ([1, 2, 3], 0.5, False, [0, 1]),
    ],
)
def test_elite_direction_small_sets_rounding_and_boundary_ties(
    values, fraction, maximize, expected
):
    rows = [{"objective": value} for value in values]
    assert select_elite_indices(rows, fraction, maximize) == expected


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan"), float("inf")])
def test_invalid_elite_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        select_elite_indices([], fraction)


def test_screening_uses_strict_medians_and_sorts_cost_without_division():
    cost = {
        "candidate": 0.8,
        "expensive_important": 0.9,
        "cheap": 0.1,
        "median": 0.3,
        "second": 0.7,
    }
    objective = {
        "candidate": 0.0,
        "expensive_important": 0.9,
        "cheap": 0.1,
        "median": 0.3,
        "second": 0.05,
    }
    result = importance_comparison(cost, objective)
    assert result["cost_median"] == 0.7
    assert result["objective_median"] == 0.1
    assert [row["parameter"] for row in result["candidates"]] == ["candidate"]
    # Two candidates sorted by descending cost, then name.
    cost["second"] = 0.85
    cost["median"] = 0.5
    cost["middle"] = 0.6
    objective["middle"] = 0.4
    result = importance_comparison(cost, objective)
    assert [row["parameter"] for row in result["candidates"]] == ["second", "candidate"]
    assert all(row["classification"] == "Candidate" for row in result["candidates"])


def test_threshold_edges_equal_importances_and_missing_estimates():
    assert importance_comparison({"x": 0.5}, {"x": 0.5})["candidates"] == []
    result = importance_comparison({"x": 1, "invalid": np.nan}, {"x": 0, "invalid": 1})
    assert result["parameters"] == ["x"]
    assert importance_comparison({}, {})["cost_median"] is None


def test_elite_summary_and_zero_duration_ratio_are_json_safe():
    assert elite_summary([])["count"] == 0
    stats = elite_summary([{"duration": 1}, {"duration": 2}, {"duration": 10}])
    assert stats == {"count": 3, "minimum": 1, "median": 2, "maximum": 10, "ratio": 10}
    assert elite_summary([{"duration": 0}, {"duration": 1}])["ratio"] is None


def test_fanova_adapter_preserves_constant_parent_conditions_and_source_run():
    cs = ConfigurationSpace(space={"fixed": Constant("fixed", "on"), "x": (0.0, 1.0)})
    cs.add(EqualsCondition(cs["x"], cs["fixed"], "on"))
    run = DeepCAVERun("importance", configspace=cs, objectives=[Objective("score")])
    before = copy.deepcopy(cs.to_serialized_dict())
    rows = [
        {
            "configuration": {"fixed": "on", "x": index / 30},
            "duration": index + 1,
            "objective": 30 - index,
        }
        for index in range(30)
    ]
    results = calculate_importances(run, rows, n_trees=4)
    assert results["errors"] == {}
    for target in ("cost", "objective"):
        assert results[target]["x"] == pytest.approx(1)
        assert results[target]["fixed"] == 0
    assert run.configspace is cs
    assert cs.to_serialized_dict() == before
    assert run.history == []


@pytest.mark.parametrize("objective_id", [0, 1])
def test_normalized_targets_match_importances_for_objective_and_duration(objective_id):
    run = DeepCAVERun(
        "normalized",
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)}),
        objectives=[
            Objective("score", lower=0.0, upper=1.0, optimize="upper"),
            Objective("loss", lower=0.0, upper=1000.0),
            Objective("Time"),
        ],
    )
    for index, (x, y) in enumerate(np.random.RandomState(0).uniform(size=(30, 2))):
        run.add(
            config={"x": x, "y": y},
            costs=[0.58 + 0.001 * (0.7 * x + 0.3 * y), 100 + 30 * x * y, (index + 1) * 10],
            start_time=0, end_time=(index + 1) * 10, budget=0, seed=0,
        )
    rows = extract_trial_data(run, objective_id)
    for row in rows:
        row["configuration"] = dict(run.get_config(row["config_id"]))
    result = calculate_importances(run, rows, n_trees=4, objective_id=objective_id)
    assert result["errors"] == {}
    for target, selected_id in (("objective", objective_id), ("cost", 2)):
        native = Importances.process(
            run, {"objective_id1": selected_id, "objective_id2": -1, "method": "global", "n_trees": 4}
        )
        expected = {record["members"][0]: record["importance"] for record in native[0]}
        assert result[target] == pytest.approx(expected, abs=1e-12)


def test_importance_insufficient_data_and_constant_targets():
    run = DeepCAVERun("empty", configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}))
    assert calculate_importances(run, [])["cost"] == {}
    rows = [
        {"configuration": {"x": index / 10}, "duration": 1, "objective": 0} for index in range(10)
    ]
    result = calculate_importances(run, rows)
    assert result["cost"] == result["objective"] == {}
    assert "no variation" in result["errors"]["cost"]


def test_trend_handles_repeated_values_and_is_robust_to_duration_outliers():
    rows = [
        {"configuration": {"x": value}, "duration": duration}
        for value in (1, 2, 3)
        for duration in (1, 2, 3, 100)
    ]
    trend = duration_trend(rows, "x")
    assert trend["x"] == [1, 2, 3]
    assert trend["median"] == [2.5, 2.5, 2.5]
    assert duration_trend([], "x")["x"] == []
