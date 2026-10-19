"""Validate plans against independent Cartesian coverage counts and legal native values."""

import json
from copy import deepcopy
from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    ForbiddenEqualsClause,
    NormalFloatHyperparameter,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.evaluators import coverage_planning as planner
from deepcave.evaluators.combinatorial import discretize_combinatorial
from deepcave.evaluators.coverage import normalize_configurations, prepare_coverage
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.group import Group
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status


def binary_data(rows, dimensions=3):
    space = ConfigurationSpace()
    names = [f"p{i}" for i in range(dimensions)]
    space.add([CategoricalHyperparameter(name, [0, 1]) for name in names])
    return normalize_configurations(space, [dict(zip(names, row)) for row in rows])


def request(**overrides):
    return {
        "strategy": "maximize_coverage",
        "resolution": 2,
        "strength_mode": "custom",
        "target_strength": 2,
        "max_additional_trials": 8,
        **overrides,
    }


def assert_coverage_trajectory(data, plan):
    points, sizes = discretize_combinatorial(data, plan["resolution"])
    groups = list(combinations(range(len(sizes)), plan["target_strength"]))
    possible = {
        (group, values)
        for group in groups
        for values in product(*(range(sizes[column]) for column in group))
    }
    observed = {
        (group, tuple(point[column] for column in group)) for point in points for group in groups
    }
    assert plan["initial_coverage"] == len(observed) / len(possible)
    for trial in plan["trials"]:
        Configuration(data.configspace, trial["parameters"]).check_valid_configuration()
        encoded, _ = discretize_combinatorial(
            normalize_configurations(data.configspace, [trial["parameters"]]), plan["resolution"]
        )
        added = {(group, tuple(encoded[0, column] for column in group)) for group in groups}
        assert trial["new_combinations"] == len(added - observed) > 0
        observed |= added
        assert trial["cumulative_coverage"] == len(observed) / len(possible)
        assert trial["remaining_missing"] == len(possible - observed)
    assert plan["projected_coverage"] == len(observed) / len(possible)
    assert plan["missing_tuples_after"] == len(possible - observed)
    json.dumps(plan, allow_nan=False)


@pytest.mark.parametrize(
    "rows,expected",
    [([(0, 0, 0), (0, 1, 1), (1, 0, 1)], 1), ([(0, 0, 0), (0, 0, 1), (0, 1, 0)], 2)],
)
def test_minimize_trials_completes_known_binary_extensions(rows, expected):
    data = binary_data(rows)
    original = deepcopy(data.values)
    plan = planner.generate_coverage_trial_plan(
        data, request(strategy="minimize_trials", max_additional_trials=None)
    )
    assert len(plan["trials"]) == expected
    assert plan["target_complete"] and plan["optimality_proven"]
    assert plan["lower_bound_trials"] == expected
    assert data.values == original
    assert_coverage_trajectory(data, plan)


@pytest.mark.parametrize("strategy", ["maximize_coverage", "minimize_trials"])
def test_budget_below_lower_bound_returns_an_honest_improving_plan(strategy):
    data = binary_data([], dimensions=6)
    plan = planner.generate_coverage_trial_plan(data, request(strategy=strategy, target_strength=4))
    assert len(plan["trials"]) == 8
    assert plan["lower_bound_trials"] == 16
    assert plan["budget_target_impossible"]
    assert 0 < plan["projected_coverage"] <= 0.5
    assert not plan["target_complete"] and not plan["optimality_proven"]
    assert plan["termination_reason"] == "budget_exhausted"
    assert_coverage_trajectory(data, plan)


def test_next_and_maximum_strength_stop_without_zero_gain_proposals():
    data = binary_data([(0, 0, 0), (1, 1, 1)])
    plan = planner.generate_coverage_trial_plan(data, request(strength_mode="next"))
    assert plan["target_strength"] == 2
    assert plan["target_complete"]
    assert len(plan["trials"]) < 8
    assert_coverage_trajectory(data, plan)
    complete = binary_data(list(product([0, 1], repeat=3)))
    for mode in ("next", "maximum"):
        plan = planner.generate_coverage_trial_plan(complete, request(strength_mode=mode))
        assert plan["target_strength"] == 3
        assert plan["trials"] == [] and plan["projected_coverage"] == 1


def test_mixed_log_stepped_ordinal_and_constant_values_round_trip_reproducibly():
    space = ConfigurationSpace(seed=9)
    space.add(
        [
            UniformFloatHyperparameter("log", 0.001, 1, log=True),
            UniformIntegerHyperparameter("integer", 0, 8, meta={"step": 2}),
            UniformFloatHyperparameter("stepped", 0.1, 0.9, meta={"step": 0.2}),
            CategoricalHyperparameter("category", ["a", "b", "c"]),
            OrdinalHyperparameter("ordinal", ["small", "medium", "large"]),
            Constant("constant", 4),
        ]
    )
    original_rng = deepcopy(space.random.get_state())
    data = normalize_configurations(
        space,
        [
            {
                "log": 0.01,
                "integer": 2,
                "stepped": 0.3,
                "category": "a",
                "ordinal": "small",
                "constant": 4,
            }
        ],
    )
    plan = planner.generate_coverage_trial_plan(data, request(seed=42))
    assert planner.generate_coverage_trial_plan(data, request(seed=42)) == plan
    assert plan["target_strength"] == 2
    assert len(plan["search_space"]["parameters"]) == 5
    for trial in plan["trials"]:
        assert trial["parameters"]["constant"] == 4
        assert trial["parameters"]["integer"] % 2 == 0
        assert np.isclose(
            (trial["parameters"]["stepped"] - 0.1) / 0.2,
            round((trial["parameters"]["stepped"] - 0.1) / 0.2),
        )
    after = space.random.get_state()
    assert original_rng[0] == after[0]
    np.testing.assert_array_equal(original_rng[1], after[1])
    assert original_rng[2:] == after[2:]
    assert_coverage_trajectory(data, plan)


def test_normal_sampling_levels_use_the_same_cdf_for_generation_and_analysis():
    space = ConfigurationSpace()
    space.add(NormalFloatHyperparameter("x", mu=0.4, sigma=0.1, lower=0, upper=1))
    data = normalize_configurations(space, [{"x": 0.4}])
    plan = planner.generate_coverage_trial_plan(data, request(target_strength=1, resolution=4))
    assert plan["target_complete"]
    assert_coverage_trajectory(data, plan)


def test_failed_and_pruned_trials_do_not_close_gaps_and_duplicates_add_no_coverage():
    data = binary_data([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
    values = data.values + [data.values[0]]
    run = SimpleNamespace(
        configspace=data.configspace,
        history=[
            SimpleNamespace(config_id=i, status=status, seed=0, budget=1)
            for i, status in enumerate(
                [Status.SUCCESS, Status.SUCCESS, Status.SUCCESS, Status.FAILED, Status.SUCCESS]
            )
        ],
        get_config=lambda index: values[index],
    )
    prepared = prepare_coverage(run)
    plan = planner.generate_coverage_trial_plan(prepared, request())
    assert prepared.count == 4 and prepared.overview["unique_configurations"] == 4
    assert plan["initial_coverage"] == 0.75
    assert plan["trials"][0]["parameters"] == {"p0": 1, "p1": 1, "p2": 0}
    run.history[3].status = Status.PRUNED
    assert (
        planner.generate_coverage_trial_plan(prepare_coverage(run), request())["initial_coverage"]
        == 0.75
    )


def test_group_uses_merged_ranges_independently_of_member_sampler_metadata():
    runs = []
    for index, bounds in enumerate([(0.0, 0.4), (0.6, 1.0)]):
        run = DeepCAVERun(
            f"random-{index}",
            configspace=ConfigurationSpace(space={"x": bounds}),
            objectives=Objective("Score", lower=0, upper=1),
            meta={"sampler": "RandomSampler"},
        )
        run.add(costs=0.5, config={"x": 0.1 if index == 0 else 0.9}, seed=0, budget=0)
        runs.append(run)
    group = Group("merged-random", runs, require_configspace_congruence=False)
    assert planner.planning_source_error(group) is None
    data = prepare_coverage(group)
    plan = planner.generate_coverage_trial_plan(data, request(target_strength=1, resolution=4))
    assert data.count == 2
    assert plan["search_space"]["parameters"][0]["edges"] == [0, 0.25, 0.5, 0.75, 1]
    assert len(plan["trials"]) == 2
    assert all(0.25 <= trial["parameters"]["x"] < 0.75 for trial in plan["trials"])
    assert_coverage_trajectory(data, plan)
    runs[1].meta["sampler"] = "TPESampler"
    assert planner.planning_source_error(group) is None


@pytest.mark.parametrize("constraint", ["conditional", "forbidden"])
def test_constraints_are_rejected_before_candidate_generation(constraint):
    space = ConfigurationSpace()
    parent = CategoricalHyperparameter("parent", [0, 1], default_value=1)
    child = UniformFloatHyperparameter("child", 0, 1)
    space.add([parent, child])
    space.add(
        EqualsCondition(child, parent, 1)
        if constraint == "conditional"
        else ForbiddenEqualsClause(parent, 0)
    )
    data = normalize_configurations(space, [{"parent": 1, "child": 0.5}])
    with pytest.raises(ValueError, match="without constraints"):
        planner.generate_coverage_trial_plan(data, request())


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_additional_trials": None},
        {"max_additional_trials": 0},
        {"max_additional_trials": True},
        {"target_strength": 0},
        {"target_strength": 4},
        {"resolution": 5},
        {"solver_timeout": -1},
        {"solver_timeout": float("nan")},
        {"seed": -1},
    ],
)
def test_invalid_requests_do_not_produce_plans(overrides):
    with pytest.raises(ValueError):
        planner.generate_coverage_trial_plan(binary_data([(0, 0, 0)]), request(**overrides))


def test_timeout_retains_valid_progress_without_claiming_impossibility(monkeypatch):
    ticks = iter([0, 0, 0, 0, 0, 50])
    monkeypatch.setattr(planner, "monotonic", lambda: next(ticks, 50))
    data = binary_data([], dimensions=6)
    plan = planner.generate_coverage_trial_plan(data, request(target_strength=4))
    assert 0 < len(plan["trials"]) < 8
    assert plan["termination_reason"] == "timeout"
    assert not plan["optimality_proven"]
    assert_coverage_trajectory(data, plan)


def test_tuple_and_work_limits_are_reported_instead_of_approximating(monkeypatch):
    data = binary_data([], dimensions=6)
    monkeypatch.setattr(planner, "MAX_TUPLES", 10)
    with pytest.raises(ValueError, match="tuple limit"):
        planner.generate_coverage_trial_plan(data, request(target_strength=4))
    monkeypatch.setattr(planner, "MAX_TUPLES", 5_000_000)
    monkeypatch.setattr(planner, "MAX_GROUPS", 1)
    with pytest.raises(ValueError, match="work limit"):
        planner.generate_coverage_trial_plan(data, request(target_strength=4))


def test_large_space_missing_tuple_candidates_make_reproducible_positive_progress():
    data = binary_data([(0,) * 14], dimensions=14)
    plan = planner.generate_coverage_trial_plan(data, request(max_additional_trials=3, seed=3))
    assert len(plan["trials"]) == 3
    assert (
        planner.generate_coverage_trial_plan(data, request(max_additional_trials=3, seed=3)) == plan
    )
    assert_coverage_trajectory(data, plan)
