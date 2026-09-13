"""Tests for efficient running-best trajectory calculation."""

from typing import Optional

import ConfigSpace
import numpy as np
import pytest

from deepcave.runs import AbstractRun, Status
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective


def _make_run(optimize: str = "lower") -> DeepCAVERun:
    configspace = ConfigSpace.ConfigurationSpace()
    configspace.add(ConfigSpace.Float("x", bounds=(0.0, 1.0)))
    return DeepCAVERun(
        "trajectory",
        configspace=configspace,
        objectives=Objective("score", lower=0.0, upper=1.0, optimize=optimize),
    )


def _legacy_trajectory(
    run: AbstractRun,
    objective: Objective,
    budget: float,
    seed: Optional[int] = None,
):
    """Calculate the former prefix-by-prefix implementation as a parity oracle."""
    order = sorted(enumerate(run.history), key=lambda item: item[1].end_time)
    current_cost = np.inf if objective.optimize == "lower" else -np.inf
    times = []
    costs = []
    ids = []
    config_ids = []

    for index, (trial_id, trial) in enumerate(order):
        try:
            _, cost = run.get_incumbent(
                objectives=objective,
                budget=budget,
                seed=seed,
                selected_ids=[item[0] for item in order[: index + 1]],
            )
        except RuntimeError:
            continue
        improvement = cost < current_cost if objective.optimize == "lower" else cost > current_cost
        if improvement:
            current_cost = cost
            times.append(trial.end_time)
            costs.append(cost)
            ids.append(trial_id)
            config_ids.append(trial.config_id)

    return times, costs, [0.0] * len(costs), ids, config_ids


@pytest.mark.parametrize("optimize", ["lower", "upper"])
def test_running_best_matches_previous_behavior_for_seeds_and_duplicates(optimize: str) -> None:
    run = _make_run(optimize)
    trials = [
        (0.1, 0, 0.8, 4.0, Status.SUCCESS),
        (0.2, 0, 0.7, 1.0, Status.SUCCESS),
        (0.1, 1, 0.4, 5.0, Status.SUCCESS),
        (0.2, 1, 0.9, 2.0, Status.SUCCESS),
        (0.3, 0, None, 3.0, Status.FAILED),
        (0.2, 1, 0.3, 6.0, Status.SUCCESS),
    ]
    for x, seed, cost, end_time, status in trials:
        run.add(
            costs=cost,
            config={"x": x},
            budget=0.0,
            seed=seed,
            end_time=end_time,
            status=status,
            allow_duplicate_trials=True,
        )

    objective = run.get_objective(0)
    assert objective is not None
    expected = _legacy_trajectory(run, objective, budget=0.0)

    assert run.get_trajectory(objective, budget=0.0) == expected


def test_running_best_no_longer_recomputes_incumbent_for_every_trial(monkeypatch) -> None:
    run = _make_run()
    for index in range(100):
        run.add(
            costs=1.0 - index / 100,
            config={"x": index / 100},
            budget=0.0,
            seed=0,
            end_time=float(index),
        )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("get_incumbent must not be called while building a trajectory")

    monkeypatch.setattr(run, "get_incumbent", fail_if_called)
    objective = run.get_objective(0)
    assert objective is not None

    times, costs, _, ids, _ = run.get_trajectory(objective, budget=0.0)

    assert len(times) == len(costs) == len(ids) == 100
