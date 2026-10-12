"""Trial extraction, subset filtering, and UI-independent Pareto calculations."""

from types import SimpleNamespace

import pytest

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.utils.performance_cost import (
    extract_trial_data,
    filter_trial_data,
    pareto_indices,
    prepare_plot_data,
)


@pytest.mark.parametrize(
    "points,maximize,expected",
    [
        ([(3, 8), (1, 4), (2, 7), (4, 6)], True, [1, 2, 0]),
        ([(3, 2), (1, 6), (2, 3), (4, 4)], False, [1, 2, 0]),
        ([(1, 2), (1, 3), (2, 3)], True, [1]),
        ([(1, 2), (1, 3), (2, 2)], False, [0]),
        ([(2, 3), (1, 3)], True, [1]),
        ([(2, 3), (1, 3)], False, [1]),
        ([(1, 3), (1, 3), (2, 4)], True, [0, 1, 2]),
        ([(1, 3), (1, 3), (2, 2)], False, [0, 1, 2]),
        ([(1, 3)], True, [0]),
        ([(1, 3)], False, [0]),
        ([], False, []),
    ],
)
def test_pareto_respects_direction_strict_dominance_and_duplicates(points, maximize, expected):
    assert pareto_indices(points, maximize=maximize) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None])
def test_pareto_rejects_invalid_inputs(value):
    with pytest.raises(ValueError, match="finite"):
        pareto_indices([(1, value)])


def _trial(**changes):
    values = {
        "status": Status.SUCCESS,
        "start_time": 0.0,
        "end_time": 2.0,
        "costs": [0.8],
        "config_id": 0,
        "budget": 1.0,
        "additional": {"optuna_trial_number": 17},
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _run(trials, objectives=None):
    return SimpleNamespace(
        history=trials,
        get_objectives=lambda: objectives or [Objective("score", optimize="upper")],
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": Status.FAILED},
        {"status": Status.PRUNED},
        {"status": Status.NOT_EVALUATED},
        {"start_time": None},
        {"end_time": None},
        {"start_time": float("nan")},
        {"end_time": float("inf")},
        {"end_time": -1},
        {"costs": [None]},
        {"costs": [float("nan")]},
        {"costs": [float("inf")]},
        {"costs": [-float("inf")]},
        {"costs": []},
    ],
)
def test_extraction_excludes_invalid_and_incomplete_trials(changes):
    run = _run([_trial(), _trial(**changes)])
    rows = extract_trial_data(run, 0)
    assert len(rows) == 1
    assert rows[0]["trial_id"] == 17
    assert rows[0]["objective"] == 0.8
    assert rows[0]["duration"] == 2.0


def test_extraction_reuses_precise_time_and_does_not_invent_missing_duration():
    run = _run(
        [
            _trial(costs=[0.8, 0.003], end_time=0.0),
            _trial(costs=[0.9, None]),
            _trial(costs=[0.9, -1]),
            _trial(costs=[0.9, float("inf")]),
        ],
        [Objective("score"), Objective("Time")],
    )
    rows = extract_trial_data(run, 0)
    assert [row["duration"] for row in rows] == [0.003]


def test_extraction_preserves_duplicates_and_uses_history_id_without_optuna_number():
    rows = extract_trial_data(_run([_trial(additional={}), _trial(additional={})]), 0)
    assert [row["trial_id"] for row in rows] == [0, 1]
    assert prepare_plot_data(rows)["pareto_indices"] == [0, 1]


def test_pareto_is_computed_only_from_filtered_visible_subset():
    rows = extract_trial_data(
        _run(
            [
                _trial(costs=[10], end_time=1, budget=1),
                _trial(costs=[5], end_time=2, budget=2),
                _trial(costs=[6], end_time=3, budget=2),
            ]
        ),
        0,
    )
    assert prepare_plot_data(filter_trial_data(rows), maximize=True)["pareto_indices"] == [0]
    selected = prepare_plot_data(filter_trial_data(rows, budget=2), maximize=True)
    assert len(selected["trials"]) == 2
    assert selected["pareto_indices"] == [0, 1]


@pytest.mark.parametrize(
    "durations,expected",
    [
        ([], "linear"),
        ([1], "linear"),
        ([1, 10], "linear"),
        ([1, 100], "log"),
        ([0, 1, 100], "linear"),
    ],
)
def test_log_scale_requires_a_wide_positive_range(durations, expected):
    rows = [{"duration": duration, "objective": 1} for duration in durations]
    result = prepare_plot_data(rows)
    assert result["xaxis_type"] == expected
    assert len(result["trials"]) == len(durations)


def test_zero_duration_remains_valid_and_linear():
    rows = extract_trial_data(_run([_trial(end_time=0)]), 0)
    result = prepare_plot_data(rows)
    assert result["trials"][0]["duration"] == 0
    assert result["pareto_indices"] == [0]
    assert result["xaxis_type"] == "linear"
