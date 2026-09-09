from types import SimpleNamespace

import pytest

from deepcave.plugins.objective import cost_over_time
from deepcave.runs.status import Status


class DummyRun:
    def __init__(self, run_id, name, objective):
        self.id = run_id
        self.name = name
        self.history = []
        self._objective = objective

    def get_objective(self, _objective_id):
        return self._objective


def test_process_collects_successful_objective_values():
    objective = SimpleNamespace(name="CV Average Precision", optimize="upper")

    class ProcessRun(DummyRun):
        history = [
            SimpleNamespace(status=Status.SUCCESS, budget=0.0, costs=[0.2]),
            SimpleNamespace(status=Status.SUCCESS, budget=0.0, costs=[0.4]),
            SimpleNamespace(status=Status.FAILED, budget=0.0, costs=[None]),
        ]

        def __init__(self, run_id, name, objective):
            self.id = run_id
            self.name = name
            self._objective = objective

        def get_budget(self, budget_id):
            assert budget_id == 0
            return 0.0

        def get_trajectory(self, objective, budget):
            assert budget == 0.0
            return [1.0, 2.0], [0.2, 0.4], [0.0, 0.0], [0, 1], [0, 1]

        def get_objective_id(self, selected_objective):
            assert selected_objective is objective
            return 0

    output = cost_over_time.CostOverTime.process(
        ProcessRun("study-1", "Study 1", objective),
        {"objective_id": 0, "budget_id": 0},
    )

    assert output["objective_values"] == pytest.approx([0.2, 0.4])


def test_load_outputs_overlays_selected_study_distributions(monkeypatch):
    monkeypatch.setattr(cost_over_time, "save_image", lambda *_args: None)
    objective = SimpleNamespace(name="CV Average Precision", optimize="upper")
    runs = [
        DummyRun("study-1", "Study 1", objective),
        DummyRun("study-2", "Study 2", objective),
    ]
    outputs = {
        "study-1": {
            "times": [1.0, 2.0],
            "costs_mean": [0.2, 0.4],
            "costs_std": [0.0, 0.0],
            "ids": [0, 1],
            "config_ids": [0, 1],
            "objective_values": [0.1, 0.2, 0.4],
        },
        "study-2": {
            "times": [1.5, 3.0],
            "costs_mean": [0.3, 0.5],
            "costs_std": [0.0, 0.0],
            "ids": [0, 1],
            "config_ids": [0, 1],
            "objective_values": [0.15, 0.3, 0.5],
        },
    }

    figure = cost_over_time.CostOverTime.load_outputs(
        runs,
        {
            "objective_id": 0,
            "xaxis": "trials",
        },
        outputs,
    )

    histograms = [trace for trace in figure.data if trace.type == "histogram"]
    running_best = [
        trace for trace in figure.data if trace.type == "scatter" and trace.showlegend is not False
    ]
    assert [list(trace.x) for trace in histograms] == [
        [0.1, 0.2, 0.4],
        [0.15, 0.3, 0.5],
    ]
    assert all(trace.bingroup == "objective-distribution" for trace in histograms)
    assert all(trace.showlegend is False for trace in histograms)
    assert [trace.name for trace in running_best] == ["Study 1", "Study 2"]
    assert {trace.legendgroup for trace in figure.data} == {"study-1", "study-2"}
    assert sum(trace.showlegend is True for trace in figure.data) == 2
    assert len(figure.layout.shapes) == 2
    assert figure.layout.barmode == "overlay"
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y < 0
    assert figure.layout.legend.x == pytest.approx(0)
    assert figure.layout.margin.b >= 120
    assert figure.layout.xaxis2.title.text == "Number of evaluated trials"


def test_load_outputs_only_renders_passed_studies(monkeypatch):
    monkeypatch.setattr(cost_over_time, "save_image", lambda *_args: None)
    objective = SimpleNamespace(name="Objective", optimize="lower")
    runs = [
        DummyRun("study-1", "Study 1", objective),
        DummyRun("study-2", "Study 2", objective),
    ]
    outputs = {
        "study-1": {
            "times": [1.0],
            "costs_mean": [0.2],
            "costs_std": [0.0],
            "ids": [0],
            "config_ids": [0],
            "objective_values": [0.2],
        }
    }

    figure = cost_over_time.CostOverTime.load_outputs(
        [runs[0]],
        {"objective_id": 0, "xaxis": "times"},
        outputs,
    )

    assert {trace.legendgroup for trace in figure.data if trace.legendgroup} == {"study-1"}
    assert len([trace for trace in figure.data if trace.type == "histogram"]) == 1
