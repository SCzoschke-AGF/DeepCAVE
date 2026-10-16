from types import SimpleNamespace
from pathlib import Path

import pytest
from dash import dcc
from ConfigSpace import ConfigurationSpace

from deepcave.plugins.objective import cost_over_time
from deepcave.plugins.objective import performance_cost
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status


class DummyRun:
    def __init__(self, run_id, name, objective):
        self.id = run_id
        self.name = name
        self.history = []
        self._objective = objective

    def get_objective(self, _objective_id):
        return self._objective


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


def test_all_controls_share_the_compact_input_toolbar():
    from deepcave.config import Config

    plugins = Config().PLUGINS
    assert plugins["Summary"][0].name == "Run Metadata"
    assert plugins["Objective Analysis"][0].name == "Multi-Run Comparison"
    assert all(plugin.id != "performance_cost" for plugin in plugins["Objective Analysis"])
    registrations = []

    def register(name, *_args, **kwargs):
        registrations.append((name, kwargs))
        return name

    additional_runs = cost_over_time.CostOverTime.get_additional_run_input_layout(register)
    (toolbar,) = cost_over_time.CostOverTime.get_input_layout(register)

    assert "analysis-toolbar" in additional_runs.className
    assert "analysis-toolbar" in toolbar.className
    assert cost_over_time.CostOverTime.get_filter_layout(register) == []
    assert [name for name, _kwargs in registrations] == [
        "additional_runs",
        "budget_id",
        "xaxis",
        "show_pareto",
        "elite_fraction",
    ]
    assert registrations[-1][1]["filter"] is True
    budget = next(
        component for component in _walk_components(toolbar)
        if getattr(component, "id", None) == "budget_id"
    )
    assert budget.style == {"display": "none"}
    assert not any(
        getattr(component, "children", None) == "Budget"
        for component in _walk_components(toolbar)
    )

    dropdowns = [
        component
        for component in _walk_components(additional_runs)
        if isinstance(component, dcc.Dropdown)
    ]
    assert len(dropdowns) == 1
    assert dropdowns[0].multi is True


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

        def get_objectives(self):
            return [objective]

    run = ProcessRun("study-1", "Study 1", objective)
    for index, trial in enumerate(run.history):
        trial.start_time = 0
        trial.end_time = index + 1
        trial.config_id = index
        trial.additional = {}
    output = cost_over_time.CostOverTime.process(
        run,
        {"objective_id": 0, "budget_id": 0},
    )

    assert output["objective_values"] == pytest.approx([0.2, 0.4])
    assert [row["duration"] for row in output["performance_cost"]["trials"]] == [1, 2]


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

    for data in outputs.values():
        data["performance_cost"] = {"trials": []}
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
    assert figure.layout.xaxis3.title.text == "Trial duration [s]"


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

    outputs["study-1"]["performance_cost"] = {"trials": []}
    figure = cost_over_time.CostOverTime.load_outputs(
        [runs[0]],
        {"objective_id": 0, "xaxis": "times"},
        outputs,
    )

    assert {trace.legendgroup for trace in figure.data if trace.legendgroup} == {"study-1"}
    assert len([trace for trace in figure.data if trace.type == "histogram"]) == 1


def _comparison_run(name):
    run = DeepCAVERun(
        name,
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}),
        objectives=[Objective("score", optimize="upper"), Objective("loss")],
    )
    run.path = Path("/tmp") / name
    for index, (duration, score, budget) in enumerate([(1, 4, 1), (2, 7, 2), (3, 6, 2)]):
        run.add(
            config={"x": index / 4}, costs=[score, 10 - score],
            start_time=0, end_time=duration, budget=budget, seed=0,
        )
    return run


@pytest.mark.parametrize("objective_id", [0, 1])
def test_performance_cost_comparison_reuses_run_colors_frontiers_and_drilldown(
    monkeypatch, objective_id
):
    monkeypatch.setattr(cost_over_time, "save_image", lambda *_: None)
    monkeypatch.setattr(performance_cost, "save_image", lambda *_: None)
    monkeypatch.setattr(
        performance_cost.Configurations, "get_link",
        lambda run, config_id: f"/configs/{run.name}/{config_id}",
    )
    runs = [_comparison_run("first"), _comparison_run("second")]
    inputs = {"objective_id": objective_id, "budget_id": 1, "xaxis": "times"}
    outputs = {run.id: cost_over_time.CostOverTime.process(run, inputs) for run in runs}
    # Exercise persisted trajectory caches produced before the cost plot was integrated.
    outputs[runs[1].id].pop("performance_cost")
    figure = cost_over_time.CostOverTime.load_outputs(runs, inputs, outputs)
    trials = [trace for trace in figure.data if trace.xaxis == "x3" and trace.mode == "markers"]
    fronts = [trace for trace in figure.data if trace.xaxis == "x3" and trace.mode == "lines+markers"]
    assert len(trials) == len(fronts) == 2
    for index, run in enumerate(runs):
        assert list(trials[index].x) == [2, 3]
        assert list(fronts[index].x) == [2]
        assert list(trials[index].y) == ([7, 6] if objective_id == 0 else [3, 4])
        assert trials[index].legendgroup == fronts[index].legendgroup == run.id
        assert trials[index].marker.color == fronts[index].line.color == cost_over_time.get_color(index)
        assert f"<a href='/configs/{run.name}/1'" in trials[index].hovertext[0]
        elite = [
            trace for trace in figure.data
            if trace.xaxis == "x4" and trace.legendgroup == run.id
        ]
        assert [list(trace.x) for trace in elite] == [[2], [2]]
        assert elite[0].marker.color == cost_over_time.get_color(index)
        assert all(trace.marker.symbol != "star" for trace in elite)
        assert elite[1].mode == "lines+markers"
        assert "Objective rank: 1 / 2" in elite[0].hovertext[0]
    assert figure.layout.yaxis3.title.text == ("score" if objective_id == 0 else "loss")
    assert figure.layout.xaxis3.type == "linear"
    assert figure.layout.xaxis3.domain == figure.layout.xaxis.domain
    hidden = cost_over_time.CostOverTime.load_outputs(runs, {**inputs, "show_pareto": False}, outputs)
    assert len([trace for trace in hidden.data if trace.xaxis == "x3"]) == 2
    assert len([trace for trace in hidden.data if trace.xaxis == "x4"]) == 2
    single = cost_over_time.CostOverTime.load_outputs([runs[0]], inputs, outputs)
    assert {trace.legendgroup for trace in single.data} == {runs[0].id}


@pytest.mark.parametrize("objective_id", [0, 1])
def test_objective_overview_elite_fraction_reuses_compute_cost_selection(monkeypatch, objective_id):
    monkeypatch.setattr(cost_over_time, "save_image", lambda *_: None)
    monkeypatch.setattr(performance_cost, "save_image", lambda *_: None)
    monkeypatch.setattr(performance_cost.Configurations, "get_link", lambda *_: "/configs/study")
    run = _comparison_run("study")
    for index in range(17):
        run.add(
            config={"x": (index + 1) / 20}, costs=[index + 10, -index],
            start_time=0, end_time=index + 4, budget=2, seed=1,
        )
    inputs = {"objective_id": objective_id, "budget_id": 1, "xaxis": "times"}
    outputs = {run.id: cost_over_time.CostOverTime.process(run, inputs)}
    for fraction, count in [("0.05", 1), ("0.2", 4)]:
        view_inputs = {**inputs, "elite_fraction": fraction}
        figure = cost_over_time.CostOverTime.load_outputs([run], view_inputs, outputs)
        elite = [trace for trace in figure.data if trace.xaxis == "x4"]
        expected, _ = cost_over_time.elite_figure(
            run, view_inputs, outputs[run.id]["performance_cost"]["trials"],
            run.get_objective(objective_id).optimize == "upper",
        )
        assert len(elite[0].x) == count
        assert list(elite[0].x) == list(expected.data[0].x)
        assert list(elite[0].y) == list(expected.data[0].y)
        assert list(elite[-1].customdata) == list(expected.data[-1].customdata)


def test_combined_budget_cost_plot_uses_highest_budget_per_configuration():
    run = _comparison_run("study")
    run.add(
        config={"x": 0.0}, costs=[8, 2], start_time=0, end_time=4, budget=2, seed=0,
    )
    data = cost_over_time.performance_cost_data(run, {"objective_id": 0, "budget_id": 2})
    assert [row["duration"] for row in data["trials"]] == [2, 3, 4]
    assert all(row["budget"] == 2 for row in data["trials"])
