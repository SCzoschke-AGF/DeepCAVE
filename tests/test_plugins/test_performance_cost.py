"""Native performance/cost plugin integration and plotting behavior."""

import pickle
from datetime import datetime, timedelta

import dash_bootstrap_components as dbc
import optuna
import pytest
from ConfigSpace import ConfigurationSpace
from dash import dcc
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial

from deepcave.config import Config
from deepcave.plugins.objective import performance_cost as plugin
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.converters.optuna import OptunaRun
from deepcave.runs.group import Group
from deepcave.runs.objective import Objective
from deepcave.utils.performance_cost import extract_trial_data


def _run(name="study", optimize="upper"):
    run = DeepCAVERun(
        name,
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}),
        objectives=[Objective("score", optimize=optimize), Objective("loss")],
    )
    for number, (duration, score, budget) in enumerate([(1, 4, 1), (2, 7, 2), (3, 6, 2)]):
        run.add(
            config={"x": number / 4},
            costs=[score, 10 - score],
            start_time=0,
            end_time=duration,
            budget=budget,
            seed=0,
        )
    return run


def _inputs(**changes):
    return {"objective_id": 0, "trials": "all", "show_pareto": True, **changes}


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk(child)


def test_control_row_and_single_plot_reuse_native_components():
    instance = plugin.PerformanceCost()
    controls = instance.get_input_layout(instance.register_input)
    components = [child for control in controls for child in _walk(control)]
    assert [child.label for child in components if isinstance(child, dbc.Checkbox)] == [
        "Pareto front"
    ]
    assert len([child for child in components if isinstance(child, dbc.Select)]) == 2
    assert not any(isinstance(child, dcc.Dropdown) for child in components)
    card = instance.get_output_layout(instance.register_output)
    assert card.children[0].children == "Performance vs. Computational Cost"
    assert len([child for child in _walk(card) if isinstance(child, dcc.Graph)]) == 1
    assert instance.load_inputs()["show_pareto"]["value"] is True
    assert all(item.id != instance.id for item in Config().PLUGINS["Objective Analysis"])


def test_dependency_inputs_expose_all_trials_and_native_budget_subsets():
    result = plugin.PerformanceCost().load_dependency_inputs(
        _run(), {}, {"trials": {"value": "stale"}}
    )
    assert result["trials"] == {
        "value": "all",
        "options": [
            {"label": "All trials", "value": "all"},
            {"label": "Budget 1", "value": "budget:0"},
            {"label": "Budget 2", "value": "budget:1"},
        ],
    }
    assert result["objective_id"]["value"] == 0


@pytest.fixture(autouse=True)
def _disable_image_export_and_supply_native_link(monkeypatch):
    monkeypatch.setattr(plugin, "save_image", lambda *_: None)
    monkeypatch.setattr(
        plugin.Configurations, "get_link", lambda run, config_id: f"/configs/{run.name}/{config_id}"
    )


def test_plot_keeps_trials_muted_and_draws_sorted_frontier_without_a_legend():
    run = _run()
    outputs = plugin.PerformanceCost.process(run, _inputs())
    figure = plugin.PerformanceCost.load_outputs(run, _inputs(), outputs)
    trials, pareto = figure.data
    assert list(trials.x) == [1, 2, 3]
    assert list(trials.y) == [4, 7, 6]
    assert trials.marker.opacity == 0.5
    assert list(pareto.x) == [1, 2]
    assert list(pareto.y) == [4, 7]
    assert pareto.mode == "lines+markers"
    assert "Trial 1" in pareto.hovertext[1]
    assert "score: 7" in pareto.hovertext[1]
    assert "Trial duration: 2 s" in pareto.hovertext[1]
    assert "Pareto-efficient" in pareto.hovertext[1]
    assert "<a href='/configs/study/1'" in pareto.hovertext[1]
    assert figure.layout.showlegend is False
    assert figure.layout.xaxis.title.text == "Trial duration [s]"
    assert figure.layout.yaxis.title.text == "score"
    assert not figure.layout.annotations


def test_unchecked_pareto_renders_scatter_only():
    run = _run()
    inputs = _inputs(show_pareto=False)
    outputs = plugin.PerformanceCost.process(run, inputs)
    figure = plugin.PerformanceCost.load_outputs(run, inputs, outputs)
    assert len(figure.data) == 1
    assert figure.data[0].mode == "markers"


def test_global_objective_change_updates_values_direction_and_plot_label():
    run = _run()
    instance = plugin.PerformanceCost()
    inputs = {"objective_id": {"value": 0}}
    instance.apply_global_main_objective(
        inputs, run, {"value": "objective:1"}, include_cache_signature=True
    )
    clean = _inputs(objective_id=inputs["objective_id"]["value"])
    outputs = instance.process(run, clean)
    figure = instance.load_outputs(run, clean, outputs)
    assert list(figure.data[0].y) == [6, 3, 4]
    assert outputs["pareto_indices"] == [0, 1]
    assert figure.layout.yaxis.title.text == "loss"
    assert inputs["_global_main_objective"]["value"] == "objective:1"


def test_budget_filter_changes_both_points_and_frontier():
    run = _run()
    inputs = _inputs(trials="budget:1")
    outputs = plugin.PerformanceCost.process(run, inputs)
    figure = plugin.PerformanceCost.load_outputs(run, inputs, outputs)
    assert list(figure.data[0].x) == [2, 3]
    assert list(figure.data[1].x) == [2]


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_trial_plots(count):
    run = _run()
    run.history = run.history[:count]
    outputs = plugin.PerformanceCost.process(run, _inputs())
    figure = plugin.PerformanceCost.load_outputs(run, _inputs(), outputs)
    if count:
        assert len(figure.data[1].x) == 1
        assert outputs["pareto_indices"] == [0]
    else:
        assert not figure.data
        assert "No valid completed trials" in figure.layout.annotations[0].text


def test_group_preserves_repeated_evaluations_and_trial_identity():
    first, second = _run("first"), _run("second")
    first.add(
        costs=[9, 1],
        config={"x": 0.0},
        budget=1,
        seed=0,
        start_time=0,
        end_time=4,
        allow_duplicate_trials=True,
    )
    group = Group("group", [first, second])
    outputs = plugin.PerformanceCost.process(group, _inputs())
    figure = plugin.PerformanceCost.load_outputs(group, _inputs(), outputs)
    assert len(outputs["trials"]) == 7
    assert list(figure.data[0].customdata[3]) == [0, 3]
    assert list(figure.data[0].customdata[4]) == [1, 0]
    assert "Run: second" in figure.data[0].hovertext[4]


def test_optuna_conversion_supplies_precise_duration_and_excludes_missing_timing(tmp_path):
    study = optuna.create_study(direction="maximize")
    for number in range(3):
        study.add_trial(
            create_trial(
                params={"x": number / 4},
                distributions={"x": FloatDistribution(0.0, 1.0)},
                value=number,
            )
        )
    start = datetime(2026, 1, 1)
    trials = study.get_trials(deepcopy=False)
    trials[0].datetime_start = start
    trials[0].datetime_complete = start + timedelta(microseconds=3000)
    trials[1].datetime_start = None
    trials[1].datetime_complete = start + timedelta(seconds=10)
    trials[2].datetime_start = start
    trials[2].datetime_complete = None
    with (tmp_path / "study.pkl").open("wb") as file:
        pickle.dump(study, file)
    run = OptunaRun.from_path(tmp_path)
    rows = extract_trial_data(run, 0)
    assert [row["trial_id"] for row in rows] == [0]
    assert rows[0]["duration"] == pytest.approx(0.003)
