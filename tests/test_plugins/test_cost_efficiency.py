"""Native page structure, linked controls, direction, trial identity and plotting tests."""

import json
from types import SimpleNamespace

import dash_bootstrap_components as dbc
import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, EqualsCondition
from dash import dash_table, dcc

from deepcave.config import Config
from deepcave.plugins.cost_efficiency import (
    CostEfficiency,
    elite_figure,
    importance_figures,
    parameter_figure,
)
from deepcave.plugins.objective.performance_cost import Configurations
from deepcave.runs import Status
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.group import Group
from deepcave.runs.objective import Objective


def _run(name="study"):
    cs = ConfigurationSpace(space={"x": (0.0, 1.0), "kind": ["slow", "fast"]})
    run = DeepCAVERun(
        name, configspace=cs, objectives=[Objective("score", optimize="upper"), Objective("loss")]
    )
    for index in range(20):
        run.add(
            config={"x": index / 20, "kind": "slow" if index % 2 else "fast"},
            costs=[index, 20 - index],
            start_time=0,
            end_time=index + 1,
            budget=1 if index < 10 else 2,
            seed=0,
        )
    return run


def _inputs(**changes):
    return {
        "objective_id": 0,
        "trials": "all",
        "parameter": "x",
        "elite_fraction": "0.05",
        "show_pareto": True,
        "show_trend": False,
        "mark_elite": False,
        "detail_view": 0,
        "importance_view": 0,
        **changes,
    }


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk(child)


@pytest.fixture(autouse=True)
def _native_links_and_no_image_export(monkeypatch):
    from deepcave.plugins.objective import performance_cost

    monkeypatch.setattr(performance_cost, "save_image", lambda *_: None)
    monkeypatch.setattr(
        Configurations, "get_link", lambda run, config_id: f"/configs/{run.name}/{config_id}"
    )


def test_page_registration_structure_and_global_objective_only():
    plugin = CostEfficiency()
    assert plugin.name == "Compuational Cost"
    assert [item.name for item in Config().PLUGINS["Hyperparameter Analysis"]] == [
        "1-D Analyses",
        "2-D Analyses",
        "HPO Planner",
        "Compuational Cost",
        "Importances",
        "Ablation Paths",
        "Partial Dependencies",
    ]
    assert "Cost & Efficiency" not in Config().PLUGINS
    inputs = plugin.get_input_layout(plugin.register_input)
    filters = plugin.get_filter_layout(lambda a, b: plugin.register_input(a, b, filter=True))
    components = [child for control in inputs + filters for child in _walk(control)]
    assert [child.label for child in components if isinstance(child, dbc.Checkbox)] == [
        "Pareto front",
        "Duration trend",
        "Mark elite trials",
    ]
    assert not any(
        getattr(child, "id", None) == plugin.get_internal_input_id("objective_id")
        for child in components
    )
    assert {name for name, attribute, filtered, _ in plugin.inputs if filtered} == {
        "parameter",
        "show_pareto",
        "elite_fraction",
        "show_trend",
        "mark_elite",
        "detail_view",
        "importance_view",
    }
    page = plugin.get_output_layout(plugin.register_output)
    assert len([item for item in _walk(page) if isinstance(item, dcc.Graph)]) == 6
    view = next(
        item for item in _walk(page)
        if isinstance(item, dbc.RadioItems)
        and item.id == plugin.get_internal_input_id("detail_view")
    )
    assert view.id == plugin.get_internal_input_id("detail_view")
    assert view.value == 0
    assert view.options == [{"label": "Table", "value": 0}, {"label": "Objective", "value": 1}]
    assert view.className == "multi-metrics-segmented"
    assert len([item for item in _walk(page) if isinstance(item, dash_table.DataTable)]) == 1
    rows = [child for child in page.children if isinstance(child, dbc.Row)]
    assert [
        next(
            item.id for item in _walk(column) if isinstance(item, (dcc.Graph, dash_table.DataTable))
        )
        for row in rows
        for column in row.children
    ] == [
        plugin.get_internal_output_id("cost_importance"),
        plugin.get_internal_output_id("parameter_duration"),
        plugin.get_internal_output_id("comparison"),
        plugin.get_internal_output_id("candidates"),
        plugin.get_internal_output_id("performance"),
        plugin.get_internal_output_id("elite"),
    ]
    table = next(item for item in _walk(page) if isinstance(item, dash_table.DataTable))
    assert [column["id"] for column in table.columns] == [
        "parameter",
        "cost_importance",
        "objective_importance",
    ]
    assert table.page_action == "none"
    assert table.style_data_conditional[0]["if"]["filter_query"] == "{cost_candidate} = 1"
    assert plugin.load_inputs()["show_trend"]["value"] is True
    assert (
        next(
            child.value
            for child in components
            if isinstance(child, dbc.Checkbox) and child.label == "Duration trend"
        )
        is True
    )
    assert plugin.main_objective_input == "objective_id"


def test_dependency_selects_parameter_and_keeps_valid_previous_selection():
    plugin, run = CostEfficiency(), _run()
    result = plugin.load_dependency_inputs(run, {}, {"parameter": {"value": "kind"}})
    assert result["parameter"]["value"] == "kind"
    assert result["trials"]["value"] == "all"
    assert len(result["trials"]["options"]) == 3
    result = plugin.load_dependency_inputs(run, {}, {"parameter": {"value": "missing"}})
    assert result["parameter"]["value"] in run.configspace


def test_process_serializes_and_elite_plot_contains_no_nonelite_trials():
    run, inputs = _run(), _inputs(elite_fraction="0.2")
    outputs = CostEfficiency.process(run, inputs)
    json.dumps(outputs, allow_nan=False)
    figures = CostEfficiency.load_outputs(run, inputs, outputs)
    assert len(figures["performance"]["figure"].data[0].x) == 20
    elite, pareto = figures["elite"]["figure"].data
    assert list(elite.y) == [19, 18, 17, 16]
    assert list(pareto.x) == [17, 18, 19, 20]
    assert list(pareto.y) == [16, 17, 18, 19]
    assert all(trace.marker.symbol != "star" for trace in figures["elite"]["figure"].data)
    assert "Objective rank: 4 / 20" in elite.hovertext[-1]
    assert "<a href='/configs/study/16'" in elite.hovertext[-1]
    assert "Elite trials: 4" in figures["elite_summary"]["children"]
    assert outputs["importances"]["errors"] == {}


def test_objective_direction_and_budget_filter_apply_to_every_analysis():
    run, inputs = _run(), _inputs(objective_id=1, trials="budget:0", elite_fraction="0.2")
    outputs = CostEfficiency.process(run, inputs)
    figures = CostEfficiency.load_outputs(run, inputs, outputs)
    assert outputs["maximize"] is False
    assert len(outputs["trials"]) == 10
    assert list(figures["elite"]["figure"].data[0].y) == [11, 12]
    assert figures["performance"]["figure"].layout.yaxis.title.text == "loss"
    assert list(figures["parameter_duration"]["figure"].data[0].y) == list(range(1, 11))
    assert list(figures["elite"]["figure"].data[1].x) == [9, 10]


@pytest.mark.parametrize(
    "maximize,objective_id,expected_durations",
    [(True, 0, [1, 3]), (False, 1, [1, 2, 4])],
)
def test_elite_pareto_front_respects_direction_and_visibility(
    maximize, objective_id, expected_durations
):
    run = _run()
    rows = CostEfficiency.process(run, _inputs())["trials"][:4]
    for row, duration, objective in zip(rows, [1, 2, 3, 4], [3, 2, 4, 1]):
        row.update(duration=duration, objective=objective)
    inputs = _inputs(objective_id=objective_id, elite_fraction="1")
    figure, _ = elite_figure(run, inputs, rows, maximize)
    assert list(figure.data[1].x) == expected_durations
    assert figure.data[1].name == "Pareto front"
    assert figure.data[1].mode == "lines+markers"
    assert [trace.name for trace in figure.data] == ["Elite trials", "Pareto front"]
    hidden, _ = elite_figure(run, {**inputs, "show_pareto": False}, rows, maximize)
    assert [trace.name for trace in hidden.data] == ["Elite trials"]


def test_singleton_elite_pareto_front_remains_visible_without_star():
    run = _run()
    rows = CostEfficiency.process(run, _inputs())["trials"][:4]
    for index, row in enumerate(rows):
        row.update(duration=index + 1, objective=4 - index)
    figure, _ = elite_figure(run, _inputs(elite_fraction="1"), rows, True)
    elite, front = figure.data
    assert list(front.x) == [1]
    assert list(front.y) == [4]
    assert front.mode == "lines+markers"
    assert front.marker.size > elite.marker.size
    assert "Pareto-efficient" in front.hovertext[0]
    assert all(trace.marker.symbol != "star" for trace in figure.data)


def test_filters_redraw_from_same_cached_data_and_categorical_distribution():
    run, inputs = _run(), _inputs()
    outputs = CostEfficiency.process(run, inputs)
    rendered = CostEfficiency.load_outputs(
        run, _inputs(parameter="kind", show_pareto=False, elite_fraction="0.1"), outputs
    )
    assert rendered["parameter_duration"]["figure"].data[0].type == "box"
    assert list(rendered["parameter_duration"]["figure"].layout.xaxis.categoryarray) == [
        "slow",
        "fast",
    ]
    assert len(rendered["performance"]["figure"].data) == 1
    assert len(rendered["elite"]["figure"].data[0].x) == 2


def test_importance_click_payload_colors_and_candidate_thresholds_match():
    run = _run()
    importances = {
        "cost": {"x": 0.9, "kind": 0.1},
        "objective": {"x": 0.1, "kind": 0.9},
        "errors": {},
    }
    from deepcave.utils.cost_efficiency import importance_comparison

    outputs = {
        "importances": importances,
        "comparison": importance_comparison(importances["cost"], importances["objective"]),
    }
    cost, comparison = importance_figures(run, outputs)
    assert list(cost.data[0].customdata) == ["x", "kind"]
    assert list(comparison.data[0].customdata) == ["kind", "x"]
    color_by_name = dict(zip(cost.data[0].customdata, cost.data[0].marker.color))
    assert all(
        color_by_name[name] == color
        for name, color in zip(comparison.data[0].customdata, comparison.data[0].marker.color)
    )
    assert not comparison.layout.shapes
    assert not comparison.layout.annotations
    assert comparison.layout.xaxis.title.text == "Cost Importance"
    assert comparison.layout.yaxis.title.text == "Objective Importance"
    assert list(comparison.data[0].x) == [0.1, 0.9]
    assert list(comparison.data[0].y) == [0.9, 0.1]
    assert outputs["comparison"]["candidates"][0]["parameter"] == "x"


def test_importance_table_includes_non_candidates_and_marks_candidates():
    from deepcave.utils.cost_efficiency import importance_comparison

    run = _run()
    outputs = CostEfficiency.process(run, _inputs())
    outputs["importances"] = {
        "cost": {"x": 0.9, "kind": 0.1},
        "objective": {"x": 0.1, "kind": 0.9},
        "errors": {},
    }
    outputs["comparison"] = importance_comparison(
        outputs["importances"]["cost"], outputs["importances"]["objective"]
    )
    rendered = CostEfficiency.load_outputs(run, _inputs(), outputs)
    rows = rendered["candidates"]["data"]
    assert [row["parameter"] for row in rows] == ["x", "kind"]
    assert [row["cost_candidate"] for row in rows] == [1, 0]
    assert [row["id"] for row in rows] == ["x", "kind"]
    assert all("classification" not in row for row in rows)
    styles = rendered["candidates"]["style_data_conditional"]
    colors = dict(zip(
        rendered["cost_importance"]["figure"].data[0].customdata,
        rendered["cost_importance"]["figure"].data[0].marker.color,
    ))
    for index, row in enumerate(rows):
        for state in ("active", "selected"):
            style = next(s for s in styles if s["if"] == {"row_index": index, "state": state})
            assert style["border"] == f"1px solid {colors[row['parameter']]}"
            assert style["backgroundColor"] == colors[row["parameter"]].replace(", 1)", ", 0.25)")
    outputs["importances"]["cost"] = {}
    outputs["comparison"] = importance_comparison({}, outputs["importances"]["objective"])
    rows = CostEfficiency.load_outputs(run, _inputs(), outputs)["candidates"]["data"]
    assert {row["parameter"] for row in rows} == set(run.configspace.keys())
    assert all(row["cost_importance"] is None and row["cost_candidate"] == 0 for row in rows)


def test_group_preserves_trial_identity_for_elite_drilldown():
    first, second = _run("first"), _run("second")
    group = Group("both", [first, second])
    outputs = CostEfficiency.process(group, _inputs())
    assert len(outputs["trials"]) == 40
    figure, _ = elite_figure(group, _inputs(), outputs["trials"], True)
    assert len(figure.data[0].x) == 2
    assert list(figure.data[0].customdata[0]) == [0, 19]
    assert list(figure.data[0].customdata[1]) == [1, 19]
    assert "<a href='/configs/second/19'" in figure.data[0].hovertext[1]


def test_zero_importance_parameter_has_clickable_label_in_bar_order():
    run = _run()
    cost, _ = importance_figures(
        run,
        {
            "importances": {"cost": {"kind": 0, "x": 0.9}, "objective": {}, "errors": {}},
            "comparison": {"parameters": []},
        },
    )
    assert cost.layout.yaxis.showticklabels is False
    assert list(cost.layout.yaxis.categoryarray) == ["x", "kind"]
    assert [label.name for label in cost.layout.annotations] == list(cost.data[0].customdata)
    for label in cost.layout.annotations:
        assert label.text == label.name == label.y
        assert label.captureevents is True
        assert label.xref == "paper" and label.x == 0 and label.xanchor == "right"
        assert label.yref == "y" and label.yanchor == "middle"
        assert label.font.size == cost.layout.yaxis.tickfont.size


@pytest.mark.parametrize("count", [0, 1])
def test_empty_or_single_trial_retains_available_configuration_analysis(count):
    run = _run()
    run.history = run.history[:count]
    outputs = CostEfficiency.process(run, _inputs())
    rendered = CostEfficiency.load_outputs(run, _inputs(), outputs)
    assert rendered["cost_importance"]["figure"].layout.annotations
    if count:
        assert len(rendered["elite"]["figure"].data[0].x) == 1
    else:
        assert rendered["elite"]["figure"].layout.annotations


def test_conditional_parameter_omits_inactive_trials_and_keeps_numeric_plot():
    cs = ConfigurationSpace(space={"x": (0.0, 1.0), "kind": ["on", "off"]})
    cs.add(EqualsCondition(cs["x"], cs["kind"], "on"))
    run = DeepCAVERun("conditional", configspace=cs)
    rows = [
        {"trial_id": 1, "configuration": {"kind": "on", "x": 0.3}, "duration": 1, "objective": 1},
        {"trial_id": 2, "configuration": {"kind": "off"}, "duration": 100, "objective": 1},
    ]
    figure = parameter_figure(run, rows, "x")
    assert list(figure.data[0].x) == [0.3]
    assert figure.layout.yaxis.type == "linear"


def test_invalid_trial_filtering_is_shared_by_importance_and_configuration_views():
    run = _run()
    run.history[0].status = Status.PRUNED
    run.history[1].costs[0] = np.nan
    run.history[2].end_time = -1
    outputs = CostEfficiency.process(run, _inputs())
    assert len(outputs["trials"]) == 17
    assert {row["history_id"] for row in outputs["trials"]} == set(range(3, 20))


def test_optional_trend_only_for_numerical_parameters():
    run = _run()
    outputs = CostEfficiency.process(run, _inputs())
    numerical = parameter_figure(run, outputs["trials"], "x", True)
    assert len(numerical.data) == 3
    assert numerical.data[-1].name == "Binned median"
    assert numerical.data[1].fill == "toself"
    assert len(parameter_figure(run, outputs["trials"], "kind", True).data) == 1


@pytest.mark.parametrize("objective_id,parameter", [(0, "x"), (1, "x"), (0, "kind")])
def test_mark_elite_uses_selected_fraction_and_objective_direction(objective_id, parameter):
    run = _run()
    inputs = _inputs(objective_id=objective_id, parameter=parameter, elite_fraction="0.2")
    outputs = CostEfficiency.process(run, inputs)
    unmarked = CostEfficiency.load_outputs(run, inputs, outputs)["parameter_duration"]["figure"]
    marked = CostEfficiency.load_outputs(run, {**inputs, "mark_elite": True}, outputs)
    duration = marked["parameter_duration"]["figure"]
    assert len(duration.data) == len(unmarked.data) + 1
    assert list(duration.data[0].y) == list(unmarked.data[0].y)
    highlights = duration.data[-1]
    assert highlights.name == "Elite trials"
    assert list(highlights.y) == list(marked["elite"]["figure"].data[0].x)
    assert len(highlights.y) == 4
    assert highlights.marker.opacity == 1
    assert highlights.marker.symbol == "diamond"
    fewer = CostEfficiency.load_outputs(
        run, {**inputs, "mark_elite": True, "elite_fraction": "0.05"}, outputs
    )["parameter_duration"]["figure"]
    assert len(fewer.data[-1].y) == 1


@pytest.mark.parametrize("objective_id,parameter", [(0, "x"), (1, "x"), (0, "kind")])
def test_objective_panel_shares_parameter_trend_and_elite_selection(objective_id, parameter):
    run = _run()
    inputs = _inputs(
        objective_id=objective_id,
        parameter=parameter,
        detail_view=1,
        show_trend=True,
        mark_elite=True,
        elite_fraction="0.2",
    )
    outputs = CostEfficiency.process(run, inputs)
    rendered = CostEfficiency.load_outputs(run, inputs, outputs)
    objective = rendered["parameter_objective"]["figure"]
    duration = rendered["parameter_duration"]["figure"]
    assert list(objective.data[0].x) == list(duration.data[0].x)
    assert list(objective.data[0].y) == [row["objective"] for row in outputs["trials"]]
    assert objective.layout.yaxis.title.text == run.get_objective(objective_id).name
    assert objective.layout.yaxis.type == "linear"
    assert list(objective.data[-1].x) == list(duration.data[-1].x)
    assert len(objective.data[-1].y) == 4
    if parameter == "x":
        assert list(objective.data[2].x) == list(duration.data[2].x)
        assert " s" not in objective.data[2].hovertemplate
        assert objective.data[2].line.color == duration.data[2].line.color
    else:
        assert objective.data[0].type == "box"
    assert rendered["objective_panel"]["style"]["display"] == "block"
    assert rendered["importance_table"]["style"]["display"] == "none"
    table = CostEfficiency.load_outputs(run, {**inputs, "detail_view": 0}, outputs)
    assert table["objective_panel"]["style"]["display"] == "none"
    assert table["importance_table"]["style"]["display"] == "block"


def test_all_parameter_click_sources_update_native_control(monkeypatch):
    import deepcave
    from deepcave.plugins import Plugin
    from deepcave.plugins import cost_efficiency as module

    callbacks = {}
    registrations = []

    def callback(outputs, inputs, *_args, **_kwargs):
        registrations.append((outputs, inputs))

        def register(function):
            callbacks[function.__name__] = function
            return function

        return register

    monkeypatch.setattr(deepcave, "app", SimpleNamespace(callback=callback), raising=False)
    monkeypatch.setattr(deepcave, "c", SimpleNamespace(get=lambda *_: None), raising=False)
    run = _run()
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_run=lambda _: run), raising=False
    )
    plugin = CostEfficiency()
    monkeypatch.setattr(deepcave, "_api_mode", False)
    Plugin.register_callbacks(plugin)
    native_update = callbacks["plugin_input_update"]
    context = SimpleNamespace(triggered_id=None, triggered_prop_ids={})
    monkeypatch.setattr(module, "ctx", context)
    monkeypatch.setattr("deepcave.plugins.ctx", context)

    def update(events):
        values = [
            plugin.previous_inputs.get(name, {}).get(attribute)
            for name, attribute, *_ in plugin.inputs
        ]
        return native_update(
            "http://localhost/plugins/cost_efficiency",
            run.id,
            *values,
            {"value": "objective:0"},
            *events,
        )

    update([None] * len(plugin.input_update_inputs))
    for index, (_, component_id, property_name) in enumerate(plugin.input_update_inputs):
        context.triggered_id = component_id
        context.triggered_prop_ids = {f"{component_id}.{property_name}": component_id}
        parameter = "kind" if index % 2 == 0 else "x"
        events = [None] * len(plugin.input_update_inputs)
        if property_name == "active_cell":
            events[index] = {"row": 0, "row_id": parameter}
        elif property_name == "clickAnnotationData":
            events[index] = {"index": 0, "annotation": {"name": parameter, "text": parameter}}
            # Both events share a graph: only the triggered property may select a parameter.
            events[0] = {"points": [{"customdata": "kind"}]}
        else:
            events[index] = {"points": [{"customdata": parameter}]}
        result = update(events)
        assert plugin.previous_inputs["parameter"]["value"] == parameter
        assert result[len(plugin.inputs) + index] is None

    # Subsequent manual selection must not reapply an earlier chart/table click.
    context.triggered_id = plugin.get_internal_input_id("parameter")
    context.triggered_prop_ids = {f"{context.triggered_id}.value": context.triggered_id}
    plugin.previous_inputs["parameter"]["value"] = "x"
    update([None, None, {"row": 0, "row_id": "kind"}, None, None])
    assert plugin.previous_inputs["parameter"]["value"] == "x"

    outputs = [
        output
        for group, _ in registrations
        for output in (group if isinstance(group, list) else [group])
    ]
    writers = [
        output
        for output in outputs
        if output.component_id == plugin.get_internal_input_id("parameter")
        and output.component_property == "value"
    ]
    assert len(writers) == 1
    graph_ids = {
        plugin.get_internal_output_id(name)
        for name, attribute in plugin.outputs
        if attribute == "figure" and name not in ("parameter_duration", "parameter_objective")
    }
    assert {
        output.component_id for output in outputs if output.component_property == "clickData"
    } == graph_ids
    assert [
        output.component_id
        for output in outputs
        if output.component_property == "clickAnnotationData"
    ] == [
        plugin.get_internal_output_id("cost_importance"),
        plugin.get_internal_output_id("comparison"),
    ]


def test_native_render_delivers_all_registered_outputs(monkeypatch):
    import deepcave

    plugin = CostEfficiency()
    run = _run()
    outputs = plugin.process(run, _inputs())
    inputs = {name: {"value": value} for name, value in _inputs().items()}
    monkeypatch.setattr(deepcave, "_api_mode", False)
    rendered = plugin._process_raw_outputs(inputs, {run.id: outputs}, [run])
    assert len(rendered) == len(plugin.outputs) == 12
    by_name = {name: value for (name, _), value in zip(plugin.outputs, rendered)}
    assert len(by_name["cost_importance"].data) == 1
    assert len(by_name["performance"].data) == 2
    assert "Elite trials: 1" in rendered[-1]


def test_view_delivery_acknowledgement_is_independent_for_each_browser_page(monkeypatch):
    from time import monotonic

    from dash.exceptions import PreventUpdate

    import deepcave
    from deepcave.plugins import static

    callbacks = []

    def callback(*_args, **_kwargs):
        def register(function):
            callbacks.append(function)
            return function

        return register

    monkeypatch.setattr(deepcave, "app", SimpleNamespace(callback=callback), raising=False)
    monkeypatch.setattr(
        static, "ctx", SimpleNamespace(triggered_id="cost_efficiency-update-interval")
    )
    plugin = CostEfficiency()
    plugin._state = static.PluginState.READY
    plugin._expected_render_key = "view-b"
    plugin._expected_render_keys = {"page-a": "view-a", "page-b": "view-b"}
    plugin._render_attempts = {("page-a", "view-a"): monotonic()}
    plugin._callback_loop_update_status_label.__wrapped__(plugin)
    update_status = callbacks[-1]
    assert update_status(1, "view-a", "page-a")[1] is False
    assert update_status(1, "old", "page-a")[1] is True
    plugin._callback_loop_trigger_main_loop.__wrapped__(plugin)
    poll = callbacks[-1]
    assert poll(1, 0, "view-a", "page-a") == 1
    with pytest.raises(PreventUpdate):
        poll(1, 0, "old", "page-a")


@pytest.mark.parametrize(
    "filter_name,value",
    [
        ("parameter", "kind"),
        ("elite_fraction", "0.2"),
        ("show_pareto", False),
        ("show_trend", True),
        ("mark_elite", True),
        ("detail_view", 1),
        ("importance_view", 1),
    ],
)
def test_native_filter_callback_redraws_cached_results_without_reprocessing(
    monkeypatch, filter_name, value
):
    import deepcave
    from deepcave.plugins import static
    from deepcave.plugins.static import _processing_inputs_key, _render_inputs_key

    plugin, run = CostEfficiency(), _run()
    raw = plugin.process(run, _inputs())
    applied = plugin.load_inputs()
    for name, selected in _inputs().items():
        applied[name]["value"] = selected
    applied["run"] = {"value": run.id}
    current = {name: dict(attributes) for name, attributes in applied.items()}
    current[filter_name]["value"] = value
    assert _processing_inputs_key(plugin, current) == _processing_inputs_key(plugin, applied)
    assert _render_inputs_key(plugin, current) != _render_inputs_key(plugin, applied)
    callbacks = []

    def callback(*_args, **_kwargs):
        def register(function):
            callbacks.append(function)
            return function

        return register

    monkeypatch.setattr(deepcave, "app", SimpleNamespace(callback=callback), raising=False)
    monkeypatch.setattr(deepcave, "c", SimpleNamespace(get=lambda *_: applied), raising=False)
    monkeypatch.setattr(deepcave, "rc", SimpleNamespace(get=lambda *_: raw), raising=False)
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_run=lambda _: run), raising=False
    )
    monkeypatch.setattr(
        deepcave,
        "queue",
        SimpleNamespace(
            get_finished_jobs=lambda: [],
            is_running=lambda _: False,
            is_pending=lambda _: False,
            has_failed=lambda _: False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        static, "ctx", SimpleNamespace(triggered_id=plugin.get_internal_input_id(filter_name))
    )
    monkeypatch.setattr(deepcave, "_api_mode", False)
    plugin._refresh_required = False
    plugin._state = static.PluginState.READY
    plugin._callback_inputs_changed()
    values = [current[name].get(attribute) for name, attribute, *_ in plugin.inputs]
    result = callbacks[0](
        None, 0, run.id, *values, None, _render_inputs_key(plugin, applied), "page"
    )
    assert len(result) == len(plugin.outputs) + 1
    assert result[-1] == _render_inputs_key(plugin, current)
    by_name = {name: value for (name, _), value in zip(plugin.outputs, result)}
    if filter_name == "parameter":
        assert by_name["parameter_duration"].data[0].type == "box"
    elif filter_name == "elite_fraction":
        assert len(by_name["elite"].data[0].x) == 4
    elif filter_name == "show_pareto":
        assert len(by_name["performance"].data) == 1
        assert [trace.name for trace in by_name["elite"].data] == ["Elite trials"]
    elif filter_name == "show_trend":
        assert len(by_name["parameter_duration"].data) == 3
    elif filter_name == "mark_elite":
        assert by_name["parameter_duration"].data[-1].name == "Elite trials"
        assert len(by_name["parameter_duration"].data[-1].x) == 1
    elif filter_name == "importance_view":
        assert by_name["comparison"].data[0].type == "bar"
        assert by_name["comparison"].layout.xaxis.title.text == "Objective Importance"
        assert all(a.captureevents for a in by_name["comparison"].layout.annotations)
    elif filter_name == "detail_view":
        assert by_name["objective_panel"]["display"] == "block"
        assert by_name["importance_table"]["display"] == "none"
