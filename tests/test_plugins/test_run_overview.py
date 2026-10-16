from pathlib import Path

import pytest
from ConfigSpace import ConfigurationSpace
from dash import dcc

from deepcave.config import Config
from deepcave.plugins import cost_efficiency
from deepcave.plugins.objective import cost_over_time, performance_cost
from deepcave.plugins.summary import run_overview
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status


@pytest.fixture
def run(monkeypatch):
    for module in (run_overview, cost_over_time, performance_cost):
        monkeypatch.setattr(module, "save_image", lambda *_: None)
    monkeypatch.setattr(performance_cost.Configurations, "get_link", lambda *_: "/config")
    study = DeepCAVERun(
        "overview",
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}),
        objectives=[Objective("score", optimize="upper"), Objective("loss")],
    )
    study.path = Path("/tmp/overview")
    for index, (score, budget, status) in enumerate(
        [
            (1, 1, Status.SUCCESS),
            (4, 2, Status.SUCCESS),
            (3, 2, Status.SUCCESS),
            (2, 2, Status.FAILED),
        ]
    ):
        study.add(
            config={"x": index / 4},
            costs=[score, 5 - score],
            start_time=0,
            end_time=index + 10,
            budget=budget,
            seed=0,
            status=status,
        )
    return study


def test_summary_registration_and_single_run_controls(run):
    plugins = Config().PLUGINS
    assert [plugin.name for plugin in plugins["Summary"]] == ["Run Metadata", "Run Overview"]
    plugin = plugins["Summary"][1]
    assert plugin.activate_run_selection and not plugin.allow_additional_runs
    plugin.check_runs_compatibility([run])
    inputs = plugin.load_inputs()
    assert inputs["budget_id"]["value"] == 1
    registered = {}

    def register(name, *_args, **kwargs):
        registered[name] = kwargs
        return name

    layout = plugin.get_input_layout(register)
    assert list(registered) == [
        "budget_id", "highlight_mode", "top_percent", "deactivate_tooltips"
    ]
    selector = layout[0].children[2]
    assert selector.className == "multi-metrics-segmented"
    assert [option["label"] for option in selector.options] == ["None", "Pareto front", "Elite trials"]
    assert inputs["highlight_mode"]["value"] == "pareto"
    assert inputs["top_percent"]["value"] == 5
    assert registered["highlight_mode"]["filter"] is True
    assert registered["top_percent"]["filter"] is True
    assert inputs["deactivate_tooltips"]["value"] is False
    assert registered["deactivate_tooltips"]["filter"] is True
    assert plugin.get_filter_layout(
        lambda name, *args: register(name, *args, filter=True)
    ) == []
    assert list(registered) == [
        "budget_id", "highlight_mode", "top_percent", "deactivate_tooltips", "importance_view"
    ]
    assert registered["importance_view"]["filter"] is True
    assert inputs["importance_view"]["value"] == "objective"
    assert [option["value"] for option in inputs["importance_view"]["options"]] == [
        "objective", "cost", "comparison"
    ]
    assert "elite_fraction" not in inputs
    assert "xaxis" not in inputs


def test_output_layout_keeps_full_figures_in_separate_graphs():
    registered = []
    layout = run_overview.RunOverview.get_output_layout(
        lambda name, *_: registered.append(name) or name
    )
    assert registered == ["graph", "objective_importance"]
    performance, importance = layout.children
    assert performance.children[0].children == "Performance vs. Computational Cost"
    assert isinstance(performance.children[1], dcc.Graph)
    assert importance.children[0].children == "fANOVA Feature Importance"
    importance_graph, view_control = importance.children[1].children
    assert importance_graph.id == "objective_importance"
    assert importance_graph.style["height"] == "500px"
    label, selector = view_control.children
    assert label.children == "Importance"
    assert selector.id == "run_overview-importance_view-input"
    assert selector.className == "multi-metrics-segmented"
    assert selector.value == "objective"
    assert [option["value"] for option in selector.options] == ["objective", "cost", "comparison"]


@pytest.mark.parametrize("objective_id", [0, 1])
@pytest.mark.parametrize("duration", [11, 0, 0.01])
def test_overview_reuses_cost_samples_and_plots_trial_durations(
    run, monkeypatch, objective_id, duration
):
    run.history[1].start_time = run.history[1].end_time - duration
    samples = []

    def importance(_run, rows, **kwargs):
        assert kwargs["objective_id"] == objective_id
        samples.append(rows)
        return {"objective": {"x": 0.75}, "cost": {"x": 0.25}, "errors": {}}

    assert run_overview.calculate_importances is cost_efficiency.calculate_importances
    monkeypatch.setattr(run_overview, "calculate_importances", importance)
    monkeypatch.setattr(cost_efficiency, "calculate_importances", importance)
    inputs = {"objective_id": objective_id, "budget_id": 1}
    outputs = run_overview.RunOverview.process(run, inputs)
    cost_inputs = {"objective_id": objective_id, "trials": "budget:1", "importance_view": 1}
    cost_outputs = cost_efficiency.CostEfficiency.process(run, cost_inputs)
    assert samples[0] == samples[1]
    assert [row["history_id"] for row in samples[0]] == [1, 2]
    rendered = run_overview.RunOverview.load_outputs(run, inputs, outputs)
    figure = rendered["graph"]["figure"]
    assert figure.layout.height == 500
    scatter = next(
        trace for trace in figure.data if trace.type == "scatter" and trace.name == "Trials"
    )
    assert list(scatter.x) == pytest.approx([duration, 12])
    assert list(scatter.y) == ([4, 3] if objective_id == 0 else [1, 2])
    assert [list(row) for row in scatter.customdata] == [[0, 1], [0, 2]]
    assert "Trial 1" in scatter.hovertext[0]
    assert "Trial duration:" in scatter.hovertext[0]
    assert "Open configuration" in scatter.hovertext[0]
    histogram = next(trace for trace in figure.data if trace.type == "histogram")
    assert histogram.x is None
    assert list(histogram.y) == list(scatter.y)
    assert histogram.orientation == "h"
    assert "Trials: %{x}" in histogram.hovertemplate
    assert figure.layout.xaxis.domain == pytest.approx((0, 0.8))
    assert figure.layout.xaxis2.domain == pytest.approx((0.82, 1))
    assert figure.layout.yaxis.domain == figure.layout.yaxis2.domain
    assert figure.layout.yaxis2.matches == "y"
    assert figure.layout.yaxis2.title.text is None
    assert figure.layout.yaxis2.showticklabels is False
    assert figure.layout.xaxis2.showticklabels is False
    assert figure.layout.xaxis2.title.text == "Trials"
    assert figure.layout.xaxis2.type != "log"
    assert figure.layout.xaxis.title.text == "Trial duration [s]"
    assert figure.layout.yaxis.title.text == ("score" if objective_id == 0 else "loss")
    best_line, histogram_line = figure.layout.shapes
    assert best_line.xref == "x"
    assert best_line.x0 == pytest.approx(duration)
    assert best_line.x1 > max(scatter.x)
    assert histogram_line.xref == "paper"
    assert (histogram_line.x0, histogram_line.x1) == (figure.layout.xaxis.domain[1], 1)
    assert histogram_line.y0 == histogram_line.y1 == best_line.y0
    assert histogram_line.line == best_line.line
    assert best_line.yref == "y"
    assert best_line.y0 == best_line.y1 == (4 if objective_id == 0 else 1)
    assert best_line.line.dash == "dash"
    (best_label,) = figure.layout.annotations
    assert best_label.text == ("4" if objective_id == 0 else "1")
    assert best_label.xref == "paper"
    assert best_label.x == figure.layout.xaxis2.domain[0]
    assert best_label.yref == "y"
    assert best_label.y == best_line.y0
    assert best_label.yanchor == "bottom"
    assert best_label.showarrow is False
    assert figure.layout.xaxis.type == ("log" if duration == 0.01 else "linear")
    importance = rendered["objective_importance"]["figure"]
    reference = cost_efficiency.CostEfficiency.load_outputs(run, cost_inputs, cost_outputs)
    # Compare the complete native figure, including labels, spacing, axes and typography.
    assert importance.to_json() == reference["comparison"]["figure"].to_json()
    (bars,) = importance.data
    assert list(bars.x) == [0.75]
    assert list(bars.y) == ["x"]
    assert importance.layout.xaxis.title.text == "Objective Importance"
    assert importance.layout.annotations[0].text == "x"
    assert importance.layout.annotations[0].captureevents is True
    native = performance_cost.PerformanceCost.load_outputs(run, inputs, outputs["performance_cost"])
    # Preserve native trial and Pareto traces, including configuration links.
    assert [trace.to_plotly_json() for trace in figure.data[:2]] == [
        {**trace.to_plotly_json(), "xaxis": "x", "yaxis": "y"} for trace in native.data
    ]
    hidden = run_overview.RunOverview.load_outputs(run, {**inputs, "highlight_mode": "none"}, outputs)
    assert [trace.type for trace in hidden["graph"]["figure"].data] == ["scatter", "histogram"]
    assert [shape.to_plotly_json() for shape in hidden["graph"]["figure"].layout.shapes] == [
        shape.to_plotly_json() for shape in figure.layout.shapes
    ]


@pytest.mark.parametrize("objective_id", [0, 1])
@pytest.mark.parametrize("percent, count", [(5, 1), (50, 1), (100, 2)])
def test_top_percent_highlights_objective_elite_without_filtering(
    run, monkeypatch, objective_id, percent, count
):
    monkeypatch.setattr(
        run_overview, "calculate_importances",
        lambda *_, **_kwargs: {"objective": {"x": 0.75}, "cost": {"x": 0.25}, "errors": {}},
    )
    inputs = {"objective_id": objective_id, "budget_id": 1}
    outputs = run_overview.RunOverview.process(run, inputs)
    baseline = run_overview.RunOverview.load_outputs(run, inputs, outputs)
    rendered = run_overview.RunOverview.load_outputs(
        run, {**inputs, "highlight_mode": "top", "top_percent": percent}, outputs
    )
    trials, elite, histogram = rendered["graph"]["figure"].data
    assert list(trials.x) == [11, 12]
    assert list(histogram.y) == list(trials.y)
    assert elite.name == f"Top {percent}%"
    assert elite.mode == "markers"
    assert len(elite.x) == count
    assert list(elite.customdata[0]) == [0, 1]
    assert elite.y[0] == (4 if objective_id == 0 else 1)
    assert "Open configuration" in elite.hovertext[0]
    figure = rendered["graph"]["figure"]
    if count == 1:
        # The threshold coincides with the best value; draw and label it only once.
        assert len(figure.layout.shapes) == len(figure.layout.annotations) == 1
    else:
        best_line, threshold_line = figure.layout.shapes
        assert threshold_line.y0 == threshold_line.y1 == (3 if objective_id == 0 else 2)
        assert (threshold_line.x0, threshold_line.x1, threshold_line.xref) == (0, 1, "paper")
        assert threshold_line.yref == "y"
        assert threshold_line.line.dash == "dash"
        assert threshold_line.line == best_line.line
        threshold_label = figure.layout.annotations[1]
        assert threshold_label.text == ("3" if objective_id == 0 else "2")
        assert threshold_label.x == figure.layout.xaxis2.domain[0]
        assert threshold_label.y == threshold_line.y0
        lower_label, upper_label = sorted(figure.layout.annotations[:2], key=lambda a: a.y)
        assert lower_label.yanchor == "top"
        assert lower_label.yshift == -6
        assert upper_label.yanchor == "bottom"
        assert upper_label.yshift == 6
        delta_label = figure.layout.annotations[2]
        assert delta_label.text == "Δ 1"
        assert delta_label.x == 1
        assert delta_label.xref == "paper"
        assert delta_label.xanchor == "right"
        assert delta_label.y == (3.5 if objective_id == 0 else 1.5)
        assert delta_label.yref == "y"
        assert delta_label.yanchor == "middle"
    assert rendered["objective_importance"]["figure"].to_json() == (
        baseline["objective_importance"]["figure"].to_json()
    )


@pytest.mark.parametrize("importance_view", ["objective", "cost", "comparison"])
def test_tooltip_switch_disables_both_plots_and_restores_hover(run, monkeypatch, importance_view):
    monkeypatch.setattr(
        run_overview, "calculate_importances",
        lambda *_, **_kwargs: {"objective": {"x": 0.75}, "cost": {"x": 0.25}, "errors": {}},
    )
    inputs = {
        "objective_id": 0, "budget_id": 1, "highlight_mode": "top",
        "importance_view": importance_view,
    }
    outputs = run_overview.RunOverview.process(run, inputs)
    baseline = run_overview.RunOverview.load_outputs(run, inputs, outputs)
    disabled = run_overview.RunOverview.load_outputs(
        run, {**inputs, "deactivate_tooltips": True}, outputs
    )
    restored = run_overview.RunOverview.load_outputs(
        run, {**inputs, "deactivate_tooltips": False}, outputs
    )
    for name in ("graph", "objective_importance"):
        assert baseline[name]["figure"].layout.hovermode is not False
        assert disabled[name]["figure"].layout.hovermode is False
        assert restored[name]["figure"].to_json() == baseline[name]["figure"].to_json()


def test_importance_views_reuse_computational_cost_figures(run, monkeypatch):
    calls = []

    def importance(_run, rows, **_kwargs):
        calls.append(rows)
        return {"objective": {"x": 0.75}, "cost": {"x": 0.25}, "errors": {}}

    monkeypatch.setattr(run_overview, "calculate_importances", importance)
    inputs = {"objective_id": 0, "budget_id": 1, "highlight_mode": "top"}
    outputs = run_overview.RunOverview.process(run, inputs)
    cost, comparison = cost_efficiency.importance_figures(
        run,
        {
            "importances": outputs["importances"],
            "comparison": cost_efficiency.importance_comparison({"x": 0.25}, {"x": 0.75}),
        },
    )
    objective = cost_efficiency.importance_bar_figure(run, {"x": 0.75}, "Objective Importance")
    for view, expected in (("objective", objective), ("cost", cost), ("comparison", comparison)):
        rendered = run_overview.RunOverview.load_outputs(
            run, {**inputs, "importance_view": view}, outputs
        )
        assert rendered["objective_importance"]["figure"].to_json() == expected.to_json()
    assert len(calls) == 1
    assert list(comparison.data[0].x) == [0.25]
    assert list(comparison.data[0].y) == [0.75]


@pytest.mark.parametrize("view", ["objective", "cost", "comparison"])
def test_unavailable_importance_is_explained(run, monkeypatch, view):
    monkeypatch.setattr(
        run_overview,
        "calculate_importances",
        lambda *_, **_kwargs: {
            "objective": {},
            "cost": {},
            "errors": {
                "objective": "Insufficient objective trials.",
                "cost": "Insufficient duration trials.",
            },
        },
    )
    inputs = {"objective_id": 0, "budget_id": 1, "importance_view": view}
    outputs = run_overview.RunOverview.process(run, inputs)
    outputs["performance_cost"]["trials"] = []
    rendered = run_overview.RunOverview.load_outputs(run, inputs, outputs)
    figure = rendered["objective_importance"]["figure"]
    expected = "Insufficient duration trials." if view == "cost" else "Insufficient objective trials."
    message = next(a for a in figure.layout.annotations if a.text == expected)
    assert message.xref == "paper"
    assert message.yref == "paper"
    assert rendered["graph"]["figure"].layout.annotations[0].text == (
        "No valid completed trials with duration data available."
    )


@pytest.mark.parametrize("highlight_mode", ["none", "pareto", "top"])
def test_empty_trial_distribution_has_no_best_line(run, monkeypatch, highlight_mode):
    monkeypatch.setattr(
        run_overview, "calculate_importances",
        lambda *_, **_kwargs: {"objective": {}, "cost": {}, "errors": {}},
    )
    inputs = {"objective_id": 0, "budget_id": 1, "highlight_mode": highlight_mode}
    outputs = run_overview.RunOverview.process(run, inputs)
    outputs["performance_cost"]["trials"] = []
    figure = run_overview.RunOverview.load_outputs(run, inputs, outputs)["graph"]["figure"]
    histogram = next(trace for trace in figure.data if trace.type == "histogram")
    assert list(histogram.y) == []
    assert not figure.layout.shapes
    assert len(figure.layout.annotations) == 1
    assert figure.layout.annotations[0].text == (
        "No valid completed trials with duration data available."
    )
