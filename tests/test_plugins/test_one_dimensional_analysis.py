from types import SimpleNamespace

import pandas as pd
import plotly.graph_objs as go
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dash import html

from deepcave import config
from deepcave.plugins.hyperparameter import one_dimensional_analysis as analysis
from deepcave.runs.objective import Objective


def test_plugin_is_first_hyperparameter_analysis_without_ablation_paths():
    plugin_ids = [plugin.id for plugin in config.PLUGINS["Hyperparameter Analysis"]]

    assert plugin_ids[0] == "one_dimensional_analysis"
    assert "ablation_paths" not in plugin_ids


def test_toolbar_does_not_require_an_apply_button():
    (toolbar,) = analysis.OneDimensionalAnalysis.get_input_layout(
        lambda input_id, *_args, **_kwargs: input_id
    )

    assert not any(isinstance(component, html.Button) for component in _walk_components(toolbar))


def test_every_control_shares_the_single_toolbar_row():
    registered = []
    (toolbar,) = analysis.OneDimensionalAnalysis.get_input_layout(
        lambda input_id, *_args, **_kwargs: registered.append(input_id) or input_id
    )

    assert registered == [
        "hyperparameter_name",
        "elite_proportion",
    ]
    assert "analysis-toolbar" in toolbar.className


def _walk_components(component):
    """Yield a Dash component and all of its descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children"):
            yield from _walk_components(child)


@pytest.mark.parametrize(
    ("hyperparameter", "values", "expected_groups"),
    [
        (
            UniformIntegerHyperparameter("integer", lower=1, upper=3),
            [1, 2, 3],
            ["1", "2", "3"],
        ),
        (
            CategoricalHyperparameter("category", choices=["a", "b"]),
            ["b", "a"],
            ["b", "a"],
        ),
    ],
)
def test_discrete_hyperparameters_keep_observed_values(
    hyperparameter,
    values,
    expected_groups,
):
    groups = analysis._group_parameter_values(pd.Series(values), hyperparameter)

    assert groups.astype(str).tolist() == expected_groups


def test_numeric_hyperparameters_use_linear_or_logarithmic_bins():
    linear = UniformFloatHyperparameter("linear", lower=1.0, upper=101.0)
    logarithmic = UniformFloatHyperparameter("logarithmic", lower=1.0, upper=100.0, log=True)

    linear_groups = analysis._group_parameter_values(
        pd.Series([1.0, 11.0, 101.0]),
        linear,
    )
    logarithmic_groups = analysis._group_parameter_values(
        pd.Series([1.0, 10.0, 100.0]),
        logarithmic,
    )

    assert len(linear_groups.cat.categories) == 10
    assert len(logarithmic_groups.cat.categories) == 10
    assert 11.0 in linear_groups.cat.categories[0]
    assert 10.0 in logarithmic_groups.cat.categories[4]


def test_small_range_ticks_move_the_shared_exponent_into_the_axis_title():
    figure = go.Figure()
    ranges = ["(0.004, 0.007]", "(0.007, 0.009]", "(0.009, 0.011]"]

    axis_title = analysis._range_axis(figure, ranges, "depth")

    assert axis_title == "depth (×10⁻³)"
    assert figure.layout.xaxis.title.text == axis_title
    assert list(figure.layout.xaxis.ticktext) == [
        "4\u2060–\u20607",
        "7\u2060–\u20609",
        "9\u2060–\u206011",
    ]


def test_summary_uses_median_elite_rank_instead_of_mean():
    hyperparameter = CategoricalHyperparameter("group", choices=["a", "b"])
    trials = pd.DataFrame(
        {
            "trial_number": range(8),
            "hyperparameter_value": ["a", "a", "b", "a", "b", "b", "b", "b"],
            "objective": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        }
    )

    summary, tail_quantile, threshold, elite_count = analysis._summarize_trials(
        trials,
        hyperparameter,
        optimize="upper",
        elite_proportion=0.5,
    )

    group_a = next(row for row in summary if row["range"] == "a")
    assert group_a["elite_median_rank"] == pytest.approx(2.0)
    assert group_a["elite_median_rank"] != pytest.approx((1 + 2 + 4) / 3)
    assert group_a["elite_rate_pct"] == pytest.approx(100.0)
    # 100% elite in the bin against a 50% global elite share.
    assert group_a["elite_enrichment"] == pytest.approx(2.0)
    assert tail_quantile == pytest.approx(0.9)
    assert threshold == pytest.approx(6.5)
    assert elite_count == 4


def test_minimization_uses_lower_tail_and_lower_values_as_elite():
    hyperparameter = CategoricalHyperparameter("group", choices=["a", "b"])
    trials = pd.DataFrame(
        {
            "trial_number": range(5),
            "hyperparameter_value": ["a", "a", "b", "b", "b"],
            "objective": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    summary, tail_quantile, threshold, elite_count = analysis._summarize_trials(
        trials,
        hyperparameter,
        optimize="lower",
        elite_proportion=0.2,
    )

    assert tail_quantile == pytest.approx(0.1)
    assert threshold == pytest.approx(1.8)
    assert elite_count == 1
    assert summary[0]["elite_median_rank"] == pytest.approx(1.0)
    assert summary[1]["elite_median_rank"] is None


def test_dependencies_exclude_time_and_preserve_analysis_controls():
    objectives = [
        Objective("Loss", optimize="lower"),
        Objective("Accuracy", optimize="upper"),
        Objective("Time"),
    ]
    configspace = ConfigurationSpace(
        space={
            "x": (0.0, 1.0),
            "kind": ["a", "b"],
        }
    )

    class DependencyRun:
        def get_objectives(self):
            return objectives

        def get_objective_id(self, objective):
            return objectives.index(objective)

    run = DependencyRun()
    run.configspace = configspace
    inputs = {
        "objective_id": {"value": 1},
        "hyperparameter_name": {"value": "x"},
        "elite_proportion": {"value": 0.2},
    }
    previous_inputs = {
        **inputs,
        "elite_proportion": {"value": 0.1},
    }

    result = analysis.OneDimensionalAnalysis.load_dependency_inputs(
        object.__new__(analysis.OneDimensionalAnalysis),
        run,
        previous_inputs,
        inputs,
    )

    assert [option["label"] for option in result["objective_id"]["options"]] == [
        "Loss",
        "Accuracy",
    ]
    assert result["objective_id"]["value"] == 1
    assert result["elite_proportion"]["value"] == pytest.approx(0.2)


def test_all_plots_are_immediate(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Accuracy", optimize="upper")
    hyperparameter = CategoricalHyperparameter("kind", choices=["a", "b"])
    run = SimpleNamespace(
        configspace={"x": UniformFloatHyperparameter("x", 0.0, 1.0), "kind": hyperparameter},
        get_objective=lambda _objective_id: objective,
    )
    outputs = {
        "importances": {"objective": {"x": 0.2, "kind": 0.7}, "errors": {}},
        "trials": [
            {"trial_number": 2, "hyperparameter_value": "a", "objective": 0.7},
            {"trial_number": 5, "hyperparameter_value": "b", "objective": 0.9},
        ],
        "summary": [
            {
                "range": "a",
                "n": 1,
                "elite": 0,
                "elite_rate_pct": 0.0,
                "elite_enrichment": 0.0,
                "elite_median_rank": None,
                "objective_median": 0.7,
                "objective_tail": 0.7,
                "objective_max": 0.7,
            },
            {
                "range": "b",
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_enrichment": 2.0,
                "elite_median_rank": 1.0,
                "objective_median": 0.9,
                "objective_tail": 0.9,
                "objective_max": 0.9,
            },
        ],
        "tail_quantile": 0.9,
        "elite_threshold": 0.88,
        "elite_count": 1,
    }
    base_inputs = {
        "objective_id": 0,
        "hyperparameter_name": "kind",
        "elite_proportion": 0.1,
    }

    importance_figure, trials_figure, enrichment_figure = (
        analysis.OneDimensionalAnalysis.load_outputs(
            run,
            base_inputs,
            outputs,
        )
    )
    assert importance_figure.layout.xaxis.title.text == "Objective Importance"
    assert list(importance_figure.data[0].customdata) == ["kind", "x"]
    assert importance_figure.data[0].orientation == "h"
    assert importance_figure.layout.annotations[0].captureevents is True
    assert [trace.type for trace in trials_figure.data] == ["scatter"] * 7
    assert trials_figure.data[0].marker.color == importance_figure.data[0].marker.color[0]
    assert trials_figure.data[0].marker.colorscale is None
    assert trials_figure.data[0].marker.size == 7
    assert trials_figure.data[0].marker.opacity == 0.6
    assert trials_figure.layout.font == importance_figure.layout.font
    assert trials_figure.layout.margin.t == importance_figure.layout.margin.t
    assert trials_figure.layout.margin.b == importance_figure.layout.margin.b
    assert trials_figure.layout.margin.r == importance_figure.layout.margin.r
    assert trials_figure.layout.title.text is None
    assert trials_figure.layout.height is None
    assert importance_figure.layout.height is None
    assert not any(annotation.text == "Trial" for annotation in trials_figure.layout.annotations)
    assert [trace.name for trace in trials_figure.data[:4]] == ["Trials", "q90", "Median", "Best"]
    assert [trace.mode for trace in trials_figure.data] == ["markers"] + ["lines+markers"] * 3 + [
        "lines"
    ] * 3
    assert list(trials_figure.data[1].x) == ["a", "b"]
    assert list(trials_figure.data[2].y) == pytest.approx([0.7, 0.9])
    assert trials_figure.layout.showlegend is True
    assert [trace.name for trace in trials_figure.data if trace.showlegend] == [
        "q90",
        "Median",
        "Best",
    ]
    assert trials_figure.data[0].showlegend is False
    assert trials_figure.data[0].visible is None
    for main, mean in zip(trials_figure.data[1:4], trials_figure.data[4:]):
        assert main.legendgroup == mean.legendgroup
        assert main.visible == mean.visible == "legendonly"
        assert main.line.color == mean.line.color
        assert mean.showlegend is False
        assert mean.line.dash == "dash"
    assert trials_figure.layout.legend.title.text == "Show lines:"
    assert trials_figure.layout.legend.groupclick == "togglegroup"
    assert trials_figure.layout.legend.itemclick == "toggle"
    assert trials_figure.layout.legend.itemdoubleclick is False
    assert trials_figure.layout.legend.orientation == "h"
    assert trials_figure.layout.legend.yref == "container"
    ranked, without_rank, baseline = enrichment_figure.data
    assert list(ranked.y) == pytest.approx([2.0])
    assert list(ranked.marker.color) == pytest.approx([1.0])
    assert ranked.marker.reversescale is True
    assert "Median elite rank" in ranked.hovertemplate
    assert without_rank.marker.color == analysis.DEPLETED_COLOR
    assert without_rank.customdata[0][-1] == "n/a"
    assert list(baseline.y) == pytest.approx([1.0, 1.0])
    assert baseline.name == "1.0× baseline"
    assert enrichment_figure.layout.xaxis.title.text is None
    x_title = next(
        annotation
        for annotation in enrichment_figure.layout.annotations
        if annotation.name == "enrichment_x_title"
    )
    assert x_title.text == "kind"
    assert x_title.y == pytest.approx(analysis.ENRICHMENT_X_TITLE_Y)
    assert x_title.y < analysis.TOP_TRIAL_RUG_Y
    markers = [
        annotation
        for annotation in enrichment_figure.layout.annotations
        if annotation.name and annotation.name.startswith("top_trial_marker_")
    ]
    ranks = [
        annotation
        for annotation in enrichment_figure.layout.annotations
        if annotation.name and annotation.name.startswith("top_trial_rank_")
    ]
    assert [marker.x for marker in markers] == pytest.approx([0.75, 0.25])
    assert [marker.text for marker in markers] == ["◆", "◆"]
    assert [rank.text for rank in ranks] == ["1", "2"]
    assert any(
        annotation.text == "Top 5 trials" for annotation in enrichment_figure.layout.annotations
    )


def test_top_trials_rug_uses_exact_continuous_positions_and_objective_order(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Loss", optimize="lower")
    hyperparameter = UniformFloatHyperparameter("x", lower=0.0, upper=1.0)
    run = SimpleNamespace(
        configspace={"x": hyperparameter},
        get_objective=lambda _objective_id: objective,
    )
    trials = pd.DataFrame(
        {
            "trial_number": [8, 2, 5, 1, 7, 3],
            "hyperparameter_value": [0.91, 0.123, 0.126, 0.44, 0.7, 0.2],
            "objective": [6.0, 1.0, 2.0, 4.0, 5.0, 3.0],
        }
    )
    summary, tail_quantile, elite_threshold, elite_count = analysis._summarize_trials(
        trials,
        hyperparameter,
        objective.optimize,
        0.2,
    )

    _, _, enrichment_figure = analysis.OneDimensionalAnalysis.load_outputs(
        run,
        {
            "objective_id": 0,
            "hyperparameter_name": "x",
            "elite_proportion": 0.2,
        },
        {
            "importances": {"objective": {"x": 0.7}, "errors": {}},
            "trials": trials.to_dict(orient="records"),
            "summary": summary,
            "tail_quantile": tail_quantile,
            "elite_threshold": elite_threshold,
            "elite_count": elite_count,
        },
    )

    markers = [
        annotation
        for annotation in enrichment_figure.layout.annotations
        if annotation.name and annotation.name.startswith("top_trial_marker_")
    ]
    ranks = [
        annotation
        for annotation in enrichment_figure.layout.annotations
        if annotation.name and annotation.name.startswith("top_trial_rank_")
    ]
    assert [marker.xref for marker in markers] == ["paper"] * 5
    assert [marker.x for marker in markers] == pytest.approx([0.1, 0.125, 0.15, 0.375, 0.625])
    assert [rank.text for rank in ranks] == ["1", "2", "3", "4", "5"]
    assert "Trial: 2" in markers[0].hovertext
    assert [marker.y for marker in markers] == [analysis.TOP_TRIAL_RUG_Y] * 5
    assert "xaxis2" not in enrichment_figure.layout
    assert len(enrichment_figure.layout.shapes) == 1
    rug_line = enrichment_figure.layout.shapes[0]
    assert rug_line.y0 == pytest.approx(analysis.TOP_TRIAL_RUG_Y)
    assert rug_line.y1 == pytest.approx(analysis.TOP_TRIAL_RUG_Y)


def test_rug_markers_are_centered_and_spread_horizontally_inside_their_bin():
    positions = analysis._center_rug_positions_in_bins(
        [1, 1, 1, 3, 3],
        bin_count=5,
    )

    assert positions[:3] == pytest.approx([0.275, 0.3, 0.325])
    assert positions[3:] == pytest.approx([0.6875, 0.7125])


def test_combined_plot_reuses_scatter_without_accumulating_summary_traces(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    analysis.clear_one_dimensional_figure_cache()
    original_trials_figure = analysis._trials_figure
    calls = []

    def tracked_trials_figure(*args, **kwargs):
        calls.append(1)
        return original_trials_figure(*args, **kwargs)

    monkeypatch.setattr(analysis, "_trials_figure", tracked_trials_figure)
    objective = Objective("Accuracy", optimize="upper")
    hyperparameter = CategoricalHyperparameter("kind", choices=["a", "b"])
    run = SimpleNamespace(
        id="study",
        latest_change=1.0,
        history=[],
        configspace={"kind": hyperparameter},
        get_objective=lambda _objective_id: objective,
    )
    outputs = {
        "importances": {"objective": {"kind": 0.7}, "errors": {}},
        "trials": [{"trial_number": 2, "hyperparameter_value": "a", "objective": 0.7}],
        "summary": [
            {
                "range": "a",
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_enrichment": 1.0,
                "elite_median_rank": 1.0,
                "objective_median": 0.7,
                "objective_tail": 0.7,
                "objective_max": 0.7,
            }
        ],
        "tail_quantile": 0.9,
        "elite_threshold": 0.5,
        "elite_count": 1,
    }

    _, trials_figure, enrichment_figure = analysis.OneDimensionalAnalysis.load_outputs(
        run,
        {
            "objective_id": 0,
            "hyperparameter_name": "kind",
            "elite_proportion": 0.1,
        },
        outputs,
    )
    _, updated_figure, _ = analysis.OneDimensionalAnalysis.load_outputs(
        run,
        {
            "objective_id": 0,
            "hyperparameter_name": "kind",
            "elite_proportion": 0.2,
        },
        outputs,
    )
    assert len(calls) == 1
    assert len(trials_figure.data) == len(updated_figure.data) == 7
    assert not trials_figure.layout.shapes
    assert [trace.legendgroup for trace in updated_figure.data] == [
        None,
        "tail",
        "median",
        "best",
        "tail",
        "median",
        "best",
    ]

    assert list(trials_figure.layout.xaxis.categoryarray) == ["a", "b"]
    assert list(enrichment_figure.layout.xaxis.categoryarray) == ["a"]
    for figure in (trials_figure, enrichment_figure):
        assert figure.layout.xaxis.categoryorder == "array"
        assert figure.layout.margin.l == analysis.PANEL_LEFT_MARGIN
    assert trials_figure.layout.xaxis.automargin is True
    assert trials_figure.layout.yaxis.automargin is True
    assert enrichment_figure.layout.xaxis.automargin is False
    assert enrichment_figure.layout.yaxis.automargin is False

    assert trials_figure.layout.showlegend is True


@pytest.mark.parametrize(
    ("hyperparameter", "expected_centers"),
    [
        (UniformFloatHyperparameter("x", 0.0, 1.0), [0.05, 0.95]),
        (
            UniformFloatHyperparameter("x", 1.0, 100.0, log=True),
            [100**0.05, 100**0.95],
        ),
    ],
)
def test_summary_lines_use_linear_or_geometric_bin_centers(hyperparameter, expected_centers):
    trials = pd.DataFrame(
        {
            "trial_number": [0, 1, 2, 3],
            "hyperparameter_value": [hyperparameter.lower] * 2 + [hyperparameter.upper] * 2,
            "objective": [1.0, 3.0, 5.0, 7.0],
        }
    )
    summary, *_ = analysis._summarize_trials(trials, hyperparameter, "lower", 0.5)
    figure = analysis._trials_figure(trials, hyperparameter, "x", "Loss", "blue")
    analysis._add_performance_summary(figure, trials, pd.DataFrame(summary), hyperparameter, 10)

    raw, tail, median, best = figure.data
    assert list(raw.y) == [1.0, 3.0, 5.0, 7.0]
    assert tail.name == "q10"
    assert best.name == "Best"
    for trace, expected_y in ((tail, [1.2, 5.2]), (median, [2.0, 6.0]), (best, [3.0, 7.0])):
        assert trace.mode == "lines+markers"
        assert list(trace.x) == pytest.approx(expected_centers)
        assert trace.connectgaps is False
        assert list(trace.y) == pytest.approx(expected_y)
        assert list(trace.customdata[0][1:]) == [2, 2.0, 1.2, 3.0]
        assert "Best (max):" in trace.hovertemplate
        assert "Trials:" in trace.hovertemplate
    assert figure.layout.xaxis.type == ("log" if hyperparameter.log else None)


def test_discrete_numeric_summary_lines_keep_numeric_coordinates():
    hyperparameter = UniformIntegerHyperparameter("depth", 1, 3)
    trials = pd.DataFrame(
        {"trial_number": [0, 1], "hyperparameter_value": [1, 3], "objective": [4.0, 2.0]}
    )
    summary, *_ = analysis._summarize_trials(trials, hyperparameter, "lower", 0.5)
    figure = analysis._trials_figure(trials, hyperparameter, "depth", "Loss", "blue")
    analysis._add_performance_summary(figure, trials, pd.DataFrame(summary), hyperparameter, 10)
    for trace in figure.data[1:]:
        assert trace.mode == "lines+markers"
        assert list(trace.x) == [1.0, 3.0]
        assert list(trace.y) == [4.0, 2.0]


def test_output_layout_places_importance_left_of_combined_plot_above_enrichment():
    registered = []
    layout = analysis.OneDimensionalAnalysis.get_output_layout(
        lambda name, *_args, **_kwargs: registered.append(name) or name
    )

    (grid,) = layout
    assert grid.className == "analysis-grid analysis-grid--one-dimensional"
    assert registered == [
        "importance_graph",
        "trials_graph",
        "enrichment_graph",
    ]
    assert [card.className for card in grid.children] == [
        "analysis-card",
        "analysis-card",
        "analysis-card",
    ]
    assert grid.children[-1].style == {"gridColumn": "1 / -1"}
    (graph,) = grid.children[1].children
    assert graph.id == "trials_graph"
    assert "height" not in graph.style
    assert "minHeight" not in graph.style
    assert grid.children[1].style == {"alignSelf": "start"}
    assert grid.children[0].children.className == graph.className == "one-dimensional-primary-graph"
    assert "height" not in grid.children[0].children.style
    assert not analysis.OneDimensionalAnalysis.output_view_inputs


def test_bin_means_weight_finite_bins_equally_and_share_native_legend_groups():
    summary = pd.DataFrame(
        {
            "n": [1, 100, 5, 2],
            "objective_tail": [2.0, 8.0, float("nan"), float("inf")],
            "objective_median": [1.0, 5.0, float("nan"), float("-inf")],
            "objective_max": [3.0, 11.0, float("nan"), float("nan")],
        }
    )
    figure = go.Figure([go.Scatter(name=name) for name in ("Trials", "q90", "Median", "Best")])
    analysis._add_bin_means(figure, summary)
    means = figure.data[4:]
    assert [trace.y[0] for trace in means] == pytest.approx([5.0, 3.0, 7.0])
    assert [trace.y[1] for trace in means] == pytest.approx([5.0, 3.0, 7.0])
    assert [trace.line.color for trace in means] == [
        analysis.TAIL_COLOR,
        analysis.MEDIAN_COLOR,
        analysis.BEST_COLOR,
    ]
    assert [trace.legendgroup for trace in means] == ["tail", "median", "best"]
    for trace in means:
        assert trace.line.dash == "dash"
        assert trace.line.width < 3
        assert trace.opacity < 1
        assert trace.showlegend is False
        assert list(trace.x) == [0, 1]
        assert trace.xaxis == "x2"
    assert figure.layout.xaxis2.overlaying == "x"
    assert tuple(figure.layout.xaxis2.range) == (0, 1)
    assert figure.layout.xaxis2.visible is False
    assert figure.layout.xaxis2.fixedrange is True
    assert figure.data[0].visible is None


@pytest.mark.parametrize("event_index", [0, 1])
def test_importance_click_selects_parameter_and_ignores_stale_events(monkeypatch, event_index):
    plugin = analysis.OneDimensionalAnalysis()
    event_name, component_id, property_name = plugin.input_update_inputs[event_index]
    context = SimpleNamespace(triggered_prop_ids={f"{component_id}.{property_name}": component_id})
    monkeypatch.setattr(analysis, "ctx", context)
    events = {
        "importance_click": {"points": [{"customdata": "old"}]},
        "importance_label_click": {"annotation": {"name": "old"}},
    }
    events[event_name] = (
        {"points": [{"customdata": "kind"}]}
        if event_index == 0
        else {"annotation": {"name": "kind"}}
    )
    assert plugin.load_input_updates(events) == {"hyperparameter_name": {"value": "kind"}}
    assert plugin.load_input_updates({}) == {}
    context.triggered_prop_ids = {"one_dimensional_analysis-hyperparameter_name-input.value": ""}
    assert plugin.load_input_updates(events) == {}


def test_parameter_and_elite_changes_reuse_importance_estimates(monkeypatch):
    from deepcave.runs.converters.deepcave import DeepCAVERun

    analysis.clear_one_dimensional_figure_cache()
    run = DeepCAVERun(
        "importance",
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0), "kind": ["a", "b"]}),
        objectives=[Objective("score")],
    )
    for index in range(12):
        run.add(
            config={"x": index / 12, "kind": "a" if index % 2 else "b"},
            costs=[float(index)],
            start_time=0,
            end_time=index + 1,
            seed=0,
        )
    calls = []
    original = analysis.calculate_importances

    def tracked_importances(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(analysis, "calculate_importances", tracked_importances)
    inputs = {"objective_id": 0, "hyperparameter_name": "x", "elite_proportion": 0.1}
    first = analysis.OneDimensionalAnalysis.process(run, inputs)
    second = analysis.OneDimensionalAnalysis.process(
        run, {**inputs, "hyperparameter_name": "kind", "elite_proportion": 0.2}
    )
    assert calls == [1]
    assert set(first["importances"]["objective"]) == {"x", "kind"}
    assert first["importances"] == second["importances"]
    assert {row["hyperparameter_value"] for row in second["trials"]} == {"a", "b"}


def test_optuna_studies_and_optuna_only_groups_are_compatible():
    optuna_run = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    group = SimpleNamespace(
        prefix="group",
        runs=[optuna_run],
        empty=lambda: False,
        configspace={"x": 1},
    )
    mixed_group = SimpleNamespace(
        prefix="group",
        runs=[optuna_run, SimpleNamespace(prefix="SMAC3")],
        empty=lambda: False,
        configspace={"x": 1},
    )

    assert analysis.OneDimensionalAnalysis.check_run_compatibility(optuna_run)
    assert analysis.OneDimensionalAnalysis.check_run_compatibility(group)
    assert not analysis.OneDimensionalAnalysis.check_run_compatibility(mixed_group)
