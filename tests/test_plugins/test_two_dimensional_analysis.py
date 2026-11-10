from types import SimpleNamespace
from unittest.mock import Mock

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace, Constant
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from dash import dcc

from deepcave import config
from deepcave.plugins.hyperparameter import analysis_utils
from deepcave.plugins.hyperparameter import two_dimensional_analysis as analysis
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial


def test_plugin_is_registered_directly_after_one_dimensional_analysis():
    plugin_ids = [plugin.id for plugin in config.PLUGINS["Hyperparameter Analysis"]]

    one_dimensional_index = plugin_ids.index("one_dimensional_analysis")
    assert plugin_ids[one_dimensional_index + 1] == "two_dimensional_analysis"


def test_cache_key_is_versioned_for_axis_mask_geometry():
    plugin = object.__new__(analysis.TwoDimensionalAnalysis)
    inputs = {"feature": {"value": "x"}}
    legacy_key = super(analysis.TwoDimensionalAnalysis, plugin)._dict_as_key(inputs)

    assert plugin._dict_as_key(inputs) == f"{legacy_key}-v{analysis.CACHE_SCHEMA_VERSION}"


def test_planning_state_is_not_an_analysis_redraw_input():
    assert analysis.TwoDimensionalAnalysis.output_view_inputs == ()
    assert analysis.TwoDimensionalAnalysis.output_view_states == (
        ("planned_search_space_state", analysis.PLANNED_SEARCH_SPACE_STORE_ID, "data"),
    )


def _walk_components(component):
    """Yield a Dash component and all of its descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk_components(child)


def test_feature_cards_group_each_selector_with_its_range_and_bins():
    registered = []
    (controls,) = analysis.TwoDimensionalAnalysis.get_input_layout(
        lambda name, *_args, **_kwargs: registered.append(name) or name
    )
    assert registered == [
        "color_by_rank",
        "hyperparameter_name_1",
        "n_bins_1",
        "hyperparameter_name_2",
        "n_bins_2",
        "elite_proportion",
        "coverage_previous_bins",
    ]
    assert controls.id == analysis.PLANNING_CONTROLS_ID
    x_card, y_card, elite_card, context = controls.children
    for index, (card, mount) in enumerate(
        ((x_card, analysis.PLANNING_CONTROL_1_ID), (y_card, analysis.PLANNING_CONTROL_2_ID)),
        start=1,
    ):
        components = list(_walk_components(card))
        (selector,) = [c for c in components if isinstance(c, dbc.Select)]
        (bins,) = [c for c in components if isinstance(c, dbc.Input)]
        assert selector.id == f"hyperparameter_name_{index}"
        assert bins.id == f"n_bins_{index}"
        assert bins.type == "number"
        assert (bins.min, bins.max, bins.step, bins.value) == (
            analysis.MIN_BINS,
            analysis.MAX_BINS,
            1,
            analysis.N_BINS,
        )
        buttons = [c for c in components if getattr(c, "type", None) == "button"]
        assert [button.id for button in buttons] == [
            f"n_bins_{index}-decrement",
            f"n_bins_{index}-increment",
        ]
        assert any(getattr(c, "id", None) == mount for c in components)
        assert not any(getattr(c, "id", None) == analysis.PLANNING_ACTIONS_ID for c in components)
    assert x_card.children[0].children == "Hyperparameter X"
    assert y_card.children[0].children == "Hyperparameter Y"
    assert any(
        getattr(c, "id", None) == analysis.PLANNING_ACTIONS_ID for c in _walk_components(elite_card)
    )
    assert isinstance(context, dcc.Store)
    assert context.id == {
        "type": analysis.PLANNED_SEARCH_SPACE_CONTEXT_TYPE,
        "source": analysis.PLANNING_SOURCE,
    }
    # The input and output regions coexist; all mounts and controls must have unique IDs.
    output = analysis.TwoDimensionalAnalysis.get_output_layout(lambda name, *_args: name)
    components = [c for root in [controls, *output] for c in _walk_components(root)]
    ids = [str(c.id) for c in components if hasattr(c, "id")]
    assert len(ids) == len(set(ids))


def test_interactions_are_top_left_next_to_trials_above_the_binned_views():
    registered = []
    (grid,) = analysis.TwoDimensionalAnalysis.get_output_layout(
        lambda name, *_args, **_kwargs: registered.append(name) or name
    )
    assert grid.className == "analysis-grid analysis-grid--two-dimensional"
    assert registered == ["trials_graph", "elite_rate_graph", "objective_graph"]
    assert len(grid.children) == 6
    interactions, scatter, enrichment, contour, coverage, performance = grid.children
    assert "two-dimensional-trials-card" in scatter.className
    assert "two-dimensional-trials-card" not in enrichment.className + contour.className
    assert [
        card.children[0].children[0].children[0].children
        for card in (interactions, scatter, enrichment, contour)
    ] == [
        "fANOVA · Feature interactions",
        "Observed trials",
        "Elite enrichment (1x = Baseline)",
        "Empirical objective median contour",
    ]
    assert [card.children[1].id for card in (scatter, enrichment, contour)] == registered
    (heatmap,) = [c for c in _walk_components(interactions) if isinstance(c, dcc.Graph)]
    assert heatmap.id == analysis.INTERACTION_GRAPH_ID
    assert not heatmap.figure.data
    assert any(
        getattr(c, "id", None) == analysis.INTERACTION_BUTTON_ID
        for c in _walk_components(interactions)
    )
    assert all(
        graph.style == {"width": "100%", "height": f"{analysis.COVERAGE_PANEL_HEIGHT}px"}
        for graph in (heatmap, scatter.children[1], enrichment.children[1], contour.children[1])
    )
    (switch,) = [c for c in _walk_components(scatter) if isinstance(c, dbc.Switch)]
    assert switch.id == "two_dimensional_analysis-color_by_rank-input"
    assert switch.value is False
    assert not any(isinstance(c, dcc.Slider) for c in _walk_components(grid))
    assert coverage.children[0].children[0].children == "Search space coverage"
    assert coverage.style["gridColumn"] == "1"
    button = coverage.children[0].children[1]
    assert button.id == analysis.COVERAGE_BINS_BUTTON_ID
    assert button.children == "20 × 20 bins"
    assert any(
        getattr(c, "id", None) == analysis.COVERAGE_PANEL_ID for c in _walk_components(coverage)
    )
    assert (
        performance.children[0].children[0].children[0].children == "Cell performance · Best trial"
    )
    assert any(
        getattr(c, "id", None) == analysis.PERFORMANCE_PANEL_ID
        for c in _walk_components(performance)
    )


def test_coverage_projection_follows_selected_pair_and_independent_bin_counts(monkeypatch):
    import deepcave

    callbacks = {}

    def callback(*args, **kwargs):
        def register(function):
            callbacks[function.__name__] = function
            return function

        return register

    configurations = [
        {"x": 0.25, "y": 0.75},
        {"x": 0.75, "y": 0.25},
        {"x": 0.5, "y": 0.5},
    ]
    run = SimpleNamespace(
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)}),
        history=[
            SimpleNamespace(config_id=i, budget=1, seed=0, status=status, costs=[i, i * 2])
            for i, status in enumerate([Status.SUCCESS, Status.SUCCESS, Status.FAILED])
        ],
        get_config=lambda index: configurations[index],
        get_objective=lambda index: Objective(f"Objective{index}", optimize="upper"),
    )
    monkeypatch.setattr(
        deepcave,
        "app",
        SimpleNamespace(callback=callback, clientside_callback=lambda *args, **kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_run=lambda _: run), raising=False
    )
    monkeypatch.setattr(analysis.DynamicPlugin, "register_callbacks", lambda self: None)
    plugin = object.__new__(analysis.TwoDimensionalAnalysis)
    plugin.register_callbacks.__wrapped__(plugin)
    render = callbacks["render_pair_coverage"]
    label = callbacks["coverage_bins_label"]
    assert label(3, 7, [4, 4]) == "20 × 20 bins"
    assert label(20, 20, [3, 7]) == "3 × 7 bins"
    assert label(20, 20, None) == "4 × 4 bins"
    assert label(20, 7, [3, 7]) == "20 × 20 bins"
    performance = callbacks["render_pair_performance"]
    for objective_id in (0, 1):
        panel = performance("run", "x", "y", 20, 20, objective_id)
        graph = panel.children[1]
        assert graph.id == f"{analysis.PERFORMANCE_PANEL_ID}-graph"
        assert all(trace.type == "heatmap" for trace in graph.figure.data)
        values = np.asarray(graph.figure.data[1].z)
        assert values.shape == (20, 20)
        assert values[15, 5] == 0
        assert values[5, 15] == (1 if objective_id == 0 else 2)
        assert "Empty cells" not in panel.children[0].children
    assert performance("run", "x", "y", 4, 4, None) == []

    for x, y in [("x", "y"), ("y", "x")]:
        for n_bins_x, n_bins_y in [(3, 5), (7, 4), (20, 20)]:
            panel = render("run", x, y, n_bins_x, n_bins_y)
            graph = next(c for c in _walk_components(panel) if isinstance(c, dcc.Graph))
            (density,) = graph.figure.data
            assert density.type == "heatmap"
            assert np.asarray(density.z).shape == (n_bins_y, n_bins_x)
            counts = np.asarray(density.customdata, dtype=object)[:, :, 2].astype(int)
            assert counts.sum() == 2
            for configuration in configurations[:2]:
                assert (
                    counts[int(configuration[y] * n_bins_y), int(configuration[x] * n_bins_x)] >= 1
                )
            total = n_bins_x * n_bins_y
            empty = int(np.count_nonzero(counts == 0))
            assert panel.children[0].children == (
                f"Empty cells: {empty / total:.1%} ({empty} / {total})"
            )
            assert graph.figure.layout.xaxis.title.text == x
            assert graph.figure.layout.yaxis.title.text == y
            if n_bins_x == n_bins_y == 20:
                layout = graph.figure.layout
                assert layout.height - layout.margin.t - layout.margin.b >= 300
                assert graph.style["height"] == f"{layout.height}px"
                for axis in (layout.xaxis, layout.yaxis):
                    assert len(axis.tickvals) <= 5
                    assert axis.tickvals[0] == 0
                    assert axis.tickvals[-1] == 19
                    assert axis.tickangle == 0
                    assert axis.automargin is False
                assert density.customdata[19][19][0]
                assert density.customdata[19][19][1]

    for run_id, x, y in [
        (None, "x", "y"),
        ("run", None, "y"),
        ("run", "x", "x"),
        ("run", "x", "old"),
    ]:
        assert render(run_id, x, y, 4, 4) == []
    for bins in [(None, 4), (4, None), (0, 4), (21, 4)]:
        assert render("run", "x", "y", *bins) == []


@pytest.mark.parametrize(
    ("optimize", "objective_id", "best"),
    [("lower", 0, 2), ("upper", 0, 5), ("lower", 1, 20), ("upper", 1, 50)],
)
def test_cell_performance_uses_best_finite_successful_trial(optimize, objective_id, best):
    configurations = [
        {"x": 0.1, "y": 0.1},
        {"x": 0.2, "y": 0.1},
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.9},
        {"x": 0.9, "y": 0.1},
    ]
    costs = [[5, 50], [2, 20], [-100, 1000], [None, np.nan], [np.inf, -np.inf]]
    run = SimpleNamespace(
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)}),
        history=[
            SimpleNamespace(
                config_id=i,
                budget=1,
                seed=0,
                status=Status.FAILED if i == 2 else Status.SUCCESS,
                costs=values,
            )
            for i, values in enumerate(costs)
        ],
        get_config=lambda index: configurations[index],
        get_objective=lambda index: Objective(f"Objective{index}", optimize=optimize),
    )

    panel = analysis._pair_coverage(run, "x", "y", 2, 3, objective_id)
    background, trace = panel.children[1].figure.data
    values = np.asarray(trace.z)

    assert values.shape == (3, 2)
    assert values[0, 0] == best
    assert np.isfinite(values).sum() == 1
    assert trace.customdata[0][0][2] == 2
    assert trace.reversescale == (optimize == "lower")
    assert trace.colorbar.title.text == f"Objective{objective_id} · Best trial"
    assert trace.hoverongaps is False
    assert background.showscale is False
    assert background.colorscale[0][1] == "#e7d3b2"
    assert "Empty cells" not in panel.children[0].children
    assert "Occupancy" not in panel.children[0].children


def test_dense_empirical_axes_match_coverage_label_style():
    x = UniformFloatHyperparameter("x", lower=0.25, upper=25, log=True)
    y = UniformFloatHyperparameter("y", lower=0.01, upper=1, log=True)
    trials = pd.DataFrame(
        {
            "hyperparameter_1": np.geomspace(0.25, 25, 21),
            "hyperparameter_2": np.geomspace(0.01, 1, 21),
            "objective": np.linspace(0, 1, 21),
        }
    )
    _, x_groups, y_groups, threshold, _, rows = analysis._summarize_pair(
        trials, x, y, "lower", 0.1, 20, 20
    )
    x_axes = [analysis._axis_descriptor(group, x) for group in x_groups]
    y_axes = [analysis._axis_descriptor(group, y) for group in y_groups]
    figures = [
        analysis._elite_enrichment_figure(
            pd.DataFrame(rows),
            [axis["label"] for axis in x_axes],
            [axis["label"] for axis in y_axes],
            "x",
            "y",
            threshold,
            0.1,
        ),
        analysis._objective_median_figure(
            pd.DataFrame(rows), x_axes, y_axes, x, y, "x", "y", "Loss", "lower"
        ),
    ]
    assert len(x_axes) == len(y_axes) == 20
    for figure in figures:
        assert figure.layout.margin.l == 100
        assert figure.layout.margin.r == 20
        assert figure.layout.margin.b == 125
        for axis in (figure.layout.xaxis, figure.layout.yaxis):
            assert len(axis.tickvals) == 5
            assert all("<br>" in label for label in axis.ticktext)
            assert axis.tickangle == 0
            assert axis.tickfont.size == 10
            assert axis.title.standoff == 10
            assert axis.automargin is False
    assert figures[1].layout.xaxis.type == figures[1].layout.yaxis.type == "log"


def test_coverage_recommended_bins_button_restores_previous_asymmetric_binning():
    plugin = object.__new__(analysis.TwoDimensionalAnalysis)
    plugin.previous_inputs = plugin.load_inputs()
    plugin.previous_inputs["n_bins_1"]["value"] = 3
    plugin.previous_inputs["n_bins_2"]["value"] = 7
    recommended = plugin.load_input_updates({"coverage_recommended_bins": 1})
    assert recommended == {
        "n_bins_1": {"value": 20},
        "n_bins_2": {"value": 20},
        "coverage_previous_bins": {"data": [3, 7]},
    }
    plugin.previous_inputs.update(recommended)
    restored = plugin.load_input_updates({"coverage_recommended_bins": 1})
    assert restored == {
        "n_bins_1": {"value": 3},
        "n_bins_2": {"value": 7},
        "coverage_previous_bins": {"data": [3, 7]},
    }
    plugin.previous_inputs.update(restored)
    assert plugin.load_input_updates({"coverage_recommended_bins": 1}) == recommended
    plugin.previous_inputs["n_bins_1"]["value"] = 5
    assert plugin.load_input_updates({"coverage_recommended_bins": 1})[
        "coverage_previous_bins"
    ] == {"data": [5, 7]}
    assert plugin.load_input_updates({"coverage_recommended_bins": None}) == {}


def test_coverage_keeps_unobserved_categories_and_integer_values():
    configspace = ConfigurationSpace(space={"category": ["a", "b", "c"], "integer": (1, 4)})
    run = SimpleNamespace(
        configspace=configspace,
        history=[SimpleNamespace(config_id=0, budget=1, seed=0, status=Status.SUCCESS)],
        get_config=lambda _: {"category": "a", "integer": 1},
    )

    panel = analysis._pair_coverage(run, "category", "integer", 20, 20)
    figure = panel.children[1].figure
    assert np.asarray(figure.data[0].z).shape == (4, 3)
    assert panel.children[0].children == "Empty cells: 91.7% (11 / 12)"
    assert list(figure.layout.xaxis.ticktext) == ["a", "b", "c"]
    assert list(figure.layout.yaxis.ticktext) == ["1", "2", "3", "4"]


def test_coverage_logarithmic_bins_cover_full_configured_space():
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("x", lower=1, upper=10000, log=True),
            UniformFloatHyperparameter("y", lower=0, upper=1),
        ]
    )
    run = SimpleNamespace(
        configspace=configspace,
        history=[SimpleNamespace(config_id=0, budget=1, seed=0, status=Status.SUCCESS)],
        get_config=lambda _: {"x": 100, "y": 1},
    )

    panel = analysis._pair_coverage(run, "x", "y", 4, 2)
    figure = panel.children[1].figure
    assert list(figure.layout.xaxis.ticktext) == [
        "1 –<br>10",
        "10 –<br>100",
        "100 –<br>1000",
        "1000 –<br>1e+04",
    ]
    counts = np.asarray(figure.data[0].customdata, dtype=object)[:, :, 2].astype(int)
    assert counts.sum() == counts[1, 2] == 1
    assert panel.children[0].children == "Empty cells: 87.5% (7 / 8)"


def test_interaction_heatmap_uses_shared_screening_and_shows_each_pair_once(monkeypatch, tmp_path):
    objective = Objective("Loss")
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": ["a", "b"]})
    run = SimpleNamespace(
        configspace=configspace,
        get_objective=lambda objective_id: objective,
        get_highest_budget=lambda: 42,
    )
    rows = [
        {"members": ["x", "y"], "score": 0.25, "status": "Stable"},
        {"members": ["x", "z"], "score": 0.0001, "status": "Coarse"},
        {"members": ["y", "z"], "score": 0.1, "status": "Refine recommended"},
    ]
    evaluator = SimpleNamespace(screen_interactions=Mock(return_value=rows))
    model_path = tmp_path / "model.pkl"
    get_cached = Mock(return_value=(evaluator, model_path))
    save_cached = Mock()
    monkeypatch.setattr(analysis, "get_cached_fanova", get_cached)
    monkeypatch.setattr(analysis, "save_cached_fanova", save_cached)

    figure = analysis._interaction_importance_figure(run, 0)

    get_cached.assert_called_once_with(
        run, objective, 42, 10, config.CACHE_DIR / "interaction_models"
    )
    evaluator.screen_interactions.assert_called_once_with(["x", "y", "z"], top_k=5)
    save_cached.assert_called_once_with(evaluator, model_path)
    heatmap = figure.data[0]
    assert list(heatmap.x) == list(heatmap.y) == ["x", "y", "z"]
    assert list(map(list, heatmap.z)) == [
        [None, 0.25, 0.0001],
        [None, None, 0.1],
        [None, None, None],
    ]
    assert heatmap.customdata[1][2] == "Refine recommended"
    assert heatmap.hoverongaps is False
    assert heatmap.showscale is False
    assert heatmap.opacity == 0
    assert figure.layout.xaxis.side == "top"
    assert figure.layout.yaxis.range == (2.5, -0.5)
    assert [annotation.text for annotation in figure.layout.annotations[:3]] == [
        "25.00%",
        "0.01%",
        "10.00%",
    ]
    assert figure.layout.meta["interaction_pairs"] == [["x", "y"], ["x", "z"], ["y", "z"]]
    assert all(shape.type == "path" for shape in figure.layout.shapes[:3])
    assert figure.layout.shapes[0].fillcolor == "rgb(237, 218, 80)"
    assert figure.layout.shapes[1].fillcolor == "rgb(237, 240, 243)"
    assert any(annotation.text == "3 pairs" for annotation in figure.layout.annotations)
    scores = heatmap.z
    analysis._highlight_interaction_pair(figure, "y", "x")
    assert [shape.line.width for shape in figure.layout.shapes[:3]] == [2, 0, 0]
    analysis._highlight_interaction_pair(figure, "z", "y")
    assert [shape.line.width for shape in figure.layout.shapes[:3]] == [0, 0, 2]
    assert figure.data[0].z == scores


def test_interaction_screening_removes_constants_without_mutating_run(monkeypatch, tmp_path):
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    configspace.add(Constant("fixed", 1))
    configurations = {7: {"x": 0.25, "y": 0.75, "fixed": 1}}
    run = SimpleNamespace(
        configspace=configspace,
        configs=configurations,
        get_objective=lambda _: Objective("Loss"),
        get_highest_budget=lambda: 1,
    )
    evaluator = SimpleNamespace(
        screen_interactions=lambda *args, **kwargs: [
            {"members": ["x", "y"], "score": 0.1, "status": "Stable"}
        ]
    )
    get_cached = Mock(return_value=(evaluator, tmp_path / "model.pkl"))
    monkeypatch.setattr(analysis, "get_cached_fanova", get_cached)
    monkeypatch.setattr(analysis, "save_cached_fanova", Mock())

    figure = analysis._interaction_importance_figure(run, 0)

    screened_run = get_cached.call_args.args[0]
    assert list(screened_run.configspace) == ["x", "y"]
    assert screened_run.configs == {7: {"x": 0.25, "y": 0.75}}
    assert run.configspace is configspace
    assert run.configs is configurations
    assert "fixed" in run.configspace
    assert "fixed" not in figure.data[0].x


def test_heatmap_click_selects_both_axes_through_dependency_updates():
    plugin = object.__new__(analysis.TwoDimensionalAnalysis)
    run = SimpleNamespace(
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)}),
        get_objectives=lambda: [Objective("Loss")],
        get_objective_id=lambda _: 0,
    )
    inputs = plugin.load_inputs()
    inputs.update(
        plugin.load_input_updates({"interaction_cell": {"points": [{"x": "y", "y": "x"}]}})
    )

    updated = plugin.load_dependency_inputs(run, {}, inputs)

    assert plugin.input_update_inputs == (
        ("interaction_cell", analysis.INTERACTION_GRAPH_ID, "clickData"),
        ("coverage_recommended_bins", analysis.COVERAGE_BINS_BUTTON_ID, "n_clicks"),
    )
    assert updated["hyperparameter_name_1"]["value"] == "y"
    assert updated["hyperparameter_name_2"]["value"] == "x"
    for click in (None, {"points": []}, {"points": [{"x": "x", "y": "x"}]}):
        assert plugin.load_input_updates({"interaction_cell": click}) == {}


def test_interaction_callback_only_calculates_on_button_and_clears_on_context_change(monkeypatch):
    import deepcave

    callbacks = {}

    def callback(*args, **kwargs):
        def register(function):
            callbacks[function.__name__] = function
            return function

        return register

    run = object()
    monkeypatch.setattr(
        deepcave,
        "app",
        SimpleNamespace(callback=callback, clientside_callback=lambda *args, **kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_run=lambda _: run), raising=False
    )
    monkeypatch.setattr(analysis.DynamicPlugin, "register_callbacks", lambda self: None)
    plugin = object.__new__(analysis.TwoDimensionalAnalysis)
    plugin.register_callbacks.__wrapped__(plugin)
    calculate = callbacks["calculate_interactions"]
    figure = analysis.go.Figure(analysis.go.Heatmap(z=[[0.1]]))
    figure.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, line={"width": 0})
    figure.update_layout(meta={"interaction_pairs": [["x", "y"]]})
    screening = Mock(return_value=figure)
    monkeypatch.setattr(analysis, "_interaction_importance_figure", screening)

    for trigger in (None, "global-run-selector", plugin.get_internal_input_id("objective_id")):
        monkeypatch.setattr(analysis, "ctx", SimpleNamespace(triggered_id=trigger))
        assert not calculate(3, "run-id", 0, "x", "y", None).data
    screening.assert_not_called()

    monkeypatch.setattr(
        analysis, "ctx", SimpleNamespace(triggered_id=analysis.INTERACTION_BUTTON_ID)
    )
    assert calculate(4, "run-id", 0, "x", "y", None) is figure
    screening.assert_called_once_with(run, 0)
    assert figure.layout.shapes[0].line.width == 2
    monkeypatch.setattr(
        analysis,
        "ctx",
        SimpleNamespace(triggered_id=plugin.get_internal_input_id("hyperparameter_name_1")),
    )
    updated = calculate(4, "run-id", 0, "z", "y", figure.to_plotly_json())
    assert updated.layout.shapes[0].line.width == 0
    screening.assert_called_once()
    assert calculate(4, "another-run", 0, "z", "y", figure.to_plotly_json()) is analysis.no_update
    monkeypatch.setattr(
        analysis, "ctx", SimpleNamespace(triggered_id=analysis.INTERACTION_BUTTON_ID)
    )
    screening.side_effect = RuntimeError("No trees with finite positive variance")
    failed = calculate(5, "run-id", 0, "x", "y", None)
    assert not failed.data
    assert "No trees with finite positive variance" in failed.layout.annotations[0].text


def test_dependencies_exclude_time_and_enforce_distinct_hyperparameters():
    objectives = [Objective("Loss"), Objective("Accuracy", optimize="upper"), Objective("Time")]
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (1, 5), "kind": ["a", "b"]})

    class DependencyRun:
        def get_objectives(self):
            return objectives

        def get_objective_id(self, objective):
            return objectives.index(objective)

    run = DependencyRun()
    run.configspace = configspace
    inputs = {
        "objective_id": {"value": None},
        "hyperparameter_name_1": {"value": "x"},
        "hyperparameter_name_2": {"value": "x"},
        "elite_proportion": {"value": 0.2},
    }

    result = analysis.TwoDimensionalAnalysis.load_dependency_inputs(
        object.__new__(analysis.TwoDimensionalAnalysis),
        run,
        {},
        inputs,
    )

    assert [option["label"] for option in result["objective_id"]["options"]] == [
        "Loss",
        "Accuracy",
    ]
    assert result["hyperparameter_name_1"]["value"] == "x"
    assert result["hyperparameter_name_2"]["value"] != "x"
    assert "x" not in [option["value"] for option in result["hyperparameter_name_2"]["options"]]
    assert result["n_bins_1"]["value"] == analysis.N_BINS
    assert result["n_bins_2"]["value"] == analysis.N_BINS
    assert result["color_by_rank"]["value"] is False


def test_numeric_grouping_uses_four_linear_or_logarithmic_bins():
    linear = UniformFloatHyperparameter("linear", lower=1.0, upper=101.0)
    logarithmic = UniformFloatHyperparameter("logarithmic", lower=1.0, upper=10000.0, log=True)

    linear_groups = analysis_utils._group_parameter_values(
        pd.Series([1.0, 21.0, 41.0, 61.0, 101.0]),
        linear,
        n_bins=4,
        discrete_numeric=True,
    )
    logarithmic_groups = analysis_utils._group_parameter_values(
        pd.Series([1.0, 10.0, 100.0, 1000.0, 10000.0]),
        logarithmic,
        n_bins=4,
        discrete_numeric=True,
    )

    assert len(linear_groups.cat.categories) == 4
    assert len(logarithmic_groups.cat.categories) == 4
    assert 26.0 in linear_groups.cat.categories[0]
    assert 10.0 in logarithmic_groups.cat.categories[0]


def test_logarithmic_group_coordinate_stays_inside_positive_domain():
    hyperparameter = UniformFloatHyperparameter(
        "gamma",
        lower=0.0001,
        upper=20.0,
        log=True,
    )
    # pandas expands the first interval below zero for include_lowest=True.
    group = pd.Interval(-0.0009, 0.00211, closed="right")

    coordinate = analysis._group_coordinate(group, hyperparameter)

    assert coordinate == pytest.approx(np.sqrt(hyperparameter.lower * group.right))
    assert coordinate > 0

    # Previously cached analyses can still contain the invalid zero coordinate.
    assert analysis._contour_coordinates(
        [{"coordinate": 0.0}, {"coordinate": 0.01}],
        hyperparameter,
    ) == pytest.approx([hyperparameter.lower, 0.01])


def test_axis_groups_trim_empty_edges_but_keep_internal_gaps():
    groups = pd.Series(
        pd.Categorical(
            ["used-1", "used-2"],
            categories=["empty-edge", "used-1", "empty-middle", "used-2"],
            ordered=True,
        )
    )

    assert analysis._observed_categories(groups) == [
        "used-1",
        "empty-middle",
        "used-2",
    ]


@pytest.mark.parametrize(
    ("optimize", "objectives", "expected_threshold", "expected_quantiles"),
    [
        ("upper", [10.0, 9.0, 8.0, 7.0], 8.5, [1.0, 0.75, 0.5, 0.25]),
        ("lower", [1.0, 2.0, 3.0, 4.0], 2.5, [1.0, 0.75, 0.5, 0.25]),
    ],
)
def test_pair_summary_honors_objective_direction(
    optimize,
    objectives,
    expected_threshold,
    expected_quantiles,
):
    hp1 = CategoricalHyperparameter("x", choices=["a", "b"])
    hp2 = CategoricalHyperparameter("y", choices=["c", "d"])
    trials = pd.DataFrame(
        {
            "trial_number": range(4),
            "hyperparameter_1": ["a", "a", "b", "b"],
            "hyperparameter_2": ["c", "d", "c", "d"],
            "objective": objectives,
        }
    )

    summarized, x_groups, y_groups, threshold, elite_count, cells = analysis._summarize_pair(
        trials,
        hp1,
        hp2,
        optimize,
        0.5,
    )

    assert threshold == pytest.approx(expected_threshold)
    assert elite_count == 2
    assert summarized["objective_rank"].tolist() == pytest.approx([1, 2, 3, 4])
    assert summarized["rank_quantile"].tolist() == pytest.approx(expected_quantiles)
    assert x_groups == ["a", "b"]
    assert y_groups == ["c", "d"]
    assert sum(cell["elite"] for cell in cells) == 2
    assert [cell["elite_enrichment"] for cell in cells] == pytest.approx([2.0, 2.0, 0.0, 0.0])


def test_elite_threshold_ties_are_all_included_and_joint_gaps_remain_empty():
    hp1 = CategoricalHyperparameter("x", choices=["a", "b"])
    hp2 = CategoricalHyperparameter("y", choices=["c", "d"])
    trials = pd.DataFrame(
        {
            "trial_number": range(4),
            "hyperparameter_1": ["a", "a", "b", "b"],
            "hyperparameter_2": ["c", "d", "c", "c"],
            "objective": [10.0, 9.0, 9.0, 0.0],
        }
    )

    _, x_groups, y_groups, threshold, elite_count, cells = analysis._summarize_pair(
        trials,
        hp1,
        hp2,
        "upper",
        0.5,
    )
    matrix = analysis._cell_matrix(
        pd.DataFrame(cells),
        [str(group) for group in x_groups],
        [str(group) for group in y_groups],
        "elite_rate_pct",
    )

    assert threshold == pytest.approx(9.0)
    assert elite_count == 3
    assert matrix[1][1] is None


def test_process_keeps_repeated_configurations_and_original_trial_numbers():
    objective = Objective("Accuracy", optimize="upper")
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    history = [
        Trial(0, 0, -1, [0.8], 0.0, 0.0, 1.0, Status.SUCCESS, {"optuna_trial_number": 4}),
        Trial(0, 0, -1, [0.9], 0.0, 1.0, 2.0, Status.SUCCESS, {"optuna_trial_number": 9}),
    ]
    run = SimpleNamespace(
        history=history,
        configspace=configspace,
        get_config=lambda _config_id: {"x": 0.5, "y": 0.75},
        get_objective=lambda _objective_id: objective,
    )

    outputs = analysis.TwoDimensionalAnalysis.process(
        run,
        {
            "objective_id": 0,
            "hyperparameter_name_1": "x",
            "hyperparameter_name_2": "y",
            "elite_proportion": 0.5,
        },
    )

    assert [trial["trial_number"] for trial in outputs["trials"]] == [4, 9]


def test_feature_changes_reuse_the_objective_wide_trial_table():
    objective = Objective("Accuracy", optimize="upper")
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)})
    configurations = {
        0: {"x": 0.1, "y": 0.2, "z": 0.3},
        1: {"x": 0.4, "y": 0.5, "z": 0.6},
    }

    class CachedRun:
        id = "cached-run"
        latest_change = 1.0
        configs = configurations
        history = [
            Trial(0, 0, -1, [0.8], 0.0, 0.0, 1.0, Status.SUCCESS, {}),
            Trial(1, 0, -1, [0.9], 0.0, 1.0, 2.0, Status.SUCCESS, {}),
        ]

        def __init__(self):
            self.configspace = configspace

        def get_objective(self, _objective_id):
            return objective

        def get_config(self, _config_id):
            raise AssertionError("stored configurations should be read without reconstruction")

    run = CachedRun()
    analysis_utils._TRIAL_TABLE_CACHE.clear()
    first_table = analysis_utils._get_trial_table(run, 0)
    second_table = analysis_utils._get_trial_table(run, 0)

    assert first_table is second_table
    assert list(first_table.columns) == [
        "trial_number",
        "objective",
        "x",
        "y",
        "z",
        "objective_rank",
        "rank_quantile",
    ]
    assert analysis._collect_trials(run, 0, ["x", "y"])["x"].tolist() == [0.1, 0.4]
    assert analysis._collect_trials(run, 0, ["x", "z"])["z"].tolist() == [0.3, 0.6]

    run.latest_change = 2.0
    refreshed_table = analysis_utils._get_trial_table(run, 0)
    assert refreshed_table is not first_table
    assert all(":2.0:" in key[1] for key in analysis_utils._TRIAL_TABLE_CACHE)


def test_independent_bin_counts_feed_enrichment_and_contour(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Accuracy", optimize="upper")
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    configurations = [{"x": x / 5, "y": y / 5} for y in range(6) for x in range(6)]
    history = [
        Trial(
            index,
            0,
            -1,
            [float(index)],
            0.0,
            0.0,
            float(index),
            Status.SUCCESS,
            {"optuna_trial_number": index},
        )
        for index in range(len(configurations))
    ]
    run = SimpleNamespace(
        history=history,
        configspace=configspace,
        get_config=lambda config_id: configurations[config_id],
        get_objective=lambda _objective_id: objective,
    )
    inputs = {
        "objective_id": 0,
        "hyperparameter_name_1": "x",
        "hyperparameter_name_2": "y",
        "elite_proportion": 0.25,
        "n_bins_1": 3,
        "n_bins_2": 5,
    }

    outputs = analysis.TwoDimensionalAnalysis.process(run, inputs)
    _, enrichment_figure, contour_figure = analysis.TwoDimensionalAnalysis.load_outputs(
        run,
        inputs,
        outputs,
    )

    assert len(outputs["axis_1"]) == 3
    assert len(outputs["axis_2"]) == 5
    assert all(axis["kind"] == "interval" for axis in outputs["axis_1"])
    assert all(axis["lower"] < axis["upper"] for axis in outputs["axis_1"])
    assert np.asarray(enrichment_figure.data[0].z).shape == (5, 3)
    assert np.asarray(contour_figure.data[0].z).shape == (7, 5)
    assert contour_figure.data[0].x[0] == pytest.approx(outputs["axis_1"][0]["lower"])
    assert contour_figure.data[0].x[-1] == pytest.approx(outputs["axis_1"][-1]["upper"])
    assert contour_figure.data[0].y[0] == pytest.approx(outputs["axis_2"][0]["lower"])
    assert contour_figure.data[0].y[-1] == pytest.approx(outputs["axis_2"][-1]["upper"])
    assert len(contour_figure.layout.xaxis.tickvals) == 3
    assert len(contour_figure.layout.yaxis.tickvals) == 5


def test_numeric_outputs_use_contour_and_preserve_missing_cells(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Accuracy", optimize="upper")
    run = SimpleNamespace(
        configspace={
            "x": UniformFloatHyperparameter("x", lower=1.0, upper=100.0, log=True),
            "y": UniformFloatHyperparameter("y", lower=0.0, upper=10.0),
        },
        get_objective=lambda _objective_id: objective,
    )
    inputs = {
        "objective_id": 0,
        "hyperparameter_name_1": "x",
        "hyperparameter_name_2": "y",
        "elite_proportion": 0.5,
    }
    outputs = {
        "trials": [
            {
                "trial_number": 1,
                "hyperparameter_1": 1.0,
                "hyperparameter_2": 0.0,
                "objective": 0.8,
                "objective_rank": 2.0,
                "rank_quantile": 0.5,
                "is_elite": False,
                "elite_rank": None,
            },
            {
                "trial_number": 2,
                "hyperparameter_1": 100.0,
                "hyperparameter_2": 10.0,
                "objective": 0.9,
                "objective_rank": 1.0,
                "rank_quantile": 1.0,
                "is_elite": True,
                "elite_rank": 1.0,
            },
        ],
        "cells": [
            {
                "group_1": "low-x",
                "group_2": "low-y",
                "n": 1,
                "elite": 0,
                "elite_rate_pct": 0.0,
                "elite_median_rank": None,
                "objective_median": 0.8,
            },
            {
                "group_1": "high-x",
                "group_2": "high-y",
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_median_rank": 1.0,
                "objective_median": 0.9,
            },
        ],
        "axis_1": [
            {"label": "low-x", "coordinate": 1.0},
            {"label": "high-x", "coordinate": 100.0},
        ],
        "axis_2": [
            {"label": "low-y", "coordinate": 0.0},
            {"label": "high-y", "coordinate": 10.0},
        ],
        "elite_threshold": 0.85,
        "elite_count": 1,
    }

    trials_figure, elite_rate_figure, objective_figure = (
        analysis.TwoDimensionalAnalysis.load_outputs(
            run,
            inputs,
            outputs,
        )
    )

    assert trials_figure.data[0].type == "scattergl"
    assert trials_figure.layout.xaxis.type == "log"
    assert trials_figure.data[0].marker.reversescale is False
    assert list(trials_figure.data[0].marker.color) == pytest.approx([0.8, 0.9])
    assert elite_rate_figure.data[0].type == "heatmap"
    assert elite_rate_figure.data[0].z[0][1] is None
    assert elite_rate_figure.data[0].z[1][1] == pytest.approx(2.0)
    assert elite_rate_figure.data[0].zmax == pytest.approx(2.0)
    assert "Elite enrichment: %{z:.2f}×" in elite_rate_figure.data[0].hovertemplate
    assert "Elite rate: %{customdata[2]:.1f}%" in elite_rate_figure.data[0].hovertemplate
    assert elite_rate_figure.data[0].showscale is False
    assert objective_figure.data[0].type == "contour"
    assert objective_figure.data[0].connectgaps is False
    assert objective_figure.data[0].z[0][1] is None
    assert objective_figure.layout.xaxis.type == "log"
    assert (
        trials_figure.data[0].marker.colorscale
        == elite_rate_figure.data[0].colorscale
        == objective_figure.data[0].colorscale
    )
    assert trials_figure.data[0].marker.colorbar.orientation == "v"
    assert trials_figure.data[0].marker.colorbar.lenmode == "pixels"
    assert 0 < trials_figure.data[0].marker.colorbar.len < analysis.TRIALS_PANEL_HEIGHT
    assert trials_figure.layout.height == analysis.TRIALS_PANEL_HEIGHT
    assert objective_figure.data[0].colorbar.orientation == "h"
    assert trials_figure.data[0].marker.colorbar.title.text == "Accuracy"
    assert objective_figure.data[0].colorbar.title.text == "Accuracy"
    assert trials_figure.layout.annotations == ()
    assert objective_figure.layout.annotations == ()
    assert all(
        figure.layout.title.text is None
        for figure in (trials_figure, elite_rate_figure, objective_figure)
    )

    rank_inputs = {**inputs, "color_by_rank": True}
    rank_figure, _, _ = analysis.TwoDimensionalAnalysis.load_outputs(
        run,
        rank_inputs,
        outputs,
    )
    assert list(rank_figure.data[0].marker.color) == pytest.approx([2.0, 1.0])
    assert rank_figure.data[0].marker.reversescale is True
    assert rank_figure.data[0].marker.colorbar.title.text == "Trial rank"


@pytest.mark.parametrize("n_bins", [5, 6, 7])
@pytest.mark.parametrize("swap_axes", [False, True])
def test_enrichment_axes_stay_categorical_when_integer_bins_become_single_values(n_bins, swap_axes):
    hp1 = UniformFloatHyperparameter("learning_rate", 0.004, 0.08, log=True)
    hp2 = UniformIntegerHyperparameter("depth", 10, 15)
    trials = pd.DataFrame(
        {
            "trial_number": range(6),
            "hyperparameter_1": np.geomspace(0.004, 0.08, 6),
            "hyperparameter_2": range(10, 16),
            "objective": range(6),
        }
    )
    _, xs, ys, threshold, _, cells = analysis._summarize_pair(
        trials, hp1, hp2, "lower", 0.1, n_bins_1=5, n_bins_2=n_bins
    )
    x_labels, y_labels = [str(x) for x in xs], [str(y) for y in ys]
    if n_bins >= 6:
        assert y_labels == [str(value) for value in range(10, 16)]
    else:
        assert all(isinstance(group, pd.Interval) for group in ys)
    cells = pd.DataFrame(cells)
    if swap_axes:
        x_labels, y_labels = y_labels, x_labels
        cells = cells.rename(columns={"group_1": "group_2", "group_2": "group_1"})
    figure = analysis._elite_enrichment_figure(cells, x_labels, y_labels, "x", "y", threshold, 0.1)
    assert figure.layout.xaxis.type == figure.layout.yaxis.type == "category"
    assert list(figure.data[0].x) == list(figure.layout.xaxis.categoryarray) == x_labels
    assert list(figure.data[0].y) == list(figure.layout.yaxis.categoryarray) == y_labels
    assert all(
        0 <= a.x < len(x_labels) and 0 <= a.y < len(y_labels) for a in figure.layout.annotations
    )


def test_enrichment_distinguishes_observed_zero_from_an_empty_bin():
    cells = pd.DataFrame(
        [
            {
                "group_1": "observed-zero",
                "group_2": "row",
                "n": 10,
                "elite": 0,
                "elite_rate_pct": 0.0,
                "elite_median_rank": None,
            },
            {
                "group_1": "enriched",
                "group_2": "row",
                "n": 10,
                "elite": 2,
                "elite_rate_pct": 20.0,
                "elite_median_rank": 1.5,
            },
            {
                "group_1": "empty",
                "group_2": "row",
                "n": 0,
                "elite": 0,
                "elite_rate_pct": 0.0,
                "elite_median_rank": None,
            },
        ]
    )

    figure = analysis._elite_enrichment_figure(
        cells,
        ["observed-zero", "enriched", "empty"],
        ["row"],
        "learning_rate",
        "max_depth",
        elite_threshold=0.5,
        elite_proportion=0.1,
    )

    assert list(figure.data[0].z[0]) == [0.0, 2.0, None]
    annotation_text = [annotation.text for annotation in figure.layout.annotations]
    assert "0×" in annotation_text
    assert "2.00×" in annotation_text
    assert "No data" in annotation_text
    assert annotation_text.count("n = 0") == 1
    assert figure.data[0].colorscale[0][1] == "#440154"
    assert figure.data[0].colorscale[-1][1] == "#fde725"
    assert figure.data[0].showscale is False
    assert figure.layout.xaxis.domain == (0, 1)
    assert figure.layout.xaxis.scaleanchor is None
    assert all("<br>" not in label for label in figure.layout.xaxis.ticktext)
    assert figure.layout.xaxis.title.standoff == 10
    assert figure.layout.margin.b == analysis.ENRICHMENT_BOTTOM_MARGIN
    assert "Elite enrichment" not in [annotation.text for annotation in figure.layout.annotations]

    figure_with_twenty_percent_elite = analysis._elite_enrichment_figure(
        cells,
        ["observed-zero", "enriched", "empty"],
        ["row"],
        "learning_rate",
        "max_depth",
        elite_threshold=0.5,
        elite_proportion=0.2,
    )
    assert list(figure_with_twenty_percent_elite.data[0].z[0]) == [0.0, 1.0, None]


@pytest.mark.parametrize(
    ("x_bin_count", "y_bin_count", "shows_cell_text"),
    [
        (15, 6, True),
        (16, 6, False),
        (15, 7, False),
    ],
)
def test_enrichment_hides_cell_text_for_dense_grids(
    x_bin_count,
    y_bin_count,
    shows_cell_text,
):
    x_labels = [f"x-{index}" for index in range(x_bin_count)]
    y_labels = [f"y-{index}" for index in range(y_bin_count)]
    cells = pd.DataFrame(
        [
            {
                "group_1": x_labels[0],
                "group_2": y_labels[0],
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_median_rank": 1.0,
            }
        ]
    )

    figure = analysis._elite_enrichment_figure(
        cells,
        x_labels,
        y_labels,
        "feature-1",
        "feature-2",
        elite_threshold=0.5,
        elite_proportion=0.1,
    )

    assert bool(figure.layout.annotations) is shows_cell_text
    assert "Elite enrichment" in figure.data[0].hovertemplate


def test_categorical_axis_uses_objective_heatmap(monkeypatch):
    monkeypatch.setattr(analysis, "save_image", lambda *_args: None)
    objective = Objective("Loss", optimize="lower")
    run = SimpleNamespace(
        id="study",
        configspace={
            "kind": CategoricalHyperparameter("kind", choices=["a", "b"]),
            "x": UniformFloatHyperparameter("x", lower=0.0, upper=1.0),
        },
        get_objective=lambda _objective_id: objective,
    )
    inputs = {
        "objective_id": 0,
        "hyperparameter_name_1": "kind",
        "hyperparameter_name_2": "x",
        "elite_proportion": 0.5,
        "planned_search_space_state": {
            "version": 2,
            "revision": 1,
            "runs": {"study": {"objective:0": {"kind": {"choices": ["b"]}}}},
            "event_sequences": {},
            "last_change": None,
        },
    }
    outputs = {
        "trials": [
            {
                "trial_number": 0,
                "hyperparameter_1": "a",
                "hyperparameter_2": 0.5,
                "objective": 1.0,
                "objective_rank": 1.0,
                "rank_quantile": 1.0,
                "is_elite": True,
                "elite_rank": 1.0,
            }
        ],
        "cells": [
            {
                "group_1": "a",
                "group_2": "0.5",
                "n": 1,
                "elite": 1,
                "elite_rate_pct": 100.0,
                "elite_median_rank": 1.0,
                "objective_median": 1.0,
            }
        ],
        "axis_1": [{"label": "a", "coordinate": "a"}],
        "axis_2": [{"label": "0.5", "coordinate": 0.5}],
        "elite_threshold": 1.0,
        "elite_count": 1,
    }

    _, _, objective_figure = analysis.TwoDimensionalAnalysis.load_outputs(run, inputs, outputs)

    assert objective_figure.data[0].type == "heatmap"
    assert objective_figure.data[0].reversescale is True
    assert len(objective_figure.layout.shapes) == 1
    assert objective_figure.layout.shapes[0].x0 == -0.5
    assert objective_figure.layout.shapes[0].x1 == 0.5


def test_reduced_planning_controls_adapt_to_parameter_type():
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("x", lower=0.01, upper=1.0, log=True),
            CategoricalHyperparameter("kind", choices=["a", "b"]),
        ]
    )
    numeric, categorical = analysis.configspace_metadata(configspace, ["x", "kind"])

    numeric_control = analysis._planning_control(numeric, {"min": 0.1, "max": 2.0}, 0)
    category_control = analysis._planning_control(categorical, {"choices": ["b"]}, 1)
    numeric_components = list(_walk_components(numeric_control))
    category_components = list(_walk_components(category_control))

    (slider,) = [item for item in numeric_components if isinstance(item, dcc.RangeSlider)]
    (dropdown,) = [item for item in category_components if isinstance(item, dcc.Dropdown)]
    assert slider.id == {
        "type": analysis.PLANNING_RANGE_TYPE,
        "index": 0,
        "context": "",
    }
    assert slider.min < 0
    assert slider.max > 1
    assert slider.marks is None
    assert numeric_control.children[0].children[1].children == "0.1 – 2"
    assert dropdown.id == {
        "type": analysis.PLANNING_CATEGORIES_TYPE,
        "index": 1,
        "context": "",
    }
    assert dropdown.value == ["b"]
    assert any(
        getattr(item, "className", None) == "planner-extension-segment"
        for item in numeric_components
    )


def test_planning_feature_identity_survives_other_feature_switch():
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("x", lower=0.0, upper=1.0),
            UniformFloatHyperparameter("y", lower=0.0, upper=2.0),
            CategoricalHyperparameter("kind", choices=["a", "b"]),
        ]
    )

    first_pair, first_version, first_context = analysis._selected_planning_metadata(
        configspace, ["x", "y"], 0
    )
    second_pair, second_version, second_context = analysis._selected_planning_metadata(
        configspace, ["x", "kind"], 0
    )

    assert first_version == second_version
    assert first_context == second_context
    assert first_pair[0]["index"] == second_pair[0]["index"]
    assert first_pair[0]["planning_context"] == second_pair[0]["planning_context"]
    assert first_pair[1]["index"] != second_pair[1]["index"]

    _, _, other_objective_context = analysis._selected_planning_metadata(configspace, ["x", "y"], 1)
    assert other_objective_context != first_context


@pytest.mark.parametrize("children", [None, []])
def test_empty_planning_mount_is_not_treated_as_an_existing_control(children):
    assert analysis._planning_control_is_mounted(children) is False


@pytest.mark.parametrize("children", [{"type": "Div"}, ["control"]])
def test_rendered_planning_mount_is_preserved_on_other_feature_switch(children):
    assert analysis._planning_control_is_mounted(children) is True


def test_logarithmic_interval_mask_uses_exact_fractional_bin_positions():
    parameter = {
        "name": "x",
        "parameter_type": "float",
        "current_min": 1.0,
        "current_max": 100.0,
        "log": True,
        "step": None,
    }
    axes = [
        {"kind": "interval", "lower": 1.0, "upper": 10.0},
        {"kind": "interval", "lower": 10.0, "upper": 100.0},
    ]

    included, domain = analysis._binned_axis_segments(
        axes,
        parameter,
        {"min": np.sqrt(10.0), "max": np.sqrt(1000.0)},
    )

    assert domain == (-0.5, 1.5)
    assert len(included) == 1
    assert included[0] == pytest.approx((0.0, 1.0))


def test_rectangular_mask_does_not_double_overlay_corners():
    figure = analysis.go.Figure()

    analysis._apply_rectangular_mask(
        figure,
        [(0.25, 0.75)],
        (0.0, 1.0),
        [(0.25, 0.75)],
        (0.0, 1.0),
    )

    shapes = list(figure.layout.shapes)
    assert len(shapes) == 4
    assert all(shape.fillcolor == analysis.MASK_COLOR for shape in shapes)
    vertical = [shape for shape in shapes if shape.y0 == 0.0 and shape.y1 == 1.0]
    horizontal = [shape for shape in shapes if shape.x0 == 0.25 and shape.x1 == 0.75]
    assert len(vertical) == 2
    assert len(horizontal) == 2


def test_only_suitable_optuna_studies_are_compatible():
    suitable = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1, "y": 2})
    one_parameter = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": 1})
    group = SimpleNamespace(
        prefix="group",
        runs=[suitable],
        empty=lambda: False,
        configspace={"x": 1, "y": 2},
    )

    assert analysis.TwoDimensionalAnalysis.check_run_compatibility(suitable)
    assert not analysis.TwoDimensionalAnalysis.check_run_compatibility(one_parameter)
    assert analysis.TwoDimensionalAnalysis.check_run_compatibility(group)
