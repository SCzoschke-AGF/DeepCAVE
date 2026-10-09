from types import SimpleNamespace

import dash_bootstrap_components as dbc
import pytest
from ConfigSpace import ConfigurationSpace

from deepcave.plugins.objective import multi_metrics_comparison as comparison
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial


class DummyRun:
    prefix = "Optuna"
    id = "study"
    name = "Study"

    def __init__(self, trials=None):
        self._objectives = [
            Objective("Loss", optimize="lower"),
            Objective("user_attr:score", optimize="upper"),
            Objective("Time", optimize="lower"),
        ]
        self.meta = {"user_attr_objectives": {"score": 1}}
        self.history = trials or []

    def get_objectives(self):
        return self._objectives

    def get_objective_ids(self):
        return list(range(len(self._objectives)))

    def get_objective(self, objective_id):
        if objective_id is None:
            return None
        return self._objectives[objective_id]


def _trial(number, x, y, status=Status.SUCCESS):
    return Trial(
        config_id=number,
        budget=0.0,
        seed=-1,
        costs=[x, y, 1.0],
        cpu_time=1.0,
        start_time=0.0,
        end_time=1.0,
        status=status,
        additional={"optuna_trial_number": number},
    )


def _run():
    return DummyRun(
        [
            _trial(10, 10.0, 1.0),
            _trial(11, 10.0, 4.0),
            _trial(12, 8.0, 3.0),
            _trial(13, 7.0, 2.0),
        ]
    )


def _inputs(**changes):
    inputs = {
        "x_metric": "objective:0",
        "x_direction": "maximize",
        "y_metric": "user_attr:score",
        "y_direction": "minimize",
        "trial_mode": "top_n",
        "n": 2,
        "selection": "compare",
        "scale": "values",
        "display": "selected",
        "show_pareto": True,
    }
    inputs.update(changes)
    return inputs


def test_plugin_accepts_only_individual_optuna_studies():
    assert comparison.MultiMetricsComparison.check_run_compatibility(
        SimpleNamespace(prefix="Optuna")
    )
    assert not comparison.MultiMetricsComparison.check_run_compatibility(
        SimpleNamespace(prefix="SMAC3")
    )
    assert not comparison.MultiMetricsComparison.check_run_compatibility(
        SimpleNamespace(prefix="group")
    )


def test_dependency_inputs_discover_metrics_directions_and_clamp_n():
    plugin = comparison.MultiMetricsComparison()
    inputs = plugin.load_inputs()
    dependency = plugin.load_dependency_inputs(_run(), {}, inputs)

    assert dependency["x_metric"] == {
        "options": [
            {"label": "Objective · Loss", "value": "objective:0"},
            {"label": "User attr · score", "value": "user_attr:score"},
        ],
        "value": "objective:0",
    }
    assert dependency["y_metric"]["value"] == "user_attr:score"
    assert dependency["x_direction"]["value"] == "minimize"
    assert dependency["y_direction"]["value"] == "maximize"
    assert dependency["n"] == {"value": 4, "max": 4, "disabled": True}
    assert all(option["disabled"] for option in dependency["selection"]["options"])
    assert all(option["disabled"] for option in dependency["display"]["options"])

    inputs["trial_mode"]["value"] = "top_n"
    inputs["x_metric"]["value"] = "user_attr:score"
    inputs["x_direction"]["value"] = "minimize"
    changed = plugin.load_dependency_inputs(_run(), {"x_metric": {"value": "objective:0"}}, inputs)
    assert changed["x_direction"]["value"] == "maximize"
    assert changed["n"]["disabled"] is False
    assert not any(option["disabled"] for option in changed["selection"]["options"])
    assert not any(option["disabled"] for option in changed["display"]["options"])

    preserved = plugin.load_dependency_inputs(
        _run(), {"x_metric": {"value": "user_attr:score"}}, inputs
    )
    assert preserved["x_direction"]["value"] == "minimize"

    inputs["n"]["value"] = None
    editing = plugin.load_dependency_inputs(_run(), inputs, inputs)
    assert editing["n"] == {"value": None, "max": 4, "disabled": False}


def test_input_controls_use_compact_groups_with_directions_under_advanced():
    registered = []
    (controls,) = comparison.MultiMetricsComparison.get_input_layout(
        lambda name, attribute="value", **_kwargs: registered.append(name) or name
    )

    subtitle, primary, secondary = controls.children
    assert subtitle.children == "Compare two metrics, analyze top trials and their agreement."
    assert primary.className == "multi-metrics-toolbar__primary"
    assert secondary.className == "multi-metrics-toolbar__secondary"
    assert isinstance(primary.children[2].children[1], dbc.RadioItems)
    assert primary.children[4].children.children[0].children == "Top N trials by"
    selection = primary.children[4].children.children[1]
    assert isinstance(selection, dbc.RadioItems)
    assert [option["label"] for option in selection.options] == [
        "Metric 1",
        "Metric 2",
        "Compare both",
    ]
    assert all(option["disabled"] for option in selection.options)
    assert isinstance(secondary.children[0].children[1], dbc.RadioItems)
    display = secondary.children[1].children.children[1]
    assert isinstance(display, dbc.RadioItems)
    assert [option["label"] for option in display.options] == ["All trials", "Top N only"]
    assert all(option["disabled"] for option in display.options)

    advanced = secondary.children[-1]
    assert advanced.children[0].children == "Advanced (metric direction)"
    assert [field.children[0].children for field in advanced.children[1].children] == [
        "Metric 1 direction",
        "Metric 2 direction",
    ]
    assert registered == [
        "x_metric",
        "y_metric",
        "trial_mode",
        "n",
        "selection",
        "scale",
        "display",
        "show_pareto",
        "x_direction",
        "y_direction",
    ]


def test_process_builds_global_ranks_exact_sets_overlap_and_display_context():
    outputs = comparison.MultiMetricsComparison.process(_run(), _inputs())
    trials = {row["trial_number"]: row for row in outputs["trials"]}

    assert (trials[10]["x_rank"], trials[11]["x_rank"]) == (1, 1)
    assert trials[12]["x_rank"] == 3
    assert [trials[number]["y_rank"] for number in (10, 13, 12, 11)] == [1, 2, 3, 4]
    assert outputs["set_x"] == [10, 11]
    assert outputs["set_y"] == [10, 13]
    assert outputs["analysis_set"] == [10, 11, 13]
    assert outputs["display_context"] == [12]
    assert outputs["pareto_set"] == [10]
    assert outputs["overlap"]["intersection"] == 1
    assert outputs["overlap"]["only_x"] == 1
    assert outputs["overlap"]["only_y"] == 1
    assert outputs["overlap"]["union"] == 3
    assert outputs["overlap"]["jaccard"] == pytest.approx(1 / 3)
    assert outputs["overlap"]["overlap_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [("x", [10, 11]), ("y", [10, 13]), ("compare", [10, 11, 13])],
)
def test_process_applies_each_top_n_selection(selection, expected):
    outputs = comparison.MultiMetricsComparison.process(_run(), _inputs(selection=selection))
    assert outputs["analysis_set"] == expected


def test_process_filters_invalid_trials_and_clamps_requested_n():
    run = _run()
    run.history.extend(
        [
            _trial(20, float("nan"), 2.0),
            _trial(21, 5.0, 3.0, status=Status.FAILED),
        ]
    )

    outputs = comparison.MultiMetricsComparison.process(run, _inputs(n=50))

    assert outputs["universe"] == [10, 11, 12, 13]
    assert outputs["effective_n"] == 4
    assert outputs["set_x"] == [10, 11, 12, 13]
    assert outputs["set_y"] == [10, 11, 12, 13]


def test_same_metric_is_allowed_with_independent_directions():
    outputs = comparison.MultiMetricsComparison.process(
        _run(),
        _inputs(
            y_metric="objective:0",
            x_direction="maximize",
            y_direction="minimize",
        ),
    )

    assert outputs["x_name"] == outputs["y_name"] == "Loss"
    assert outputs["set_x"] == [10, 11]
    assert outputs["set_y"] == [12, 13]


def test_correlations_are_descriptive_and_undefined_for_constant_data():
    outputs = comparison.MultiMetricsComparison.process(_run(), _inputs(trial_mode="all"))
    assert outputs["correlations"]["pearson"] is not None
    assert outputs["correlations"]["spearman"] is not None

    constant = DummyRun([_trial(1, 1.0, 2.0), _trial(2, 1.0, 3.0)])
    undefined = comparison.MultiMetricsComparison.process(constant, _inputs(trial_mode="all"))
    assert undefined["correlations"] == {"pearson": None, "spearman": None}


def test_display_context_does_not_change_pareto_and_is_only_rendered_on_request(monkeypatch):
    monkeypatch.setattr(comparison, "save_image", lambda *_args: None)
    outputs = comparison.MultiMetricsComparison.process(_run(), _inputs())

    selected = comparison.MultiMetricsComparison.load_outputs(
        _run(), _inputs(display="selected", show_pareto=False), outputs
    )
    shown = comparison.MultiMetricsComparison.load_outputs(
        _run(), _inputs(display="all", show_pareto=False), outputs
    )

    assert "Other trials" not in [trace.name for trace in selected[0].data]
    assert "Other trials" in [trace.name for trace in shown[0].data]
    assert [trace.name for trace in selected[0].data] == [
        "Top N by Metric 1",
        "Top N by Metric 2",
        "Top N by Metric 1 and Metric 2",
    ]
    assert selected[0].layout.showlegend is False
    overlap_trace = selected[0].data[-1]
    assert overlap_trace.marker.color == comparison.BLUE
    assert overlap_trace.marker.line.color == comparison.VIOLET
    assert selected[1] == {}
    agreement_tiles = selected[2][1].children
    assert [tile.children[0].children for tile in agreement_tiles] == ["Jaccard", "Overlap rate"]
    assert agreement_tiles[1].children[1].children == "50 % (1/2)"
    venn = selected[2][0].figure
    metric_labels = [
        annotation
        for annotation in venn.layout.annotations
        if annotation.text in ("Metric 1", "Metric 2")
    ]
    assert all(annotation.y > venn.layout.shapes[0].y1 for annotation in metric_labels)


def test_rank_scale_is_global_and_pareto_overlay_uses_analysis_set(monkeypatch):
    monkeypatch.setattr(comparison, "save_image", lambda *_args: None)
    outputs = comparison.MultiMetricsComparison.process(_run(), _inputs())
    rendered = comparison.MultiMetricsComparison.load_outputs(
        _run(), _inputs(scale="rank", display="all"), outputs
    )
    figure = rendered[0]

    assert figure.layout.xaxis.title.text == "<b>Rank — Loss</b>"
    assert figure.layout.yaxis.title.text == "<b>Rank — score</b>"
    pareto = next(trace for trace in figure.data if trace.name == "Pareto front")
    assert pareto.showlegend is False
    assert list(pareto.customdata) == [10]
    background = next(trace for trace in figure.data if trace.name == "Other trials")
    assert list(background.x) == [3]


def test_pareto_overlay_does_not_enable_legend_for_one_point_series(monkeypatch):
    monkeypatch.setattr(comparison, "save_image", lambda *_args: None)
    inputs = _inputs(selection="x", display="selected", show_pareto=True)
    outputs = comparison.MultiMetricsComparison.process(_run(), inputs)

    figure = comparison.MultiMetricsComparison.load_outputs(_run(), inputs, outputs)[0]

    assert [trace.name for trace in figure.data] == ["Top N by Metric 1", "Pareto front"]
    assert figure.layout.showlegend is False


def test_parallel_coordinates_show_only_global_best_trials_over_full_search_space():
    run = _run()
    run.configspace = ConfigurationSpace(
        space={
            "learning_rate": (0.0, 10.0),
            "small_gap": (0.0, 10.0),
            "depth": (0.0, 10.0),
            "kind": ["a", "b", "c"],
        }
    )
    configurations = {
        10: {"learning_rate": 2.0, "small_gap": 1.0, "depth": 9.0, "kind": "a"},
        11: {"learning_rate": 8.0, "small_gap": 3.0, "depth": 8.0, "kind": "a"},
        12: {"learning_rate": 5.0, "small_gap": 4.0, "depth": 7.0, "kind": "b"},
        13: {"learning_rate": 4.0, "small_gap": 2.0, "depth": 6.0, "kind": "a"},
    }
    run.get_config = lambda config_id: configurations[config_id]
    encoded_values = {
        "learning_rate": lambda configuration: configuration["learning_rate"] / 10.0,
        "small_gap": lambda configuration: configuration["small_gap"] / 10.0,
        "depth": lambda configuration: configuration["depth"] / 10.0,
        "kind": lambda configuration: {"a": 0.0, "b": 0.5, "c": 1.0}[configuration["kind"]],
    }
    run.encode_config = lambda configuration: [
        encoded_values[name](configuration) for name in run.configspace.keys()
    ]
    inputs = _inputs(y_direction="maximize", n=1, selection="x")
    outputs = comparison.MultiMetricsComparison.process(run, inputs)
    for row in outputs["trials"]:
        row.pop("config_id")  # Simulate processed output cached before config IDs were embedded.

    figure = comparison._best_configurations_figure(run, outputs)

    assert [trace.name for trace in figure.data] == [
        "Best Metric 1 (Loss) · Trial 10",
        "Best Metric 2 (score) · Trial 11",
    ]
    assert list(figure.data[0].y) == [0.0, 0.1, 0.2, 0.9]
    assert list(figure.data[1].y) == [0.0, 0.3, 0.8, 0.8]
    assert len(figure.layout.shapes) == 4
    assert all(shape.y0 == comparison.VALUE_RANGE[0] for shape in figure.layout.shapes)
    assert all(shape.y1 == comparison.VALUE_RANGE[1] for shape in figure.layout.shapes)
    annotation_labels = {annotation.text for annotation in figure.layout.annotations}
    assert {"0.0", "10.0", "a", "b", "c"} <= annotation_labels
    axis_labels = [
        annotation.text for annotation in figure.layout.annotations if annotation.yref == "paper"
    ]
    assert axis_labels == ["kind", "small_gap", "learning_rate", "depth"]


def test_output_layout_registers_plot_and_sidebar_outputs():
    registered = []
    (grid,) = comparison.MultiMetricsComparison.get_output_layout(
        lambda name, attribute="value": registered.append((name, attribute)) or name
    )

    assert grid.className == "multi-metrics-grid"
    sidebar = grid.children[1]
    assert [card.children[0].children for card in sidebar.children] == [
        "Analysis & Insights",
        "Agreement & overlap",
    ]
    assert registered == [
        ("graph", "figure"),
        ("agreement_container", "style"),
        ("agreement", "children"),
        ("insights", "children"),
        ("parallel_graph", "figure"),
    ]
