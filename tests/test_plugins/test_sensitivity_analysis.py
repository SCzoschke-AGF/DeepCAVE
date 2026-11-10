"""Native layout, objective wiring and diagnostic figures for sensitivity analysis."""

import copy
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dash import Dash
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

import deepcave
from deepcave.config import Config
from deepcave.plugins.sensitivity_analysis import (
    SensitivityAnalysis,
    render_detail,
    table_projection,
)
from deepcave.runs.group import Group
from deepcave.runs.handler import (
    DATA_SEED_ATTRIBUTE_CACHE_KEY,
    MODEL_SEED_ATTRIBUTE_CACHE_KEY,
)
from deepcave.utils.cache import Cache
from deepcave.utils.sensitivity_references import REFERENCE_CACHE_KEY, reference_key
from deepcave.utils.sensitivity import (
    distribution_figure,
    explorer_figure,
    global_sensitivity_figure,
    hyperparameter_figure,
    randomness_range_figure,
    seed_distribution_figure,
    seed_mean_figure,
    seed_profile_figure,
    trials_by_seed_figure,
    seed_matrix_figure,
)

from tests.test_evaluators.test_sensitivity import MAPPINGS, groups, prepared
from tests.test_utils.test_sensitivity_references import (
    reference_database,
    reference_group,
)  # noqa: F401
from tests.test_evaluators.test_sensitivity import make_run  # noqa: F401


def text(node):
    if isinstance(node, Component):
        return str(getattr(node, "label", "")) + text(getattr(node, "children", []))
    if isinstance(node, (list, tuple)):
        return " ".join(text(child) for child in node)
    return str(node)


@pytest.fixture
def design(make_run):  # noqa: F811
    return groups(
        prepared(
            make_run,
            [
                {"x": x, "m": m, "d": d, "y": x + m + d, "attrs": {"fold_auc": [x + m, None, d]}}
                for x, m, d in product((0.25, 0.5), range(4), range(4))
                if (m, d) != (3, 3)
            ],
        ),
        "combined",
    )


def test_native_registration_and_tables(design, make_run):  # noqa: F811
    assert SensitivityAnalysis.id in [p.id for p in Config().PLUGINS["Configuration Analyses"]]
    run = make_run([{"m": i} for i in range(4)])
    assert SensitivityAnalysis.check_run_compatibility(run)
    assert SensitivityAnalysis.check_run_compatibility(Group("seeds", [run]))
    assert not SensitivityAnalysis.check_run_compatibility(SimpleNamespace(prefix="SMAC"))
    layout = SensitivityAnalysis.get_output_layout(lambda name, *_: name)
    assert "Minimum seeds" not in text(layout)
    assert "Sensitivity mode" not in text(layout)
    assert "Configuration Explorer" not in text(layout)
    assert "Objective Distribution" in text(layout)
    assert "Ranking Stability" not in text(layout)
    assert "Variance Decomposition" not in text(layout)
    assert "Reference Trials" in text(layout)
    assert "Fold Diagnostics" not in text(layout)
    assert "Mean Objective by Seed" not in text(layout)
    assert "Seed axis" not in text(layout)
    assert "Configuration × Seed Pair" not in text(layout)
    assert "Pairwise Comparison" not in text(layout)
    assert "Points only" not in str(layout)
    assert "Seed Profiles" not in text(layout)
    assert "Swap axes / lines" not in text(layout)
    rows, columns = table_projection(design)
    assert len(rows) == 2
    assert {c["id"] for c in columns} >= {"mean", "sd", "iqr", "param:rate", "seeds", "evaluations"}


def test_figures_keep_seed_pairs_missing_cells_and_log_scale(design):
    figure = explorer_figure(design, True)
    assert list(figure.data[0].x) == [g["sd"] for g in design]
    assert len(figure.data[0].customdata) == 2
    matrix = seed_matrix_figure(design[0])
    assert len(matrix.data[0].z) == 4
    assert sum(cell is None for row in matrix.data[0].z for cell in row) == 1
    assert matrix.layout.xaxis.type == "category" and matrix.layout.yaxis.type == "category"
    assert hyperparameter_figure(design, "rate").layout.xaxis.type == "log"
    categorical = copy.deepcopy(design)
    for group in categorical:
        group["parameters"]["rate"] = "category"
    assert hyperparameter_figure(categorical, "rate").layout.xaxis.type == "category"
    distribution = distribution_figure(design)
    assert len(distribution.data) == 2 * len(design)
    assert distribution.layout.xaxis.title.text == "Configuration"
    for i, group in enumerate(design):
        box, points = distribution.data[2 * i : 2 * i + 2]
        assert box.type == "box" and points.type == "scatter"
        assert list(box.y) == [rep["objective"] for rep in group["replicates"]]
        assert list(points.y) == [row["objective"] for row in group["rows"]]
        assert list(points.customdata[0]) == [group["rows"][0]["id"], group["id"]]
        assert all(abs(x - i) < box.width / 2 for x in points.x)


def test_global_sensitivity_shares_scales_and_preserves_each_parameter_semantics(design):
    observed = []
    for index, count in enumerate((4, 8, 16)):
        group = copy.deepcopy(design[0])
        group.update(
            id=f"group-{index}", mean=0.56 + index * 0.005, sd=0.01 * (index + 1), seeds=count
        )
        group["parameters"] = {
            "category": ["a", "b", "c"][index],
            "count": 2 * (index + 1),
            "linear": 0.1 * (index + 1),
            "log": 10 ** (-index - 1),
            "mixed": 10 ** (-index - 1),
        }
        if index != 1:
            group["parameters"]["optional"] = index
        for row in group["rows"]:
            row["distributions"] = {
                "category": {"kind": "categorical"},
                "count": {"kind": "integer"},
                "linear": {"kind": "float"},
                "log": {"kind": "float", "log": True},
                "mixed": {"kind": "float", "log": index == 2},
                "optional": {"kind": "integer"},
            }
        observed.append(group)
    unavailable = {**observed[0], "parameters": {"unavailable": 1}, "sd": None}
    figure = global_sensitivity_figure([*observed, unavailable])
    panels = [trace for trace in figure.data if trace.customdata]
    assert len(panels) == 6
    assert [trace.xaxis for trace in panels] == ["x", "x2", "x3", "x4", "x5", "x6"]
    assert [figure.layout[f"xaxis{suffix}"].type for suffix in ("", "2", "3", "4", "5", "6")] == [
        "category",
        "linear",
        "linear",
        "log",
        "category",
        "linear",
    ]
    assert figure.layout.xaxis2.tickformat == "d"
    assert list(figure.layout.xaxis2.tickvals) == [2, 4, 6]
    assert list(figure.layout.xaxis4.ticktext) == ["1e-3", "1e-2", "0.1"]
    assert figure.layout.yaxis4.domain[1] < figure.layout.yaxis.domain[0]
    matches = [figure.layout[f"yaxis{suffix}"].matches for suffix in ("", "2", "3", "4", "5", "6")]
    assert matches.count(None) == 1
    assert len({match for match in matches if match}) == 1
    assert all(trace.marker.coloraxis == "coloraxis" for trace in panels)
    assert figure.layout.coloraxis.cmin == pytest.approx(0.56)
    assert figure.layout.coloraxis.cmax == pytest.approx(0.57)
    assert len(panels[-1].x) == 2  # Conditional parameters retain only actual observations.
    assert [item[0] for item in panels[-1].customdata] == ["group-0", "group-2"]
    assert panels[0].marker.size[-1] ** 2 / panels[0].marker.size[0] ** 2 == pytest.approx(4)
    assert [trace.name for trace in figure.data if trace.showlegend] == ["4", "8", "16"]
    assert not figure.layout.coloraxis.reversescale
    minimized = global_sensitivity_figure([{**g, "direction": "lower"} for g in observed])
    assert minimized.layout.coloraxis.reversescale


@pytest.mark.parametrize("counts", [(15, 15, 15), (150, 150, 150), (5, 15, 45), (50, 150, 450)])
def test_global_sensitivity_sizes_are_relative_to_visible_configurations(design, counts):
    observed = [{**design[0], "id": str(index), "seeds": count} for index, count in enumerate(counts)]
    figure = global_sensitivity_figure(observed)
    sizes = list(figure.data[0].marker.size)
    assert sizes[1] == pytest.approx(8)
    if len(set(counts)) == 1:
        assert sizes == pytest.approx([8, 8, 8])
    else:
        assert sizes[0] < 8 < sizes[2]
        assert sizes[2] ** 2 / sizes[0] ** 2 == pytest.approx(9)
    legend = {int(trace.name): trace.marker.size for trace in figure.data if trace.showlegend}
    assert sizes == pytest.approx([legend[count] for count in counts])


def test_distribution_overlays_assigned_reference_without_changing_seed_statistics(design):
    trial = {
        "study_name": "original-hpo",
        "trial_number": 554,
        "values": [99.0],
        "metrics": {"metric:Objective0": 99.0},
    }
    references = {reference_key(design[0]): trial}
    baseline = distribution_figure(design)
    figure = distribution_figure(design, references=references)
    assert len(figure.data) == len(baseline.data) + 1
    for before, after in zip(baseline.data, figure.data):
        assert before.to_plotly_json() == after.to_plotly_json()
    marker = figure.data[-1]
    assert list(marker.x) == [0]
    assert list(marker.y) == [99.0]
    assert marker.marker.symbol == "diamond"
    assert marker.customdata is None
    assert "original-hpo · Trial 554" in marker.hovertext[0]
    assert figure.layout.showlegend
    assert len(distribution_figure(design[1:], references=references).data) == 2


def test_seed_aggregation_distributions_and_switchable_lines(make_run):  # noqa: F811
    design = groups(
        prepared(
            make_run,
            [
                {"m": 0, "d": 0, "y": 0},
                {"m": 0, "d": 0, "y": 4},
                {"m": 0, "d": 1, "y": 10},
                {"m": 1, "d": 0, "y": 20},
                {"x": 0.5, "m": 0, "d": 1, "y": 30},
            ],
        ),
        "combined",
    )
    distributions = seed_distribution_figure(design)
    assert distributions.layout.boxmode == "group"
    assert distributions.layout.xaxis.showgrid is False
    assert [
        (shape.x0, shape.x1) for shape in distributions.layout.shapes if shape.x0 == shape.x1
    ] == [(0.5, 0.5)]
    assert distributions.layout.xaxis.showticklabels is False
    assert [label.text for label in distributions.layout.annotations[:3]] == [
        "Data seeds",
        "Model seeds",
        f"<b>{design[0]['label']}</b>",
    ]
    assert list(distributions.data[0].y) == [11, 10, 30]
    assert list(distributions.data[1].y) == [6, 20, 30]
    trials = trials_by_seed_figure(design)
    data_points, model_means, model_points, data_means = trials.data
    assert data_points.marker.color == "#2563eb"
    assert model_points.marker.color == "#d97706"
    assert list(data_points.y) == [0, 4, 10, 20, 30]
    assert list(model_points.y) == [0, 4, 20, 10, 30]
    assert list(model_means.y) == [6, 20, 30]
    assert list(data_means.y) == [11, 10, 30]
    assert model_points.marker.opacity == data_points.marker.opacity == 0.2
    assert list(data_means.error_y.array) == [9, 0, 0]
    assert list(data_means.error_y.arrayminus) == [11, 0, 0]
    assert list(model_means.error_y.array) == [4, 0, 0]
    assert list(model_means.error_y.arrayminus) == [6, 0, 0]
    assert model_means.marker.symbol == data_means.marker.symbol == "line-ew"
    assert set(trials.layout.xaxis.ticktext) == {"0", "1"}
    assert trials.layout.xaxis.range == (-0.5, 1.5)
    assert trials.layout.legend.groupclick == "togglegroup"
    assert len(trials.layout.annotations) == 5 * len(design)
    seed_labels = [label for label in trials.layout.annotations if label.y == -0.15]
    assert [label.text for label in seed_labels] == [
        "Model seeds",
        "Model seeds",
        "Data seeds",
        "Data seeds",
    ]
    top_labels = [label.text for label in trials.layout.annotations if label.y == 1.04]
    assert top_labels == [
        "Data seed<br>variability",
        "Data seed<br>variability",
        "Model seed<br>variability",
        "Model seed<br>variability",
    ]
    assert all(label.font.color == "#d97706" for label in seed_labels[:2])
    assert all(label.font.color == "#2563eb" for label in seed_labels[2:])
    assert len(
        [shape for shape in trials.layout.shapes if shape.yref == "paper" and shape.y0 < 0]
    ) == 2 * len(design)
    assert "Trial 0" in data_points.hovertext[0]
    assert all(trace.boxpoints == "all" and trace.pointpos == 0 for trace in distributions.data)
    for axis, expected in (("model_seed", [6, 20]), ("data_seed", [11, 10])):
        lines = seed_mean_figure(design, axis)
        assert len(lines.data) == 2
        assert list(lines.data[0].x) == ["0", "1"]
        assert list(lines.data[0].y) == expected
        assert list(lines.data[1].y) == ([30, None] if axis == "model_seed" else [None, 30])
        assert lines.data[0].mode == "lines+markers"
        assert lines.layout.yaxis.title.text == "Mean objective"
        assert lines.layout.xaxis.type == "category"


def test_randomness_range_points_hover_and_scale(design):
    figure = randomness_range_figure(design)
    points = [trace for trace in figure.data if trace.type == "scatter" and trace.mode == "markers"]
    assert [trace.name for trace in points] == [
        "Total range",
        "Data randomness",
        "Model randomness",
    ]
    assert [trace.marker.color for trace in points] == ["#64748b", "#2563eb", "#d97706"]
    assert figure.layout.barmode == "overlay" and figure.layout.showlegend
    assert figure.layout.legend.groupclick == "togglegroup"
    assert figure.layout.xaxis.categoryarray == tuple(group["id"] for group in design)
    assert figure.layout.yaxis.title.text == "Objective range (max − min)"
    low, high = figure.layout.yaxis.range
    assert low == 0 and high > 0
    total = points[0]
    assert total.name == "Total range" and total.mode == "markers"
    assert not any(trace.type == "bar" for trace in figure.data)
    assert list(total.y) == pytest.approx(
        [
            max(pair["objective"] for pair in group["replicates"])
            - min(pair["objective"] for pair in group["replicates"])
            for group in design
        ]
    )
    for index in range(len(design)):
        assert total.x[index] == pytest.approx(index - 0.7 / 3)
    assert "All observed seed combinations" in total.hovertemplate
    for trace, fixed_label in zip(points[1:], ("Model seed", "Data seed")):
        assert list(trace.marker.line.color) == [
            trace.marker.color if symbol == "line-ew" else "white"
            for symbol in trace.marker.symbol
        ]
        assert all(width > 0 for width in trace.marker.line.width)
        assert trace.marker.opacity > 0.9
        assert len(trace.y) == 8
        assert all(low <= value < high for value in trace.y)
        for position, span, custom in zip(trace.x, trace.y, trace.customdata):
            group = design[round(position)]
            assert custom[0] == group["label"]
            assert custom[1] == fixed_label
            assert custom[2] in ("0", "1", "2", "3")
            assert span == pytest.approx(custom[4] - custom[3])
    singleton = copy.deepcopy(design[:1])
    singleton[0]["replicates"] = singleton[0]["replicates"][:1]
    small = randomness_range_figure(singleton)
    assert all(list(trace.y) == [] for trace in small.data)
    assert small.layout.yaxis.range == (0, 1)
    assert randomness_range_figure([]).layout.yaxis.range == (0, 1)


def test_randomness_range_summary_shows_stem_maximum_and_all_points():
    group = {
        "id": "c4",
        "label": "C4",
        "replicates": [
            {"model_seed": model, "data_seed": data, "objective": value}
            for model, maximum in enumerate((0.050, 0.055, 0.089))
            for data, value in enumerate((0, maximum))
        ],
    }
    figure = randomness_range_figure([group])
    stem, summary, points = figure.data[3:6]
    assert stem.type == "scatter" and stem.mode == "lines"
    assert list(stem.y) == [0, 0.089, None]
    assert stem.x[0] == stem.x[1] == summary.x[0]
    assert summary.mode == "markers+text" and list(summary.y) == [0.089]
    assert summary.text[0] == "<b>Data<br>0.089</b>"
    assert list(points.y) == [0.050, 0.055, 0.089]
    assert points.x[-1] == summary.x[0]
    assert points.marker.size[-1] > max(points.marker.size[:-1])
    assert list(points.marker.symbol) == ["circle", "circle", "line-ew"]
    assert summary.marker.size > max(points.marker.size)
    assert all(abs(position - summary.x[0]) < 0.1225 for position in points.x)
    assert stem.legendgroup == summary.legendgroup == points.legendgroup
    assert stem.showlegend is summary.showlegend is False


@pytest.mark.parametrize("model_effect,data_effect", [(0.0347, 0.0003), (0.0003, 0.0347)])
def test_total_needle_has_its_own_head_and_label(model_effect, data_effect):
    group = {
        "id": "configuration",
        "label": "Configuration",
        "replicates": [
            {
                "model_seed": model,
                "data_seed": data,
                "objective": model * model_effect + data * data_effect,
            }
            for model, data in product(range(2), repeat=2)
        ],
    }
    figure = randomness_range_figure([group])
    stem, label, head = figure.data[:3]
    assert list(stem.y) == [0, model_effect + data_effect, None]
    assert stem.x[0] == stem.x[1] == label.x[0] == head.x[0]
    assert label.text[0] == f"<b>Total<br>{model_effect + data_effect:.3g}</b>"
    assert label.marker.size > head.marker.size[0]
    assert head.marker.symbol[0] == "line-ew"
    assert head.x[0] < figure.data[5].x[0] < figure.data[8].x[0]


def test_seed_means_and_randomness_ranges_share_horizontal_alignment(design):
    # Keep categories in their slots even when a configuration has no eligible ranges.
    sparse = copy.deepcopy(design[0])
    sparse["id"], sparse["label"] = "sparse", "Sparse configuration"
    sparse["replicates"] = sparse["replicates"][:1]
    for selected in (design, design[::-1], design[:1], [sparse, *design], []):
        means = seed_distribution_figure(selected)
        ranges = randomness_range_figure(selected)
        assert (
            means.layout.xaxis.range
            == ranges.layout.xaxis.range
            == (-0.5, max(len(selected) - 0.5, 0.5))
        )
        assert ranges.layout.xaxis.type == "linear"
        assert ranges.layout.xaxis.tickvals == tuple(range(len(selected)))
        assert ranges.layout.xaxis.ticktext == tuple(f"<b>{group['label']}</b>" for group in selected)
        assert means.layout.xaxis.categoryarray == tuple(group["id"] for group in selected)
        assert means.layout.margin.l == ranges.layout.margin.l == 70
        assert means.layout.margin.r == ranges.layout.margin.r == 35
        assert means.layout.yaxis.automargin is ranges.layout.yaxis.automargin is False
        assert means.layout.boxgap == ranges.layout.boxgap
        assert means.layout.boxgroupgap == ranges.layout.boxgroupgap
        for source, stem in enumerate(trace for trace in ranges.data if trace.mode == "lines"):
            expected = [
                index + (source - 1) * 0.7 / 3
                for index, group in enumerate(selected)
                if len(group["replicates"]) > 1
            ]
            assert list(stem.x[::3]) == pytest.approx(expected)


def test_seed_profiles_share_scale_colors_and_keep_raw_evaluations(make_run):  # noqa: F811
    records = [
        {"x": x, "m": m, "d": d, "y": x + m + d}
        for x, m, d in product((0.1, 0.2, 0.3, 0.4, 0.5), (0, 1), (2, 10))
        if (m, d) != (1, 10)
    ]
    records.append({"x": 0.1, "m": 0, "d": 2, "y": 8})
    design = groups(prepared(make_run, records), "combined")
    figure = seed_profile_figure(design)
    markers = [trace for trace in figure.data if trace.mode == "markers"]
    lines = [trace for trace in figure.data if trace.mode == "lines"]
    assert len(figure.layout.annotations) == 5
    assert sum(len(trace.y) for trace in markers) == len(records)
    assert {item[0] for trace in markers for item in trace.customdata} == {
        row["id"] for group in design for row in group["rows"]
    }
    assert sum(trace.showlegend for trace in markers) == 2
    for model in ("Model seed 0", "Model seed 1"):
        assert len({trace.marker.color for trace in markers if trace.name == model}) == 1
    assert figure.layout.xaxis.ticktext == ("2", "10")
    assert figure.layout.yaxis.range == figure.layout.yaxis5.range
    assert figure.layout.yaxis.range[0] < min(record["y"] for record in records)
    assert figure.layout.yaxis.range[1] > max(record["y"] for record in records)
    assert figure.layout.xaxis6.visible is False
    assert list(lines[0].y) == pytest.approx([(2.1 + 8) / 2, 10.1])
    assert list(lines[1].y) == [3.1, None]
    swapped = seed_profile_figure(design, x_axis="model_seed")
    swapped_markers = [trace for trace in swapped.data if trace.mode == "markers"]
    swapped_lines = [trace for trace in swapped.data if trace.mode == "lines"]
    assert swapped.layout.xaxis.ticktext == ("0", "1")
    assert swapped.layout.xaxis5.title.text == "Model seed"
    assert swapped.layout.yaxis.range == figure.layout.yaxis.range
    assert sum(len(trace.y) for trace in swapped_markers) == len(records)
    assert {trace.name for trace in swapped_markers} == {"Data seed 2", "Data seed 10"}
    assert list(swapped_lines[0].y) == pytest.approx([(2.1 + 8) / 2, 3.1])
    assert list(swapped_lines[1].y) == [10.1, None]
    for seed in ("Data seed 2", "Data seed 10"):
        assert len({trace.marker.color for trace in swapped_markers if trace.name == seed}) == 1


def test_details_empty_states_and_methodological_messages(design):
    data = {"groups": design}
    active = design[0]["id"]
    selected = [g["id"] for g in design]
    assert "Select a configuration" in text(render_detail("distribution", None, [], data))
    assert "All evaluations" not in text(render_detail("distribution", active, selected, data))
    overview = render_detail("distribution", None, [], data)
    assert not any(hasattr(child, "figure") for child in overview.children)
    trials_card = overview.children[0]
    assert trials_card.children[0].children == "Trials by Seed per Configuration"
    assert trials_card.children[1].id == "sensitivity-trials-by-seed"
    range_card = overview.children[1]
    assert range_card.children[0].children == "Potential performance impact by randomness source"
    assert range_card.children[1].id == "sensitivity-randomness-range"
    assert "Filled areas extend" not in text(overview)
    assert "No configurations" in text(render_detail("distribution", None, [], {"groups": []}))
    effects = render_detail("effects", active, [], data)
    assert len(effects.children) == 1
    assert effects.children[0].figure.data[0].type == "heatmap"
    assert "not calibrated probabilities" in text(render_detail("ranking", active, selected, data))
    assert "Uneven configuration sampling" in text(
        render_detail("global", None, [], data)
    )
    global_view = render_detail("global", None, [], data)
    assert global_view.children[1].id == "sensitivity-global-scatter"
    assert "No hyperparameters" in text(render_detail("global", None, [], {"groups": []}))
    assert "balanced counts" in text(render_detail("variance", active, [], data))


def test_callbacks_use_sidebar_objective_and_cache_groups_independently_of_active_selection(
    make_run, monkeypatch  # noqa: F811
):
    run = make_run([{"m": m, "y": m, "attrs": {"score": 10 - m}} for m in range(4)])
    app = Dash(__name__, suppress_callback_exceptions=True)
    monkeypatch.setattr(deepcave, "_api_mode", False)
    monkeypatch.setattr(deepcave, "app", app, raising=False)
    cache = Cache()
    cache.set(MODEL_SEED_ATTRIBUTE_CACHE_KEY, value=MAPPINGS["model"])
    cache.set(DATA_SEED_ATTRIBUTE_CACHE_KEY, value=MAPPINGS["data"])
    monkeypatch.setattr(deepcave, "c", cache, raising=False)
    prepared_cache = {}
    monkeypatch.setattr(
        deepcave,
        "rc",
        SimpleNamespace(
            get=lambda source, plugin, key: prepared_cache.get((source.id, key)),
            set=lambda source, plugin, key, value: prepared_cache.update({(source.id, key): value}),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        deepcave, "run_handler", SimpleNamespace(get_runs=lambda **_: [run]), raising=False
    )
    SensitivityAnalysis()
    callbacks = {
        key: value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "callback" in value
    }
    load = callbacks["sensitivity-source.data"]
    aggregate = callbacks["sensitivity-groups.data"]
    data = load(True, run.id)
    assert len(prepared_cache) == 1 and load(True, run.id) == data
    native = aggregate(data, {"value": "objective:0"})
    derived = aggregate(data, {"value": "user_attr:score", "direction": "lower"})
    assert native["mode"] == "combined"
    single_seed_type = {**data, "mappings": {"model": MAPPINGS["model"]}}
    assert aggregate(single_seed_type, {"value": "objective:0"})["mode"] == "model"
    assert native["groups"][0]["mean"] == 1.5
    assert derived["groups"][0]["mean"] == 8.5 and derived["direction"] == "lower"
    assert derived["groups"][0]["direction"] == "lower"
    aggregate_entry = app.callback_map["sensitivity-groups.data"]
    assert all(
        item["id"] not in ("sensitivity-active", "sensitivity-comparison")
        for item in aggregate_entry["inputs"]
    )
    detail_entry = next(
        entry for key, entry in app.callback_map.items() if "sensitivity-content.children" in key
    )
    assert all(
        item["id"] != "sensitivity-show-reference-objectives" for item in detail_entry["inputs"]
    )
    reference = {
        "study_name": "original-hpo",
        "trial_number": 554,
        "metrics": {"metric:Objective0": 99.0},
        "user_attrs": {},
    }
    cache.set(REFERENCE_CACHE_KEY, reference_key(native["groups"][0]), value=reference)
    toggle = next(
        entry["callback"].__wrapped__
        for key, entry in app.callback_map.items()
        if "sensitivity-distribution.figure" in key
        and entry["inputs"][0]["id"] == "sensitivity-show-reference-objectives"
    )
    patch, description = toggle(
        True, native, None, [{"id": group["id"]} for group in native["groups"]]
    )
    operations = patch.to_plotly_json()["operations"]
    assert all(operation["location"][0] in ("data", "layout") for operation in operations)
    append = next(operation for operation in operations if operation["operation"] == "Append")
    assert append["params"]["value"]["y"] == [99.0]
    assert "Red diamonds" in description
    boxplot = next(
        entry["callback"].__wrapped__
        for key, entry in app.callback_map.items()
        if "sensitivity-distribution.figure" in key
        and entry["inputs"][0]["id"] == "sensitivity-groups"
    )
    figure, _ = boxplot(native, None, [{"id": g["id"]} for g in native["groups"]], [], None, True)
    assert figure.data[-1].name == "Reference trial objective"
    assert figure.data[-1].y == (99.0,)
    details = next(
        value for key, value in callbacks.items() if "sensitivity-content.children" in key
    )
    rows = [{"id": group["id"]} for group in native["groups"]]
    content, axis_style, *_ = details(
        "distribution",
        None,
        [],
        native,
        1,
        [],
        [],
        rows,
    )
    assert "No configurations" in text(content) and axis_style == {"display": "none"}


def test_reference_callbacks_search_only_on_click_and_persist_per_configuration(
    reference_database, reference_group, monkeypatch, tmp_path  # noqa: F811
):
    import deepcave.plugins.sensitivity_analysis as module

    database, _ = reference_database
    app = Dash(__name__, suppress_callback_exceptions=True)
    settings = tmp_path / "settings.json"
    cache = Cache(settings)
    cache.set("optuna_database", value=str(database))
    monkeypatch.setattr(deepcave, "_api_mode", False)
    monkeypatch.setattr(deepcave, "app", app, raising=False)
    monkeypatch.setattr(deepcave, "c", cache, raising=False)
    monkeypatch.setattr(deepcave, "rc", SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        deepcave,
        "run_handler",
        SimpleNamespace(get_working_directory=lambda: tmp_path),
        raising=False,
    )
    search_database = Mock(wraps=module.find_reference_trials)
    monkeypatch.setattr(module, "find_reference_trials", search_database)
    SensitivityAnalysis()
    callbacks = {
        key: value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "callback" in value
    }
    search_entry = next(
        entry
        for key, entry in app.callback_map.items()
        if "sensitivity-reference-results.data" in key
    )
    assert search_entry["inputs"] == [
        {"id": "sensitivity-reference-search", "property": "n_clicks"}
    ]
    search = search_entry["callback"].__wrapped__
    render = callbacks["sensitivity-reference-controls.children"]
    save = callbacks["sensitivity-reference-save-status.children"]
    reference_group["label"] = "C-test"
    changed = copy.deepcopy(reference_group)
    changed.update(id="second-configuration", label="C-second")
    changed["parameters"]["rate"] = 0.5
    conditional = copy.deepcopy(reference_group)
    conditional.update(id="another-condition", label="C-test · data=47")
    data = {
        "groups": [reference_group, changed, conditional],
        "mappings": {"model": "trial:model_seed"},
    }
    rows = [{"id": group["id"]} for group in data["groups"]]
    with pytest.raises(PreventUpdate):
        search(0, data, None, [])
    assert len(render(data, {}, None, rows)) == 2
    assert not search_database.called
    results, message = search(1, data, None, rows)
    assert search_database.call_count == 1
    assert "Found 2 matching trials" in text(message)
    controls = render(data, results, None, rows)
    field = controls[0].children[1]
    second = controls[1].children[1]
    assert field.value is None and len(field.options) == 1
    identity = field.id
    value = field.options[0]["value"]
    second_value = second.options[0]["value"]
    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=identity))
    assert "saved" in text(save([value, None], [identity, second.id], results))
    saved = Cache(settings).get(REFERENCE_CACHE_KEY, reference_key(reference_group))
    assert saved["study_name"] == "original-hpo"
    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=second.id))
    assert "saved" in text(save([value, second_value], [identity, second.id], results))
    assert (
        Cache(settings).get(REFERENCE_CACHE_KEY, reference_key(changed))["study_name"]
        == "other-parameters"
    )
    monkeypatch.setattr(deepcave, "c", Cache(settings))
    restored = render(data, {}, None, rows)[0].children[1]
    assert restored.value == value and len(restored.options) == 1
    assert len(render(data, results, [changed["id"]], rows)) == 1
    assert "No configurations" in text(render(data, results, [], rows))
    assert search_database.call_count == 1
    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=identity))
    save([], [], results)
    assert Cache(settings).get(REFERENCE_CACHE_KEY, reference_key(reference_group))["id"] == value
    assert "cleared" in text(save([None, second_value], [identity, second.id], results))
    assert Cache(settings).get(REFERENCE_CACHE_KEY, reference_key(reference_group)) is None
    assert Cache(settings).get(REFERENCE_CACHE_KEY, reference_key(changed))["id"] == second_value
