import math
from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float
from dash import dcc, html

from deepcave import config
from deepcave.plugins.hyperparameter import near_optimal_configurations as near_optimal
from deepcave.runs.converters.optuna import OptunaRun
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status


def _payload():
    return {
        "run_id": "study",
        "objective_name": "Accuracy",
        "optimize": "upper",
        "criterion": "top_percent",
        "criterion_value": 5.0,
        "criterion_label": "Top 5%",
        "distance_metric": "normalized",
        "metric_warning": None,
        "importance_error": None,
        "importance": {"x": 1.0},
        "parameters": ["x"],
        "profile_parameters": ["x"],
        "sampler": "TPESampler",
        "sampler_note": "Observed trials only.",
        "summary": {
            "best": 0.9,
            "eligible": 2,
            "total_completed": 2,
            "selected": 2,
            "median_objective": 0.85,
            "objective_min": 0.8,
            "objective_max": 0.9,
            "median": 0.5,
            "q25": 0.5,
            "q75": 0.5,
        },
        "baseline": {"median": 0.5, "lower": 0.5, "upper": 0.5},
        "diversity_curve": [
            {"percent": percent, "count": 1, "median": None, "q25": None, "q75": None}
            for percent in (1, 2, 5, 10, 20, 50, 75, 100)
        ],
        "spreads": [
            {
                "parameter": "x",
                "near_optimal_spread": 0.5,
                "overall_spread": 0.5,
                "relative_spread": 1.0,
                "near_optimal_min": 0.18,
                "near_optimal_median": 0.33,
                "near_optimal_max": 0.49,
                "near_optimal_coverage": 0.31,
                "near_optimal_active": 1.0,
                "overall_active": 1.0,
                "kind": "numeric",
            }
        ],
        "clusters": {
            "count": 1,
            "profiles": [
                {
                    "cluster": 1,
                    "size": 2,
                    "objective_median": 0.85,
                    "objective_min": 0.8,
                    "objective_max": 0.9,
                    "parameters": [
                        {"parameter": "x", "summary": "0.5 [0.25 – 0.75]", "active": 1.0}
                    ],
                }
            ],
        },
        "heatmap": {
            "matrix": [[0, 0.5], [0.5, 0]],
            "trial_ids": [3, 8],
            "clusters": [1, 1],
            "silhouette": [0.0, 0.0],
            "linkage": [[0, 1, 0.5, 2]],
            "shown": 2,
            "total": 2,
        },
        "distributions": {
            "x": {"overall": [0.0, 1.0], "near": [0.0, 1.0], "clusters": {"1": [0.0, 1.0]}}
        },
        "trials": [
            {"id": 3, "trial_id": 3, "objective": 0.8, "cluster": 1, "hp::x": 0.0},
            {"id": 8, "trial_id": 8, "objective": 0.9, "cluster": 1, "hp::x": 1.0},
        ],
    }


def test_plugin_and_footprint_are_registered_together():
    plugin_ids = [plugin.id for plugin in config.PLUGINS["Configuration Analyses"]]
    assert plugin_ids == [
        "footprint",
        "near_optimal_configurations",
        "configuration_neighborhood",
    ]


def test_plugin_is_optuna_only():
    compatible = SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": object()})
    incompatible = SimpleNamespace(
        prefix="SMAC3v2", empty=lambda: False, configspace={"x": object()}
    )
    assert near_optimal.NearOptimalConfigurations.check_run_compatibility(compatible)
    assert not near_optimal.NearOptimalConfigurations.check_run_compatibility(incompatible)


def test_defaults_match_primary_rashomon_workflow():
    defaults = near_optimal.NearOptimalConfigurations().load_inputs()
    assert defaults["objective_id"]["value"] is None
    assert defaults["criterion"]["value"] == "top_percent"
    assert defaults["top_percent"]["value"] == 5
    assert defaults["distance_metric"]["value"] == "importance_weighted"
    assert defaults["cluster_count"]["value"] == "auto"
    assert [option["label"] for option in defaults["top_percent"]["options"]] == [
        "1%",
        "2%",
        "5%",
        "10%",
        "20%",
    ]


def test_input_layout_matches_the_two_card_control_panel():
    layout = near_optimal.NearOptimalConfigurations.get_input_layout(
        lambda name, *_args, **_kwargs: f"test-{name}"
    )
    components = [component for root in layout for component in near_optimal._walk_components(root)]
    classes = {getattr(component, "className", None) for component in components}
    ids = {getattr(component, "id", None) for component in components}

    assert "near-optimal-definition-card" in " ".join(value or "" for value in classes)
    assert "near-optimal-settings-card" in " ".join(value or "" for value in classes)
    assert near_optimal.TOP_CRITERION_PANEL_ID in ids
    assert near_optimal.SELECTION_COUNT_ID in ids
    assert near_optimal.NearOptimalConfigurations.button_caption == "Run analysis"


def test_selection_preview_uses_current_criterion():
    objective = SimpleNamespace(optimize="upper")
    run = SimpleNamespace(
        history=[
            SimpleNamespace(
                costs=[0.8],
                status=Status.SUCCESS,
                end_time=1.0,
                additional={"optuna_trial_number": 3},
            ),
            SimpleNamespace(
                costs=[0.9],
                status=Status.SUCCESS,
                end_time=2.0,
                additional={"optuna_trial_number": 8},
            ),
            SimpleNamespace(
                costs=[0.95],
                status=Status.CRASHED,
                end_time=3.0,
                additional={"optuna_trial_number": 9},
            ),
        ],
        get_objective=lambda objective_id: objective if objective_id == 0 else None,
    )

    preview = near_optimal._selection_preview(run, 0, "top_percent", 50)

    assert preview == {
        "selected": 1,
        "eligible": 2,
        "share": 0.5,
        "minimum": 0.9,
        "maximum": 0.9,
    }


def test_study_change_repopulates_every_dependent_control():
    run = SimpleNamespace(
        id="new-study",
        configspace={"depth": object(), "learning_rate": object()},
        history=[
            SimpleNamespace(status=Status.SUCCESS, end_time=10.0),
            SimpleNamespace(status=Status.SUCCESS, end_time=30.0),
        ],
    )
    empty_inputs = {
        "criterion": {"value": None},
        "top_percent": {"value": None},
        "absolute_delta": {"value": None},
        "relative_delta": {"value": None},
        "distance_metric": {"value": None},
        "cluster_count": {"value": None},
        "parameter_names": {"value": ["old_parameter"]},
    }

    populated = near_optimal.NearOptimalConfigurations().load_dependency_inputs(
        run,
        {"run": {"value": "old-study"}},
        empty_inputs,
    )

    assert populated["criterion"]["value"] == "top_percent"
    assert populated["top_percent"]["value"] == 5
    assert len(populated["top_percent"]["options"]) == 5
    assert populated["distance_metric"]["value"] == "importance_weighted"
    assert populated["cluster_count"]["value"] == "auto"
    assert populated["parameter_names"]["value"] == ["depth", "learning_rate"]


def test_process_recovers_safe_defaults_from_an_incomplete_panel(monkeypatch):
    captured = {}

    def build(*args):
        captured["args"] = args
        return {"ok": True}

    monkeypatch.setattr(near_optimal, "build_near_optimal_analysis", build)
    run = SimpleNamespace(
        configspace={"depth": object()},
        history=[SimpleNamespace(status=Status.SUCCESS, end_time=4.0)],
    )

    result = near_optimal.NearOptimalConfigurations.process(
        run,
        {
            "objective_id": None,
            "criterion": None,
            "top_percent": None,
            "cluster_count": None,
            "distance_metric": None,
            "parameter_names": None,
        },
    )

    assert result == {"ok": True}
    assert captured["args"][1:] == (
        0,
        "top_percent",
        5.0,
        "importance_weighted",
        None,
        ["depth"],
    )


def test_page_contains_all_analysis_panels_and_stores():
    payload = _payload()
    page = near_optimal._page_layout(payload)
    ids = {
        component.id
        for component in near_optimal._walk_components(page)
        if isinstance(getattr(component, "id", None), str)
    }
    assert {
        near_optimal.ANALYSIS_DATA_ID,
        near_optimal.DIVERSITY_ID,
        near_optimal.HEATMAP_ID,
        near_optimal.HEATMAP_ORDER_ID,
        near_optimal.CONSTRAINTS_ID,
        near_optimal.IMPORTANCE_ID,
        near_optimal.CLUSTER_DETAIL_ID,
        near_optimal.CLUSTER_COORDINATES_ID,
        near_optimal.CLUSTER_COORDINATES_MODE_ID,
        near_optimal.TRIAL_TABLE_ID,
    }.issubset(ids)
    assert any(
        isinstance(component, dcc.Store) for component in near_optimal._walk_components(page)
    )
    analysis_store = next(
        component
        for component in near_optimal._walk_components(page)
        if getattr(component, "id", None) == near_optimal.ANALYSIS_DATA_ID
    )
    assert isinstance(analysis_store.data, str)
    assert near_optimal._get_analysis_data(analysis_store.data) is payload
    classes = [getattr(component, "className", None) for component in page.children]
    assert "near-optimal-overview-grid" in classes
    summary = next(
        component
        for component in near_optimal._walk_components(page)
        if getattr(component, "className", None) == "analysis-card near-optimal-summary"
    )
    assert len(summary.children) == 6
    assert "Random baseline" not in [card.children[0].children for card in summary.children]
    distance_card = next(
        component
        for component in page.children
        if getattr(component, "className", None)
        == "analysis-card near-optimal-primary-card near-optimal-distance-card"
    )
    distance_ids = {
        getattr(component, "id", None)
        for component in near_optimal._walk_components(distance_card)
    }
    assert near_optimal.CLUSTER_COORDINATES_ID in distance_ids
    assert near_optimal.CLUSTER_COORDINATES_MODE_ID in distance_ids
    assert classes.index("analysis-card near-optimal-regimes") < classes.index(
        "near-optimal-secondary-grid"
    )


def test_analysis_data_cache_uses_stable_tokens_and_can_be_cleared():
    payload = _payload()
    near_optimal.clear_near_optimal_analysis_cache()

    first = near_optimal._cache_analysis_data(payload)
    second = near_optimal._cache_analysis_data(_payload())

    assert first == second
    assert near_optimal._get_analysis_data(first) is not None
    near_optimal.clear_near_optimal_analysis_cache()
    assert near_optimal._get_analysis_data(first) is None


@pytest.mark.parametrize("clusters", [[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 3, 4]])
def test_legacy_heatmap_cache_recovers_missing_silhouettes(clusters):
    payload = _payload()
    matrix = [[0, 0.1, 0.8, 0.9], [0.1, 0, 0.9, 0.8], [0.8, 0.9, 0, 0.1], [0.9, 0.8, 0.1, 0]]
    payload["heatmap"].update(
        matrix=matrix, trial_ids=[3, 8, 9, 10], clusters=clusters, linkage=[], shown=4, total=4
    )
    expected = (
        near_optimal.silhouette_samples(np.asarray(matrix), clusters, metric="precomputed")
        if len(set(clusters)) == 2
        else np.zeros(4)
    )
    payload["heatmap"].pop("silhouette")

    page = near_optimal.NearOptimalConfigurations.load_outputs(None, {}, payload)
    graph = next(
        component
        for component in near_optimal._walk_components(page)
        if getattr(component, "id", None) == near_optimal.HEATMAP_ID
    )
    strip = next(trace for trace in graph.figure.data if trace.type == "bar")
    np.testing.assert_allclose(strip.x, expected)
    assert "silhouette" not in payload["heatmap"]


def test_cluster_profiles_are_compared_in_columns_without_activity_or_graph():
    payload = _payload()
    payload["clusters"]["profiles"].append(
        {
            "cluster": 2,
            "size": 3,
            "objective_median": 0.82,
            "objective_min": 0.78,
            "objective_max": 0.86,
            "parameters": [
                {"parameter": "x", "summary": "0.2 [0.1 – 0.3]", "active": 0.5}
            ],
        }
    )

    detail = near_optimal._cluster_detail(payload)
    components = list(near_optimal._walk_components(detail))
    table = next(component for component in components if isinstance(component, html.Table))
    header = table.children[0].children.children
    rows = table.children[1].children

    assert [cell.children for cell in header] == [
        "Parameter / metric",
        "Cluster 1 (n = 2)",
        "Cluster 2 (n = 3)",
    ]
    assert [[cell.children for cell in row.children] for row in rows] == [
        ["Median objective", "0.85", "0.82"],
        ["Objective range", "0.8 – 0.9", "0.78 – 0.86"],
        ["x", "0.5 [0.25 – 0.75]", "0.2 [0.1 – 0.3]"],
    ]
    assert not any(isinstance(component, dcc.Graph) for component in components)


def test_profile_parameters_are_ordered_by_ascending_cluster_distinctness():
    records = [
        {"configuration": {"overlap": 0.0, "separated": 0.0}},
        {"configuration": {"overlap": 2.0, "separated": 2.0}},
        {"configuration": {"overlap": 0.0, "separated": 4.0}},
        {"configuration": {"overlap": 2.0, "separated": 6.0}},
    ]
    configspace = {"overlap": object(), "separated": object()}

    ordered, scores = near_optimal._profile_parameter_order(
        records,
        [1, 1, 2, 2],
        configspace,
        ["overlap", "separated"],
        {"overlap": 1.0, "separated": 0.1},
    )

    assert ordered == ["overlap", "separated"]
    assert scores == {"overlap": 0.0, "separated": 0.8}


def test_cluster_coordinates_switch_between_trials_and_median_iqr():
    payload = _payload()
    payload["clusters"]["profiles"].append(
        {
            "cluster": 2,
            "size": 2,
            "objective_median": 0.82,
            "objective_min": 0.78,
            "objective_max": 0.86,
            "parameters": [
                {"parameter": "x", "summary": "0.3 [0.2 – 0.4]", "active": 1.0}
            ],
        }
    )
    payload["distributions"]["x"]["clusters"]["2"] = [0.2, 0.4]

    summary = near_optimal.cluster_coordinates_figure(payload, "summary")
    trials = near_optimal.cluster_coordinates_figure(payload, "trials")

    assert len(summary.data) == 4
    np.testing.assert_allclose(summary.data[0].y, [0.75, 0.25])
    np.testing.assert_allclose(summary.data[1].y, [0.5])
    np.testing.assert_allclose(summary.data[2].y, [0.35, 0.25])
    np.testing.assert_allclose(summary.data[3].y, [0.3])
    assert [trace.name for trace in summary.data if trace.showlegend is not False] == [
        "Cluster 1",
        "Cluster 2",
    ]
    assert len(trials.data) == 2
    assert list(trials.data[0].y) == [0.0, None, 1.0, None]
    assert list(trials.data[1].y) == [0.2, None, 0.4, None]


def test_cluster_coordinates_encode_categorical_values_and_inactivity():
    payload = _payload()
    payload["spreads"][0]["kind"] = "categorical"
    payload["distributions"]["x"]["clusters"]["1"] = ["red", "blue", None]

    summary = near_optimal.cluster_coordinates_figure(payload, "summary")
    trials = near_optimal.cluster_coordinates_figure(payload, "trials")

    assert [annotation.text for annotation in trials.layout.annotations] == [
        "red",
        "blue",
        "inactive",
    ]
    assert list(trials.data[0].y) == [0.0, None, 0.5, None, 1.0, None]
    assert list(summary.data[1].y) == [0.0]


def test_analysis_keeps_all_200_trials_and_limits_only_the_heatmap():
    configspace = ConfigurationSpace(seed=0)
    configspace.add([Float("x", (0.0, 1.0)), Float("y", (0.0, 1.0))])
    run = OptunaRun(
        "200 trials", configspace=configspace, objectives=Objective("loss", optimize="lower")
    )
    for trial_id, configuration in enumerate(configspace.sample_configuration(200)):
        run.add(
            float(configuration["x"]) ** 2 + float(configuration["y"]),
            configuration,
            seed=0,
            additional={"optuna_trial_number": trial_id},
        )

    payload = near_optimal.build_near_optimal_analysis(
        run, 0, "top_percent", 100, "importance_weighted", None, ["x", "y"]
    )

    assert payload["summary"]["eligible"] == payload["summary"]["selected"] == 200
    assert {row["trial_id"] for row in payload["trials"]} == set(range(200))
    assert sum(profile["size"] for profile in payload["clusters"]["profiles"]) == 200
    assert payload["importance_error"] is None
    assert payload["distance_metric"] == "importance_weighted"
    assert payload["heatmap"]["shown"] == near_optimal.HEATMAP_LIMIT
    assert payload["heatmap"]["total"] == 200
    assert len(payload["heatmap"]["silhouette"]) == near_optimal.HEATMAP_LIMIT
    assert all(-1 <= value <= 1 for value in payload["heatmap"]["silhouette"])

    figure = near_optimal.heatmap_figure(payload)
    dendrograms = [trace for trace in figure.data if trace.type == "scatter"]
    assert len(figure.data) == 6
    assert len(dendrograms) == 2
    expected = near_optimal.dendrogram(np.asarray(payload["heatmap"]["linkage"]), no_plot=True)
    assert dendrograms[0].connectgaps is False
    top_x = np.asarray(dendrograms[0].x, dtype=float).reshape(-1, 5)
    top_y = np.asarray(dendrograms[0].y, dtype=float).reshape(-1, 5)
    np.testing.assert_allclose(top_x[:, :4], expected["icoord"])
    np.testing.assert_allclose(top_y[:, :4], expected["dcoord"])
    assert np.isnan(top_x[:, 4]).all() and np.isnan(top_y[:, 4]).all()
    np.testing.assert_allclose(
        np.asarray(dendrograms[1].x, dtype=float),
        np.asarray(dendrograms[0].y, dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(dendrograms[1].y, dtype=float),
        np.asarray(dendrograms[0].x, dtype=float),
    )


def test_heatmap_carries_trial_and_cluster_identity():
    figure = near_optimal.heatmap_figure(_payload())
    heatmap, top_strip, _side_strip = [trace for trace in figure.data if trace.type == "heatmap"]
    assert list(heatmap.customdata[0][0]) == [3, 3]
    assert list(top_strip.customdata[0][0]) == [1, 3]


def test_diversity_curve_is_a_right_facing_needle_plot():
    payload = _payload()
    for row in payload["diversity_curve"]:
        row.update(count=5, median=0.5, q25=0.3, q75=0.7)

    figure = near_optimal.diversity_figure(payload)

    assert list(figure.data[0].x) == [0, 0.5, None] * 8
    assert list(figure.data[2].x) == [0.5] * 8
    assert list(figure.data[2].y) == [
        "Top 1%",
        "Top 2%",
        "Top 5%",
        "Top 10%",
        "Top 20%",
        "Top 50%",
        "Top 75%",
        "Top 100%",
    ]
    assert figure.data[2].mode == "markers"
    assert list(figure.layout.xaxis.range) == [0, 1.05]
    assert list(figure.layout.yaxis.categoryarray) == [
        "Top 100%",
        "Top 75%",
        "Top 50%",
        "Top 20%",
        "Top 10%",
        "Top 5%",
        "Top 2%",
        "Top 1%",
    ]
    assert figure.layout.showlegend
    assert {trace.name for trace in figure.data if trace.showlegend is not False} == {
        "Median pairwise distance",
        "Interquartile range (IQR)",
    }
    assert list(figure.data[1].x) == [0.3, 0.7, None] * 8
    assert figure.layout.height == near_optimal.DIVERSITY_GRAPH_HEIGHT


@pytest.mark.parametrize(
    "ordering, expected_order",
    [("hierarchical", [2, 0, 1]), ("cluster", [1, 2, 0]), ("trial", [1, 0, 2])],
)
def test_heatmap_ordering_keeps_matrix_strips_and_hover_identity_aligned(ordering, expected_order):
    payload = _payload()
    matrix = np.asarray([[0, 0.2, 0.8], [0.2, 0, 0.6], [0.8, 0.6, 0]])
    payload["heatmap"].update(
        matrix=matrix.tolist(),
        trial_ids=[8, 3, 12],
        clusters=[2, 1, 1],
        silhouette=[0.2, -0.4, 0.7],
        linkage=[[0, 1, 0.2, 2], [2, 3, 0.7, 3]],
        shown=3,
        total=3,
    )

    figure = near_optimal.heatmap_figure(payload, ordering)
    heatmap, top_strip, side_strip = [trace for trace in figure.data if trace.type == "heatmap"]
    np.testing.assert_allclose(heatmap.z, matrix[np.ix_(expected_order, expected_order)])
    expected_trials = [payload["heatmap"]["trial_ids"][index] for index in expected_order]
    expected_clusters = [payload["heatmap"]["clusters"][index] for index in expected_order]
    assert [cell[1] for cell in top_strip.customdata[0]] == expected_trials
    assert [row[0][0] for row in side_strip.customdata] == expected_clusters
    assert list(figure.layout.xaxis.ticktext) == [f"trial_{trial:03d}" for trial in expected_trials]
    assert list(figure.layout.yaxis.range) == [30, 0]
    assert heatmap.zmin == 0 and heatmap.zmax == 1
    assert heatmap.showscale is False
    silhouette = next(trace for trace in figure.data if trace.type == "bar")
    assert list(silhouette.x) == [
        payload["heatmap"]["silhouette"][index] for index in expected_order
    ]
    assert list(silhouette.y) == list(heatmap.y)
    assert list(silhouette.marker.color) == [
        near_optimal._cluster_color(expected_clusters[index])
        for index in range(len(expected_clusters))
    ]
    assert list(figure.layout.xaxis4.range) == [-1, 1]
    assert figure.layout.xaxis4.zeroline
    dendrograms = [trace for trace in figure.data if trace.type == "scatter"]
    assert len(dendrograms) == (2 if ordering == "hierarchical" else 0)


@pytest.mark.parametrize("count", [0, 1])
def test_heatmap_handles_sets_without_pairwise_distances(count):
    payload = _payload()
    payload["heatmap"].update(
        matrix=[[0]] if count else [],
        trial_ids=[3] if count else [],
        clusters=[1] if count else [],
        silhouette=[0.0] if count else [],
        linkage=[],
        shown=count,
        total=count,
    )
    figure = near_optimal.heatmap_figure(payload)
    assert figure.layout.height == near_optimal.PRIMARY_GRAPH_HEIGHT
    if count:
        assert len([trace for trace in figure.data if trace.type == "heatmap"]) == 3
    else:
        assert figure.layout.annotations[0].text == "No pairwise distances available."


def test_constraints_summary_shows_normalized_elite_interval_and_coverage():
    summary = near_optimal._constraints_table(_payload())
    components = list(near_optimal._walk_components(summary))
    classes = [getattr(component, "className", "") or "" for component in components]
    rows = [
        component
        for component in components
        if isinstance(getattr(component, "id", None), dict)
        and component.id.get("type") == "near-optimal-constraint-row"
    ]
    tracks = [
        component
        for component in components
        if getattr(component, "className", None) == "near-optimal-interval__track"
    ]

    assert summary.id == near_optimal.CONSTRAINTS_ID
    assert len(rows) == 1
    assert rows[0].id["parameter"] == "x"
    assert [track.style for track in tracks] == [
        {
            "--interval-start": "18.00%",
            "--interval-width": "31.00%",
            "--interval-median": "33.00%",
        }
    ]
    assert "near-optimal-interval__range" in classes
    assert "near-optimal-interval__median" in classes
    assert "near-optimal-constraints__coverage" in classes


def test_importance_plot_centers_coverage_quadrants_and_keeps_points_clear():
    payload = _payload()
    payload["parameters"].append("y")
    payload["importance"]["y"] = 0.2
    payload["spreads"].append(
        {
            "parameter": "y",
            "near_optimal_spread": 0.4,
            "overall_spread": 0.5,
            "relative_spread": 0.8,
            "near_optimal_coverage": 0.79,
            "near_optimal_active": 1.0,
            "overall_active": 1.0,
            "kind": "numeric",
        }
    )

    figure = near_optimal.importance_figure(payload)

    assert figure.layout.xaxis.type == "linear"
    assert list(figure.layout.xaxis.range) == [-0.03, 1.03]
    assert list(figure.layout.xaxis.ticktext) == ["0%", "25%", "50%", "75%", "100%"]
    assert list(figure.data[0].x) == [0.31, 0.79]
    assert figure.layout.yaxis.type == "log"
    assert figure.layout.yaxis.dtick == 1
    log_y_center = sum(figure.layout.yaxis.range) / 2
    assert abs(10**log_y_center - 0.6) < 1e-12
    log_y_radius = (figure.layout.yaxis.range[1] - figure.layout.yaxis.range[0]) / 2
    assert all(
        abs(math.log10(value) - log_y_center) / log_y_radius < 0.75 for value in figure.data[0].y
    )
    assert len(figure.layout.shapes) == 2
    annotation_texts = {annotation.text for annotation in figure.layout.annotations}
    assert any("critical setting" in text for text in annotation_texts)
    assert any("alternative regimes" in text for text in annotation_texts)
    assert any("Higher than median importance" in text for text in annotation_texts)
    assert any("Lower than median importance" in text for text in annotation_texts)
    assert any("flexible" in text for text in annotation_texts)
    assert figure.layout.yaxis.title.text == "Hyperparameter importance"
    assert list(figure.data[0].marker.color) == ["#636EFA", "#EF553B"]
    assert any(shape.x0 == shape.x1 == 0.5 for shape in figure.layout.shapes)
    assert any(shape.y0 == shape.y1 == 0.6 for shape in figure.layout.shapes)
    assert figure.data[0].mode == "markers"
    assert list(figure.data[0].text) == ["", ""]
    assert figure.layout.xaxis.title.text == "Search-space coverage"
    selected_figure = near_optimal.importance_figure(payload, "y")
    assert selected_figure.data[0].mode == "markers+text"
    assert list(selected_figure.data[0].text) == ["", "y"]
    assert list(figure.data[0].customdata[0][:5]) == ["x", 1.0, 0.5, 0.5, 1.0]


def test_importance_plot_keeps_relative_spread_in_hover_data():
    payload = _payload()
    payload["spreads"][0]["relative_spread"] = 1.11

    figure = near_optimal.importance_figure(payload)

    assert figure.data[0].x[0] == 0.31
    assert figure.data[0].customdata[0][4] == 1.11
    assert list(figure.layout.xaxis.range) == [-0.03, 1.03]
