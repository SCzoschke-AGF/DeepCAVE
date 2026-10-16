import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from plotly.utils import PlotlyJSONEncoder

from deepcave import config
from deepcave.evaluators.parameter_dependencies import (
    conditional_effect,
    encode_parameter,
    orient_pair,
    pair_summary,
    sampling_dependency,
)
from deepcave.plugins.hyperparameter import parameter_dependencies as analysis
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status
from deepcave.runs.trial import Trial


def _pair(objectives, counts=None):
    values = np.asarray(objectives, dtype=float)
    counts = np.full(values.shape, 10) if counts is None else np.asarray(counts)
    codes_a, codes_b, performance = [], [], []
    for a, b in np.ndindex(values.shape):
        codes_a.extend([a] * counts[a, b])
        codes_b.extend([b] * counts[a, b])
        performance.extend([values[a, b]] * counts[a, b])
    return pair_summary(
        np.asarray(codes_a),
        np.asarray(codes_b),
        [str(i) for i in range(values.shape[0])],
        [str(i) for i in range(values.shape[1])],
        np.asarray(performance),
    )


def _run():
    space = ConfigurationSpace()
    space.add(
        [
            UniformFloatHyperparameter("x", 0, 1),
            CategoricalHyperparameter("category", ["left", "right", "never sampled"]),
            Constant("fixed", 1),
        ]
    )
    configurations = {
        0: {"x": 0.1, "category": "left", "fixed": 1},
        1: {"x": 0.9, "category": "right", "fixed": 1},
        2: {"x": 0.9, "category": "left", "fixed": 1},
        3: {"x": None, "category": "right", "fixed": 1},
    }
    objectives = [Objective("Loss", optimize="lower"), Objective("Accuracy", optimize="upper")]
    return SimpleNamespace(
        prefix="Optuna",
        configspace=space,
        configs=configurations,
        history=[
            Trial(0, 0, -1, [1.0, 0.8], 0, 0, 1, Status.SUCCESS, {}),
            Trial(1, 0, -1, [2.0, None], 0, 1, 2, Status.SUCCESS, {}),
            Trial(2, 0, -1, [100.0, 0.9], 0, 2, 3, Status.CRASHED, {}),
            Trial(3, 0, -1, [3.0, float("inf")], 0, 3, 4, Status.SUCCESS, {}),
            Trial(0, 0, -1, [1.5, 0.7], 0, 4, 5, Status.SUCCESS, {}),
        ],
        get_objective=lambda i: objectives[i],
        get_objectives=lambda: objectives,
        get_objective_id=lambda objective: objectives.index(objective),
        empty=lambda: False,
    )


def _inputs():
    return {
        "objective_id": 0,
        "n_bins": 4,
        "min_support": 2,
        "target": "x",
        "conditioning": "category",
    }


def test_normalized_mi_independence_symmetry_and_deterministic_dependency():
    independent = sampling_dependency(np.array([[10, 20], [20, 40]]))
    dependent = sampling_dependency(np.diag([10, 30]))
    assert independent["score"] == pytest.approx(0, abs=1e-12)
    assert dependent["score"] == pytest.approx(1)
    counts = np.array([[3, 5, 2], [8, 1, 4]])
    assert sampling_dependency(counts)["score"] == pytest.approx(
        sampling_dependency(counts.T)["score"]
    )
    assert 0 <= sampling_dependency(counts)["score"] <= 1


def test_expected_counts_and_relative_deviations_explain_same_joint_table():
    result = sampling_dependency(np.array([[30, 0, 0], [10, 60, 0]]))
    np.testing.assert_allclose(result["expected"], [[12, 18, 0], [28, 42, 0]])
    assert result["deviation"][0] == [1.5, -1.0, None]
    assert result["deviation"][1][0] == pytest.approx(-18 / 28)


@pytest.mark.parametrize(
    "counts", [np.zeros((2, 2)), np.array([[1, 0], [0, 0]]), np.array([[3, 2]])]
)
def test_empty_single_observation_and_constant_marginal_are_unavailable(counts):
    assert sampling_dependency(counts)["score"] is None


@pytest.mark.parametrize(
    "hyperparameter,values",
    [
        (UniformFloatHyperparameter("x", 0, 100), list(range(100))),
        (UniformFloatHyperparameter("x", 1e-6, 1, log=True), np.geomspace(1e-6, 1, 100)),
        (UniformIntegerHyperparameter("x", 0, 99), list(range(100))),
        (CategoricalHyperparameter("x", ["a", "b", "c", "d"]), list("abcd") * 25),
        (OrdinalHyperparameter("x", ["a", "b", "c", "d"]), list("abcd") * 25),
    ],
)
def test_all_parameter_types_share_normalized_dependency_concept(hyperparameter, values):
    codes, labels = encode_parameter(pd.Series(values), hyperparameter, 4)
    pair = pair_summary(codes, codes, labels, labels, np.ones(len(codes)))
    assert len(labels) == 4
    assert pair["dependency"]["score"] == pytest.approx(1)


def test_numeric_ties_low_cardinality_and_inactive_values():
    hp = UniformFloatHyperparameter("x", 0, 1)
    codes, labels = encode_parameter(pd.Series([0.1, 0.1, 0.9, None, np.inf]), hp, 4)
    assert labels == ["0.1", "0.9"]
    assert list(codes) == [0, 0, 1, -1, -1]
    codes, labels = encode_parameter(pd.Series([0] * 90 + list(np.linspace(0.1, 1, 10))), hp, 4)
    assert (codes >= 0).all()
    assert len(labels) <= 4


def test_pure_conditioning_offset_does_not_count_as_interaction():
    pair = _pair([[2, 12], [0, 10], [3, 13]])
    effect = conditional_effect(pair, "lower", 5)
    assert effect["interaction_score"] == pytest.approx(0)
    assert effect["interaction"] == "Low"
    assert effect["stability"] == "High"
    assert effect["diagnosis"] == "Robust"


def test_opposing_conditional_effects_have_low_dependency_and_strong_interaction():
    effect = conditional_effect(_pair([[0, 10], [5, 5], [10, 0]]), "lower", 5)
    assert effect["sampling"] == "Low"
    assert effect["interaction"] == "Strong"
    assert effect["stability"] == "Low"
    assert effect["diagnosis"] == "Interaction"


def test_high_dependency_with_missing_overlap_is_potentially_confounded():
    pair = _pair([[0, 10], [10, 0]], [[10, 0], [0, 10]])
    effect = conditional_effect(pair, "lower", 5)
    assert effect["sampling"] == "High"
    assert effect["diagnosis"] == "Potentially confounded"
    assert effect["interaction_score"] is None
    assert effect["stability"] == "Not estimable"


def test_high_dependency_alone_does_not_establish_confounding():
    pair = _pair([[0, 0], [1, 1]], [[100, 5], [5, 100]])
    effect = conditional_effect(pair, "lower", 5)
    assert effect["sampling"] == "High"
    assert effect["diagnosis"] == "Stable within observed overlap"
    assert effect["stability"] == "High"


def test_sparse_observed_region_cannot_be_discarded_to_claim_robustness():
    effect = conditional_effect(
        _pair([[1, 1, 1], [0, 0, 0], [2, 2, 2]], [[10, 10, 1]] * 3), "lower", 5
    )
    assert effect["regions"] == 3
    assert effect["overlap"] == 0
    assert effect["diagnosis"] == "Insufficient support"


def test_shared_bins_must_cover_the_marginal_optimum():
    pair = _pair([[0, 0], [2, 2], [3, 3]], [[20, 1], [10, 10], [10, 10]])
    effect = conditional_effect(pair, "lower", 5)
    assert effect["overlap"] == pytest.approx(2 / 3)
    assert effect["interaction_score"] is None
    assert effect["diagnosis"] != "Robust"


def test_objective_direction_changes_optimum_stability():
    pair = _pair([[0, 2], [2, 0], [5, 5]])
    assert conditional_effect(pair, "lower", 5)["stability"] == "Low"
    assert conditional_effect(pair, "upper", 5)["stability"] == "High"


def test_regional_optimum_outside_overlap_cannot_appear_stable():
    # The marginal optimum and two thirds of the target bins overlap, but the
    # second conditioning region has its best supported value outside that overlap.
    pair = _pair([[0, 0], [2, 2], [5, -1]], [[100, 10], [100, 10], [1, 10]])
    effect = conditional_effect(pair, "lower", 5)
    assert effect["overlap"] == pytest.approx(2 / 3)
    assert effect["interaction_score"] is None
    assert effect["diagnosis"] != "Robust"


def test_flat_profiles_have_no_identifiable_sweet_spot():
    effect = conditional_effect(_pair([[1, 1], [1, 1]]), "lower", 5)
    assert effect["interaction"] == "Low"
    assert effect["stability"] == "Not identifiable"
    assert effect["diagnosis"] == "Inconclusive"


def test_reversing_pair_preserves_dependency_and_transposes_every_drilldown():
    pair = _pair([[0, 2], [2, 0], [5, 5]])
    reverse = orient_pair(pair, True)
    assert reverse["dependency"]["score"] == pair["dependency"]["score"]
    assert reverse["labels_a"] == pair["labels_b"]
    np.testing.assert_equal(reverse["counts"], np.asarray(pair["counts"]).T)
    assert orient_pair(reverse, True) == pair


def test_sampling_includes_failures_repeats_and_missing_objectives_and_is_objective_independent():
    run = _run()
    first = analysis.ParameterDependencies.process(run, _inputs())
    second = analysis.ParameterDependencies.process(run, dict(_inputs(), objective_id=1))
    a = analysis.selected_pair(first, "x", "category")
    b = analysis.selected_pair(second, "x", "category")
    assert first["n_trials"] == 5
    assert a["dependency"] == b["dependency"]
    assert a["counts"] == [[2, 0, 0], [1, 1, 0]]
    assert a["excluded"] == 1
    assert np.sum(a["performance_counts"]) == 3
    assert np.sum(b["performance_counts"]) == 2
    assert np.sum(analysis.selected_pair(first, "fixed", "category")["counts"]) == 5
    json.dumps(first, allow_nan=False)


def test_matrix_and_sorted_ranking_clicks_select_same_pair_and_diagonal_is_ignored():
    assert analysis.selection_updates(
        {"matrix_click": {"points": [{"x": "category", "y": "x"}]}}
    ) == {"target": {"value": "x"}, "conditioning": {"value": "category"}}
    assert analysis.selection_updates({"ranking_cell": {"row": 999, "row_id": "category"}}) == {
        "conditioning": {"value": "category"}
    }
    assert analysis.selection_updates({"matrix_click": {"points": [{"x": "x", "y": "x"}]}}) == {}


def test_dependency_controls_retain_target_and_choose_distinct_valid_conditioner():
    plugin = analysis.ParameterDependencies()
    inputs = plugin.load_inputs()
    inputs["target"]["value"] = "x"
    inputs["conditioning"]["value"] = "x"
    result = plugin.load_dependency_inputs(_run(), {}, inputs)
    assert result["target"]["value"] == "x"
    assert result["conditioning"]["value"] != "x"
    assert result["objective_id"]["value"] == 0


def test_registered_output_order_matches_rendered_content_and_display_filters_reuse_cache():
    plugin = analysis.ParameterDependencies()
    assert plugin.outputs == [
        ("matrix", "figure"),
        ("pair_title", "children"),
        ("joint", "figure"),
        ("deviation", "figure"),
        ("ranking", "data"),
        ("diagnostics", "children"),
        ("conditional", "figure"),
        ("support", "figure"),
    ]
    run, inputs = _run(), _inputs()
    outputs = plugin.process(run, inputs)
    rendered = plugin.load_outputs(run, inputs, outputs)
    assert len(rendered) == len(plugin.outputs)
    assert rendered[1] == "x × category"
    assert {row["id"] for row in rendered[4]} == {"category", "fixed"}
    # Rendering must be serializable even for missing objectives and constant parameters.
    json.dumps(rendered, cls=PlotlyJSONEncoder)
    nested = {key: {"value": value} for key, value in inputs.items()}
    changed = {
        **nested,
        "target": {"value": "fixed"},
        "conditioning": {"value": "x"},
        "min_support": {"value": 20},
    }
    assert plugin._dict_as_key(nested, remove_filters=True) == plugin._dict_as_key(
        changed, remove_filters=True
    )


def test_unsupported_cells_are_gaps_and_zero_expected_cells_are_unavailable():
    pair = _pair([[0, 1], [2, 3], [4, 5]], [[10, 10], [1, 10], [10, 10]])
    effect = conditional_effect(pair, "lower", 5)
    figure = analysis.conditional_figure(pair, effect, "x", "b", "Loss")
    assert list(figure.data[0].y) == [0, None, 4]
    assert figure.data[0].connectgaps is False
    run = _run()
    pair = analysis.selected_pair(
        analysis.ParameterDependencies.process(run, _inputs()), "x", "category"
    )
    enrichment = analysis.joint_figure(pair, "x", "category", "deviation")
    assert np.isnan(enrichment.data[0].z[2]).all()
    assert list(enrichment.data[0].z[1])[0] == -1


def test_entirely_inactive_parameter_and_no_performance_are_handled():
    run = _run()
    for values in run.configs.values():
        values["x"] = None
    outputs = analysis.ParameterDependencies.process(run, _inputs())
    pair = analysis.selected_pair(outputs, "x", "category")
    assert pair["dependency"]["score"] is None
    assert conditional_effect(pair, "lower", 5)["performance_n"] == 0
    reverse = analysis.selected_pair(outputs, "category", "x")
    assert conditional_effect(reverse, "lower", 5)["regions"] == 0
    rendered = analysis.ParameterDependencies.load_outputs(run, _inputs(), outputs)
    assert rendered[6].layout.annotations[0].text.startswith("No successful trials")


def test_plugin_registration_and_run_compatibility():
    ids = [plugin.id for plugin in config.PLUGINS["Configuration Analyses"]]
    assert analysis.ParameterDependencies.id in ids
    assert analysis.ParameterDependencies.check_run_compatibility(_run())
    assert not analysis.ParameterDependencies.check_run_compatibility(
        SimpleNamespace(prefix="Optuna", empty=lambda: False, configspace={"x": None})
    )
    group = SimpleNamespace(
        prefix="group", runs=[_run()], empty=lambda: False, configspace={"x": None, "b": None}
    )
    assert analysis.ParameterDependencies.check_run_compatibility(group)
