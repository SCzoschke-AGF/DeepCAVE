"""Exact pairs, nearest-neighbor isolation, empirical bins and threshold reuse."""

from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    EqualsCondition,
    UniformFloatHyperparameter,
)
from scipy.spatial.distance import pdist, squareform

from deepcave.evaluators import performance_coverage as coverage
from deepcave.evaluators.rashomon import (
    cached_pairwise_configuration_distances,
    pairwise_configuration_distances,
)
from deepcave.runs.status import Status


def make_run(configs=None, costs=None):
    configs = configs or [{"x": 0.0}, {"x": 0.0}, {"x": 0.4}, {"x": 1.0}]
    costs = costs or [[0.0, 4.0], [0.02, 3.0], [0.1, 2.0], [0.5, 1.0]]
    objectives = [SimpleNamespace(name="AP"), SimpleNamespace(name="Loss")]
    return SimpleNamespace(
        id="performance-study",
        name="Performance study",
        latest_change=1,
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}),
        history=[
            SimpleNamespace(
                config_id=i,
                costs=value,
                status=Status.SUCCESS,
                additional={"optuna_trial_number": i},
            )
            for i, value in enumerate(costs)
        ],
        get_config=lambda index: configs[index],
        get_objective=lambda index: objectives[index],
        get_objective_ids=lambda: [0, 1],
        get_highest_budget=lambda: 1,
    )


def analyze(run, tmp_path):
    prepared = coverage.prepare_performance_trials(run, 0)
    distances = pairwise_configuration_distances(
        prepared["matrix"], run.configspace, weights={"x": 1.0}
    )
    result = coverage.analyze_performance_distances(
        distances, prepared["objectives"], prepared["trial_ids"], tmp_path / "differences.npy"
    )
    return result, prepared, distances


def test_all_pairs_and_zero_distance_duplicates_are_retained(tmp_path):
    result, prepared, distances = analyze(make_run(), tmp_path)
    assert result["trials"] == 4 and result["pairs"] == 6
    assert len(prepared["trial_ids"]) == 4
    assert distances[0] == 0
    assert result["plot"]["differences"][0] == 0.02
    assert sum(group["count"] for group in result["bins"]) == 6
    assert sum(group["count"] for group in result["nearest_bins"]) == 4


def test_nearest_neighbors_match_exhaustive_reference_with_deterministic_ties(tmp_path):
    run = make_run(configs=[{"x": 0.0}, {"x": 0.5}, {"x": 1.0}, {"x": 0.5}])
    run.history[0].additional["optuna_trial_number"] = 20
    result, prepared, distances = analyze(run, tmp_path)
    matrix = squareform(distances)
    np.fill_diagonal(matrix, np.inf)
    expected = np.argmin(matrix, axis=1)
    assert result["nearest"]["neighbor_indices"] == expected.tolist()
    assert result["nearest"]["distances"] == pytest.approx(matrix.min(axis=1))
    assert result["nearest"]["differences"] == pytest.approx(
        np.abs(prepared["objectives"] - prepared["objectives"][expected])
    )
    # Trial 2 is equally far from duplicate trials 1 and 3; lowest ID wins.
    row = result["nearest"]["trial_ids"].index(2)
    assert result["nearest"]["neighbor_ids"][row] == 1


def test_all_identical_configurations_have_one_group_and_no_self_neighbors(tmp_path):
    run = make_run(configs=[{"x": 0.0}] * 4)
    result, _, _ = analyze(run, tmp_path)
    assert len(result["bins"]) == len(result["nearest_bins"]) == 1
    assert result["nearest"]["distances"] == [0.0] * 4
    assert result["nearest"]["neighbor_ids"] == [1, 0, 0, 0]


def test_missing_nonfinite_failed_and_incompatible_trials_are_excluded(tmp_path):
    run = make_run(
        configs=[
            {"x": 0.0},
            {"x": 0.2},
            {"x": 0.4},
            {"x": 0.6},
            {"x": 0.8},
            {"x": 2.0},
            {"x": 1.0},
        ],
        costs=[[0.0], [None], [np.nan], [np.inf], [0.8], [0.7], [1.0]],
    )
    run.history[4].status = Status.FAILED
    result, prepared, _ = analyze(run, tmp_path)
    assert prepared["trial_ids"] == [0, 6]
    assert prepared["excluded"] == 5
    assert result["pairs"] == 1


def test_existing_log_categorical_and_conditional_normalization_is_preserved():
    run = make_run(
        configs=[{"model": "a", "rate": 1e-4}, {"model": "a", "rate": 1e-2}, {"model": "b"}],
        costs=[[0], [1], [2]],
    )
    space = ConfigurationSpace()
    model = CategoricalHyperparameter("model", ["a", "b"])
    rate = UniformFloatHyperparameter("rate", 1e-4, 1.0, log=True)
    space.add([model, rate, EqualsCondition(rate, model, "a")])
    run.configspace = space
    prepared = coverage.prepare_performance_trials(run, 0)
    distances = pairwise_configuration_distances(
        prepared["matrix"], space, weights={"model": 1, "rate": 3}
    )
    assert distances == pytest.approx([np.sqrt(3 * 0.5**2 / 4), 1.0, 1.0])


@pytest.mark.parametrize("values", [[0] * 20, [0] * 19 + [1], [0, 0, 0.1, 0.1, 0.5, 1]])
def test_quantile_groups_merge_ties_and_cover_every_observation(values):
    array = np.asarray(values, dtype=float)
    groups, labels = coverage.quantile_groups(array, 3)
    assert len(groups) == len(values) and len(labels) <= 3
    for value in np.unique(array):
        assert len(np.unique(groups[array == value])) == 1
    assert sum(np.sum(groups == label) for label in labels) == len(values)


def test_pair_statistics_and_threshold_updates_match_exact_observations(tmp_path):
    run = make_run()
    result, prepared, distances = analyze(run, tmp_path)
    groups, labels = coverage.quantile_groups(distances, 10)
    differences = np.asarray(result["plot"]["differences"])
    for label, stats in zip(labels, result["bins"]):
        observed = differences[groups == label]
        assert stats["mean"] == pytest.approx(np.mean(observed))
        assert [stats[key] for key in ("median", "p90", "p95")] == pytest.approx(
            np.quantile(observed, [0.5, 0.9, 0.95])
        )
    for threshold in (0.0, 0.02, 0.1, 100.0):
        updated = coverage.threshold_statistics(result, threshold, tmp_path / "differences.npy")
        for label, stats in zip(labels, updated["bins"]):
            assert stats["relevant_fraction"] == np.mean(differences[groups == label] > threshold)
        nn = updated["nearest"]
        nn_groups, nn_labels = coverage.quantile_groups(np.asarray(nn["distances"]), 3)
        for label, stats in zip(nn_labels, updated["nearest_bins"]):
            observed = np.asarray(nn["differences"])[nn_groups == label]
            assert stats["mean"] == pytest.approx(np.mean(observed))
            assert stats["relevant_fraction"] == np.mean(observed > threshold)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, True, None])
def test_invalid_threshold_is_rejected(tmp_path, threshold):
    result, _, _ = analyze(make_run(), tmp_path)
    with pytest.raises(ValueError, match="Relevant difference"):
        coverage.threshold_statistics(result, threshold, tmp_path / "differences.npy")


def test_density_counts_include_all_pairs_and_zero_differences(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage, "SCATTER_LIMIT", 2)
    run = make_run(costs=[[0.0]] * 4)
    result, _, _ = analyze(run, tmp_path)
    assert result["plot"]["kind"] == "density"
    assert np.sum(result["plot"]["counts"]) == 6
    assert all(stats["median"] == 0 for stats in result["bins"])
    assert min(result["plot"]["y_edges"]) == 0


def test_blocked_rashomon_distances_match_compiled_reference():
    space = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    matrix = np.random.default_rng(5).uniform(size=(2051, 2))
    actual = pairwise_configuration_distances(matrix, space, weights={"x": 1, "y": 3})
    expected = pdist(matrix * np.sqrt([1, 3])) / 2
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_shared_distance_cache_checks_actual_inputs_in_addition_to_caller_key():
    space = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    matrix = np.asarray([[0, 0], [1, 0]], dtype=float)
    first = cached_pairwise_configuration_distances(
        ("same-caller-key",), matrix, space, weights={"x": 1, "y": 0}
    )
    changed = cached_pairwise_configuration_distances(
        ("same-caller-key",), matrix, space, weights={"x": 0, "y": 1}
    )
    assert first.tolist() == [1.0] and changed.tolist() == [0.0]


def test_source_key_changes_with_objective_eligibility_and_configuration():
    run = make_run()
    original = coverage.performance_source_key(run, 0)
    assert coverage.performance_source_key(run, 1) != original
    run.history[0].costs[0] = 0.3
    assert coverage.performance_source_key(run, 0) != original
    run.history[0].costs[0] = 0.0
    run.history[0].status = Status.FAILED
    assert coverage.performance_source_key(run, 0) != original
    run.history[0].status = Status.SUCCESS
    run.get_config = lambda _: {"x": 0.2}
    assert coverage.performance_source_key(run, 0) != original


def test_fewer_than_two_eligible_trials_is_explained(tmp_path):
    with pytest.raises(ValueError, match="At least two"):
        coverage.analyze_performance_distances(
            np.asarray([]), np.asarray([0.0]), [0], tmp_path / "differences.npy"
        )
