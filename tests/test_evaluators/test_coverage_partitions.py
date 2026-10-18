"""Validate experimental partitions, discrete atoms, held-out calibration, and forecasts."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace

from deepcave.evaluators import coverage
from examples.api.coverage_partitions import (
    Node,
    PartitionEnsemble,
    RandomPartition,
    aggregate,
    coordinates,
    forecast,
)
from examples.api.validate_coverage_mmd import draw, make_spaces
from examples.api.validate_coverage_partitions import joint_scores, mmd_comparison


def test_discrete_partition_cannot_create_more_cells_than_configurations():
    points = np.array(list(product([0.0, 1.0], repeat=2)))
    weights = np.array([49, 49, 1, 1])
    axes = [SimpleNamespace(kind="categorical")] * 2
    tree = RandomPartition(points, weights, axes, 80, 17)
    assert tree.cells == 4
    assert len(set(tree.assign(points))) == 4
    assert sorted(tree.training_mass) == pytest.approx([0.01, 0.01, 0.49, 0.49])
    assert (
        tree.assign(points).tolist()
        == RandomPartition(points, weights, axes, 80, 17).assign(points).tolist()
    )


def test_numeric_inactivity_is_a_separate_state_and_category_splits_are_sets():
    node = Node(threshold=0.5, missing_left=False)
    values = np.array([-1.0, 0.0, 0.5, 1.0])
    assert node.mask(values).tolist() == [False, True, True, False]
    node.missing_left = True
    assert node.mask(values).tolist() == [True, True, True, False]
    node.categories = (-1.0, 7.0)
    assert node.mask(np.array([-1.0, 0.0, 2.0, 7.0])).tolist() == [True, False, False, True]


def test_equiprobable_cells_tv_identity_at_k_equals_n_and_counterexample_elsewhere():
    space = ConfigurationSpace(space={"c": ["a", "b", "c", "d"]})
    complete = coverage.normalize_configurations(space, [{"c": c} for c in "abcd"])
    forest = PartitionEnsemble(complete, complete, 4, 5, 3)
    observed = coverage.normalize_configurations(space, [{"c": c} for c in "aaab"])
    measured = forest.measure(coordinates(observed)[0], 1, 4)
    assert measured["tv"] == pytest.approx(np.full((1, 5), 0.5))
    assert measured["empty_mass"] == pytest.approx(measured["tv"])
    assert measured["empty_fraction"] == pytest.approx(measured["tv"])
    diagnostic = forest.diagnostics(4)
    assert diagnostic["expected_empty_fraction"] == pytest.approx((3 / 4) ** 4)
    assert diagnostic["expected_empty_mass"] == pytest.approx((3 / 4) ** 4)
    # At N != K an overrepresented occupied cell can break the identity.
    uneven = coverage.normalize_configurations(space, [{"c": c} for c in "aaaaabcd"])
    eight = forest.measure(coordinates(uneven)[0], 1, 8)
    assert np.all(eight["tv"] > eight["empty_mass"])


def test_conditional_balancing_uses_legal_configs_and_unseen_values_are_assignable():
    definition = next(d for d in make_spaces([6]) if d["name"] == "conditional_6d")
    space = definition["space"]
    training, masses, held_out = [
        draw(space, n, seed) for n, seed in ((512, 1), (2048, 2), (80, 3))
    ]
    forest = PartitionEnsemble(training, masses, 20, 8, 4)
    assert (coordinates(training)[0] == -1).any()
    for values in held_out.values:
        Configuration(space, values=values)
    for tree, mass in zip(forest.partitions, forest.masses):
        labels = tree.assign(coordinates(held_out)[0])
        assert tree.cells == 20
        assert labels.min() >= 0 and labels.max() < tree.cells
        assert mass.sum() == pytest.approx(1)
        assert (mass > 0).all()
    assert forest.diagnostics(80)["median_mass_cv"] < 0.5


def test_statistics_fit_only_on_independent_fit_designs_and_calibrate_selection():
    empty = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.3], [0.4, 0.2], [0.9, 0.9]])
    measures = {"empty_mass": empty, "tv": empty + 0.1}
    scores = aggregate(measures, 3)
    joint = joint_scores(scores, 3)
    altered = {key: values.copy() for key, values in measures.items()}
    for value in altered.values():
        value[-1] = 100
    updated = aggregate(altered, 3)
    assert np.allclose(scores["max_z_empty_mass"][:-1], updated["max_z_empty_mass"][:-1])
    assert np.allclose(joint[:-1], joint_scores(updated, 3)[:-1])
    assert joint[-1] > joint[3]


def test_forecast_reuses_fixed_trees_and_same_future_configs_and_matches_direct_counts():
    space = ConfigurationSpace(space={"x": (0.0, 1.0), "c": ["a", "b"]})
    training, masses = draw(space, 256, 4), draw(space, 512, 5)
    ensembles = {k: PartitionEnsemble(training, masses, k, 4, 6) for k in (4, 8)}
    observed, future = draw(space, 4, 7), draw(space, 8, 8)
    predictions = forecast(ensembles, observed, future, 2, 4, [0, 1, 2, 3, 4])
    for k, ensemble in ensembles.items():
        expected = np.zeros((2, 5))
        for simulation in range(2):
            for step in range(5):
                points = np.r_[
                    coordinates(observed)[0],
                    coordinates(future)[0][simulation * 4 : simulation * 4 + step],
                ]
                expected[simulation, step] = ensemble.measure(points, 1, len(points))[
                    "empty_mass"
                ].mean()
        prediction = predictions[str(k)]
        assert prediction["initial_mean_empty_mass"] == pytest.approx(expected[0, 0])
        assert np.all(np.diff(expected, axis=1) <= 1e-12)
        for step, budget in enumerate(prediction["budgets"]):
            assert budget["mean_empty_mass"] == pytest.approx(expected[:, step].mean())
            assert budget["successes"] == np.sum(expected[:, step] <= 0.8 * expected[:, 0])


def test_mmd_comparison_refuses_different_replicate_counts(tmp_path):
    import json

    prior = dict(seed=1, dimensions=[6], calibration_designs=199, replicates=100, groups=[])
    path = tmp_path / "results.json"
    path.write_text(json.dumps(prior))
    assert mmd_comparison({**prior, "replicates": 10}, path) == {}


def test_eighty_fixed_cells_have_the_claimed_random_occupancy_expectation():
    occupied = 80 * (1 - (79 / 80) ** 80)
    assert occupied == pytest.approx(50.7549, abs=0.001)
