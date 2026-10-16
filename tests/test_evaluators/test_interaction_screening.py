"""Check the interaction definition, quadrature diagnostics and selected exact pairs."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace

from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.evaluators.fanova import fANOVA
from deepcave.evaluators.interaction_screening import (
    _grid,
    estimate_pair,
    screen_interactions,
)


def _analytic_forest(function, variance=1 / 9, trees=2):
    def marginal(tree, values):
        mean = function(tree, values)
        return SimpleNamespace(mean=lambda: mean, sum_of_weights=lambda: 1.0)

    return SimpleNamespace(
        n_params=10,
        cs=ConfigurationSpace(space={f"x{i}": (0.0, 1.0) for i in range(10)}),
        bounds=[(0.0, 1.0)] * 10,
        trees_total_variance=[variance] * trees,
        _model=SimpleNamespace(marginal_prediction_stat_of_tree=marginal),
        _interaction_screening_cache={"pairs": {}, "lower": {}},
    )


def test_pure_interaction_is_found_without_main_effects_and_before_tree_averaging():
    def product(tree, values):
        if np.isfinite(values[0]) and np.isfinite(values[1]):
            return (-1) ** tree * (2 * values[0] - 1) * (2 * values[1] - 1)
        return 0.0

    forest = _analytic_forest(product)
    # Opposite trees have a zero ensemble mean, but both contain the same interaction.
    forest.trees_total_variance.extend([0.0, np.nan])
    rows = screen_interactions(forest, range(10), top_k=1)
    assert len(rows) == 45
    assert rows[0]["members"] == ["x0", "x1"]
    assert rows[0]["score"] == pytest.approx((1 - 1 / 16**2) ** 2)
    assert rows[0]["offset"] < 1e-25
    assert rows[0]["status"] == "Stable"
    assert all(row["score"] == 0 for row in rows[1:])
    assert sum(row["audit_difference"] is not None for row in rows) == 2


def test_additive_surface_does_not_create_pair_interactions():
    forest = _analytic_forest(
        lambda tree, values: sum((value if np.isfinite(value) else 0.5) for value in values),
        variance=10 / 12,
    )
    rows = screen_interactions(forest, range(4), top_k=1)
    assert max(row["score"] for row in rows) < 1e-25
    assert max(row["offset"] for row in rows) < 1e-25


def test_offset_is_squared_per_tree_without_signed_cancellation():
    def shifted(tree, values):
        if np.isfinite(values[0]) and np.isfinite(values[1]):
            return (-1) ** tree * 0.5
        return 0.0

    forest = _analytic_forest(shifted, variance=1)
    result = estimate_pair(forest, (0, 1), 8)
    assert result == {"score": 0.25, "offset": 0.25}
    rows = screen_interactions(forest, range(4), top_k=1)
    assert rows[0]["status"] == "Refine recommended"


def test_shifted_grid_audit_promotes_a_low_rank_narrow_interaction():
    def surface(tree, values):
        value = 0.0
        for first, second in ((0, 1), (2, 3), (4, 5)):
            if np.isfinite(values[first]) and np.isfinite(values[second]):
                value += (2 * values[first] - 1) * (2 * values[second] - 1)
        if np.isfinite(values[8]) and np.isfinite(values[9]):

            def narrow(x):
                return 1 if x < 0.04 else -0.04 / 0.96

            value += narrow(values[8]) * narrow(values[9])
        return value

    forest = _analytic_forest(surface, variance=1, trees=1)
    rows = screen_interactions(forest, range(10), top_k=1, offset_tolerance=10)
    narrow = next(row for row in rows if row["members"] == ["x8", "x9"])
    assert narrow["audit_difference"] > 0.001
    assert narrow["medium"] is not None
    assert narrow["stability"] > 0.001


def test_grid_respects_encoded_log_bounds_and_categorical_mass():
    numeric = _grid((0, 1), 8, 0.5)
    assert numeric[0] == (1 / 16, 1 / 8)
    assert sum(weight for _value, weight in numeric) == 1
    categories = _grid((13, np.nan), 8, 0.5)
    assert sum(weight for _value, weight in categories) == 13
    assert all(value.is_integer() and 0 <= value < 13 for value, _weight in categories)
    assert _grid((3, np.nan), 8, 0.5) == [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]


def test_native_forest_roundtrip_retains_predictions_marginals_and_exact_pair_cache():
    cs = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0), "kind": ["a", "b"]})
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(80, 3))
    # ConfigSpace's encoded order is kind, x, y.
    X[:, 0] = rng.randint(0, 2, size=len(X))
    model = FanovaForest(cs, n_trees=3)
    model.train(X, X[:, 1] * X[:, 2] + 0.1 * X[:, 0])
    model.compute_marginals([1, 2], depth=2)
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(model.predict(X)[0], restored.predict(X)[0])
    sample = [np.nan, 0.3, 0.7]
    for tree in range(3):
        first = model._model.marginal_prediction_stat_of_tree(tree, sample).mean()
        second = restored._model.marginal_prediction_stat_of_tree(tree, sample).mean()
        assert first == second
    assert restored.V_U_individual == model.V_U_individual
    assert screen_interactions(restored, range(3), top_k=1) == screen_interactions(
        model, range(3), top_k=1
    )


def test_exact_pair_api_does_not_calculate_unselected_cross_pairs():
    cs = ConfigurationSpace(space={name: (0.0, 1.0) for name in "abcd"})
    X = np.random.RandomState(0).uniform(size=(60, 4))
    forest = FanovaForest(cs, n_trees=2)
    forest.train(X, X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3])
    evaluator = fANOVA(SimpleNamespace(configspace=cs))
    evaluator._model, evaluator.n_trees = forest, 2
    pairs = evaluator.get_pair_importances([("b", "a"), ("c", "d")])
    assert set(pairs) == {("a", "b"), ("c", "d")}
    computed = {key for key in forest.V_U_individual if len(key) == 2}
    assert computed == {(0, 1), (2, 3)}
