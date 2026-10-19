"""Exact checks, bounded allocation and independent backtests for trial forecasts."""

import json
from copy import deepcopy
from itertools import combinations, product
from math import comb

import numpy as np
import pytest
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    ForbiddenEqualsClause,
    NormalFloatHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.evaluators import coverage_forecast as forecast
from deepcave.evaluators.combinatorial import (
    CoverageSpec,
    compute_combinatorial_coverage,
)
from deepcave.evaluators.coverage import normalize_configurations, run_coverage_analysis


def binary_data(rows, dimensions=3, weights=None):
    space = ConfigurationSpace()
    names = [f"p{i}" for i in range(dimensions)]
    space.add([CategoricalHyperparameter(name, [0, 1], weights=weights) for name in names])
    return normalize_configurations(space, [dict(zip(names, row)) for row in rows])


def options(**changes):
    return forecast.forecast_options({"simulations": 128, **changes})


def exact_uniform_probability(total, missing, trials):
    return sum((-1) ** j * comb(missing, j) * (1 - j / total) ** trials for j in range(missing + 1))


def success_rate(history, strength, budget, seed=294, repeats=4000):
    """Independent, direct occupancy backtest without the forecast's state implementation."""
    dimensions = len(history[0])
    future = np.random.default_rng(seed).integers(2, size=(repeats, budget, dimensions))
    complete = np.ones(repeats, dtype=bool)
    for group in combinations(range(dimensions), strength):
        occupied = np.zeros((repeats, 2**strength), dtype=bool)
        for row in history:
            code = sum(row[column] * 2**j for j, column in enumerate(group))
            occupied[:, code] = True
        codes = sum(future[:, :, column] * 2**j for j, column in enumerate(group))
        occupied[np.arange(repeats)[:, None], codes] = True
        complete &= occupied.all(axis=1)
    return complete.mean()


def test_deterministic_without_mutating_trials_search_space_or_measured_coverage():
    data = binary_data([(0, 0, 0), (1, 1, 1)])
    original_values = deepcopy(data.values)
    original_rng = deepcopy(data.configspace.random.get_state())
    measured = compute_combinatorial_coverage(data)
    snapshot = deepcopy(measured)
    first = forecast.compute_coverage_forecast(data, options(), observed=measured)
    assert first == forecast.compute_coverage_forecast(data, options(), observed=measured)
    assert measured == snapshot and data.values == original_values
    after = data.configspace.random.get_state()
    assert after[0] == original_rng[0]
    np.testing.assert_array_equal(after[1], original_rng[1])
    assert after[2:] == original_rng[2:]
    assert all(row["next"]["method"] == "MC" for row in first["rows"])
    json.dumps(first, allow_nan=False)


def test_complete_history_has_zero_budget_and_no_next_target():
    result = forecast.compute_coverage_forecast(
        binary_data(list(product([0, 1], repeat=3))), options()
    )
    for row in result["rows"]:
        assert row["current_strength"] == row["full_strength_target"] == 3
        assert row["next_strength_target"] is None
        assert row["next_strength_trials_p80"] == row["full_strength_trials_p80"] == 0
        assert row["forecast_status"] == "Completed"


def test_full_target_is_dimension_even_above_trial_count_bound():
    data = binary_data([(0, 0, 0)])
    result = forecast.compute_coverage_forecast(data, options())
    for row in result["rows"]:
        assert row["current_strength"] == 0
        assert row["next_strength_target"] == 1
        assert row["full_strength_target"] == 3
        assert row["full_strength_trials_p80"] >= row["next_strength_trials_p80"]


def test_next_equals_full_reuses_identical_analytic_result():
    result = forecast.compute_coverage_forecast(
        binary_data([(0, 0), (1, 1)], dimensions=2), options()
    )
    for row in result["rows"]:
        assert row["next_strength_target"] == row["full_strength_target"] == 2
        assert row["next"] == row["full"]


def test_probability_monotonicity_with_common_random_continuations():
    data = binary_data([(0, 0, 0), (1, 1, 1)])
    lower = forecast.compute_coverage_forecast(data, options(probability=0.8))
    higher = forecast.compute_coverage_forecast(data, options(probability=0.95))
    for low, high in zip(lower["rows"], higher["rows"]):
        for name in ("next", "full"):
            assert low[name]["trials_p80"] <= high[name]["trials_p80"]


def test_more_history_cannot_increase_budget_at_a_fixed_target():
    first = binary_data([(0, 0, 0), (1, 1, 1)])
    second = binary_data([(0, 0, 0), (1, 1, 1), (0, 1, 0)])
    predictions = []
    for data in (first, second):
        spec = CoverageSpec(data, 2)
        predictions.append(
            forecast._next_forecast(
                spec, 2, forecast.level_probabilities(spec), options(), float("inf")
            )
        )
    assert predictions[1]["trials_p80"] <= predictions[0]["trials_p80"]
    for a, b in zip(
        forecast.compute_coverage_forecast(first, options())["rows"],
        forecast.compute_coverage_forecast(second, options())["rows"],
    ):
        assert b["full"]["trials_p80"] <= a["full"]["trials_p80"]


@pytest.mark.parametrize("total,missing", [(2, 1), (4, 3), (8, 5), (16, 12)])
@pytest.mark.parametrize("probability", [0.5, 0.8, 0.95])
def test_uniform_exact_path_against_inclusion_exclusion(total, missing, probability):
    result = forecast.uniform_coupon_forecast(total, missing, probability)
    n = result["trials_p80"]
    assert result["method"] == "Exact"
    assert exact_uniform_probability(total, missing, n - 1) < probability
    assert exact_uniform_probability(total, missing, n) >= probability - 1e-12


def test_nonuniform_exact_path_with_two_missing_coupons():
    masses = np.array([0.05, 0.25])
    result = forecast._coupon_forecast(masses, 0.8)
    n = result["trials_p80"]
    cdf = lambda budget: 1 - 0.95**budget - 0.75**budget + 0.7**budget
    assert result["method"] == "Exact"
    assert cdf(n - 1) < 0.8 <= cdf(n)


def test_poisson_approximations_against_independent_fixed_trial_multinomials():
    rng = np.random.default_rng(901)
    for masses, result in (
        (np.full(100, 1 / 128), forecast.uniform_coupon_forecast(128, 100, 0.8)),
        (
            np.r_[np.full(19, 0.025), 0.001],
            forecast._coupon_forecast(np.r_[np.full(19, 0.025), 0.001], 0.8),
        ),
    ):
        assert result["method"] == "Approximation"
        p = np.r_[masses, 1 - masses.sum()]
        counts = rng.multinomial(result["trials_p80"], p, size=10000)
        success = (counts[:, :-1] > 0).all(axis=1).mean()
        assert abs(success - 0.8) < 0.025
        bounds = result["uncertainty"]
        assert bounds["lower"] <= result["trials_p80"] <= bounds["upper"]


def test_zero_probability_missing_configuration_is_unavailable():
    result = forecast.compute_coverage_forecast(binary_data([(0, 0)], 2, [1, 0]), options())
    assert result["rows"][0]["full"]["trials_p80"] is None
    assert "zero probability" in result["rows"][0]["full"]["reason"]


def test_unknown_distribution_is_not_assumed_uniform():
    from types import SimpleNamespace

    spec = CoverageSpec(binary_data([(0, 0, 0)]), 2)
    spec.axes[0].hp = SimpleNamespace(name="custom")
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        forecast.level_probabilities(spec)


def test_analytic_search_honours_the_shared_time_limit():
    with pytest.raises(TimeoutError):
        forecast._coupon_forecast(np.full(20, 0.01), 0.8, deadline=-1)


def test_analytic_timeout_is_censored_without_discarding_target_metadata(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("Shared forecast time limit reached.")

    monkeypatch.setattr(forecast, "_full_forecast", timeout)
    result = forecast.compute_coverage_forecast(binary_data([(0, 0), (1, 1)], 2), options())
    for row in result["rows"]:
        assert row["full_strength_target"] == row["next_strength_target"] == 2
        assert row["full"]["status"] == row["next"]["status"] == "Censored"
        assert row["full"]["trials_p80"] is None


def test_rare_combinations_require_more_random_trials_regardless_of_history_frequencies():
    history = [(0, 0)] * 100 + [(1, 0), (0, 1)]
    uniform = forecast.compute_coverage_forecast(binary_data(history, 2), options())
    weighted = forecast.compute_coverage_forecast(binary_data(history, 2, [99, 1]), options())
    assert (
        weighted["rows"][0]["full"]["trials_p80"] > 100 * uniform["rows"][0]["full"]["trials_p80"]
    )
    assert weighted["rows"][0]["level_probabilities"] == [[0.99, 0.01], [0.99, 0.01]]


def test_probabilities_and_boundaries_follow_existing_cdf_bins_and_discrete_steps():
    space = ConfigurationSpace()
    space.add(
        [
            UniformFloatHyperparameter("log", 0.001, 1, log=True),
            NormalFloatHyperparameter("normal", mu=0.4, sigma=0.1, lower=0, upper=1),
            UniformIntegerHyperparameter("step", 0, 8, meta={"step": 2}),
            CategoricalHyperparameter("category", ["a", "b", "c"], weights=[8, 1, 1]),
            Constant("fixed", 17),
        ]
    )
    data = normalize_configurations(
        space, [{"log": 0.01, "normal": 0.4, "step": 4, "category": "a", "fixed": 17}]
    )
    spec = CoverageSpec(data, 2)
    masses = dict(zip(spec.names, forecast.level_probabilities(spec)))
    np.testing.assert_allclose(masses["log"], [0.5, 0.5])
    np.testing.assert_allclose(masses["normal"], [0.5, 0.5])
    np.testing.assert_allclose(masses["step"], [0.4, 0.6])
    np.testing.assert_allclose(masses["category"], [0.8, 0.1, 0.1])
    assert "fixed" not in spec.names
    result = forecast.compute_coverage_forecast(data, options())
    assert result["rows"][0]["binning"] == spec.serialize()
    assert result["rows"][0]["full_strength_target"] == 4


@pytest.mark.parametrize("constraint", ["conditional", "forbidden"])
def test_constraints_are_unsupported_instead_of_counting_impossible_combinations(constraint):
    space = ConfigurationSpace()
    a, b = [CategoricalHyperparameter(name, [0, 1], default_value=1) for name in "ab"]
    space.add([a, b])
    space.add(
        EqualsCondition(b, a, 1) if constraint == "conditional" else ForbiddenEqualsClause(a, 0)
    )
    result = forecast.compute_coverage_forecast(
        normalize_configurations(space, [{"a": 1, "b": 0}]), options()
    )
    assert all(row["forecast_status"] == "Unsupported" for row in result["rows"])
    assert all(row["full"]["trials_p80"] is None for row in result["rows"])
    assert "constraints" in result["rows"][0]["full"]["reason"]


def test_large_uniform_space_is_analytic_without_enumerating_full_configurations(monkeypatch):
    space = ConfigurationSpace()
    space.add([UniformFloatHyperparameter(f"x{i}", 0, 1) for i in range(10)])
    data = normalize_configurations(space, [{f"x{i}": 0.1 for i in range(10)}])

    def unexpected(*args):
        pytest.fail("Uniform full forecasts must not materialize a coupon vector")

    monkeypatch.setattr(forecast, "_coupon_forecast", unexpected)
    result = forecast.compute_coverage_forecast(data, options())
    full = result["rows"][-1]["full"]
    assert full["total_combinations"] == 8**10
    assert full["observed_combinations"] == 1
    assert full["status"] == "Approximate"
    assert 20_000_000_000 < full["trials_p80"] < 30_000_000_000


def test_large_nonuniform_space_returns_a_bound_instead_of_a_uniform_estimate():
    data = binary_data([(0,) * 10], 10, weights=[9, 1])
    result = forecast.compute_coverage_forecast(data, options(max_explicit_coupons=16))
    full = result["rows"][0]["full"]
    assert full["kind"] == "upper_bound"
    assert full["trials_p80"] is None
    assert full["upper_bound_trials"] > 10**10


def test_censored_continuations_do_not_produce_a_quantile_from_successes():
    result = forecast.compute_coverage_forecast(
        binary_data([(0, 0, 0), (1, 1, 1)]), options(max_trials=1)
    )
    next_result = result["rows"][0]["next"]
    assert next_result["status"] == "Censored"
    assert next_result["trials_p80"] is None
    assert next_result["simulations_censored"] == 128
    assert result["rows"][0]["full"]["trials_p80"] is not None


def test_time_limit_reports_all_unstarted_simulations():
    spec = CoverageSpec(binary_data([(0, 0, 0), (1, 1, 1)]), 2)
    result = forecast._next_forecast(spec, 2, forecast.level_probabilities(spec), options(), -1)
    assert result["status"] == "Censored" and result["trials_p80"] is None
    assert result["simulations_not_started"] == 128


def test_tuple_limit_uses_a_history_conditioned_bound_for_dependent_groups():
    history = [(0, 0, 0), (1, 1, 1)]
    result = forecast.compute_coverage_forecast(binary_data(history), options(max_tuples=1))
    bound = result["rows"][0]["next"]
    assert bound["kind"] == "upper_bound" and bound["missing"] == 6
    assert bound["trials_p80"] is None
    assert success_rate(history, 2, bound["upper_bound_trials"]) >= 0.8


def test_work_limited_current_strength_has_no_invented_next_target():
    data = binary_data([(0, 0, 0), (1, 1, 1)])
    observed = compute_combinatorial_coverage(data)
    for row in observed["rows"]:
        row["strength_is_lower_bound"] = True
    result = forecast.compute_coverage_forecast(data, options(), observed)
    assert all(row["next_strength_target"] is None for row in result["rows"])
    assert all(row["next"]["status"] == "Unsupported" for row in result["rows"])
    assert all(row["full"]["trials_p80"] is not None for row in result["rows"])


def test_synthetic_random_history_backtesting():
    rates = {"next": [], "full": []}
    previous_rates = {"next": [], "full": []}
    rng = np.random.default_rng(219)
    for size in (2, 4, 6, 8, 10, 12):
        history = rng.integers(2, size=(size, 3)).tolist()
        row = forecast.compute_coverage_forecast(binary_data(history), options(simulations=512))[
            "rows"
        ][0]
        for name in ("next", "full"):
            n = row[name]["trials_p80"]
            strength = row[f"{name}_strength_target"]
            if not n or strength is None:
                continue
            rates[name].append(success_rate(history, strength, n))
            previous_rates[name].append(success_rate(history, strength, n - 1))
    for name in rates:
        assert 0.76 <= np.mean(rates[name]) <= 0.9
        assert np.mean(previous_rates[name]) <= 0.84


@pytest.mark.parametrize(
    "settings",
    [
        {"simulations": 0},
        {"simulations": True},
        {"probability": 1},
        {"probability": float("nan")},
        {"timeout": 0},
        {"max_tuples": 0},
        {"max_trials": 1.5},
        {"seed": -1},
    ],
)
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        forecast.forecast_options(settings)


def test_worker_dispatch_uses_forecast_settings():
    data = binary_data([(0, 0, 0)])
    result = run_coverage_analysis("forecast", data, data.configspace, options(probability=0.95))
    assert result["probability_threshold"] == 0.95
