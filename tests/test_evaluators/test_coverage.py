"""Sampling-space statistics and reproducible coverage simulations."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import (
    BetaFloatHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    NormalFloatHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from scipy.stats import beta, qmc, truncnorm

from deepcave.evaluators import coverage
from deepcave.runs.status import Status


def make_run(space, configurations, statuses=None):
    statuses = statuses or [Status.SUCCESS] * len(configurations)
    return SimpleNamespace(
        id="coverage-test",
        name="Coverage test",
        configspace=space,
        latest_change=1,
        history=[
            SimpleNamespace(config_id=i, budget=1, seed=0, status=status)
            for i, status in enumerate(statuses)
        ],
        get_config=lambda index: configurations[index],
    )


def test_log_sampling_space_and_trial_statuses():
    space = ConfigurationSpace()
    space.add(UniformFloatHyperparameter("lr", 0.005, 0.08, log=True))
    run = make_run(
        space,
        [{"lr": 0.005}, {"lr": 0.02}, {"lr": 0.08}],
        [Status.SUCCESS, Status.CRASHED, Status.PRUNED],
    )
    data = coverage.prepare_coverage(run)
    np.testing.assert_allclose(data.points[:, 0], [0, 0.5, 1])
    assert data.overview["trials"] == 3
    assert data.overview["complete"] == 1
    assert data.overview["failed"] == 1
    assert data.overview["statuses"]["PRUNED"] == 1
    (row,) = coverage.compute_univariate_coverage(data, bins=2)
    assert row["range_fraction"] == 1
    assert row["max_gap"] == pytest.approx(0.5)
    assert row["raw_max_gap"] == pytest.approx(0.06)
    assert row["coverage"] == 1
    assert row["expected"] == [1.5, 1.5]
    assert row["distribution"] == "LogUniformFloat"


def test_nonuniform_continuous_distributions_use_actual_sampling_cdf():
    space = ConfigurationSpace()
    space.add(
        [
            BetaFloatHyperparameter("beta", alpha=2, beta=3, lower=0, upper=1),
            NormalFloatHyperparameter("normal", mu=0, sigma=1, lower=-3, upper=3),
        ]
    )
    data = coverage.normalize_configurations(space, [{"beta": 0.25, "normal": 1}])
    assert data.points[0, 0] == pytest.approx(beta.cdf(0.25, 2, 3))
    assert data.points[0, 1] == pytest.approx(truncnorm.cdf(1, -3, 3))


def test_weighted_categories_and_inactive_parameters():
    space = ConfigurationSpace()
    category = CategoricalHyperparameter("kind", ["a", "b", "c"], weights=[1, 2, 1])
    child = UniformFloatHyperparameter("child", 0, 1)
    space.add([category, child, EqualsCondition(child, category, "a")])
    data = coverage.normalize_configurations(
        space,
        [
            {"kind": "a", "child": 0.2},
            {"kind": "a", "child": 0.8},
            {"kind": "b"},
            {"kind": "c"},
        ],
    )
    rows = {row["name"]: row for row in coverage.compute_univariate_coverage(data)}
    assert rows["kind"]["expected"] == [1, 2, 1]
    assert rows["kind"]["relative_deviation"] == [1, -0.5, 0]
    assert rows["child"]["conditional"] is True
    assert rows["child"]["active_trials"] == 2
    assert rows["child"]["inactive_trials"] == 2
    assert np.isnan(data.points[2:, 1]).all()
    (pair,) = coverage.compute_pairwise_coverage(data)["pairs"]
    assert pair["active_trials"] == 2
    assert pair["discrepancy"] is None


def test_integer_steps_and_missing_edge_runs():
    space = ConfigurationSpace()
    space.add(UniformIntegerHyperparameter("depth", 1, 9, meta={"step": 2}))
    data = coverage.normalize_configurations(space, [{"depth": 1}, {"depth": 5}])
    (row,) = coverage.compute_univariate_coverage(data)
    assert row["labels"] == ["1", "3", "5", "7", "9"]
    assert row["frequencies"] == [1, 0, 1, 0, 0]
    assert row["expected"] == pytest.approx([0.4] * 5)
    assert row["coverage"] == pytest.approx(0.4)
    assert row["largest_missing_run"] == 2
    assert row["unseen"] == ["3", "7", "9"]


def test_log_integer_mass_matches_native_sampling():
    space = ConfigurationSpace()
    hp = UniformIntegerHyperparameter("depth", 1, 10, log=True)
    space.add(hp)
    axis = coverage.CoverageAxis(hp)
    left, right = axis.mass_edges(np.arange(10))
    assert sum(right - left) == pytest.approx(1)
    # Native ConfigSpace quantization boundaries in value space are 1 + k*9/10.
    expected = np.diff(np.log10(1 + np.arange(11) * 0.9))
    np.testing.assert_allclose(right - left, expected)


def test_huge_integer_ranges_do_not_materialize_the_lattice():
    space = ConfigurationSpace()
    space.add(UniformIntegerHyperparameter("integer", 1, 1_000_000_000))
    (row,) = coverage.compute_univariate_coverage(
        coverage.normalize_configurations(space, [{"integer": 1}, {"integer": 100}])
    )
    assert len(row["labels"]) == 2
    assert row["frequency_table_complete"] is False
    assert row["unseen_count"] == 999_999_998
    assert row["largest_missing_run"] == 999_999_900


def test_discrepancy_matches_scipy_and_labels_the_projection():
    space = ConfigurationSpace()
    space.add(
        [
            UniformFloatHyperparameter("x", 0, 1),
            UniformIntegerHyperparameter("z", 1, 3, default_value=1),
        ]
    )
    data = coverage.normalize_configurations(space, [{"x": 0.25, "z": 1}, {"x": 0.75, "z": 3}])
    result = coverage.compute_global_discrepancy(data)
    assert result["parameters"] == ["x"]
    assert result["centered"] == pytest.approx(qmc.discrepancy([[0.25], [0.75]]))
    assert result["l2_star"] == pytest.approx(qmc.discrepancy([[0.25], [0.75]], method="L2-star"))
    space.add(ForbiddenEqualsClause(space["z"], 2))
    constrained = coverage.normalize_configurations(space, data.values)
    assert coverage.compute_global_discrepancy(constrained)["centered"] is None
    assert "forbidden" in coverage.compute_global_discrepancy(constrained)["reason"]


def test_pairwise_grid_occupancy_and_ranking():
    space = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)})
    data = coverage.normalize_configurations(
        space,
        [
            {"x": 0, "y": 0, "z": 0},
            {"x": 0, "y": 1, "z": 0},
            {"x": 1, "y": 0, "z": 1},
            {"x": 1, "y": 1, "z": 1},
        ],
    )
    results = coverage.compute_pairwise_coverage(data, bins=2)["pairs"]
    assert results[0]["x"] == "x" and results[0]["y"] == "z"
    assert results[0]["occupancy"] == 0.5
    assert results[0]["empty_cell_rate"] == 0.5
    assert results[0]["hole_score"] == 0.5
    assert results[-1]["occupancy"] == 1
    assert all(row["discrepancy"] is not None for row in results)


def test_radius_quantiles_and_decoded_regions():
    space = ConfigurationSpace(space={"x": (0.0, 1.0)})
    data = coverage.normalize_configurations(space, [{"x": 0}, {"x": 1}])
    references = coverage.normalize_configurations(
        space, [{"x": x} for x in [0, 0.25, 0.5, 0.75, 1]]
    )
    result = coverage.compute_coverage_radius(data, references)
    assert result["median"] == 0.25
    assert result["p95"] == pytest.approx(0.45)
    assert result["max"] == 0.5
    assert result["regions"][0] == {"x": 0.5, "distance": 0.5}
    assert sum(result["histogram_y"]) == 5
    assert coverage.compute_coverage_radius(data, references, holes=0)["regions"] == []


def test_mixed_radius_uses_hamming_and_activation_distance():
    space = ConfigurationSpace()
    category = CategoricalHyperparameter("kind", ["a", "b"])
    child = UniformFloatHyperparameter("x", 0, 1)
    space.add([category, child, EqualsCondition(child, category, "a")])
    data = coverage.normalize_configurations(space, [{"kind": "a", "x": 0.2}])
    refs = coverage.normalize_configurations(space, [{"kind": "a", "x": 0.7}, {"kind": "b"}])
    np.testing.assert_allclose(coverage._nearest_distances(data, refs), [0.5 / np.sqrt(2), 1])
    repeated = coverage.normalize_configurations(space, [{"kind": "b"}] * 4)
    result = coverage.compute_coverage_radius(data, repeated)
    assert len(result["regions"]) == 1


def test_random_sampling_respects_steps_conditions_forbiddens_and_rng():
    space = ConfigurationSpace(seed=19)
    category = CategoricalHyperparameter("kind", ["a", "b"], weights=[3, 1])
    child = UniformFloatHyperparameter("x", 0, 1, default_value=0.25, meta={"step": 0.25})
    integer = UniformIntegerHyperparameter("i", 1, 9, meta={"step": 2})
    space.add([category, child, integer, EqualsCondition(child, category, "a")])
    space.add(
        ForbiddenAndConjunction(
            ForbiddenEqualsClause(category, "a"), ForbiddenEqualsClause(child, 0.5)
        )
    )
    original = deepcopy(space)
    data = coverage.sample_reference_design(space, 100, 8)
    again = coverage.sample_reference_design(space, 100, 8)
    assert data.values == again.values
    assert dict(space.sample_configuration()) == dict(original.sample_configuration())
    for values in data.values:
        Configuration(space, values=values).check_valid_configuration()
        assert values["i"] in (1, 3, 5, 7, 9)
        assert ("x" in values) == (values["kind"] == "a")
        if "x" in values:
            assert values["x"] in (0, 0.25, 0.75, 1)


def test_custom_step_sampling_handles_nested_inactive_parents():
    space = ConfigurationSpace()
    root = CategoricalHyperparameter("root", ["a", "b"])
    parent = UniformIntegerHyperparameter("parent", 1, 5, meta={"step": 2})
    child = UniformFloatHyperparameter("child", 0, 1)
    space.add(
        [root, parent, child, EqualsCondition(parent, root, "a"), EqualsCondition(child, parent, 3)]
    )
    design = coverage.sample_reference_design(space, 64, 3)
    assert any(values["root"] == "b" for values in design.values)
    for values in design.values:
        assert ("parent" in values) == (values["root"] == "a")
        assert ("child" in values) == (values.get("parent") == 3)
        Configuration(space, values=values).check_valid_configuration()


def test_growth_uses_nested_prefixes_and_fixed_references():
    space = ConfigurationSpace(space={"x": (0.0, 1.0)})
    data = coverage.normalize_configurations(space, [{"x": 0}, {"x": 1}, {"x": 0.5}, {"x": 0.25}])
    refs = coverage.sample_reference_design(space, 64, 11)
    result = coverage.compute_coverage_growth(data, refs, [1, 2, 3, 4])
    radii = [row["p95_radius"] for row in result["checkpoints"]]
    assert all(a >= b for a, b in zip(radii, radii[1:]))
    final = result["checkpoints"][-1]
    assert final["p95_radius"] == coverage.compute_coverage_radius(data, refs)["p95"]
    assert final["centered_discrepancy"] == coverage.compute_global_discrepancy(data)["centered"]
    assert final["max_radius"] == coverage.compute_coverage_radius(data, refs)["max"]
    assert final["marginal_max_gap"] == coverage.compute_univariate_coverage(data)[0]["max_gap"]
    checkpoints = coverage.growth_checkpoints(2000)
    assert checkpoints[0] == 1 and checkpoints[-1] == 2000
    assert len(checkpoints) <= 20 and checkpoints == sorted(set(checkpoints))


def test_random_baselines_are_reproducible_and_report_percentiles():
    space = ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)})
    data = coverage.normalize_configurations(space, [{"x": 0, "y": 0}, {"x": 1, "y": 1}])
    result = coverage.compute_random_baseline(data, space, designs=3, reference_samples=32, seed=13)
    assert result == coverage.compute_random_baseline(
        data, space, designs=3, reference_samples=32, seed=13
    )
    for row in result["metrics"].values():
        assert 0 <= row["percentile"] <= 100
        assert row["lower"] <= row["upper"]
    growth = coverage.compute_growth_baseline(data, space, designs=3, reference_samples=32, seed=13)
    assert (
        growth["bands"][-1]["p95_radius"]["observed"] == result["metrics"]["p95_radius"]["observed"]
    )
    assert growth["bands"][-1]["p95_radius"]["mean"] == result["metrics"]["p95_radius"]["mean"]
    for metric in ("max_radius", "pairwise_occupancy", "marginal_max_gap"):
        assert growth["bands"][-1][metric]["observed"] == result["metrics"][metric]["observed"]
        assert growth["bands"][-1][metric]["mean"] == result["metrics"][metric]["mean"]
    json.dumps([result, growth], allow_nan=False)


def test_ties_and_zero_baseline_variance():
    row = coverage.summarize_baseline(0.5, [0.5, 0.5])
    assert row["percentile"] == 50
    assert row["standardized_deviation"] is None
    assert coverage.summarize_baseline(None, [0.5])["percentile"] is None


def test_fingerprints_capture_trial_and_definition_changes():
    space = ConfigurationSpace(space={"x": (0.0, 1.0)})
    values = [{"x": 0.5}]
    run = make_run(space, values)
    initial = coverage.prepare_coverage(run).fingerprint
    values[0]["x"] = 0.6
    assert coverage.prepare_coverage(run).fingerprint != initial
    values[0]["x"] = 0.5
    run.history[0].status = Status.FAILED
    assert coverage.prepare_coverage(run).fingerprint != initial
    run.history[0].status = Status.SUCCESS
    space["x"].meta = {"step": 0.1}
    assert coverage.prepare_coverage(run).fingerprint != initial


def test_empty_history_has_explicit_unavailable_metrics():
    space = ConfigurationSpace(space={"x": (0.0, 1.0)})
    data = coverage.prepare_coverage(make_run(space, []))
    (row,) = coverage.compute_univariate_coverage(data)
    assert row["active_trials"] == 0 and row["coverage"] == 0
    assert row["ks"] is None
    assert coverage.compute_global_discrepancy(data)["centered"] is None
    with pytest.raises(ValueError, match="sampled trials"):
        coverage.compute_coverage_radius(data, coverage.sample_reference_design(space, 32, 0))
    json.dumps(row, allow_nan=False)
