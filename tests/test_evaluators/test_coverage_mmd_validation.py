"""Kernel validity, controlled dependencies, and finite-sample MMD sensitivity."""

from collections import Counter
from itertools import product

import numpy as np
import pytest
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.evaluators import coverage
from examples.api import validate_coverage_mmd as validation


@pytest.mark.parametrize("width", [0.05, 0.2, 0.5])
def test_mixed_and_conditional_kernel_is_positive_semidefinite_and_matches_production(
    monkeypatch, width
):
    monkeypatch.setattr(coverage, "MIXED_MMD_KERNEL_SCALE", width)
    space = ConfigurationSpace()
    space.add(
        [
            CategoricalHyperparameter("family", ["linear", "tree"]),
            UniformIntegerHyperparameter("depth", 1, 3),
            UniformFloatHyperparameter("rate", 0.0001, 1, log=True),
            UniformFloatHyperparameter("x", 0, 1),
            OrdinalHyperparameter("level", ["low", "mid", "high"]),
        ]
    )
    space.add(EqualsCondition(space["depth"], space["family"], "tree"))
    data = coverage.sample_reference_design(space, 24, 7)
    points, axes = coverage._mixed_coordinates(data)
    assert np.isnan(points).any()
    matrices = validation.kernel_matrices(points, points, axes)
    assert np.array_equal(matrices["product"], coverage._mixed_kernel(points, points, axes))
    for matrix in matrices.values():
        assert np.isfinite(matrix).all()
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 1)
        assert matrix.min() >= -1e-12 and matrix.max() <= 1 + 1e-12
        assert np.linalg.eigvalsh(matrix).min() >= -1e-10


def test_mixed_kernel_uses_sampling_coordinates_for_log_parameters():
    coordinates = np.array([0.1, 0.3, 0.7, 0.9])
    spaces = [ConfigurationSpace(), ConfigurationSpace()]
    spaces[0].add(UniformFloatHyperparameter("x", 0, 1))
    spaces[1].add(UniformFloatHyperparameter("x", 0.0001, 1, log=True))
    values = [coordinates, np.exp(np.log(0.0001) * (1 - coordinates))]
    matrices = []
    for space, native in zip(spaces, values):
        data = coverage.normalize_configurations(space, [{"x": float(x)} for x in native])
        points, axes = coverage._mixed_coordinates(data)
        matrices.append(coverage._mixed_kernel(points, points, axes))
    assert np.allclose(matrices[0], matrices[1], atol=1e-12)


@pytest.mark.parametrize("dimensions", [2, 3])
def test_joint_only_defects_are_invisible_to_marginal_control(dimensions):
    space = ConfigurationSpace(space={f"c{i}": ["a", "b"] for i in range(dimensions)})
    complete = [
        dict(zip(space.keys(), values)) for values in product(["a", "b"], repeat=dimensions)
    ]
    incomplete = [config for config in complete if sum(v == "b" for v in config.values()) % 2 == 0]
    # Preserve every marginal exactly while removing half the combinations.
    for name in space:
        assert Counter(c[name] for c in complete) == Counter(c[name] for c in incomplete * 2)
    reference = coverage.normalize_configurations(space, complete)
    observed = coverage.normalize_configurations(space, incomplete * 2)
    points, axes = coverage._mixed_coordinates(reference)
    means = validation.kernel_means(points, points, axes)
    scores = dict(zip(validation.KERNELS, validation.mmd_scores(observed, reference, means)))
    assert scores["product"] > 0.2
    assert scores["marginal"] == pytest.approx(0, abs=1e-8)
    assert scores["interaction_mixture"] > 0.1
    if dimensions == 3:
        # All pairwise marginals are also identical: only the full interaction contributes.
        assert scores["interaction_mixture"] ** 2 == pytest.approx(scores["product"] ** 2 / 3)


@pytest.mark.parametrize("definition", validation.make_spaces([6]), ids=lambda item: item["name"])
def test_synthetic_draws_remain_legal_and_introduce_the_declared_defect(definition):
    space = definition["space"]
    for scenario in ["random", *definition["scenarios"]]:
        data = validation.draw(space, 80, 17, scenario)
        assert len(data.values) == 80
        for values in data.values:
            Configuration(space, values=values)
        if scenario in ("missing_category", "missing_branch"):
            name = "category" if "category" in space else "c0"
            assert {v[name] for v in data.values} == {"a"}
        elif scenario == "missing_rare_category":
            assert all(v["category"] != "rare" for v in data.values)
        elif scenario == "marginal_bias":
            assert max(v["x0"] for v in data.values) <= 0.5
        elif scenario == "conditional_bias":
            assert max(v.get("child", 0) for v in data.values) <= 0.25
        elif scenario == "missing_combinations":
            if "integer" in space:
                assert all(v["integer"] == int(v["category"] == "b") for v in data.values)
            else:
                assert all(v["c0"] == v["c1"] for v in data.values)
        elif scenario == "three_way_parity":
            assert all(sum(v[f"c{i}"] == "b" for i in range(3)) % 2 == 0 for v in data.values)


def test_validation_matches_midrank_ties_and_reports_binomial_uncertainty():
    result = validation.flag_rates(np.arange(100), np.array([0, 50, 99]))
    assert result["upper_count"] == 1
    assert result["outside_count"] == 2
    assert result["upper_95ci"][0] < 1 / 3 < result["upper_95ci"][1]
    ties = validation.flag_rates(np.ones(100), np.ones(10))
    assert ties["median_metric_percentile"] == 50
    assert ties["upper_rate"] == ties["outside_rate"] == 0


def test_strong_six_parameter_defects_exceed_the_independent_random_baseline():
    space = validation.make_spaces([6])[0]["space"]
    reference = validation.draw(space, 512, 20261017)
    points, axes = coverage._mixed_coordinates(reference)
    reference_means = validation.kernel_means(points, points, axes)
    calibration = np.array(
        [
            validation.mmd_scores(
                validation.draw(space, 80, 20262017 + i), reference, reference_means
            )[0]
            for i in range(99)
        ]
    )
    for scenario in ("missing_category", "marginal_bias", "missing_combinations"):
        scores = np.array(
            [
                validation.mmd_scores(
                    validation.draw(space, 80, 20271017 + i, scenario), reference, reference_means
                )[0]
                for i in range(16)
            ]
        )
        assert validation.flag_rates(calibration, scores)["upper_rate"] >= 0.75, scenario
