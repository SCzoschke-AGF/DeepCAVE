import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
)

from deepcave.evaluators.rashomon import (
    cluster_configurations,
    diversity_summary,
    pairwise_configuration_distances,
    parameter_spreads,
    random_subset_baseline,
    select_near_optimal_indices,
    subset_condensed_distances,
    subset_distance_matrix,
)
from deepcave.utils.configspace import normalized_configuration_matrix


def test_selection_is_direction_aware_and_deterministic():
    values = [0.8, 0.9, 0.9, 0.5]
    trial_ids = [10, 12, 11, 13]

    maximize = select_near_optimal_indices(values, trial_ids, "upper", "top_percent", 50)
    minimize = select_near_optimal_indices(values, trial_ids, "lower", "top_percent", 50)

    assert maximize.tolist() == [2, 1]
    assert minimize.tolist() == [3, 0]


def test_absolute_and_relative_selection_use_worse_direction():
    values = [10.0, 10.5, 11.1]
    ids = [0, 1, 2]
    assert select_near_optimal_indices(values, ids, "lower", "absolute", 0.5).tolist() == [0, 1]
    assert select_near_optimal_indices(values, ids, "lower", "relative", 5).tolist() == [0, 1]

    negative = [-10.0, -9.8, -9.0]
    assert select_near_optimal_indices(negative, ids, "lower", "relative", 2).tolist() == [0, 1]
    assert select_near_optimal_indices([0.0, 0.01], ids[:2], "lower", "relative", 10).tolist() == [
        0
    ]


def test_rms_distance_is_bounded_weighted_and_handles_inactive_values():
    configspace = ConfigurationSpace()
    model = CategoricalHyperparameter("model", ["a", "b"])
    child = UniformFloatHyperparameter("alpha", lower=0.0, upper=1.0)
    fixed = Constant("fixed", 1)
    configspace.add([model, child, fixed, EqualsCondition(child, model, "a")])
    configurations = [
        Configuration(configspace, values={"model": "b", "fixed": 1}),
        Configuration(configspace, values={"model": "a", "alpha": 0.0, "fixed": 1}),
        Configuration(configspace, values={"model": "a", "alpha": 1.0, "fixed": 1}),
    ]
    matrix = normalized_configuration_matrix(configurations, configspace)

    distances = pairwise_configuration_distances(matrix, configspace)
    weighted = pairwise_configuration_distances(
        matrix, configspace, weights={"model": 0.0, "alpha": 1.0}
    )

    assert distances.tolist() == pytest.approx([1.0, 1.0, np.sqrt(0.5)])
    assert weighted.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert np.all((distances >= 0) & (distances <= 1))


def test_subset_distance_helpers_and_diversity_are_exact():
    condensed = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    subset = subset_condensed_distances(condensed, 4, [0, 2, 3])
    matrix = subset_distance_matrix(condensed, 4, [0, 2, 3])

    assert subset.tolist() == pytest.approx([0.2, 0.3, 0.6])
    assert matrix == pytest.approx(np.asarray([[0, 0.2, 0.3], [0.2, 0, 0.6], [0.3, 0.6, 0]]))
    assert diversity_summary(subset) == pytest.approx({"median": 0.3, "q25": 0.25, "q75": 0.45})


def test_parameter_spread_includes_normalized_elite_interval():
    configspace = ConfigurationSpace(space={"x": (0.0, 1.0), "category": ["a", "b"], "fixed": 3})
    configurations = [
        Configuration(configspace, values={"x": value, "category": category, "fixed": 3})
        for value, category in [(0.0, "a"), (0.25, "a"), (0.75, "b"), (1.0, "b")]
    ]
    matrix = normalized_configuration_matrix(configurations, configspace)
    rows = {row["parameter"]: row for row in parameter_spreads(matrix, [0, 1], configspace)}

    assert rows["x"]["near_optimal_spread"] == pytest.approx(0.125)
    assert rows["x"]["overall_spread"] == pytest.approx(0.625)
    assert rows["x"]["near_optimal_min"] == pytest.approx(0.0)
    assert rows["x"]["near_optimal_median"] == pytest.approx(0.125)
    assert rows["x"]["near_optimal_max"] == pytest.approx(0.25)
    assert rows["x"]["near_optimal_coverage"] == pytest.approx(0.25)
    assert rows["category"]["near_optimal_coverage"] == pytest.approx(0.5)
    assert rows["category"]["near_optimal_spread"] == 0
    assert rows["category"]["overall_spread"] == pytest.approx(1)
    assert rows["fixed"]["relative_spread"] == 0


def test_parameter_coverage_uses_log_normalized_search_space():
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("rate", lower=1e-4, upper=1.0, log=True))
    configurations = [
        Configuration(configspace, values={"rate": value}) for value in (1e-4, 1e-2, 1.0)
    ]
    matrix = normalized_configuration_matrix(configurations, configspace)

    row = parameter_spreads(matrix, [0, 1], configspace)[0]

    assert row["near_optimal_min"] == pytest.approx(0.0)
    assert row["near_optimal_median"] == pytest.approx(0.25)
    assert row["near_optimal_max"] == pytest.approx(0.5)
    assert row["near_optimal_coverage"] == pytest.approx(0.5)


def test_random_baseline_is_reproducible_and_uses_equal_subset_size():
    condensed = np.linspace(0.0, 1.0, 45, dtype=np.float32)
    first = random_subset_baseline(condensed, 10, 4, repeats=20, seed=7)
    second = random_subset_baseline(condensed, 10, 4, repeats=20, seed=7)
    assert first == second
    assert first["lower"] <= first["median"] <= first["upper"]


def test_hierarchical_clustering_returns_stable_cluster_order():
    matrix = np.asarray(
        [
            [0, 0.1, 0.9, 0.9],
            [0.1, 0, 0.9, 0.9],
            [0.9, 0.9, 0, 0.1],
            [0.9, 0.9, 0.1, 0],
        ]
    )
    condensed = matrix[np.triu_indices(4, 1)]
    result = cluster_configurations(condensed, 4, requested_clusters=2)

    assert result["n_clusters"] == 2
    assert result["labels"][0] == result["labels"][1]
    assert result["labels"][2] == result["labels"][3]
    assert result["labels"][0] != result["labels"][2]
