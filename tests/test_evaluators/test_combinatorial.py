"""Exact mixed-level combinatorial coverage and whole-study random calibration."""

import json
from copy import deepcopy
from itertools import combinations, product

import numpy as np
import pytest
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    ForbiddenEqualsClause,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.evaluators import combinatorial as ccm
from deepcave.evaluators.coverage import normalize_configurations, run_coverage_analysis


def binary_data(values):
    space = ConfigurationSpace()
    space.add([CategoricalHyperparameter(name, [0, 1]) for name in "xyz"])
    return normalize_configurations(space, [dict(zip("xyz", row)) for row in values])


def test_strength_requires_every_group_and_every_combination():
    data = binary_data([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)] * 2)
    result = ccm.compute_combinatorial_coverage(data)
    for row in result["rows"]:
        assert row["observed_strength"] == 2
        assert row["trial_bound"] == 3
        assert not row["strength_is_lower_bound"]
        triples = row["cells"][2]
        assert triples["coverage"] == 0.5
        assert triples["complete_groups"] == 0
        assert triples["missing"] == 4
        assert triples["deficits"][0]["parameters"] == ["x", "y", "z"]
    complete = ccm.compute_combinatorial_coverage(binary_data(list(product([0, 1], repeat=3))))
    assert all(row["observed_strength"] == 3 for row in complete["rows"])
    json.dumps(result, allow_nan=False)


def test_missing_marginal_has_strength_zero_and_trial_bound_is_separate():
    result = ccm.compute_combinatorial_coverage(binary_data([(0, 0, 0), (0, 1, 1)]))
    for row in result["rows"]:
        assert row["observed_strength"] == 0
        assert row["trial_bound"] == 1
        cell = row["cells"][0]
        assert cell["coverage"] == pytest.approx(5 / 6)
        assert cell["fully_covered_rate"] == pytest.approx(2 / 3)
        assert cell["deficits"][0]["parameters"] == ["x"]
        assert len(row["cells"]) == 1

    complete_marginals = ccm.compute_combinatorial_coverage(binary_data([(0, 0, 0), (1, 1, 1)]))
    for row in complete_marginals["rows"]:
        assert row["observed_strength"] == 1
        assert row["cells"][-1]["status"] == "trial_limit"
        assert row["cells"][-1]["required_trials"] == 4
        assert not row["strength_is_lower_bound"]


def test_mixed_levels_weight_denominator_and_exclude_constants():
    space = ConfigurationSpace()
    space.add(
        [
            CategoricalHyperparameter("category", ["a", "b", "c"], weights=[8, 1, 1]),
            UniformFloatHyperparameter("number", 0, 1),
            Constant("fixed", 4),
        ]
    )
    data = normalize_configurations(space, [{"category": "a", "number": 0.1, "fixed": 4}] * 8)
    result = ccm.compute_combinatorial_coverage(data, minimum_strengths={2: 2})
    row = result["rows"][0]
    assert result["parameters"] == ["category", "number"]
    assert result["excluded_constants"] == 1
    assert row["levels"] == {"category": 3, "number": 2}
    assert row["cells"][0]["coverage"] == 2 / 5
    assert row["cells"][1]["coverage"] == 1 / 6
    assert row["cells"][1]["total"] == 6


def test_deficits_rank_missing_counts_before_coverage_fraction():
    space = ConfigurationSpace()
    space.add(
        [
            CategoricalHyperparameter(name, list(range(size)))
            for name, size in zip("abcd", [2, 3, 5, 6])
        ]
    )
    configs = [
        dict(zip("abcd", [0, 0, c, d])) for c, d in list(product(range(5), range(6)))[:20]
    ] * 10
    data = normalize_configurations(space, configs)
    pairs = ccm.compute_combinatorial_coverage(data, minimum_strengths={2: 2})["rows"][0]["cells"][
        1
    ]
    assert pairs["deficits"][0]["parameters"] == ["b", "d"]
    assert pairs["deficits"][0]["missing"] == 12
    assert [group["missing"] for group in pairs["deficits"]] == sorted(
        [group["missing"] for group in pairs["deficits"]], reverse=True
    )


def test_levels_follow_definition_including_log_steps_and_endpoints():
    space = ConfigurationSpace()
    space.add(
        [
            UniformFloatHyperparameter("log", 0.001, 1, log=True),
            UniformIntegerHyperparameter("integer", 0, 4),
            UniformIntegerHyperparameter("step", 0, 8, meta={"step": 2}),
            UniformIntegerHyperparameter("weighted", 0, 3, meta={"probabilities": [7, 1, 1, 1]}),
        ]
    )
    data = normalize_configurations(
        space,
        [
            {"log": 0.001, "integer": 0, "step": 0, "weighted": 0},
            {"log": 0.01, "integer": 2, "step": 4, "weighted": 1},
            {"log": 1, "integer": 4, "step": 8, "weighted": 3},
        ],
    )
    points, sizes = ccm.discretize_combinatorial(data, 2)
    columns = dict(zip([axis.name for axis in data.axes], points.T.tolist()))
    assert sizes == [2, 2, 2, 2]
    assert columns["log"] == [0, 0, 1]
    assert columns["integer"] == columns["step"] == [0, 1, 1]
    assert columns["weighted"] == [0, 1, 1]
    _, sizes = ccm.discretize_combinatorial(data, 8)
    assert dict(zip([axis.name for axis in data.axes], sizes)) == {
        "integer": 5,
        "log": 8,
        "step": 5,
        "weighted": 4,
    }


def test_exact_counts_match_cartesian_reference():
    rng = np.random.RandomState(17)
    space = ConfigurationSpace()
    sizes, names = [2, 3, 4, 5], list("abcd")
    space.add(
        [CategoricalHyperparameter(name, list(range(size))) for name, size in zip(names, sizes)]
    )
    configs = [dict(zip(names, [rng.randint(size) for size in sizes])) for _ in range(120)]
    result = ccm.compute_combinatorial_coverage(normalize_configurations(space, configs))
    for cell in result["rows"][0]["cells"]:
        covered, total, complete = 0, 0, 0
        for group in combinations(names, cell["strength"]):
            possible = set(product(*[range(sizes[names.index(name)]) for name in group]))
            observed = {tuple(config[name] for name in group) for config in configs}
            covered += len(observed & possible)
            total += len(possible)
            complete += observed == possible
        assert cell["covered"] == covered
        assert cell["total"] == total
        assert cell["complete_groups"] == complete


@pytest.mark.parametrize("constraint", ["condition", "forbidden"])
def test_constraints_are_explicitly_unsupported(constraint):
    space = ConfigurationSpace()
    parent = CategoricalHyperparameter("parent", [0, 1])
    child = UniformFloatHyperparameter("child", 0, 1)
    space.add([parent, child])
    space.add(
        EqualsCondition(child, parent, 1)
        if constraint == "condition"
        else ForbiddenEqualsClause(parent, 1)
    )
    with pytest.raises(ValueError, match="independent"):
        ccm.compute_combinatorial_coverage(normalize_configurations(space, [{"parent": 0}]))


def test_empty_and_constant_only_spaces_are_rejected():
    with pytest.raises(ValueError, match="sampled trials"):
        ccm.compute_combinatorial_coverage(binary_data([]))
    space = ConfigurationSpace()
    space.add(Constant("fixed", 1))
    with pytest.raises(ValueError, match="varying"):
        ccm.compute_combinatorial_coverage(normalize_configurations(space, [{"fixed": 1}]))


def test_work_limit_cannot_claim_exact_strength(monkeypatch):
    monkeypatch.setattr(ccm, "MAX_GROUPS", 3)
    for row in ccm.compute_combinatorial_coverage(binary_data(list(product([0, 1], repeat=3))))[
        "rows"
    ]:
        assert row["observed_strength"] == 1
        assert row["strength_is_lower_bound"]
        assert row["cells"][1]["status"] == "work_limit"


def test_strength_above_six_is_evaluated_through_the_parameter_count():
    space = ConfigurationSpace()
    space.add([CategoricalHyperparameter(f"p{i}", [0, 1]) for i in range(7)])
    data = normalize_configurations(
        space, [dict(zip(space.keys(), row)) for row in product([0, 1], repeat=7)]
    )
    result = ccm.compute_combinatorial_coverage(data)
    assert result["max_strength"] == 7
    for row in result["rows"]:
        assert row["observed_strength"] == 7
        assert row["trial_bound"] == 7
        assert not row["strength_is_lower_bound"]
        assert row["cells"][-1]["coverage"] == 1


def test_work_limit_stops_before_later_smaller_groups(monkeypatch):
    space = ConfigurationSpace()
    space.add([CategoricalHyperparameter(f"p{i}", [0, 1]) for i in range(7)])
    data = normalize_configurations(
        space, [{name: value for name in space.keys()} for value in [0, 1]] * 64
    )
    # Singles and six-way groups fit, but the pair count exceeds the work limit.
    monkeypatch.setattr(ccm, "MAX_GROUPS", 15)
    row = ccm.compute_combinatorial_coverage(data)["rows"][0]
    assert row["cells"][1]["status"] == "work_limit"
    assert len(row["cells"]) == 2
    assert row["observed_strength"] == 1
    assert row["strength_is_lower_bound"]


def test_strength_seven_and_next_strength_coverage_are_measured():
    space = ConfigurationSpace()
    space.add([CategoricalHyperparameter(f"p{i}", [0, 1]) for i in range(8)])
    # Even parity covers all seven-way combinations and half the eight-way tuples.
    configs = [
        dict(zip(space.keys(), (*values, sum(values) % 2))) for values in product([0, 1], repeat=7)
    ] * 2
    result = ccm.compute_combinatorial_coverage(normalize_configurations(space, configs))
    assert result["max_strength"] == 8
    for row in result["rows"]:
        assert row["observed_strength"] == 7
        assert row["trial_bound"] == 8
        assert not row["strength_is_lower_bound"]
        assert row["cells"][-1]["strength"] == 8
        assert row["cells"][-1]["coverage"] == 0.5


def test_evaluation_stops_at_first_gap_despite_higher_trial_bound(monkeypatch):
    space = ConfigurationSpace()
    space.add([CategoricalHyperparameter(f"p{i}", [0, 1]) for i in range(8)])
    configs = [dict.fromkeys(space.keys(), value) for value in [0, 1]] * 128
    strengths = []
    count_groups = ccm._group_counts

    def record(points, sizes, groups):
        strengths.append(len(groups[0]))
        return count_groups(points, sizes, groups)

    monkeypatch.setattr(ccm, "_group_counts", record)
    result = ccm.compute_combinatorial_coverage(normalize_configurations(space, configs))
    assert strengths == [1, 2] * 4
    assert result["max_strength"] == 2
    for row in result["rows"]:
        assert row["trial_bound"] == 8
        assert row["observed_strength"] == 1
        assert row["cells"][-1]["coverage"] == 0.5
        assert not row["strength_is_lower_bound"]


def test_random_comparison_is_reproducible_and_preserves_search_space_rng():
    data = binary_data([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)] * 2)
    untouched = deepcopy(data.configspace)
    options = {"designs": 5, "seed": 13}
    result = run_coverage_analysis("combinatorial_baseline", data, data.configspace, options)
    assert result == run_coverage_analysis(
        "combinatorial_baseline", data, data.configspace, options
    )
    assert data.configspace.sample_configuration() == untouched.sample_configuration()
    for cell in result["cells"]:
        assert cell["deficit"] == pytest.approx(cell["observed"] - cell["mean"])
        assert cell["lower"] <= cell["median"] <= cell["upper"]
        if cell["degenerate"]:
            assert cell["percentile"] is None
        else:
            assert 0 <= cell["percentile"] <= 100
    assert result["baseline_version"] == ccm.BASELINE_VERSION
    for row in result["strengths"]:
        assert len(row["samples"]) == 5
        assert row["median"] == np.median(row["samples"])
    assert run_coverage_analysis("combinatorial", data, data.configspace, {}) == result["observed"]
    json.dumps(result, allow_nan=False)


def test_random_reference_medians_and_strengths_come_from_whole_designs(monkeypatch):
    observed = binary_data([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)] * 2)
    references = iter(
        [
            binary_data(list(product([0, 1], repeat=3))),
            observed,
            binary_data([(0, 0, 0)] * 8),
        ]
    )
    monkeypatch.setattr(ccm, "sample_reference_design", lambda *args: next(references))
    result = ccm.compute_combinatorial_baseline(observed, observed.configspace, designs=3)
    for row in result["strengths"]:
        assert row["samples"] == [3, 2, 0]
        assert row["median"] == 2
        assert row["lower"] == 0 and row["upper"] == 3
        assert row["percentile"] == 50
        assert not row["censored"]
    triples = next(
        cell for cell in result["cells"] if cell["resolution"] == 2 and cell["strength"] == 3
    )
    assert triples["median"] == 0.5
    assert triples["mean"] == pytest.approx((1 + 0.5 + 0.125) / 3)


def test_constant_references_and_censored_strengths_have_no_precise_rank(monkeypatch):
    data = binary_data(list(product([0, 1], repeat=3)))
    monkeypatch.setattr(ccm, "sample_reference_design", lambda *args: data)
    result = ccm.compute_combinatorial_baseline(data, data.configspace, designs=3)
    assert all(cell["degenerate"] and cell["percentile"] is None for cell in result["cells"])
    assert all(row["degenerate"] and row["percentile"] is None for row in result["strengths"])
    monkeypatch.setattr(ccm, "MAX_GROUPS", 0)
    result = ccm.compute_combinatorial_baseline(data, data.configspace, designs=3)
    assert all(row["censored"] and row["percentile"] is None for row in result["strengths"])
