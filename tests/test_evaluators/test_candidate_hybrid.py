"""Geometry, legal shell supply and model-independent hybrid selection contracts."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import (
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.evaluators import candidate_geometry as geometry
from deepcave.evaluators import coverage_candidates as scoring
from deepcave.evaluators.candidate_selection import (
    configuration_key,
    rank_candidates,
    select_hybrid_candidates,
    select_ranked_hybrid,
)
from deepcave.evaluators.rashomon import pairwise_configuration_distances
from deepcave.runs.status import Status
from deepcave.utils.configspace import normalized_configuration_matrix

from tests.test_evaluators.test_coverage_candidates import OPTIONS, make_run


@pytest.mark.parametrize("dimensions", [1, 2, 7])
def test_normalized_rms_and_constants(dimensions):
    cs = ConfigurationSpace()
    cs.add([UniformFloatHyperparameter(f"p{i}", 10, 20) for i in range(dimensions)])
    cs.add(Constant("fixed", 42))
    reference = {**{f"p{i}": 10 for i in range(dimensions)}, "fixed": 42}
    query = {**reference, "p0": 20}
    assert geometry.incumbent_distances(cs, reference, [query])[0] == pytest.approx(
        1 / np.sqrt(dimensions)
    )
    matrix = normalized_configuration_matrix([reference, query], cs)
    assert geometry.incumbent_distances(cs, reference, [query])[0] == pytest.approx(
        pairwise_configuration_distances(matrix, cs)[0]
    )


def test_log_coordinates_and_equal_width_boundaries():
    cs = ConfigurationSpace()
    cs.add(UniformFloatHyperparameter("rate", 0.001, 1, log=True))
    assert geometry.incumbent_distances(cs, {"rate": 0.001}, [{"rate": 0.01}])[0] == pytest.approx(
        1 / 3
    )
    distances = np.array([0, 0.099, 0.1, 0.2, 0.3, 0.4, 0.5, 0.99])
    assert geometry.distance_bins(distances, 0.5).tolist() == [1, 1, 2, 3, 4, 5, 5, 5]
    assert geometry.distance_bins(np.array([0, 0.1]), 0).tolist() == [5, 5]


@pytest.mark.parametrize("maximize", [True, False])
def test_elite_is_ceil_ten_percent_of_finite_successes_and_matches_incumbent(maximize):
    run = make_run(maximize)
    run.history[0].status = Status.FAILED
    run.history[1].costs[0] = float("nan")
    result = geometry.historical_geometry(run, 0)
    valid = run.history[2:]
    expected = sorted(valid, key=lambda t: (-1 if maximize else 1) * t.costs[0])[:6]
    assert result["elite_trials"] == [t.additional["optuna_trial_number"] for t in expected]
    assert result["incumbent"] == run.get_config(expected[0].config_id)
    assert result["radius"] == max(result["elite_distances"])
    assert result["elite_distances"][0] == 0


def test_supply_reaches_one_thousand_per_bin_without_duplicates_and_is_deterministic():
    run = make_run()
    initial, _ = scoring.generate_full_space_candidates(run, OPTIONS)
    geo = geometry.historical_geometry(run, 0)
    result = scoring.augment_full_space_candidates(run, initial, geo, 17)
    assert result["supply"]["ready"]
    assert all(b["count"] >= 1000 for b in result["supply"]["bins"])
    assert [c["configuration"] for c in result["candidates"][: len(initial)]] == [
        c["configuration"] for c in initial
    ]
    keys = [configuration_key(c["configuration"]) for c in result["candidates"]]
    assert len(set(keys)) == len(keys)
    assert not set(keys) & {configuration_key(run.get_config(t.config_id)) for t in run.history}
    distances = geometry.incumbent_distances(
        run.configspace, geo["incumbent"], [c["configuration"] for c in result["candidates"]]
    )
    np.testing.assert_array_equal(
        geometry.distance_bins(distances, geo["radius"]), [c["bin"] for c in result["candidates"]]
    )
    np.testing.assert_allclose(
        distances, [c["distance"] for c in result["candidates"]], atol=0, rtol=0
    )
    assert result == scoring.augment_full_space_candidates(run, initial, geo, 17)
    # Generation does not read historical objective values once geometry is defined.
    for trial in run.history:
        trial.costs = [float("nan"), None]
    assert result == scoring.augment_full_space_candidates(run, initial, geo, 17)


def test_radial_sampling_obeys_log_integer_and_float_grids_forbidden_and_bounds():
    run = make_run()
    run.configspace = ConfigurationSpace()
    run.configspace.add(
        [
            UniformFloatHyperparameter("a", 0, 1, meta={"step": 0.2}),
            UniformIntegerHyperparameter("b", 1, 99, log=True),
            UniformIntegerHyperparameter("grid", 2, 20, meta={"step": 2}),
            UniformFloatHyperparameter("c", 0.001, 1, log=True),
            Constant("constant", "fixed"),
        ]
    )
    run.configspace.add(ForbiddenEqualsClause(run.configspace["b"], 3))
    reference = {"a": 0, "b": 1, "c": 0.001, "constant": "fixed", "grid": 2}
    proposals = geometry.radial_configurations(
        run.configspace, reference, 0, 0.5, np.random.RandomState(12), 2000
    )
    assert len(proposals) > 100
    for config in proposals:
        scoring._validate_config(run, config, scoring.numeric_space(run))
        assert config["b"] != 3
        assert config["constant"] == "fixed"
    # A too-distant shell cannot be made valid by clipping to a box face.
    assert (
        geometry.radial_configurations(
            run.configspace, reference, 1.1, 1.2, np.random.RandomState(12), 100
        )
        == []
    )


def test_finite_and_degenerate_supply_fail_explicitly():
    run = make_run()
    run.configspace = ConfigurationSpace(space={"n": (1, 3)})
    run.history = run.history[:1]
    run.get_config = lambda i: {"n": 1}
    initial, _ = scoring.generate_full_space_candidates(run, OPTIONS)
    for radius, reason in [(0.5, "Finite legal"), (0, "R=0")]:
        result = scoring.augment_full_space_candidates(
            run, initial, {"incumbent": {"n": 1}, "radius": radius}, 17
        )
        assert not result["supply"]["ready"]
        assert len(result["candidates"]) == 2
        assert any(reason in b["reason"] for b in result["supply"]["bins"] if b["reason"])
        with pytest.raises(ValueError, match="Insufficient candidate supply"):
            select_hybrid_candidates([dict(c, kde_score=0) for c in result["candidates"]], 1)


def test_restricted_shell_stops_at_sampling_budget(monkeypatch):
    run = make_run()
    monkeypatch.setattr(scoring, "BIN_SAMPLING_BUDGET", 12)
    monkeypatch.setattr(scoring, "radial_configurations", lambda *args: [])
    result = scoring.augment_full_space_candidates(
        run, [], geometry.historical_geometry(run, 0), 17
    )
    assert not result["supply"]["ready"]
    assert all(
        b["attempts"] == 12 and "budget reached" in b["reason"] for b in result["supply"]["bins"]
    )


@pytest.mark.parametrize("dimensions", [1, 4, 12])
def test_supply_for_different_dimensions_and_boundary_incumbents(dimensions):
    run = make_run()
    run.configspace = ConfigurationSpace()
    run.configspace.add(
        [UniformFloatHyperparameter(f"coordinate_{i}", 0, 1) for i in range(dimensions)]
    )
    reference = {name: 0.0 for name in run.configspace}
    run.get_config = lambda i: reference
    initial, _ = scoring.generate_full_space_candidates(run, OPTIONS)
    result = scoring.augment_full_space_candidates(
        run, initial, {"incumbent": reference, "radius": 0.4}, 42
    )
    assert result["supply"]["ready"]
    assert all(b["count"] >= 1000 for b in result["supply"]["bins"])


def synthetic_pool():
    return [
        dict(id=f"{i:05}", configuration={"arbitrary_name": i}, kde_score=-float(i), bin=i % 5 + 1)
        for i in range(5000)
    ]


@pytest.mark.parametrize("budget", [1, 10, 20, 27, 40, 5000])
def test_hybrid_split_global_optimum_round_robin_and_exact_unique_budget(budget):
    pool = synthetic_pool()
    selected = select_hybrid_candidates(pool, budget)
    global_count = (budget + 1) // 2
    assert len(selected) == len({configuration_key(c["configuration"]) for c in selected}) == budget
    assert [c["id"] for c in selected[:global_count]] == [c["id"] for c in pool[:global_count]]
    assert all(c["selection"] == "Global KDE" for c in selected[:global_count])
    assert all(c["selection"] == "Exploration" for c in selected[global_count:])
    if budget <= 40:
        assert [c["bin"] for c in selected[global_count:]] == [
            i % 5 + 1 for i in range(budget - global_count)
        ]
    assert [c["selection_rank"] for c in selected] == list(range(1, budget + 1))


def test_ties_deduplication_and_insufficient_budget():
    pool = synthetic_pool()
    for c in pool:
        c["kde_score"] = 1
    expected = select_hybrid_candidates(pool, 27)
    assert select_hybrid_candidates(list(reversed(pool)) + [pool[0]], 27) == expected
    assert [c["kde_rank"] for c in rank_candidates(pool)] == list(range(1, 5001))
    with pytest.raises(ValueError, match="only 5000 unique"):
        select_hybrid_candidates(pool, 5001)


def test_frozen_catboost_pool_reproduces_archived_14_13_selection_and_metrics():
    folder = Path(__file__).parent / "fixtures" / "candidate_hybrid"
    with gzip.open(folder / "cat38.json.gz", "rt") as stream:
        frozen = json.load(stream)
    selected = select_hybrid_candidates(frozen["pool"], 27)
    assert [c["id"] for c in selected] == frozen["expected_ids"]
    assert sum(c["selection"] == "Global KDE" for c in selected) == 14
    assert sum(c["configuration"]["depth"] == 12 for c in selected) == 4
    assert sum(c["configuration"]["depth"] == 15 for c in selected) == 16
    assert np.mean([c["kde_score"] for c in selected]) == pytest.approx(
        5.0177401959435866, abs=1e-12
    )
    cs = ConfigurationSpace.from_serialized_dict(frozen["configspace"])
    matrix = normalized_configuration_matrix([c["configuration"] for c in selected], cs)
    assert pairwise_configuration_distances(matrix, cs).astype(float).mean() == pytest.approx(
        0.27532818453370506, abs=1e-9
    )


def test_frozen_xgboost_proposal_arms_reproduce_retrospective_splits():
    path = Path(__file__).parent / "fixtures" / "candidate_hybrid" / "xgb.json.gz"
    with gzip.open(path, "rt") as stream:
        studies = json.load(stream)
    assert {s["study_id"] for s in studies} == {30, 32}
    for study in studies:
        assert study["cases"]
        for case in study["cases"]:
            selected = select_ranked_hybrid(
                study["global_rows"], study["exploration_rows"], case["n_global"], case["budget"]
            )
            assert [c["id"] for c in selected] == case["expected_ids"]


def test_round_robin_skips_exhausted_bins_and_detects_incomplete_arms():
    pool = synthetic_pool()[:10]
    exploration = [pool[0], pool[4], pool[5]]
    selected = select_ranked_hybrid(pool, exploration, 2, 4)
    assert [c["id"] for c in selected] == ["00000", "00001", "00005", "00004"]
    with pytest.raises(ValueError, match="Exploration pool exhausted"):
        select_ranked_hybrid(pool, exploration, 2, 5)
