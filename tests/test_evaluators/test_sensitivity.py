"""Synthetic Optuna designs exercise seed statistics independently of the interface."""

import copy
import pickle
from itertools import product

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

from deepcave.evaluators.sensitivity import (
    available_modes,
    bootstrap_means,
    build_groups,
    canonical,
    elite_subset,
    fold_attributes,
    fold_diagnostics,
    pairwise,
    pareto_ids,
    prepare_study,
    ranking,
    randomness_ranges,
    resolve_attribute,
    seed_design,
    summary,
    variance_decomposition,
)
from deepcave.runs.converters.optuna import OptunaRun

MAPPINGS = {"model": "trial:network_rng", "data": "trial:split_rng"}
CONTEXT = {
    "experiment_id": "exp",
    "model_type": "xgb",
    "dataset": "cohort-A",
    "feature_set": ["a", "b"],
    "cv_strategy": "stratified-5",
    "objective_definition": "auc-v2",
}


@pytest.fixture
def make_run(tmp_path):
    def make(records, name="study", direction="maximize", context=None):
        study = optuna.create_study(study_name=name, direction=direction)
        for key, value in (context or {}).items():
            study.set_user_attr(key, value)
        for item in records:
            attrs = {
                "network_rng": item.get("m", 0),
                "split_rng": item.get("d", 0),
                **item.get("attrs", {}),
            }
            for key in item.get("missing", []):
                attrs.pop(key, None)
            parameters = {
                "rate": item.get("x", 0.25),
                "network_rng": item.get("m", 0),
                "split_rng": item.get("d", 0),
            }
            distributions = {
                "rate": FloatDistribution(0.00001, 1, log=True),
                "network_rng": IntDistribution(0, 100),
                "split_rng": IntDistribution(0, 100),
            }
            state = item.get("state", TrialState.COMPLETE)
            study.add_trial(
                create_trial(
                    params=parameters,
                    distributions=distributions,
                    value=item.get("y", 1.0) if state == TrialState.COMPLETE else None,
                    state=state,
                    user_attrs=attrs,
                )
            )
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        with (path / "study.pkl").open("wb") as file:
            pickle.dump(study, file)
        return OptunaRun.from_path(path)

    return make


def prepared(make_run, records, **kwargs):
    return prepare_study(make_run(records, **kwargs))


def groups(data, mode="model", mappings=None):
    return build_groups(data, mappings or MAPPINGS, mode, "metric:Objective0")["groups"]


def test_mapping_modes_and_scope():
    assert available_modes({}) == []
    assert available_modes({"data": "trial:custom"}) == ["data"]
    assert available_modes(MAPPINGS) == ["model", "data", "combined"]
    assert resolve_attribute("study:rng", {"rng": "A"}, {"rng": "B"}) == "A"
    assert resolve_attribute("trial:rng", {}, {"rng": 7}) == 7
    for invalid in (None, True, [], {}, float("nan"), float("inf")):
        assert resolve_attribute("trial:rng", {}, {"rng": invalid}) is None


@pytest.mark.parametrize("mode", ["model", "data"])
def test_single_seed_group_statistics_and_direction(make_run, mode):
    records = [
        {"m" if mode == "model" else "d": i, "y": value}
        for i, value in enumerate([1.0, 2.0, 4.0, 9.0])
    ]
    for direction in ("maximize", "minimize"):
        data = prepared(make_run, records, name=direction, direction=direction)
        mappings = {mode: MAPPINGS[mode]}
        # The unassigned seed parameter is constant and remains a real parameter.
        result = groups(data, mode, mappings)
        assert len(result) == 1
        group = result[0]
        assert group["seeds"] == group["evaluations"] == 4
        assert group["mean"] == 4
        assert group["median"] == 3
        assert group["sd"] == pytest.approx(np.std([1, 2, 4, 9], ddof=1))
        assert group["iqr"] == pytest.approx(3.5)
        assert group["best_gap"] == (5 if direction == "maximize" else 3)
        assert group["minimum"] == 1 and group["maximum"] == 9


def test_conditional_modes_do_not_pool_other_seed(make_run):
    data = prepared(
        make_run, [{"m": m, "d": d, "y": m + 100 * d} for m, d in product(range(3), range(2))]
    )
    model = groups(data)
    assert len(model) == 2
    assert [g["sd"] for g in model] == pytest.approx([1, 1])
    assert all(g["seeds"] == 3 for g in model)
    assert len(groups(data, "data")) == 3
    combined = groups(data, "combined")[0]
    assert combined["seeds"] == 6 and combined["sd"] > 50
    assert seed_design(combined)["complete"]
    assert summary(model)["configurations"] == 1
    assert summary(model)["comparable"] == 1


def test_duplicate_seed_pairs_have_equal_pair_weight_and_raw_records(make_run):
    data = prepared(make_run, [{"m": 0, "y": 1}, {"m": 0, "y": 3}, {"m": 1, "y": 8}])
    group = groups(data)[0]
    assert group["evaluations"] == 3 and group["seeds"] == 2 and group["duplicates"] == 1
    assert group["mean"] == 5
    assert group["sd"] == pytest.approx(np.std([2, 8], ddof=1))
    assert len(group["rows"]) == 3


@pytest.mark.parametrize(
    "records, expected",
    [
        (
            [(0, 0, 0), (0, 1, 10), (1, 0, 20), (1, 1, 30)],
            {
                "data_seed": {0: (0, 10, 10), 1: (20, 30, 10)},
                "model_seed": {0: (0, 20, 20), 1: (10, 30, 20)},
            },
        ),
        (
            [(0, 0, 0), (0, 0, 4), (0, 1, 10), (1, 0, 20)],
            {
                "data_seed": {0: (2, 10, 8)},
                "model_seed": {0: (2, 20, 18)},
            },
        ),
    ],
)
def test_randomness_ranges_use_pair_means_and_fixed_seeds(make_run, records, expected):
    design = groups(
        prepared(
            make_run,
            [
                {"x": x, "m": m, "d": d, "y": value + offset}
                for x, offset in ((0.25, 0), (0.5, 100))
                for m, d, value in records
            ],
        ),
        "combined",
    )
    assert len(design) == 2
    for group, offset in zip(design, (0, 100)):
        for axis, values in expected.items():
            observations = randomness_ranges(group, axis)
            assert {item["fixed_seed"] for item in observations} == set(values)
            for item in observations:
                minimum, maximum, span = values[item["fixed_seed"]]
                assert item["minimum"] == pytest.approx(minimum + offset)
                assert item["maximum"] == pytest.approx(maximum + offset)
                assert item["range"] == pytest.approx(span)


def test_randomness_ranges_require_two_distinct_known_seeds():
    pairs = [
        {"objective": 2, "data_seed": None, "model_seed": 0},
        {"objective": 4, "data_seed": 0, "model_seed": None},
        {"objective": 6, "data_seed": 1, "model_seed": 1},
    ]
    for axis in ("data_seed", "model_seed"):
        assert randomness_ranges({"replicates": pairs}, axis) == []
        assert randomness_ranges({"replicates": pairs[2:] * 2}, axis) == []
        assert randomness_ranges({"replicates": []}, axis) == []
    # Equal objectives still produce an observed zero range if distinct seeds exist.
    pairs = [
        {"objective": 2, "data_seed": 0, "model_seed": 0},
        {"objective": 2, "data_seed": 1, "model_seed": 0},
    ]
    assert randomness_ranges({"replicates": pairs}, "data_seed") == [
        {"fixed_seed": 0, "minimum": 2, "maximum": 2, "range": 0}
    ]


def test_typed_canonical_values_and_unrounded_original_optuna_params(make_run):
    assert len({canonical(value) for value in (1, 1.0, True, "1", None)}) == 5
    assert canonical({"a": 2, "b": 1}) == canonical({"b": 1, "a": 2})
    data = prepared(make_run, [{"x": 0.1, "m": 1}, {"x": np.nextafter(0.1, 1), "m": 2}])
    assert len(groups(data)) == 2
    for row in data["rows"]:
        row["user_attrs"]["config_id"] = "same-user-label"
    assert len(groups(data)) == 2


def test_cross_study_merging_requires_full_matching_context_and_avoids_id_collisions(make_run):
    first = prepared(make_run, [{"m": 1}], name="one", context=CONTEXT)
    second = prepared(make_run, [{"m": 2}], name="two", context=CONTEXT)
    first["rows"].extend(second["rows"])
    result = groups(first)
    assert len(result) == 1 and result[0]["seeds"] == 2
    assert len({r["id"] for r in result[0]["rows"]}) == 2
    assert {r["trial_number"] for r in result[0]["rows"]} == {0}
    changed = copy.deepcopy(first)
    changed["rows"][1]["study_attrs"]["dataset"] = "other-cohort"
    assert len(groups(changed)) == 2
    for row in first["rows"]:
        row["study_attrs"] = {"experiment_id": "same-label"}
    assert len(groups(first)) == 2


def test_missing_values_unfinished_and_single_repetitions(make_run):
    data = prepared(
        make_run,
        [
            {"m": 0},
            {"m": 1, "missing": ["split_rng"]},
            {"m": 2, "state": TrialState.FAIL},
            {"m": 3, "y": 2},
        ],
    )
    data["rows"][-1]["metrics"]["metric:Objective0"] = float("inf")
    result = build_groups(data, MAPPINGS, "model", "metric:Objective0")
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["sd"] is None and group["seeds"] == 1
    assert sum(result["excluded"].values()) == 3
    assert summary([group])["comparable"] == 0
    assert pareto_ids([group], True) == []
    assert "reason" in bootstrap_means([group])


def test_incomplete_and_fully_coupled_seed_designs(make_run):
    data = prepared(make_run, [{"m": m, "d": d} for m, d in ((0, 0), (0, 1), (1, 1))])
    design = seed_design(groups(data, "combined")[0])
    assert design["crossed"] and not design["complete"]
    assert (design["observed"], design["possible"]) == (3, 4)
    data = prepared(make_run, [{"m": i, "d": i} for i in range(4)], name="coupled")
    group = groups(data, "combined")[0]
    assert not seed_design(group)["crossed"]
    assert "reason" in variance_decomposition([group])


def test_pareto_frontier_directions_and_ties():
    rows = [
        {"id": str(i), "mean": mean, "sd": sd, "seeds": 5}
        for i, (mean, sd) in enumerate([(3, 1), (2, 2), (4, 3), (3, 1), (1, 0.5)])
    ]
    assert set(pareto_ids(rows, True)) == {"0", "2", "3", "4"}
    assert pareto_ids(rows, False) == ["4"]


def paired_groups(make_run, offset=2):
    records = [
        {
            "x": x,
            "m": m,
            "y": m + (offset if x == 0.5 else 0),
            "attrs": {"cv_split_ids": ["partition-A", "partition-B"]},
        }
        for x, m in product((0.25, 0.5), range(6))
    ]
    return sorted(groups(prepared(make_run, records)), key=lambda g: g["mean"], reverse=True)


def test_paired_differences_bootstrap_and_partition_evidence(make_run):
    first, second = paired_groups(make_run)
    result = pairwise(first, second, 500)
    assert result["paired"] and result["difference"] == 2
    assert result["differences"] == [2] * 6
    assert result["interval"] == pytest.approx([2, 2])
    assert result["win_frequency"] == 1
    for rep in second["replicates"]:
        rep["partition"] = None
    assert not pairwise(first, second, 500)["paired"]
    for rep in second["replicates"]:
        rep["partition"] = "different-partitions"
    assert not pairwise(first, second, 500)["paired"]


def test_unpaired_superiority_and_insufficient_bootstrap_samples(make_run):
    first, second = paired_groups(make_run)
    for group in (first, second):
        for rep in group["replicates"]:
            rep["partition"] = None
    result = pairwise(first, second, 500)
    expected = np.mean(
        [a["objective"] > b["objective"] for a in first["replicates"] for b in second["replicates"]]
    )
    assert not result["paired"] and result["win_frequency"] == expected
    second["replicates"] = second["replicates"][:2]
    second["seeds"] = 2
    assert pairwise(first, second)["interval"] is None


def test_ranking_direction_and_tie_frequencies(make_run):
    first, second = paired_groups(make_run)
    result = ranking([first, second], 500)
    assert [row["mean_rank"] for row in result["rows"]] == [1, 2]
    assert result["rows"][0]["top1"] == 1
    for group in (first, second):
        group["direction"] = "lower"
    assert ranking([first, second], 500)["rows"][1]["top1"] == 1
    tied = [copy.deepcopy(first) for _ in range(4)]
    result = ranking(tied, 500)
    assert all(
        row["mean_rank"] == 2.5 and row["top1"] == 0.25 and row["top3"] == 0.75
        for row in result["rows"]
    )


def test_crossed_bootstrap_and_balanced_variance_components(make_run):
    records = [
        {"x": x, "m": m, "d": d, "y": x + m + 2 * d + repeat * 0.1}
        for x, m, d, repeat in product((0.25, 0.5), range(4), range(4), range(2))
    ]
    result = groups(prepared(make_run, records), "combined")
    bootstrap = bootstrap_means(result, 500)
    assert bootstrap["means"].shape == (500, 2)
    assert "crossed-factor" in bootstrap["method"]
    decomposition = variance_decomposition(result)
    assert len(decomposition["rows"]) == 8
    estimates = {r["component"]: r["estimate"] for r in decomposition["rows"]}
    assert estimates["Configuration"] == pytest.approx(np.var([0.25, 0.5], ddof=1))
    assert estimates["Model Seed"] == pytest.approx(np.var(range(4), ddof=1))
    assert estimates["Data Seed"] == pytest.approx(np.var([0, 2, 4, 6], ddof=1))
    assert estimates["Repeat residual"] == pytest.approx(0.005)
    assert sum(row["share"] for row in decomposition["rows"]) == pytest.approx(1)
    result[0]["rows"].pop()
    assert "reason" in variance_decomposition(result)


def test_two_factor_variance_and_elite_threshold_ties(make_run):
    records = [
        {"x": x, "m": m, "y": x + m + repeat * 0.1}
        for x, m, repeat in product((0.25, 0.5), range(4), range(2))
    ]
    result = groups(prepared(make_run, records))
    assert len(variance_decomposition(result)["rows"]) == 4
    assert len(elite_subset(result, 0.5)) == 1
    assert elite_subset(result, 1) == result
    result[1]["mean"] = result[0]["mean"]
    assert len(elite_subset(result, 0.1)) == 2


def test_fold_diagnostics_partition_and_missing_fold_rules(make_run):
    records = [
        {"m": m, "attrs": {"fold_auc": [m, m + 1], "cv_split_ids": ["a", "b"]}} for m in range(4)
    ]
    group = groups(prepared(make_run, records))[0]
    result = fold_diagnostics(group, "fold_auc")
    assert result["comparable"]
    assert result["summary"][0]["mean"] == 1.5
    assert result["summary"][0]["sd"] == pytest.approx(np.std(range(4), ddof=1))
    group["rows"][0]["user_attrs"]["fold_auc"].append(4)
    assert not fold_diagnostics(group, "fold_auc")["comparable"]
    assert len(fold_diagnostics(group, "fold_auc")["labels"]) == 3
    assert fold_attributes({"fold_metrics": {"auc": [1, None, float("inf")]}}) == {
        "fold_metrics / auc": {"1": 1.0, "2": None, "3": None}
    }
    assert fold_diagnostics(group, "missing")["rows"] == []
    for row in group["rows"]:
        row["partition"] = None
        row["data_seed"] = row["model_seed"]
    assert not fold_diagnostics(group, "fold_auc")["comparable"]


def test_conflicting_trial_context_and_duplicate_snapshot_rows(make_run):
    data = prepared(
        make_run, [{"m": 0}, {"m": 1, "attrs": {"dataset": "different"}}], context=CONTEXT
    )
    data["rows"].append(copy.deepcopy(data["rows"][0]))
    result = groups(data)
    assert len(result) == 2
    assert sum(g["evaluations"] for g in result) == 2
