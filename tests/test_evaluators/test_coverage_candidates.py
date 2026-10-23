"""Shared benchmark math, objective-free CHR pools and sequential virtual plans."""

from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.evaluators import coverage_candidates as scoring
from deepcave.evaluators import kde
from deepcave.evaluators.coverage import (
    _nearest_distances,
    normalize_configurations,
    prepare_coverage,
    sample_reference_design,
)
from deepcave.runs.status import Status


def make_run(maximize=True):
    rng = np.random.default_rng(8)
    configs = [{"x": float(rng.uniform(0.01, 1)), "n": int(rng.integers(1, 6))} for _ in range(60)]
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("x", 0.01, 1, log=True),
            UniformIntegerHyperparameter("n", 1, 5, meta={"step": 2}),
        ]
    )
    # Preserve the declared integer lattice in every historical observation.
    for c in configs:
        c["n"] = min(5, 1 + 2 * ((c["n"] - 1) // 2))
    objectives = [
        SimpleNamespace(name="AP", optimize="upper" if maximize else "lower"),
        SimpleNamespace(name="Loss", optimize="lower"),
    ]
    return SimpleNamespace(
        id="kde-study",
        name="KDE study",
        meta={},
        configspace=configspace,
        history=[
            SimpleNamespace(
                config_id=i,
                status=Status.SUCCESS,
                budget=0,
                seed=0,
                costs=[c["x"] + c["n"] / 20, c["x"]],
                additional={"optuna_trial_number": i + 100},
            )
            for i, c in enumerate(configs)
        ],
        get_config=lambda i: configs[i],
        get_objective=lambda i: objectives[i],
        get_objective_ids=lambda: [0, 1],
    )


OPTIONS = dict(
    pool_size=60,
    holes=4,
    plan_size=4,
    reference_samples=128,
    seed=42,
    global_pool_size=80,
    global_seed=17,
)


@pytest.mark.parametrize("maximize", [True, False])
def test_production_matches_benchmark_encoding_fit_split_and_reference(tmp_path, maximize):
    from tests.test_tpe_kde_benchmark import benchmark

    run = make_run(maximize)
    space = scoring.numeric_space(run)
    trials = [SimpleNamespace(params=run.get_config(i)) for i in range(len(run.history))]
    encoded, var_type, _ = benchmark.kde_encoding(trials, space)
    values = np.array([t.costs[0] for t in run.history])
    good, bad = benchmark.split_good_bad(values, np.arange(len(trials)), maximize)
    fitted = scoring.fit_candidate_kde(run, 0, str(tmp_path))
    assert fitted.metadata["good_trials"] == [int(i + 100) for i in good]
    assert fitted.metadata["bad_trials"] == [int(i + 100) for i in bad]
    assert fitted.metadata["reference_trial_number"] == int(good[0] + 100)
    assert fitted.metadata["reference_objective"] == values[good[0]]
    assert benchmark.kde_encoding is kde.kde_encoding
    assert benchmark.kde_logpdf is kde.kde_logpdf
    assert benchmark.loo_objective is kde.loo_objective
    for model, indices in ((fitted.good, good), (fitted.bad, bad)):
        reference, _ = benchmark.fit_kde(encoded[indices], var_type, space)
        np.testing.assert_array_equal(model.bw, reference.bw)
        np.testing.assert_array_equal(
            kde.kde_logpdf(model, encoded), benchmark.kde_logpdf(reference, encoded)
        )
    assert fitted.metadata["reference_score"] == fitted.score([run.get_config(int(good[0]))])[0]


def test_log_and_ordinal_transformations_use_declared_bounds():
    run = make_run()
    encoded, kind, _ = kde.kde_encoding(
        [SimpleNamespace(params={"x": 0.1, "n": 3})], scoring.numeric_space(run)
    )
    assert kind == "oc"  # ConfigSpace's deterministic parameter order is n, x.
    np.testing.assert_allclose(encoded, [[1, 0.5]])


def test_model_cache_survives_geometry_settings_but_not_new_objectives(tmp_path, monkeypatch):
    run = make_run()
    original = kde.fit_kde
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args[0].copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(kde, "fit_kde", tracked)
    first = scoring.score_coverage_candidates(run, 0, str(tmp_path), OPTIONS)
    assert len(calls) == 2
    second = scoring.score_coverage_candidates(run, 0, str(tmp_path), {**OPTIONS, "pool_size": 70})
    assert len(calls) == 2
    assert second["cache_key"] != first["cache_key"]
    assert second["model"] == first["model"]
    assert second["full_space_candidates"] == first["full_space_candidates"]
    resized = scoring.score_coverage_candidates(
        run, 0, str(tmp_path), {**OPTIONS, "global_pool_size": 90, "global_seed": 18}
    )
    assert len(calls) == 2
    assert resized["cache_key"] != first["cache_key"]
    assert resized["model"] == first["model"]
    assert len(resized["full_space_candidates"]) == 90
    fitted = scoring.fit_candidate_kde(run, 0, str(tmp_path))
    np.testing.assert_array_equal(
        [c["kde_score"] for c in first["full_space_candidates"]],
        fitted.score([c["configuration"] for c in first["full_space_candidates"]]),
    )
    assert scoring.score_coverage_candidates(run, 0, str(tmp_path), OPTIONS) == first
    run.history[0].costs[0] += 5
    changed = scoring.score_coverage_candidates(run, 0, str(tmp_path), OPTIONS)
    assert len(calls) == 4
    assert changed["source_key"] != first["source_key"]
    assert changed["reference_trial_number"] == 100
    scoring.fit_candidate_kde(run, 1, str(tmp_path))
    assert len(calls) == 6


def test_holes_and_generation_ignore_objectives_and_respect_distributions():
    run = make_run()
    data = prepare_coverage(run)
    references = sample_reference_design(run.configspace, 128, 42)
    holes = scoring.detect_holes(data, references, 4)
    candidates = scoring.generate_candidates(run, data, holes, OPTIONS)
    for t in run.history:
        t.costs = [-1000, float("nan")]
    again = prepare_coverage(run)
    assert scoring.detect_holes(again, references, 4) == holes
    assert scoring.generate_candidates(run, again, holes, OPTIONS) == candidates
    assert len({_key(c) for c in candidates}) == len(candidates)
    space = scoring.numeric_space(run)
    for c in candidates:
        assert c["configuration"]["n"] in (1, 3, 5)
        scoring._validate_config(run, c["configuration"], space)
        center = next(h["center"] for h in holes if h["id"] == c["hole_id"])
        point = normalize_configurations(run.configspace, [c["configuration"]])
        center_data = normalize_configurations(run.configspace, [center])
        distance = _nearest_distances(center_data, point)[0]
        assert distance < c["hole_radius"]
        assert distance < _nearest_distances(data, point)[0]


def _key(candidate):
    return tuple(sorted(candidate["configuration"].items()))


def test_sequential_plan_updates_geometry_and_retains_one_cohort(tmp_path):
    run = make_run()
    result = scoring.score_coverage_candidates(run, 0, str(tmp_path), OPTIONS)
    selected = sorted(
        [c for c in result["candidates"] if c["selected"]], key=lambda c: c["plan_order"]
    )
    assert [c["id"] for c in selected] == result["plan"]
    assert len(set(result["plan"])) == len(result["plan"]) == 4
    virtual = prepare_coverage(run)
    references = sample_reference_design(
        run.configspace, OPTIONS["reference_samples"], OPTIONS["seed"]
    )
    for candidate, step in zip(selected, result["steps"]):
        assert step["radius_after"] < step["radius_before"]
        assert step["max_radius_after"] <= step["max_radius_before"]
        current = scoring.detect_holes(virtual, references, OPTIONS["holes"])
        assert current[0]["id"] == candidate["plan_hole_id"]
        assert current[0]["radius"] == step["radius_before"]
        virtual = normalize_configurations(
            run.configspace, virtual.values + [candidate["configuration"]]
        )
        assert max(_nearest_distances(virtual, references)) == pytest.approx(
            step["max_radius_after"]
        )
    for c in result["candidates"]:
        assert c["relative_score"] == c["kde_score"] - result["reference_score"]
        assert c["reference_score"] == result["model"]["reference_score"]
        assert c["reference_trial_number"] in result["model"]["training_trials"]
    assert len(result["model"]["training_trials"]) == len(run.history)
    assert result["reference_score"] != 0


def test_kde_cannot_override_geometric_admissibility(tmp_path, monkeypatch):
    run = make_run()
    run.configspace = ConfigurationSpace(space={"x": (0.0, 1.0)})
    run.get_config = lambda i: {"x": 0.0 if i % 2 == 0 else 1.0}
    refs = normalize_configurations(run.configspace, [{"x": 0.5}, {"x": 0.25}, {"x": 0.75}])
    monkeypatch.setattr(scoring, "sample_reference_design", lambda *a: refs)

    def generate(run, data, holes, options):
        hole = holes[0]
        return [
            {
                "id": str(x),
                "configuration": {"x": x},
                "hole_id": hole["id"],
                "hole_radius": hole["radius"],
                "center_distance": abs(0.5 - x),
                "radius_reduction": 0.5 - abs(0.5 - x),
                "selected": False,
                "plan_order": None,
            }
            for x in (0.49, 0.51, 0.9)
        ]

    monkeypatch.setattr(scoring, "generate_candidates", generate)
    fake = SimpleNamespace(
        metadata={
            "reference_score": 2.0,
            "reference_trial_number": 100,
            "reference_objective": 0.0,
            "objective": "AP",
        },
        score=lambda configs: np.array([c["x"] * 1000 for c in configs]),
    )
    monkeypatch.setattr(scoring, "fit_candidate_kde", lambda *a: fake)
    result = scoring.score_coverage_candidates(run, 0, str(tmp_path), {**OPTIONS, "plan_size": 2})
    assert result["plan"][0] == "0.51"  # .9 has the highest score but is inadmissible.
    assert result["steps"][1]["hole_id"] != result["steps"][0]["hole_id"]
    assert result["steps"][1]["max_radius_before"] == result["steps"][0]["max_radius_after"]


def test_training_excludes_pending_failed_and_nonfinite_trials(tmp_path):
    run = make_run()
    run.history[0].status = Status.FAILED
    run.history[1].costs[0] = float("inf")
    run.history[2].costs[0] = None
    run.history[3].status = Status.PRUNED
    fitted = scoring.fit_candidate_kde(run, 0, str(tmp_path))
    assert len(fitted.metadata["training_trials"]) == 56
    assert not set(range(100, 104)) & set(fitted.metadata["training_trials"])


def test_failed_optimization_is_not_scored_or_cached(tmp_path, monkeypatch):
    def failed(*args):
        return None, {"success": False, "message": "test convergence failure"}

    monkeypatch.setattr(kde, "fit_kde", failed)
    with pytest.raises(ValueError, match="good bandwidth optimization failed: test convergence"):
        scoring.fit_candidate_kde(make_run(), 0, str(tmp_path))
    assert not list(tmp_path.rglob("*.npz"))


def test_stepped_float_candidates_and_log_integer_encoding_preserve_optuna_semantics():
    from optuna.distributions import FloatDistribution, IntDistribution

    run = make_run()
    run.configspace = ConfigurationSpace()
    run.configspace.add(
        [
            UniformFloatHyperparameter("x", 0.0, 1.0, meta={"step": 0.2}),
            UniformIntegerHyperparameter("n", 1, 100, log=True),
        ]
    )
    run.get_config = lambda i: {"x": 0.2 if i % 2 else 0.8, "n": 10}
    data = prepare_coverage(run)
    refs = sample_reference_design(run.configspace, 128, 42)
    holes = scoring.detect_holes(data, refs, 4)
    candidates = scoring.generate_candidates(run, data, holes, OPTIONS)
    for candidate in candidates:
        assert candidate["configuration"]["x"] / 0.2 == pytest.approx(
            round(candidate["configuration"]["x"] / 0.2)
        )
        assert candidate["configuration"]["n"] in range(1, 101)
    encoded, kind, _ = kde.kde_encoding(
        [SimpleNamespace(params={"x": 0.4, "n": 10})],
        {"x": FloatDistribution(0, 1, step=0.2), "n": IntDistribution(1, 100, log=True)},
    )
    assert kind == "oc"
    np.testing.assert_allclose(encoded, [[2, 0.5]])


def test_numerical_forbidden_constraints_are_preserved_by_candidate_generation():
    from ConfigSpace import ForbiddenEqualsClause

    run = make_run()
    run.configspace.add(ForbiddenEqualsClause(run.configspace["n"], 5))
    original_get_config = run.get_config
    run.get_config = lambda i: {
        **original_get_config(i),
        "n": min(3, original_get_config(i)["n"]),
    }
    data = prepare_coverage(run)
    refs = sample_reference_design(run.configspace, 128, 42)
    holes = scoring.detect_holes(data, refs, 4)
    candidates = scoring.generate_candidates(run, data, holes, OPTIONS)
    assert candidates
    for candidate in candidates:
        assert candidate["configuration"]["n"] != 5
        scoring._validate_config(run, candidate["configuration"], scoring.numeric_space(run))


def test_conditional_and_weighted_integer_spaces_are_rejected():
    from ConfigSpace import EqualsCondition

    run = make_run()
    run.configspace.add(EqualsCondition(run.configspace["x"], run.configspace["n"], 1))
    with pytest.raises(ValueError, match="conditional"):
        scoring.numeric_space(run)
    run = make_run()
    run.configspace["n"].meta["probabilities"] = [0.1, 0.2, 0.7]
    with pytest.raises(ValueError, match="explicit sampling probabilities"):
        scoring.numeric_space(run)


def test_missing_data_unsupported_spaces_and_dynamic_audit_are_explicit(tmp_path):
    run = make_run()
    run.history = run.history[:5]
    with pytest.raises(ValueError, match="Insufficient completed"):
        scoring.fit_candidate_kde(run, 0, str(tmp_path))
    run = make_run()
    run.configspace = ConfigurationSpace(space={"c": ["a", "b"]})
    with pytest.raises(ValueError, match="Categorical"):
        scoring.numeric_space(run)
    run = make_run()
    run.meta["optuna_space_error"] = "conditional or changing Optuna distributions: x"
    with pytest.raises(ValueError, match="changing Optuna"):
        scoring.numeric_space(run)


@pytest.mark.parametrize(
    "key,value",
    [
        ("pool_size", 0),
        ("holes", 1.5),
        ("seed", True),
        ("plan_size", None),
        ("reference_samples", float("nan")),
        ("global_pool_size", 0),
        ("global_pool_size", 100001),
        ("global_seed", -1),
    ],
)
def test_invalid_settings_are_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        scoring.candidate_options({key: value})


def test_global_sampling_is_reproducible_objective_free_and_respects_log_lattices():
    from ConfigSpace import ForbiddenEqualsClause

    run = make_run()
    run.configspace.add(ForbiddenEqualsClause(run.configspace["n"], 5))
    options = {**OPTIONS, "global_pool_size": 512}
    candidates, sampling = scoring.generate_full_space_candidates(run, options)
    assert len(candidates) == sampling["count"] == sampling["requested"] == 512
    assert sampling["stop_reason"] == "target_reached"
    assert len({_key(c) for c in candidates}) == len(candidates)
    history = {scoring._key(run.get_config(t.config_id)) for t in run.history}
    space = scoring.numeric_space(run)
    for candidate in candidates:
        config = candidate["configuration"]
        assert config["n"] in (1, 3)
        assert scoring._key(config) not in history
        scoring._validate_config(run, config, space)
        assert "hole_id" not in candidate
    # A uniform draw in log space places its midpoint near x=0.1, not x=0.5.
    assert 0.07 < np.median([c["configuration"]["x"] for c in candidates]) < 0.14
    for trial in run.history:
        trial.costs = [999, None]
    assert scoring.generate_full_space_candidates(run, options) == (candidates, sampling)
    larger, _ = scoring.generate_full_space_candidates(run, {**options, "global_pool_size": 1200})
    assert larger[:512] == candidates
    changed, _ = scoring.generate_full_space_candidates(run, {**options, "global_seed": 18})
    assert changed != candidates


def test_global_finite_space_excludes_all_historical_statuses_and_handles_exhaustion():
    run = make_run()
    run.configspace = ConfigurationSpace(space={"n": (1, 3)})
    run.history = run.history[:1]
    run.history[0].status = Status.FAILED
    run.get_config = lambda i: {"n": 1}
    candidates, sampling = scoring.generate_full_space_candidates(run, OPTIONS)
    assert {c["configuration"]["n"] for c in candidates} == {2, 3}
    assert sampling["count"] == 2 and sampling["stop_reason"] == "finite_space_exhausted"
    run.history = [SimpleNamespace(config_id=i) for i in range(3)]
    run.get_config = lambda i: {"n": i + 1}
    candidates, sampling = scoring.generate_full_space_candidates(run, OPTIONS)
    assert candidates == [] and sampling["draws"] == 0
    assert sampling["stop_reason"] == "finite_space_exhausted"


def test_global_sampling_stops_at_budget_when_draws_repeat(monkeypatch):
    run = make_run()
    calls = []

    def repeated(space, count, seed):
        calls.append(count)
        return SimpleNamespace(values=[run.get_config(0)] * count)

    monkeypatch.setattr(scoring, "sample_reference_design", repeated)
    candidates, sampling = scoring.generate_full_space_candidates(run, OPTIONS)
    assert candidates == []
    assert sampling["draws"] == sum(calls) == 1024
    assert sampling["stop_reason"] == "sampling_budget_reached"


def test_global_scoring_without_holes_uses_one_fit_and_bounded_batches(tmp_path, monkeypatch):
    run = make_run()
    fits, batches = [], []

    def score(configs):
        batches.append(len(configs))
        return np.array([c["x"] for c in configs])

    def fit(*args):
        fits.append(args)
        return SimpleNamespace(
            score=score,
            metadata={
                "reference_score": 0.5,
                "reference_trial_number": 100,
                "reference_objective": 0.7,
                "objective": "AP",
            },
        )

    monkeypatch.setattr(scoring, "detect_holes", lambda *args: [])
    monkeypatch.setattr(scoring, "fit_candidate_kde", fit)
    result = scoring.score_coverage_candidates(
        run, 0, str(tmp_path), {**OPTIONS, "global_pool_size": 2050}
    )
    assert len(fits) == 1 and batches == [1024, 1024, 2]
    assert result["candidates"] == result["holes"] == result["plan"] == []
    assert len(result["full_space_candidates"]) == 2050
    for candidate in result["full_space_candidates"]:
        assert candidate["kde_score"] == candidate["configuration"]["x"]
        assert candidate["relative_score"] == candidate["kde_score"] - 0.5
        assert candidate["reference_score"] == result["reference_score"]
        assert 0 < candidate["percentile"] <= 100
    assert max(c["percentile"] for c in result["full_space_candidates"]) == 100
