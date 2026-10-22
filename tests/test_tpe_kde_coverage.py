"""Focused checks for coverage-only strata, frozen scoring and prospective selection."""

import itertools
from types import SimpleNamespace

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import create_trial

from examples.api import benchmark_tpe_kde_coverage as experiment


def test_coverage_strata_distinguish_unseen_rare_and_dense_using_existing_grid():
    space = {"a": IntDistribution(0, 3), "b": IntDistribution(0, 3)}
    cs = experiment.configspace_from_distributions(space)
    history = [{"a": 0, "b": 0}] * 6 + [{"a": 1, "b": 1}] + [{"a": 2, "b": 2}] * 3
    candidates = [{"a": 3, "b": 3}, {"a": 1, "b": 1}, {"a": 0, "b": 0}]
    reference = [dict(zip(("a", "b"), values)) for values in itertools.product(range(4), repeat=2)]
    exposure, profile, spec, state, _ = experiment.coverage_profile(
        cs, history, candidates, reference, 4, 2
    )
    np.testing.assert_array_equal(exposure["missing"], [1, 0, 0])
    np.testing.assert_array_equal(exposure["rare"], [0, 1, 0])
    np.testing.assert_array_equal(exposure["minimum_count"], [0, 1, 6])
    np.testing.assert_array_equal(exposure["missing"], state.gains(spec.encode(candidates)))
    masks = experiment.populations(exposure, profile)
    np.testing.assert_array_equal(masks["coverage_seeking"], [True, False, False])
    np.testing.assert_array_equal(masks["regular"], [False, True, True])
    assert not np.any(masks["dense_regular"] & masks["rare_seeking"])
    assert not np.any(masks["coverage_seeking"] & masks["rare_seeking"])


def test_declared_logs_and_integer_steps_roundtrip_through_coverage_materialization():
    space = {
        "rate": FloatDistribution(0.001, 1, log=True),
        "n": IntDistribution(300, 1500, step=100),
    }
    cs = experiment.configspace_from_distributions(space)
    history = [{"rate": 0.01, "n": 300}, {"rate": 0.1, "n": 1000}]
    spec = experiment.CoverageSpec(experiment.normalize_configurations(cs, history), 4)
    rng = np.random.default_rng(42)
    for levels in itertools.product(range(4), repeat=2):
        values, encoded = spec.materialize(np.array(levels), rng)
        np.testing.assert_array_equal(encoded, levels)
        for name, distribution in space.items():
            assert distribution._contains(distribution.to_internal_repr(values[name]))


def test_primary_grid_selection_uses_only_training_reference_profiles():
    profiles = [
        {
            "fold": f,
            "resolution": r,
            "strength": t,
            "reference_missing_fraction": (0.4 if (r, t) == (4, 4) else 0.99),
        }
        for f in range(5)
        for r, t in experiment.VIEWS
    ]
    assert experiment.choose_primary(profiles) == (4, 4)
    assert experiment.choose_primary(profiles[::-1]) == (4, 4)


def test_curves_require_twenty_points_per_bin_and_at_least_three_bins():
    rows, bins = experiment.evaluation(np.arange(59.0), np.arange(59.0), 0)
    assert rows["spearman"] == pytest.approx(1) and not bins and np.isnan(rows["monotone"])
    rows, bins = experiment.evaluation(np.arange(60.0), np.arange(60.0), 3)
    assert len(bins) == 3 and all(row["n"] == 20 for row in bins)
    assert rows["monotone"] == 1 and rows["decreases"] == 0
    with pytest.raises(ValueError, match="Insufficient"):
        experiment.evaluation(np.arange(59.0), np.arange(59.0), 3)


def test_regular_and_seeking_use_equal_bin_counts_across_folds():
    contexts = {}
    for fold in range(5):
        for resolution, strength in experiment.VIEWS:
            masks = {p: np.ones(400, dtype=bool) for p in experiment.POPULATIONS}
            masks["coverage_seeking"][100 + fold:] = False
            contexts[fold, resolution, strength] = (None, masks)
    counts = experiment.shared_bin_counts(contexts)
    for resolution, strength in experiment.VIEWS:
        assert counts[resolution, strength, "regular"] == 5
        assert counts[resolution, strength, "coverage_seeking"] == 5


def test_new_scoring_path_cannot_see_heldout_objectives():
    rng = np.random.default_rng(12)
    space = {"rate": FloatDistribution(0.001, 1, log=True), "n": IntDistribution(1, 4)}
    trials = []
    for i in range(100):
        rate, n = float(np.exp(rng.uniform(np.log(0.001), 0))), int(rng.integers(1, 5))
        trials.append(
            create_trial(params={"rate": rate, "n": n}, distributions=space, value=rate + n / 10)
        )
    train = np.arange(80)
    changed = [
        SimpleNamespace(number=i, params=t.params, value=t.value if i < 80 else -1000 - i)
        for i, t in enumerate(trials)
    ]
    # create_trial does not assign numbers outside a Study; assign a consistent identity.
    original = [
        SimpleNamespace(number=i, params=t.params, value=t.value) for i, t in enumerate(trials)
    ]
    before, diagnostics = experiment.fit_scores(original, space, train, original[80:])
    after, _ = experiment.fit_scores(changed, space, train, changed[80:])
    for method in experiment.METHODS:
        np.testing.assert_array_equal(before[method], after[method])
    good_bad_ids = [[g["trial_ids"] for g in d["groups"]] for d in diagnostics]
    assert good_bad_ids[0] == good_bad_ids[1]


def test_prospective_selection_preserves_score_range_coverage_and_disagreements():
    rng = np.random.default_rng(8)
    pool = [
        {
            "candidate_id": i,
            "tpe_score": float(rng.normal()),
            "kde_optimized_score": float(rng.normal()),
            "missing_gain": int(rng.integers(1, 20)),
        }
        for i in range(200)
    ]
    selected = experiment.select_diverse(pool)
    assert len(selected) == len({r["candidate_id"] for r in selected}) == 25
    assert selected == experiment.select_diverse(pool)
    for key in ("tpe_score", "kde_optimized_score", "missing_gain"):
        assert min(r[key] for r in selected) == min(r[key] for r in pool)
        assert max(r[key] for r in selected) == max(r[key] for r in pool)
    assert all(r["objective"] == "" and r["evaluation_status"] == "not_executed" for r in selected)
