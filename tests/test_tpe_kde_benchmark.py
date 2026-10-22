"""Focused checks for held-out ranking, exact LOO gradients and bounded optimization."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

SCRIPT = Path(__file__).resolve().parents[1] / "examples/api/benchmark_tpe_kde.py"
SPEC = importlib.util.spec_from_file_location("tpe_kde_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def study_fixture():
    """Make a reproducible mixed space without launching objective computations."""
    study = optuna.create_study(direction="maximize")
    rng = np.random.default_rng(8)
    for index in range(150):
        x = float(rng.uniform(0.1, 1))
        n = int(rng.integers(1, 5))
        study.add_trial(
            create_trial(
                params={"x": x, "n": n},
                distributions={
                    "x": FloatDistribution(0.1, 1, log=True),
                    "n": IntDistribution(1, 4),
                },
                value=x + n / 10 + float(rng.normal(0, 0.05)),
            )
        )
    return study


def test_grouped_folds_are_balanced_reproducible_and_do_not_split_duplicates():
    keys = [str(i) for i in range(100)] + ["0", "1", "1"]
    folds = benchmark.make_folds(keys)
    np.testing.assert_array_equal(folds, benchmark.make_folds(keys))
    assert folds[0] == folds[100]
    assert folds[1] == folds[101] == folds[102]
    assert max(np.bincount(folds)) - min(np.bincount(folds)) <= 1


def test_good_bad_uses_only_training_values_and_respects_minimization():
    values = np.array([5.0, 2, 4, 3, 1, -100, 100])
    train = np.arange(5)
    good, bad = benchmark.split_good_bad(values, train, maximize=False)
    assert good.tolist() == [4]
    assert set(bad) == {0, 1, 2, 3}
    values[5:] *= -100
    np.testing.assert_array_equal(good, benchmark.split_good_bad(values, train, False)[0])
    assert benchmark.split_good_bad(values, train, True)[0].tolist() == [0]


def test_activity_audit_rejects_implicit_imputation_and_supports_explicit_intersection():
    study = study_fixture()
    study.add_trial(
        create_trial(
            params={"x": 0.5}, distributions={"x": FloatDistribution(0.1, 1, log=True)}, value=0.3
        )
    )
    with pytest.raises(ValueError, match="Conditional or changing"):
        benchmark.prepare(study)
    _, space, _, audit = benchmark.prepare(study, subspace="intersection")
    assert list(space) == ["x"]
    assert audit["excluded_parameters"] == ["n"]
    assert audit["activity_audit"]["n"]["active_trials"] == 150


def test_filters_failed_and_nonfinite_trials():
    study = study_fixture()
    study.add_trial(create_trial(state=TrialState.FAIL))
    study.add_trial(create_trial(value=float("inf")))
    trials, _, _, audit = benchmark.prepare(study)
    assert len(trials) == 150
    assert audit["total_trials"] == 152


def test_encoding_respects_logs_steps_and_unordered_categories():
    space = {
        "x": FloatDistribution(1, 100, log=True),
        "n": IntDistribution(10, 30, step=10),
        "c": CategoricalDistribution(["a", "b"]),
    }
    trials = [create_trial(params={"x": 10, "n": 20, "c": "b"}, distributions=space, value=1)]
    x, var_type, _ = benchmark.kde_encoding(trials, space)
    assert var_type == "cou"
    np.testing.assert_allclose(x, [[0.5, 1, 1]])


def test_log_kde_matches_statsmodels_pdf_and_survives_pdf_underflow():
    rng = np.random.default_rng(42)
    data = np.column_stack([rng.normal(size=80), rng.integers(0, 4, 80), rng.integers(0, 3, 80)])
    space = {
        "x": FloatDistribution(-5, 5),
        "n": IntDistribution(0, 3),
        "c": CategoricalDistribution(["a", "b", "c"]),
    }
    model, _ = benchmark.fit_kde(data, "cou", space, "normal_reference")
    queries = np.array([[0.1, 2, 1], [-0.5, 0, 2]])
    np.testing.assert_allclose(
        benchmark.kde_logpdf(model, queries), np.log(model.pdf(queries)), atol=1e-12
    )
    far = np.array([[1000.0, 1, 1]])
    assert model.pdf(far) == 0
    assert np.isfinite(benchmark.kde_logpdf(model, far)).all()


def test_missing_category_is_an_explicit_blocker():
    data = np.array([[0.1, 0], [0.2, 0], [0.3, 0], [0.4, 0]])
    with pytest.raises(ValueError, match="lacks declared categories"):
        benchmark.fit_kde(
            data, "cu", {"x": FloatDistribution(0, 1), "c": CategoricalDistribution(["a", "b"])}
        )


def test_optimizer_timeout_is_explicit_and_never_falls_back(monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def immediate_timeout(seconds):
        raise TimeoutError("test CV time limit")
        yield

    monkeypatch.setattr(benchmark, "time_limit", immediate_timeout)
    model, info = benchmark.fit_kde(
        np.arange(20.0).reshape(-1, 1), "c", {"x": FloatDistribution(0, 20)}
    )
    assert info["used"] == "bounded_ml"
    assert info["fallback"] is None
    assert not info["success"] and not info["optimization_verified"]
    assert info["message"] == "test CV time limit"
    assert np.isfinite(model.bw).all()


def test_vectorized_loo_matches_statsmodels_and_analytic_gradient():
    from scipy.optimize import check_grad
    from statsmodels.nonparametric.kernel_density import KDEMultivariate

    rng = np.random.default_rng(42)
    data = np.column_stack([rng.normal(size=30), rng.integers(0, 4, 30), np.tile([0, 1, 2], 10)])
    theta = np.array([np.log(0.4), 0.6, 0.3])
    objective = benchmark.loo_objective(data, "cou")
    loss, gradient = objective(theta)
    model = KDEMultivariate(data, "cou", bw=[0.4, 0.6, 0.3])
    expected = model.loo_likelihood(model.bw, np.log) / len(data) + np.log(len(data) - 1)
    assert np.isclose(loss, expected, atol=1e-12)
    assert check_grad(lambda x: objective(x)[0], lambda x: objective(x)[1], theta) < 1e-5
    assert np.isfinite(gradient).all()


def test_optimizer_improves_valid_loo_without_post_fit_clipping():
    trials, space, _, _ = benchmark.prepare(study_fixture())
    x, var_type, _ = benchmark.kde_encoding(trials, space)
    model, info = benchmark.fit_kde(x, var_type, space)
    assert info["success"] and info["optimization_verified"]
    assert info["mean_loo_loglikelihood_gain"] > 1e-6
    assert info["bandwidth_changed"] and not info["post_fit_clipping"]
    assert info["evaluations"] > 1
    ordinal = var_type.index("o")
    assert info["initial_bandwidth"][ordinal] == 0.5
    assert 0 < model.bw[ordinal] < 1
    assert np.all(np.array(info["initial_bandwidth"]) >= info["lower_bounds"])
    assert np.all(np.array(info["initial_bandwidth"]) <= info["upper_bounds"])


@pytest.mark.parametrize("initial", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_robustness_start_changes_only_discrete_initialization(initial):
    rng = np.random.default_rng(12)
    data = np.column_stack([rng.uniform(size=60), np.tile([0, 1, 2], 20), np.tile([0, 1], 30)])
    space = {
        "x": FloatDistribution(0, 1),
        "n": IntDistribution(0, 2),
        "category": CategoricalDistribution(["a", "b"]),
    }
    _, info = benchmark.fit_kde(data, "cou", space, discrete_initial=initial)
    np.testing.assert_allclose(info["initial_bandwidth"][1:], initial)
    assert info["initial_bandwidth"][0] == info["raw_normal_reference_bandwidth"][0]
    assert info["success"]
    assert np.isfinite(info["final_mean_loo_loglikelihood"])


def test_robustness_compares_identical_folds_and_reports_all_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "plot_results", lambda *args: None)
    monkeypatch.setattr(benchmark, "plot_robustness", lambda *args: None)
    summary = benchmark.robustness(study_fixture(), tmp_path)
    assert len(summary) == 5
    assert all(r["total_fits"] == r["converged_fits"] == 10 for r in summary)
    with (tmp_path / "ranking_pairs.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 50
    assert {int(r["fold"]) for r in pairs} == set(range(5))
    assert all(float(r["spearman"]) > 0.99 for r in pairs)
    with (tmp_path / "paired_loo.csv").open() as stream:
        likelihoods = list(csv.DictReader(stream))
    assert len(likelihoods) == 50
    assert all(float(r["mean_loo_gap_to_best_same_fold_group"]) < 1e-6 for r in likelihoods)
    with (tmp_path / "decile_metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 250


def test_robustness_reports_all_timeouts_without_claiming_convergence(tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def immediate_timeout(seconds):
        raise TimeoutError("test robustness timeout")
        yield

    monkeypatch.setattr(benchmark, "time_limit", immediate_timeout)
    monkeypatch.setattr(benchmark, "plot_results", lambda *args: None)
    monkeypatch.setattr(benchmark, "plot_robustness", lambda *args: None)
    summary = benchmark.robustness(study_fixture(), tmp_path)
    assert all(r["converged_fits"] == r["verified_fits"] == 0 for r in summary)
    assert all(r["total_fits"] == 10 for r in summary)
    assert all(r["min_iterations"] is r["max_iterations"] is None for r in summary)
    with (tmp_path / "paired_loo.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 50
    assert all(r["status"] == "-1" and r["message"] == "test robustness timeout" for r in rows)


def test_deciles_are_fold_local_equal_sized_and_detect_a_reversal():
    folds = np.repeat([0, 1], 100)
    scores = {method: np.tile(np.arange(100.0), 2) + folds * 10000 for method in benchmark.METHODS}
    objectives = np.tile(np.repeat(np.arange(10.0), 10), 2)
    objectives[80:90] = 4.0
    details, aggregate, monotonicity = benchmark.score_deciles(objectives, folds, scores, True)
    assert len(details) == 60 and len(aggregate) == 30
    assert all(row["n_trials"] == 10 for row in details)
    assert monotonicity["tpe"]["aggregate_decile_decreases"] == 1
    assert monotonicity["tpe"]["total_fold_decile_decreases"] == 1


def test_parzen_adapter_uses_uniform_multivariate_weights_and_finite_scores():
    study = study_fixture()
    trials, space, _, _ = benchmark.prepare(study)
    adapter = benchmark.OptunaParzenAdapter(space)
    observations = adapter.encode(trials)
    indices = np.arange(30)
    model = adapter.fit(observations, indices)
    query = {name: values[30:40] for name, values in observations.items()}
    scores = model.log_pdf(query)
    assert np.isfinite(scores).all()
    assert len(scores) == 10
    assert adapter.parameters.multivariate
    np.testing.assert_array_equal(adapter.parameters.weights(30), np.ones(30))


def test_parzen_density_ratio_prefers_nearby_good_observations():
    space = {"x": FloatDistribution(0.01, 1, log=True)}
    trials = [
        create_trial(params={"x": x}, distributions=space, value=0)
        for x in [0.10, 0.11, 0.12, 0.13, 0.14, 0.8, 0.85, 0.9, 0.95, 0.99]
    ]
    adapter = benchmark.OptunaParzenAdapter(space)
    observations = adapter.encode(trials)
    good = adapter.fit(observations, np.arange(5))
    bad = adapter.fit(observations, np.arange(5, 10))
    queries = {"x": np.array([0.12, 0.9])}
    ratios = good.log_pdf(queries) - bad.log_pdf(queries)
    assert ratios[0] > 0 > ratios[1]


def test_sqlite_loader_detaches_study_without_changing_source(tmp_path):
    database = tmp_path / "source.db"
    storage = optuna.storages.RDBStorage(f"sqlite:///{database}")
    source = optuna.create_study(
        study_name="read-only-source", storage=storage, direction="minimize"
    )
    source.add_trial(create_trial(value=0.25))
    source.add_trial(create_trial(state=TrialState.FAIL))
    source.set_user_attr("sampler", "random")
    storage.engine.dispose()
    before = database.read_bytes()
    detached = benchmark.load_study(database, "read-only-source")
    assert database.read_bytes() == before
    database.unlink()
    assert detached.direction == optuna.study.StudyDirection.MINIMIZE
    assert len(detached.trials) == 2
    assert detached.trials[0].value == 0.25
    assert detached.user_attrs["sampler"] == "random"


def test_heldout_objective_permutation_cannot_change_own_fold_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "plot_results", lambda *args: None)
    study = study_fixture()
    trials, _, keys, _ = benchmark.prepare(study)
    folds = benchmark.make_folds(keys)
    benchmark.benchmark(study, tmp_path / "original", fold=0)
    changed = optuna.create_study(direction="maximize")
    values = [t.value for i, t in enumerate(trials) if folds[i] == 0][::-1]
    for index, trial in enumerate(trials):
        changed.add_trial(
            create_trial(
                params=trial.params,
                distributions=trial.distributions,
                value=values.pop(0) if folds[index] == 0 else trial.value,
            )
        )
    benchmark.benchmark(changed, tmp_path / "changed", fold=0)

    def read(folder):
        with (tmp_path / folder / "oof_scores.csv").open() as stream:
            return [row for row in csv.DictReader(stream) if row["fold"] == "0"]

    original, changed_rows = read("original"), read("changed")
    for before, after in zip(original, changed_rows):
        assert before["trial_number"] == after["trial_number"]
        assert before["tpe_score"] == after["tpe_score"]
        assert before["kde_reference_score"] == after["kde_reference_score"]
        assert before["kde_optimized_score"] == after["kde_optimized_score"]


def test_minimization_metrics_choose_low_objectives():
    folds = np.repeat(np.arange(5), 25)
    objectives = np.tile(np.arange(25.0), 5)
    scores = {method: -objectives for method in benchmark.METHODS}
    metrics, _ = benchmark.evaluate(objectives, folds, scores, maximize=False)
    assert all(np.isclose(row["spearman"], 1) for row in metrics)
    assert all(row["top20"] == row["oracle20"] == 9.5 for row in metrics)
    assert all(row["improvement"] == 2.5 for row in metrics)


def test_learning_samples_are_nested_and_full_training_sets_are_not_replicated():
    folds = np.repeat(np.arange(5), 30)
    samples = list(benchmark.learning_samples(folds, (20, 60, 120), repetitions=5))
    assert len(samples) == 55
    assert len([s for s in samples if s[1] == 120]) == 5
    for fold in range(5):
        for rep in range(5):
            selected = [s for s in samples if s[0] == fold and s[2] == rep]
            assert set(selected[0][4]) <= set(selected[1][4])
            for _, size, _, seed, train, test in selected:
                assert len(train) == size
                assert not set(train) & set(test)
                np.testing.assert_array_equal(test, np.flatnonzero(folds == fold))
                permutation = np.random.default_rng(np.random.SeedSequence(seed)).permutation(
                    np.flatnonzero(folds != fold)
                )
                np.testing.assert_array_equal(train, np.sort(permutation[:size]))


def test_learning_metrics_detect_reversals_and_constant_scores():
    objectives = np.repeat(np.arange(10.0), 4)
    objectives[32:36] = 4
    metrics, means, bins = benchmark.learning_metrics(objectives, np.arange(40.0))
    assert metrics["decreases"] == 1 and metrics["monotone"] == 0
    assert metrics["max_drop"] == metrics["sum_drops"] == 3
    assert metrics["delta_ap"] == 9
    np.testing.assert_array_equal(np.bincount(bins)[1:], np.repeat(4, 10))
    with pytest.raises(ValueError, match="Undefined Spearman"):
        benchmark.learning_metrics(objectives, np.ones(40))


def test_learning_curve_shares_groups_reproduces_baseline_and_saves_paired_metrics(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(benchmark, "plot_results", lambda *args: None)
    monkeypatch.setattr(benchmark, "plot_learning_curve", lambda *args: None)
    study = study_fixture()
    benchmark.benchmark(study, tmp_path / "baseline")
    trials, space, _, _ = benchmark.prepare(study)
    x, _, _ = benchmark.kde_encoding(trials, space)
    original_tpe_fit, original_kde_fit = benchmark.OptunaParzenAdapter.fit, benchmark.fit_kde
    tpe_groups, kde_groups = [], []

    def tpe_fit(self, observations, indices):
        tpe_groups.append(indices.copy())
        return original_tpe_fit(self, observations, indices)

    def kde_fit(data, *args, **kwargs):
        kde_groups.append(data.copy())
        return original_kde_fit(data, *args, **kwargs)

    monkeypatch.setattr(benchmark.OptunaParzenAdapter, "fit", tpe_fit)
    monkeypatch.setattr(benchmark, "fit_kde", kde_fit)
    output = tmp_path / "learning"
    summary = benchmark.learning_curve(study, output, sizes=(20, 120), repetitions=2,
                                      reference_oof=tmp_path / "baseline/oof_scores.csv")
    assert len(tpe_groups) == len(kde_groups) == 30
    for indices, data in zip(tpe_groups, kde_groups):
        np.testing.assert_array_equal(x[indices], data)
    assert all(r["attempted"] == r["valid_evaluations"] for r in summary)
    assert all(r["attempted"] == 5 and r["n_stability_pairs"] == 0
               for r in summary if r["n_train"] == 120)
    checks = json.loads((output / "reference_checks.json").read_text())
    assert len(checks) == 10 and all(r["matches"] for r in checks)
    with (output / "fit_metrics.csv").open() as stream:
        metrics = {(int(r["fold"]), int(r["n_train"]), int(r["repetition"]), r["method"]): r
                   for r in csv.DictReader(stream)}
    with (output / "paired_metrics.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 15
    for pair in pairs:
        key = tuple(int(pair[k]) for k in ("fold", "n_train", "repetition"))
        assert pair["valid_pair"] == "1"
        assert float(pair["delta_spearman"]) == pytest.approx(
            float(metrics[(*key, "tpe")]["spearman"])
            - float(metrics[(*key, "kde_optimized")]["spearman"]))


def test_learning_curve_records_failures_without_fallback_or_partial_pair_metrics(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(benchmark, "plot_learning_curve", lambda *args: None)

    def failed_fit(*args, **kwargs):
        raise ValueError("test unsupported training group")

    monkeypatch.setattr(benchmark, "fit_kde", failed_fit)
    summary = benchmark.learning_curve(study_fixture(), tmp_path, sizes=(20,), repetitions=1)
    assert next(r for r in summary if r["method"] == "tpe")["valid_evaluations"] == 5
    kde = next(r for r in summary if r["method"] == "kde_optimized")
    assert kde["attempted"] == 5 and kde["valid_evaluations"] == kde["fit_successes"] == 0
    with (tmp_path / "paired_metrics.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 5 and all(r["valid_pair"] == "0" for r in pairs)
    assert all(np.isnan(float(r["delta_spearman"])) for r in pairs)
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    failed = [g for r in diagnostics if r["method"] == "kde_optimized" for g in r["groups"]]
    assert len(failed) == 10 and all("unsupported training group" in r["message"] for r in failed)
