"""Compare Optuna Parzen and statsmodels KDE rankings on held-out real trials.

Run ``uv run --locked python examples/api/benchmark_tpe_kde.py --help``.
Optional benchmark dependencies: optuna, statsmodels, scipy, matplotlib.
SQLite inputs are opened read-only and backed up before Optuna loads them.
Only load pickle snapshots from a trusted source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import sqlite3
import tempfile
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState
from scipy.stats import spearmanr

from deepcave.evaluators.kde import (
    fit_kde as _fit_kde,
    kde_encoding,
    kde_logpdf,
    loo_objective,
    split_good_bad,
    time_limit,
)

SEED = 42
METHODS = ("tpe", "kde_reference", "kde_optimized")
LABELS = {
    "tpe": "Optuna TPE",
    "kde_reference": "KDE normal reference",
    "kde_optimized": "KDE optimized LOO",
}


def load_study(source, study_name=None):
    """Load a trusted snapshot, or detach a study from a read-only SQLite backup."""
    source = Path(source).resolve(strict=True)
    if source.is_dir():
        source = source / "optuna_study.pkl"
    if source.suffix == ".pkl":
        with source.open("rb") as stream:
            return pickle.load(stream)
    if not study_name:
        raise ValueError("--study-name is required for SQLite input")
    with tempfile.TemporaryDirectory(prefix="tpe-kde-") as directory:
        backup = Path(directory) / "study.db"
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
            with sqlite3.connect(backup) as destination:
                original.backup(destination)
        storage = optuna.storages.RDBStorage(f"sqlite:///{backup}")
        try:
            memory = optuna.storages.InMemoryStorage()
            optuna.copy_study(from_study_name=study_name, from_storage=storage, to_storage=memory)
            return optuna.load_study(study_name=study_name, storage=memory)
        finally:
            storage.engine.dispose()


def configuration_key(trial, names):
    """Identify configurations independently of objective, trial number and dictionary order."""
    return json.dumps({name: trial.params[name] for name in names}, sort_keys=True)


def prepare(study, limit=None, subspace="strict"):
    """Audit activity/distributions and select finite completed single-objective trials."""
    if len(study.directions) != 1:
        raise ValueError("Select a single-objective study; multi-objective studies are unsupported")
    all_trials = study.get_trials(deepcopy=False)
    trials = [
        trial
        for trial in all_trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(trial.value)
    ]
    if not trials:
        raise ValueError("No finite completed trials")
    trials.sort(key=lambda trial: trial.number)
    union = sorted(set().union(*(trial.distributions for trial in trials)))
    audit = {}
    common = {}
    for name in union:
        distributions = [trial.distributions[name] for trial in trials if name in trial.params]
        signatures = sorted({optuna.distributions.distribution_to_json(d) for d in distributions})
        audit[name] = {"active_trials": len(distributions), "distributions": signatures}
        if len(distributions) == len(trials) and len(signatures) == 1:
            common[name] = distributions[0]
    excluded = sorted(set(union) - set(common))
    if excluded and subspace == "strict":
        raise ValueError(
            f"Conditional or changing distributions: {excluded}. "
            "Explicitly use --subspace intersection to compare the same fixed common subspace."
        )
    space = {name: d for name, d in common.items() if not d.single()}
    if not space:
        raise ValueError("No shared nonconstant parameters")
    # The optional pilot sample is drawn independently of objective values.
    if limit is not None and limit < len(trials):
        indices = sorted(np.random.default_rng(SEED).choice(len(trials), limit, replace=False))
        trials = [trials[index] for index in indices]
    keys = [configuration_key(trial, sorted(space)) for trial in trials]
    metadata = {
        "total_trials": len(all_trials),
        "finite_completed_trials": sum(
            t.state == TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)
            for t in all_trials
        ),
        "selected_trials": len(trials),
        "activity_audit": audit,
        "excluded_parameters": excluded,
        "constant_parameters": sorted(set(common) - set(space)),
        "unique_configurations": len(set(keys)),
        "subspace": subspace,
    }
    return trials, space, keys, metadata


def make_folds(keys):
    """Assign shuffled configuration groups to the least populated fold, seed 42."""
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    if len(groups) < 5:
        raise ValueError("At least five distinct configurations are required")
    shuffled = np.random.default_rng(SEED).permutation(len(groups))
    # Large duplicate groups first; shuffle determines ties without using objectives.
    members = list(groups.values())
    shuffled = sorted(shuffled, key=lambda index: -len(members[index]))
    counts = np.zeros(5, dtype=int)
    folds = np.empty(len(keys), dtype=int)
    for index in shuffled:
        fold = int(np.argmin(counts))
        folds[members[index]] = fold
        counts[fold] += len(members[index])
    return folds


def uniform_weights(n):
    """Avoid imposing adaptive-sampling recency weights on historical random trials."""
    return np.ones(n)


class OptunaParzenAdapter:
    """Keep all private Optuna API access in this version-checked adapter."""

    def __init__(self, space):
        try:
            from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator

            sampler = optuna.samplers.TPESampler(
                seed=SEED,
                multivariate=True,
                weights=uniform_weights,
                consider_prior=True,
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=False,
            )
            self.parameters = sampler._parzen_estimator_parameters
            self.estimator = _ParzenEstimator
        except (ImportError, AttributeError, TypeError) as error:
            raise RuntimeError(
                f"Parzen adapter incompatible with Optuna {optuna.__version__}: {error}"
            ) from error
        self.space = space

    def encode(self, trials):
        """Preserve actual distributions; Optuna transforms log parameters internally."""
        return {
            name: np.array([distribution.to_internal_repr(t.params[name]) for t in trials])
            for name, distribution in self.space.items()
        }

    def fit(self, observations, indices):
        return self.estimator(
            {name: values[indices] for name, values in observations.items()},
            self.space,
            self.parameters,
        )


def fit_kde(data, var_type, space, bandwidth="bounded_ml", timeout=120, discrete_initial=0.5):
    """Use the production implementation, retaining the benchmark's timeout hook."""
    return _fit_kde(data, var_type, space, bandwidth, timeout, discrete_initial,
                    _time_limit=time_limit)


def write_csv(path, rows):
    """Write portable CSV output with explicit column names."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(objectives, folds, scores, maximize):
    """Rank within folds, keeping the top-k decision independent of held-out targets."""
    utility = objectives if maximize else -objectives
    rows, curves = [], []
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        if len(indices) < 20:
            raise ValueError("Each test fold needs at least 20 trials")
        random_mean = float(np.mean(objectives[indices]))
        oracle_order = indices[np.argsort(-utility[indices], kind="stable")]
        for method in METHODS:
            predicted_order = indices[np.argsort(-scores[method][indices], kind="stable")]
            rho = float(spearmanr(scores[method][indices], utility[indices]).statistic)
            if not math.isfinite(rho):
                raise ValueError(f"Undefined Spearman correlation: {method}, fold {fold}")
            top20 = float(np.mean(objectives[predicted_order[:20]]))
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "n_test": len(indices),
                    "spearman": rho,
                    "top20": top20,
                    "random": random_mean,
                    "oracle20": float(np.mean(objectives[oracle_order[:20]])),
                    "improvement": (top20 - random_mean) * (1 if maximize else -1),
                }
            )
            for k in (5, 10, 20, 50, 100):
                if k <= len(indices):
                    curves.append(
                        {
                            "fold": fold,
                            "method": method,
                            "k": k,
                            "objective": float(np.mean(objectives[predicted_order[:k]])),
                            "random": random_mean,
                            "oracle": float(np.mean(objectives[oracle_order[:k]])),
                        }
                    )
    return rows, curves


def score_deciles(objectives, folds, scores, maximize):
    """Bin ranks within each held-out fold; never pool absolute scores across folds."""
    details, aggregate = [], []
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        for method in scores:
            ordered = indices[np.argsort(scores[method][indices], kind="stable")]
            for decile, members in enumerate(np.array_split(ordered, 10), start=1):
                details.append(
                    {
                        "fold": int(fold),
                        "method": method,
                        "decile": decile,
                        "n_trials": len(members),
                        "mean_objective": float(np.mean(objectives[members])),
                        "score_min": float(np.min(scores[method][members])),
                        "score_max": float(np.max(scores[method][members])),
                    }
                )
    for method in scores:
        for decile in range(1, 11):
            means = [
                row["mean_objective"]
                for row in details
                if row["method"] == method and row["decile"] == decile
            ]
            aggregate.append(
                {
                    "method": method,
                    "decile": decile,
                    "mean_objective": float(np.mean(means)),
                    "sd_fold_mean_objective": (
                        float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
                    ),
                }
            )
    monotonicity = {}
    for method in scores:
        means = [row["mean_objective"] for row in aggregate if row["method"] == method]
        utility = np.array(means) * (1 if maximize else -1)
        changes = np.diff(utility)
        fold_decreases = []
        for fold in np.unique(folds):
            values = [
                row["mean_objective"]
                for row in details
                if row["method"] == method and row["fold"] == fold
            ]
            fold_decreases.append(
                int(np.count_nonzero(np.diff(values) * (1 if maximize else -1) < -1e-12))
            )
        monotonicity[method] = {
            "aggregate_decile_decreases": int(np.count_nonzero(changes < -1e-12)),
            "aggregate_decile_min_utility_change": float(np.min(changes)),
            "aggregate_decile_spearman": float(spearmanr(np.arange(10), utility).statistic),
            "total_fold_decile_decreases": sum(fold_decreases),
            "mean_fold_decile_decreases": float(np.mean(fold_decreases)),
        }
    return details, aggregate, monotonicity


def plot_results(output, objectives, scores, curves, deciles):
    """Produce standalone publication/export artifacts, with fold-local top-k aggregation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for method in METHODS:
        selected = [row for row in deciles if row["method"] == method]
        decile = np.array([row["decile"] for row in selected])
        mean = np.array([row["mean_objective"] for row in selected])
        sd = np.array([row["sd_fold_mean_objective"] for row in selected])
        (line,) = ax.plot(decile, mean, marker="o", label=LABELS[method])
        ax.fill_between(decile, mean - sd, mean + sd, alpha=0.12, color=line.get_color())
    ax.set(
        xlabel="Within-fold score decile (1 = lowest, 10 = highest)",
        ylabel="Mean actual AP",
        xticks=np.arange(1, 11),
        title="Out-of-fold score-performance relationship",
    )
    ax.legend(title="Mean of fold means; shading = +/- 1 fold SD")
    ax.grid(alpha=0.2)
    fig.savefig(output / "score_performance_deciles.png", dpi=180)
    fig.savefig(output / "score_performance_deciles.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        ax.scatter(scores[method], objectives, s=9, alpha=0.4)
        ax.set(
            xlabel="Out-of-fold log density ratio", ylabel="Actual objective", title=LABELS[method]
        )
        ax.grid(alpha=0.2)
    fig.suptitle("Higher scores predict better trials; score scales are method/fold specific")
    fig.savefig(output / "score_vs_objective.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ks = sorted({row["k"] for row in curves})
    for method in METHODS:
        means = [
            np.mean(
                [row["objective"] for row in curves if row["method"] == method and row["k"] == k]
            )
            for k in ks
        ]
        ax.plot(ks, means, marker="o", label=LABELS[method])
    for key, label in (("random", "Random expectation"), ("oracle", "Oracle")):
        ax.plot(
            ks,
            [
                np.mean([row[key] for row in curves if row["method"] == "tpe" and row["k"] == k])
                for k in ks
            ],
            "--",
            label=label,
        )
    ax.set(xlabel="k selected trials per test fold", ylabel="Mean actual objective (five folds)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "top_k.png", dpi=170)
    plt.close(fig)


def benchmark(
    study,
    output,
    limit=None,
    timeout=120,
    subspace="strict",
    reference_oof=None,
    fold=None,
    discrete_initial=0.5,
    allow_unverified=False,
):
    """Run the complete five-fold experiment and write results and diagnostics."""
    trials, space, keys, audit = prepare(study, limit, subspace)
    folds = make_folds(keys)
    if min(np.bincount(folds)) < 20:
        raise ValueError("At least 20 held-out trials per fold are needed")
    objectives = np.array([trial.value for trial in trials])
    historical = None
    if reference_oof is not None:
        with Path(reference_oof).open() as stream:
            historical = {int(row["trial_number"]): row for row in csv.DictReader(stream)}
        if set(historical) != {trial.number for trial in trials}:
            raise ValueError("Historical trial IDs differ from selected study")
        for index, trial in enumerate(trials):
            row = historical[trial.number]
            if (
                int(row["fold"]) != folds[index]
                or float(row["objective"]) != trial.value
                or json.loads(row["params_json"]) != trial.params
            ):
                raise ValueError(f"Historical fold/configuration/objective differs: {trial.number}")
        print(
            "Historical trial IDs, configurations, objectives and all five folds match.", flush=True
        )
    maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    adapter = OptunaParzenAdapter(space)
    observations = adapter.encode(trials)
    x, var_type, transforms = kde_encoding(trials, space)
    scores = {method: np.full(len(trials), np.nan) for method in METHODS}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    timings, diagnostics, optimization_diagnostics = [], [], []
    selected_folds = range(5) if fold is None else [fold]
    for fold_id in selected_folds:
        train, test = np.flatnonzero(folds != fold_id), np.flatnonzero(folds == fold_id)
        good, bad = split_good_bad(objectives, train, maximize)
        # Observation order is chronological, not target rank: equal TPE weights.
        good, bad = np.sort(good), np.sort(bad)
        query = {name: values[test] for name, values in observations.items()}
        start = perf_counter()
        l_tpe, g_tpe = adapter.fit(observations, good), adapter.fit(observations, bad)
        tpe_fit = perf_counter() - start
        start = perf_counter()
        scores["tpe"][test] = l_tpe.log_pdf(query) - g_tpe.log_pdf(query)
        tpe_score = perf_counter() - start
        timings.append(
            {"fold": fold_id, "method": "tpe", "fit_seconds": tpe_fit, "score_seconds": tpe_score}
        )
        print(f"Fold {fold_id + 1}/5: TPE complete; KDE fits ({len(good)}/{len(bad)})", flush=True)
        for method, bandwidth in (
            ("kde_reference", "normal_reference"),
            ("kde_optimized", "bounded_ml"),
        ):
            models, info = [], []
            for group, members in (("good", good), ("bad", bad)):
                model, details = fit_kde(
                    x[members], var_type, space, bandwidth, timeout, discrete_initial
                )
                models.append(model)
                info.append(details)
                if method == "kde_optimized":
                    record = {
                        "fold": fold_id,
                        "group": group,
                        "n_train": len(members),
                        "parameter_names": list(space),
                        **details,
                    }
                    optimization_diagnostics.append(record)
                    (output / "optimization_diagnostics.json").write_text(
                        json.dumps(optimization_diagnostics, indent=2) + "\n"
                    )
                    print(json.dumps(record), flush=True)
                    if not details["optimization_verified"] and not allow_unverified:
                        raise RuntimeError(
                            f"Unverified optimization, fold {fold_id}, {group}: "
                            f"{details['message']}. "
                            f"Diagnostics saved in {output}; no baseline fallback."
                        )
            start = perf_counter()
            scores[method][test] = kde_logpdf(models[0], x[test]) - kde_logpdf(models[1], x[test])
            timings.append(
                {
                    "fold": fold_id,
                    "method": method,
                    "fit_seconds": sum(record["fit_seconds"] for record in info),
                    "score_seconds": perf_counter() - start,
                }
            )
            diagnostics.append(
                {
                    "fold": fold_id,
                    "method": method,
                    "n_good": len(good),
                    "n_bad": len(bad),
                    "good": info[0],
                    "bad": info[1],
                }
            )
        if not all(np.isfinite(scores[method][test]).all() for method in METHODS):
            raise ValueError(f"Nonfinite scores in fold {fold_id}")
    active = np.flatnonzero(np.isfinite(scores["kde_optimized"]))
    active_scores = {method: values[active] for method, values in scores.items()}
    reference_check = {}
    if historical is not None:
        for method, column in (("tpe", "tpe_score"), ("kde_reference", "kde_score")):
            expected = [float(historical[trials[i].number][column]) for i in active]
            difference = float(np.max(np.abs(scores[method][active] - expected)))
            reference_check[method + "_max_absolute_difference"] = difference
            if not np.allclose(scores[method][active], expected, atol=1e-12, rtol=1e-12):
                raise ValueError(f"{method} scores differ from historical baseline: {difference}")
    metrics, curves = evaluate(objectives[active], folds[active], active_scores, maximize)
    decile_details, deciles, monotonicity = score_deciles(
        objectives[active], folds[active], active_scores, maximize
    )
    summary = []
    for method in METHODS:
        rows = [row for row in metrics if row["method"] == method]
        times = [row for row in timings if row["method"] == method]
        summary.append(
            {
                "method": method,
                "mean_spearman": float(np.mean([row["spearman"] for row in rows])),
                "sd_spearman": (
                    float(np.std([row["spearman"] for row in rows], ddof=1))
                    if len(rows) > 1
                    else 0.0
                ),
                "mean_objective_top20": float(np.mean([row["top20"] for row in rows])),
                "mean_random": float(np.mean([row["random"] for row in rows])),
                "mean_oracle20": float(np.mean([row["oracle20"] for row in rows])),
                "improvement_over_random": float(np.mean([row["improvement"] for row in rows])),
                "fit_seconds": sum(row["fit_seconds"] for row in times),
                "score_seconds": sum(row["score_seconds"] for row in times),
                **monotonicity[method],
            }
        )
    payload = [
        {
            "trial_number": t.number,
            "objective": t.value,
            "params": t.params,
            "distributions": {
                n: optuna.distributions.distribution_to_json(d) for n, d in t.distributions.items()
            },
        }
        for t in trials
    ]
    metadata = {
        "study_name": study.study_name,
        "study_user_attrs": study.user_attrs,
        "direction": study.direction.name,
        "seed": SEED,
        "n_folds": 5,
        "good_fraction": 0.2,
        "rounding": "ceil",
        "top_k_ties": "trial number ascending",
        "grouping": "identical configurations in selected subspace never cross folds",
        "evaluated_folds": list(map(int, selected_folds)),
        "scored_trials": len(active),
        "reference_oof": str(reference_oof) if reference_oof is not None else None,
        "historical_score_check": reference_check,
        "optimizer": {
            "method": "L-BFGS-B",
            "objective": "exact mean negative LOO log density",
            "jacobian": "analytic",
            "continuous_coordinates": "log bandwidth",
            "continuous_bounds": [1e-6, 10.0],
            "discrete_bounds": [1e-6, 1 - 1e-6],
            "discrete_initial": discrete_initial,
            "maxiter": 200,
            "ftol": 1e-10,
            "gtol": 1e-5,
            "timeout_per_group_seconds": timeout,
            "fallback": "none",
        },
        "reference_regularization": "unchanged: floor 1e-8, discrete upper bound 1-1e-8",
        "deciles": "ascending fold-local score ranks, stable trial-number tie break; array_split",
        "tpe": {
            "multivariate": True,
            "weights": "uniform",
            "prior_weight": 1.0,
            "consider_prior": True,
            "consider_magic_clip": True,
            "consider_endpoints": False,
        },
        "kde_transforms": transforms,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("optuna", "statsmodels", "numpy", "scipy", "matplotlib")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        **audit,
        "kde_folds": diagnostics,
    }
    write_csv(
        output / "oof_scores.csv",
        [
            {
                "trial_number": t.number,
                "objective": t.value,
                "fold": int(folds[i]),
                "tpe_score": scores["tpe"][i],
                "kde_reference_score": scores["kde_reference"][i],
                "kde_optimized_score": scores["kde_optimized"][i],
                "params_json": json.dumps(t.params, sort_keys=True),
            }
            for i, t in enumerate(trials)
            if np.isfinite(scores["kde_optimized"][i])
        ],
    )
    write_csv(output / "fold_metrics.csv", metrics)
    write_csv(output / "timings.csv", timings)
    write_csv(output / "top_k.csv", curves)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "decile_metrics.csv", decile_details)
    write_csv(output / "decile_summary.csv", deciles)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(output, objectives[active], active_scores, curves, deciles)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def robustness(study, output, limit=None, timeout=120, subspace="strict", reference_oof=None):
    """Measure five initializations without selecting or changing any fitted solution.

    Every variant calls the existing benchmark, including its original encoding,
    folds, optimizer and kernels. Unsuccessful finite fits remain visible in this
    diagnostic experiment; they are never relabeled as successful or replaced.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    starts = (0.1, 0.3, 0.5, 0.7, 0.9)
    variants, fits, scores, metrics = {}, {}, {}, {}
    for initial in starts:
        label = str(initial)
        folder = output / f"start_{label}"
        benchmark(
            study,
            folder,
            limit,
            timeout,
            subspace,
            reference_oof,
            discrete_initial=initial,
            allow_unverified=True,
        )
        with (folder / "oof_scores.csv").open() as stream:
            variants[label] = list(csv.DictReader(stream))
        fits[label] = json.loads((folder / "optimization_diagnostics.json").read_text())
        with (folder / "fold_metrics.csv").open() as stream:
            metrics[label] = [r for r in csv.DictReader(stream) if r["method"] == "kde_optimized"]
        scores[label] = np.array([float(r["kde_optimized_score"]) for r in variants[label]])
    baseline = variants["0.5"]
    for rows in variants.values():
        for row, expected in zip(rows, baseline):
            for key in ("trial_number", "fold", "objective", "params_json"):
                if row[key] != expected[key]:
                    raise ValueError(f"Robustness inputs differ: {key}")
        if len(rows) != len(baseline):
            raise ValueError("Robustness trial counts differ")
    folds = np.array([int(r["fold"]) for r in baseline])
    objectives = np.array([float(r["objective"]) for r in baseline])
    maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    pairs = []
    for fold_id in range(5):
        test = folds == fold_id
        for a, first in enumerate(scores):
            for second in list(scores)[a + 1 :]:
                pairs.append(
                    {
                        "fold": fold_id,
                        "start_a": first,
                        "start_b": second,
                        "spearman": float(
                            spearmanr(scores[first][test], scores[second][test]).statistic
                        ),
                        "max_absolute_score_difference": float(
                            np.max(np.abs(scores[first][test] - scores[second][test]))
                        ),
                    }
                )
    details, deciles, monotonicity = score_deciles(objectives, folds, scores, maximize)
    comparisons, bandwidths, summaries = [], [], []
    for label, records in fits.items():
        for record, reference in zip(records, fits["0.5"]):
            if (record["fold"], record["group"], record["n_train"]) != (
                reference["fold"],
                reference["group"],
                reference["n_train"],
            ):
                raise ValueError("LOO comparisons require identical fold/group")
            best_ll = max(
                r["final_mean_loo_loglikelihood"]
                for run in fits.values()
                for r in run
                if (r["fold"], r["group"]) == (record["fold"], record["group"])
            )
            comparisons.append(
                {
                    "start": label,
                    "fold": record["fold"],
                    "group": record["group"],
                    "n_train": record["n_train"],
                    "success": record["success"],
                    "status": record["status"],
                    "message": record["message"],
                    "iterations": record["iterations"],
                    "evaluations": record["evaluations"],
                    "fit_seconds": record["fit_seconds"],
                    "mean_loo_loglikelihood": record["final_mean_loo_loglikelihood"],
                    "sum_loo_loglikelihood": record["final_sum_loo_loglikelihood"],
                    "mean_loo_difference_from_start_0.5": (
                        record["final_mean_loo_loglikelihood"]
                        - reference["final_mean_loo_loglikelihood"]
                    ),
                    "mean_loo_gap_to_best_same_fold_group": best_ll
                    - record["final_mean_loo_loglikelihood"],
                    "lower_bound_hits": sum(record["at_lower_bound"]),
                    "upper_bound_hits": sum(record["at_upper_bound"]),
                }
            )
            for i, name in enumerate(record["parameter_names"]):
                bandwidths.append(
                    {
                        "start": label,
                        "fold": record["fold"],
                        "group": record["group"],
                        "parameter": name,
                        "initial": record["initial_bandwidth"][i],
                        "final": record["bandwidth"][i],
                        "difference_from_start_0.5": record["bandwidth"][i]
                        - reference["bandwidth"][i],
                        "lower": record["lower_bounds"][i],
                        "upper": record["upper_bounds"][i],
                        "at_lower": record["at_lower_bound"][i],
                        "at_upper": record["at_upper_bound"][i],
                    }
                )
        correlations = [float(r["spearman"]) for r in metrics[label]]
        stability = [r["spearman"] for r in pairs if label in (r["start_a"], r["start_b"])]
        summaries.append(
            {
                "start": label,
                "mean_loo_good": float(
                    np.mean(
                        [r["final_mean_loo_loglikelihood"] for r in records if r["group"] == "good"]
                    )
                ),
                "mean_loo_bad": float(
                    np.mean(
                        [r["final_mean_loo_loglikelihood"] for r in records if r["group"] == "bad"]
                    )
                ),
                "max_paired_mean_loo_gap": max(
                    r["mean_loo_gap_to_best_same_fold_group"]
                    for r in comparisons
                    if r["start"] == label
                ),
                "mean_spearman": float(np.mean(correlations)),
                "sd_spearman": float(np.std(correlations, ddof=1)),
                "min_ranking_spearman": min(stability),
                "mean_ranking_spearman": float(np.mean(stability)),
                "converged_fits": sum(r["success"] for r in records),
                "verified_fits": sum(r["optimization_verified"] for r in records),
                "total_fits": len(records),
                "fit_seconds": sum(r["fit_seconds"] for r in records),
                "min_iterations": min(
                    (r["iterations"] for r in records if r["iterations"] is not None),
                    default=None,
                ),
                "max_iterations": max(
                    (r["iterations"] for r in records if r["iterations"] is not None),
                    default=None,
                ),
                **monotonicity[label],
            }
        )
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "ranking_pairs.csv", pairs)
    write_csv(output / "paired_loo.csv", comparisons)
    write_csv(output / "bandwidths.csv", bandwidths)
    write_csv(output / "decile_metrics.csv", details)
    write_csv(output / "decile_summary.csv", deciles)
    write_csv(
        output / "oof_scores.csv",
        [
            {
                "trial_number": row["trial_number"],
                "fold": row["fold"],
                "objective": row["objective"],
                **{f"score_start_{label}": values[i] for label, values in scores.items()},
            }
            for i, row in enumerate(baseline)
        ],
    )
    plot_robustness(output, scores, details, deciles)
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


def plot_robustness(output, scores, details, deciles):
    """Export aggregate and individual-fold performance curves for all starts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for panel, ax in enumerate(axes.flat):
        for label in scores:
            source = deciles if panel == 0 else [r for r in details if r["fold"] == panel - 1]
            selected = [r for r in source if r["method"] == label]
            ax.plot(
                [r["decile"] for r in selected],
                [r["mean_objective"] for r in selected],
                marker="o",
                markersize=3,
                alpha=0.75,
                label=f"start {label}",
            )
        ax.set(
            title="Mean over five folds" if panel == 0 else f"Fold {panel - 1}",
            xlabel="Within-fold score decile (ascending)",
            ylabel="Mean actual objective",
            xticks=range(1, 11),
        )
        ax.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.savefig(output / "robustness_deciles.png", dpi=180)
    fig.savefig(output / "robustness_deciles.svg")
    plt.close(fig)


LEARNING_METHODS = ("tpe", "kde_optimized")


def learning_samples(folds, sizes=(100, 200, 500, 1000, 1600), repetitions=5):
    """Yield shared nested samples; full training sets occur only once per fold."""
    if repetitions < 1 or not sizes or min(sizes) < 10 or len(set(sizes)) != len(sizes):
        raise ValueError("Use distinct training sizes >= 10 and positive repetitions")
    for fold in range(5):
        pool, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if max(sizes) > len(pool) or len(test) < 20:
            raise ValueError("Training sizes exceed fold capacity or test fold is too small")
        for repetition in range(repetitions):
            seed = [SEED, fold, repetition]
            permutation = np.random.default_rng(np.random.SeedSequence(seed)).permutation(pool)
            for size in sorted(sizes):
                if size == len(pool) and repetition:
                    continue
                # Chronological order preserves historical tie handling and numerical results.
                train = np.sort(permutation[:size])
                yield fold, size, repetition, seed, train, test


def learning_metrics(objectives, scores, maximize=True):
    """Evaluate complete test predictions, retaining deterministic trial-order score ties."""
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete/nonfinite test scores")
    rho = float(spearmanr(scores, objectives if maximize else -objectives).statistic)
    if not np.isfinite(rho):
        raise ValueError("Undefined Spearman correlation (constant scores/objectives)")
    bins = np.array_split(np.argsort(scores, kind="stable"), 10)
    means = np.array([np.mean(objectives[members]) for members in bins])
    changes = np.diff(means) * (1 if maximize else -1)
    drops = np.maximum(-changes, 0)
    decile_ids = np.empty(len(scores), dtype=int)
    for decile, members in enumerate(bins, 1):
        decile_ids[members] = decile
    return {
        "spearman": rho,
        "decreases": int(np.count_nonzero(changes < -1e-12)),
        "monotone": int(np.all(changes >= -1e-12)),
        "max_drop": float(np.max(drops)),
        "sum_drops": float(np.sum(drops)),
        "delta_ap": float(means[-1] - means[0]),
    }, means, decile_ids


def learning_curve(study, output, sizes=(100, 200, 500, 1000, 1600), repetitions=5,
                   timeout=120, subspace="strict", reference_oof=None):
    """Compare unchanged estimators on shared subsamples, with explicit failure accounting."""
    import os

    trials, space, keys, audit = prepare(study, subspace=subspace)
    folds = make_folds(keys)
    samples = list(learning_samples(folds, sizes, repetitions))
    objectives = np.array([t.value for t in trials])
    maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    x, var_type, transforms = kde_encoding(trials, space)
    adapter = OptunaParzenAdapter(space)
    observations = adapter.encode(trials)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    historical = None
    if reference_oof is not None:
        with Path(reference_oof).open() as stream:
            historical = {int(r["trial_number"]): r for r in csv.DictReader(stream)}
        if set(historical) != {t.number for t in trials}:
            raise ValueError("Historical trial IDs differ")
        for i, t in enumerate(trials):
            row = historical[t.number]
            if (int(row["fold"]) != folds[i] or float(row["objective"]) != t.value
                    or json.loads(row["params_json"]) != t.params):
                raise ValueError(f"Historical fold/configuration/objective differs: {t.number}")
    payload = [{"trial_number": t.number, "objective": t.value, "params": t.params,
                "distributions": {n: optuna.distributions.distribution_to_json(d)
                                  for n, d in t.distributions.items()}} for t in trials]
    metadata = {
        "study_name": study.study_name, "study_user_attrs": study.user_attrs,
        "direction": study.direction.name, "seed": SEED, "sizes": list(sizes),
        "repetitions": repetitions, "n_folds": 5, "good_fraction": 0.2,
        "sampling": "SeedSequence([42, fold, repetition]); sorted permutation prefixes",
        "ties": "training objectives and test scores: stable trial-number order",
        "full_training_set": "one fit per method/fold; no duplicated repetitions",
        "tpe": {"adapter": "unchanged OptunaParzenAdapter", "multivariate": True,
                "weights": "uniform", "prior_weight": 1.0, "consider_prior": True,
                "consider_magic_clip": True, "consider_endpoints": False,
                "split": "explicit shared ceil(0.2*n) good IDs; sampler gamma bypassed"},
        "kde": {"fit": "unchanged fit_kde defaults, bounded_ml", "discrete_initial": 0.5,
                "timeout_per_group_seconds": timeout, "fallback": None},
        "kde_transforms": transforms,
        "versions": {p: importlib.metadata.version(p)
                     for p in ("optuna", "statsmodels", "numpy", "scipy", "matplotlib")},
        "python": platform.python_version(), "platform": platform.platform(),
        "threads": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "reference_oof": str(reference_oof) if reference_oof else None,
        "uncertainty": "descriptive SD only; overlapping folds/repetitions are dependent",
        "memory": "not measured; no per-method peak-memory claims", **audit,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_csv(output / "trials.csv", [
        {"trial_number": t.number, "fold": int(folds[i]), "objective": t.value,
         "params_json": json.dumps(t.params, sort_keys=True)} for i, t in enumerate(trials)])
    metrics, diagnostics, memberships, predictions, deciles, paired = [], [], [], [], [], []
    stability_cache, reference_checks = {}, []
    metric_names = ("spearman", "decreases", "monotone", "max_drop", "sum_drops", "delta_ap")
    for fold, size, repetition, seed, train, test in samples:
        identity = {"fold": fold, "n_train": size, "repetition": repetition}
        good, bad = split_good_bad(objectives, train, maximize)
        good, bad = np.sort(good), np.sort(bad)
        memberships.append({**identity, "seed": seed,
                            "train_ids": [trials[i].number for i in train],
                            "good_ids": [trials[i].number for i in good],
                            "bad_ids": [trials[i].number for i in bad]})
        query = {n: v[test] for n, v in observations.items()}
        sample_scores, sample_rows = {}, {}
        for method in LEARNING_METHODS:
            scores = np.full(len(test), np.nan)
            info, models, error, score_error = [], [], "", ""
            start = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for group, members in (("good", good), ("bad", bad)):
                    try:
                        if method == "tpe":
                            model = adapter.fit(observations, members)
                            details = {"success": True}
                        else:
                            model, details = fit_kde(x[members], var_type, space, timeout=timeout)
                        models.append(model)
                    except Exception as exc:
                        models.append(None)
                        details = {"success": False, "message": f"{type(exc).__name__}: {exc}"}
                    info.append({"group": group, "n_group": len(members), **details})
                fit_seconds = perf_counter() - start
                fit_success = all(d["success"] for d in info)
                if not fit_success:
                    error = "; ".join(f"{d['group']}: {d.get('message', '')}"
                                      for d in info if not d["success"])
                start = perf_counter()
                if all(m is not None for m in models):
                    try:
                        if method == "tpe":
                            scores = models[0].log_pdf(query) - models[1].log_pdf(query)
                        else:
                            scores = kde_logpdf(models[0], x[test]) - kde_logpdf(models[1], x[test])
                    except Exception as exc:
                        score_error = f"{type(exc).__name__}: {exc}"
                score_seconds = perf_counter() - start
            success = fit_success and not score_error and bool(np.isfinite(scores).all())
            values = dict.fromkeys(metric_names, float("nan"))
            if success:
                try:
                    values, means, bins = learning_metrics(objectives[test], scores, maximize)
                    for decile, mean in enumerate(means, 1):
                        deciles.append({**identity, "method": method, "decile": decile,
                                        "mean_ap": float(mean), "n_test": int(sum(bins == decile))})
                    stability_cache[fold, size, repetition, method] = (scores.copy(), bins)
                except ValueError as exc:
                    success, score_error = False, str(exc)
            row = {**identity, "method": method, "n_good": len(good), "n_bad": len(bad),
                   "n_test": len(test), "fit_success": int(fit_success), "success": int(success),
                   "n_finite": int(np.isfinite(scores).sum()), "error": error or score_error,
                   "fit_seconds": fit_seconds, "score_seconds": score_seconds, **values}
            metrics.append(row)
            sample_rows[method], sample_scores[method] = row, scores
            diagnostics.append({**identity, "method": method, "groups": info,
                                "warnings": [str(w.message) for w in caught],
                                "score_error": score_error})
            if historical is not None and size == len(train) == np.count_nonzero(folds != fold):
                column = "tpe_score" if method == "tpe" else "kde_optimized_score"
                expected = np.array([float(historical[trials[i].number][column]) for i in test])
                reference_checks.append({**identity, "method": method,
                    "max_absolute_difference": float(np.max(np.abs(scores - expected))),
                    "matches": bool(np.allclose(scores, expected, atol=1e-12, rtol=1e-12))})
        valid_pair = all(r["success"] for r in sample_rows.values())
        paired.append({**identity, "valid_pair": int(valid_pair),
                       **{f"delta_{k}": (sample_rows["tpe"][k] - sample_rows["kde_optimized"][k]
                                        if valid_pair else float("nan")) for k in metric_names},
                       "delta_fit_seconds": sample_rows["tpe"]["fit_seconds"]
                                            - sample_rows["kde_optimized"]["fit_seconds"],
                       "delta_score_seconds": sample_rows["tpe"]["score_seconds"]
                                              - sample_rows["kde_optimized"]["score_seconds"]})
        for j, i in enumerate(test):
            predictions.append({**identity, "trial_number": trials[i].number,
                                "objective": float(objectives[i]),
                                **{f"{m}_score": float(sample_scores[m][j]) for m in LEARNING_METHODS},
                                **{f"{m}_valid": sample_rows[m]["success"] for m in LEARNING_METHODS}})
        (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        print(f"Learning curve fold={fold} n={size} repetition={repetition}: "
              + ", ".join(f"{m} rho={sample_rows[m]['spearman']:.4f} "
                          f"ok={sample_rows[m]['success']}" for m in LEARNING_METHODS), flush=True)
    stability = []
    for fold in range(5):
        for size in sorted(sizes):
            for method in LEARNING_METHODS:
                for a in range(repetitions):
                    for b in range(a + 1, repetitions):
                        left = stability_cache.get((fold, size, a, method))
                        right = stability_cache.get((fold, size, b, method))
                        if left is None or right is None:
                            continue
                        stability.append({"fold": fold, "n_train": size, "method": method,
                            "repetition_a": a, "repetition_b": b,
                            "rank_spearman": float(spearmanr(left[0], right[0]).statistic),
                            "decile_changed_fraction": float(np.mean(left[1] != right[1])),
                            "mean_absolute_decile_shift": float(np.mean(abs(left[1] - right[1])))})
    summary = learning_summary(metrics, stability)
    for name, rows in (("fit_metrics", metrics), ("paired_metrics", paired),
                       ("oof_scores", predictions), ("deciles", deciles),
                       ("stability_pairs", stability), ("summary", summary)):
        if rows:
            write_csv(output / f"{name}.csv", rows)
    (output / "samples.json").write_text(json.dumps(memberships) + "\n")
    (output / "reference_checks.json").write_text(json.dumps(reference_checks, indent=2) + "\n")
    plot_learning_curve(output, metrics, deciles, paired, study.study_name.split("_data:")[0])
    return summary


def learning_summary(metrics, stability):
    """Separate within-fold repeat variation from between-fold variation, descriptively."""
    summary = []
    for size in sorted({r["n_train"] for r in metrics}):
        for method in LEARNING_METHODS:
            attempted = [r for r in metrics if r["n_train"] == size and r["method"] == method]
            rows = [r for r in attempted if r["success"]]
            result = {"n_train": size, "method": method, "attempted": len(attempted),
                      "fit_successes": sum(r["fit_success"] for r in attempted),
                      "valid_evaluations": len(rows)}
            for key in ("spearman", "decreases", "monotone", "max_drop", "sum_drops", "delta_ap"):
                values = [r[key] for r in rows]
                result[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
                result[f"sd_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            groups = [[r["spearman"] for r in rows if r["fold"] == f] for f in range(5)]
            means = [np.mean(g) for g in groups if g]
            within = [np.std(g, ddof=1) for g in groups if len(g) > 1]
            result["sd_fold_mean_spearman"] = float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
            result["mean_within_fold_sd_spearman"] = float(np.mean(within)) if within else float("nan")
            for key in ("fit_seconds", "score_seconds"):
                result[f"mean_{key}"] = float(np.mean([r[key] for r in attempted]))
            pairs = [r for r in stability if r["n_train"] == size and r["method"] == method]
            result["n_stability_pairs"] = len(pairs)
            for key in ("rank_spearman", "decile_changed_fraction", "mean_absolute_decile_shift"):
                result[f"mean_{key}"] = float(np.mean([r[key] for r in pairs])) if pairs else float("nan")
            summary.append(result)
    return summary


def plot_learning_curve(output, metrics, deciles, paired, title):
    """Export shared method curves and individual fold/repetition decile curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({r["n_train"] for r in metrics})
    colors = {"tpe": "#d47716", "kde_optimized": "#1875b5"}

    def save(fig, name):
        for extension in ("png", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for method in LEARNING_METHODS:
        mean, sd = [], []
        for size in sizes:
            rows = [r for r in metrics if r["method"] == method and r["n_train"] == size and r["success"]]
            values = [r["spearman"] for r in rows]
            mean.append(np.mean(values) if values else np.nan)
            sd.append(np.std(values, ddof=1) if len(values) > 1 else np.nan)
            for fold in range(5):
                values_fold = [r["spearman"] for r in rows if r["fold"] == fold]
                ax.scatter([size] * len(values_fold), values_fold, s=10, alpha=0.18, color=colors[method])
        ax.plot(sizes, mean, "o-", color=colors[method], label=LABELS[method])
        ax.fill_between(sizes, np.array(mean) - sd, np.array(mean) + sd, color=colors[method], alpha=0.12)
    ax.set(xlabel="Historical training trials", ylabel="Spearman correlation: score vs actual AP",
           title=f"{title}: learning curves (shading: descriptive +/- 1 SD, not CI)", xticks=sizes)
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, "learning_curve")
    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 4.5), sharey=True, constrained_layout=True)
    for ax, size in zip(np.atleast_1d(axes), sizes):
        for method in LEARNING_METHODS:
            rows = [r for r in deciles if r["n_train"] == size and r["method"] == method]
            mean, sd = [], []
            for d in range(1, 11):
                values = [r["mean_ap"] for r in rows if r["decile"] == d]
                mean.append(np.mean(values) if values else np.nan)
                sd.append(np.std(values, ddof=1) if len(values) > 1 else np.nan)
            ax.plot(range(1, 11), mean, "o-", markersize=3, color=colors[method], label=LABELS[method])
            ax.fill_between(range(1, 11), np.array(mean) - sd, np.array(mean) + sd,
                            color=colors[method], alpha=0.12)
        ax.set(title=f"{size} training trials", xlabel="Score decile (ascending)", xticks=[1, 3, 5, 7, 10])
        ax.grid(alpha=0.2)
    np.atleast_1d(axes)[0].set_ylabel("Mean actual AP")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"{title}: score-performance curves; descriptive +/- 1 SD, not CI")
    save(fig, "score_performance")
    fig, axes = plt.subplots(5, len(sizes), figsize=(4 * len(sizes), 13), sharey=True,
                             squeeze=False, constrained_layout=True)
    for fold in range(5):
        for col, size in enumerate(sizes):
            ax = axes[fold, col]
            for method in LEARNING_METHODS:
                for rep in sorted({r["repetition"] for r in deciles}):
                    rows = [r for r in deciles if r["fold"] == fold and r["n_train"] == size
                            and r["method"] == method and r["repetition"] == rep]
                    if rows:
                        ax.plot([r["decile"] for r in rows], [r["mean_ap"] for r in rows],
                                color=colors[method], alpha=0.45, linewidth=1,
                                label=LABELS[method] if rep == 0 else None)
            ax.set(title=f"Fold {fold}, n={size}", xticks=[1, 5, 10])
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{title}: all individual fold/repetition decile curves")
    save(fig, "individual_deciles")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for size in sizes:
        values = [r["delta_spearman"] for r in paired if r["n_train"] == size and r["valid_pair"]]
        ax.scatter([size] * len(values), values, s=18, alpha=0.4, color="#754098")
        if values:
            ax.plot(size, np.mean(values), "D", color="black")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(xlabel="Historical training trials", ylabel="Paired Spearman difference: TPE - KDE",
           title=f"{title}: paired comparisons (black diamonds: means)", xticks=sizes)
    ax.grid(alpha=0.2)
    save(fig, "paired_spearman")


def main():
    """Expose SQLite and DeepCAVE Optuna snapshot inputs with explicit experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="SQLite DB or trusted study pickle/directory"
    )
    parser.add_argument("--study-name", help="Required with SQLite input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Objective-independent pilot sample, e.g. 200")
    parser.add_argument(
        "--bandwidth-timeout",
        type=float,
        default=120,
        help="Bounded LOO seconds per good/bad fit; failure is explicit, no fallback (POSIX)",
    )
    parser.add_argument(
        "--fold", type=int, choices=range(5), help="Run one original fold as a pilot"
    )
    parser.add_argument(
        "--reference-oof",
        type=Path,
        help="Historical OOF CSV: require matching trials, folds, TPE/reference scores",
    )
    parser.add_argument("--subspace", choices=("strict", "intersection"), default="strict")
    parser.add_argument("--learning-curve", action="store_true",
                        help="Paired TPE/KDE comparison at 100/200/500/1000/1600 training trials")
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run all five discrete starts and compare identical OOF predictions",
    )
    args = parser.parse_args()
    if args.bandwidth_timeout <= 0 or (args.limit is not None and args.limit < 100):
        parser.error("Use a positive timeout and at least 100 pilot trials")
    study = load_study(args.source, args.study_name)
    if args.learning_curve:
        if args.robustness or args.fold is not None or args.limit is not None:
            parser.error("--learning-curve uses all five original folds; omit robustness/fold/limit")
        learning_curve(study, args.output, timeout=args.bandwidth_timeout,
                       subspace=args.subspace, reference_oof=args.reference_oof)
        return
    if args.robustness:
        if args.fold is not None:
            parser.error("--robustness evaluates all five folds; omit --fold")
        robustness(
            study,
            args.output,
            args.limit,
            args.bandwidth_timeout,
            args.subspace,
            args.reference_oof,
        )
        return
    benchmark(
        study,
        args.output,
        args.limit,
        args.bandwidth_timeout,
        args.subspace,
        args.reference_oof,
        args.fold,
    )


if __name__ == "__main__":
    main()
