"""Validate unchanged TPE/KDE scores in missing and sparsely covered regions.

Run from the repository root with PYTHONPATH=. and one BLAS/OpenMP thread.
Phase B prepares scored, diverse candidates, but never silently substitutes CPU
training for the historical GPU objective. All sources are opened read-only.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from scipy.stats import rankdata, spearmanr

from deepcave.evaluators.combinatorial import CoverageSpec
from deepcave.evaluators.coverage import normalize_configurations, sample_reference_design
from deepcave.evaluators.coverage_planning import _candidates
from deepcave.evaluators.coverage_state import CoverageState
from examples.api import benchmark_tpe_kde as scoring

METHODS = ("tpe", "kde_optimized")
VIEWS = tuple((resolution, strength) for resolution in (4, 8) for strength in (2, 3, 4))
POPULATIONS = ("all", "regular", "coverage_seeking", "dense_regular", "rare_seeking")
REFERENCE_SEED = 20261023
REFERENCE_COUNT = 5000
RARE_LIMIT = 5
MIN_BIN_SIZE = 20


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configspace_from_distributions(space):
    """Use the existing Optuna converter's bounds/log/step mapping, without loading a run."""
    result = ConfigurationSpace()
    for name, distribution in space.items():
        if isinstance(distribution, CategoricalDistribution):
            parameter = Categorical(name, items=distribution.choices)
        elif isinstance(distribution, (FloatDistribution, IntDistribution)):
            factory = Float if isinstance(distribution, FloatDistribution) else Integer
            parameter = factory(
                name,
                bounds=(distribution.low, distribution.high),
                log=distribution.log,
                meta={"step": distribution.step},
            )
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        result.add(parameter)
    return result


def coverage_profile(configspace, history, candidates, references, resolution, strength):
    """Reuse Lighthouse tuple IDs/gains; count rare occupied cells on the same grid."""
    spec = CoverageSpec(normalize_configurations(configspace, history), resolution)
    state = CoverageState(spec, strength)
    counts = np.bincount(state.tuple_ids(spec.points).ravel(), minlength=state.total)

    def exposure(values):
        points = spec.encode(values)
        occupancy = counts[state.tuple_ids(points)]
        missing = np.count_nonzero(occupancy == 0, axis=1)
        rare = np.count_nonzero((occupancy >= 1) & (occupancy <= RARE_LIMIT), axis=1)
        np.testing.assert_array_equal(missing, state.gains(points))
        return {
            "missing": missing,
            "rare": rare,
            "minimum_count": occupancy.min(axis=1),
            "mean_count": occupancy.mean(axis=1),
        }

    reference = exposure(references)
    covered_rare = reference["rare"][reference["missing"] == 0]
    low, high = (
        np.quantile(covered_rare, [0.25, 0.75], method="inverted_cdf").astype(int)
        if len(covered_rare)
        else (-1, -1)
    )
    description = {
        "resolution": resolution,
        "strength": strength,
        "parameter_groups": len(state.groups),
        "total_tuples": state.total,
        "missing_tuples": int(state.missing.sum()),
        "coverage": float(1 - state.missing.mean()),
        "reference_missing_fraction": float(np.mean(reference["missing"] > 0)),
        "reference_missing_count_mean": float(np.mean(reference["missing"])),
        "rare_limit": RARE_LIMIT,
        "rare_lower_quartile": int(low),
        "rare_upper_quartile": int(high),
        "reference_covered_count": len(covered_rare),
        "discretization": spec.serialize(),
        "occupancy_histogram": np.bincount(counts).tolist(),
    }
    return exposure(candidates), description, spec, state, counts


def populations(exposure, profile):
    """Define groups from coverage alone; rare-only groups contain no missing tuple."""
    missing, rare = exposure["missing"], exposure["rare"]
    low, high = profile["rare_lower_quartile"], profile["rare_upper_quartile"]
    return {
        "all": np.ones(len(missing), dtype=bool),
        "regular": missing == 0,
        "coverage_seeking": missing > 0,
        "dense_regular": (missing == 0) & (rare <= low),
        "rare_seeking": (missing == 0) & ((rare >= high) if high > low else (rare > high)),
    }


def choose_primary(profiles):
    """Choose a shared grid from training-only reference group sizes, before test outcomes."""
    choices = []
    for resolution, strength in VIEWS:
        rows = [r for r in profiles if r["resolution"] == resolution and r["strength"] == strength]
        fractions = [r["reference_missing_fraction"] for r in rows]
        balanced = all(0.15 <= value <= 0.85 for value in fractions)
        choices.append(
            (not balanced, float(np.mean(np.abs(np.array(fractions) - 0.5))), resolution, strength)
        )
    _, _, resolution, strength = min(choices)
    return resolution, strength


def fit_scores(trials, space, train, queries):
    """Fit the original estimators on shared chronological good/bad groups, with no tuning."""
    objectives = np.array([t.value for t in trials])
    good, bad = scoring.split_good_bad(objectives, np.sort(train), maximize=True)
    good, bad = np.sort(good), np.sort(bad)
    adapter = scoring.OptunaParzenAdapter(space)
    observations = adapter.encode(trials)
    query = adapter.encode(queries)
    x, var_type, _ = scoring.kde_encoding(trials, space)
    query_x, _, _ = scoring.kde_encoding(queries, space)
    scores, diagnostics = {}, []
    for method in METHODS:
        models, groups = [], []
        start = perf_counter()
        for group, indices in (("good", good), ("bad", bad)):
            if method == "tpe":
                model, info = adapter.fit(observations, indices), {"success": True}
            else:
                model, info = scoring.fit_kde(x[indices], var_type, space)
                if not info["success"] or not info["optimization_verified"]:
                    raise RuntimeError(f"Unverified KDE fit: {info}")
            models.append(model)
            groups.append(
                {"group": group, "trial_ids": [trials[i].number for i in indices], **info}
            )
        fit_seconds = perf_counter() - start
        start = perf_counter()
        scores[method] = (
            models[0].log_pdf(query) - models[1].log_pdf(query)
            if method == "tpe"
            else scoring.kde_logpdf(models[0], query_x) - scoring.kde_logpdf(models[1], query_x)
        )
        if not np.isfinite(scores[method]).all():
            raise FloatingPointError(f"Nonfinite {method} score")
        diagnostics.append(
            {
                "method": method,
                "fit_seconds": fit_seconds,
                "score_seconds": perf_counter() - start,
                "groups": groups,
            }
        )
    return scores, diagnostics


def evaluation(values, scores, n_bins):
    """Avoid small artificial deciles; correlations below 30 observations are exploratory."""
    n = len(values)
    rho = float(spearmanr(scores, values).statistic) if n >= 10 else float("nan")
    result = {
        "n": n,
        "spearman": rho,
        "exploratory": int(n < 30),
        "n_bins": n_bins,
        "objective_mean": float(np.mean(values)) if n else float("nan"),
        "objective_sd": float(np.std(values, ddof=1)) if n > 1 else float("nan"),
        "objective_iqr": (
            float(np.diff(np.quantile(values, [0.25, 0.75]))[0]) if n else float("nan")
        ),
        **dict.fromkeys(("decreases", "monotone", "max_drop", "sum_drops", "spread"), float("nan")),
    }
    bins = []
    if n_bins >= 3:
        if n // n_bins < MIN_BIN_SIZE:
            raise ValueError("Insufficient observations per score bin")
        parts = np.array_split(np.argsort(scores, kind="stable"), n_bins)
        means = np.array([np.mean(values[p]) for p in parts])
        changes = np.diff(means)
        result.update(
            decreases=int(sum(changes < -1e-12)),
            monotone=int(np.all(changes >= -1e-12)),
            max_drop=float(max(np.maximum(-changes, 0))),
            sum_drops=float(sum(np.maximum(-changes, 0))),
            spread=float(means[-1] - means[0]),
        )
        bins = [
            {
                "bin": b + 1,
                "n_bins": n_bins,
                "rank_fraction": (b + 0.5) / n_bins,
                "n": len(part),
                "mean_ap": float(means[b]),
            }
            for b, part in enumerate(parts)
        ]
    return result, bins


def safe_rho(left, right):
    if len(left) < 10 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def shared_bin_counts(contexts):
    """Use equal regular/seeking curve resolution without looking at outcomes."""
    counts = {}
    for resolution, strength in VIEWS:
        for population in POPULATIONS:
            minimum_n = min(
                int(contexts[f, resolution, strength][1][population].sum()) for f in range(5)
            )
            number = min(10, minimum_n // MIN_BIN_SIZE)
            counts[resolution, strength, population] = number if number >= 3 else 0
        common = min(counts[resolution, strength, p] for p in ("regular", "coverage_seeking"))
        for population in ("regular", "coverage_seeking"):
            counts[resolution, strength, population] = common
    return counts


def summary_rows(metrics):
    summary = []
    for resolution, strength in VIEWS:
        for population in POPULATIONS:
            for method in METHODS:
                rows = [
                    r
                    for r in metrics
                    if (r["resolution"], r["strength"], r["population"], r["method"])
                    == (resolution, strength, population, method)
                ]
                record = {
                    "resolution": resolution,
                    "strength": strength,
                    "population": population,
                    "method": method,
                    "n_configurations": sum(r["n"] for r in rows),
                    "n_folds_with_rho": sum(np.isfinite(r["spearman"]) for r in rows),
                    "n_curves": sum(r["n_bins"] >= 3 for r in rows),
                }
                for key in ("spearman", "decreases", "monotone", "max_drop", "sum_drops", "spread"):
                    values = [r[key] for r in rows if np.isfinite(r[key])]
                    record[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
                    record[f"sd_{key}"] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
                    )
                summary.append(record)
    return summary


def phase_a(study, output, reference_oof):
    """Lock coverage design without held-out targets, then fit and evaluate both methods."""
    trials, space, keys, audit = scoring.prepare(study)
    if len(trials) != 2000 or study.direction.name != "MAXIMIZE":
        raise ValueError("This protocol requires 2000 finite random trials and maximized AP")
    folds = scoring.make_folds(keys)
    configspace = configspace_from_distributions(space)
    references = sample_reference_design(configspace, REFERENCE_COUNT, REFERENCE_SEED).values
    output.mkdir(parents=True, exist_ok=True)
    profiles, contexts = [], {}
    for fold in range(5):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        for resolution, strength in VIEWS:
            exposure, profile, _, _, _ = coverage_profile(
                configspace,
                [trials[i].params for i in train],
                [trials[i].params for i in test],
                references,
                resolution,
                strength,
            )
            profile = {"fold": fold, **profile}
            profiles.append(profile)
            contexts[fold, resolution, strength] = (exposure, populations(exposure, profile))
    primary = choose_primary(profiles)
    bin_counts = shared_bin_counts(contexts)
    design = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_resolution": primary[0],
        "primary_strength": primary[1],
        "reference_seed": REFERENCE_SEED,
        "reference_count": REFERENCE_COUNT,
        "rare_occupancy": [1, RARE_LIMIT],
        "views": [list(v) for v in VIEWS],
        "selection": "prefer views with 15%-85% missing-seeking reference points in every fold; "
        "then minimize mean absolute distance to 50%; ties by resolution, strength",
        "regular": "no currently missing tuple in this view; can still contain rare occupied tuples",
        "coverage_seeking": "at least one currently missing tuple; independent one-trial gain",
        "rare_seeking": "no missing tuple and rare count in upper reference quartile; strict > if tied",
        "dense_regular": "no missing tuple and rare count in lower reference quartile",
        "bin_rule": "shared across folds and regular/seeking: min(10, floor(min group n / 20)); "
        "omit curves below 3 bins; rare-only subgroups use their own common fold resolution",
        "design_locked_before": "test objectives, KDE/TPE scores and scoring quality are inspected",
        "bins": [
            {"resolution": r, "strength": t, "population": p, "n_bins": n}
            for (r, t, p), n in bin_counts.items()
        ],
    }
    write_json(output / "design.json", design)
    write_json(output / "training_coverage.json", profiles)
    historical = {int(r["trial_number"]): r for r in read_csv(reference_oof)}
    if set(historical) != {t.number for t in trials}:
        raise ValueError("Historical trial IDs differ")
    metadata = {
        "study_name": study.study_name,
        "study_user_attrs": study.user_attrs,
        **audit,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("optuna", "statsmodels", "numpy", "scipy", "matplotlib", "ConfigSpace")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(scoring.__file__),
                Path("deepcave/evaluators/combinatorial.py"),
                Path("deepcave/evaluators/coverage_state.py"),
                Path("deepcave/evaluators/coverage_planning.py"),
            )
        },
        "reference_oof": str(reference_oof),
        "inference": "descriptive fold means/SD and paired differences; folds are dependent",
    }
    write_json(output / "metadata.json", metadata)
    scores_rows, diagnostics, coverage_rows, metrics, bin_rows, error_rows = [], [], [], [], [], []
    reference_differences = []
    for fold in range(5):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        scores, info = fit_scores(trials, space, train, [trials[i] for i in test])
        diagnostics.extend({"fold": fold, **record} for record in info)
        actual = np.array([trials[i].value for i in test])
        actual_rank = (rankdata(actual) - 0.5) / len(test)
        rank_error = {
            m: abs((rankdata(scores[m]) - 0.5) / len(test) - actual_rank) for m in METHODS
        }
        for j, i in enumerate(test):
            old = historical[trials[i].number]
            assert int(old["fold"]) == fold and float(old["objective"]) == trials[i].value
            assert json.loads(old["params_json"]) == trials[i].params
            scores_rows.append(
                {
                    "trial_number": trials[i].number,
                    "fold": fold,
                    "objective": trials[i].value,
                    "params_json": json.dumps(trials[i].params, sort_keys=True),
                    **{f"{m}_score": float(scores[m][j]) for m in METHODS},
                    **{f"{m}_rank_error": float(rank_error[m][j]) for m in METHODS},
                }
            )
        for method in METHODS:
            expected = np.array(
                [float(historical[trials[i].number][f"{method}_score"]) for i in test]
            )
            error = float(max(abs(scores[method] - expected)))
            if not np.allclose(scores[method], expected, atol=1e-12, rtol=1e-12):
                raise ValueError(f"Historical {method} scores differ: {error}")
            reference_differences.append(
                {"fold": fold, "method": method, "max_absolute_difference": error}
            )
        for resolution, strength in VIEWS:
            identity = {"fold": fold, "resolution": resolution, "strength": strength}
            exposure, masks = contexts[fold, resolution, strength]
            for j, i in enumerate(test):
                coverage_rows.append(
                    {
                        **identity,
                        "trial_number": trials[i].number,
                        "missing_gain": int(exposure["missing"][j]),
                        "rare_count": int(exposure["rare"][j]),
                        "minimum_occupancy": int(exposure["minimum_count"][j]),
                        "mean_occupancy": float(exposure["mean_count"][j]),
                        **{p: int(mask[j]) for p, mask in masks.items()},
                    }
                )
            for population, mask in masks.items():
                for method in METHODS:
                    row, bins = evaluation(
                        actual[mask],
                        scores[method][mask],
                        bin_counts[resolution, strength, population],
                    )
                    metrics.append({**identity, "population": population, "method": method, **row})
                    bin_rows.extend(
                        {**identity, "population": population, "method": method, **b} for b in bins
                    )
            for method in METHODS:
                error_rows.append(
                    {
                        **identity,
                        "method": method,
                        "rho_missing_gain_rank_error": safe_rho(
                            exposure["missing"], rank_error[method]
                        ),
                        "rho_rare_count_rank_error": safe_rho(exposure["rare"], rank_error[method]),
                        "mean_error_regular": (
                            float(np.mean(rank_error[method][masks["regular"]]))
                            if masks["regular"].any()
                            else float("nan")
                        ),
                        "mean_error_seeking": (
                            float(np.mean(rank_error[method][masks["coverage_seeking"]]))
                            if masks["coverage_seeking"].any()
                            else float("nan")
                        ),
                    }
                )
        print(
            f"Phase A {study.study_name.split('_data:')[0]} fold {fold}: complete; primary={primary}",
            flush=True,
        )
    by_key = {
        (r["fold"], r["resolution"], r["strength"], r["population"], r["method"]): r
        for r in metrics
    }
    paired = []
    for row in metrics:
        if row["method"] != "tpe":
            continue
        other = by_key[
            row["fold"], row["resolution"], row["strength"], row["population"], "kde_optimized"
        ]
        paired.append(
            {k: row[k] for k in ("fold", "resolution", "strength", "population", "n")}
            | {
                f"delta_{k}": row[k] - other[k]
                for k in ("spearman", "decreases", "monotone", "max_drop", "spread")
            }
        )
    group_differences = []
    for fold in range(5):
        for resolution, strength in VIEWS:
            for method in METHODS:
                regular = by_key[fold, resolution, strength, "regular", method]
                seeking = by_key[fold, resolution, strength, "coverage_seeking", method]
                group_differences.append(
                    {
                        "fold": fold,
                        "resolution": resolution,
                        "strength": strength,
                        "method": method,
                        "delta_spearman_seeking_minus_regular": seeking["spearman"]
                        - regular["spearman"],
                        "n_regular": regular["n"],
                        "n_seeking": seeking["n"],
                    }
                )
    for filename, rows in (
        ("oof_scores", scores_rows),
        ("oof_coverage", coverage_rows),
        ("fold_metrics", metrics),
        ("score_bins", bin_rows),
        ("paired_metrics", paired),
        ("group_differences", group_differences),
        ("coverage_rank_errors", error_rows),
        ("summary", summary_rows(metrics)),
    ):
        scoring.write_csv(output / f"{filename}.csv", rows)
    write_json(output / "fit_diagnostics.json", diagnostics)
    write_json(output / "reference_checks.json", reference_differences)
    return trials, space, configspace, primary


def select_diverse(pool, count=25):
    """Predeclare score/coverage extremes, then maximin in rank and disagreement space."""
    if len(pool) < count:
        raise ValueError("Candidate pool smaller than evaluation sample")
    features = np.column_stack(
        [
            rankdata([r[k] for r in pool]) / len(pool)
            for k in ("tpe_score", "kde_optimized_score", "missing_gain")
        ]
    )
    features = np.column_stack([features, abs(features[:, 0] - features[:, 1])])
    selected, reasons = [], {}
    for column in range(4):
        for index, side in (
            (int(np.argmin(features[:, column])), "min"),
            (int(np.argmax(features[:, column])), "max"),
        ):
            if index not in selected:
                selected.append(index)
                reasons[index] = f"anchor feature {column} {side}"
    while len(selected) < count:
        distance = np.min(
            np.sum((features[:, None, :] - features[selected][None, :, :]) ** 2, axis=2), axis=1
        )
        distance[selected] = -1
        index = int(np.argmax(distance))
        selected.append(index)
        reasons[index] = "maximin score/coverage/disagreement ranks"
    return [
        {
            **pool[i],
            "selection_order": position + 1,
            "selection_reason": reasons[i],
            "tpe_rank_fraction": float(features[i, 0]),
            "kde_rank_fraction": float(features[i, 1]),
            "coverage_rank_fraction": float(features[i, 2]),
            "rank_disagreement": float(features[i, 3]),
            "objective": "",
            "evaluation_status": "not_executed",
        }
        for position, i in enumerate(selected)
    ]


def prospective_candidates(trials, space, configspace, primary, output):
    """Freeze full-history scores and a diverse 25-point test sample before any training."""
    history = [t.params for t in trials]
    spec = CoverageSpec(normalize_configurations(configspace, history), primary[0])
    state = CoverageState(spec, primary[1])
    counts = np.bincount(state.tuple_ids(spec.points).ravel(), minlength=state.total)
    rng = np.random.default_rng(REFERENCE_SEED + 1)
    seen = {json.dumps(p, sort_keys=True) for p in history}
    proposals = []
    for _ in range(100):
        if not state.missing.any():
            raise ValueError("No missing tuples at the chosen primary view")
        points = _candidates(spec, state, rng)
        for point in points[rng.permutation(len(points))]:
            if state.gains(point[None, :])[0] == 0:
                continue
            params, encoded = spec.materialize(point, rng)
            for name, distribution in space.items():
                if not distribution._contains(distribution.to_internal_repr(params[name])):
                    raise ValueError(f"Invalid Optuna value: {name}={params[name]}")
            key = json.dumps(params, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            occupancy = counts[state.tuple_ids(encoded)[0]]
            proposals.append(
                {
                    "candidate_id": len(proposals),
                    "params_json": key,
                    "resolution": primary[0],
                    "strength": primary[1],
                    "missing_gain": int(sum(occupancy == 0)),
                    "rare_count": int(sum((occupancy >= 1) & (occupancy <= RARE_LIMIT))),
                }
            )
            if len(proposals) == 512:
                break
        if len(proposals) == 512:
            break
    scores, diagnostics = fit_scores(
        trials,
        space,
        np.arange(len(trials)),
        [SimpleNamespace(params=json.loads(p["params_json"])) for p in proposals],
    )
    pool = [
        {**row, **{f"{m}_score": float(scores[m][i]) for m in METHODS}}
        for i, row in enumerate(proposals)
    ]
    scoring.write_csv(output / "prospective_pool.csv", pool)
    selected = select_diverse(pool)
    scoring.write_csv(output / "prospective_selected.csv", selected)
    write_json(
        output / "prospective_design.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": REFERENCE_SEED + 1,
            "history_count": len(trials),
            "resolution": primary[0],
            "strength": primary[1],
            "pool_count": len(pool),
            "selected_count": len(selected),
            "generator": "unchanged Lighthouse _candidates, CoverageState and CoverageSpec.materialize",
            "gain": "each candidate independently against the same frozen full history; not cumulative",
            "selection": "anchor min/max TPE rank, KDE rank, coverage rank, absolute rank disagreement; "
            "then maximin squared distance in those four rank coordinates, ties by candidate ID",
            "model_fit_diagnostics": diagnostics,
            "scores_saved_before_training": True,
            "maximum_trials_per_study": 25,
            "proposed_wall_budget_per_study_seconds": 1800,
            "execution_status": "not_executed; see phase_b_feasibility.json",
        },
    )
    return selected


def feasibility(source, studies, output):
    """Audit original GPU requirements and durations without touching the source studies."""
    root = Path(source).parent
    gpu = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if shutil.which("nvidia-smi")
        else None
    )
    records = []
    with sqlite3.connect((root / "mlflow.db").as_uri() + "?mode=ro", uri=True) as db:
        for family, study in studies.items():
            first = study.get_trials(deepcopy=False)[0]
            run_id = first.user_attrs["mlflow_run_id"]
            params = dict(db.execute("select key,value from params where run_uuid=?", (run_id,)))
            durations = [
                t.duration.total_seconds() for t in study.get_trials(deepcopy=False) if t.duration
            ]
            records.append(
                {
                    "family": family,
                    "historical_run_id": run_id,
                    "historical_params": params,
                    "historical_duration_median_seconds": float(np.median(durations)),
                    "historical_duration_p90_seconds": float(np.quantile(durations, 0.9)),
                    "estimated_25_trial_median_seconds": float(25 * np.median(durations)),
                    "estimated_25_trial_p90_seconds": float(25 * np.quantile(durations, 0.9)),
                    "original_dataset_present": (root / params["data_path"]).exists(),
                }
            )
    result = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "nvidia_smi_returncode": gpu.returncode if gpu else None,
        "nvidia_smi_output": (gpu.stdout + gpu.stderr).strip() if gpu else "not installed",
        "nvidia_devices": [str(p) for p in Path("/dev").glob("nvidia*")],
        "current_environment_packages": {
            p: importlib.util.find_spec(p) is not None for p in ("xgboost", "catboost")
        },
        "source_environment_exists": (root / ".venv/bin/python").exists(),
        "studies": records,
        "status": "not_executed",
        "actual_new_training_trials": 0,
        "reason": "Historical objectives explicitly require CUDA/GPU; no usable NVIDIA driver/device "
        "is accessible in this execution environment. No CPU substitution or altered objective.",
        "source_studies_modified": False,
    }
    write_json(output / "phase_b_feasibility.json", result)
    return result


def plot_results(root):
    """Show paired fold correlations and rank-local performance without pooling score scales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"tpe": "#d47716", "kde_optimized": "#1875b5"}
    labels = {"tpe": "Optuna TPE", "kde_optimized": "Optimized KDE"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    for ax, family in zip(axes, ("xgboost", "catboost")):
        folder = root / family
        design = json.loads((folder / "design.json").read_text())
        primary = (design["primary_resolution"], design["primary_strength"])
        rows = [
            r
            for r in read_csv(folder / "fold_metrics.csv")
            if (int(r["resolution"]), int(r["strength"])) == primary
        ]
        for position, population in enumerate(("regular", "coverage_seeking")):
            for method, offset in (("tpe", -0.12), ("kde_optimized", 0.12)):
                values = [
                    float(r["spearman"])
                    for r in rows
                    if r["population"] == population and r["method"] == method
                ]
                ax.scatter(
                    np.full(len(values), position + offset),
                    values,
                    s=25,
                    alpha=0.6,
                    color=colors[method],
                )
                ax.errorbar(
                    position + offset,
                    np.mean(values),
                    yerr=np.std(values, ddof=1),
                    fmt="D",
                    capsize=5,
                    color=colors[method],
                    label=labels[method] if position == 0 else None,
                )
            for fold in range(5):
                paired = [
                    next(
                        float(r["spearman"])
                        for r in rows
                        if r["population"] == population
                        and r["method"] == m
                        and int(r["fold"]) == fold
                    )
                    for m in METHODS
                ]
                ax.plot(
                    [position - 0.12, position + 0.12],
                    paired,
                    color="gray",
                    alpha=0.25,
                    linewidth=0.7,
                )
        ax.set(
            title=f"{family}: {primary[0]} levels, strength {primary[1]}",
            xticks=[0, 1],
            xticklabels=["Regular (all tuples seen)", "Coverage-seeking (missing tuples)"],
            ylabel="Spearman correlation: score vs actual AP",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Ranking quality by coverage; error bars = descriptive fold SD, not CI")
    for suffix in ("png", "svg"):
        fig.savefig(root / f"ranking_by_coverage.{suffix}", dpi=180)
    plt.close(fig)
    for family in ("xgboost", "catboost"):
        folder = root / family
        design = json.loads((folder / "design.json").read_text())
        primary = (design["primary_resolution"], design["primary_strength"])
        coverage = [
            r
            for r in read_csv(folder / "oof_coverage.csv")
            if (int(r["resolution"]), int(r["strength"])) == primary
        ]
        score_rows = {int(r["trial_number"]): r for r in read_csv(folder / "oof_scores.csv")}
        bins = [
            r
            for r in read_csv(folder / "score_bins.csv")
            if (int(r["resolution"]), int(r["strength"])) == primary
        ]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
        for ax, population in zip(axes, ("regular", "coverage_seeking")):
            selected = [r for r in coverage if r[population] == "1"]
            for method in METHODS:
                for fold in range(5):
                    rows = [
                        score_rows[int(r["trial_number"])]
                        for r in selected
                        if int(r["fold"]) == fold
                    ]
                    rank = (rankdata([float(r[f"{method}_score"]) for r in rows]) - 0.5) / len(rows)
                    ax.scatter(
                        rank,
                        [float(r["objective"]) for r in rows],
                        s=6,
                        alpha=0.12,
                        color=colors[method],
                    )
                    local = [
                        r
                        for r in bins
                        if r["population"] == population
                        and r["method"] == method
                        and int(r["fold"]) == fold
                    ]
                    ax.plot(
                        [float(r["rank_fraction"]) for r in local],
                        [float(r["mean_ap"]) for r in local],
                        color=colors[method],
                        alpha=0.22,
                        linewidth=0.9,
                    )
                local = [r for r in bins if r["population"] == population and r["method"] == method]
                positions = sorted({float(r["rank_fraction"]) for r in local})
                means = [
                    np.mean([float(r["mean_ap"]) for r in local if float(r["rank_fraction"]) == x])
                    for x in positions
                ]
                ax.plot(
                    positions, means, "o-", color=colors[method], label=labels[method], linewidth=2
                )
            ax.set(
                title=f"{population.replace('_', ' ')} (n={len(selected)})",
                xlabel="Within-fold/population score rank fraction",
                ylabel="Actual AP",
            )
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle(
            f"{family}: raw observations, individual fold curves and means; >=20 points/bin"
        )
        for suffix in ("png", "svg"):
            fig.savefig(folder / f"score_performance.{suffix}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("../streamingmodel_NonRouterHHs/optuna.db")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/validation/tpe_kde/coverage_regions")
    )
    args = parser.parse_args()
    inventory = json.loads(
        Path("docs/validation/tpe_kde/robustness/study_inventory.json").read_text()
    )
    studies = {}
    for family, study_id in (("xgboost", 1), ("catboost", 6)):
        name = next(s["study_name"] for s in inventory["studies"] if s["study_id"] == study_id)
        studies[family] = scoring.load_study(args.source, name)
        trials, space, configspace, primary = phase_a(
            studies[family],
            args.output / family,
            Path(f"docs/validation/tpe_kde/robustness/{family}/start_0.5/oof_scores.csv"),
        )
        print(f"Phase A {family} complete", flush=True)
        prospective_candidates(trials, space, configspace, primary, args.output / family)
        print(f"Phase B {family}: pool and diverse sample scored and saved", flush=True)
    feasibility(args.source.resolve(), studies, args.output)
    plot_results(args.output)


if __name__ == "__main__":
    main()
