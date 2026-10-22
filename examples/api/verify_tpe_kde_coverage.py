"""Independently recount saved coverage tuples and verify coverage-stratified metrics."""

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from math import prod
from pathlib import Path

import numpy as np
from optuna.distributions import json_to_distribution
from scipy.stats import rankdata, spearmanr

from deepcave.evaluators.combinatorial import discretize_combinatorial
from deepcave.evaluators.coverage import normalize_configurations
from examples.api import benchmark_tpe_kde_coverage as experiment


def verify(root):
    verification, comparison = {}, []
    for family in ("xgboost", "catboost"):
        folder = root / family
        read = lambda name: experiment.read_csv(folder / f"{name}.csv")
        metadata = json.loads((folder / "metadata.json").read_text())
        design = json.loads((folder / "design.json").read_text())
        profiles = json.loads((folder / "training_coverage.json").read_text())
        profile_map = {(r["fold"], r["resolution"], r["strength"]): r for r in profiles}
        scores = sorted(read("oof_scores"), key=lambda r: int(r["trial_number"]))
        assert len(scores) == len({r["trial_number"] for r in scores}) == 2000
        folds = np.array([int(r["fold"]) for r in scores])
        ids = np.array([int(r["trial_number"]) for r in scores])
        actual = np.array([float(r["objective"]) for r in scores])
        space = {name: json_to_distribution(audit["distributions"][0])
                 for name, audit in metadata["activity_audit"].items()}
        cs = experiment.configspace_from_distributions(space)
        params = [json.loads(r["params_json"]) for r in scores]
        encoded = {r: discretize_combinatorial(normalize_configurations(cs, params), r)
                   for r in (4, 8)}
        coverage_rows = read("oof_coverage")
        coverage = {(int(r["fold"]), int(r["resolution"]), int(r["strength"]), int(r["trial_number"])): r
                    for r in coverage_rows}
        assert len(coverage) == len(coverage_rows) == 12000
        metric_rows = read("fold_metrics")
        metrics = {(int(r["fold"]), int(r["resolution"]), int(r["strength"]), r["population"], r["method"]): r
                   for r in metric_rows}
        assert len(metric_rows) == len(metrics) == 300
        bins = read("score_bins")
        bin_lookup = {}
        for row in bins:
            key = (int(row["fold"]), int(row["resolution"]), int(row["strength"]), row["population"], row["method"])
            bin_lookup.setdefault(key, []).append(row)
        checked_bins, max_rho_error = 0, 0.0
        for fold in range(5):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            assert len(train) == 1600 and len(test) == 400
            for resolution, strength in experiment.VIEWS:
                points, sizes = encoded[resolution]
                missing, rare = np.zeros(400, dtype=int), np.zeros(400, dtype=int)
                minimum, mean = np.full(400, 1601), np.zeros(400)
                total_missing, group_count = 0, 0
                for columns in combinations(range(points.shape[1]), strength):
                    # Independent tuple dictionary, without CoverageState IDs/offsets/gains.
                    counts = Counter(map(tuple, points[train][:, columns]))
                    occupancy = np.array([counts.get(tuple(p), 0) for p in points[test][:, columns]])
                    missing += occupancy == 0
                    rare += (occupancy >= 1) & (occupancy <= 5)
                    minimum = np.minimum(minimum, occupancy)
                    mean += occupancy
                    total_missing += prod(sizes[c] for c in columns) - len(counts)
                    group_count += 1
                mean /= group_count
                profile = profile_map[fold, resolution, strength]
                assert total_missing == profile["missing_tuples"]
                masks = {"all": np.ones(400, dtype=bool), "regular": missing == 0,
                         "coverage_seeking": missing > 0,
                         "dense_regular": (missing == 0) & (rare <= profile["rare_lower_quartile"]),
                         "rare_seeking": (missing == 0) & (
                             (rare >= profile["rare_upper_quartile"])
                             if profile["rare_upper_quartile"] > profile["rare_lower_quartile"]
                             else (rare > profile["rare_upper_quartile"]))}
                for j, i in enumerate(test):
                    row = coverage[fold, resolution, strength, ids[i]]
                    assert int(row["missing_gain"]) == missing[j] and int(row["rare_count"]) == rare[j]
                    assert int(row["minimum_occupancy"]) == minimum[j]
                    np.testing.assert_allclose(float(row["mean_occupancy"]), mean[j], atol=1e-15)
                    for population, mask in masks.items():
                        assert int(row[population]) == mask[j]
                for population, mask in masks.items():
                    for method in experiment.METHODS:
                        key = (fold, resolution, strength, population, method)
                        metric = metrics[key]
                        selected = test[mask]
                        observed = actual[selected]
                        predicted = np.array([float(scores[i][f"{method}_score"]) for i in selected])
                        assert int(metric["n"]) == len(selected)
                        expected = float(spearmanr(predicted, observed).statistic) if len(selected) >= 10 else np.nan
                        np.testing.assert_allclose(float(metric["spearman"]), expected, atol=1e-15, equal_nan=True)
                        if np.isfinite(expected):
                            max_rho_error = max(max_rho_error, abs(expected - float(metric["spearman"])))
                        n_bins = int(metric["n_bins"])
                        if population in ("regular", "coverage_seeking"):
                            other = "regular" if population == "coverage_seeking" else "coverage_seeking"
                            assert n_bins == int(metrics[fold, resolution, strength, other, method]["n_bins"])
                        recorded = sorted(bin_lookup.get(key, []), key=lambda r: int(r["bin"]))
                        assert len(recorded) == n_bins
                        if n_bins:
                            parts = np.array_split(np.argsort(predicted, kind="stable"), n_bins)
                            means = [np.mean(observed[part]) for part in parts]
                            for part, mean_ap, row in zip(parts, means, recorded):
                                assert len(part) >= 20 and int(row["n"]) == len(part)
                                np.testing.assert_allclose(float(row["mean_ap"]), mean_ap, atol=1e-15)
                                checked_bins += 1
                            changes = np.diff(means)
                            drops = np.maximum(-changes, 0)
                            np.testing.assert_allclose(
                                [float(metric[k]) for k in ("decreases", "monotone", "max_drop", "sum_drops", "spread")],
                                [sum(changes < -1e-12), int(np.all(changes >= -1e-12)), max(drops), sum(drops), means[-1] - means[0]], atol=1e-15)
            for method in experiment.METHODS:
                predicted = np.array([float(scores[i][f"{method}_score"]) for i in test])
                error = abs(rankdata(predicted) - rankdata(actual[test])) / 400
                np.testing.assert_allclose(error, [float(scores[i][f"{method}_rank_error"]) for i in test], atol=1e-15)
        assert checked_bins == len(bins)
        paired = read("paired_metrics")
        assert len(paired) == 150
        for row in paired:
            prefix = (int(row["fold"]), int(row["resolution"]), int(row["strength"]), row["population"])
            for name in ("spearman", "decreases", "monotone", "max_drop", "spread"):
                expected = float(metrics[(*prefix, "tpe")][name]) - float(metrics[(*prefix, "kde_optimized")][name])
                np.testing.assert_allclose(float(row[f"delta_{name}"]), expected, atol=1e-15, equal_nan=True)
        pool, selected = read("prospective_pool"), read("prospective_selected")
        assert len(pool) == 512 and len(selected) == 25
        history_keys = {r["params_json"] for r in scores}
        pool_keys = {r["params_json"] for r in pool}
        assert len(pool_keys) == 512 and not history_keys & pool_keys
        assert len({r["candidate_id"] for r in selected}) == 25
        by_id = {r["candidate_id"]: r for r in pool}
        resolution, strength = design["primary_resolution"], design["primary_strength"]
        prospective_points = discretize_combinatorial(normalize_configurations(cs,
            [json.loads(r["params_json"]) for r in pool]), resolution)[0]
        prospective_gain = np.zeros(512, dtype=int)
        full_points, _ = encoded[resolution]
        for columns in combinations(range(full_points.shape[1]), strength):
            seen = set(map(tuple, full_points[:, columns]))
            prospective_gain += [tuple(p) not in seen for p in prospective_points[:, columns]]
        for i, row in enumerate(pool):
            assert int(row["missing_gain"]) == prospective_gain[i] > 0
            values = json.loads(row["params_json"])
            for name, distribution in space.items():
                assert distribution._contains(distribution.to_internal_repr(values[name]))
            assert all(np.isfinite(float(row[f"{m}_score"])) for m in experiment.METHODS)
        for row in selected:
            assert row["objective"] == "" and row["evaluation_status"] == "not_executed"
            for name in by_id[row["candidate_id"]]:
                assert row[name] == by_id[row["candidate_id"]][name]
        for key in ("tpe_score", "kde_optimized_score", "missing_gain"):
            assert min(float(r[key]) for r in selected) == min(float(r[key]) for r in pool)
            assert max(float(r[key]) for r in selected) == max(float(r[key]) for r in pool)
        reference = json.loads((folder / "reference_checks.json").read_text())
        assert len(reference) == 10 and max(r["max_absolute_difference"] for r in reference) == 0
        summary = read("summary")
        for population in ("regular", "coverage_seeking"):
            rows = {m: next(r for r in summary if (int(r["resolution"]), int(r["strength"])) == (resolution, strength)
                           and r["population"] == population and r["method"] == m) for m in experiment.METHODS}
            comparison.append({"study": family, "population": population, "phase": "retrospective",
                "kde_spearman": rows["kde_optimized"]["mean_spearman"],
                "tpe_spearman": rows["tpe"]["mean_spearman"], "n_evaluated": rows["tpe"]["n_configurations"],
                "n_prepared": 0, "resolution": resolution, "strength": strength})
        comparison.append({"study": family, "population": "coverage_seeking", "phase": "prospective_not_executed",
            "kde_spearman": "", "tpe_spearman": "", "n_evaluated": 0, "n_prepared": 25,
            "resolution": resolution, "strength": strength})
        verification[family] = {"oof_configurations": len(scores), "coverage_rows_recounted": len(coverage),
            "fold_population_method_metrics_checked": len(metrics), "score_bin_means_checked": checked_bins,
            "method_pairs_checked": len(paired), "prospective_candidates_checked": len(pool),
            "selected_candidates_checked": len(selected), "max_spearman_error": max_rho_error,
            "max_historical_score_difference": 0.0}
    experiment.scoring.write_csv(root / "comparison.csv", comparison)
    experiment.write_json(root / "verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/validation/tpe_kde/coverage_regions"))
    verify(parser.parse_args().root)
