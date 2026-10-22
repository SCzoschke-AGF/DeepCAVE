"""Independently audit saved paired learning curves without repeating any fits."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def aggregate(root):
    """Reproduce the cross-study descriptive tables from saved individual results."""
    paired_summary, decile_summary, stability_paired = [], [], []
    for family in ("xgboost", "catboost"):
        folder = root / family
        pairs = read_csv(folder / "paired_metrics.csv")
        for size in (100, 200, 500, 1000, 1600):
            selected = [r for r in pairs if int(r["n_train"]) == size and r["valid_pair"] == "1"]
            rho = [float(r["delta_spearman"]) for r in selected]
            row = {"study": family, "n_train": size, "valid_pairs": len(selected),
                   "tpe_spearman_wins": sum(v > 0 for v in rho)}
            for key in ("spearman", "decreases", "monotone", "max_drop", "sum_drops", "delta_ap",
                        "fit_seconds", "score_seconds"):
                values = [float(r[f"delta_{key}"]) for r in selected]
                row.update({f"mean_delta_{key}": float(np.mean(values)),
                            f"sd_delta_{key}": float(np.std(values, ddof=1))})
            row.update({"min_delta_spearman": min(rho), "max_delta_spearman": max(rho)})
            paired_summary.append(row)
        ds = read_csv(folder / "deciles.csv")
        for size in (100, 200, 500, 1000, 1600):
            for method in ("tpe", "kde_optimized"):
                for decile in range(1, 11):
                    values = [float(r["mean_ap"]) for r in ds if int(r["n_train"]) == size
                              and r["method"] == method and int(r["decile"]) == decile]
                    decile_summary.append({"study": family, "n_train": size, "method": method,
                        "decile": decile, "n_curves": len(values), "mean_ap": float(np.mean(values)),
                        "sd_ap": float(np.std(values, ddof=1))})
        stability = read_csv(folder / "stability_pairs.csv")
        grouped = {tuple(r[k] for k in ("fold", "n_train", "repetition_a", "repetition_b", "method")): r
                   for r in stability}
        for key, left in grouped.items():
            if key[-1] != "tpe":
                continue
            right = grouped[(*key[:-1], "kde_optimized")]
            stability_paired.append({"study": family,
                **{k: int(left[k]) for k in ("fold", "n_train", "repetition_a", "repetition_b")},
                **{f"delta_{k}": float(left[k]) - float(right[k]) for k in
                   ("rank_spearman", "decile_changed_fraction", "mean_absolute_decile_shift")}})
    for filename, rows in (("paired_summary", paired_summary), ("decile_summary", decile_summary),
                           ("stability_paired", stability_paired)):
        with (root / f"{filename}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def verify(root):
    """Check membership, nesting, every saved metric, stability and historical scores."""
    verification = {}
    for family in ("xgboost", "catboost"):
        folder = root / family
        metadata = json.loads((folder / "metadata.json").read_text())
        trials = {int(r["trial_number"]): r for r in read_csv(folder / "trials.csv")}
        samples = json.loads((folder / "samples.json").read_text())
        assert len(trials) == 2000 and len(samples) == 105
        sample_map = {(r["fold"], r["n_train"], r["repetition"]): r for r in samples}
        assert len(sample_map) == 105
        for sample in samples:
            fold, size, rep = (sample[k] for k in ("fold", "n_train", "repetition"))
            train, good, bad = (sample[k] for k in ("train_ids", "good_ids", "bad_ids"))
            pool = sorted(i for i, row in trials.items() if int(row["fold"]) != fold)
            permutation = np.random.default_rng(np.random.SeedSequence([42, fold, rep])).permutation(pool)
            assert train == sorted(permutation[:size])
            assert len(set(train)) == size and len(good) == size // 5
            assert not set(good) & set(bad) and set(good) | set(bad) == set(train)
            ordered = sorted(train, key=lambda i: (-float(trials[i]["objective"]), i))
            assert good == sorted(ordered[:size // 5])
            if size == 1600:
                assert rep == 0
            if size < 1000:
                next_size = {100: 200, 200: 500, 500: 1000}[size]
                assert set(train) <= set(sample_map[fold, next_size, rep]["train_ids"])
        predictions = defaultdict(list)
        for row in read_csv(folder / "oof_scores.csv"):
            key = tuple(int(row[k]) for k in ("fold", "n_train", "repetition"))
            predictions[key].append(row)
        assert set(predictions) == set(sample_map)
        metric_rows = read_csv(folder / "fit_metrics.csv")
        metrics = {tuple(int(r[k]) for k in ("fold", "n_train", "repetition")) + (r["method"],): r
                   for r in metric_rows}
        assert len(metric_rows) == len(metrics) == 210
        decile_rows = read_csv(folder / "deciles.csv")
        deciles = {tuple(int(r[k]) for k in ("fold", "n_train", "repetition"))
                   + (r["method"], int(r["decile"])): r for r in decile_rows}
        assert len(decile_rows) == len(deciles) == 2100
        bin_cache, score_cache = {}, {}
        max_rho_error, max_mean_error = 0.0, 0.0
        for key, rows in predictions.items():
            test_ids = sorted(i for i, r in trials.items() if int(r["fold"]) == key[0])
            assert [int(r["trial_number"]) for r in rows] == test_ids
            assert len(rows) == 400 and not set(test_ids) & set(sample_map[key]["train_ids"])
            actual = np.array([float(r["objective"]) for r in rows])
            np.testing.assert_array_equal(actual, [float(trials[i]["objective"]) for i in test_ids])
            for method in ("tpe", "kde_optimized"):
                scores = np.array([float(r[f"{method}_score"]) for r in rows])
                assert np.isfinite(scores).all() and all(r[f"{method}_valid"] == "1" for r in rows)
                metric = metrics[(*key, method)]
                assert metric["success"] == metric["fit_success"] == "1"
                rho = float(spearmanr(scores, actual).statistic)
                max_rho_error = max(max_rho_error, abs(rho - float(metric["spearman"])))
                bins = np.array_split(np.argsort(scores, kind="stable"), 10)
                means = np.array([np.mean(actual[b]) for b in bins])
                labels = np.empty(400, dtype=int)
                for d, members in enumerate(bins, 1):
                    labels[members] = d
                    recorded = deciles[(*key, method, d)]
                    assert int(recorded["n_test"]) == len(members) == 40
                    max_mean_error = max(max_mean_error, abs(means[d - 1] - float(recorded["mean_ap"])))
                drops = np.maximum(-np.diff(means), 0)
                assert int(metric["decreases"]) == np.count_nonzero(np.diff(means) < -1e-12)
                assert int(metric["monotone"]) == int(not np.any(np.diff(means) < -1e-12))
                np.testing.assert_allclose([float(metric[k]) for k in ("max_drop", "sum_drops", "delta_ap")],
                                           [max(drops), sum(drops), means[-1] - means[0]], atol=1e-15)
                bin_cache[(*key, method)], score_cache[(*key, method)] = labels, scores
        pairs = read_csv(folder / "paired_metrics.csv")
        assert len(pairs) == 105 and all(r["valid_pair"] == "1" for r in pairs)
        for row in pairs:
            key = tuple(int(row[k]) for k in ("fold", "n_train", "repetition"))
            for name in ("spearman", "decreases", "monotone", "max_drop", "sum_drops", "delta_ap",
                         "fit_seconds", "score_seconds"):
                expected = float(metrics[(*key, "tpe")][name]) - float(metrics[(*key, "kde_optimized")][name])
                np.testing.assert_allclose(float(row[f"delta_{name}"]), expected, atol=1e-15)
        stability = read_csv(folder / "stability_pairs.csv")
        assert len(stability) == 400
        for row in stability:
            fold, size = int(row["fold"]), int(row["n_train"])
            left = (fold, size, int(row["repetition_a"]), row["method"])
            right = (fold, size, int(row["repetition_b"]), row["method"])
            np.testing.assert_allclose(float(row["rank_spearman"]),
                                       spearmanr(score_cache[left], score_cache[right]).statistic, atol=1e-15)
            np.testing.assert_allclose(float(row["decile_changed_fraction"]),
                                       np.mean(bin_cache[left] != bin_cache[right]), atol=1e-15)
            np.testing.assert_allclose(float(row["mean_absolute_decile_shift"]),
                                       np.mean(abs(bin_cache[left] - bin_cache[right])), atol=1e-15)
        reference = json.loads((folder / "reference_checks.json").read_text())
        assert len(reference) == 10 and all(r["matches"] for r in reference)
        old_metadata = json.loads((root.parent / f"robustness/{family}/start_0.5/metadata.json").read_text())
        assert metadata["input_sha256"] == old_metadata["input_sha256"]
        assert metadata["versions"] == old_metadata["versions"]
        assert max_rho_error < 1e-14 and max_mean_error < 1e-14
        verification[family] = {
            "training_samples_checked": len(samples), "test_configurations_per_sample": 400,
            "method_evaluations_checked": len(metrics), "decile_means_checked": len(deciles),
            "paired_comparisons_checked": len(pairs), "stability_pairs_checked": len(stability),
            "max_spearman_error": max_rho_error, "max_decile_mean_error": max_mean_error,
            "historical_input_and_versions_match": True,
            "max_historical_score_difference": max(r["max_absolute_difference"] for r in reference),
        }
    aggregate(root)
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/validation/tpe_kde/learning_curves"))
    verify(parser.parse_args().root)
