"""Benchmark fixed random space partitions against the existing MMD sensitivity experiment.

Run ``.venv/bin/python -m examples.api.validate_coverage_partitions`` from the repository root.
All partition choices and statistic combinations are fixed before held-out evaluation.
"""

from typing import Dict

import argparse
import json
from pathlib import Path
import time

import numpy as np

from examples.api.coverage_partitions import PartitionEnsemble, aggregate, coordinates, forecast
from examples.api.validate_coverage_mmd import draw, flag_rates, make_spaces, wilson


def joint_scores(scores: Dict[str, np.ndarray], fit: int) -> np.ndarray:
    """Include selection across cell counts and aggregates inside the calibrated statistic."""
    matrix = np.column_stack(list(scores.values()))
    centre = matrix[:fit].mean(axis=0)
    scale = np.maximum(matrix[:fit].std(axis=0, ddof=1), 1e-12)
    return ((matrix - centre) / scale).max(axis=1)


def experiment(args: argparse.Namespace) -> dict:
    """Use disjoint training, mass estimation, standardization, calibration, and evaluation."""
    result = {
        "seed": args.seed,
        "trials": args.trials,
        "dimensions": args.dimensions,
        "partitions_per_resolution": args.partitions,
        "cell_counts": sorted({max(1, args.trials // d) for d in (1, 2, 4)}),
        "training_samples": args.training,
        "mass_samples": args.mass_samples,
        "fit_designs": args.fit,
        "calibration_designs": args.calibration,
        "replicates": args.replicates,
        "groups": [],
        "forecasts": [],
    }
    for index, definition in enumerate(make_spaces(args.dimensions)):
        started = time.monotonic()
        space = definition["space"]
        seed = args.seed + index * 100_000
        training = draw(space, args.training, seed + 50_000)
        mass_reference = draw(space, args.mass_samples, seed + 60_000)
        specs = [
            ("fit", range(seed + 2000, seed + 2000 + args.fit), "random"),
            ("calibration", range(seed + 1000, seed + 1000 + args.calibration), "random"),
        ] + [
            (scenario, range(seed + 10_000, seed + 10_000 + args.replicates), scenario)
            for scenario in ["random", *definition["scenarios"]]
        ]
        blocks, slices, offset = [], {}, 0
        for name, seeds, scenario in specs:
            blocks.extend(coordinates(draw(space, args.trials, s, scenario))[0] for s in seeds)
            slices[name] = slice(offset, offset + len(seeds))
            offset += len(seeds)
        points = np.concatenate(blocks)
        scores, diagnostics, ensembles = {}, {}, {}
        for cells in result["cell_counts"]:
            ensemble = PartitionEnsemble(
                training, mass_reference, cells, args.partitions, seed + 70_000 + cells * 211
            )
            ensembles[cells] = ensemble
            measured = ensemble.measure(points, offset, args.trials)
            scores.update(
                {f"k{cells}_{key}": value for key, value in aggregate(measured, args.fit).items()}
            )
            diagnostic = ensemble.diagnostics(args.trials)
            for metric in measured:
                null_means = measured[metric][slices["calibration"]].mean(axis=1)
                diagnostic[f"random_{metric}_quartiles"] = np.quantile(
                    null_means, [0.25, 0.5, 0.75]
                ).tolist()
            diagnostics[str(cells)] = diagnostic
        scores["joint"] = joint_scores(scores, args.fit)
        conditions = []
        for scenario in ["random", *definition["scenarios"]]:
            stats = {}
            for key, values in scores.items():
                row = flag_rates(values[slices["calibration"]], values[slices[scenario]])
                row["median_statistic"] = row.pop("median_mmd")
                stats[key] = row
            conditions.append({"scenario": scenario, "statistics": stats})
        result["groups"].append(
            {
                "space": definition["name"],
                "dimensions": len(coordinates(training)[1]),
                "training_seed": seed + 50_000,
                "mass_seed": seed + 60_000,
                "partition_seed_formula": f"{seed + 70_000} + K*211 + partition_index",
                "fit_seeds": [seed + 2000, seed + 1999 + args.fit],
                "calibration_seeds": [seed + 1000, seed + 999 + args.calibration],
                "evaluation_seeds": [seed + 10_000, seed + 9999 + args.replicates],
                "diagnostics": diagnostics,
                "conditions": conditions,
            }
        )
        if definition["name"] == "mixed_20d" and args.forecast_simulations:
            future = draw(space, args.forecast_simulations * args.forecast_horizon, seed + 80_000)
            checkpoints = sorted(
                {
                    0,
                    *[s for s in (10, 20, 40, 80, 160, 320) if s < args.forecast_horizon],
                    args.forecast_horizon,
                }
            )
            for scenario in ("random", "missing_category"):
                observed = draw(space, args.trials, seed + 10_000, scenario)
                predictions = forecast(
                    ensembles,
                    observed,
                    future,
                    args.forecast_simulations,
                    args.forecast_horizon,
                    checkpoints,
                )
                for prediction in predictions.values():
                    sufficient = []
                    for budget in prediction["budgets"]:
                        budget["probability"] = budget["successes"] / args.forecast_simulations
                        budget["probability_95ci"] = wilson(
                            budget["successes"], args.forecast_simulations
                        )
                        if budget["probability_95ci"][0] >= 0.8:
                            sufficient.append(budget["additional_trials"])
                    prediction["first_tested_budget_lower95_at_least_80pct"] = (
                        min(sufficient) if sufficient else None
                    )
                result["forecasts"].append(
                    {
                        "space": definition["name"],
                        "observed_scenario": scenario,
                        "observed_seed": seed + 10_000,
                        "future_seed": seed + 80_000,
                        "resolutions": predictions,
                    }
                )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Completed {definition['name']} in {time.monotonic() - started:.1f}s", flush=True)
    return result


def mmd_comparison(result: dict, source: Path) -> dict:
    """Reuse measurements only when seeds, sample sizes, and scenarios match exactly."""
    if not source.exists():
        return {}
    previous = json.loads(source.read_text())
    if any(
        previous[key] != result[key]
        for key in ("seed", "dimensions", "calibration_designs", "replicates")
    ):
        return {}
    matched = {}
    for group in result["groups"]:
        candidates = [
            g
            for g in previous["groups"]
            if g["space"] == group["space"] and g["trials"] == result["trials"]
        ]
        if not candidates:
            continue
        prior = candidates[0]
        if any(prior[key] != group[key] for key in ("calibration_seeds", "evaluation_seeds")):
            continue
        for condition in prior["conditions"]:
            matched[(group["space"], condition["scenario"])] = condition["kernels"]
    return matched


def report(result: dict, directory: Path, comparison_path: Path) -> None:
    """Write all rates, intervals, balance diagnostics, and a comparison figure."""
    comparison = mmd_comparison(result, comparison_path)
    columns = (
        [f"k{k}_mean_empty_mass" for k in result["cell_counts"]]
        + [f"k{k}_max_z_empty_mass" for k in result["cell_counts"]]
        + ["joint"]
    )
    labels = (
        [f"Mean K={k}" for k in result["cell_counts"]]
        + [f"Max K={k}" for k in result["cell_counts"]]
        + ["Joint"]
    )
    if comparison:
        labels += ["MMD product", "MMD mixture"]
    lines = [
        "# Randomized space partitioning: sensitivity measurements",
        "",
        f"{result['trials']} trials; {result['partitions_per_resolution']} fixed partitions per K; "
        f"K={result['cell_counts']}; seed {result['seed']}. "
        f"Independent pools: {result['training_samples']} for partition construction, "
        f"{result['mass_samples']} for cell masses, {result['fit_designs']} IID designs "
        f"for standardization, {result['calibration_designs']} for calibration, "
        f"{result['replicates']} held-out designs per scenario.",
        "",
        "Entries are upper-tail flag rates above the 95th midrank percentile of IID designs. "
        "`random` rows are held-out false-alarm rates. These are measurements at a nominal "
        "5% threshold, conditional on fixed partitions, mass estimates, standardization, and "
        "calibration pools; not guaranteed exact 5% rates. JSON includes 95% Wilson intervals "
        "and two-sided rates. No statistic is selected using the held-out defect labels.",
        "",
        "Mean averages unvisited probability mass across partitions. Max selects the largest "
        "per-partition standardized unvisited mass. Joint takes the largest standardized "
        "aggregate among mean and max of unvisited mass and TV at all three K values. "
        "The complete selection is calibrated against the same IID designs. All component "
        "statistics including TV are retained in JSON.",
        "",
        "MMD columns reuse the earlier results only with matching evaluation and calibration "
        "seeds, counts and scenarios. Reference budgets differ: MMD used 512 points; partitions "
        "use the larger construction and mass pools above. This is a practical sensitivity "
        "comparison, not a comparison at equal computation or reference budget.",
        "",
        "| Space | Scenario | " + " | ".join(labels) + " |",
        "|---|---|" + "---:|" * len(labels),
    ]
    matrix, row_labels = [], []
    for group in result["groups"]:
        for condition in group["conditions"]:
            rates = [condition["statistics"][key] for key in columns]
            if comparison:
                previous = comparison[(group["space"], condition["scenario"])]
                rates += [previous["product"], previous["interaction_mixture"]]
            matrix.append([row["upper_rate"] * 100 for row in rates])
            row_labels.append(f"{group['space']} / {condition['scenario']}")
            cells = []
            for row in rates:
                low, high = row["upper_95ci"]
                cells.append(f"{100*row['upper_rate']:.0f}% [{100*low:.0f}, {100*high:.0f}]")
            lines.append(
                f"| {group['space']} | {condition['scenario']} | " + " | ".join(cells) + " |"
            )
    lines += [
        "",
        "## Cell balance and random occupancy",
        "",
        "Mass is estimated on a separate reference pool, not assumed to be 1/K. "
        "The occupancy expectations use these estimated probabilities. IQR describes "
        "the IID distribution of the average unvisited mass, not a confidence interval.",
        "",
        "| Space | K | Actual cells | Zero estimated mass cells | Median mass CV | Expected missing mass | Random median [IQR] |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for group in result["groups"]:
        for key, diagnostic in group["diagnostics"].items():
            q1, median, q3 = diagnostic["random_empty_mass_quartiles"]
            lines.append(
                f"| {group['space']} | {key} | {diagnostic['actual_cells_min']}–{diagnostic['actual_cells_max']} | "
                f"{diagnostic['zero_mass_cells']} | {diagnostic['median_mass_cv']:.2f} | {diagnostic['expected_empty_mass']:.3f} | "
                f"{median:.3f} [{q1:.3f}, {q3:.3f}] |"
            )
    lines += [
        "",
        "## Fixed-partition forecasts",
        "",
        "Independent IID future configurations are appended to one observed design; "
        "each simulation uses the same future configurations across all partitions. "
        "Partitions and K remain fixed. The target is a 20% reduction in average "
        "unvisited mass. This is not the union of spatial holes across overlapping "
        "partitions. The budgets below are the first tested grid points whose 95% "
        "Wilson lower bound is at least 80%; they are not exact minimum trial counts. "
        "Intervals condition on the observed design, partitions and estimated masses. "
        "Forecasts assume future IID sampling from ConfigSpace, not an adaptive optimizer.",
        "",
        "| Observed design | K | Initial missing mass | First tested sufficient budget |",
        "|---|---:|---:|---:|",
    ]
    for item in result["forecasts"]:
        for key, prediction in item["resolutions"].items():
            lines.append(
                f"| {item['observed_scenario']} | {key} | {prediction['initial_mean_empty_mass']:.3f} | "
                f"{prediction['first_tested_budget_lower95_at_least_80pct']} |"
            )
    (directory / "measurements.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, panel = plt.subplots(figsize=(12.5, 10))
    values = np.array(matrix)
    displayed = panel.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    panel.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    panel.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    panel.set_title(
        f"Detection / false-alarm rates at nominal 5% threshold — {result['trials']} trials"
    )
    for row, col in np.ndindex(values.shape):
        panel.text(
            col,
            row,
            f"{values[row, col]:.0f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if values[row, col] > 55 else "#152238",
        )
    figure.colorbar(displayed, ax=panel, label="Flagged held-out designs (%)", shrink=0.7)
    figure.tight_layout()
    figure.savefig(directory / "sensitivity.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Run the reproducible experiment, or a smaller pilot with explicit parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[6, 12, 20])
    parser.add_argument("--partitions", type=int, default=200)
    parser.add_argument("--training", type=int, default=8192)
    parser.add_argument("--mass-samples", type=int, default=32768)
    parser.add_argument("--fit", type=int, default=128)
    parser.add_argument("--calibration", type=int, default=199)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--forecast-simulations", type=int, default=200)
    parser.add_argument("--forecast-horizon", type=int, default=320)
    parser.add_argument(
        "--mmd-results", type=Path, default=Path("docs/validation/coverage_mmd/results.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/deepcave-partition-validation")
    )
    args = parser.parse_args()
    if (
        args.trials < 4
        or min(args.dimensions) < 3
        or args.partitions < 1
        or args.fit < 3
        or args.calibration < 20
        or args.replicates < 2
        or args.training < args.trials
        or args.mass_samples < args.trials
        or args.forecast_simulations < 0
        or args.forecast_horizon < 1
    ):
        parser.error(
            "Use positive sample counts; N>=4, dimensions>=3, fit>=3, calibration>=20, replicates>=2."
        )
    result = experiment(args)
    report(result, args.output_dir, args.mmd_results)


if __name__ == "__main__":
    main()
