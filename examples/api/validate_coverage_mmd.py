"""Reproduce mixed-space MMD power measurements without changing the dashboard kernel.

Run with ``python -m examples.api.validate_coverage_mmd --output-dir /tmp/mmd-validation``.
The interaction mixture is an experimental comparator, not the production method.
"""

from typing import Any, Dict, List

import argparse
import json
from pathlib import Path

import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    EqualsCondition,
    OrdinalHyperparameter,
)

from deepcave.evaluators import coverage

KERNELS = ("product", "marginal", "interaction_mixture")


def kernel_matrices(left: np.ndarray, right: np.ndarray, axes: list) -> Dict[str, np.ndarray]:
    """Compare the production product with marginal and low-order interaction kernels."""
    product = np.ones((len(left), len(right)))
    total = np.zeros_like(product)
    squares = np.zeros_like(product)
    for j, axis in enumerate(axes):
        # Use the actual production factor, including its inactive-state handling.
        factor = coverage._mixed_kernel(left[:, [j]], right[:, [j]], [axis])
        product *= factor
        total += factor
        squares += factor**2
    dimensions = len(axes)
    marginal = total / dimensions
    pairwise = (
        (total**2 - squares) / (dimensions * (dimensions - 1)) if dimensions > 1 else marginal
    )
    return {
        "product": product,
        "marginal": marginal,
        "interaction_mixture": (marginal + pairwise + product) / 3,
    }


def kernel_means(left: np.ndarray, right: np.ndarray, axes: list) -> np.ndarray:
    """Return kernel means in a stable, named order."""
    matrices = kernel_matrices(left, right, axes)
    return np.array([matrices[name].mean() for name in KERNELS])


def mmd_scores(data: Any, references: Any, reference_means: np.ndarray) -> np.ndarray:
    """Evaluate the production biased estimator and two fixed comparator kernels."""
    left, axes = coverage._mixed_coordinates(data)
    right, _ = coverage._mixed_coordinates(references)
    squared = kernel_means(left, left, axes) + reference_means - 2 * kernel_means(left, right, axes)
    return np.sqrt(np.maximum(0, squared))


def make_spaces(dimensions: List[int]) -> List[dict]:
    """Define independent spaces with controlled defects and nuisance dimensions."""
    spaces = []
    for count in dimensions:
        space = ConfigurationSpace(
            space={
                **{f"x{i}": (0.0, 1.0) for i in range(count - 2)},
                "category": ["a", "b"],
                "integer": (0, 1),
            }
        )
        spaces.append(
            {
                "name": f"mixed_{count}d",
                "space": space,
                "scenarios": ["missing_category", "marginal_bias", "missing_combinations"],
            }
        )
    spaces.append(
        {
            "name": "categorical_20d",
            "space": ConfigurationSpace(space={f"c{i}": ["a", "b"] for i in range(20)}),
            "scenarios": ["missing_category", "missing_combinations", "three_way_parity"],
        }
    )
    conditional = ConfigurationSpace(
        space={
            "category": ["a", "b"],
            "child": (0.0, 1.0),
            **{f"x{i}": (0.0, 1.0) for i in range(3)},
        }
    )
    conditional.add(OrdinalHyperparameter("level", ["low", "mid", "high"]))
    conditional.add(EqualsCondition(conditional["child"], conditional["category"], "b"))
    spaces.append(
        {
            "name": "conditional_6d",
            "space": conditional,
            "scenarios": ["conditional_bias", "missing_branch"],
        }
    )
    rare = ConfigurationSpace(
        space={
            **{f"x{i}": (0.0, 1.0) for i in range(4)},
            "integer": (0, 1),
        }
    )
    rare.add(CategoricalHyperparameter("category", ["a", "b", "rare"], weights=[0.49, 0.49, 0.02]))
    spaces.append(
        {"name": "rare_category_6d", "space": rare, "scenarios": ["missing_rare_category"]}
    )
    return spaces


def draw(space: ConfigurationSpace, count: int, seed: int, scenario: str = "random") -> Any:
    """Draw legal IID designs, or introduce one known distributional defect."""
    data = coverage.sample_reference_design(space, count, seed)
    if scenario == "random":
        return data
    values = [dict(config) for config in data.values]
    rng = np.random.default_rng(seed)
    categorical = "category" if "category" in space else "c0"
    for config in values:
        if scenario == "missing_category":
            config[categorical] = "a"
        elif scenario == "missing_rare_category":
            if config["category"] == "rare":
                config["category"] = str(rng.choice(["a", "b"]))
        elif scenario == "marginal_bias":
            config["x0"] *= 0.5
        elif scenario == "missing_combinations":
            if "integer" in space:
                config["integer"] = int(config[categorical] == "b")
            else:
                config["c1"] = config["c0"]
        elif scenario == "three_way_parity":
            config["c2"] = "b" if (config["c0"] == "b") != (config["c1"] == "b") else "a"
        elif scenario == "conditional_bias":
            if "child" in config:
                config["child"] *= 0.25
        elif scenario == "missing_branch":
            config["category"] = "a"
            config.pop("child", None)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    return coverage.normalize_configurations(space, values)


def wilson(successes: int, total: int) -> List[float]:
    """Calculate a 95% binomial interval, conditional on the fixed calibration."""
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1 + z**2 / total
    centre = (fraction + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(fraction * (1 - fraction) / total + z**2 / (4 * total**2)) / denominator
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def flag_rates(calibration: np.ndarray, observed: np.ndarray) -> dict:
    """Match the dashboard's midrank rule, with disjoint calibration/evaluation draws."""
    ranks = (
        100
        * (
            (calibration[None, :] < observed[:, None]).sum(axis=1)
            + 0.5 * (calibration[None, :] == observed[:, None]).sum(axis=1)
        )
        / len(calibration)
    )
    upper = int(np.sum(ranks > 95))
    outside = int(np.sum((ranks < 5) | (ranks > 95)))
    return {
        "replicates": len(observed),
        "upper_count": upper,
        "upper_rate": upper / len(observed),
        "upper_95ci": wilson(upper, len(observed)),
        "outside_count": outside,
        "outside_rate": outside / len(observed),
        "outside_95ci": wilson(outside, len(observed)),
        "median_mmd": float(np.median(observed)),
        "median_metric_percentile": float(np.median(ranks)),
    }


def experiment(
    trials: List[int],
    dimensions: List[int],
    calibration: int,
    replicates: int,
    seed: int,
    references: int = 512,
) -> dict:
    """Measure calibration and detection rates across independent synthetic replicates."""
    groups = []
    for index, definition in enumerate(make_spaces(dimensions)):
        space = definition["space"]
        group_seed = seed + index * 100_000
        reference = draw(space, references, group_seed)
        points, axes = coverage._mixed_coordinates(reference)
        reference_means = kernel_means(points, points, axes)
        production = coverage._mixed_kernel(points, points, axes)
        off_diagonal = production[~np.eye(len(points), dtype=bool)]
        for count in trials:
            null = np.array(
                [
                    mmd_scores(
                        draw(space, count, group_seed + 1000 + i), reference, reference_means
                    )
                    for i in range(calibration)
                ]
            )
            conditions = []
            for scenario in ["random", *definition["scenarios"]]:
                scores = np.array(
                    [
                        mmd_scores(
                            draw(space, count, group_seed + 10_000 + i, scenario),
                            reference,
                            reference_means,
                        )
                        for i in range(replicates)
                    ]
                )
                conditions.append(
                    {
                        "scenario": scenario,
                        "kernels": {
                            name: flag_rates(null[:, j], scores[:, j])
                            for j, name in enumerate(KERNELS)
                        },
                    }
                )
            groups.append(
                {
                    "space": definition["name"],
                    "dimensions": len(axes),
                    "trials": count,
                    "reference_seed": group_seed,
                    "calibration_seeds": [group_seed + 1000, group_seed + 999 + calibration],
                    "evaluation_seeds": [group_seed + 10_000, group_seed + 9999 + replicates],
                    "product_off_diagonal_mean": float(off_diagonal.mean()),
                    "product_off_diagonal_median": float(np.median(off_diagonal)),
                    "product_off_diagonal_zero_fraction": float(np.mean(off_diagonal == 0)),
                    "conditions": conditions,
                }
            )
            print(f"Completed {definition['name']}, n={count}", flush=True)
    return {
        "seed": seed,
        "trials": trials,
        "dimensions": dimensions,
        "reference_samples": references,
        "calibration_designs": calibration,
        "replicates": replicates,
        "kernel_width": coverage.MIXED_MMD_KERNEL_SCALE,
        "production_kernel_version": coverage.MIXED_MMD_VERSION,
        "production_trial_limit": coverage.MIXED_MMD_MAX_TRIALS,
        "kernels": list(KERNELS),
        "groups": groups,
    }


def report(result: dict, directory: Path) -> None:
    """Write reproducible results, confidence intervals, and an exportable figure."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Mixed-space MMD sensitivity measurements",
        "",
        f"Seed {result['seed']}; {result['calibration_designs']} independent calibration designs, "
        f"{result['replicates']} held-out replicates per scenario; "
        f"{result['reference_samples']} shared reference configurations; kernel width {result['kernel_width']}.",
        "",
        "The table measures the fraction with MMD above the 95th midrank percentile of "
        "correct IID designs. This is an upper-tail detection rate, not the dashboard's "
        "two-sided 5–95% status. Both rates and their 95% Wilson intervals are in results.json. "
        "Intervals are conditional on the fixed reference and calibration designs. "
        "Null evaluation seeds are disjoint from calibration and reference seeds.",
        "",
        "`product` is the current production kernel. `marginal` is a negative control "
        "for joint-only defects. `interaction_mixture` is an experimental equal-weight "
        "mixture of averaged one-dimensional kernels, averaged pair products, and the "
        "complete product. No comparator is selected using evaluation labels.",
        "",
        "| Space | Trials | Scenario | Product | Marginal control | Interaction mixture |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for group in result["groups"]:
        for condition in group["conditions"]:
            cells = []
            for name in KERNELS:
                row = condition["kernels"][name]
                low, high = row["upper_95ci"]
                cells.append(f"{100*row['upper_rate']:.0f}% [{100*low:.0f}, {100*high:.0f}]")
            lines.append(
                f"| {group['space']} | {group['trials']} | {condition['scenario']} | "
                + " | ".join(cells)
                + " |"
            )
    lines += [
        "",
        "A missing 2% category naturally occurs with probability "
        "0.98^n under correct IID sampling. At n=80 this is about 20%; absence "
        "alone is not a reliable anomaly at a 5% false-alarm budget.",
        "",
    ]
    (directory / "measurements.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [group for group in result["groups"] if group["space"].startswith("mixed_")]
    figure, panels = plt.subplots(
        1, len(result["trials"]), figsize=(6 * len(result["trials"]), 4), squeeze=False
    )
    colors = {
        "missing_category": "#315cd6",
        "marginal_bias": "#d97706",
        "missing_combinations": "#0f766e",
    }
    for panel, count in zip(panels[0], result["trials"]):
        for scenario, color in colors.items():
            groups = [g for g in selected if g["trials"] == count]
            for name, style in (("product", "-"), ("interaction_mixture", "--")):
                rows = [
                    next(c for c in g["conditions"] if c["scenario"] == scenario)["kernels"][name]
                    for g in groups
                ]
                values = np.array([row["upper_rate"] for row in rows])
                errors = np.array(
                    [
                        [row["upper_rate"] - row["upper_95ci"][0] for row in rows],
                        [row["upper_95ci"][1] - row["upper_rate"] for row in rows],
                    ]
                )
                panel.errorbar(
                    [g["dimensions"] for g in groups],
                    values * 100,
                    yerr=errors * 100,
                    linestyle=style,
                    capsize=2,
                    color=color,
                    marker="o",
                    label=f"{scenario}: {name}",
                )
        panel.axhline(5, color="#94a3b8", linestyle=":", linewidth=1)
        panel.set(
            title=f"{count} trials",
            xlabel="Total parameters",
            ylabel="Upper-tail detection rate (%)",
            ylim=(-2, 102),
        )
        panel.set_xticks(result["dimensions"])
        panel.grid(alpha=0.15)
    panels[0, -1].legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    figure.savefig(directory / "sensitivity.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the fixed benchmark or a smaller pilot with explicit parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs="+", type=int, default=[80, 256])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[6, 12, 20])
    parser.add_argument("--calibration", type=int, default=199)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/deepcave-mmd-validation"))
    args = parser.parse_args()
    if (
        min(args.trials) < 2
        or min(args.dimensions) < 3
        or args.calibration < 20
        or args.replicates < 2
    ):
        parser.error("Use trials >= 2, dimensions >= 3, calibration >= 20, and replicates >= 2.")
    result = experiment(args.trials, args.dimensions, args.calibration, args.replicates, args.seed)
    report(result, args.output_dir)


if __name__ == "__main__":
    main()
