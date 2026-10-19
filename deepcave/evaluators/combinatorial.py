"""Exact combinatorial coverage of independent search-space parameters."""

from typing import Any, Dict, List, Optional, Tuple

from itertools import combinations
from math import comb, prod

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from deepcave.evaluators.coverage import (
    CoverageAxis,
    CoverageData,
    normalize_configurations,
    sample_reference_design,
    summarize_baseline,
)

RESOLUTIONS = (2, 3, 4, 8)
MAX_GROUPS = 100_000
MAX_PROJECTIONS = 20_000_000
BASELINE_VERSION = 3


def _random_summary(observed: float, samples: List[float]) -> Dict[str, Any]:
    """Summarize empirical references without implying a rank when all values tie."""
    summary = summarize_baseline(observed, samples)
    degenerate = min(samples) == max(samples)
    summary.update(median=float(np.median(samples)), degenerate=degenerate)
    if degenerate:
        summary["percentile"] = None
    return summary


def _discrete_cuts(axis: CoverageAxis, levels: int) -> np.ndarray:
    """Split ordered support near theoretical quantiles, keeping every bin nonempty."""
    size = axis.size
    assert size is not None
    cuts = []
    for level in range(1, levels):
        lower = cuts[-1] + 1 if cuts else 1
        upper = size - (levels - level)
        lo, hi = lower, upper
        target = level / levels
        while lo < hi:
            middle = (lo + hi) // 2
            mass = axis.mass_edges(np.array([middle - 1]))[1][0]
            if mass < target:
                lo = middle + 1
            else:
                hi = middle
        candidates = np.array(sorted({max(lower, lo - 1), lo}))
        masses = axis.mass_edges(candidates - 1)[1]
        cuts.append(int(candidates[np.argmin(np.abs(masses - target))]))
    return np.asarray(cuts, dtype=int)


def _discretization(data: CoverageData, resolution: int, definitions: bool = False) -> tuple:
    """Encode levels and optionally retain their exact definitions for later drilldowns."""
    columns, sizes, level_definitions = [], [], {}
    for index, axis in enumerate(data.axes):
        if axis.size == 1:
            continue
        if axis.kind in ("categorical", "ordinal"):
            size = len(axis.choices)
            codes = axis.ranks([value[axis.name] for value in data.values])
            if definitions:
                ranges = [str(value) for value in axis.choices]
        elif axis.size is not None:
            size = min(resolution, axis.size)
            cuts = _discrete_cuts(axis, size)
            ranks = axis.ranks([value[axis.name] for value in data.values])
            codes = np.searchsorted(cuts, ranks, side="right")
            if definitions:
                edges = np.r_[0, cuts, axis.size]
                lower = axis.values_at(edges[:-1])
                upper = axis.values_at(edges[1:] - 1)
                ranges = [f"[{lo}, {hi}]" for lo, hi in zip(lower, upper)]
        else:
            size = resolution
            codes = np.minimum(size - 1, (data.points[:, index] * size).astype(int))
            if definitions:
                # Invert precisely the sampling CDF used by CoverageAxis.encode.
                quantiles = np.linspace(0, 1, size + 1)
                rv = getattr(axis.hp._vector_dist, "rv", None)
                vectors = rv.ppf(quantiles) if rv is not None else quantiles
                edges = np.asarray(axis.hp.to_value(vectors), dtype=float)
                edges[0], edges[-1] = axis.hp.lower, axis.hp.upper
                ranges = [
                    f"[{edges[i]:.12g}, {edges[i + 1]:.12g}{']' if i == size - 1 else ')'}"
                    for i in range(size)
                ]
        if definitions:
            if axis.kind in ("categorical", "ordinal"):
                labels = ranges
            elif size == 2:
                labels = ["Low", "High"]
            elif size == 3:
                labels = ["Low", "Medium", "High"]
            else:
                labels = [f"L{i + 1}" for i in range(size)]
            level_definitions[axis.name] = [
                {"label": label, "range": bounds} for label, bounds in zip(labels, ranges)
            ]
        columns.append(codes)
        sizes.append(int(size))
    points = np.column_stack(columns) if columns else np.empty((data.count, 0), dtype=int)
    return points, sizes, level_definitions


def discretize_combinatorial(data: CoverageData, resolution: int) -> Tuple[np.ndarray, List[int]]:
    """Encode theoretical levels; categories/ordinals retain their original values."""
    points, sizes, _ = _discretization(data, resolution)
    return points, sizes


class CoverageSpec:
    """Use the measured discretization to generate and re-encode concrete values."""

    def __init__(self, data: CoverageData, resolution: int) -> None:
        if data.constrained or any(axis.conditional for axis in data.axes):
            raise ValueError(
                "Coverage planning requires independent parameters without constraints."
            )
        if data.configspace is None:
            raise ValueError("Coverage planning requires a search-space definition.")
        if not np.isfinite(data.points).all():
            raise ValueError("Every successful trial must define every parameter.")
        self.configspace = data.configspace
        self.resolution = resolution
        self.axes = [axis for axis in data.axes if axis.size != 1]
        self.constants = {
            axis.name: axis.values_at(np.array([0]))[0] for axis in data.axes if axis.size == 1
        }
        if not self.axes:
            raise ValueError("Coverage planning requires at least one varying parameter.")
        self.points, self.sizes, self.definitions = _discretization(data, resolution, True)
        self.names = [axis.name for axis in self.axes]
        self.rank_edges = {
            axis.name: np.r_[0, _discrete_cuts(axis, size), axis.size]
            for axis, size in zip(self.axes, self.sizes)
            if axis.size is not None and axis.kind not in ("categorical", "ordinal")
        }

    def materialize(self, point: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
        """Choose legal values within levels without modifying the search-space RNG."""
        values = dict(self.constants)
        for axis, size, level in zip(self.axes, self.sizes, point):
            level = int(level)
            if axis.kind in ("categorical", "ordinal"):
                value = axis.choices[level]
            elif axis.size is not None:
                edges = self.rank_edges[axis.name]
                rank = int(rng.integers(edges[level], edges[level + 1]))
                value = axis.values_at(np.array([rank]))[0]
            else:
                # Stay away from numerical boundaries; use the inverse of encode's CDF.
                quantile = (level + rng.uniform(0.1, 0.9)) / size
                rv = getattr(axis.hp._vector_dist, "rv", None)
                vector = rv.ppf(quantile) if rv is not None else quantile
                value = float(axis.hp.to_value(vector))
            values[axis.name] = value
        Configuration(self.configspace, values=values).check_valid_configuration()
        actual = self.encode([values])[0]
        if not np.array_equal(actual, point):
            raise ValueError(
                "A proposed configuration cannot realize its intended coverage levels."
            )
        return values

    def encode(self, configurations: List[Dict[str, Any]]) -> np.ndarray:
        """Apply the original discretization to concrete configurations."""
        data = normalize_configurations(self.configspace, configurations)
        return discretize_combinatorial(data, self.resolution)[0]

    def serialize(self) -> Dict[str, Any]:
        """Retain exact numerical boundaries as well as readable level definitions."""
        parameters = []
        for axis, size in zip(self.axes, self.sizes):
            parameter = {
                "name": axis.name,
                "type": axis.kind,
                "levels": self.definitions[axis.name],
            }
            if axis.kind in ("categorical", "ordinal"):
                parameter["values"] = axis.choices
            elif axis.size is not None:
                parameter["rank_edges"] = self.rank_edges[axis.name].tolist()
                parameter["lower"] = axis.hp.lower
                parameter["step"] = axis.step
            else:
                quantiles = np.linspace(0, 1, size + 1)
                rv = getattr(axis.hp._vector_dist, "rv", None)
                vectors = rv.ppf(quantiles) if rv is not None else quantiles
                edges = np.asarray(axis.hp.to_value(vectors), dtype=float)
                edges[0], edges[-1] = axis.hp.lower, axis.hp.upper
                parameter["edges"] = edges.tolist()
            parameters.append(parameter)
        return {
            "resolution": self.resolution,
            "parameters": parameters,
            "constants": self.constants,
        }


def _group_counts(points: np.ndarray, sizes: List[int], groups: List[tuple]) -> np.ndarray:
    """Count observed mixed-radix tuples in batches without enumerating empty cells."""
    counts = []
    cardinalities = np.asarray(sizes, dtype=np.int64)
    for start in range(0, len(groups), 256):
        indices = np.asarray(groups[start : start + 256])
        codes = np.zeros((len(points), len(indices)), dtype=np.int64)
        for position in range(indices.shape[1]):
            columns = indices[:, position]
            codes *= cardinalities[columns]
            codes += points[:, columns]
        codes.sort(axis=0)
        counts.extend((1 + np.sum(codes[1:] != codes[:-1], axis=0)).tolist())
    return np.asarray(counts, dtype=np.int64)


def compute_combinatorial_coverage(
    data: CoverageData,
    details: bool = True,
    minimum_strengths: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Measure all groups at each reported strength; never sample parameter groups.

    Trial-count feasibility is only a necessary upper bound. Cells beyond that
    bound are omitted; computational limits are reported separately. Constants
    do not increase interaction strength. Conditional/constrained spaces require
    a different validity model and are explicitly unsupported here. Each resolution
    stops at its first incomplete strength or a trial/work limit. Calibration may
    request higher minimum strengths to compare identical statistics in every study.
    """
    if data.constrained or any(axis.conditional for axis in data.axes):
        raise ValueError(
            "Combinatorial coverage requires independent parameters without constraints."
        )
    axes = [axis for axis in data.axes if axis.size != 1]
    if not axes:
        raise ValueError("Combinatorial coverage requires at least one varying parameter.")
    if not data.count:
        raise ValueError("Combinatorial coverage requires sampled trials.")
    if not np.isfinite(data.points).all():
        raise ValueError(
            "Combinatorial coverage requires a value for every parameter in every trial."
        )
    names = [axis.name for axis in axes]
    rows = []
    for resolution in RESOLUTIONS:
        points, sizes, definitions = _discretization(data, resolution, definitions=details)
        required, upper_bound = 1, 0
        for size in sorted(sizes, reverse=True):
            required *= size
            if required > data.count:
                break
            upper_bound += 1
        cells, observed_strength = [], 0
        used_groups = 0
        minimum = (minimum_strengths or {}).get(resolution, 0)
        for strength in range(1, len(axes) + 1):
            required = prod(sorted(sizes, reverse=True)[:strength])
            group_count = comb(len(axes), strength)
            cell: Dict[str, Any] = {
                "strength": strength,
                "groups": group_count,
                "required_trials": required,
                "coverage": None,
            }
            if required > data.count:
                cell["status"] = "trial_limit"
            elif (
                used_groups + group_count > MAX_GROUPS
                or (used_groups + group_count) * data.count > MAX_PROJECTIONS
            ):
                cell["status"] = "work_limit"
            else:
                groups = list(combinations(range(len(axes)), strength))
                totals = np.array([prod(sizes[index] for index in group) for group in groups])
                counts = _group_counts(points, sizes, groups)
                complete = counts == totals
                covered, total = int(counts.sum()), int(totals.sum())
                cell.update(
                    status="computed",
                    covered=covered,
                    total=total,
                    missing=total - covered,
                    coverage=covered / total,
                    complete_groups=int(complete.sum()),
                    fully_covered_rate=float(complete.mean()),
                )
                if details:
                    # All counts, in combinations() order, rather than only the top 20.
                    # Missing tuples are intentionally not materialized here.
                    cell["group_covered"] = counts.tolist()
                    worst = sorted(
                        np.flatnonzero(~complete),
                        key=lambda index: (
                            -int(totals[index] - counts[index]),
                            counts[index] / totals[index],
                        ),
                    )[:20]
                    cell["deficits"] = [
                        {
                            "parameters": [names[j] for j in groups[index]],
                            "covered": int(counts[index]),
                            "total": int(totals[index]),
                            "missing": int(totals[index] - counts[index]),
                            "coverage": float(counts[index] / totals[index]),
                        }
                        for index in worst
                    ]
                used_groups += group_count
                if bool(complete.all()):
                    observed_strength = strength
            cells.append(cell)
            if cell["status"] != "computed" or (cell["missing"] and strength >= minimum):
                break
        rows.append(
            {
                "resolution": resolution,
                "levels": dict(zip(names, sizes)),
                "observed_strength": observed_strength,
                "strength_is_lower_bound": (
                    observed_strength < upper_bound
                    and (
                        observed_strength == len(cells)
                        or cells[observed_strength]["status"] == "work_limit"
                    )
                ),
                "trial_bound": upper_bound,
                "cells": cells,
                **(
                    {
                        "discretization": {
                            "points": points.tolist(),
                            "definitions": definitions,
                        }
                    }
                    if details
                    else {}
                ),
            }
        )
    return {
        "trials": data.count,
        "parameters": names,
        "excluded_constants": len(data.axes) - len(axes),
        "max_strength": max(len(row["cells"]) for row in rows),
        "rows": rows,
    }


def compute_combinatorial_baseline(
    data: CoverageData, configspace: ConfigurationSpace, designs: int = 200, seed: int = 0
) -> Dict[str, Any]:
    """Calibrate whole-matrix statistics with independent complete random designs."""
    if isinstance(designs, bool) or int(designs) != designs or not 1 <= designs <= 2000:
        raise ValueError("Reference designs must be an integer between 1 and 2000.")
    observed = compute_combinatorial_coverage(data)
    minimum_strengths = {
        row["resolution"]: max(
            (cell["strength"] for cell in row["cells"] if cell["status"] == "computed"),
            default=0,
        )
        for row in observed["rows"]
    }
    samples: Dict[tuple, list] = {
        (row["resolution"], cell["strength"]): []
        for row in observed["rows"]
        for cell in row["cells"]
        if cell["status"] == "computed"
    }
    strength_samples: Dict[int, list] = {row["resolution"]: [] for row in observed["rows"]}
    strength_censored = dict.fromkeys(strength_samples, 0)
    for index in range(designs):
        random = sample_reference_design(configspace, data.count, seed + index + 1)
        result = compute_combinatorial_coverage(
            random, details=False, minimum_strengths=minimum_strengths
        )
        for row in result["rows"]:
            strength_samples[row["resolution"]].append(row["observed_strength"])
            strength_censored[row["resolution"]] += int(row["strength_is_lower_bound"])
            for cell in row["cells"]:
                key = (row["resolution"], cell["strength"])
                if key in samples:
                    samples[key].append(cell["coverage"])
    comparisons = []
    for row in observed["rows"]:
        for cell in row["cells"]:
            key = (row["resolution"], cell["strength"])
            if key in samples:
                summary = _random_summary(cell["coverage"], samples[key])
                comparisons.append(
                    {
                        "resolution": key[0],
                        "strength": key[1],
                        **summary,
                        "deficit": cell["coverage"] - summary["mean"],
                    }
                )
    strengths = []
    for row in observed["rows"]:
        resolution = row["resolution"]
        summary = _random_summary(row["observed_strength"], strength_samples[resolution])
        lower, upper = np.quantile(
            strength_samples[resolution], [0.05, 0.95], method="inverted_cdf"
        )
        summary.update(lower=int(lower), upper=int(upper))
        censored = bool(strength_censored[resolution] or row["strength_is_lower_bound"])
        if censored:
            summary["percentile"] = None
        strengths.append(
            {
                "resolution": resolution,
                **summary,
                "censored": censored,
                "samples": strength_samples[resolution],
                "lower_bound_designs": strength_censored[resolution],
            }
        )
    return {
        "baseline_version": BASELINE_VERSION,
        "observed": observed,
        "designs": designs,
        "seed": seed,
        "cells": comparisons,
        "strengths": strengths,
    }
