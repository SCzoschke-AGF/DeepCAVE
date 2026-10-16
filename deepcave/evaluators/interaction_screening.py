"""Deterministic quadrature screening of a fitted fANOVA forest's pair interactions."""

from typing import Any, Dict, List, Sequence, Tuple

import itertools

import numpy as np


def _grid(bounds: Tuple[float, float], size: int, phase: float) -> List[Tuple[float, float]]:
    """Use the exact evaluator's encoded bounds and categorical counting measure."""
    lower, upper = bounds
    if np.isnan(upper):
        count = int(lower)
        if count <= 0:
            return [(0.0, 1.0)]
        edges = np.linspace(0, count, min(size, count) + 1, dtype=int)
        return [
            (float(min(stop - 1, start + int(phase * (stop - start)))), float(stop - start))
            for start, stop in zip(edges[:-1], edges[1:])
        ]
    if lower == upper:
        return [(float(lower), 1.0)]
    width = (upper - lower) / size
    return [(float(lower + (index + phase) * width), float(width)) for index in range(size)]


def estimate_pair(
    forest: Any, pair: Tuple[int, int], size: int, phase: float = 0.5
) -> Dict[str, float]:
    """Integrate squared pure interaction per tree, retaining exact lower-order marginals.

    The native marginal predictor integrates unspecified dimensions exactly. Grid weights
    include its marginal mass, just as in the exact evaluator. Never average predictions
    across trees before squaring, and never subtract the sampled residual mean.
    """
    if size < 2 or not 0 < phase < 1:
        raise ValueError("Grid size must be at least two and phase must lie in (0, 1).")
    cache = forest._interaction_screening_cache
    key = (pair, size, phase)
    if key in cache["pairs"]:
        return dict(cache["pairs"][key])
    grids = [_grid(forest.bounds[index], size, phase) for index in pair]
    scores, offsets = [], []

    def marginal(
        tree: int, dimensions: Tuple[int, ...], values: Tuple[float, ...]
    ) -> Tuple[float, float]:
        sample = np.full(forest.n_params, np.nan)
        sample[list(dimensions)] = values
        stat = forest._model.marginal_prediction_stat_of_tree(tree, sample.tolist())
        return float(stat.mean()), float(stat.sum_of_weights())

    for tree, variance in enumerate(forest.trees_total_variance):
        if not np.isfinite(variance) or variance <= 0:
            continue
        constant_key = (tree, (), ())
        if constant_key not in cache["lower"]:
            cache["lower"][constant_key] = marginal(tree, (), ())[0]
        constant = cache["lower"][constant_key]
        lower = []
        for dimension, grid in zip(pair, grids):
            means = {}
            for value, _weight in grid:
                lower_key = (tree, (dimension,), (value,))
                if lower_key not in cache["lower"]:
                    cache["lower"][lower_key] = marginal(tree, (dimension,), (value,))[0]
                means[value] = cache["lower"][lower_key]
            lower.append(means)
        weight_sum = squared_sum = residual_sum = 0.0
        for (a, wa), (b, wb) in itertools.product(*grids):
            joint, mass = marginal(tree, pair, (a, b))
            residual = joint - lower[0][a] - lower[1][b] + constant
            weight = wa * wb * mass
            if weight > 0 and np.isfinite(residual):
                weight_sum += weight
                squared_sum += weight * residual * residual
                residual_sum += weight * residual
        if weight_sum > 0:
            scores.append(squared_sum / weight_sum / variance)
            offsets.append((residual_sum / weight_sum) ** 2 / variance)
    if not scores:
        raise RuntimeError(
            "No trees with finite positive variance available for interaction screening."
        )
    result = {"score": float(np.mean(scores)), "offset": float(np.mean(offsets))}
    cache["pairs"][key] = result
    return dict(result)


def screen_interactions(
    forest: Any,
    hp_ids: Sequence[int],
    top_k: int = 5,
    coarse_size: int = 8,
    medium_size: int = 16,
    absolute_tolerance: float = 0.002,
    relative_tolerance: float = 0.2,
    offset_tolerance: float = 0.001,
    audit_pairs: int = 2,
) -> List[Dict[str, Any]]:
    """Rank all pairs, refine candidates and offsets, and audit part of the low-score tail.

    Stability describes numerical resolution sensitivity, not statistical confidence.
    Cached grid estimates are reusable for different shortlists on the same fitted forest.
    """
    if top_k < 0 or coarse_size < 2 or medium_size <= coarse_size:
        raise ValueError("Require top_k >= 0 and 2 <= coarse_size < medium_size.")
    names = list(forest.cs.keys())
    rows = []
    for pair in itertools.combinations(sorted(set(hp_ids)), 2):
        estimate = estimate_pair(forest, pair, coarse_size)
        rows.append(
            {
                "pair": pair,
                "members": [names[index] for index in pair],
                "coarse": estimate["score"],
                "medium": None,
                "score": estimate["score"],
                "offset": estimate["offset"],
                "stability": None,
                "audit_difference": None,
                "status": "Coarse",
            }
        )

    def ranking(row: Dict[str, Any]) -> Tuple[float, List[str]]:
        return -row["score"], row["members"]

    def changed(first: float, second: float) -> bool:
        return abs(first - second) > max(
            absolute_tolerance, relative_tolerance * max(first, second)
        )

    rows.sort(key=ranking)
    candidate_count = min(len(rows), max(1, top_k or 5))
    selected = {row["pair"] for row in rows[: min(len(rows), candidate_count + 2)]}
    cutoff = rows[candidate_count - 1]["score"] if rows else 0
    selected.update(
        row["pair"]
        for row in rows
        if row["offset"] > offset_tolerance or (cutoff > 0 and row["score"] >= 0.8 * cutoff)
    )
    tail = [row for row in rows if row["pair"] not in selected]
    for index in np.linspace(0, len(tail) - 1, min(len(tail), audit_pairs), dtype=int):
        row = tail[index]
        audit = estimate_pair(forest, row["pair"], coarse_size, phase=0.25)
        row["audit_difference"] = abs(row["coarse"] - audit["score"])
        row["status"] = "Audited coarse"
        if changed(row["coarse"], audit["score"]) or audit["offset"] > offset_tolerance:
            selected.add(row["pair"])

    def refine(row: Dict[str, Any]) -> None:
        estimate = estimate_pair(forest, row["pair"], medium_size)
        row.update(
            medium=estimate["score"],
            score=estimate["score"],
            offset=estimate["offset"],
            stability=abs(row["coarse"] - estimate["score"]),
            status=(
                "Refine recommended"
                if (
                    changed(row["coarse"], estimate["score"])
                    or estimate["offset"] > offset_tolerance
                    or estimate["score"] > 1 + absolute_tolerance
                )
                else "Stable"
            ),
        )

    for row in rows:
        if row["pair"] in selected:
            refine(row)
    # Refining leaders may promote a previously coarse pair into the shortlist.
    while True:
        rows.sort(key=ranking)
        cutoff = rows[candidate_count - 1]["score"] if rows else 0
        pending = [
            row
            for index, row in enumerate(rows)
            if row["medium"] is None
            and (index < candidate_count or (cutoff > 0 and row["score"] >= 0.8 * cutoff))
        ]
        if not pending:
            break
        for row in pending:
            refine(row)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        del row["pair"]
    return rows
