"""Experimental fixed partitions of legal sampling space; no dashboard integration."""

from typing import Any, Dict, List, Optional, Tuple

from dataclasses import dataclass

import numpy as np

from deepcave.evaluators import coverage


def coordinates(data: Any) -> Tuple[np.ndarray, list]:
    """Use ordered sampling coordinates, category identities, and a separate inactive symbol."""
    points, axes = coverage._mixed_coordinates(data)
    return np.where(np.isnan(points), -1.0, points), axes


@dataclass
class Node:
    """A numerical threshold or unordered category subset, or a terminal cell."""

    axis: int = -1
    threshold: float = 0
    categories: Optional[tuple] = None
    missing_left: bool = False
    left: int = -1
    right: int = -1
    leaf: int = -1

    def mask(self, values: np.ndarray) -> np.ndarray:
        """Route inactive values without giving them a numerical ordering."""
        if self.categories is not None:
            if len(self.categories) == 1:
                return values == self.categories[0]
            return np.isin(values, self.categories)
        return ((values >= 0) & (values <= self.threshold)) | ((values < 0) & self.missing_left)


class RandomPartition:
    """Random axis splits balanced against an independent weighted reference pool."""

    def __init__(
        self, unique: np.ndarray, weights: np.ndarray, axes: list, cells: int, seed: int
    ) -> None:
        if cells < 1 or len(unique) < 1:
            raise ValueError("Positive cell count and nonempty training pool are required.")
        self.nodes: List[Node] = []
        self.training_mass: List[float] = []
        rng = np.random.default_rng(seed)

        def build(indices: np.ndarray, requested: int) -> int:
            node_index = len(self.nodes)
            node = Node()
            self.nodes.append(node)
            requested = min(requested, len(indices))
            if requested > 1:
                goal_left = requested // 2
                target = weights[indices].sum() * goal_left / requested
                for column in rng.permutation(len(axes)):
                    values = unique[indices, column]
                    labels, inverse = np.unique(values, return_inverse=True)
                    if len(labels) < 2:
                        continue
                    masses = np.bincount(inverse, weights=weights[indices])
                    node.axis = int(column)
                    if axes[column].kind == "categorical":
                        order = rng.permutation(len(labels))
                        cut = int(np.argmin(np.abs(np.cumsum(masses[order])[:-1] - target))) + 1
                        node.categories = tuple(labels[order[:cut]].tolist())
                    else:
                        missing = labels < 0
                        missing_mass = float(masses[missing].sum())
                        active_labels, active_masses = labels[~missing], masses[~missing]
                        thresholds = np.r_[
                            -np.inf, (active_labels[:-1] + active_labels[1:]) / 2, np.inf
                        ]
                        cumulative = np.r_[0, np.cumsum(active_masses)]
                        candidates = []
                        for missing_left in ([False, True] if missing_mass else [False]):
                            left_mass = cumulative + missing_mass * missing_left
                            valid = (left_mass > 0) & (left_mass < masses.sum())
                            costs = np.where(valid, np.abs(left_mass - target), np.inf)
                            for cut in np.flatnonzero(costs == costs.min()) if valid.any() else []:
                                candidates.append(
                                    (
                                        abs(left_mass[cut] - target),
                                        float(thresholds[cut]),
                                        missing_left,
                                    )
                                )
                        if not candidates:
                            continue
                        best = min(candidate[0] for candidate in candidates)
                        tied = [candidate for candidate in candidates if candidate[0] == best]
                        _, node.threshold, node.missing_left = tied[int(rng.integers(len(tied)))]
                    left = node.mask(values)
                    left_indices, right_indices = indices[left], indices[~left]
                    # Do not invent fractional categories or more cells than distinct
                    # configurations. Reallocate the leaf budget when one side is atomic.
                    goal_left = int(
                        np.clip(
                            goal_left,
                            max(1, requested - len(right_indices)),
                            min(len(left_indices), requested - 1),
                        )
                    )
                    node.left = build(left_indices, goal_left)
                    node.right = build(right_indices, requested - goal_left)
                    return node_index
            node.leaf = len(self.training_mass)
            self.training_mass.append(float(weights[indices].sum() / weights.sum()))
            return node_index

        build(np.arange(len(unique)), cells)
        self.cells = len(self.training_mass)

    def assign(self, points: np.ndarray) -> np.ndarray:
        """Classify existing legal configurations; generate no synthetic configurations."""
        result = np.empty(len(points), dtype=int)
        stack = [(0, np.arange(len(points)))]
        while stack:
            index, rows = stack.pop()
            if not len(rows):
                continue
            node = self.nodes[index]
            if node.leaf >= 0:
                result[rows] = node.leaf
            else:
                left = node.mask(points[rows, node.axis])
                stack.append((node.left, rows[left]))
                stack.append((node.right, rows[~left]))
        return result


class PartitionEnsemble:
    """Hold independent-of-observation partitions and separately estimated cell masses."""

    def __init__(
        self, training: Any, mass_reference: Any, cells: int, partitions: int, seed: int
    ) -> None:
        points, axes = coordinates(training)
        unique, weights = np.unique(points, axis=0, return_counts=True)
        mass_points, _ = coordinates(mass_reference)
        self.partitions = [
            RandomPartition(unique, weights, axes, cells, seed + i) for i in range(partitions)
        ]
        self.masses = [
            np.bincount(tree.assign(mass_points), minlength=tree.cells) / len(mass_points)
            for tree in self.partitions
        ]
        self.requested_cells = cells

    def measure(self, points: np.ndarray, designs: int, trials: int) -> Dict[str, np.ndarray]:
        """Measure each design against every fixed partition, preserving design identity."""
        if trials < 1 or designs * trials != len(points):
            raise ValueError("The point pool must contain equally sized nonempty designs.")
        result = {
            key: np.empty((designs, len(self.partitions)))
            for key in ("empty_mass", "tv", "empty_fraction")
        }
        design_ids = np.repeat(np.arange(designs), trials)
        for index, (tree, mass) in enumerate(zip(self.partitions, self.masses)):
            labels = tree.assign(points)
            counts = np.bincount(design_ids * tree.cells + labels, minlength=designs * tree.cells)
            counts = counts.reshape(designs, tree.cells)
            result["empty_mass"][:, index] = ((counts == 0) * mass).sum(axis=1)
            result["empty_fraction"][:, index] = (counts == 0).mean(axis=1)
            result["tv"][:, index] = 0.5 * np.abs(counts / trials - mass).sum(axis=1)
        return result

    def diagnostics(self, trials: int) -> dict:
        """Disclose unequal cell masses instead of assuming exact equiprobability."""
        masses = np.concatenate(self.masses)
        return {
            "requested_cells": self.requested_cells,
            "actual_cells_min": min(tree.cells for tree in self.partitions),
            "actual_cells_max": max(tree.cells for tree in self.partitions),
            "mass_min": float(masses.min()),
            "mass_max": float(masses.max()),
            "zero_mass_cells": int(np.sum(masses == 0)),
            "median_mass_cv": float(np.median([p.std() / p.mean() for p in self.masses])),
            "expected_empty_mass": float(
                np.mean([np.sum(p * (1 - p) ** trials) for p in self.masses])
            ),
            "expected_empty_fraction": float(
                np.mean([np.mean((1 - p) ** trials) for p in self.masses])
            ),
        }


def aggregate(measures: Dict[str, np.ndarray], fit_designs: int) -> Dict[str, np.ndarray]:
    """Fit per-partition scales on independent IID draws, then apply fixed aggregates."""
    result = {}
    for name in ("empty_mass", "tv"):
        matrix = measures[name]
        centre = matrix[:fit_designs].mean(axis=0)
        scale = np.maximum(matrix[:fit_designs].std(axis=0, ddof=1), 1e-12)
        result[f"mean_{name}"] = matrix.mean(axis=1)
        result[f"max_z_{name}"] = ((matrix - centre) / scale).max(axis=1)
    return result


def forecast(
    ensembles: Dict[int, PartitionEnsemble],
    observed: Any,
    future: Any,
    simulations: int,
    horizon: int,
    checkpoints: List[int],
    reduction: float = 0.2,
) -> dict:
    """Append the same future configurations across all fixed partitions in each simulation."""
    if len(future.points) != simulations * horizon:
        raise ValueError("Future pool must contain simulations times horizon configurations.")
    initial_points, _ = coordinates(observed)
    future_points, _ = coordinates(future)
    curves = {}
    for cells, ensemble in ensembles.items():
        losses = np.zeros((simulations, horizon + 1))
        for tree, mass in zip(ensemble.partitions, ensemble.masses):
            initially_empty = np.bincount(tree.assign(initial_points), minlength=tree.cells) == 0
            first_hit = np.full((simulations, tree.cells), horizon + 1, dtype=int)
            labels = tree.assign(future_points).reshape(simulations, horizon)
            np.minimum.at(
                first_hit,
                (np.repeat(np.arange(simulations), horizon), labels.ravel()),
                np.tile(np.arange(1, horizon + 1), simulations),
            )
            events = np.zeros((simulations, horizon + 1))
            rows, cells_hit = np.where((first_hit <= horizon) & initially_empty[None, :])
            np.add.at(events, (rows, first_hit[rows, cells_hit]), mass[cells_hit])
            losses += mass[initially_empty].sum() - np.cumsum(events, axis=1)
        losses /= len(ensemble.partitions)
        initial = float(losses[0, 0])
        curves[cells] = losses
        if initial <= 0:
            continue
        if np.any(np.diff(losses, axis=1) > 1e-12):
            raise AssertionError("Fixed-partition unvisited mass must be monotone.")
    return {
        str(cells): {
            "initial_mean_empty_mass": float(losses[0, 0]),
            "reduction_target": reduction,
            "simulations": simulations,
            "budgets": [
                {
                    "additional_trials": step,
                    "successes": int(np.sum(losses[:, step] <= (1 - reduction) * losses[:, 0])),
                    "mean_empty_mass": float(losses[:, step].mean()),
                }
                for step in checkpoints
            ],
        }
        for cells, losses in curves.items()
        if losses[0, 0] > 0
    }
