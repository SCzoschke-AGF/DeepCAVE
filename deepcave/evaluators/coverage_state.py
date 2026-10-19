"""Shared, bounded missing-tuple accounting for coverage planning and forecasts."""

from typing import TYPE_CHECKING

from itertools import combinations
from math import comb, prod

import numpy as np

if TYPE_CHECKING:
    from deepcave.evaluators.combinatorial import CoverageSpec


class CoverageState:
    """Index missing tuples without imposing the current trial-count bound."""

    def __init__(
        self,
        spec: "CoverageSpec",
        strength: int,
        max_groups: int = 10_000,
        max_tuples: int = 5_000_000,
        max_projections: int = 20_000_000,
    ) -> None:
        dimensions = len(spec.axes)
        if not 1 <= strength <= dimensions:
            raise ValueError(f"Target strength must be between 1 and {dimensions}.")
        group_count = comb(dimensions, strength)
        if group_count > max_groups or group_count * len(spec.points) > max_projections:
            raise ValueError("Coverage exceeds the parameter-group work limit.")
        groups = list(combinations(range(dimensions), strength))
        totals = [prod(spec.sizes[column] for column in group) for group in groups]
        self.total = sum(totals)
        if self.total > max_tuples:
            raise ValueError("Coverage exceeds the tuple limit.")
        self.groups = np.asarray(groups, dtype=np.int64)
        self.sizes = np.asarray(spec.sizes, dtype=np.int64)
        self.offsets = np.r_[0, np.cumsum(totals[:-1])].astype(np.int64)
        self.missing = np.ones(self.total, dtype=bool)
        batch = max(1, 250_000 // group_count)
        for start in range(0, len(spec.points), batch):
            self.missing[self.tuple_ids(spec.points[start : start + batch])] = False

    def tuple_ids(self, points: np.ndarray) -> np.ndarray:
        """Encode one mixed-radix tuple per parameter group and configuration."""
        points = np.atleast_2d(points)
        codes = np.zeros((len(points), len(self.groups)), dtype=np.int64)
        for columns in self.groups.T:
            codes *= self.sizes[columns]
            codes += points[:, columns]
        return codes + self.offsets

    def gains(self, points: np.ndarray) -> np.ndarray:
        """Count additional covered tuples in bounded batches."""
        batch = max(1, 250_000 // len(self.groups))
        return (
            np.concatenate(
                [
                    self.missing[self.tuple_ids(points[start : start + batch])].sum(axis=1)
                    for start in range(0, len(points), batch)
                ]
            )
            if len(points)
            else np.empty(0, dtype=int)
        )

    def add(self, point: np.ndarray) -> np.ndarray:
        """Commit a configuration and return newly covered tuple IDs."""
        ids = self.tuple_ids(point)[0]
        new = ids[self.missing[ids]]
        self.missing[ids] = False
        return new

    def lower_bound(self) -> int:
        """One configuration covers at most one missing tuple in each group."""
        return int(np.add.reduceat(self.missing.astype(np.int64), self.offsets).max())

    def complete_tuple(self, point: np.ndarray, tuple_id: int) -> None:
        """Fix one missing tuple's levels in a complete candidate."""
        group = int(np.searchsorted(self.offsets, tuple_id, side="right") - 1)
        code = int(tuple_id - self.offsets[group])
        for column in self.groups[group, ::-1]:
            point[column] = code % self.sizes[column]
            code //= self.sizes[column]
