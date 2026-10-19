"""Independent search-space coverage analyses in sampling coordinates.

Inactive parameters remain NaN. Discrete coordinates are probability-mass
midpoints; categorical distance uses equality rather than category order.
No expensive analysis is called by preprocessing or univariate coverage.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.types import NotSet
from scipy.spatial import cKDTree
from scipy.stats import kstest, qmc

from deepcave.runs.status import Status

VERSION = 6
DEFAULT_BINS = 20
DEFAULT_REFERENCES = 2048
PAIR_REFERENCES = 512
PAIR_MMD_MAX_TRIALS = 1024
PAIR_KERNEL_SCALE = 0.2
MIXED_MMD_VERSION = 1
MIXED_MMD_MAX_TRIALS = 1024
MIXED_MMD_REFERENCES = 512
MIXED_MMD_KERNEL_SCALE = 0.2


def _python(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


@dataclass
class CoverageAxis:
    """A parameter definition and its sampling-space adapter."""

    hp: Any
    conditional: bool = False

    @property
    def name(self) -> str:
        """Return the parameter name."""
        return str(self.hp.name)

    @property
    def kind(self) -> str:
        """Return the semantic parameter type."""
        if isinstance(self.hp, Constant):
            return "constant"
        if isinstance(self.hp, CategoricalHyperparameter):
            return "categorical"
        if isinstance(self.hp, OrdinalHyperparameter):
            return "ordinal"
        if isinstance(self.hp, IntegerHyperparameter):
            return "integer"
        return "continuous"

    @property
    def step(self) -> Optional[float]:
        """Return the discrete lattice step, when defined."""
        step = (self.hp.meta or {}).get("step", getattr(self.hp, "step", None))
        if step is not None and float(step) > 0:
            return float(step)
        return 1.0 if self.kind == "integer" else None

    @property
    def choices(self) -> List[Any]:
        """Return category, ordinal or constant values in definition order."""
        if self.kind == "categorical":
            return [_python(v) for v in self.hp.choices]
        if self.kind == "ordinal":
            return [_python(v) for v in self.hp.sequence]
        if self.kind == "constant":
            return [_python(self.hp.value)]
        return []

    @property
    def size(self) -> Optional[int]:
        """Return finite cardinality without materializing numerical lattices."""
        if self.kind in ("categorical", "ordinal", "constant"):
            return len(self.choices)
        if self.step is not None:
            return int(np.floor((self.hp.upper - self.hp.lower) / self.step + 1e-8)) + 1
        return None

    def ranks(self, values: Sequence[Any]) -> np.ndarray:
        """Return lattice indices without enumerating large integer ranges."""
        if self.kind in ("categorical", "ordinal", "constant"):
            choices = self.choices
            return np.array([choices.index(_python(v)) for v in values], dtype=int)
        return np.rint((np.asarray(values, dtype=float) - self.hp.lower) / self.step).astype(int)

    def values_at(self, ranks: np.ndarray) -> List[Any]:
        """Decode discrete lattice indices into native values."""
        if self.kind in ("categorical", "ordinal", "constant"):
            return [self.choices[int(i)] for i in ranks]
        values = self.hp.lower + np.asarray(ranks) * self.step
        if self.kind == "integer":
            return values.astype(int).tolist()
        return values.tolist()

    def _cdf(self, vectors: np.ndarray) -> np.ndarray:
        # ConfigSpace 1.2 stores the actual (possibly truncated) sampling RV in
        # vector coordinates. Using its CDF also supports Normal and Beta parameters.
        rv = getattr(self.hp._vector_dist, "rv", None)
        return np.asarray(rv.cdf(vectors) if rv is not None else vectors, dtype=float)

    @property
    def integer_probabilities(self) -> Optional[np.ndarray]:
        """Return explicit masses in support order, when the integer definition has them."""
        if self.kind != "integer":
            return None
        probabilities = (self.hp.meta or {}).get("probabilities")
        if probabilities is None:
            probabilities = getattr(self.hp, "probabilities", None)
        if probabilities is None:
            probabilities = getattr(self.hp._vector_dist, "probabilities", None)
        if probabilities is None:
            return None
        probabilities = np.asarray(probabilities, dtype=float)
        if (
            probabilities.shape != (self.size,)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or probabilities.sum() <= 0
        ):
            raise ValueError(
                f"{self.name}: probabilities must contain one nonnegative, finite mass "
                "per valid support value and have a positive sum."
            )
        return probabilities / probabilities.sum()

    def mass_edges(self, ranks: np.ndarray) -> tuple:
        """Return the CDF interval occupied by each discrete value."""
        size = self.size
        if self.kind == "categorical":
            weights = np.asarray(self.hp.probabilities, dtype=float)
            edges = np.r_[0.0, np.cumsum(weights)]
            return edges[ranks], edges[ranks + 1]
        if self.kind in ("ordinal", "constant"):
            return ranks / size, (ranks + 1) / size
        probabilities = self.integer_probabilities
        if probabilities is not None:
            edges = np.r_[0.0, np.cumsum(probabilities)]
            edges[-1] = 1.0
            return edges[ranks], edges[ranks + 1]
        left, right = ranks / size, (ranks + 1) / size
        if getattr(self.hp, "log", False):
            # ConfigSpace quantizes log integers in value space before converting
            # back to vector coordinates (not by rounding log coordinates).
            width = float(self.hp.upper - self.hp.lower)
            lo = np.log(float(self.hp.lower))
            span = np.log(float(self.hp.upper)) - lo
            left = (np.log(self.hp.lower + left * width) - lo) / span
            right = (np.log(self.hp.lower + right * width) - lo) / span
        return self._cdf(left), self._cdf(right)

    def encode(self, values: Sequence[Any]) -> np.ndarray:
        """Map active native values into sampling coordinates."""
        if self.size is not None:
            left, right = self.mass_edges(self.ranks(values))
            return (left + right) / 2
        return np.clip(self._cdf(self.hp.to_vector(np.asarray(values, dtype=float))), 0, 1)

    def metadata(self) -> Dict[str, Any]:
        """Describe the definition, distribution and transformation."""
        distribution = type(self.hp).__name__.replace("Hyperparameter", "")
        if distribution.startswith("Uniform") and getattr(self.hp, "log", False):
            distribution = "Log" + distribution
        numeric = self.kind in ("continuous", "integer")
        integer_probabilities = self.integer_probabilities
        return {
            "name": self.name,
            "type": self.kind,
            "conditional": self.conditional,
            "distribution": distribution,
            "lower": _python(self.hp.lower) if numeric else None,
            "upper": _python(self.hp.upper) if numeric else None,
            "choices": self.choices,
            "probabilities": (
                list(map(float, self.hp.probabilities))
                if self.kind == "categorical"
                else (integer_probabilities.tolist() if integer_probabilities is not None else None)
            ),
            "step": self.step,
            "cardinality": self.size,
            "transformation": (
                "Discrete CDF midpoint"
                if self.size is not None
                else (
                    "Sampling CDF (log scale)" if getattr(self.hp, "log", False) else "Sampling CDF"
                )
            ),
        }


@dataclass
class CoverageData:
    """Reusable normalized history, retaining trial order and inactive values."""

    axes: List[CoverageAxis]
    values: List[Dict[str, Any]]
    points: np.ndarray
    fingerprint: str
    overview: Dict[str, Any]
    constrained: bool = False
    configspace: Optional[ConfigurationSpace] = None

    @property
    def count(self) -> int:
        """Return the number of sampled trials, including repetitions."""
        return len(self.values)


def normalize_configurations(
    configspace: ConfigurationSpace, configurations: Sequence[Mapping[str, Any]]
) -> CoverageData:
    """Prepare all coordinates once, without global or pairwise calculations."""
    axes = [
        CoverageAxis(hp, bool(configspace.parent_conditions_of[hp.name]))
        for hp in configspace.values()
    ]
    values = [{str(k): _python(v) for k, v in dict(c).items()} for c in configurations]
    points = np.full((len(values), len(axes)), np.nan)
    for j, axis in enumerate(axes):
        active = [i for i, config in enumerate(values) if config.get(axis.name) is not None]
        if active:
            points[active, j] = axis.encode([values[i][axis.name] for i in active])
    return CoverageData(
        axes, values, points, "", {}, bool(configspace.forbidden_clauses), configspace
    )


def prepare_coverage(run: Any) -> CoverageData:
    """Normalize successful trials in history order, retaining repetitions."""
    trials = [trial for trial in run.history if trial.status == Status.SUCCESS]
    configs: Dict[int, Dict[str, Any]] = {}
    for trial in trials:
        if trial.config_id not in configs:
            configs[trial.config_id] = dict(run.get_config(trial.config_id))
    data = normalize_configurations(run.configspace, [configs[t.config_id] for t in trials])
    definition = run.configspace.to_serialized_dict()
    signature = [
        VERSION,
        definition,
        [[t.config_id, t.budget, t.seed, int(t.status)] for t in run.history],
        data.values,
    ]
    data.fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True, default=_python).encode()
    ).hexdigest()
    statuses = {status.name: sum(t.status == status for t in run.history) for status in Status}
    failures = (Status.TIMEOUT, Status.MEMORYOUT, Status.CRASHED, Status.ABORTED, Status.FAILED)
    data.overview = {
        "trials": data.count,
        "total_trials": len(run.history),
        "complete": statuses["SUCCESS"],
        "failed": sum(statuses[s.name] for s in failures),
        "statuses": statuses,
        "unique_configurations": len(configs),
        "parameters": len(data.axes),
        "types": {
            kind: sum(a.kind == kind for a in data.axes)
            for kind in ("continuous", "integer", "categorical", "ordinal", "constant")
        },
        "conditional": sum(a.conditional for a in data.axes),
    }
    return data


def _integer_ecdf(axis: CoverageAxis, visited: np.ndarray, counts: np.ndarray) -> Dict:
    """Compute support-based CDFs and an exact descriptive distance, without a p-value."""
    size = axis.size
    assert size is not None
    cumulative = np.r_[0, np.cumsum(counts)]
    count = int(cumulative[-1])

    def observed_at(ranks: np.ndarray) -> np.ndarray:
        return cumulative[np.searchsorted(visited, ranks, side="right")] / count

    # Between observations the empirical CDF is constant and the reference is monotone.
    # Its largest difference is thus at a visited rank or just before it, including edges.
    check_ranks = np.unique(np.r_[0, size - 1, visited, visited[visited > 0] - 1])
    _, expected = axis.mass_edges(check_ranks)
    ks = float(np.max(np.abs(observed_at(check_ranks) - expected))) if count else None
    complete = size <= 1000
    if complete:
        ranks = np.arange(size)
    else:
        selected = visited[
            np.unique(np.linspace(0, len(visited) - 1, min(len(visited), 1000), dtype=int))
        ]
        ranks = np.unique(
            np.r_[
                np.linspace(0, size - 1, 1000, dtype=int),
                selected,
                selected[selected > 0] - 1,
            ]
        )
    support = axis.values_at(ranks)
    _, expected = axis.mass_edges(ranks)
    return {
        "ecdf_x": support if count else [],
        "ecdf_y": observed_at(ranks).tolist() if count else [],
        "expected_cdf_x": support,
        "expected_cdf_y": expected.tolist(),
        "ecdf_complete": complete,
        "ks": ks,
    }


def compute_univariate_coverage(data: CoverageData, bins: int = DEFAULT_BINS) -> List[Dict]:
    """Compute cheap marginals, continuous sampling ECDFs and integer support ECDFs."""
    results = []
    for j, axis in enumerate(data.axes):
        active = np.isfinite(data.points[:, j])
        u = np.sort(data.points[active, j])
        values = [config[axis.name] for config in data.values if config.get(axis.name) is not None]
        row = axis.metadata()
        row.update(active_trials=len(u), inactive_trials=data.count - len(u))
        if axis.size is not None:
            visited, counts = np.unique(axis.ranks(values), return_counts=True)
            ranks = np.arange(axis.size) if axis.size <= 1000 else visited
            left, right = axis.mass_edges(ranks)
            lookup = dict(zip(visited.tolist(), counts.tolist()))
            observed = [lookup.get(int(rank), 0) for rank in ranks]
            expected = (right - left) * len(u)
            gaps = np.diff(np.r_[-1, visited, axis.size]) - 1
            row.update(
                labels=[str(v) for v in axis.values_at(ranks)],
                frequencies=observed,
                expected=expected.tolist(),
                relative_deviation=[
                    float((actual - exp) / exp) if exp > 0 else None
                    for actual, exp in zip(observed, expected)
                ],
                unique=len(visited),
                unseen_count=axis.size - len(visited),
                unseen=[str(v) for v, n in zip(axis.values_at(ranks), observed) if n == 0],
                coverage=float(len(visited) / axis.size),
                largest_missing_run=int(max(gaps)) if len(gaps) else axis.size,
                frequency_table_complete=axis.size <= 1000,
                ks=None,
            )
        else:
            frequencies, edges = np.histogram(u, bins=bins, range=(0, 1))
            gap = None
            if len(u):
                boundaries = np.r_[0, u, 1]
                raw_boundaries = np.r_[axis.hp.lower, np.sort(values), axis.hp.upper]
                index = int(np.argmax(np.diff(boundaries)))
                gap = dict(
                    lower=float(boundaries[index]),
                    upper=float(boundaries[index + 1]),
                    lower_value=float(raw_boundaries[index]),
                    upper_value=float(raw_boundaries[index + 1]),
                    lower_observed=index > 0,
                    upper_observed=index < len(u),
                    width=float(boundaries[index + 1] - boundaries[index]),
                )
            row.update(
                trial_coordinates=u.tolist(),
                largest_gap=gap,
                labels=((edges[:-1] + edges[1:]) / 2).tolist(),
                frequencies=frequencies.tolist(),
                expected=[len(u) / bins] * bins,
                unique=len(np.unique(np.asarray(values))),
                coverage=float(np.count_nonzero(frequencies) / bins),
                ks=float(kstest(u, "uniform").statistic) if len(u) else None,
            )
        if len(u):
            numeric = axis.kind in ("continuous", "integer")
            raw = np.sort(np.unique(np.asarray(values, dtype=float))) if numeric else np.array([])
            row.update(
                observed_min=float(raw[0]) if numeric else None,
                observed_max=float(raw[-1]) if numeric else None,
                range_fraction=(
                    float((raw[-1] - raw[0]) / (axis.hp.upper - axis.hp.lower))
                    if numeric and axis.hp.upper > axis.hp.lower
                    else None
                ),
                max_gap=float(np.max(np.diff(np.r_[0, u, 1]))),
                raw_max_gap=float(np.max(np.diff(raw))) if len(raw) > 1 else None,
                edge_left=float(u[0]),
                edge_right=float(1 - u[-1]),
                occupied_bins=int(np.count_nonzero(np.histogram(u, bins=bins, range=(0, 1))[0])),
            )
        else:
            row.update(
                observed_min=None,
                observed_max=None,
                range_fraction=None,
                max_gap=1.0,
                raw_max_gap=None,
                edge_left=None,
                edge_right=None,
                occupied_bins=0,
            )
        # Bound browser payloads while retaining exact statistics above.
        indices = np.unique(np.linspace(0, max(0, len(u) - 1), min(len(u), 1000), dtype=int))
        row.update(
            bins=bins,
            ecdf_x=u[indices].tolist(),
            ecdf_y=((indices + 1) / len(u)).tolist() if len(u) else [],
        )
        if axis.kind == "integer":
            row.update(_integer_ecdf(axis, visited, counts))
        results.append(row)
    return results


def _projection(data: CoverageData) -> tuple:
    indices = [
        i
        for i, a in enumerate(data.axes)
        if a.kind == "continuous" and a.size is None and not a.conditional
    ]
    names = [data.axes[i].name for i in indices]
    if data.constrained:
        return np.empty((data.count, 0)), [], "Unavailable for spaces with forbidden combinations."
    if not names:
        return np.empty((data.count, 0)), [], "No unconditional continuous parameters."
    points = data.points[:, indices]
    if not np.isfinite(points).all():
        return points, names, "The continuous projection contains missing values."
    return points, names, None


def compute_global_discrepancy(
    data: CoverageData, methods: Sequence[str] = ("CD", "L2-star", "WD")
) -> Dict[str, Any]:
    """Calculate cube discrepancy only on an eligible continuous projection."""
    points, names, reason = _projection(data)
    if not data.count:
        reason = "No sampled trials."
    if reason:
        return {"parameters": names, "reason": reason, "centered": None}
    labels = {"CD": "centered", "L2-star": "l2_star", "WD": "wrap_around"}
    return {
        "parameters": names,
        "reason": None,
        **{labels[method]: float(qmc.discrepancy(points, method=method)) for method in methods},
    }


def _mixed_coordinates(data: CoverageData) -> tuple:
    """Retain inactive states and encode categories by identity, independent of weights."""
    columns = [i for i, axis in enumerate(data.axes) if axis.kind != "constant"]
    axes = [data.axes[i] for i in columns]
    points = data.points[:, columns].copy()
    for j, axis in enumerate(axes):
        if axis.kind == "categorical":
            active = np.flatnonzero(np.isfinite(points[:, j]))
            points[active, j] = axis.ranks([data.values[i][axis.name] for i in active])
    return points, axes


def _mixed_kernel(left: np.ndarray, right: np.ndarray, axes: List[CoverageAxis]) -> np.ndarray:
    """Product kernel with a separate inactive state and no category ordering."""
    kernel = np.ones((len(left), len(right)))
    for j, axis in enumerate(axes):
        lhs, rhs = left[:, j, None], right[None, :, j]
        active_left, active_right = np.isfinite(lhs), np.isfinite(rhs)
        similarity = (
            (lhs == rhs).astype(float)
            if axis.kind == "categorical"
            else np.exp(-0.5 * ((lhs - rhs) / MIXED_MMD_KERNEL_SCALE) ** 2)
        )
        kernel *= np.where(active_left & active_right, similarity, ~(active_left | active_right))
    return kernel


def _mixed_reference(references: CoverageData) -> tuple:
    """Prepare one bounded reference sample shared by observed and random designs."""
    if not references.count:
        raise ValueError("Sampling distribution requires reference configurations.")
    points, axes = _mixed_coordinates(references)
    if len(points) > MIXED_MMD_REFERENCES:
        indices = np.random.default_rng(1).choice(len(points), MIXED_MMD_REFERENCES, replace=False)
        points = points[indices]
    return points, axes, float(_mixed_kernel(points, points, axes).mean())


def _sampling_uniformity(data: CoverageData, reference: tuple) -> Dict[str, Any]:
    """Extend continuous discrepancy with a joint mixed-space empirical MMD."""
    points, axes = _mixed_coordinates(data)
    reference_points, reference_axes, reference_mean = reference
    if [a.name for a in axes] != [a.name for a in reference_axes]:
        raise ValueError("Sampling references must use the same parameter axes in the same order.")
    reason = (
        "No sampled trials." if not data.count else "No varying parameters." if not axes else None
    )
    rows = compute_discrepancy_trajectory(data)
    # For ordinary runs, reuse both kernel matrices for all nested prefixes.
    # Larger histories use the same bounded, reproducible subsampling rule for
    # the observed history and every random design at each checkpoint.
    kernel = cross = None
    if not reason and data.count <= MIXED_MMD_MAX_TRIALS:
        kernel = _mixed_kernel(points, points, axes)
        cross = _mixed_kernel(points, reference_points, axes)
    for row in rows:
        count = row["trials"]
        used = min(count, MIXED_MMD_MAX_TRIALS)
        value = None
        if not reason:
            if kernel is not None and cross is not None:
                self_mean = float(kernel[:count, :count].mean())
                cross_mean = float(cross[:count].mean())
            else:
                indices = np.random.default_rng(0).choice(count, used, replace=False)
                selected = points[indices]
                self_mean = float(_mixed_kernel(selected, selected, axes).mean())
                cross_mean = float(_mixed_kernel(selected, reference_points, axes).mean())
            value = float(np.sqrt(max(0.0, self_mean + reference_mean - 2 * cross_mean)))
        row.update(mixed_mmd=value, mixed_trials=used)
    return {
        **compute_global_discrepancy(data),
        "mixed_mmd": rows[-1]["mixed_mmd"] if rows else None,
        "mixed_parameters": [axis.name for axis in axes],
        "mixed_parameter_types": {axis.name: axis.kind for axis in axes},
        "mixed_conditional_parameters": [axis.name for axis in axes if axis.conditional],
        "mixed_reason": reason,
        "mixed_trials": min(data.count, MIXED_MMD_MAX_TRIALS),
        "mixed_reference_samples": len(reference_points),
        "mixed_kernel_scale": MIXED_MMD_KERNEL_SCALE,
        "mixed_version": MIXED_MMD_VERSION,
        "checkpoints": rows,
    }


def compute_sampling_uniformity(data: CoverageData, references: CoverageData) -> Dict[str, Any]:
    """Compare the complete legal mixed search space with reference configurations."""
    return _sampling_uniformity(data, _mixed_reference(references))


def _pair_projection(data: CoverageData, columns: List[int], active: np.ndarray) -> CoverageData:
    """Use valid support coordinates for ordered values and identities for categories."""
    values = [data.values[i] for i in np.flatnonzero(active)]
    points = data.points[active][:, columns].copy()
    axes = [data.axes[j] for j in columns]
    for j, axis in enumerate(axes):
        if axis.size is not None:
            ranks = axis.ranks([config[axis.name] for config in values])
            points[:, j] = ranks if axis.kind == "categorical" else ranks / max(1, axis.size - 1)
    return CoverageData(axes, values, points, "", {})


def _pair_kernel_mean(left: np.ndarray, right: np.ndarray, axes: List[CoverageAxis]) -> float:
    """Average a product of Gaussian numeric kernels and categorical equality kernels."""
    total = 0.0
    for start in range(0, len(left), 128):
        for offset in range(0, len(right), 128):
            lhs, rhs = left[start : start + 128], right[offset : offset + 128]
            kernel = np.ones((len(lhs), len(rhs)))
            for j, axis in enumerate(axes):
                if axis.kind == "categorical":
                    kernel *= lhs[:, j, None] == rhs[None, :, j]
                else:
                    difference = (lhs[:, j, None] - rhs[None, :, j]) / PAIR_KERNEL_SCALE
                    kernel *= np.exp(-0.5 * difference**2)
            total += float(kernel.sum())
    return total / (len(left) * len(right))


def _pair_mmd(observed: CoverageData, references: CoverageData) -> tuple:
    """Compute the nonnegative biased empirical MMD, without a significance test."""
    points = observed.points
    if observed.count > PAIR_MMD_MAX_TRIALS:
        indices = np.random.default_rng(0).choice(
            observed.count, PAIR_MMD_MAX_TRIALS, replace=False
        )
        points = points[indices]
    reference_points = references.points
    if references.count > PAIR_REFERENCES:
        indices = np.random.default_rng(1).choice(references.count, PAIR_REFERENCES, replace=False)
        reference_points = reference_points[indices]
    squared = (
        _pair_kernel_mean(points, points, observed.axes)
        + _pair_kernel_mean(reference_points, reference_points, observed.axes)
        - 2 * _pair_kernel_mean(points, reference_points, observed.axes)
    )
    return float(np.sqrt(max(0.0, squared))), len(points), len(reference_points)


def _pair_grid(data: CoverageData, columns: List[int], bins: int) -> tuple:
    """Count joint-active trials in the shared continuous-bin/discrete-support grid."""
    active = np.isfinite(data.points[:, columns]).all(axis=1)
    indices = np.flatnonzero(active)
    shape = tuple(min(bins, data.axes[j].size or bins) for j in columns)
    cells = []
    for column, count in zip(columns, shape):
        axis = data.axes[column]
        if axis.size is not None:
            ranks = axis.ranks([data.values[k][axis.name] for k in indices])
            cells.append(np.minimum(count - 1, ranks * count // axis.size))
        else:
            cells.append(np.minimum(count - 1, (data.points[active, column] * count).astype(int)))
    grid = np.zeros(shape, dtype=int)
    np.add.at(grid, tuple(cells), 1)
    return grid, active


def compute_pairwise_coverage(
    data: CoverageData,
    bins: int = DEFAULT_BINS,
    include_distances: bool = True,
    references: Optional[CoverageData] = None,
) -> Dict[str, Any]:
    """Compute occupancy, MMD and maximum coverage radius for every parameter type."""
    if include_distances and references is None:
        if data.configspace is None:
            raise ValueError(
                "Pairwise distances require the configuration space or a reference design."
            )
        references = sample_reference_design(data.configspace, PAIR_REFERENCES, 0)
    if include_distances and references is not None:
        if [axis.name for axis in data.axes] != [axis.name for axis in references.axes]:
            raise ValueError(
                "Pairwise references must use the same parameter axes in the same order."
            )
    rows = []
    for i, x in enumerate(data.axes):
        if x.kind == "constant":
            continue
        for j in range(i + 1, len(data.axes)):
            y = data.axes[j]
            if y.kind == "constant":
                continue
            grid, active = _pair_grid(data, [i, j], bins)
            indices = np.flatnonzero(active)
            occupancy = float(np.count_nonzero(grid) / grid.size)
            mmd, radius, metric_trials, metric_references, reference_count = None, None, 0, 0, 0
            reason = "No joint observations." if not len(indices) else None
            if include_distances and references is not None:
                ref_active = np.isfinite(references.points[:, [i, j]]).all(axis=1)
                reference_count = int(np.count_nonzero(ref_active))
                if len(indices) and reference_count:
                    observed = _pair_projection(data, [i, j], active)
                    reference = _pair_projection(references, [i, j], ref_active)
                    mmd, metric_trials, metric_references = _pair_mmd(observed, reference)
                    radius = float(np.max(_nearest_distances(observed, reference)))
                elif len(indices):
                    reason = "No joint reference samples; increase the reference sample count."
            rows.append(
                {
                    "x": x.name,
                    "y": y.name,
                    "active_trials": len(indices),
                    "occupied_cells": int(np.count_nonzero(grid)),
                    "total_cells": int(grid.size),
                    "occupancy": occupancy,
                    "empty_cell_rate": 1 - occupancy,
                    "mmd": mmd,
                    "max_radius": radius,
                    "mmd_trials": metric_trials,
                    "mmd_reference_samples": metric_references,
                    "reference_samples": reference_count,
                    "distance_reason": reason,
                }
            )
    return {
        "pairs": sorted(rows, key=lambda row: (row["occupancy"], -(row["mmd"] or 0))),
        "bins": bins,
        "note": "Nominal projection cells; conditional or forbidden cells may be unreachable. "
        "Discrete axes above the bin limit are grouped. MMD and maximum radius compare joint-active "
        "trials with legal reference configurations for every parameter type; lower is better. "
        f"Reference configurations: {references.count if references is not None else 0}. "
        f"Gaussian kernel scale: {PAIR_KERNEL_SCALE}; "
        f"categorical equality kernel. MMD uses at most {PAIR_MMD_MAX_TRIALS} trials and "
        f"{PAIR_REFERENCES} references; "
        "radius uses all joint-active trials. No p-value is computed.",
    }


def sample_reference_design(configspace: ConfigurationSpace, count: int, seed: int) -> CoverageData:
    """Sample a reproducible legal design, including metadata-defined lattices."""
    if count < 1:
        raise ValueError("At least one reference sample is required.")
    space = deepcopy(configspace)
    axes = [CoverageAxis(hp) for hp in space.values()]
    custom_steps = any(
        (a.step is not None and (a.kind == "continuous" or a.step != 1))
        or a.integer_probabilities is not None
        for a in axes
    )
    space.seed(int(seed))
    if not custom_steps:
        samples = space.sample_configuration(size=count)
        samples = [samples] if isinstance(samples, Configuration) else samples
        return normalize_configurations(space, samples)
    rng = np.random.RandomState(seed)
    configurations: List[Dict[str, Any]] = []
    attempts = 0
    while len(configurations) < count:
        attempts += 1
        if attempts > max(10000, count * 1000):
            raise ValueError("Could not sample enough legal configurations on the step lattice.")
        values = {}
        condition_values = dict.fromkeys(space.keys(), NotSet)
        for axis in axes:
            conditions = space.parent_conditions_of[axis.name]
            if not all(condition.satisfied_by_value(condition_values) for condition in conditions):
                continue
            if axis.size is not None and axis.kind in ("continuous", "integer"):
                # Draw from the same quantization intervals as mass_edges, without
                # enumerating a potentially enormous integer lattice.
                probabilities = axis.integer_probabilities
                if probabilities is not None:
                    rank = np.array([rng.choice(axis.size, p=probabilities)])
                else:
                    rv = getattr(axis.hp._vector_dist, "rv", None)
                    v = float(rv.rvs(random_state=rng)) if rv is not None else float(rng.rand())
                    if getattr(axis.hp, "log", False):
                        v = (
                            np.exp(
                                np.log(axis.hp.lower) + v * np.log(axis.hp.upper / axis.hp.lower)
                            )
                            - axis.hp.lower
                        ) / (axis.hp.upper - axis.hp.lower)
                    rank = np.array([min(axis.size - 1, int(v * axis.size))])
                value = axis.values_at(rank)[0]
            else:
                value = _python(axis.hp.sample_value(seed=rng))
            values[axis.name] = value
            condition_values[axis.name] = value
        try:
            configurations.append(dict(Configuration(space, values=values)))
        except ForbiddenValueError:
            continue
    return normalize_configurations(space, configurations)


def _nearest_distances(data: CoverageData, references: CoverageData) -> np.ndarray:
    axes = [i for i, a in enumerate(data.axes) if a.kind != "constant"]
    if not data.count:
        raise ValueError("Coverage radius requires sampled trials.")
    if not axes:
        return np.zeros(references.count)
    # Common numerical spaces use a tree. Mixed/conditional spaces use bounded
    # chunks, avoiding an unbounded references × trials × parameters allocation.
    if all(data.axes[j].kind != "categorical" for j in axes) and (
        np.isfinite(data.points[:, axes]).all() and np.isfinite(references.points[:, axes]).all()
    ):
        distances, _ = cKDTree(data.points[:, axes]).query(references.points[:, axes], k=1)
        return distances / np.sqrt(len(axes))
    result = np.full(references.count, np.inf)
    for start in range(0, references.count, 128):
        ref = references.points[start : start + 128]
        best = np.full(len(ref), np.inf)
        for offset in range(0, data.count, 1024):
            points = data.points[offset : offset + 1024]
            squared = np.zeros((len(ref), len(points)))
            for j in axes:
                left, right = ref[:, j, None], points[None, :, j]
                active_left, active_right = np.isfinite(left), np.isfinite(right)
                difference = (
                    (left != right).astype(float)
                    if data.axes[j].kind == "categorical"
                    else (left - right) ** 2
                )
                squared += np.where(
                    active_left & active_right, difference, active_left != active_right
                )
            best = np.minimum(best, np.min(squared, axis=1))
        result[start : start + len(ref)] = np.sqrt(best / len(axes))
    return result


def compute_coverage_radius(
    data: CoverageData, references: CoverageData, holes: int = 20
) -> Dict[str, Any]:
    """Measure nearest-trial RMS distance and decode the largest sampled holes."""
    distances = _nearest_distances(data, references)
    summary = dict(
        zip(
            ("median", "p90", "p95", "p99", "max"),
            map(float, np.quantile(distances, [0.5, 0.9, 0.95, 0.99, 1])),
        )
    )
    # Repeated reference configurations describe the same region; show it once.
    regions, seen = [], set()
    for index in np.argsort(distances)[::-1] if holes > 0 else []:
        key = json.dumps(references.values[index], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        regions.append({**references.values[index], "distance": float(distances[index])})
        if len(regions) >= holes:
            break
    frequencies, edges = np.histogram(distances, bins=30, range=(0, 1))
    return {
        **summary,
        "reference_samples": references.count,
        "histogram_x": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "histogram_y": frequencies.tolist(),
        "regions": regions,
        "metric": "RMS sampling-space distance; category/activation mismatch contributes 1.",
    }


def _metrics(data: CoverageData, references: CoverageData) -> Dict[str, Optional[float]]:
    discrepancy = compute_global_discrepancy(data, methods=("CD",))
    radius = compute_coverage_radius(data, references, holes=0)
    pairs = compute_pairwise_coverage(data, include_distances=False)["pairs"]
    marginals = compute_univariate_coverage(data)
    return {
        "centered_discrepancy": discrepancy["centered"],
        "p95_radius": radius["p95"],
        "max_radius": radius["max"],
        "pairwise_occupancy": float(np.mean([p["occupancy"] for p in pairs])) if pairs else None,
        "marginal_max_gap": max(
            (m["max_gap"] for m in marginals if m["type"] in ("continuous", "integer")),
            default=None,
        ),
    }


def summarize_baseline(observed: Optional[float], samples: Sequence[float]) -> Dict[str, Any]:
    """Describe a metric against random expectation, with midrank percentiles."""
    if observed is None or not samples:
        return {"observed": observed, "mean": None, "percentile": None}
    values = np.asarray(samples, dtype=float)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "observed": observed,
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, 0.05)),
        "upper": float(np.quantile(values, 0.95)),
        "percentile": float(
            100 * (np.sum(values < observed) + 0.5 * np.sum(values == observed)) / len(values)
        ),
        "standardized_deviation": float((observed - np.mean(values)) / std) if std > 0 else None,
    }


def compute_random_baseline(
    data: CoverageData,
    configspace: ConfigurationSpace,
    designs: int = 500,
    reference_samples: int = DEFAULT_REFERENCES,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare identical-size random designs using shared reference points."""
    _validate_options(designs, reference_samples)
    if not data.count:
        raise ValueError("Random baseline requires sampled trials.")
    references = sample_reference_design(configspace, reference_samples, seed)
    observed = _metrics(data, references)
    mixed_reference = _mixed_reference(references)
    uniformity = _sampling_uniformity(data, mixed_reference)
    trajectory = uniformity["checkpoints"]
    discrepancy_keys = ("centered", "l2_star", "wrap_around", "mixed_mmd")
    observed.update({key: trajectory[-1].get(key) for key in discrepancy_keys[1:]})
    trajectory_samples: List[Dict[str, List[float]]] = [
        {key: [] for key in discrepancy_keys} for _ in trajectory
    ]
    samples: Dict[str, List[float]] = {key: [] for key in observed}
    for i in range(designs):
        random = sample_reference_design(configspace, data.count, seed + i + 1)
        metrics = _metrics(random, references)
        random_trajectory = _sampling_uniformity(random, mixed_reference)["checkpoints"]
        metrics.update({key: random_trajectory[-1].get(key) for key in discrepancy_keys[1:]})
        for checkpoint, row in zip(trajectory_samples, random_trajectory):
            for key in discrepancy_keys:
                if row.get(key) is not None:
                    checkpoint[key].append(row[key])
        for key, value in metrics.items():
            if value is not None:
                samples[key].append(value)
    summaries = {key: summarize_baseline(value, samples[key]) for key, value in observed.items()}
    for key, summary in summaries.items():
        summary["median"] = float(np.median(samples[key])) if samples[key] else None
    return {
        "designs": designs,
        "seed": seed,
        "reference_samples": reference_samples,
        "metrics": summaries,
        "discrepancy_checkpoints": trajectory,
        "discrepancy_bands": [
            {
                "trials": row["trials"],
                **{
                    key: {
                        **summarize_baseline(row.get(key), checkpoint[key]),
                        "median": float(np.median(checkpoint[key])) if checkpoint[key] else None,
                    }
                    for key in discrepancy_keys
                },
            }
            for row, checkpoint in zip(trajectory, trajectory_samples)
        ],
        "discrepancy_parameters": _projection(data)[1],
        **{key: value for key, value in uniformity.items() if key.startswith("mixed_")},
    }


def growth_checkpoints(count: int, checkpoints: int = 20) -> List[int]:
    """Select unique logarithmic history prefixes, always including the final trial."""
    if count < 1:
        return []
    return np.unique(np.rint(np.geomspace(1, count, min(count, checkpoints))).astype(int)).tolist()


def _prefix(data: CoverageData, count: int) -> CoverageData:
    return CoverageData(
        data.axes, data.values[:count], data.points[:count], data.fingerprint, {}, data.constrained
    )


def compute_discrepancy_trajectory(data: CoverageData) -> List[Dict[str, Any]]:
    """Evaluate all discrepancy definitions on the same nested trial prefixes."""
    return [
        {"trials": int(count), **compute_global_discrepancy(_prefix(data, int(count)))}
        for count in growth_checkpoints(data.count)
    ]


def compute_coverage_growth(
    data: CoverageData, references: CoverageData, checkpoints: Optional[Sequence[int]] = None
) -> Dict[str, Any]:
    """Evaluate trial prefixes against one fixed reference design."""
    checkpoints = list(checkpoints) if checkpoints is not None else growth_checkpoints(data.count)
    rows = []
    for count in checkpoints:
        if count < 1 or count > data.count:
            raise ValueError("Growth checkpoints must be within the sampled trial history.")
        prefix = _prefix(data, count)
        rows.append({"trials": count, **_metrics(prefix, references)})
    return {
        "checkpoints": rows,
        "reference_samples": references.count,
        "discrepancy_parameters": _projection(data)[1],
    }


def compute_growth_baseline(
    data: CoverageData,
    configspace: ConfigurationSpace,
    designs: int = 500,
    reference_samples: int = DEFAULT_REFERENCES,
    seed: int = 0,
) -> Dict[str, Any]:
    """Simulate independent full random designs and reuse their nested prefixes."""
    _validate_options(designs, reference_samples)
    if not data.count:
        raise ValueError("Growth baseline requires sampled trials.")
    references = sample_reference_design(configspace, reference_samples, seed)
    observed = compute_coverage_growth(data, references)
    random_rows = []
    checkpoints = [row["trials"] for row in observed["checkpoints"]]
    for i in range(designs):
        random = sample_reference_design(configspace, data.count, seed + i + 1)
        random_rows.append(compute_coverage_growth(random, references, checkpoints)["checkpoints"])
    bands = []
    for i, row in enumerate(observed["checkpoints"]):
        bands.append(
            {
                "trials": row["trials"],
                **{
                    key: summarize_baseline(
                        row[key], [r[i][key] for r in random_rows if r[i][key] is not None]
                    )
                    for key in (
                        "p95_radius",
                        "max_radius",
                        "centered_discrepancy",
                        "pairwise_occupancy",
                        "marginal_max_gap",
                    )
                },
            }
        )
    return {"bands": bands, "designs": designs, "seed": seed, **observed}


def _validate_options(designs: int, reference_samples: int) -> None:
    if isinstance(designs, bool) or int(designs) != designs or not 1 <= designs <= 2000:
        raise ValueError("Reference designs must be an integer between 1 and 2000.")
    if (
        isinstance(reference_samples, bool)
        or int(reference_samples) != reference_samples
        or not 32 <= reference_samples <= 20000
    ):
        raise ValueError("Reference samples must be an integer between 32 and 20000.")


def run_coverage_analysis(
    analysis: str,
    data: CoverageData,
    configspace: ConfigurationSpace,
    options: Dict[str, Any],
    observed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dispatch exactly one explicitly requested worker job."""
    if analysis == "forecast":
        from deepcave.evaluators.coverage_forecast import compute_coverage_forecast

        return compute_coverage_forecast(data, options, observed)
    if analysis == "planning":
        from deepcave.evaluators.coverage_planning import generate_coverage_trial_plan

        return generate_coverage_trial_plan(data, options)
    if analysis in ("combinatorial", "combinatorial_baseline"):
        from deepcave.evaluators.combinatorial import (
            compute_combinatorial_baseline,
            compute_combinatorial_coverage,
        )

        if analysis == "combinatorial":
            return compute_combinatorial_coverage(data)
        return compute_combinatorial_baseline(data, configspace, **options)
    if analysis == "pairwise":
        references = sample_reference_design(
            configspace, options.get("reference_samples", PAIR_REFERENCES), options.get("seed", 0)
        )
        return compute_pairwise_coverage(data, references=references)
    if analysis == "discrepancy":
        return compute_sampling_uniformity(
            data,
            sample_reference_design(
                configspace,
                options.get("reference_samples", DEFAULT_REFERENCES),
                options.get("seed", 0),
            ),
        )
    if analysis == "radius":
        return compute_coverage_radius(
            data,
            sample_reference_design(configspace, options["reference_samples"], options["seed"]),
        )
    if analysis == "baseline":
        return compute_random_baseline(data, configspace, **options)
    if analysis == "growth":
        return compute_coverage_growth(
            data,
            sample_reference_design(configspace, options["reference_samples"], options["seed"]),
        )
    if analysis == "growth_baseline":
        return compute_growth_baseline(data, configspace, **options)
    raise ValueError(f"Unknown coverage analysis: {analysis}")
