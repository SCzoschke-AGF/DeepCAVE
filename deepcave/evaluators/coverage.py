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
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import kstest, qmc

from deepcave.runs.status import Status

VERSION = 2
DEFAULT_BINS = 20
DEFAULT_REFERENCES = 2048


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

    def mass_edges(self, ranks: np.ndarray) -> tuple:
        """Return the CDF interval occupied by each discrete value."""
        size = self.size
        if self.kind == "categorical":
            weights = np.asarray(self.hp.probabilities, dtype=float)
            edges = np.r_[0.0, np.cumsum(weights)]
            return edges[ranks], edges[ranks + 1]
        if self.kind in ("ordinal", "constant"):
            return ranks / size, (ranks + 1) / size
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
        return {
            "name": self.name,
            "type": self.kind,
            "conditional": self.conditional,
            "distribution": distribution,
            "lower": _python(self.hp.lower) if numeric else None,
            "upper": _python(self.hp.upper) if numeric else None,
            "choices": self.choices,
            "probabilities": (
                list(map(float, self.hp.probabilities)) if self.kind == "categorical" else None
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
    return CoverageData(axes, values, points, "", {}, bool(configspace.forbidden_clauses))


def prepare_coverage(run: Any) -> CoverageData:
    """Normalize every sampled trial, including failed and repeated configurations."""
    configs: Dict[int, Dict[str, Any]] = {}
    for trial in run.history:
        if trial.config_id not in configs:
            configs[trial.config_id] = dict(run.get_config(trial.config_id))
    data = normalize_configurations(run.configspace, [configs[t.config_id] for t in run.history])
    definition = run.configspace.to_serialized_dict()
    signature = [
        VERSION,
        definition,
        [
            [t.config_id, t.budget, t.seed, int(t.status), data.values[i]]
            for i, t in enumerate(run.history)
        ],
    ]
    data.fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True, default=_python).encode()
    ).hexdigest()
    statuses = {status.name: sum(t.status == status for t in run.history) for status in Status}
    failures = (Status.TIMEOUT, Status.MEMORYOUT, Status.CRASHED, Status.ABORTED, Status.FAILED)
    data.overview = {
        "trials": data.count,
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


def compute_univariate_coverage(data: CoverageData, bins: int = DEFAULT_BINS) -> List[Dict]:
    """Compute cheap marginal frequencies, gaps and sampling-space ECDFs."""
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
            row.update(
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


def compute_pairwise_coverage(
    data: CoverageData, bins: int = DEFAULT_BINS, include_discrepancy: bool = True
) -> Dict[str, Any]:
    """Rank nominal projection grids by occupancy; compute numerical hole scores."""
    rows = []
    for i, x in enumerate(data.axes):
        if x.kind == "constant":
            continue
        for j in range(i + 1, len(data.axes)):
            y = data.axes[j]
            if y.kind == "constant":
                continue
            active = np.isfinite(data.points[:, [i, j]]).all(axis=1)
            indices = np.flatnonzero(active)
            shape = (min(bins, x.size or bins), min(bins, y.size or bins))
            cells = []
            for axis, column, count in ((x, i, shape[0]), (y, j, shape[1])):
                if axis.size is not None:
                    ranks = axis.ranks([data.values[k][axis.name] for k in indices])
                    cells.append(np.minimum(count - 1, ranks * count // axis.size))
                else:
                    cells.append(
                        np.minimum(count - 1, (data.points[active, column] * count).astype(int))
                    )
            grid = np.zeros(shape, dtype=int)
            np.add.at(grid, tuple(cells), 1)
            occupancy = float(np.count_nonzero(grid) / grid.size)
            continuous = x.size is None and y.size is None
            discrepancy = (
                float(qmc.discrepancy(data.points[active][:, [i, j]], method="CD"))
                if include_discrepancy
                and len(indices)
                and continuous
                and not x.conditional
                and not y.conditional
                and not data.constrained
                else None
            )
            # Padding occupied boundaries measures distance to samples or the grid edge.
            hole = (
                float(distance_transform_edt(np.pad(grid == 0, 1))[1:-1, 1:-1].max() / bins)
                if continuous
                else None
            )
            rows.append(
                {
                    "x": x.name,
                    "y": y.name,
                    "active_trials": len(indices),
                    "occupied_cells": int(np.count_nonzero(grid)),
                    "total_cells": int(grid.size),
                    "occupancy": occupancy,
                    "empty_cell_rate": 1 - occupancy,
                    "hole_score": hole,
                    "discrepancy": discrepancy,
                }
            )
    return {
        "pairs": sorted(rows, key=lambda row: (row["occupancy"], -(row["discrepancy"] or 0))),
        "bins": bins,
        "note": "Nominal projection cells; conditional or forbidden cells may be unreachable. "
        "Discrete axes above the bin limit are grouped. Hole score is an approximate grid radius.",
    }


def sample_reference_design(configspace: ConfigurationSpace, count: int, seed: int) -> CoverageData:
    """Sample a reproducible legal design, including metadata-defined lattices."""
    if count < 1:
        raise ValueError("At least one reference sample is required.")
    space = deepcopy(configspace)
    axes = [CoverageAxis(hp) for hp in space.values()]
    custom_steps = any(a.step is not None and (a.kind == "continuous" or a.step != 1) for a in axes)
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
                rv = getattr(axis.hp._vector_dist, "rv", None)
                v = float(rv.rvs(random_state=rng)) if rv is not None else float(rng.rand())
                if getattr(axis.hp, "log", False):
                    v = (
                        np.exp(np.log(axis.hp.lower) + v * np.log(axis.hp.upper / axis.hp.lower))
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
    pairs = compute_pairwise_coverage(data, include_discrepancy=False)["pairs"]
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
    samples: Dict[str, List[float]] = {key: [] for key in observed}
    for i in range(designs):
        random = sample_reference_design(configspace, data.count, seed + i + 1)
        metrics = _metrics(random, references)
        for key, value in metrics.items():
            if value is not None:
                samples[key].append(value)
    return {
        "designs": designs,
        "seed": seed,
        "reference_samples": reference_samples,
        "metrics": {
            key: summarize_baseline(value, samples[key]) for key, value in observed.items()
        },
        "discrepancy_parameters": _projection(data)[1],
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
    analysis: str, data: CoverageData, configspace: ConfigurationSpace, options: Dict[str, int]
) -> Dict[str, Any]:
    """Dispatch exactly one explicitly requested worker job."""
    if analysis == "pairwise":
        return compute_pairwise_coverage(data)
    if analysis == "discrepancy":
        return compute_global_discrepancy(data)
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
