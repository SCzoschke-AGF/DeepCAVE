"""Objective-aware geometry and objective-free radial proposals for numerical spaces."""

from typing import Any, Dict, List

import math

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import Constant

from deepcave.evaluators.coverage import CoverageAxis
from deepcave.evaluators.performance_coverage import prepare_performance_trials
from deepcave.evaluators.rashomon import select_near_optimal_indices
from deepcave.utils.configspace import (
    configuration_euclidean_distances,
    normalized_configuration_distance_components,
    normalized_configuration_matrix,
)

BIN_COUNT = 5
MIN_BIN_CANDIDATES = 1000
GEOMETRY_VERSION = 1


def incumbent_distances(
    configspace: Any, reference: Dict, configurations: List[Dict]
) -> np.ndarray:
    """Use Lighthouse's normalized ConfigSpace components and unweighted RMS distance."""
    matrix = normalized_configuration_matrix([reference, *configurations], configspace)
    components = normalized_configuration_distance_components(matrix[0], matrix[1:], configspace)
    dimensions = sum(not isinstance(hp, Constant) for hp in configspace.values())
    return configuration_euclidean_distances(components, configspace) / math.sqrt(
        max(1, dimensions)
    )


def distance_bins(distances: np.ndarray, radius: float) -> np.ndarray:
    """Assign bins 1..5; internal boundaries belong right, and bin 5 is unbounded."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Elite radius must be finite and nonnegative.")
    return (
        np.searchsorted(np.arange(1, BIN_COUNT) * radius / BIN_COUNT, distances, side="right") + 1
    )


def historical_geometry(run: Any, objective_id: int) -> Dict[str, Any]:
    """Center five equal-width shells on the incumbent, using the best ceil(10% * n)."""
    objective = run.get_objective(objective_id)
    if objective is None or objective.optimize not in ("lower", "upper"):
        raise ValueError("Select an available objective with a known optimization direction.")
    trials = prepare_performance_trials(run, objective_id)
    elite = select_near_optimal_indices(
        trials["objectives"], trials["trial_ids"], objective.optimize, "top_percent", 10
    )
    if not len(elite):
        raise ValueError(
            "Geometry requires successful historical trials with finite objective values."
        )
    configs = [
        dict(run.get_config(run.history[trials["history_indices"][i]].config_id)) for i in elite
    ]
    distances = incumbent_distances(run.configspace, configs[0], configs)
    radius = float(distances.max())
    return {
        "version": GEOMETRY_VERSION,
        "objective_id": objective_id,
        "optimize": objective.optimize,
        "incumbent": configs[0],
        "incumbent_trial": trials["trial_ids"][int(elite[0])],
        "elite_trials": [trials["trial_ids"][int(i)] for i in elite],
        "elite_distances": distances.tolist(),
        "radius": radius,
        "edges": (np.arange(BIN_COUNT + 1) * radius / BIN_COUNT).tolist(),
        "bin_count": BIN_COUNT,
        "minimum_per_bin": MIN_BIN_CANDIDATES,
    }


def radial_configurations(
    configspace: Any,
    reference: Dict,
    lower: float,
    upper: float,
    rng: np.random.RandomState,
    count: int,
) -> List[Dict]:
    """Draw shell-volume radii along feasible rays, then decode declared numerical grids.

    Bounds restrict each ray before radius sampling; proposals are never clipped.
    Quantization can move a proposal between bins, so callers must check its actual distance.
    """
    axes = [CoverageAxis(hp) for hp in configspace.values()]
    varying = [j for j, axis in enumerate(axes) if axis.kind != "constant"]
    dimensions = len(varying)
    anchor = normalized_configuration_matrix([reference], configspace)[0, varying]
    directions = rng.normal(size=(count, dimensions))
    norms = np.linalg.norm(directions, axis=1)
    directions /= np.where(norms > 0, norms, 1)[:, None]
    # At an exact box face, sample the feasible hemisphere directly.
    directions[:, anchor == 0] = np.abs(directions[:, anchor == 0])
    directions[:, anchor == 1] = -np.abs(directions[:, anchor == 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        ray_limits = np.where(directions > 0, (1 - anchor) / directions, -anchor / directions)
    ray_limits[directions == 0] = np.inf
    maxima = np.minimum(upper, np.min(ray_limits, axis=1) / np.sqrt(dimensions))
    usable = (maxima > lower) & (norms > 0)
    directions, maxima = directions[usable], maxima[usable]
    # This ratio form avoids underflow from small radii in high dimensions.
    u = rng.uniform(size=len(maxima))
    radii = maxima * (u + (1 - u) * (lower / maxima) ** dimensions) ** (1 / dimensions)
    vectors = anchor + directions * radii[:, None] * np.sqrt(dimensions)
    result = []
    for vector in vectors:
        if np.any((vector < 0) | (vector > 1)):
            continue
        config = {axis.name: axis.hp.value for axis in axes if axis.kind == "constant"}
        for axis_index, coordinate in zip(varying, vector):
            axis = axes[axis_index]
            if axis.step is not None:
                # Decode continuously before snapping, including log integer parameters.
                hp = axis.hp
                value = (
                    np.exp(np.log(hp.lower) + coordinate * np.log(hp.upper / hp.lower))
                    if hp.log
                    else hp.lower + coordinate * (hp.upper - hp.lower)
                )
                rank = int(np.rint((value - hp.lower) / axis.step))
                if not 0 <= rank < axis.size:
                    break
                value = axis.values_at(np.array([rank]))[0]
            else:
                value = float(axis.hp.to_value(coordinate))
            config[axis.name] = value
        else:
            try:
                Configuration(configspace, values=config).check_valid_configuration()
            except ValueError:
                continue
            result.append(config)
    return result
