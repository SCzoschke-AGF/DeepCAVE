# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core calculations for observed near-optimal (Rashomon) configurations."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import hashlib
import json
import math
from collections import OrderedDict
from threading import RLock

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
)
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from sklearn.metrics import silhouette_score

from deepcave.utils.configspace import normalized_configuration_distance_components

TOP_PERCENT_PRESETS = (1, 2, 5, 10, 20)
DISTANCE_CACHE_SIZE = 2

_DISTANCE_CACHE: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
_DISTANCE_CACHE_LOCK = RLock()


def clear_rashomon_distance_cache() -> None:
    """Clear the bounded in-process cache of condensed distance vectors."""
    with _DISTANCE_CACHE_LOCK:
        _DISTANCE_CACHE.clear()


def select_near_optimal_indices(
    objectives: Sequence[float],
    trial_ids: Sequence[int],
    optimize: str,
    criterion: str,
    value: float,
) -> np.ndarray:
    """Select direction-aware near-optimal rows in deterministic best-first order."""
    objective_values = np.asarray(objectives, dtype=float)
    ids = np.asarray(trial_ids, dtype=int)
    if objective_values.ndim != 1 or ids.shape != objective_values.shape:
        raise ValueError("Objectives and trial IDs must be equally sized one-dimensional arrays.")
    if optimize not in ("lower", "upper"):
        raise ValueError("Optimize must be either 'lower' or 'upper'.")
    if value < 0:
        raise ValueError("The near-optimal tolerance must not be negative.")
    if objective_values.size == 0:
        return np.asarray([], dtype=int)
    if not np.all(np.isfinite(objective_values)):
        raise ValueError("Near-optimal selection requires finite objective values.")

    primary = objective_values if optimize == "lower" else -objective_values
    order = np.lexsort((ids, primary))
    best = float(objective_values[order[0]])

    if criterion == "top_percent":
        if value <= 0 or value > 100:
            raise ValueError("Top percentage must be in the interval (0, 100].")
        count = max(1, int(math.ceil(objective_values.size * value / 100.0)))
        return order[:count]
    if criterion == "absolute":
        tolerance = float(value)
    elif criterion == "relative":
        tolerance = abs(best) * float(value) / 100.0
    else:
        raise ValueError(f"Unknown near-optimal criterion: {criterion}")

    if optimize == "lower":
        selected = np.flatnonzero(objective_values <= best + tolerance)
    else:
        selected = np.flatnonzero(objective_values >= best - tolerance)
    selected_set = set(selected.tolist())
    return np.asarray([index for index in order if index in selected_set], dtype=int)


def _distance_parameter_indices(
    configspace: ConfigurationSpace,
    parameter_names: Optional[Sequence[str]],
) -> np.ndarray:
    names = list(configspace.keys())
    selected = set(parameter_names) if parameter_names is not None else set(names)
    unknown = selected - set(names)
    if unknown:
        raise ValueError(f"Unknown hyperparameters: {sorted(unknown)}")
    return np.asarray(
        [
            index
            for index, (name, hyperparameter) in enumerate(configspace.items())
            if name in selected and not isinstance(hyperparameter, Constant)
        ],
        dtype=int,
    )


def pairwise_configuration_distances(
    matrix: np.ndarray,
    configspace: ConfigurationSpace,
    parameter_names: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Return exact condensed RMS distances for normalized mixed-type configurations."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(configspace):
        raise ValueError("Configuration matrix does not match the ConfigSpace dimensions.")
    n_rows = values.shape[0]
    if n_rows < 2:
        return np.asarray([], dtype=np.float32)

    indices = _distance_parameter_indices(configspace, parameter_names)
    if indices.size == 0:
        return np.zeros(n_rows * (n_rows - 1) // 2, dtype=np.float32)

    names = list(configspace.keys())
    if weights is None:
        weight_values = np.ones(indices.size, dtype=float)
    else:
        weight_values = np.asarray(
            [max(0.0, float(weights.get(names[index], 0.0))) for index in indices],
            dtype=float,
        )
        if not np.all(np.isfinite(weight_values)) or float(weight_values.sum()) <= 0:
            raise ValueError("Importance weights must contain at least one finite positive value.")
    denominator = float(weight_values.sum())

    condensed = np.empty(n_rows * (n_rows - 1) // 2, dtype=np.float32)
    offset = 0
    for row in range(n_rows - 1):
        for start in range(row + 1, n_rows, 2048):
            stop = min(start + 2048, n_rows)
            components = normalized_configuration_distance_components(
                values[row], values[start:stop], configspace
            )[:, indices]
            count = stop - start
            condensed[offset : offset + count] = np.sqrt(
                np.sum(np.square(components) * weight_values[np.newaxis, :], axis=1) / denominator
            ).astype(np.float32)
            offset += count
    return condensed


def cached_pairwise_configuration_distances(
    cache_key: Tuple[Any, ...],
    matrix: np.ndarray,
    configspace: ConfigurationSpace,
    parameter_names: Optional[Sequence[str]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Return a cached condensed vector while retaining at most two large arrays."""
    # Callers can share results only when the actual metric inputs also match.
    signature = hashlib.sha256(np.asarray(matrix, dtype=float).tobytes())
    signature.update(
        json.dumps(
            [configspace.to_serialized_dict(), parameter_names, weights, "rashomon-rms-v1"],
            sort_keys=True,
            default=str,
        ).encode()
    )
    cache_key = (*cache_key, signature.hexdigest())
    with _DISTANCE_CACHE_LOCK:
        cached = _DISTANCE_CACHE.get(cache_key)
        if cached is not None:
            _DISTANCE_CACHE.move_to_end(cache_key)
            return cached
    distances = pairwise_configuration_distances(matrix, configspace, parameter_names, weights)
    with _DISTANCE_CACHE_LOCK:
        _DISTANCE_CACHE[cache_key] = distances
        while len(_DISTANCE_CACHE) > DISTANCE_CACHE_SIZE:
            _DISTANCE_CACHE.popitem(last=False)
    return distances


def condensed_pair_indices(n_items: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Map pairs of square-matrix indices to SciPy condensed-vector indices."""
    left = np.minimum(first, second).astype(np.int64)
    right = np.maximum(first, second).astype(np.int64)
    if np.any(left == right):
        raise ValueError("Condensed vectors do not contain diagonal entries.")
    return n_items * left - left * (left + 1) // 2 + right - left - 1


def subset_condensed_distances(
    condensed: np.ndarray, n_items: int, indices: Sequence[int]
) -> np.ndarray:
    """Extract a condensed distance vector for an arbitrary ordered subset."""
    selected = np.asarray(indices, dtype=int)
    if selected.size < 2:
        return np.asarray([], dtype=np.float32)
    row, column = np.triu_indices(selected.size, 1)
    positions = condensed_pair_indices(n_items, selected[row], selected[column])
    return np.asarray(condensed, dtype=np.float32)[positions]


def subset_distance_matrix(
    condensed: np.ndarray, n_items: int, indices: Sequence[int]
) -> np.ndarray:
    """Construct only the requested square submatrix from a condensed vector."""
    selected = np.asarray(indices, dtype=int)
    result = np.zeros((selected.size, selected.size), dtype=np.float32)
    if selected.size < 2:
        return result
    row, column = np.triu_indices(selected.size, 1)
    positions = condensed_pair_indices(n_items, selected[row], selected[column])
    result[row, column] = np.asarray(condensed, dtype=np.float32)[positions]
    result[column, row] = result[row, column]
    return result


def diversity_summary(distances: Sequence[float]) -> Dict[str, Optional[float]]:
    """Summarize a condensed pairwise-distance vector robustly."""
    values = np.asarray(distances, dtype=float)
    if values.size == 0:
        return {"median": None, "q25": None, "q75": None}
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75)}


def random_subset_baseline(
    condensed: np.ndarray,
    pool_size: int,
    subset_size: int,
    repeats: int = 200,
    max_pairs: int = 100_000,
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    """Estimate the equal-size random-subset median-distance reference distribution."""
    if subset_size < 2 or pool_size < subset_size:
        return {"median": None, "lower": None, "upper": None}
    if subset_size == pool_size:
        median = float(np.median(np.asarray(condensed, dtype=float)))
        return {"median": median, "lower": median, "upper": median}
    rng = np.random.RandomState(seed)
    medians = []
    for _ in range(repeats):
        subset = np.sort(rng.choice(pool_size, size=subset_size, replace=False))
        pair_count = subset_size * (subset_size - 1) // 2
        if pair_count <= max_pairs:
            distances = subset_condensed_distances(condensed, pool_size, subset)
        else:
            first = rng.randint(0, subset_size, size=max_pairs)
            second = rng.randint(0, subset_size - 1, size=max_pairs)
            second += second >= first
            positions = condensed_pair_indices(
                pool_size,
                subset[first],
                subset[second],
            )
            distances = np.asarray(condensed)[positions]
        medians.append(float(np.median(distances)))
    lower, median, upper = np.quantile(medians, [0.025, 0.5, 0.975])
    return {"median": float(median), "lower": float(lower), "upper": float(upper)}


def _normalized_entropy(values: np.ndarray, possible_states: int) -> float:
    if values.size == 0 or possible_states <= 1:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(possible_states)


def parameter_spreads(
    matrix: np.ndarray,
    selected_indices: Sequence[int],
    configspace: ConfigurationSpace,
    parameter_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Summarize per-parameter coverage in the selected and complete matrices."""
    values = np.asarray(matrix, dtype=float)
    selected = values[np.asarray(selected_indices, dtype=int)]
    allowed = set(parameter_names) if parameter_names is not None else set(configspace.keys())
    rows = []
    for column, (name, hyperparameter) in enumerate(configspace.items()):
        if name not in allowed:
            continue
        overall_values = values[:, column]
        elite_values = selected[:, column]
        overall_active = np.isfinite(overall_values)
        elite_active = np.isfinite(elite_values)

        elite_positions = elite_values[elite_active]
        is_categorical = isinstance(hyperparameter, CategoricalHyperparameter)
        if is_categorical:
            denominator = max(1, len(hyperparameter.choices) - 1)
            elite_positions = elite_positions / denominator
        if elite_positions.size:
            elite_min = float(np.min(elite_positions))
            elite_median = float(np.median(elite_positions))
            elite_max = float(np.max(elite_positions))
            if is_categorical:
                elite_coverage = float(
                    len(np.unique(elite_values[elite_active])) / len(hyperparameter.choices)
                )
            else:
                elite_coverage = elite_max - elite_min
        else:
            elite_min = elite_median = elite_max = elite_coverage = None

        if isinstance(hyperparameter, Constant):
            overall_spread: Optional[float] = 0.0
            elite_spread: Optional[float] = 0.0
        elif isinstance(hyperparameter, CategoricalHyperparameter):
            include_inactive = not np.all(overall_active)
            possible = len(hyperparameter.choices) + int(include_inactive)
            overall_codes = np.where(overall_active, overall_values, -1)
            elite_codes = np.where(elite_active, elite_values, -1)
            overall_spread = _normalized_entropy(overall_codes, possible)
            elite_spread = _normalized_entropy(elite_codes, possible)
        else:
            overall_finite = overall_values[overall_active]
            elite_finite = elite_values[elite_active]
            overall_spread = (
                float(np.subtract(*np.quantile(overall_finite, [0.75, 0.25])))
                if overall_finite.size >= 2
                else None
            )
            elite_spread = (
                float(np.subtract(*np.quantile(elite_finite, [0.75, 0.25])))
                if elite_finite.size >= 2
                else None
            )

        if overall_spread in (None, 0.0):
            relative = 0.0 if elite_spread == 0.0 else None
        elif elite_spread is None:
            relative = None
        else:
            relative = float(elite_spread / overall_spread)
        rows.append(
            {
                "parameter": name,
                "near_optimal_spread": elite_spread,
                "overall_spread": overall_spread,
                "relative_spread": relative,
                "near_optimal_min": elite_min,
                "near_optimal_median": elite_median,
                "near_optimal_max": elite_max,
                "near_optimal_coverage": elite_coverage,
                "near_optimal_active": float(np.mean(elite_active)) if elite_active.size else 0.0,
                "overall_active": float(np.mean(overall_active)) if overall_active.size else 0.0,
                "kind": (
                    "categorical"
                    if isinstance(hyperparameter, CategoricalHyperparameter)
                    else (
                        "ordinal"
                        if isinstance(hyperparameter, OrdinalHyperparameter)
                        else "constant" if isinstance(hyperparameter, Constant) else "numeric"
                    )
                ),
            }
        )
    return rows


def cluster_configurations(
    condensed: Sequence[float],
    n_items: int,
    requested_clusters: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Cluster a precomputed distance vector and return stable dendrogram-order labels."""
    distances = np.asarray(condensed, dtype=float)
    if n_items <= 1:
        return {"labels": [1] * n_items, "order": list(range(n_items)), "n_clusters": n_items}
    tree = linkage(distances, method="average")
    order = leaves_list(tree).astype(int)

    if requested_clusters is not None:
        n_clusters = max(1, min(int(requested_clusters), 8, n_items))
    elif n_items < 3:
        n_clusters = 1
    else:
        rng = np.random.RandomState(seed)
        sample = np.sort(rng.choice(n_items, size=min(500, n_items), replace=False))
        sample_matrix = subset_distance_matrix(distances, n_items, sample)
        best_score = -1.0
        n_clusters = 1
        for candidate in range(2, min(8, n_items - 1) + 1):
            candidate_labels = fcluster(tree, candidate, criterion="maxclust")
            sampled_labels = candidate_labels[sample]
            if np.unique(sampled_labels).size < 2:
                continue
            score = float(silhouette_score(sample_matrix, sampled_labels, metric="precomputed"))
            if score > best_score:
                best_score = score
                n_clusters = candidate
        if best_score < 0.25:
            n_clusters = 1

    raw_labels = (
        np.ones(n_items, dtype=int)
        if n_clusters == 1
        else fcluster(tree, n_clusters, criterion="maxclust")
    )
    label_order = []
    for index in order:
        label = int(raw_labels[index])
        if label not in label_order:
            label_order.append(label)
    remap = {label: index + 1 for index, label in enumerate(label_order)}
    labels = [remap[int(label)] for label in raw_labels]
    return {
        "labels": labels,
        "order": order.tolist(),
        "n_clusters": len(label_order),
        "linkage": tree.tolist(),
    }
