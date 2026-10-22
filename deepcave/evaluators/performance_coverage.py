"""Exact empirical distance/performance diagnostics over observed successful trials.

The caller supplies the existing Rashomon distances. Pair statistics and leave-one-out
nearest-neighbor statistics are deliberately separate. No pair sampling is performed.
"""

from typing import Any, Dict, List, Tuple

import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration

from deepcave.runs.status import Status
from deepcave.utils.configspace import normalized_configuration_matrix

PERFORMANCE_VERSION = 1
SCATTER_LIMIT = 12000


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def performance_source_key(run: Any, objective_id: int) -> str:
    """Fingerprint the study, configurations, eligibility and selected objective values."""
    configs = {}
    rows = []
    for index, trial in enumerate(run.history):
        if trial.config_id not in configs:
            try:
                configs[trial.config_id] = dict(run.get_config(trial.config_id))
            except (KeyError, ValueError, TypeError):
                configs[trial.config_id] = None
        costs = getattr(trial, "costs", [])
        value = costs[objective_id] if objective_id < len(costs) else None
        rows.append(
            [
                index,
                str(trial.status),
                trial.config_id,
                getattr(trial, "additional", {}),
                getattr(trial, "budget", None),
                getattr(trial, "seed", None),
                value,
            ]
        )
    objective = run.get_objective(objective_id)
    return _digest(
        [
            PERFORMANCE_VERSION,
            run.id,
            objective_id,
            vars(objective) if objective is not None else None,
            run.configspace.to_serialized_dict(),
            configs,
            rows,
        ]
    )


def prepare_performance_trials(run: Any, objective_id: int) -> Dict[str, Any]:
    """Keep every successful finite compatible trial, including duplicate configurations."""
    records = []
    encoded = {}
    for index, trial in enumerate(run.history):
        costs = getattr(trial, "costs", [])
        if trial.status != Status.SUCCESS or objective_id >= len(costs):
            continue
        value = costs[objective_id]
        if value is None or not np.isfinite(value):
            continue
        if trial.config_id not in encoded:
            try:
                config = Configuration(
                    run.configspace, values=dict(run.get_config(trial.config_id))
                )
                config.check_valid_configuration()
                encoded[trial.config_id] = normalized_configuration_matrix(
                    [config], run.configspace
                )[0]
            except (ValueError, TypeError, KeyError):
                encoded[trial.config_id] = None
        vector = encoded[trial.config_id]
        if vector is None:
            continue
        trial_id = int(getattr(trial, "additional", {}).get("optuna_trial_number", index))
        records.append((trial_id, index, float(value), vector))
    records.sort(key=lambda record: (record[0], record[1]))
    return {
        "trial_ids": [row[0] for row in records],
        "history_indices": [row[1] for row in records],
        "objectives": np.asarray([row[2] for row in records], dtype=float),
        "matrix": np.asarray([row[3] for row in records], dtype=float).reshape(
            len(records), len(run.configspace)
        ),
        "excluded": len(run.history) - len(records),
    }


def quantile_groups(distances: np.ndarray, count: int) -> Tuple[np.ndarray, List[int]]:
    """Assign observed-distance quantiles without splitting equal distances."""
    values = np.asarray(distances)
    if values.size == 0:
        return np.asarray([], dtype=np.uint8), []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, count + 1)))
    groups = np.searchsorted(edges[1:-1], values, side="right").astype(np.uint8)
    return groups, np.unique(groups).tolist()


def _summary(distances: np.ndarray, sorted_differences: np.ndarray) -> Dict[str, Any]:
    median, p90, p95 = np.quantile(sorted_differences, [0.5, 0.9, 0.95])
    return {
        "lower": float(np.min(distances)),
        "upper": float(np.max(distances)),
        "distance_median": float(np.median(distances)),
        "count": int(len(sorted_differences)),
        "mean": float(np.mean(sorted_differences)),
        "median": float(median),
        "p90": float(p90),
        "p95": float(p95),
    }


def _density_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Keep constant nonnegative observations visible without negative histogram ranges."""
    lower, upper = float(np.min(values)), float(np.max(values))
    if lower == upper:
        upper = lower + max(1e-6, abs(lower) * 0.05)
    return np.linspace(lower, upper, bins + 1)


def analyze_performance_distances(
    distances: np.ndarray,
    objectives: np.ndarray,
    trial_ids: List[int],
    differences_path: Path,
) -> Dict[str, Any]:
    """Analyze a condensed vector with linear-size row blocks and no square matrices.

    Sorted differences per distance bin are persisted in one memory-mappable array.
    This supports exact threshold changes with a binary search per bin. Nearest-neighbor
    ties select the earliest row (the caller orders rows by trial ID, then history index).
    """
    values = np.asarray(objectives, dtype=float)
    n = len(values)
    pairs = n * (n - 1) // 2
    if n < 2:
        raise ValueError("At least two successful trials with a finite objective are required.")
    if len(trial_ids) != n or np.shape(distances) != (pairs,):
        raise ValueError("Trial metadata and condensed distances must have matching sizes.")
    if not np.isfinite(values).all() or not np.isfinite(distances).all():
        raise ValueError("Objectives and configuration distances must be finite.")
    differences = np.empty(pairs, dtype=float)
    nearest = np.full(n, -1, dtype=int)
    radii = np.full(n, np.inf)
    offset = 0
    for row in range(n - 1):
        size = n - row - 1
        block = distances[offset : offset + size]
        differences[offset : offset + size] = np.abs(values[row] - values[row + 1 :])
        candidate = int(np.argmin(block))
        if block[candidate] < radii[row]:
            radii[row] = block[candidate]
            nearest[row] = row + 1 + candidate
        better = block < radii[row + 1 :]
        radii[row + 1 :][better] = block[better]
        nearest[row + 1 :][better] = row
        offset += size

    groups, labels = quantile_groups(distances, 10)
    differences_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_values = np.lib.format.open_memmap(
        differences_path, mode="w+", dtype=float, shape=(pairs,)
    )
    bins = []
    offset = 0
    for label in labels:
        selected = groups == label
        ordered = np.sort(differences[selected])
        stats = _summary(distances[selected], ordered)
        stats["offset"] = offset
        sorted_values[offset : offset + len(ordered)] = ordered
        bins.append(stats)
        offset += len(ordered)
    sorted_values.flush()
    del sorted_values

    nn_differences = np.abs(values - values[nearest])
    nn_groups, nn_labels = quantile_groups(radii, 3)
    nn_bins = []
    for label in nn_labels:
        selected = nn_groups == label
        ordered = np.sort(nn_differences[selected])
        nn_bins.append({**_summary(radii[selected], ordered), "differences": ordered.tolist()})

    if pairs <= SCATTER_LIMIT:
        plot = {
            "kind": "scatter",
            "distances": distances.tolist(),
            "differences": differences.tolist(),
            "bins": groups.tolist(),
            "labels": labels,
        }
    else:
        x_edges = _density_edges(distances, 48)
        y_edges = _density_edges(differences, 40)
        counts = np.zeros((48, 40), dtype=np.int64)
        # NumPy's histogram2d allocates index arrays for every input pair. Bound these
        # temporaries while still counting every pair, including zero differences.
        for start in range(0, pairs, 262144):
            stop = min(start + 262144, pairs)
            block_counts, _, _ = np.histogram2d(
                distances[start:stop], differences[start:stop], bins=(x_edges, y_edges)
            )
            counts += block_counts.astype(np.int64)
        plot = {
            "kind": "density",
            "counts": counts.astype(int).T.tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
        }
    return {
        "trials": n,
        "pairs": pairs,
        "bins": bins,
        "nearest_bins": nn_bins,
        "nearest": {
            "trial_ids": trial_ids,
            "neighbor_ids": [trial_ids[index] for index in nearest],
            "neighbor_indices": nearest.tolist(),
            "distances": radii.tolist(),
            "differences": nn_differences.tolist(),
        },
        "plot": plot,
    }


def threshold_statistics(
    result: Dict[str, Any], threshold: float, differences_path: Path
) -> Dict[str, Any]:
    """Update exact empirical exceedance fractions without recalculating distances."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("Relevant difference must be a finite number of at least zero.")
    values = np.load(differences_path, mmap_mode="r", allow_pickle=False)
    bins = []
    for stats in result["bins"]:
        start, count = stats["offset"], stats["count"]
        ordered = values[start : start + count]
        fraction = (count - int(np.searchsorted(ordered, threshold, side="right"))) / count
        bins.append({**stats, "relevant_fraction": fraction})
    nn_bins = []
    for stats in result["nearest_bins"]:
        ordered = stats["differences"]
        count = stats["count"]
        fraction = (count - int(np.searchsorted(ordered, threshold, side="right"))) / count
        nn_bins.append({**stats, "relevant_fraction": fraction})
    return {**result, "bins": bins, "nearest_bins": nn_bins, "threshold": float(threshold)}
