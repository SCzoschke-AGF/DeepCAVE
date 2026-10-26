"""CHR candidates and hybrid full-space recommendations scored by one good/bad KDE.

CHR describes sampled empty balls, not covering-array tuples. A candidate is
eligible for its nearest hole center only if it is inside that empty ball AND
closer to the center than to any observed configuration.
This Voronoi criterion introduces no coverage tolerance or performance weight.
"""

from typing import Any, Callable, Dict, List, Optional

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from scipy.spatial.distance import cdist

from deepcave.evaluators.candidate_geometry import (
    BIN_COUNT,
    GEOMETRY_VERSION,
    MIN_BIN_CANDIDATES,
    distance_bins,
    historical_geometry,
    incumbent_distances,
    radial_configurations,
)
from deepcave.evaluators.candidate_selection import (
    SELECTION_VERSION,
)
from deepcave.evaluators.candidate_selection import configuration_key as _key
from deepcave.evaluators.candidate_selection import (
    rank_candidates,
    select_hybrid_candidates,
)
from deepcave.evaluators.coverage import (
    CoverageAxis,
    CoverageData,
    _nearest_distances,
    normalize_configurations,
    prepare_coverage,
    sample_reference_design,
)
from deepcave.evaluators.performance_coverage import _digest, performance_source_key
from deepcave.runs.status import Status

CANDIDATE_VERSION = 5
KDE_VERSION = 4  # Recommendation rules do not invalidate historical KDE fits.
SAMPLING_VERSION = 1
BIN_SAMPLING_BUDGET = 100000
FINITE_ENUMERATION_LIMIT = 50000


def candidate_options(inputs: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Validate candidate pool sizes independently of KDE model settings."""
    inputs = inputs or {}
    result = {}
    for name, default, low, high in (
        ("pool_size", 2000, 1, 10000),
        ("holes", 15, 1, 100),
        ("reference_samples", 2048, 32, 16384),
        ("seed", 42, 0, 2**32 - 2),
        ("global_pool_size", 10000, 1, 100000),
        ("global_seed", 42, 0, 2**32 - 2),
        ("number_of_candidates", 20, 1, 100000),
    ):
        value = inputs.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
        if not np.isfinite(value) or int(value) != value or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
        result[name] = int(value)
    return result


def numeric_space(run: Any) -> Dict[str, Any]:
    """Adapt a fixed numerical ConfigSpace to the benchmark's Optuna encoding."""
    try:
        from optuna.distributions import FloatDistribution, IntDistribution
        from statsmodels.nonparametric.kernel_density import (  # noqa: F401
            KDEMultivariate,
        )
    except ImportError as error:
        raise ValueError(
            "KDE scoring requires Optuna and statsmodels. Install deepcave[optuna]."
        ) from error
    configspace = run.configspace
    source_error = getattr(run, "meta", {}).get("optuna_space_error")
    if source_error:
        raise ValueError(source_error)
    if configspace.conditions:
        raise ValueError("KDE scoring does not support conditional search spaces.")
    space = {}
    for name, hp in configspace.items():
        if isinstance(hp, Constant):
            continue
        if not isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            raise ValueError(
                f"KDE scoring supports fixed uniform/log float and integer distributions only; "
                f"{name} uses {type(hp).__name__}. Categorical and ordinal spaces are unsupported."
            )
        if (hp.meta or {}).get("probabilities") is not None:
            raise ValueError(
                f"KDE scoring does not support explicit sampling probabilities: {name}."
            )
        step = CoverageAxis(hp).step
        distribution: Any
        if isinstance(hp, UniformIntegerHyperparameter):
            assert step is not None
            distribution = IntDistribution(int(hp.lower), int(hp.upper), log=hp.log, step=int(step))
        else:
            distribution = FloatDistribution(hp.lower, hp.upper, log=hp.log, step=step)
        space[name] = distribution
    if not space:
        raise ValueError("KDE scoring requires at least one nonconstant numerical parameter.")
    return space


def scoring_source_key(run: Any, objective_id: int) -> str:
    """Include source-distribution audits in the study/objective fingerprint."""
    return _digest(
        [
            KDE_VERSION,
            performance_source_key(run, objective_id),
            getattr(run, "meta", {}).get("optuna_space_error"),
        ]
    )


def _encode(configs: List[Dict[str, Any]], space: Dict[str, Any]) -> np.ndarray:
    from deepcave.evaluators.kde import kde_encoding

    return kde_encoding([SimpleNamespace(params=c) for c in configs], space)[0]


def _validate_config(run: Any, config: Dict[str, Any], space: Dict[str, Any]) -> None:
    Configuration(run.configspace, values=config).check_valid_configuration()
    for name, distribution in space.items():
        if not distribution._contains(distribution.to_internal_repr(config[name])):
            raise ValueError(f"Configuration violates the declared distribution of {name}.")


@dataclass
class CandidateKDE:
    """One historical fit and its in-sample best-trial anchor."""

    space: Dict[str, Any]
    good: Any
    bad: Any
    metadata: Dict[str, Any]

    def score(self, configurations: List[Dict[str, Any]]) -> np.ndarray:
        """Score a batch without fitting or learning from generated candidates."""
        from deepcave.evaluators.kde import kde_logpdf

        queries = _encode(configurations, self.space)
        scores = kde_logpdf(self.good, queries) - kde_logpdf(self.bad, queries)
        if not np.isfinite(scores).all():
            raise FloatingPointError("KDE produced nonfinite candidate scores.")
        return scores


def fit_candidate_kde(run: Any, objective_id: int, directory: str) -> CandidateKDE:
    """Fit once per historical snapshot; persist numerical arrays without pickle."""
    space = numeric_space(run)
    from statsmodels.nonparametric.kernel_density import KDEMultivariate

    from deepcave.evaluators.kde import fit_kde, kde_encoding, split_good_bad

    objective = run.get_objective(objective_id)
    if objective is None or objective.optimize not in ("lower", "upper"):
        raise ValueError("Select an available objective with a known optimization direction.")
    records = []
    for index, trial in enumerate(run.history):
        if trial.status != Status.SUCCESS or objective_id >= len(trial.costs):
            continue
        value = trial.costs[objective_id]
        if value is None or not np.isfinite(value):
            continue
        config = dict(run.get_config(trial.config_id))
        _validate_config(run, config, space)
        number = int(getattr(trial, "additional", {}).get("optuna_trial_number", index))
        records.append((number, index, float(value), config))
    records.sort(key=lambda record: (record[0], record[1]))
    values = np.asarray([r[2] for r in records])
    good, bad = split_good_bad(values, np.arange(len(records)), objective.optimize == "upper")
    if min(len(good), len(bad)) <= len(space):
        raise ValueError(
            f"Insufficient completed finite trials: good/bad groups ({len(good)}/{len(bad)}) "
            f"must each contain more than {len(space)} observations. Good = best 20%."
        )
    source_key = scoring_source_key(run, objective_id)
    folder = Path(directory) / "models"
    path = folder / f"{source_key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            models = [
                KDEMultivariate(
                    saved[f"{group}_data"], metadata["var_type"], bw=saved[f"{group}_bw"]
                )
                for group in ("good", "bad")
            ]
        return CandidateKDE(space, models[0], models[1], metadata)
    encoded, var_type, transforms = kde_encoding(
        [SimpleNamespace(params=r[3]) for r in records], space
    )
    models, diagnostics = [], []
    for label, members in (("good", good), ("bad", bad)):
        model, info = fit_kde(encoded[members], var_type, space)
        if not info["success"]:
            raise ValueError(f"KDE {label} bandwidth optimization failed: {info['message']}")
        models.append(model)
        diagnostics.append(info)
    metadata = {
        "source_key": source_key,
        "var_type": var_type,
        "transforms": transforms,
        "diagnostics": diagnostics,
        "training_trials": [r[0] for r in records],
        "good_trials": [records[i][0] for i in good],
        "bad_trials": [records[i][0] for i in bad],
        "reference_trial_number": records[good[0]][0],
        "reference_objective": records[good[0]][2],
        "reference_configuration": records[good[0]][3],
        "objective": objective.name,
        "objective_id": objective_id,
    }
    fitted = CandidateKDE(space, models[0], models[1], metadata)
    metadata["reference_score"] = float(fitted.score([metadata["reference_configuration"]])[0])
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{uuid4().hex}.npz"
    np.savez_compressed(
        temporary,
        metadata=json.dumps(metadata, allow_nan=False),
        good_data=models[0].data,
        bad_data=models[1].data,
        good_bw=models[0].bw,
        bad_bw=models[1].bw,
    )
    temporary.replace(path)
    return fitted


def detect_holes(data: CoverageData, references: CoverageData, count: int) -> List[Dict[str, Any]]:
    """Decode the largest unique CHR reference centers without consulting objectives."""
    radii = _nearest_distances(data, references)
    holes, seen = [], set()
    for index in np.argsort(-radii, kind="stable"):
        config = references.values[index]
        key = _key(config)
        if key in seen or radii[index] <= 0:
            continue
        seen.add(key)
        holes.append(
            {
                "id": f"H-{_digest(config)[:12]}",
                "center": config,
                "radius": float(radii[index]),
                "reference_index": int(index),
            }
        )
        if len(holes) == count:
            break
    return holes


def _membership(data: CoverageData, candidates: CoverageData, holes: List[Dict]) -> tuple:
    """Assign to nearest center's Voronoi cell, clipped to its open CHR ball."""
    if not holes or not candidates.count:
        return np.full(candidates.count, -1), np.zeros(candidates.count)
    if data.configspace is None:
        raise ValueError("Candidate membership requires the declared configuration space.")
    centers = normalize_configurations(data.configspace, [h["center"] for h in holes])
    axes = [i for i, axis in enumerate(data.axes) if axis.kind != "constant"]
    distances = cdist(candidates.points[:, axes], centers.points[:, axes]) / np.sqrt(len(axes))
    assigned = np.argmin(distances, axis=1)
    center_distance = distances[np.arange(candidates.count), assigned]
    clearance = _nearest_distances(data, candidates)
    radii = np.asarray([h["radius"] for h in holes])
    valid = (center_distance < radii[assigned]) & (center_distance < clearance)
    return np.where(valid, assigned, -1), center_distance


def generate_candidates(
    run: Any, data: CoverageData, holes: List[Dict], options: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Generate a geometry-only pool, preferring larger holes and valid lattices."""
    space = numeric_space(run)
    seen = {_key(dict(run.get_config(t.config_id))) for t in run.history}
    configs = []
    # Centers guarantee alternatives even in small or constrained finite spaces.
    for config in [h["center"] for h in holes] + sample_reference_design(
        run.configspace, max(options["pool_size"] * 8, 256), options["seed"] + 1
    ).values:
        key = _key(config)
        if key not in seen:
            _validate_config(run, config, space)
            seen.add(key)
            configs.append(config)
    if not configs:
        return []
    proposed = normalize_configurations(run.configspace, configs)
    membership, distances = _membership(data, proposed, holes)
    groups = [np.flatnonzero(membership == i).tolist() for i in range(len(holes))]
    # Weighted round robin allocates more alternatives to larger empty balls.
    selected: List[Dict] = []
    taken = np.zeros(len(holes), dtype=int)
    while len(selected) < options["pool_size"]:
        available = [i for i, group in enumerate(groups) if taken[i] < len(group)]
        if not available:
            break
        group = max(available, key=lambda i: holes[i]["radius"] / (taken[i] + 1))
        index = groups[group][taken[group]]
        taken[group] += 1
        hole = holes[group]
        selected.append(
            {
                "id": f"C-{_digest(configs[index])[:16]}",
                "configuration": configs[index],
                "hole_id": hole["id"],
                "hole_radius": hole["radius"],
                "center_distance": float(distances[index]),
                "radius_reduction": hole["radius"] - float(distances[index]),
            }
        )
    return selected


def generate_full_space_candidates(run: Any, options: Dict[str, int]) -> tuple:
    """Sample unseen legal configurations independently of objectives and CHR geometry."""
    space = numeric_space(run)
    seen = {_key(dict(run.get_config(t.config_id))) for t in run.history}
    sizes = [CoverageAxis(hp).size for hp in run.configspace.values()]
    cardinality = math.prod(sizes) if all(size is not None for size in sizes) else None
    requested = options["global_pool_size"]
    target = min(requested, max(0, cardinality - len(seen))) if cardinality else requested
    rng = np.random.RandomState(options["global_seed"])
    candidates: List[Dict[str, Any]] = []
    draws, budget = 0, max(1024, requested * 10)
    while len(candidates) < target and draws < budget:
        # Fixed-size batches preserve the candidate prefix when the requested pool grows.
        count = min(1024, budget - draws)
        sampled = sample_reference_design(
            run.configspace, count, int(rng.randint(0, 2**32 - 1))
        ).values
        draws += len(sampled)
        for config in sampled:
            key = _key(config)
            if key in seen:
                continue
            _validate_config(run, config, space)
            seen.add(key)
            candidates.append(
                {
                    "id": f"G-{_digest(config)[:16]}",
                    "configuration": config,
                }
            )
            if len(candidates) == target:
                break
    reason = (
        "target_reached"
        if len(candidates) == requested
        else (
            "finite_space_exhausted"
            if cardinality is not None and len(seen) == cardinality
            else "sampling_budget_reached"
        )
    )
    return candidates, {
        "requested": requested,
        "count": len(candidates),
        "seed": options["global_seed"],
        "draws": draws,
        "stop_reason": reason,
    }


def augment_full_space_candidates(
    run: Any, initial: List[Dict], geometry: Dict, seed: int
) -> Dict[str, Any]:
    """Supply deficient shells using only geometry, with bounded, deterministic sampling."""
    space = numeric_space(run)
    pool = [dict(c, origin="global") for c in initial]
    seen = {_key(dict(run.get_config(t.config_id))) for t in run.history}
    seen.update(_key(c["configuration"]) for c in pool)
    reference, radius = geometry["incumbent"], geometry["radius"]
    counts = np.zeros(BIN_COUNT, dtype=int)

    def annotate(rows: List[Dict]) -> None:
        distances = incumbent_distances(
            run.configspace, reference, [c["configuration"] for c in rows]
        )
        bins = distance_bins(distances, radius)
        for row, distance, bin_id in zip(rows, distances, bins):
            row.update(distance=float(distance), bin=int(bin_id))
            counts[bin_id - 1] += 1

    annotate(pool)
    initial_count = len(pool)
    reasons, attempts = {}, np.zeros(BIN_COUNT, dtype=int)

    def accept(configs: List[Dict], target_bin: Optional[int] = None) -> None:
        new = []
        for config in configs:
            key = _key(config)
            if key in seen:
                continue
            try:
                _validate_config(run, config, space)
            except ValueError:
                continue
            seen.add(key)
            new.append(
                {"id": f"G-{_digest(config)[:16]}", "configuration": config, "origin": "targeted"}
            )
        distances = incumbent_distances(
            run.configspace, reference, [c["configuration"] for c in new]
        )
        for row, distance, bin_id in zip(new, distances, distance_bins(distances, radius)):
            if (target_bin is None or bin_id == target_bin) and counts[
                bin_id - 1
            ] < MIN_BIN_CANDIDATES:
                row.update(distance=float(distance), bin=int(bin_id))
                counts[bin_id - 1] += 1
                pool.append(row)
            else:
                # A proposal rejected for this bin can still be proposed for its actual bin.
                seen.remove(_key(row["configuration"]))

    axes = [CoverageAxis(hp) for hp in run.configspace.values()]
    sizes = [axis.size for axis in axes]
    cardinality = math.prod(sizes) if all(size is not None for size in sizes) else None
    if radius == 0:
        reasons = {
            b: "Degenerate historical elite radius R=0; bins 1–4 are empty."
            for b in range(1, BIN_COUNT + 1)
        }
    elif cardinality is not None and cardinality <= FINITE_ENUMERATION_LIMIT:
        # Exhaust small grids exactly, including forbidden clauses, instead of retrying duplicates.
        supports = [axis.values_at(np.arange(axis.size)) for axis in axes]
        batch = []
        for values in product(*supports):
            batch.append(dict(zip(run.configspace.keys(), values)))
            if len(batch) == 1024:
                accept(batch)
                batch = []
        accept(batch)
        reasons = {
            b: "Finite legal search space exhausted for this bin." for b in range(1, BIN_COUNT + 1)
        }
    else:
        for bin_id in range(1, BIN_COUNT + 1):
            if counts[bin_id - 1] >= MIN_BIN_CANDIDATES:
                continue
            # Independent streams prevent deficits in one bin from changing another's proposals.
            rng = np.random.RandomState(
                np.random.SeedSequence([seed, SAMPLING_VERSION, bin_id]).generate_state(1)[0]
            )
            lower = (bin_id - 1) * radius / BIN_COUNT
            upper = bin_id * radius / BIN_COUNT if bin_id < BIN_COUNT else 1.0
            while (
                counts[bin_id - 1] < MIN_BIN_CANDIDATES
                and attempts[bin_id - 1] < BIN_SAMPLING_BUDGET
            ):
                count = min(1024, BIN_SAMPLING_BUDGET - attempts[bin_id - 1])
                configs = radial_configurations(
                    run.configspace, reference, lower, upper, rng, count
                )
                attempts[bin_id - 1] += count
                accept(configs, bin_id)
            reasons[bin_id] = (
                "Geometric sampling budget reached; "
                "constraints or discrete support may limit this bin."
            )
    bins = [
        {
            "bin": b,
            "count": int(counts[b - 1]),
            "attempts": int(attempts[b - 1]),
            "reason": reasons.get(b) if counts[b - 1] < MIN_BIN_CANDIDATES else None,
        }
        for b in range(1, BIN_COUNT + 1)
    ]
    return {
        "candidates": pool,
        "supply": {
            "ready": bool(np.all(counts >= MIN_BIN_CANDIDATES)),
            "initial_count": initial_count,
            "added_count": len(pool) - initial_count,
            "final_count": len(pool),
            "minimum_per_bin": MIN_BIN_CANDIDATES,
            "bins": bins,
        },
    }


def _cached_json(path: Path, compute: Callable[[], Any]) -> Any:
    """Persist one deterministic stage atomically, independently of later selection settings."""
    if path.exists():
        return json.loads(path.read_text())
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(result, allow_nan=False))
    temporary.replace(path)
    return result


def score_coverage_candidates(
    run: Any, objective_id: int, directory: str, inputs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Cache historical KDE scoring separately from the hybrid recommendation budget."""
    options = candidate_options(inputs)
    numeric_space(run)
    source_key = scoring_source_key(run, objective_id)
    cache_key = _digest(
        [
            source_key,
            options,
            CANDIDATE_VERSION,
            GEOMETRY_VERSION,
            SAMPLING_VERSION,
            SELECTION_VERSION,
            BIN_COUNT,
            MIN_BIN_CANDIDATES,
            BIN_SAMPLING_BUDGET,
            FINITE_ENUMERATION_LIMIT,
        ]
    )
    result_path = Path(directory) / "results" / f"{cache_key}.json"
    return _cached_json(
        result_path,
        lambda: _recommendations(run, objective_id, directory, options, source_key, cache_key),
    )


def _recommendations(
    run: Any, objective_id: int, directory: str, options: Dict, source_key: str, cache_key: str
) -> Dict:
    pool_options = {k: v for k, v in options.items() if k != "number_of_candidates"}
    pool_key = _digest(
        [
            source_key,
            pool_options,
            CANDIDATE_VERSION,
            GEOMETRY_VERSION,
            SAMPLING_VERSION,
            BIN_COUNT,
            MIN_BIN_CANDIDATES,
            BIN_SAMPLING_BUDGET,
            FINITE_ENUMERATION_LIMIT,
        ]
    )
    base = _cached_json(
        Path(directory) / "scored" / f"{pool_key}.json",
        lambda: _scored_pools(run, objective_id, directory, options, source_key),
    )
    recommendations, error = [], None
    if not base["supply"]["ready"]:
        error = "Hybrid selection unavailable. " + "; ".join(
            f"Bin {b['bin']}: {b['count']}/{MIN_BIN_CANDIDATES} — {b['reason']}"
            for b in base["supply"]["bins"]
            if b["reason"]
        )
    else:
        try:
            recommendations = select_hybrid_candidates(
                base["full_space_candidates"], options["number_of_candidates"]
            )
        except ValueError as exception:
            error = f"Hybrid selection unavailable. {exception}"
    return {
        **base,
        "cache_key": cache_key,
        "options": options,
        "recommendations": recommendations,
        "hybrid": {
            "available": error is None,
            "error": error,
            "version": SELECTION_VERSION,
            "global_count": sum(c["selection"] == "Global KDE" for c in recommendations),
            "exploration_count": sum(c["selection"] == "Exploration" for c in recommendations),
        },
    }


def _scored_pools(
    run: Any, objective_id: int, directory: str, options: Dict, source_key: str
) -> Dict:
    """Preserve CHR behavior and score every source with the same historical KDE pair."""
    data = prepare_coverage(run)
    references = sample_reference_design(
        run.configspace, options["reference_samples"], options["seed"]
    )
    holes = detect_holes(data, references, options["holes"])
    candidates = generate_candidates(run, data, holes, options)
    fitted = fit_candidate_kde(run, objective_id, directory)
    reference_score = fitted.metadata["reference_score"]

    def score(rows: List[Dict]) -> None:
        for start in range(0, len(rows), 1024):
            batch = rows[start : start + 1024]
            scores = fitted.score([row["configuration"] for row in batch])
            for row, value in zip(batch, scores):
                row.update(
                    kde_score=float(value),
                    reference_score=reference_score,
                    relative_score=float(value - reference_score),
                    reference_trial_number=fitted.metadata["reference_trial_number"],
                    reference_objective=fitted.metadata["reference_objective"],
                )

    score(candidates)
    global_key = _digest(
        [
            SAMPLING_VERSION,
            run.id,
            run.configspace.to_serialized_dict(),
            sorted({_key(dict(run.get_config(t.config_id))) for t in run.history}),
            options["global_pool_size"],
            options["global_seed"],
        ]
    )
    initial, global_sampling = _cached_json(
        Path(directory) / "global" / f"{global_key}.json",
        lambda: generate_full_space_candidates(run, options),
    )
    geometry = historical_geometry(run, objective_id)
    geometry_key = _digest([source_key, geometry])
    augmented_key = _digest(
        [global_key, geometry_key, SAMPLING_VERSION, BIN_SAMPLING_BUDGET, FINITE_ENUMERATION_LIMIT]
    )
    augmented = _cached_json(
        Path(directory) / "augmented" / f"{augmented_key}.json",
        lambda: augment_full_space_candidates(run, initial, geometry, options["global_seed"]),
    )
    full_space_candidates = augmented["candidates"]
    score(full_space_candidates)
    ranks = {c["id"]: c["kde_rank"] for c in rank_candidates(full_space_candidates)}
    for row in full_space_candidates:
        row["kde_rank"] = ranks[row["id"]]
    for pool in (candidates, full_space_candidates):
        all_scores = np.sort([c["kde_score"] for c in pool])
        for candidate in pool:
            candidate["percentile"] = float(
                100
                * np.searchsorted(all_scores, candidate["kde_score"], side="right")
                / len(all_scores)
            )
    result = {
        "candidate_version": CANDIDATE_VERSION,
        "source_key": source_key,
        "stage_keys": {
            "global": global_key,
            "geometry": geometry_key,
            "augmented": augmented_key,
            "kde": source_key,
        },
        "holes": holes,
        "candidates": candidates,
        "full_space_candidates": full_space_candidates,
        "global_sampling": global_sampling,
        "geometry": geometry,
        "supply": augmented["supply"],
        "model": fitted.metadata,
        "objective_id": objective_id,
        "reference_score": reference_score,
        "reference_trial_number": fitted.metadata["reference_trial_number"],
        "reference_objective": fitted.metadata["reference_objective"],
        "top_relative_score": max((c["relative_score"] for c in candidates), default=None),
    }
    return result
