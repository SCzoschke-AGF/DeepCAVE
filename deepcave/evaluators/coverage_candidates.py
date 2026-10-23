"""Objective-independent CHR candidates, scored by the validated good/bad KDE.

CHR describes sampled empty balls, not covering-array tuples. A candidate is
eligible for its nearest hole center only if it is inside that empty ball AND
closer to the center than to any observed/virtually selected configuration.
This Voronoi criterion introduces no coverage tolerance or performance weight.
"""

from typing import Any, Dict, List, Optional

import json
from dataclasses import dataclass
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

CANDIDATE_VERSION = 2


def candidate_options(inputs: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Validate pool/planning sizes independently of KDE model settings."""
    inputs = inputs or {}
    result = {}
    for name, default, low, high in (
        ("pool_size", 2000, 1, 10000),
        ("holes", 15, 1, 100),
        ("plan_size", 8, 1, 100),
        ("reference_samples", 2048, 32, 16384),
        ("seed", 42, 0, 2**32 - 2),
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
            CANDIDATE_VERSION,
            performance_source_key(run, objective_id),
            getattr(run, "meta", {}).get("optuna_space_error"),
        ]
    )


def _key(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True)


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
                "selected": False,
                "plan_order": None,
            }
        )
    return selected


def score_coverage_candidates(
    run: Any, objective_id: int, directory: str, inputs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Detect → generate → score → sequentially select against virtual geometry."""
    options = candidate_options(inputs)
    numeric_space(run)
    source_key = scoring_source_key(run, objective_id)
    cache_key = _digest([source_key, options, CANDIDATE_VERSION])
    result_path = Path(directory) / "results" / f"{cache_key}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    data = prepare_coverage(run)
    references = sample_reference_design(
        run.configspace, options["reference_samples"], options["seed"]
    )
    initial_holes = detect_holes(data, references, options["holes"])
    candidates = generate_candidates(run, data, initial_holes, options)
    if not candidates:
        raise ValueError("No unseen valid candidates remain in the sampled coverage holes.")
    fitted = fit_candidate_kde(run, objective_id, directory)
    reference_score = fitted.metadata["reference_score"]

    def score(rows: List[Dict]) -> None:
        scores = fitted.score([row["configuration"] for row in rows])
        for row, value in zip(rows, scores):
            row.update(
                kde_score=float(value),
                reference_score=reference_score,
                relative_score=float(value - reference_score),
                reference_trial_number=fitted.metadata["reference_trial_number"],
                reference_objective=fitted.metadata["reference_objective"],
            )

    score(candidates)
    known_holes = {h["id"]: h for h in initial_holes}
    virtual = data
    plan, steps = [], []
    excluded = {_key(dict(run.get_config(t.config_id))) for t in run.history}
    for order in range(1, options["plan_size"] + 1):
        holes = detect_holes(virtual, references, options["holes"])
        if not holes:
            break
        additions = []
        candidate_keys = {_key(c["configuration"]) for c in candidates}
        for hole in holes:
            if hole["id"] not in known_holes:
                original = normalize_configurations(run.configspace, [hole["center"]])
                known_holes[hole["id"]] = {
                    **hole,
                    "radius": float(_nearest_distances(data, original)[0]),
                }
                center = hole["center"]
                if _key(center) not in candidate_keys and _key(center) not in excluded:
                    h = known_holes[hole["id"]]
                    additions.append(
                        {
                            "id": f"C-{_digest(center)[:16]}",
                            "configuration": center,
                            "hole_id": h["id"],
                            "hole_radius": h["radius"],
                            "center_distance": 0.0,
                            "radius_reduction": h["radius"],
                            "selected": False,
                            "plan_order": None,
                        }
                    )
                    candidate_keys.add(_key(center))
        if additions:
            score(additions)
            candidates.extend(additions)
        remaining = [c for c in candidates if not c["selected"]]
        proposed = normalize_configurations(
            run.configspace, [c["configuration"] for c in remaining]
        )
        assigned, distances = _membership(virtual, proposed, holes)
        # A new largest hole can require a new candidate after virtual updates.
        # Add its center (same fitted KDE), rather than skipping it for a lower hole.
        if not np.any(assigned == 0):
            center = holes[0]["center"]
            existing = next((c for c in remaining if c["configuration"] == center), None)
            if existing is None and _key(center) not in excluded:
                h = known_holes[holes[0]["id"]]
                existing = {
                    "id": f"C-{_digest(center)[:16]}",
                    "configuration": center,
                    "hole_id": h["id"],
                    "hole_radius": h["radius"],
                    "center_distance": 0.0,
                    "radius_reduction": h["radius"],
                    "selected": False,
                    "plan_order": None,
                }
                score([existing])
                candidates.append(existing)
                remaining.append(existing)
                assigned = np.append(assigned, 0)
                distances = np.append(distances, 0.0)
        eligible = np.flatnonzero(assigned == 0)
        if not len(eligible):
            break
        chosen_index = max(eligible, key=lambda i: remaining[i]["kde_score"])
        chosen = remaining[chosen_index]
        hole = holes[0]
        chosen.update(
            selected=True,
            plan_order=order,
            plan_hole_id=hole["id"],
            selection_radius_before=hole["radius"],
            selection_radius_after=float(distances[chosen_index]),
        )
        before = float(np.max(_nearest_distances(virtual, references)))
        virtual = normalize_configurations(
            run.configspace, virtual.values + [chosen["configuration"]]
        )
        after = float(np.max(_nearest_distances(virtual, references)))
        steps.append(
            {
                "candidate_id": chosen["id"],
                "hole_id": hole["id"],
                "radius_before": hole["radius"],
                "radius_after": chosen["selection_radius_after"],
                "max_radius_before": before,
                "max_radius_after": after,
            }
        )
        plan.append(chosen["id"])
        excluded.add(_key(chosen["configuration"]))
    all_scores = np.sort([c["kde_score"] for c in candidates])
    for candidate in candidates:
        candidate["percentile"] = float(
            100
            * np.searchsorted(all_scores, candidate["kde_score"], side="right")
            / len(all_scores)
        )
    result = {
        "candidate_version": CANDIDATE_VERSION,
        "cache_key": cache_key,
        "source_key": source_key,
        "options": options,
        "holes": list(known_holes.values()),
        "candidates": candidates,
        "plan": plan,
        "steps": steps,
        "model": fitted.metadata,
        "objective_id": objective_id,
        "reference_score": reference_score,
        "reference_trial_number": fitted.metadata["reference_trial_number"],
        "reference_objective": fitted.metadata["reference_objective"],
        "top_relative_score": max(c["relative_score"] for c in candidates),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(result, allow_nan=False))
    temporary.replace(result_path)
    return result
