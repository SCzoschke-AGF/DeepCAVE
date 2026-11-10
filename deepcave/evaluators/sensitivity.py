"""Seed replication statistics, independent of Dash and Plotly.

Original Optuna parameters define configuration identity. User config IDs are audit labels,
not identity keys. Without a complete experiment signature, comparisons stay study-local.
Repeated evaluations of a seed pair contribute one pair mean to descriptive seed statistics;
their raw evaluations are retained for diagnostics and balanced variance decomposition.
"""

from typing import Any, Dict, List, Optional, Sequence

import hashlib
import json
import math
from collections import Counter

import numpy as np
from scipy.stats import rankdata

from deepcave.utils.candidates import CANDIDATE_COLORS, candidate_data

CONTEXT_FIELDS = (
    "experiment_id",
    "model",
    "model_type",
    "model_family",
    "dataset",
    "dataset_id",
    "data_hash",
    "dataset_hash",
    "feature_set",
    "features",
    "feature_set_id",
    "cv_strategy",
    "cv",
    "n_splits",
    "objective_definition",
    "objective",
    "metric",
)
PARTITION_FIELDS = ("cv_split_ids", "split_ids", "partition_id", "data_split_id")
MODES = {"model": "Model Seed", "data": "Data Seed", "combined": "Combined"}


def canonical(value: Any) -> str:
    """Encode types and exact float values, without display rounding or bool/int collisions."""

    def typed(item: Any) -> Any:
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, dict):
            return ["dict", [[key, typed(val)] for key, val in sorted(item.items())]]
        if isinstance(item, (list, tuple)):
            return ["list", [typed(val) for val in item]]
        if isinstance(item, float):
            return ["float", item.hex()]
        return [type(item).__name__, item]

    return json.dumps(typed(value), ensure_ascii=False, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def resolve_attribute(mapping: Optional[str], study: Dict, trial: Dict) -> Any:
    """Resolve the same scoped attributes offered by General's metadata controls."""
    if not mapping:
        return None
    source, name = mapping.split(":", 1) if ":" in mapping else ("trial", mapping)
    value = (study if source == "study" else trial).get(name)
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    return None


def available_modes(mappings: Dict[str, Optional[str]]) -> List[str]:
    """Offer only modes supported by centrally configured seed fields."""
    modes = [key for key in ("model", "data") if mappings.get(key)]
    return modes + (["combined"] if len(modes) == 2 else [])


def prepare_study(run: Any) -> Dict[str, Any]:
    """Reuse candidate identities and join the existing full trial attributes by trial number."""
    data = candidate_data(run)
    attrs = {
        trial.additional.get("optuna_trial_number"): trial.additional.get("user_attrs", {})
        for trial in run.history
    }
    for row in data["rows"]:
        row["user_attrs"] = dict(attrs.get(row["trial_number"], row["user_attrs"]))
        row["study_attrs"] = run.get_meta().get("optuna_user_attrs", {})
    return data


def experiment_context(row: Dict, objective: str) -> Dict[str, Any]:
    """Retain every known context discriminator; permit cross-study pooling only with evidence.

    An experiment ID alone is not a definition of model, data, features or CV. Complete
    signatures can span studies; absent signatures are conservatively scoped by source.
    Conflicting study/trial context declarations are retained rather than overwritten.
    """
    study, trial = row.get("study_attrs", {}), row.get("user_attrs", {})
    attrs = {**study, **trial}
    known = {name: attrs[name] for name in CONTEXT_FIELDS if attrs.get(name) is not None}
    conflicts = {
        name: study[name]
        for name in CONTEXT_FIELDS
        if name in study and name in trial and canonical(study[name]) != canonical(trial[name])
    }
    complete = all(
        any(attrs.get(name) is not None for name in aliases)
        for aliases in (
            ("model", "model_type", "model_family"),
            ("dataset", "dataset_id", "dataset_hash", "data_hash"),
            ("feature_set", "features", "feature_set_id"),
            ("cv_strategy", "cv"),
            ("objective_definition", "objective", "metric"),
        )
    )
    info = row["metric_info"][objective]
    return {
        "metadata": known,
        "study_conflicts": conflicts,
        "scope": None if complete else [row["storage"], row["study_name"]],
        "objective": [objective, info["optimize"]],
    }


def _parameters(row: Dict, mappings: Dict, seeds: Dict) -> Dict:
    omitted = set()
    aliases = {
        "model": ("model_seed", "random_state", "random_seed", "seed"),
        "data": ("data_seed", "split_seed", "data_random_state"),
    }
    for kind, mapping in mappings.items():
        if not mapping:
            continue
        name = mapping.split(":", 1)[-1]
        omitted.add(name)
        for alias in aliases[kind]:
            if alias in row["parameters"] and canonical(row["parameters"][alias]) == canonical(
                seeds[kind]
            ):
                omitted.add(alias)
    return {name: value for name, value in row["parameters"].items() if name not in omitted}


def build_groups(data: Dict, mappings: Dict, mode: str, objective: str) -> Dict[str, Any]:
    """Resolve metadata, establish context, group configurations, then summarize seed pairs."""
    groups: Dict[str, Dict] = {}
    excluded: Counter[str] = Counter()
    if mode not in available_modes(mappings):
        return {"groups": [], "excluded": {}, "mode": mode, "objective": objective}
    seen = set()
    for source in data.get("rows", []):
        if source["id"] in seen:
            continue
        seen.add(source["id"])
        if source["state"] != "COMPLETE":
            excluded["unfinished trials"] += 1
            continue
        value = source["metrics"].get(objective)
        if not _finite(value):
            excluded["missing or non-finite objectives"] += 1
            continue
        study, attrs = source.get("study_attrs", {}), source.get("user_attrs", {})
        seeds = {
            kind: resolve_attribute(mappings.get(kind), study, attrs) for kind in ("model", "data")
        }
        if any(mappings.get(kind) and seeds[kind] is None for kind in seeds):
            excluded["missing seed values"] += 1
            continue
        params = _parameters(source, mappings, seeds)
        context = experiment_context(source, objective)
        config_key = _digest([params, context])
        other = "data" if mode == "model" else "model"
        condition = {other: seeds[other]} if mode != "combined" and mappings.get(other) else {}
        key = _digest([config_key, condition])
        if key not in groups:
            label = "C-" + config_key[:10]
            if condition:
                label += f" · {other}={seeds[other]}"
            groups[key] = {
                "id": key,
                "config_id": config_key,
                "label": label,
                "parameters": params,
                "context": context,
                "condition": condition,
                "mode": mode,
                "rows": [],
                "direction": source["metric_info"][objective]["optimize"],
            }
        partitions = {
            name: attrs.get(name, study.get(name))
            for name in PARTITION_FIELDS
            if attrs.get(name, study.get(name)) is not None
        }
        groups[key]["rows"].append(
            {
                **source,
                "objective": float(value),
                "model_seed": seeds["model"],
                "data_seed": seeds["data"],
                "seed_key": canonical(seeds),
                "partition": canonical(partitions) if partitions else None,
            }
        )
    config_ids = sorted({group["config_id"] for group in groups.values()})
    colors = {key: CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)] for i, key in enumerate(config_ids)}
    for group in groups.values():
        group["color"] = colors[group["config_id"]]
        pairs: Dict[str, List[Dict]] = {}
        for row in group["rows"]:
            pairs.setdefault(row["seed_key"], []).append(row)
        group["replicates"] = []
        for key, rows in sorted(pairs.items()):
            pair_partitions = {row["partition"] for row in rows}
            group["replicates"].append(
                {
                    "key": key,
                    "model_seed": rows[0]["model_seed"],
                    "data_seed": rows[0]["data_seed"],
                    "objective": float(np.mean([row["objective"] for row in rows])),
                    "n": len(rows),
                    "partition": (
                        next(iter(pair_partitions)) if len(pair_partitions) == 1 else None
                    ),
                }
            )
        values = np.array([rep["objective"] for rep in group["replicates"]])
        best = values.max() if group["direction"] == "upper" else values.min()
        group.update(
            {
                "evaluations": len(group["rows"]),
                "seeds": len(values),
                "duplicates": len(group["rows"]) - len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "best_gap": float(abs(best - values.mean())),
            }
        )
    return {
        "groups": list(groups.values()),
        "excluded": dict(excluded),
        "mode": mode,
        "objective": objective,
    }


def randomness_ranges(group: Dict, varying_axis: str) -> List[Dict]:
    """Measure observed ranges of pair means while holding the other seed fixed."""
    fixed_axis = "model_seed" if varying_axis == "data_seed" else "data_seed"
    levels: Dict[str, Dict] = {}
    for pair in group["replicates"]:
        if pair[fixed_axis] is None or pair[varying_axis] is None:
            continue
        level = levels.setdefault(
            canonical(pair[fixed_axis]), {"seed": pair[fixed_axis], "values": {}}
        )
        level["values"][canonical(pair[varying_axis])] = pair["objective"]
    ranges = []
    for level in levels.values():
        if len(level["values"]) < 2:
            continue
        minimum, maximum = min(level["values"].values()), max(level["values"].values())
        ranges.append(
            {
                "fixed_seed": level["seed"],
                "minimum": minimum,
                "maximum": maximum,
                "range": maximum - minimum,
            }
        )
    return ranges


def summary(groups: Sequence[Dict], minimum: int = 2) -> Dict:
    """Count original configurations separately from conditional rows."""
    return {
        "configurations": len({group["config_id"] for group in groups}),
        "evaluations": sum(group["evaluations"] for group in groups),
        "median_seeds": float(np.median([g["seeds"] for g in groups])) if groups else 0,
        "comparable": len({g["config_id"] for g in groups if g["seeds"] >= max(2, minimum)}),
        "model_seeds": len(
            {
                canonical(r["model_seed"])
                for g in groups
                for r in g["replicates"]
                if r["model_seed"] is not None
            }
        ),
        "data_seeds": len(
            {
                canonical(r["data_seed"])
                for g in groups
                for r in g["replicates"]
                if r["data_seed"] is not None
            }
        ),
    }


def pareto_ids(groups: Sequence[Dict], maximize: bool) -> List[str]:
    """Find nondominated means and sample SDs, including ties on the frontier."""
    valid = [g for g in groups if g["sd"] is not None and g["seeds"] >= 2]
    result = []
    for group in valid:
        score = -group["mean"] if maximize else group["mean"]
        dominated = any(
            other["sd"] <= group["sd"]
            and (-other["mean"] if maximize else other["mean"]) <= score
            and (other["sd"] < group["sd"] or other["mean"] != group["mean"])
            for other in valid
        )
        if not dominated:
            result.append(group["id"])
    return result


def seed_design(group: Dict) -> Dict:
    """Describe coverage without inventing separable effects for a coupled design."""
    reps = group["replicates"]
    models = {canonical(r["model_seed"]) for r in reps if r["model_seed"] is not None}
    data = {canonical(r["data_seed"]) for r in reps if r["data_seed"] is not None}
    pairs = {(canonical(r["model_seed"]), canonical(r["data_seed"])) for r in reps}
    crossed = (
        len(models) > 1
        and len(data) > 1
        and any(sum(m == model for m, _ in pairs) > 1 for model in models)
        and any(sum(d == datum for _, d in pairs) > 1 for datum in data)
    )
    return {
        "model_levels": len(models),
        "data_levels": len(data),
        "observed": len(pairs),
        "possible": max(1, len(models)) * max(1, len(data)),
        "crossed": crossed,
        "complete": bool(models and data and len(pairs) == len(models) * len(data)),
    }


def _pairable(groups: Sequence[Dict]) -> bool:
    """Require matching experiment, seed keys and explicit partition evidence for pairing."""
    first = groups[0]
    reference = {r["key"]: r["partition"] for r in first["replicates"]}
    return (
        all(reference.values())
        and all(canonical(g["context"]) == canonical(first["context"]) for g in groups)
        and all({r["key"]: r["partition"] for r in g["replicates"]} == reference for g in groups)
    )


def _weights(group: Dict, draws: int, rng: Any) -> np.ndarray:
    reps = sorted(group["replicates"], key=lambda r: r["key"])
    if group["mode"] == "combined" and seed_design(group)["crossed"]:
        # Pigeonhole bootstrap: resample model and data levels independently. Missing
        # cells stay missing; shared levels are never treated as independent pairs.
        models = sorted({canonical(r["model_seed"]) for r in reps})
        data = sorted({canonical(r["data_seed"]) for r in reps})
        model_counts = rng.multinomial(
            len(models), np.full(len(models), 1 / len(models)), size=draws
        )
        data_counts = rng.multinomial(len(data), np.full(len(data), 1 / len(data)), size=draws)
        return np.array(
            [
                model_counts[:, models.index(canonical(r["model_seed"]))]
                * data_counts[:, data.index(canonical(r["data_seed"]))]
                for r in reps
            ]
        ).T
    return rng.multinomial(len(reps), np.full(len(reps), 1 / len(reps)), size=draws)


def effective_replicates(group: Dict) -> int:
    """Count independent factor levels conservatively for crossed bootstrap eligibility."""
    design = seed_design(group)
    if group["mode"] == "combined" and design["crossed"]:
        return min(design["model_levels"], design["data_levels"])
    return group["seeds"]


def bootstrap_means(groups: Sequence[Dict], draws: int = 2000, seed: int = 17) -> Dict:
    """Resample distinct seed units, sharing draws only for verified paired observations."""
    if not groups or any(effective_replicates(group) < 4 for group in groups):
        return {
            "reason": (
                "At least four distinct seeds (four levels of each crossed seed factor) "
                "per configuration are required for uncertainty estimates."
            )
        }
    if (
        len({canonical(g["context"]) for g in groups}) > 1
        or len({canonical(g["condition"]) for g in groups}) > 1
    ):
        return {
            "reason": (
                "Select configurations from the same experiment context and fixed-seed condition."
            )
        }
    rng = np.random.default_rng(seed)
    paired = _pairable(groups)
    shared = _weights(groups[0], draws * 2, rng) if paired else None
    columns = []
    for group in groups:
        reps = sorted(group["replicates"], key=lambda r: r["key"])
        weights = shared if shared is not None else _weights(group, draws * 2, rng)
        values = np.array([r["objective"] for r in reps])
        denominators = weights.sum(axis=1)
        columns.append(
            np.divide(
                weights @ values,
                denominators,
                out=np.full(draws * 2, np.nan),
                where=denominators > 0,
            )
        )
    means = np.column_stack(columns)
    means = means[np.isfinite(means).all(axis=1)][:draws]
    if len(means) < draws:
        return {"reason": "Seed coverage is too sparse for the requested resampling."}
    crossed = any(g["mode"] == "combined" and seed_design(g)["crossed"] for g in groups)
    method = ("Paired" if paired else "Unpaired") + (
        " crossed-factor bootstrap" if crossed else " seed-replication bootstrap"
    )
    return {"means": means, "paired": paired, "method": method, "draws": draws}


def pairwise(first: Dict, second: Dict, draws: int = 2000) -> Dict:
    """Compare seed means; use paired differences only with verified partition identities."""
    if canonical(first["context"]) != canonical(second["context"]) or canonical(
        first["condition"]
    ) != canonical(second["condition"]):
        return {
            "reason": "Configurations have different experiment contexts or fixed-seed conditions."
        }
    paired = _pairable([first, second])
    a = np.array([r["objective"] for r in sorted(first["replicates"], key=lambda r: r["key"])])
    b = np.array([r["objective"] for r in sorted(second["replicates"], key=lambda r: r["key"])])
    count: Any
    if paired:
        differences = a - b
        unit = "matched seed pairs"
        count = len(a)
    else:
        # Count all cross-pairs without allocating an n × m array. These comparisons
        # describe observed superiority; they are not independent statistical units.
        ordered = np.sort(b)
        wins = np.searchsorted(ordered, a, side="left")
        losses = len(b) - np.searchsorted(ordered, a, side="right")
        frequency = float(
            np.sum(wins if first["direction"] == "upper" else losses) / (len(a) * len(b))
        )
        differences = None
        count = [len(a), len(b)]
        unit = "unpaired seed replications (A, B)"
    if paired:
        frequency = float(
            np.mean(differences > 0 if first["direction"] == "upper" else differences < 0)
        )
    bootstrap = bootstrap_means([first, second], draws)
    interval = None
    if "means" in bootstrap:
        interval = np.quantile(
            bootstrap["means"][:, 0] - bootstrap["means"][:, 1], [0.025, 0.975]
        ).tolist()
    return {
        "paired": paired,
        "difference": first["mean"] - second["mean"],
        "interval": interval,
        "count": count,
        "unit": unit,
        "win_frequency": frequency,
        "differences": differences.tolist() if paired else [],
        "method": bootstrap.get("method", "Paired" if paired else "Unpaired"),
        "reason": bootstrap.get("reason"),
    }


def ranking(groups: Sequence[Dict], draws: int = 2000) -> Dict:
    """Estimate rank uncertainty; ties share rank and top-one frequency."""
    if len(groups) < 2:
        return {"reason": "Select at least two configurations for ranking stability."}
    result = bootstrap_means(groups, draws)
    if "means" not in result:
        return result
    means = result.pop("means")
    scores = -means if groups[0]["direction"] == "upper" else means
    ranks = rankdata(scores, axis=1, method="average")
    best = scores == scores.min(axis=1, keepdims=True)
    top_one = best / best.sum(axis=1, keepdims=True)
    low_rank = rankdata(scores, axis=1, method="min")
    high_rank = rankdata(scores, axis=1, method="max")
    top_three = np.clip((3 - low_rank + 1) / (high_rank - low_rank + 1), 0, 1)
    result["rows"] = [
        {
            "label": group["label"],
            "color": group["color"],
            "mean_rank": float(ranks[:, i].mean()),
            "top1": float(top_one[:, i].mean()),
            "top3": float(top_three[:, i].mean()),
            "rank_low": float(np.quantile(ranks[:, i], 0.025)),
            "rank_high": float(np.quantile(ranks[:, i], 0.975)),
            "mean_low": float(np.quantile(means[:, i], 0.025)),
            "mean_high": float(np.quantile(means[:, i], 0.975)),
            "ranks": ranks[:, i].tolist(),
        }
        for i, group in enumerate(groups)
    ]
    return result


def fold_attributes(attrs: Dict) -> Dict[str, Dict[str, Optional[float]]]:
    """Discover existing fold arrays/maps, including objective-keyed nested results."""
    result = {}

    def collect(name: str, value: Any) -> None:
        if (
            isinstance(value, list)
            and value
            and all(v is None or isinstance(v, (int, float)) for v in value)
        ):
            result[name] = {
                str(i + 1): float(v) if _finite(v) else None for i, v in enumerate(value)
            }
        elif isinstance(value, dict) and value:
            if all(v is None or isinstance(v, (int, float)) for v in value.values()):
                result[name] = {str(k): float(v) if _finite(v) else None for k, v in value.items()}
            else:
                for key, item in value.items():
                    collect(name + " / " + str(key), item)

    for name, value in attrs.items():
        if "fold" in name.lower():
            collect(name, value)
    return result


def fold_diagnostics(group: Dict, attribute: str) -> Dict:
    """Keep raw trial/fold cells; aggregate pair means only within comparable partitions."""
    rows = []
    for row in group["rows"]:
        folds = fold_attributes(row["user_attrs"]).get(attribute, {})
        if folds:
            rows.append({"trial": row, "folds": folds})
    labels = sorted({key for row in rows for key in row["folds"]})
    partitions = {row["trial"]["partition"] for row in rows}
    data_seeds = {canonical(row["trial"]["data_seed"]) for row in rows}
    comparable = (
        bool(rows)
        and len({tuple(sorted(r["folds"])) for r in rows}) == 1
        and (
            len(partitions) == 1
            and None not in partitions
            or len(data_seeds) == 1
            and all(r["trial"]["data_seed"] is not None for r in rows)
            and len(partitions) == 1
        )
    )
    statistics = []
    if comparable:
        for label in labels:
            pairs: Dict[str, List[float]] = {}
            for row in rows:
                value = row["folds"].get(label)
                if value is not None:
                    pairs.setdefault(row["trial"]["seed_key"], []).append(value)
            values = [np.mean(v) for v in pairs.values()]
            if values:
                statistics.append(
                    {
                        "fold": label,
                        "mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "minimum": float(min(values)),
                        "maximum": float(max(values)),
                        "seeds": len(values),
                    }
                )
    return {"rows": rows, "labels": labels, "comparable": comparable, "summary": statistics}


def elite_subset(groups: Sequence[Dict], proportion: float) -> List[Dict]:
    """Use Lighthouse's objective-quantile elite convention, including threshold ties."""
    if not groups:
        return []
    maximize = groups[0]["direction"] == "upper"
    threshold = np.quantile([g["mean"] for g in groups], 1 - proportion if maximize else proportion)
    return (
        [g for g in groups if g["mean"] >= threshold]
        if maximize
        else [g for g in groups if g["mean"] <= threshold]
    )


def variance_decomposition(groups: Sequence[Dict]) -> Dict:
    """Balanced crossed random-effects ANOVA with all interactions and a repeat residual.

    Report method-of-moments variance estimates only for a fully crossed, balanced design
    with repeated cells. Unequal or unreplicated designs cannot identify this full model.
    Negative estimates are reported explicitly and truncated to zero for shares only.
    """
    reason = (
        "Variance components require at least two configurations in one context, "
        "identical fully crossed seed levels, and at least two evaluations in every "
        "cell with balanced counts."
    )
    if (
        len(groups) < 2
        or len({canonical(g["context"]) for g in groups}) != 1
        or len({canonical(g["condition"]) for g in groups}) != 1
    ):
        return {"reason": reason}
    mode = groups[0]["mode"]
    kinds = ["model", "data"] if mode == "combined" else [mode]
    levels = [sorted({canonical(r[kind + "_seed"]) for r in groups[0]["rows"]}) for kind in kinds]
    if any(len(level) < 2 for level in levels):
        return {"reason": reason}
    shape = (len(groups), *[len(level) for level in levels])
    cells: Dict[tuple, List[float]] = {}
    for i, group in enumerate(groups):
        for row in group["rows"]:
            keys = [canonical(row[kind + "_seed"]) for kind in kinds]
            if any(key not in level for key, level in zip(keys, levels)):
                return {"reason": reason}
            index = (i, *[level.index(key) for key, level in zip(keys, levels)])
            cells.setdefault(index, []).append(row["objective"])
    counts = {len(values) for values in cells.values()}
    if len(cells) != math.prod(shape) or len(counts) != 1 or min(counts) < 2:
        return {"reason": reason}
    repeats = next(iter(counts))
    values = np.empty((*shape, repeats))
    for key, cell in cells.items():
        values[key] = cell
    means = values.mean(axis=-1)
    residual = float(np.sum((values - means[..., None]) ** 2) / (math.prod(shape) * (repeats - 1)))
    dimensions = len(shape)
    effects: Dict[tuple, np.ndarray] = {(): np.full([1] * dimensions, means.mean())}
    ms = {}
    from itertools import combinations

    for size in range(1, dimensions + 1):
        for axes in combinations(range(dimensions), size):
            others = tuple(axis for axis in range(dimensions) if axis not in axes)
            effect = means.mean(axis=others, keepdims=True) if others else means.copy()
            for subset, previous in effects.items():
                if set(subset) < set(axes):
                    effect = effect - previous
            effects[axes] = effect
            df = math.prod(shape[axis] - 1 for axis in axes)
            ms[axes] = float(
                repeats * math.prod(shape[axis] for axis in others) * np.sum(effect**2) / df
            )
    components: Dict[tuple, float] = {}
    for size in range(dimensions, 0, -1):
        for axes in combinations(range(dimensions), size):
            remaining = ms[axes] - residual
            for superset, component in components.items():
                if set(axes) < set(superset):
                    remaining -= (
                        repeats
                        * math.prod(
                            shape[axis] for axis in range(dimensions) if axis not in superset
                        )
                        * component
                    )
            components[axes] = remaining / (
                repeats * math.prod(shape[axis] for axis in range(dimensions) if axis not in axes)
            )
    names = ["Configuration"] + [MODES[kind] for kind in kinds]
    rows: List[Dict[str, Any]] = [
        {
            "component": " × ".join(names[axis] for axis in axes),
            "estimate": value,
            "variance": max(0.0, value),
        }
        for axes, value in sorted(components.items())
    ]
    rows.append({"component": "Repeat residual", "estimate": residual, "variance": residual})
    total = sum(row["variance"] for row in rows)
    for row in rows:
        row["share"] = row["variance"] / total if total > 0 else 0.0
    return {
        "rows": rows,
        "repeats": repeats,
        "method": "Balanced crossed random-effects ANOVA (method of moments)",
    }
