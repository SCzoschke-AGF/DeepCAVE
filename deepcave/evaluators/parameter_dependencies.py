"""Empirical sampling dependencies and supported conditional performance effects."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
)


def encode_parameter(
    values: pd.Series, hyperparameter: Any, n_bins: int
) -> Tuple[np.ndarray, List[str]]:
    """Use declared discrete levels or adaptive quantile bins fitted to sampled values.

    Inactive values receive code -1 and are excluded pairwise. Numeric parameters with
    few distinct observations retain their exact levels. Bins are shared between the
    sampling and performance analyses, so the latter cannot silently redefine a pair.
    """
    if isinstance(hyperparameter, CategoricalHyperparameter):
        groups = pd.Categorical(values, categories=list(hyperparameter.choices), ordered=True)
    elif isinstance(hyperparameter, OrdinalHyperparameter):
        groups = pd.Categorical(values, categories=list(hyperparameter.sequence), ordered=True)
    elif isinstance(hyperparameter, Constant):
        groups = pd.Categorical(values, categories=[hyperparameter.value], ordered=True)
    else:
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        unique = sorted(numeric.dropna().unique())
        if len(unique) <= n_bins:
            groups = pd.Categorical(numeric, categories=unique, ordered=True)
        else:
            groups = pd.qcut(numeric, q=n_bins, duplicates="drop").array
    return np.asarray(groups.codes), [str(level) for level in groups.categories]


def sampling_dependency(counts: np.ndarray) -> Dict[str, Any]:
    """Compute symmetric NMI and the observed-versus-independent explanation.

    NMI = 2 I(A; B) / (H(A) + H(B)). A constant marginal, or fewer than two
    jointly active trials, leaves the dependency undefined rather than implying
    independence. Zero expected counts similarly leave enrichment undefined.
    """
    counts = np.asarray(counts, dtype=float)
    total = float(counts.sum())
    expected = np.zeros_like(counts)
    deviation = np.full_like(counts, np.nan)
    score = None
    if total:
        expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
        np.divide(counts - expected, expected, out=deviation, where=expected > 0)
        probabilities = counts / total
        rows = probabilities.sum(axis=1)
        columns = probabilities.sum(axis=0)
        if total >= 2 and np.count_nonzero(rows) > 1 and np.count_nonzero(columns) > 1:
            occupied = probabilities > 0
            independent = expected / total
            mi = float(
                np.sum(
                    probabilities[occupied]
                    * np.log(probabilities[occupied] / independent[occupied])
                )
            )
            entropy_a = -float(np.sum(rows[rows > 0] * np.log(rows[rows > 0])))
            entropy_b = -float(np.sum(columns[columns > 0] * np.log(columns[columns > 0])))
            score = float(np.clip(2 * mi / (entropy_a + entropy_b), 0, 1))
    return {
        "score": score,
        "n": int(total),
        "expected": expected.tolist(),
        "deviation": [
            [float(value) if np.isfinite(value) else None for value in row] for row in deviation
        ],
    }


def dependency_label(score: Any) -> str:
    """Return transparent descriptive bands, without significance claims."""
    if score is None:
        return "Not estimable"
    return "Low" if score < 0.1 else "Medium" if score < 0.3 else "High"


def pair_summary(
    codes_a: np.ndarray,
    codes_b: np.ndarray,
    labels_a: List[str],
    labels_b: List[str],
    objective: np.ndarray,
) -> Dict[str, Any]:
    """Aggregate sampling and performance separately over the same joint bins."""
    active = (codes_a >= 0) & (codes_b >= 0)
    shape = (len(labels_a), len(labels_b))
    counts = np.zeros(shape, dtype=int)
    np.add.at(counts, (codes_a[active], codes_b[active]), 1)
    usable = active & np.isfinite(objective)
    frame = pd.DataFrame(
        {"a": codes_a[usable], "b": codes_b[usable], "objective": objective[usable]}
    )
    performance_counts = np.zeros(shape, dtype=int)
    medians = np.full(shape, np.nan)
    for (a, b), group in frame.groupby(["a", "b"], sort=False):
        performance_counts[a, b] = len(group)
        medians[a, b] = float(group["objective"].median())
    marginal = np.full(shape[0], np.nan)
    for a, group in frame.groupby("a", sort=False):
        marginal[a] = float(group["objective"].median())
    marginal_b = np.full(shape[1], np.nan)
    for b, group in frame.groupby("b", sort=False):
        marginal_b[b] = float(group["objective"].median())
    return {
        "labels_a": labels_a,
        "labels_b": labels_b,
        "counts": counts.tolist(),
        "dependency": sampling_dependency(counts),
        "excluded": int((~active).sum()),
        "performance_counts": performance_counts.tolist(),
        "medians": [
            [float(value) if np.isfinite(value) else None for value in row] for row in medians
        ],
        "marginal_a": [float(value) if np.isfinite(value) else None for value in marginal],
        "marginal_b": [float(value) if np.isfinite(value) else None for value in marginal_b],
        "objective_scale": float(np.std(objective[usable])) if usable.any() else 0.0,
    }


def orient_pair(pair: Dict[str, Any], reverse: bool = False) -> Dict[str, Any]:
    """Orient one symmetric aggregate with the requested target on its first axis."""
    if not reverse:
        return pair
    shape = (len(pair["labels_a"]), len(pair["labels_b"]))
    return {
        **pair,
        "labels_a": pair["labels_b"],
        "labels_b": pair["labels_a"],
        "marginal_a": pair["marginal_b"],
        "marginal_b": pair["marginal_a"],
        **{
            key: np.asarray(pair[key], dtype=object).reshape(shape).T.tolist()
            for key in ("counts", "performance_counts", "medians")
        },
        "dependency": {
            **pair["dependency"],
            **{
                key: np.asarray(pair["dependency"][key], dtype=object).reshape(shape).T.tolist()
                for key in ("expected", "deviation")
            },
        },
    }


def conditional_effect(pair: Dict[str, Any], optimize: str, min_support: int) -> Dict[str, Any]:
    """Assess nonparallel median profiles only on sufficiently supported overlap.

    Every observed conditioning region must cover at least two common target bins,
    at least 60% of observed target bins, and the marginal and supported regional
    optima. Centering each region's profile removes conditioning-only offsets. The RMS spread
    between these centered profiles is scaled by the pooled objective standard
    deviation. This is descriptive, not an inferential or causal interaction test.
    """
    shape = (len(pair["labels_a"]), len(pair["labels_b"]))
    counts = np.asarray(pair["performance_counts"], dtype=int).reshape(shape)
    medians = np.asarray(pair["medians"], dtype=float).reshape(shape)
    supported = counts >= min_support
    target = counts.sum(axis=1) > 0
    regions = counts.sum(axis=0) > 0
    common = (
        supported[:, regions].all(axis=1) & target
        if regions.any()
        else np.zeros(len(target), dtype=bool)
    )
    overlap = float(common.sum() / target.sum()) if target.any() else 0.0
    score = pair["dependency"]["score"]
    sampling = dependency_label(score)
    scale = pair["objective_scale"]
    direction = -1 if optimize == "upper" else 1
    marginal = np.asarray(pair["marginal_a"], dtype=float) * direction
    tolerance = scale * 0.05
    marginal_best = np.zeros(len(target), dtype=bool)
    if target.any():
        marginal_best = target & (marginal <= np.nanmin(marginal) + tolerance)
    regional_optima_covered = True
    for b in np.flatnonzero(regions):
        supported_bins = supported[:, b]
        if not supported_bins.any():
            regional_optima_covered = False
            break
        profile = medians[:, b] * direction
        regional_best = supported_bins & (profile <= profile[supported_bins].min() + tolerance)
        if not (regional_best & common).any():
            regional_optima_covered = False
            break
    enough = (
        regions.sum() >= 2
        and common.sum() >= 2
        and overlap >= 0.6
        and bool((common & marginal_best).any())
        and regional_optima_covered
    )
    result = {
        "sampling": sampling,
        "interaction": "Not estimable",
        "interaction_score": None,
        "stability": "Not estimable",
        "supported": supported.tolist(),
        "common": common.tolist(),
        "overlap": overlap,
        "regions": int(regions.sum()),
        "performance_n": int(counts.sum()),
        "diagnosis": "Potentially confounded" if sampling == "High" else "Insufficient support",
        "explanation": (
            "Conditional effects cannot be compared reliably: shared support is insufficient "
            "or an observed optimum lacks overlap."
        ),
    }
    if not enough or score is None:
        return result
    profiles = medians[np.ix_(common, regions)]
    centered = profiles - profiles.mean(axis=0, keepdims=True)
    interaction_score = (
        float(np.sqrt(np.mean(np.var(centered, axis=1))) / scale) if scale > 0 else 0.0
    )
    strength = (
        "Low" if interaction_score < 0.1 else "Medium" if interaction_score < 0.3 else "Strong"
    )
    directed = profiles * direction
    best = directed <= directed.min(axis=0, keepdims=True) + tolerance
    flat = bool(np.all(np.ptp(profiles, axis=0) <= tolerance))
    stability = "Not identifiable" if flat else "High" if best.all(axis=1).any() else "Low"
    if flat:
        diagnosis = "Inconclusive"
        explanation = "Supported profiles are flat; no distinct sweet spot is identifiable."
    elif strength != "Low" or stability == "Low":
        diagnosis = "Interaction"
        explanation = (
            "Conditional profiles change shape or their optima shift across supported regions; "
            "this is an empirical interaction hint."
        )
    elif sampling == "Low":
        diagnosis = "Robust"
        explanation = (
            "Similar optimum and conditional effect across sufficiently supported regions, "
            "with low sampling dependency."
        )
    else:
        diagnosis = "Stable within observed overlap"
        explanation = (
            "Conditional effects are stable in the shared observed bins; "
            "sampling remains dependent."
        )
    result.update(
        interaction=strength,
        interaction_score=interaction_score,
        stability=stability,
        diagnosis=diagnosis,
        explanation=explanation,
    )
    return result
