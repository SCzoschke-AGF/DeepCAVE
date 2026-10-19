"""Conditional trial forecasts under independent search-space random sampling.

Full-strength coupons are disjoint. Poissonization is used only for those
coupons; dependent t-way events use Monte Carlo or a rigorous union bound.
"""

from typing import Any, Callable, Dict, List, Optional

import hashlib
import json
from itertools import combinations
from math import ceil, comb, exp, fsum, isfinite, log, log1p, prod
from time import monotonic

import numpy as np
from scipy.stats import binom

from deepcave.evaluators.combinatorial import (
    RESOLUTIONS,
    CoverageSpec,
    _group_counts,
    compute_combinatorial_coverage,
)
from deepcave.evaluators.coverage import CoverageData, _python
from deepcave.evaluators.coverage_state import CoverageState

FORECAST_VERSION = 1
SAMPLING_MODEL = "Independent random sampling from the displayed ConfigSpace"
ASSUMPTION = (
    "Additional trials are independent random configurations that contribute to coverage. "
    "Failure and pruning rates are not modelled. This is not a forecast of adaptive TPE sampling."
)


def forecast_options(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Validate reproducible settings and hard allocation/work limits."""
    options: Dict[str, Any] = {}
    for key, default, lower, upper in (
        ("simulations", 2000, 20, 10000),
        ("seed", 0, 0, 2**32 - 1),
        ("max_trials", 100_000, 1, 10_000_000),
        ("max_groups", 10_000, 1, 100_000),
        ("max_tuples", 1_000_000, 1, 5_000_000),
        ("max_projections", 20_000_000, 1, 100_000_000),
        ("max_explicit_coupons", 100_000, 1, 1_000_000),
    ):
        value = inputs.get(key, default)
        try:
            valid = not isinstance(value, bool) and int(value) == value and lower <= value <= upper
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{key} must be an integer between {lower} and {upper}.")
        options[key] = int(value)
    for key, real_default, real_lower, real_upper in (
        ("probability", 0.8, 0.01, 0.99),
        ("timeout", 30.0, 0.1, 120.0),
    ):
        value = inputs.get(key, real_default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not real_lower <= value <= real_upper
        ):
            raise ValueError(f"{key} must be between {real_lower} and {real_upper}.")
        options[key] = float(value)
    return options


def _result(
    trials: Optional[int] = None,
    method: Optional[str] = None,
    status: str = "Unsupported",
    reason: str = "",
    **metadata: Any,
) -> Dict[str, Any]:
    return dict(trials_p80=trials, method=method, status=status, reason=reason, **metadata)


def level_probabilities(spec: CoverageSpec) -> List[np.ndarray]:
    """Derive fixed bin masses from supported definitions, never from trial counts."""
    supported = {
        "UniformFloatHyperparameter",
        "UniformIntegerHyperparameter",
        "NormalFloatHyperparameter",
        "NormalIntegerHyperparameter",
        "BetaFloatHyperparameter",
        "BetaIntegerHyperparameter",
        "CategoricalHyperparameter",
        "OrdinalHyperparameter",
    }
    probabilities = []
    for axis, size in zip(spec.axes, spec.sizes):
        if type(axis.hp).__name__ not in supported:
            raise ValueError(f"{axis.name}: random-sampling distribution cannot be reconstructed.")
        if axis.kind == "categorical":
            masses = np.asarray(axis.hp.probabilities, dtype=float)
        elif axis.kind == "ordinal":
            masses = np.full(size, 1 / size)
        elif axis.size is not None:
            edges = spec.rank_edges[axis.name]
            right = axis.mass_edges(edges[1:] - 1)[1]
            masses = np.diff(np.r_[0.0, right])
        else:
            # Existing continuous bins split the theoretical sampling CDF.
            masses = np.full(size, 1 / size)
        if (
            len(masses) != size
            or not np.isfinite(masses).all()
            or np.any(masses < 0)
            or not np.isclose(masses.sum(), 1, atol=1e-10)
        ):
            raise ValueError(f"{axis.name}: invalid or unknown bin probabilities.")
        probabilities.append(masses / masses.sum())
    return probabilities


def _check_deadline(deadline: float) -> None:
    if monotonic() >= deadline:
        raise TimeoutError("Shared forecast time limit reached.")


def _search_quantile(
    cdf: Callable[[int], float],
    probability: float,
    upper: int,
    deadline: float = float("inf"),
) -> int:
    lower = 0
    while cdf(upper) < probability:
        _check_deadline(deadline)
        upper *= 2
    while lower < upper:
        _check_deadline(deadline)
        middle = (lower + upper) // 2
        if cdf(middle) >= probability:
            upper = middle
        else:
            lower = middle + 1
    return lower


def _upper_bound(missing: int, log_min_probability: float, probability: float) -> Optional[int]:
    """Union bound m exp(-n p_min), also valid for overlapping t-way events."""
    if not missing:
        return 0
    if not isfinite(log_min_probability):
        return None
    log_budget = log(log(missing) - log1p(-probability)) - log_min_probability
    if log_budget > 690:
        return None
    # A slight outward rounding protects the bound against floating-point error.
    return max(1, ceil(exp(log_budget) * (1 + 1e-12)))


def uniform_coupon_forecast(
    total: int,
    missing: int,
    probability: float,
    deadline: float = float("inf"),
) -> Dict[str, Any]:
    """Use an exact occupancy chain for small spaces, Poissonization otherwise."""
    if not missing:
        return _result(0, "Exact", "Completed", uncertainty=None)
    if total == 1:
        return _result(1, "Exact", "Completed", uncertainty=None)
    upper = _upper_bound(missing, -log(total), probability)
    if upper is None:
        return _result(reason="Trial budget exceeds the supported numerical range.")
    if missing == 1:
        return _result(
            ceil(log1p(-probability) / log1p(-1 / total)),
            "Exact",
            "Completed",
            uncertainty=None,
        )
    if total <= 64:
        transition = np.zeros((missing + 1, missing + 1))
        ranks = np.arange(missing + 1)
        transition[ranks, ranks] = 1 - ranks / total
        transition[ranks[1:], ranks[1:] - 1] = ranks[1:] / total
        powers: List[np.ndarray] = [transition]

        def cdf(trials: int) -> float:
            distribution: np.ndarray = np.zeros(missing + 1)
            distribution[missing] = 1
            bit = 0
            while trials:
                if bit == len(powers):
                    powers.append(powers[-1] @ powers[-1])
                if trials & 1:
                    distribution = distribution @ powers[bit]
                trials >>= 1
                bit += 1
            return float(distribution[0])

        trials = _search_quantile(cdf, probability, upper, deadline)
        return _result(trials, "Exact", "Completed", uncertainty=None)
    # log(-expm1(log(q)/m)) avoids losing 1 - q**(1/m) for very large m.
    trials = ceil(-total * float(np.log(-np.expm1(log(probability) / missing))))
    lower = max(missing, ceil(log1p(-probability) / log1p(-1 / total)))
    return _result(
        trials,
        "Approximation",
        "Approximate",
        "Poissonized coupon collector.",
        uncertainty={"kind": "rigorous_quantile_bounds", "lower": lower, "upper": upper},
    )


def _coupon_forecast(
    masses: np.ndarray,
    probability: float,
    deadline: float = float("inf"),
) -> Dict[str, Any]:
    if not len(masses):
        return _result(0, "Exact", "Completed", uncertainty=None)
    if np.any(masses <= 0):
        return _result(reason="Missing combinations have zero probability under random sampling.")
    minimum = float(masses.min())
    upper = _upper_bound(len(masses), log(minimum), probability)
    if upper is None:
        return _result(reason="Trial budget exceeds the supported numerical range.")
    if minimum == 1:
        return _result(1, "Exact", "Completed", uncertainty=None)
    lower = max(len(masses), ceil(log1p(-probability) / log1p(-minimum)))
    if len(masses) <= 12:
        sums, signs = np.array([0.0]), np.array([1.0])
        for mass in masses:
            sums, signs = np.r_[sums, sums + mass], np.r_[signs, -signs]
        logs = np.full(len(sums), -np.inf)
        valid = sums < 1
        logs[valid] = np.log1p(-np.maximum(0, sums[valid]))

        def cdf(trials: int) -> float:
            if trials == 0:
                return 0.0
            return fsum((signs * np.exp(trials * logs)).tolist())

        return _result(
            _search_quantile(cdf, probability, upper, deadline),
            "Exact",
            "Completed",
            uncertainty=None,
        )

    def poisson_cdf(trials: int) -> float:
        if not trials:
            return 0.0
        return float(np.exp(np.log(-np.expm1(-trials * masses)).sum()))

    return _result(
        _search_quantile(poisson_cdf, probability, upper, deadline),
        "Approximation",
        "Approximate",
        "Poissonized nonuniform coupon collector.",
        uncertainty={"kind": "rigorous_quantile_bounds", "lower": lower, "upper": upper},
    )


def _full_forecast(
    spec: CoverageSpec,
    probabilities: List[np.ndarray],
    options: Dict[str, Any],
    deadline: float = float("inf"),
) -> Dict[str, Any]:
    _check_deadline(deadline)
    total = prod(spec.sizes)
    unique = np.unique(spec.points, axis=0)
    missing = total - len(unique)
    common = dict(total_combinations=total, observed_combinations=len(unique), missing=missing)
    if not missing:
        return {**_result(0, "Exact", "Completed", uncertainty=None), **common}
    uniform = all(np.allclose(p, 1 / len(p), rtol=1e-12, atol=0) for p in probabilities)
    if uniform:
        return {
            **uniform_coupon_forecast(total, missing, options["probability"], deadline),
            **common,
        }
    if total <= options["max_explicit_coupons"]:
        masses = np.array([1.0])
        codes = np.zeros(len(unique), dtype=np.int64)
        for column, p in enumerate(probabilities):
            masses = np.multiply.outer(masses, p).ravel()
            codes = codes * len(p) + unique[:, column]
        masses[codes] = 0
        mask = np.ones(total, dtype=bool)
        mask[codes] = False
        result = _coupon_forecast(masses[mask], options["probability"], deadline)
    else:
        log_min = sum(log(float(p.min())) if p.min() > 0 else -np.inf for p in probabilities)
        upper = _upper_bound(missing, log_min, options["probability"])
        result = (
            _result(
                method="Approximation",
                status="Approximate",
                kind="upper_bound",
                upper_bound_trials=upper,
                uncertainty=None,
                reason="Conservative union bound; nonuniform space exceeds explicit coupon limit.",
            )
            if upper is not None
            else _result(
                reason="Zero-probability levels or numerical limits prevent a reliable bound."
            )
        )
    return {**result, **common}


def _next_bound(
    spec: CoverageSpec,
    strength: int,
    probabilities: List[np.ndarray],
    options: Dict[str, Any],
    missing: Optional[int] = None,
) -> Dict[str, Any]:
    groups_count = comb(len(spec.axes), strength)
    if (
        groups_count > options["max_groups"]
        or groups_count * len(spec.points) > options["max_projections"]
    ):
        return _result(reason="Next-strength parameter-group work limit exceeded.")
    if missing is None:
        groups = list(combinations(range(len(spec.axes)), strength))
        total = sum(prod(spec.sizes[j] for j in group) for group in groups)
        covered = (
            int(_group_counts(spec.points, spec.sizes, groups).sum()) if len(spec.points) else 0
        )
        missing = total - covered
    minima = sorted(log(float(p.min())) if p.min() > 0 else -np.inf for p in probabilities)
    upper = _upper_bound(missing, sum(minima[:strength]), options["probability"])
    if upper is None:
        return _result(
            reason="Zero-probability levels or numerical limits prevent a reliable bound."
        )
    return _result(
        method="Approximation",
        status="Approximate",
        kind="upper_bound",
        upper_bound_trials=upper,
        missing=missing,
        uncertainty=None,
        reason="Conservative union bound over missing t-way combinations; allows dependent events.",
    )


def _next_forecast(
    spec: CoverageSpec,
    strength: int,
    probabilities: List[np.ndarray],
    options: Dict[str, Any],
    deadline: float,
) -> Dict[str, Any]:
    try:
        state = CoverageState(
            spec, strength, options["max_groups"], options["max_tuples"], options["max_projections"]
        )
    except ValueError:
        return _next_bound(spec, strength, probabilities, options)
    initial_missing = int(state.missing.sum())
    if not initial_missing:
        return _result(0, "Exact", "Completed", uncertainty=None)
    bound = _next_bound(spec, strength, probabilities, options, initial_missing)
    upper = bound.get("upper_bound_trials")
    if upper is None:
        return bound
    if upper * options["simulations"] * len(state.groups) > options["max_projections"]:
        return bound
    cumulative = [np.cumsum(p) for p in probabilities]
    for cdf in cumulative:
        cdf[-1] = 1.0
    durations = []
    censored = 0
    projections = 0
    termination = "completed"
    for simulation in range(options["simulations"]):
        if monotonic() >= deadline:
            termination = "timeout"
            break
        rng = np.random.default_rng(
            np.random.SeedSequence([options["seed"], spec.resolution, simulation])
        )
        missing = state.missing.copy()
        remaining, elapsed = initial_missing, 0
        while remaining and elapsed < options["max_trials"]:
            if monotonic() >= deadline:
                termination = "timeout"
                break
            count = min(
                min(256, max(8, ceil(upper / 4))),
                options["max_trials"] - elapsed,
                (options["max_projections"] - projections) // len(state.groups),
                max(1, 250_000 // len(state.groups)),
            )
            if count <= 0:
                termination = "work_limit"
                break
            uniforms = rng.random((count, len(spec.axes)))
            points = np.column_stack(
                [
                    np.searchsorted(cdf, uniforms[:, column], side="right")
                    for column, cdf in enumerate(cumulative)
                ]
            )
            ids = state.tuple_ids(points).ravel()
            hits = np.flatnonzero(missing[ids])
            new, first = np.unique(ids[hits], return_index=True)
            missing[new] = False
            remaining -= len(new)
            projections += count * len(state.groups)
            if not remaining:
                durations.append(elapsed + int((hits[first] // len(state.groups)).max()) + 1)
            elapsed += count
        if remaining:
            censored += 1
        if termination != "completed":
            break
    completed = len(durations)
    common = dict(
        simulations_requested=options["simulations"],
        simulations_completed=completed,
        simulations_censored=censored,
        simulations_not_started=options["simulations"] - completed - censored,
        simulated_projections=projections,
        missing=initial_missing,
    )
    # Never form a quantile from just the successful subset of censored continuations.
    if completed != options["simulations"]:
        return _result(
            method="MC",
            status="Censored",
            uncertainty=None,
            **common,
            upper_bound_trials=upper,
            reason=f"Simulation {termination if termination != 'completed' else 'trial_limit'}; quantile unavailable.",
        )
    durations.sort()
    quantile = durations[ceil(options["probability"] * completed) - 1]
    lo, hi = binom.ppf([0.025, 0.975], completed, options["probability"]).astype(int)
    return _result(
        quantile,
        "MC",
        "Completed",
        **common,
        uncertainty={
            "kind": "quantile_confidence_interval",
            "confidence": 0.95,
            "lower": durations[lo - 1] if lo > 0 else 0,
            "upper": durations[hi] if hi < completed else None,
        },
    )


def compute_coverage_forecast(
    data: CoverageData,
    options: Optional[Dict[str, Any]] = None,
    observed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Forecast additional trials from the unchanged historical coverage state."""
    options = forecast_options(options or {})
    deadline = monotonic() + options["timeout"]
    dimensions = sum(axis.size != 1 for axis in data.axes)
    unsupported = None
    if data.configspace is None:
        unsupported = "A reliable search-space definition is required."
    elif data.constrained or any(axis.conditional for axis in data.axes):
        unsupported = (
            "Conditional parameters and constraints are unsupported by the coverage model."
        )
    elif not dimensions:
        unsupported = "At least one varying parameter is required."
    elif not np.isfinite(data.points).all():
        unsupported = "Every observed configuration must define every parameter."
    if not unsupported and observed is None and data.count:
        observed = compute_combinatorial_coverage(data)
    rows = []
    for resolution in RESOLUTIONS:
        measurement: Dict[str, Any] = next(
            (r for r in (observed or {}).get("rows", []) if r["resolution"] == resolution), {}
        )
        current = measurement.get("observed_strength", 0)
        target = current + 1 if current < dimensions else None
        definition, probabilities = None, []
        if unsupported:
            next_result = full_result = _result(reason=unsupported)
        else:
            try:
                spec = CoverageSpec(data, resolution)
                probabilities = level_probabilities(spec)
                definition = spec.serialize()
            except (ValueError, AttributeError, TypeError, OverflowError) as error:
                next_result = full_result = _result(reason=str(error))
            else:
                try:
                    full_result = _full_forecast(spec, probabilities, options, deadline)
                except TimeoutError as error:
                    full_result = _result(status="Censored", reason=str(error))
                except (ValueError, AttributeError, TypeError, OverflowError) as error:
                    full_result = _result(reason=str(error))
                if target is None:
                    next_result = _result(
                        0,
                        "Exact",
                        "Completed",
                        "All strengths are already covered.",
                        uncertainty=None,
                    )
                elif measurement.get("strength_is_lower_bound"):
                    target = None
                    next_result = _result(
                        reason="Current strength is only a lower bound; next target is unknown."
                    )
                elif target == dimensions:
                    next_result = dict(full_result)
                elif monotonic() >= deadline:
                    next_result = _result(
                        method="MC", status="Censored", reason="Forecast time limit reached."
                    )
                else:
                    try:
                        next_result = _next_forecast(spec, target, probabilities, options, deadline)
                    except (ValueError, AttributeError, TypeError, OverflowError) as error:
                        next_result = _result(reason=str(error))
        statuses = {next_result["status"], full_result["status"]}
        status = next(
            s for s in ("Censored", "Unsupported", "Approximate", "Completed") if s in statuses
        )
        row = dict(
            resolution=resolution,
            current_strength=current,
            strength_is_lower_bound=measurement.get("strength_is_lower_bound", False),
            next_strength_target=target,
            full_strength_target=dimensions,
            forecast_status=status,
            binning=definition,
            level_probabilities=[p.tolist() for p in probabilities],
            next=next_result,
            full=full_result,
        )
        for name, result in (("next", next_result), ("full", full_result)):
            for field in ("trials_p80", "method", "uncertainty", "status"):
                row[f"{name}_strength_{field}"] = result.get(field)
        rows.append(row)
    definition = data.configspace.to_serialized_dict() if data.configspace is not None else None
    space_version = hashlib.sha256(
        json.dumps(definition, sort_keys=True, default=_python).encode()
    ).hexdigest()
    return dict(
        forecast_version=FORECAST_VERSION,
        source_fingerprint=data.fingerprint,
        search_space_version=space_version,
        sampling_model=SAMPLING_MODEL,
        assumption=ASSUMPTION,
        probability_threshold=options["probability"],
        options=options,
        seed=options["seed"],
        simulations=options["simulations"],
        observed_trials=data.count,
        rows=rows,
        observed=observed,
    )
