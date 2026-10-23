"""Validated statsmodels KDE shared by production and the ranking benchmark.

The transforms, product kernels, analytic LOO gradient and bounded L-BFGS-B
settings are extracted unchanged from examples/api/benchmark_tpe_kde.py.
"""

from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
)

import math
import signal
from contextlib import contextmanager
from time import perf_counter

import numpy as np
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from scipy.optimize import minimize
from scipy.special import logsumexp


def split_good_bad(
    objectives: np.ndarray, train: np.ndarray, maximize: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Use only training targets; ties preserve original trial-number order."""
    utility = objectives if maximize else -objectives
    ordered = train[np.argsort(-utility[train], kind="stable")]
    n_good = math.ceil(0.2 * len(train))
    return ordered[:n_good], ordered[n_good:]


def kde_encoding(trials: Sequence[Any], space: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict]:
    """Encode c/u/o kernels using declared bounds, never held-out objective values.

    Non-log integer and stepped float values use ordinal grid indices. Log integers
    use a continuous log relaxation (explicitly recorded), since Wang-Ryzin assumes
    an equally spaced lattice. Continuous values use declared-bound normalization.
    """
    columns, types, description = [], [], {}
    column: Any
    for name, distribution in space.items():
        values = [trial.params[name] for trial in trials]
        if isinstance(distribution, CategoricalDistribution):
            column = [distribution.to_internal_repr(value) for value in values]
            kind, transform = "u", "unordered category index"
        elif isinstance(distribution, (FloatDistribution, IntDistribution)):
            column = np.asarray(values, dtype=float)
            if distribution.log:
                column = (np.log(column) - np.log(distribution.low)) / (
                    np.log(distribution.high) - np.log(distribution.low)
                )
                kind = "c"
                transform = "log, declared bounds to [0,1]"
                if isinstance(distribution, IntDistribution):
                    transform += "; continuous relaxation of log integer"
            elif isinstance(distribution, IntDistribution) or distribution.step is not None:
                column = np.rint(
                    (column - distribution.low) / distribution.step,  # type: ignore[operator]
                )
                kind, transform = "o", "ordinal grid index (value-low)/step"
            else:
                column = (column - distribution.low) / (distribution.high - distribution.low)
                kind, transform = "c", "declared bounds to [0,1]"
        else:
            raise ValueError(f"Unsupported distribution for {name}: {distribution}")
        columns.append(column)
        types.append(kind)
        description[name] = {"kernel_type": kind, "transform": transform}
    return np.column_stack(columns), "".join(types), description


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Bound one CV fit on POSIX; restore the caller's signal handler afterwards."""

    def timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Bounded LOO optimization exceeded {seconds:g} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def loo_objective(data: np.ndarray, var_type: str) -> Callable:
    """Cache pairwise distances for exact log-space KDE LOO and its analytic gradient.

    This is the statsmodels Gaussian/Wang-Ryzin/Aitchison-Aitken product kernel,
    excluding the diagonal and dividing by n-1. statsmodels.loo_likelihood omits
    the latter bandwidth-independent factor; tests check this identity. Continuous
    optimizer coordinates are log bandwidths; discrete coordinates are bandwidths.
    """
    n = len(data)
    features = []
    levels = []
    for index, kind in enumerate(var_type):
        delta = data[:, index, None] - data[None, :, index]
        features.append(delta**2 if kind == "c" else (np.abs(delta) if kind == "o" else delta == 0))
        levels.append(len(np.unique(data[:, index])))

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        logs = np.zeros((n, n))
        for index, kind in enumerate(var_type):
            feature = features[index]
            value = theta[index]
            if kind == "c":
                logs += -0.5 * feature * np.exp(-2 * value) - value - 0.5 * np.log(2 * np.pi)
            elif kind == "o":
                logs += np.log1p(-value) + feature * np.log(value)
                logs -= (feature != 0) * np.log(2)
            else:
                logs += np.where(
                    feature, np.log1p(-value), np.log(value) - np.log(levels[index] - 1)
                )
        np.fill_diagonal(logs, -np.inf)
        row_sums = logsumexp(logs, axis=1)
        weights = np.exp(logs - row_sums[:, None])
        gradient = np.empty(len(var_type))
        for index, kind in enumerate(var_type):
            feature = features[index]
            value = theta[index]
            expectation = np.sum(weights * feature) / n
            if kind == "c":
                gradient[index] = 1 - expectation * np.exp(-2 * value)
            elif kind == "o":
                gradient[index] = 1 / (1 - value) - expectation / value
            else:
                gradient[index] = expectation / (1 - value) - (1 - expectation) / value
        loss = -float(np.mean(row_sums - np.log(n - 1)))
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite bounded LOO objective/gradient")
        return loss, gradient

    return objective


def fit_kde(
    data: np.ndarray,
    var_type: str,
    space: Dict[str, Any],
    bandwidth: str = "bounded_ml",
    timeout: float = 120,
    discrete_initial: float = 0.5,
    _time_limit: Optional[Callable[[float], ContextManager]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Use the original regularized baseline or verified bounded LOO optimization.

    The optimized model never falls back to normal_reference. A timeout or a
    convergence failure is returned explicitly and must not count as a successful
    fit. No bandwidth is clipped after optimization.
    """
    from statsmodels.nonparametric.kernel_density import KDEMultivariate

    if bandwidth not in ("bounded_ml", "normal_reference"):
        raise ValueError(f"Unsupported bandwidth method: {bandwidth}")
    if not 1e-6 <= discrete_initial <= 1 - 1e-6:
        raise ValueError("Discrete initial bandwidth must lie within the unchanged bounds")
    if len(data) <= data.shape[1]:
        raise ValueError("KDE needs more training observations per good/bad group than dimensions")
    for index, (name, distribution) in enumerate(space.items()):
        if var_type[index] == "u":
            _, counts = np.unique(data[:, index], return_counts=True)
            if len(counts) != len(distribution.choices):
                raise ValueError(f"KDE group lacks declared categories of {name}")
            if bandwidth == "bounded_ml" and min(counts) < 2:
                raise ValueError(f"LOO requires at least two observations per category of {name}")
    start = perf_counter()
    reference = KDEMultivariate(data, var_type, bw="normal_reference")
    raw = reference.bw.copy()
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite normal-reference bandwidth")
    diagnostics = {
        "requested": bandwidth,
        "used": bandwidth,
        "fallback": None,
        "raw_normal_reference_bandwidth": raw.tolist(),
    }
    if bandwidth == "normal_reference":
        # Preserve the previously executed baseline exactly, including its repair.
        reference.bw = np.maximum(raw, 1e-8)
        for index, kind in enumerate(var_type):
            if kind != "c":
                reference.bw[index] = min(reference.bw[index], 1 - 1e-8)
        diagnostics.update(
            {
                "bandwidth": reference.bw.tolist(),
                "regularized": bool(np.any(raw != reference.bw)),
                "fit_seconds": perf_counter() - start,
            }
        )
        return reference, diagnostics

    continuous = np.array([kind == "c" for kind in var_type])
    lower = np.full(len(var_type), 1e-6)
    upper = np.where(continuous, 10.0, 1 - 1e-6)
    initial = np.where(continuous, np.clip(raw, 1e-5, 1.0), discrete_initial)
    theta = initial.copy()
    theta[continuous] = np.log(theta[continuous])
    bounds = [
        (np.log(lo), np.log(hi)) if kind == "c" else (lo, hi)
        for lo, hi, kind in zip(lower, upper, var_type)
    ]
    objective = loo_objective(data, var_type)
    initial_loss, _ = objective(theta)
    best = [initial_loss, theta.copy()]
    evaluations = 0

    def tracked_objective(value: np.ndarray) -> Tuple[float, np.ndarray]:
        nonlocal evaluations
        loss, gradient = objective(value)
        evaluations += 1
        if loss < best[0]:
            best[:] = [loss, value.copy()]
        return loss, gradient

    try:
        with (_time_limit or time_limit)(timeout):
            result = minimize(
                tracked_objective,
                theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-5, "maxls": 40},
            )
        final_theta = result.x
        success, status, message = bool(result.success), int(result.status), str(result.message)
        iterations = int(result.nit)
    except TimeoutError as error:
        final_theta = best[1]
        success, status, message, iterations = False, -1, str(error), None
    final = final_theta.copy()
    final[continuous] = np.exp(final[continuous])
    if not (
        np.isfinite(final).all()
        and (final >= lower * (1 - 1e-14)).all()
        and (final <= upper * (1 + 1e-14)).all()
    ):
        raise ValueError("Optimizer returned bandwidth outside declared bounds")
    final_loss, gradient = objective(final_theta)
    gain = initial_loss - final_loss
    moved = bool(np.max(np.abs(final - initial)) > 1e-6)
    verified = success and gain > 1e-6 and moved
    model = KDEMultivariate(data, var_type, bw=final)
    diagnostics.update(
        {
            "success": success,
            "status": status,
            "message": message,
            "iterations": iterations,
            "evaluations": evaluations,
            "optimization_verified": verified,
            "initial_bandwidth": initial.tolist(),
            "bandwidth": final.tolist(),
            "lower_bounds": lower.tolist(),
            "upper_bounds": upper.tolist(),
            "initial_mean_loo_loglikelihood": -initial_loss,
            "final_mean_loo_loglikelihood": -final_loss,
            "initial_sum_loo_loglikelihood": -initial_loss * len(data),
            "final_sum_loo_loglikelihood": -final_loss * len(data),
            "mean_loo_loglikelihood_gain": gain,
            "bandwidth_changed": moved,
            "post_fit_clipping": False,
            "gradient_inf_norm": float(np.max(np.abs(gradient))),
            "at_lower_bound": np.isclose(final, lower, rtol=1e-5, atol=1e-10).tolist(),
            "at_upper_bound": np.isclose(final, upper, rtol=1e-5, atol=1e-10).tolist(),
            "fit_seconds": perf_counter() - start,
        }
    )
    return model, diagnostics


def kde_logpdf(model: Any, queries: np.ndarray) -> np.ndarray:
    """Evaluate exactly the statsmodels product KDE in log space, avoiding pdf underflow.

    Kernels are Gaussian, Wang-Ryzin and Aitchison-Aitken, with the same observed
    category counts as KDEMultivariate.pdf. No probability floor or extra surrogate.
    """
    result = []
    for start in range(0, len(queries), 64):
        points = queries[start : start + 64]
        logs = np.zeros((len(points), len(model.data)))
        for index, kind in enumerate(model.var_type):
            delta = points[:, index, None] - model.data[None, :, index]
            h = model.bw[index]
            if kind == "c":
                logs += -0.5 * (delta / h) ** 2 - np.log(h) - 0.5 * np.log(2 * np.pi)
            elif kind == "o":
                logs += np.log1p(-h) + np.where(
                    delta == 0, 0, np.abs(delta) * np.log(h) - np.log(2)
                )
            else:
                levels = len(np.unique(model.data[:, index]))
                logs += np.where(delta == 0, np.log1p(-h), np.log(h) - np.log(levels - 1))
        result.extend(logsumexp(logs, axis=1) - np.log(len(model.data)))
    return np.asarray(result)
