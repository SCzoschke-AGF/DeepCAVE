# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# noqa: D400
"""Shared empirical-analysis helpers for Optuna hyperparameter plugins."""

from typing import Any, Dict, List, Optional, Tuple

from collections import OrderedDict
from threading import Lock, RLock

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)

from deepcave.runs import AbstractRun
from deepcave.runs.status import Status

N_BINS = 10
TRIAL_TABLE_CACHE_SIZE = 8


# Feature-specific plugin caches cannot be reused when a different feature is selected. Keep a
# small second-level cache containing the expensive, feature-independent trial table instead.
_TRIAL_TABLE_CACHE: "OrderedDict[Tuple[str, str, int, str], pd.DataFrame]" = OrderedDict()
_TRIAL_TABLE_CACHE_LOCK = RLock()
_TRIAL_TABLE_BUILD_LOCKS: Dict[Tuple[str, str, int, str], Lock] = {}
_TRIAL_TABLE_CACHE_GENERATION = 0


def clear_trial_table_cache() -> None:
    """Clear all in-process analytical tables without touching user planning state."""
    global _TRIAL_TABLE_CACHE_GENERATION
    with _TRIAL_TABLE_CACHE_LOCK:
        _TRIAL_TABLE_CACHE.clear()
        _TRIAL_TABLE_BUILD_LOCKS.clear()
        _TRIAL_TABLE_CACHE_GENERATION += 1


def _trial_table_cache_key(
    run: AbstractRun,
    objective_id: int,
    optimize: str,
) -> Optional[Tuple[str, str, int, str]]:
    """Return a content-aware cache key, or no key for lightweight test doubles."""
    try:
        run_id = run.id
        latest_change = run.latest_change
    except (AttributeError, RuntimeError):
        return None

    if not isinstance(run_id, str) or not isinstance(latest_change, (int, float)):
        return None
    # The object identity makes an explicit study refresh invalidate the table immediately. The
    # timestamp additionally covers in-place updates without hashing the complete Optuna pickle on
    # every feature switch.
    run_version = f"{id(run)}:{latest_change}:{len(run.history)}"
    return run_id, run_version, objective_id, optimize


def _configuration_values(run: AbstractRun, config_id: int) -> Any:
    """Read an already stored configuration without reconstructing it through ConfigSpace."""
    configurations = getattr(run, "configs", None)
    if configurations is not None and config_id in configurations:
        return configurations[config_id]
    return run.get_config(config_id)


def _build_trial_table(
    run: AbstractRun,
    objective_id: int,
    optimize: str,
) -> pd.DataFrame:
    """Build the reusable objective and feature table for all successful trials."""
    hyperparameter_names = list(run.configspace.keys())
    configuration_cache: Dict[int, Dict[str, Any]] = {}
    rows = []

    for history_index, trial in enumerate(run.history):
        objective_value = trial.costs[objective_id]
        if (
            trial.status != Status.SUCCESS
            or objective_value is None
            or not np.isfinite(objective_value)
        ):
            continue

        if trial.config_id not in configuration_cache:
            configuration = _configuration_values(run, trial.config_id)
            configuration_cache[trial.config_id] = {
                name: configuration.get(name) for name in hyperparameter_names
            }

        row = {
            "trial_number": int(trial.additional.get("optuna_trial_number", history_index)),
            "objective": float(objective_value),
        }
        row.update(configuration_cache[trial.config_id])
        rows.append(row)

    columns = ["trial_number", "objective", *hyperparameter_names]
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        table["objective_rank"] = pd.Series(dtype=float)
        table["rank_quantile"] = pd.Series(dtype=float)
        return table

    table["objective_rank"] = table["objective"].rank(
        ascending=optimize == "lower",
        method="average",
    )
    table["rank_quantile"] = table["objective"].rank(
        ascending=optimize == "upper",
        method="average",
        pct=True,
    )
    return table


def _get_trial_table(run: AbstractRun, objective_id: int) -> pd.DataFrame:
    """Return one cached all-feature trial table per run version and objective."""
    objective = run.get_objective(objective_id)
    optimize = objective.optimize if objective is not None else "lower"
    key = _trial_table_cache_key(run, objective_id, optimize)
    if key is None:
        return _build_trial_table(run, objective_id, optimize)

    with _TRIAL_TABLE_CACHE_LOCK:
        table = _TRIAL_TABLE_CACHE.get(key)
        if table is not None:
            _TRIAL_TABLE_CACHE.move_to_end(key)
            return table

        build_lock = _TRIAL_TABLE_BUILD_LOCKS.setdefault(key, Lock())
        generation = _TRIAL_TABLE_CACHE_GENERATION

    # Only callers for this exact run/version/objective wait for one another. Large independent
    # studies can build in parallel instead of being serialized by the LRU bookkeeping lock.
    with build_lock:
        with _TRIAL_TABLE_CACHE_LOCK:
            table = _TRIAL_TABLE_CACHE.get(key)
            if table is not None:
                _TRIAL_TABLE_CACHE.move_to_end(key)
                return table

        table = _build_trial_table(run, objective_id, optimize)

        with _TRIAL_TABLE_CACHE_LOCK:
            _TRIAL_TABLE_BUILD_LOCKS.pop(key, None)
            if generation == _TRIAL_TABLE_CACHE_GENERATION:
                # Discard obsolete versions of the same run instead of retaining large
                # dataframes until the size limit happens to evict them.
                stale_keys = [
                    cached_key
                    for cached_key in _TRIAL_TABLE_CACHE
                    if cached_key[0] == key[0] and cached_key[1] != key[1]
                ]
                for stale_key in stale_keys:
                    del _TRIAL_TABLE_CACHE[stale_key]

                _TRIAL_TABLE_CACHE[key] = table
                while len(_TRIAL_TABLE_CACHE) > TRIAL_TABLE_CACHE_SIZE:
                    _TRIAL_TABLE_CACHE.popitem(last=False)
        return table


def _is_optuna_run_or_group(run: AbstractRun) -> bool:
    """Return whether a run is an Optuna study or a group made only from Optuna studies."""
    if run.prefix == "Optuna":
        return True
    if run.prefix != "group":
        return False

    members = getattr(run, "runs", [])
    return bool(members) and all(member.prefix == "Optuna" for member in members)


def _group_parameter_values(
    values: pd.Series,
    hyperparameter: Hyperparameter,
    n_bins: int = N_BINS,
    discrete_numeric: bool = False,
) -> pd.Series:
    """Assign observed hyperparameter values to ordered display bins."""
    if isinstance(hyperparameter, CategoricalHyperparameter):
        groups = pd.Categorical(values, categories=list(hyperparameter.choices), ordered=True)
        return pd.Series(groups, index=values.index, name="group")

    if isinstance(hyperparameter, OrdinalHyperparameter):
        groups = pd.Categorical(values, categories=list(hyperparameter.sequence), ordered=True)
        return pd.Series(groups, index=values.index, name="group")

    if isinstance(hyperparameter, Constant):
        return pd.Series(
            pd.Categorical(values, categories=[hyperparameter.value], ordered=True),
            index=values.index,
            name="group",
        )

    unique_values = sorted(values.dropna().unique())
    if len(unique_values) <= n_bins and (
        isinstance(hyperparameter, IntegerHyperparameter) or discrete_numeric
    ):
        groups = pd.Categorical(values, categories=unique_values, ordered=True)
        return pd.Series(groups, index=values.index, name="group")

    if hyperparameter.log:
        edges = np.geomspace(hyperparameter.lower, hyperparameter.upper, n_bins + 1)
    else:
        edges = np.linspace(hyperparameter.lower, hyperparameter.upper, n_bins + 1)

    return pd.Series(
        pd.cut(values, edges, include_lowest=True, duplicates="drop"),
        index=values.index,
        name="group",
    )


def _add_trial_statistics(
    trials: pd.DataFrame,
    optimize: str,
    elite_proportion: float,
) -> Tuple[pd.DataFrame, float, int]:
    """Add direction-aware ranks, rank quantiles, and elite membership."""
    threshold_quantile = 1 - elite_proportion if optimize == "upper" else elite_proportion
    elite_threshold = float(trials["objective"].quantile(threshold_quantile))
    if optimize == "upper":
        is_elite = trials["objective"] >= elite_threshold
    else:
        is_elite = trials["objective"] <= elite_threshold

    objective_rank = trials.get("objective_rank")
    if objective_rank is None:
        objective_rank = trials["objective"].rank(
            ascending=optimize == "lower",
            method="average",
        )
    rank_quantile = trials.get("rank_quantile")
    if rank_quantile is None:
        rank_quantile = trials["objective"].rank(
            ascending=optimize == "upper",
            method="average",
            pct=True,
        )
    return (
        trials.assign(
            objective_rank=objective_rank,
            rank_quantile=rank_quantile,
            is_elite=is_elite,
            elite_rank=objective_rank.where(is_elite),
        ),
        elite_threshold,
        int(is_elite.sum()),
    )


def _collect_trials(
    run: AbstractRun,
    objective_id: int,
    hyperparameter_names: List[str],
) -> pd.DataFrame:
    """Select active parameters from the cached objective-wide trial table."""
    table = _get_trial_table(run, objective_id)
    columns = [
        "trial_number",
        "objective",
        "objective_rank",
        "rank_quantile",
        *hyperparameter_names,
    ]
    return table.loc[:, columns].dropna(subset=hyperparameter_names).copy()


def _get_objectives_without_time(run: AbstractRun) -> List[Any]:
    """Return real objectives while omitting DeepCAVE's synthetic time objective."""
    objectives = run.get_objectives()
    if objectives and objectives[-1].name == "Time":
        objectives = objectives[:-1]
    return objectives
