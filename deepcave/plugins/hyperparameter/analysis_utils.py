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

from typing import Any, List, Tuple

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

    objective_rank = trials["objective"].rank(
        ascending=optimize == "lower",
        method="average",
    )
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
    """Collect successful finite trials in which all selected parameters are active."""
    rows = []
    for history_index, trial in enumerate(run.history):
        objective_value = trial.costs[objective_id]
        if (
            trial.status != Status.SUCCESS
            or objective_value is None
            or not np.isfinite(objective_value)
        ):
            continue

        configuration = run.get_config(trial.config_id)
        values = [configuration.get(name) for name in hyperparameter_names]
        if any(value is None for value in values):
            continue

        row = {
            "trial_number": int(trial.additional.get("optuna_trial_number", history_index)),
            "objective": float(objective_value),
        }
        row.update(dict(zip(hyperparameter_names, values)))
        rows.append(row)

    return pd.DataFrame(rows)


def _get_objectives_without_time(run: AbstractRun) -> List[Any]:
    """Return real objectives while omitting DeepCAVE's synthetic time objective."""
    objectives = run.get_objectives()
    if objectives and objectives[-1].name == "Time":
        objectives = objectives[:-1]
    return objectives
