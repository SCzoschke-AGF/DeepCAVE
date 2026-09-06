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

#  noqa: D400
"""
# PED-ANOVA

This module provides PED-ANOVA importance evaluation for DeepCAVE runs.

The implementation follows the algorithm and Scott Parzen estimator used by Optuna 4.9 while
operating directly on DeepCAVE's ``AbstractRun`` representation.

## Classes
    - PEDANOVA: Calculate single-objective hyperparameter importances in a target region.
"""

from typing import Dict, List, Optional, Tuple, Union

from math import ceil, erf, sqrt

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)

from deepcave.runs import AbstractRun, Status
from deepcave.runs.objective import Objective


class PEDANOVA:
    """Calculate PED-ANOVA importances directly from a DeepCAVE run."""

    def __init__(self, run: AbstractRun):
        """Initialize the evaluator with a DeepCAVE run."""
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        self.run = run
        self.cs = run.configspace
        self.hp_names = list(self.cs.keys())
        self._importances: Dict[str, float] = {}
        self._n_steps = 50
        self._prior_weight = 1.0
        self._min_n_top_trials = 2

    @staticmethod
    def _normal_cdf(values: np.ndarray) -> np.ndarray:
        """Evaluate the standard normal cumulative distribution function."""
        values = np.asarray(values, dtype=float)
        flat = values.ravel()
        cdf = np.fromiter(
            (0.5 * (1.0 + erf(value / sqrt(2.0))) for value in flat),
            dtype=float,
            count=flat.size,
        )
        return cdf.reshape(values.shape)

    @classmethod
    def _normal_mass(cls, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return the standard normal probability mass between two bounds."""
        return cls._normal_cdf(upper) - cls._normal_cdf(lower)

    def _get_top_mask(
        self,
        values: np.ndarray,
        quantile: float,
        is_lower_better: bool,
    ) -> np.ndarray:
        """Select a tie-inclusive empirical quantile with a minimum sample count."""
        sign = 1.0 if is_lower_better else -1.0
        losses = sign * np.asarray(values, dtype=float)
        if losses.size < self._min_n_top_trials:
            raise RuntimeError(
                "PED-ANOVA requires at least " f"{self._min_n_top_trials} completed finite trials."
            )

        quantile_index = int(ceil(quantile * losses.size)) - 1
        cutoff = max(
            np.partition(losses, self._min_n_top_trials - 1)[self._min_n_top_trials - 1],
            np.partition(losses, quantile_index)[quantile_index],
        )
        return losses <= cutoff

    def _numerical_pdf(
        self,
        values: np.ndarray,
        hp: Union[FloatHyperparameter, IntegerHyperparameter],
    ) -> np.ndarray:
        """Estimate a numerical density on a discretized grid using Scott's rule."""
        n_steps = self._n_steps
        lower = float(hp.lower)
        upper = float(hp.upper)
        is_log = bool(getattr(hp, "log", False))

        if isinstance(hp, IntegerHyperparameter):
            domain_size = int(hp.upper - hp.lower + 1)
            if is_log:
                log2_domain_size = int(ceil(np.log(domain_size) / np.log(2))) + 1
                n_steps = min(log2_domain_size, n_steps)
            else:
                n_steps = min(domain_size, n_steps)

        transform = np.log if is_log else np.asarray
        grids = np.linspace(transform(lower), transform(upper), n_steps)
        transformed_values = transform(np.asarray(values, dtype=float))
        step_size = grids[1] - grids[0]
        grid_indices = np.searchsorted(grids, transformed_values - step_size / 2)

        unique_indices, unique_counts = np.unique(grid_indices, return_counts=True)
        counts = np.zeros(n_steps, dtype=float)
        counts[unique_indices] += unique_counts

        non_zero = counts > 0
        mus = np.arange(n_steps, dtype=float)[non_zero]
        counts_non_zero = counts[non_zero]
        n_trials = float(np.sum(counts))
        weights = counts_non_zero / n_trials
        mean_estimate = mus @ weights
        sigma_estimate = np.sqrt(
            ((mus - mean_estimate) ** 2) @ counts_non_zero / max(1.0, n_trials - 1.0)
        )

        cumulative_counts = np.cumsum(counts_non_zero)
        idx_q25 = int(np.searchsorted(cumulative_counts, int(n_trials) // 4, side="left"))
        idx_q75 = int(np.searchsorted(cumulative_counts, int(n_trials) * 3 // 4, side="right"))
        interquantile_range = mus[min(mus.size - 1, idx_q75)] - mus[idx_q25]
        sigma_estimate = (
            1.059 * min(interquantile_range / 1.34, sigma_estimate) * n_trials ** (-0.2)
        )
        sigma_estimate = max(float(sigma_estimate), 0.5 / 1.64)

        sigmas = np.full(mus.shape, sigma_estimate, dtype=float)
        mus = np.append(mus, 0.5 * (n_steps - 1))
        sigmas = np.append(sigmas, float(n_steps))
        mixture_weights = np.append(counts_non_zero, self._prior_weight)
        mixture_weights /= np.sum(mixture_weights)

        samples = np.arange(n_steps, dtype=float)[:, np.newaxis]
        component_mass = self._normal_mass(
            (samples - 0.5 - mus[np.newaxis, :]) / sigmas[np.newaxis, :],
            (samples + 0.5 - mus[np.newaxis, :]) / sigmas[np.newaxis, :],
        )
        normalization = self._normal_mass(
            (-0.5 - mus) / sigmas,
            (n_steps - 0.5 - mus) / sigmas,
        )
        component_pdfs = component_mass / normalization[np.newaxis, :]
        return component_pdfs @ mixture_weights

    def _categorical_pdf(
        self,
        values: np.ndarray,
        choices: Tuple,
    ) -> np.ndarray:
        """Estimate a categorical density using Optuna-compatible kernels."""
        choice_to_index = {choice: index for index, choice in enumerate(choices)}
        indices = np.asarray([choice_to_index[value] for value in values], dtype=int)
        unique_indices, unique_counts = np.unique(indices, return_counts=True)
        n_choices = len(choices)
        n_kernels = len(unique_indices) + 1

        kernels = np.full(
            (n_kernels, n_choices),
            fill_value=self._prior_weight / n_kernels,
            dtype=float,
        )
        kernels[np.arange(len(unique_indices)), unique_indices] += 1.0
        kernels /= np.sum(kernels, axis=1, keepdims=True)

        mixture_weights = np.append(unique_counts.astype(float), self._prior_weight)
        mixture_weights /= np.sum(mixture_weights)
        return mixture_weights @ kernels

    def _pdf(self, values: np.ndarray, hp_name: str) -> np.ndarray:
        """Estimate the density for one supported ConfigSpace hyperparameter."""
        hp = self.cs[hp_name]
        if isinstance(hp, CategoricalHyperparameter):
            return self._categorical_pdf(values, tuple(hp.choices))
        if isinstance(hp, OrdinalHyperparameter):
            return self._categorical_pdf(values, tuple(hp.sequence))
        if isinstance(hp, (FloatHyperparameter, IntegerHyperparameter)):
            return self._numerical_pdf(values, hp)
        raise TypeError(f"Unsupported hyperparameter type for PED-ANOVA: {type(hp)}")

    def calculate(
        self,
        objective: Objective,
        budget: Optional[Union[int, float]] = None,
        target_quantile: float = 0.1,
        region_quantile: float = 1.0,
    ) -> None:
        """Calculate PED-ANOVA importance scores for one objective and budget.

        Parameters
        ----------
        objective : Objective
            Objective to analyze.
        budget : Optional[Union[int, float]], optional
            Budget to analyze. The highest budget is used if no budget is specified.
        target_quantile : float, optional
            Fraction of best trials defining the target region. By default 0.1.
        region_quantile : float, optional
            Fraction of best trials defining the reference region. By default 1.0.

        Raises
        ------
        RuntimeError
            If the quantiles are invalid or fewer than two suitable trials are available.
        """
        if not 0.0 < target_quantile < region_quantile <= 1.0:
            raise RuntimeError("PED-ANOVA requires 0 < target_quantile < region_quantile <= 1.")

        if budget is None:
            budget = self.run.get_highest_budget()

        data = self.run.get_encoded_data(
            objective,
            budget,
            statuses=Status.SUCCESS,
            include_config_ids=True,
        )
        objective_values = data[objective.name].to_numpy(dtype=float)
        config_ids = data["config_id"].to_numpy(dtype=int)

        active_rows = []
        for row, config_id in enumerate(config_ids):
            config = self.run.configs[int(config_id)]
            if all(hp_name in config for hp_name in self.hp_names):
                active_rows.append(row)

        if len(active_rows) == 0:
            raise RuntimeError("PED-ANOVA found no trials containing all selected hyperparameters.")

        active_rows_array = np.asarray(active_rows, dtype=int)
        objective_values = objective_values[active_rows_array]
        finite = np.isfinite(objective_values)
        objective_values = objective_values[finite]
        config_ids = config_ids[active_rows_array][finite]

        if objective_values.size < self._min_n_top_trials:
            raise RuntimeError(
                "PED-ANOVA requires at least " f"{self._min_n_top_trials} completed finite trials."
            )

        if objective_values.size <= self._min_n_top_trials:
            equal_importance = 1.0 / len(self.hp_names)
            self._importances = {hp_name: equal_importance for hp_name in self.hp_names}
            return

        is_lower_better = objective.optimize == "lower"
        target_mask = self._get_top_mask(
            objective_values,
            target_quantile,
            is_lower_better,
        )
        region_mask = self._get_top_mask(
            objective_values,
            region_quantile,
            is_lower_better,
        )
        relative_quantile = float(np.sum(target_mask) / np.sum(region_mask))

        raw_importances = {}
        for hp_name in self.hp_names:
            hp = self.cs[hp_name]
            if isinstance(hp, Constant):
                raw_importances[hp_name] = 0.0
                continue

            values = np.asarray(
                [self.run.configs[int(config_id)][hp_name] for config_id in config_ids]
            )
            target_pdf = self._pdf(values[target_mask], hp_name) + 1e-12
            region_pdf = self._pdf(values[region_mask], hp_name) + 1e-12
            divergence = float(region_pdf @ ((target_pdf / region_pdf - 1.0) ** 2))
            raw_importances[hp_name] = relative_quantile**2 * divergence

        total_importance = float(sum(raw_importances.values()))
        if total_importance == 0.0:
            equal_importance = 1.0 / len(raw_importances)
            self._importances = {hp_name: equal_importance for hp_name in raw_importances}
        else:
            self._importances = {
                hp_name: importance / total_importance
                for hp_name, importance in raw_importances.items()
            }

        self._importances = dict(
            sorted(self._importances.items(), key=lambda item: item[1], reverse=True)
        )

    def get_importances(
        self,
        hp_names: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Return normalized importance and zero estimator variance per hyperparameter."""
        if hp_names is None:
            hp_names = self.hp_names

        if len(self._importances) == 0:
            raise RuntimeError("Call calculate before requesting PED-ANOVA importances.")

        unknown_hp_names = set(hp_names) - set(self.hp_names)
        if len(unknown_hp_names) > 0:
            raise ValueError(f"Unknown hyperparameters: {sorted(unknown_hp_names)}")

        return {hp_name: (float(self._importances[hp_name]), 0.0) for hp_name in hp_names}
