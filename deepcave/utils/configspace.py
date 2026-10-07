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
# ConfigSpace

This module samples random as well as border configurations.
"""

from typing import Any, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    IntegerHyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import deactivate_inactive_hyperparameters


ConfigurationLike = Union[Configuration, Mapping[str, Any]]


def normalized_configuration_matrix(
    configs: Sequence[ConfigurationLike],
    configspace: ConfigurationSpace,
) -> np.ndarray:
    """Encode configurations for mixed-type distance calculations.

    ConfigSpace already maps numerical hyperparameters to its vector space, including
    logarithmic transformations. Ordinals are additionally scaled to the unit interval;
    categorical vector values remain integer codes so equality can be evaluated explicitly.
    Inactive hyperparameters remain ``NaN`` and are handled by
    :func:`normalized_configuration_distance_components`.
    """
    hyperparameters = list(configspace.values())
    matrix = np.empty((len(configs), len(hyperparameters)), dtype=float)

    for row, config in enumerate(configs):
        if not isinstance(config, Configuration):
            config = Configuration(configspace, values=dict(config))
        values = np.asarray(config.get_array(), dtype=float).copy()
        for column, hyperparameter in enumerate(hyperparameters):
            if isinstance(hyperparameter, Constant):
                values[column] = 0.0
            elif isinstance(hyperparameter, OrdinalHyperparameter) and np.isfinite(
                values[column]
            ):
                denominator = max(1, len(hyperparameter.sequence) - 1)
                values[column] /= denominator
        matrix[row] = values

    return matrix


def normalized_configuration_distance_components(
    reference: np.ndarray,
    configs: np.ndarray,
    configspace: ConfigurationSpace,
) -> np.ndarray:
    """Return one normalized distance contribution per configuration and hyperparameter.

    Two inactive values contribute zero, while an active/inactive mismatch contributes one.
    Constants never contribute. Categorical parameters use a binary mismatch and all remaining
    parameter types use their absolute distance in ConfigSpace's normalized vector space.
    """
    reference_array = np.asarray(reference, dtype=float)
    config_array = np.asarray(configs, dtype=float)
    if reference_array.ndim != 1:
        raise ValueError("The reference configuration must be a one-dimensional vector.")
    if config_array.ndim != 2 or config_array.shape[1] != reference_array.shape[0]:
        raise ValueError("Configuration matrix and reference vector have incompatible shapes.")
    if config_array.shape[1] != len(configspace):
        raise ValueError("Configuration vectors do not match the ConfigSpace dimensions.")

    components = np.zeros_like(config_array, dtype=float)
    reference_active = np.isfinite(reference_array)
    configs_active = np.isfinite(config_array)

    for column, hyperparameter in enumerate(configspace.values()):
        if isinstance(hyperparameter, Constant):
            continue

        exactly_one_active = np.logical_xor(reference_active[column], configs_active[:, column])
        components[exactly_one_active, column] = 1.0
        both_active = reference_active[column] & configs_active[:, column]
        if not np.any(both_active):
            continue

        if isinstance(hyperparameter, CategoricalHyperparameter):
            components[both_active, column] = (
                config_array[both_active, column] != reference_array[column]
            ).astype(float)
        else:
            components[both_active, column] = np.abs(
                config_array[both_active, column] - reference_array[column]
            )

    return components


def configuration_euclidean_distances(
    components: np.ndarray,
    configspace: ConfigurationSpace,
) -> np.ndarray:
    """Aggregate normalized component differences to a Euclidean distance.

    Each component remains normalized to the unit interval, but the aggregate is not divided by
    the number of non-constant dimensions.
    """
    component_array = np.asarray(components, dtype=float)
    if component_array.ndim != 2 or component_array.shape[1] != len(configspace):
        raise ValueError("Distance components do not match the ConfigSpace dimensions.")
    return np.sqrt(np.sum(np.square(component_array), axis=1))


def compute_configuration_distances(
    reference_config: ConfigurationLike,
    configs: Sequence[ConfigurationLike],
    configspace: ConfigurationSpace,
) -> np.ndarray:
    """Compute Euclidean distances between normalized configuration components."""
    normalized = normalized_configuration_matrix(
        [reference_config, *configs],
        configspace,
    )
    components = normalized_configuration_distance_components(
        normalized[0],
        normalized[1:],
        configspace,
    )
    return configuration_euclidean_distances(components, configspace)


def sample_border_config(configspace: ConfigurationSpace) -> Iterator[Configuration]:
    """
    Generate border configurations from the configuration space.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The configuration space from which the hyperparameters are drawn from.

    Yields
    ------
    Iterator[Configuration]
        Border configurations.
    """
    rng = np.random.RandomState(0)

    while True:
        config = {}
        # Iterates over the hyperparameters to get considered values
        for hp_name, hp in configspace.items():
            if isinstance(hp, CategoricalHyperparameter):
                borders = list(hp.choices)
            elif isinstance(hp, Constant):
                borders = [hp.value]
            elif isinstance(hp, OrdinalHyperparameter):
                borders = [hp.sequence[0], hp.sequence[-1]]
            else:
                assert isinstance(hp, NumericalHyperparameter)
                borders = [hp.lower, hp.upper]

            # Get a random choice
            value = rng.choice(borders)
            config[hp_name] = value

        try:
            configuration = deactivate_inactive_hyperparameters(config, configspace)
            configuration.check_valid_configuration()
        except Exception:
            continue

        yield configuration


def sample_random_config(
    configspace: ConfigurationSpace, d: Optional[int] = None
) -> Iterator[Configuration]:
    """
    Generate random configurations from the configuration space.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The configspace from which the hyperparameters are drawn from.
    d : Optional[int], optional
        The possible hyperparameter values can be reduced by this argument as the range gets
        discretized. For example, an integer or float hyperparameter has only four possible
        values if d=4. By default, None (no discretization is done).

    Yields
    ------
    Iterator[Configuration]
        Random configurations.
    """
    if d is None:
        for config in configspace.sample_configuration(99999):
            yield config

        return

    rng = np.random.RandomState(0)

    while True:
        config_dict = {}

        # Iterates over the hyperparameters to get considered values
        for hp_name, hp in configspace.items():
            if isinstance(hp, CategoricalHyperparameter):
                values = list(hp.choices)
            elif isinstance(hp, Constant):
                values = [hp.value]
            elif isinstance(hp, OrdinalHyperparameter):
                values = list(hp.sequence)
            else:
                assert isinstance(hp, NumericalHyperparameter)
                if hp.log:
                    values = []
                    value = hp.lower
                    while value < hp.upper:
                        values += [value]
                        value = value * 10

                    if hp.upper not in values:
                        values += [hp.upper]
                else:
                    values = list(np.linspace(hp.lower, hp.upper, d))

                if isinstance(hp, IntegerHyperparameter):
                    values = [int(i) for i in values]

            # Get a random choice
            value = rng.choice(values)
            config_dict[hp_name] = value

        try:
            configuration = deactivate_inactive_hyperparameters(config_dict, configspace)
            configuration.check_valid_configuration()
        except Exception:
            continue

        yield configuration
