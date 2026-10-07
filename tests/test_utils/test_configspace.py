import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from deepcave.utils.configspace import (
    compute_configuration_distances,
    normalized_configuration_distance_components,
    normalized_configuration_matrix,
)


def test_normalized_configuration_distance_is_zero_and_symmetric():
    configspace = ConfigurationSpace(
        space={"x": (0.0, 10.0), "category": ["a", "b"]}
    )
    first = Configuration(configspace, values={"x": 2.0, "category": "a"})
    second = Configuration(configspace, values={"x": 8.0, "category": "b"})

    assert compute_configuration_distances(first, [first], configspace)[0] == 0
    forward = compute_configuration_distances(first, [second], configspace)[0]
    backward = compute_configuration_distances(second, [first], configspace)[0]
    assert forward == pytest.approx(backward)


def test_numerical_raw_scales_receive_equal_normalized_weight():
    configspace = ConfigurationSpace()
    configspace.add(
        [
            UniformFloatHyperparameter("wide", lower=0.0, upper=1000.0),
            UniformFloatHyperparameter("narrow", lower=0.0, upper=1.0),
        ]
    )
    reference = Configuration(configspace, values={"wide": 0.0, "narrow": 0.0})
    candidate = Configuration(configspace, values={"wide": 100.0, "narrow": 0.1})
    matrix = normalized_configuration_matrix([reference, candidate], configspace)
    components = normalized_configuration_distance_components(
        matrix[0], matrix[1:], configspace
    )

    assert components[0, 0] == pytest.approx(0.1)
    assert components[0, 1] == pytest.approx(0.1)
    assert compute_configuration_distances(reference, [candidate], configspace)[0] == pytest.approx(
        np.sqrt(0.1**2 + 0.1**2)
    )


def test_logarithmic_parameter_uses_log_space():
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("rate", lower=1.0, upper=100.0, log=True))
    reference = Configuration(configspace, values={"rate": 1.0})
    midpoint = Configuration(configspace, values={"rate": 10.0})

    assert compute_configuration_distances(reference, [midpoint], configspace)[0] == pytest.approx(
        0.5
    )


def test_categorical_ordinal_and_constant_components():
    configspace = ConfigurationSpace()
    configspace.add(
        [
            CategoricalHyperparameter("kind", choices=["a", "b", "c"]),
            OrdinalHyperparameter("size", sequence=["small", "medium", "large"]),
            Constant("fixed", value=7),
            UniformIntegerHyperparameter("depth", lower=1, upper=11),
        ]
    )
    reference = Configuration(
        configspace,
        values={"kind": "a", "size": "small", "fixed": 7, "depth": 1},
    )
    candidate = Configuration(
        configspace,
        values={"kind": "c", "size": "medium", "fixed": 7, "depth": 6},
    )
    matrix = normalized_configuration_matrix([reference, candidate], configspace)
    components = normalized_configuration_distance_components(
        matrix[0], matrix[1:], configspace
    )[0]
    by_name = dict(zip(configspace.keys(), components))

    assert by_name["kind"] == 1
    assert by_name["size"] == pytest.approx(0.5)
    assert by_name["fixed"] == 0
    assert by_name["depth"] == pytest.approx(0.5)


def test_inactive_parameters_never_produce_nan_distance():
    configspace = ConfigurationSpace()
    parent = CategoricalHyperparameter("model", choices=["a", "b"])
    child = UniformFloatHyperparameter("alpha", lower=0.0, upper=1.0)
    configspace.add([parent, child, EqualsCondition(child, parent, "a")])
    inactive_a = Configuration(configspace, values={"model": "b"})
    inactive_b = Configuration(configspace, values={"model": "b"})
    active = Configuration(configspace, values={"model": "a", "alpha": 0.5})

    both_inactive = compute_configuration_distances(inactive_a, [inactive_b], configspace)[0]
    one_inactive = compute_configuration_distances(inactive_a, [active], configspace)[0]

    assert both_inactive == 0
    assert np.isfinite(one_inactive)
    assert one_inactive == pytest.approx(np.sqrt(2.0))
