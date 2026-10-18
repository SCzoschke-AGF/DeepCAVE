from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from ConfigSpace import ConfigurationSpace

from deepcave.plugins.hyperparameter import pdp
from deepcave.plugins.hyperparameter.pdp import PartialDependencies
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective


def test_controls_and_dependency_inputs():
    plugin = PartialDependencies()
    registered = {name for name, *_ in plugin.inputs}
    assert "budget_id" not in registered
    assert "show_confidence" not in registered
    assert "show_ice" in registered
    assert "show_confidence" not in plugin.load_inputs()

    run = DeepCAVERun(
        "pdp",
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}),
        objectives=[Objective("loss")],
    )
    result = plugin.load_dependency_inputs(
        run, {}, {"objective_id": {"value": None}, "hyperparameter_name_1": {"value": None}}
    )
    assert result["hyperparameter_name_1"]["value"] == "x"
    assert "budget_id" not in result


@pytest.mark.parametrize("show_ice", [False, True])
def test_confidence_is_enabled_only_without_ice(monkeypatch, show_ice):
    run = DeepCAVERun(
        "pdp",
        configspace=ConfigurationSpace(space={"x": (0.0, 1.0)}),
        objectives=[Objective("loss")],
    )
    monkeypatch.setattr(pdp, "save_image", lambda *_: None)
    outputs = {
        "x": [[0.0], [1.0]],
        "y": [1.0, 2.0],
        "variances": [0.25, 0.25],
        "x_ice": [[[0.0], [1.0]]],
        "y_ice": [[0.5, 2.5]],
    }
    figure = PartialDependencies.load_outputs(
        run,
        {
            "objective_id": 0,
            "hyperparameter_name_1": "x",
            "hyperparameter_name_2": None,
            "show_ice": show_ice,
        },
        outputs,
    )
    assert len(figure.data) == (2 if show_ice else 3)
    assert any(trace.fill == "tonexty" for trace in figure.data) is (not show_ice)
    np.testing.assert_array_equal(figure.data[-1].y, outputs["y"])
    if show_ice:
        np.testing.assert_array_equal(figure.data[0].y, outputs["y_ice"][0])


@pytest.mark.parametrize("budgets", [[1], [1, 2, -1]])
def test_process_uses_default_budget_without_budget_input(monkeypatch, budgets):
    run = Mock()
    run.configspace = ConfigurationSpace(space={"x": (0.0, 1.0)})
    run.get_budgets.return_value = budgets
    objective = Objective("loss")
    run.get_objective.return_value = objective
    run.get_encoded_data.return_value = pd.DataFrame({"x": [0.0, 1.0], "loss": [1.0, 2.0]})
    monkeypatch.setattr(pdp, "RandomForestSurrogate", Mock())
    result = Mock()
    result.x_pdp = result.y_pdp = result.y_variances = np.array([])
    result._ice.x_ice = result._ice.y_ice = np.array([])
    monkeypatch.setattr(pdp.PDP, "from_random_points", Mock(return_value=result))

    PartialDependencies.process(
        run, {"objective_id": 0, "hyperparameter_name_1": "x", "hyperparameter_name_2": None}
    )
    assert run.get_encoded_data.call_args.args == (objective, budgets[-1])
