# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List

import json
from copy import deepcopy
from unittest.mock import MagicMock, patch

import ConfigSpace
import pytest

from deepcave.plugins.hyperparameter.importances import Importances
from deepcave.runs.converters.smac3v1 import SMAC3v1Run


def _dependency_run() -> MagicMock:
    configspace = ConfigSpace.ConfigurationSpace()
    for name in ["a", "b", "c", "d", "e"]:
        configspace.add(ConfigSpace.Float(name, bounds=(0.0, 1.0)))

    run = MagicMock()
    run.configspace = configspace
    run.get_objective_names.return_value = ["loss", "time"]
    run.get_objective_ids.return_value = [0, 1]
    run.get_budgets.return_value = [1.0]
    run.get_budget_ids.return_value = [0]
    return run


def _dependency_inputs(plugin: Importances) -> Dict[str, Dict[str, Any]]:
    inputs = plugin.load_inputs()
    inputs.update(
        {
            "objective_id1": {"value": 0},
            "objective_id2": {"value": -1},
        }
    )
    return inputs


def _record(members: List[str], importance: float, variance: float = 0.0) -> Dict[str, Any]:
    return {
        "members": members,
        "order": len(members),
        "importance": importance,
        "variance": variance,
        "total_importance": importance,
        "total_variance": variance,
    }


def test_interaction_dependency_inputs_and_selection_limit() -> None:
    plugin = Importances()
    run = _dependency_run()
    inputs = _dependency_inputs(plugin)
    inputs["method"]["value"] = "global"
    inputs["interaction_order"]["value"] = 2
    inputs["interaction_hp_names"]["value"] = ["e", "d", "c", "b", "a"]

    updated = plugin.load_dependency_inputs(run, {}, inputs)

    assert updated["interaction_order"] == {
        "options": [
            {"label": "1 — Main effects", "value": 1},
            {"label": "2 — Main effects + pairwise interactions", "value": 2},
        ],
        "value": 2,
        "disabled": False,
    }
    assert updated["interaction_hp_names"]["value"] == ["a", "b", "c", "d"]
    assert updated["interaction_hp_names"]["disabled"] is False
    assert next(
        option for option in updated["interaction_hp_names"]["options"] if option["value"] == "e"
    )["disabled"]

    inputs["method"]["value"] = "local"
    updated = plugin.load_dependency_inputs(run, {}, inputs)
    assert updated["interaction_order"]["value"] == 1
    assert updated["interaction_order"]["disabled"] is True
    assert updated["interaction_hp_names"]["value"] == []
    assert updated["interaction_hp_names"]["disabled"] is True


def test_interaction_inputs_are_part_of_processing_cache_key() -> None:
    plugin = Importances()
    inputs = {
        "interaction_order": {"value": 2},
        "interaction_hp_names": {"value": ["a", "b"]},
        "hyperparameter_names": {"value": ["a", "b"]},
    }
    changed_interactions = deepcopy(inputs)
    changed_interactions["interaction_hp_names"]["value"] = ["a", "c"]
    changed_filter = deepcopy(inputs)
    changed_filter["hyperparameter_names"]["value"] = ["a"]

    get_key = Importances._dict_as_key.__wrapped__
    key = get_key(plugin, inputs, remove_filters=True)

    assert key != get_key(plugin, changed_interactions, remove_filters=True)
    assert key == get_key(plugin, changed_filter, remove_filters=True)


def test_process_serializes_main_and_pairwise_importances() -> None:
    run = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
    hp_names = list(run.configspace.keys())
    inputs = {
        "objective_id1": 0,
        "objective_id2": -1,
        "method": "global",
        "n_trees": 4,
        "interaction_order": 2,
        "interaction_hp_names": hp_names[:2],
    }

    outputs = Importances.process(run, inputs)
    records = outputs[0]

    assert len([record for record in records if record["order"] == 1]) == len(hp_names)
    pairs = [record for record in records if record["order"] == 2]
    assert len(pairs) == 1
    assert pairs[0]["members"] == hp_names[:2]
    assert pairs[0]["importance"] <= pairs[0]["total_importance"]
    json.dumps(outputs)


@pytest.mark.parametrize(
    ("method", "objective_id2", "interaction_hp_names", "message"),
    [
        ("local", -1, ["alpha", "batch_size"], "only available"),
        ("global", 1, ["alpha", "batch_size"], "only available"),
        ("global", -1, ["alpha"], "between two and four"),
        (
            "global",
            -1,
            ["alpha", "batch_size", "depth", "learning_rate_init", "unknown"],
            "between two and four",
        ),
    ],
)
def test_process_rejects_invalid_interaction_inputs(
    method: str, objective_id2: int, interaction_hp_names: List[str], message: str
) -> None:
    run = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
    inputs = {
        "objective_id1": 0,
        "objective_id2": objective_id2,
        "method": method,
        "n_trees": 4,
        "interaction_order": 2,
        "interaction_hp_names": interaction_hp_names,
    }

    with pytest.raises(RuntimeError, match=message):
        Importances.process(run, inputs)


def test_global_plot_aligns_effects_across_budgets_and_filters_pairs() -> None:
    outputs = {
        0: [
            _record(["b"], 0.2),
            _record(["a"], 0.8),
            _record(["a", "b"], 0.3),
        ],
        1: [
            _record(["a", "b"], 0.9),
            _record(["a"], 0.4),
            _record(["b"], 0.1),
        ],
    }
    run = MagicMock()
    run.get_budget.side_effect = lambda budget_id, human: f"budget-{budget_id}"
    inputs = {
        "objective_id2": -1,
        "method": "global",
        "hyperparameter_names": ["a", "b"],
        "budget_ids": [0, 1],
        "n_hps": 3,
    }

    with patch("deepcave.plugins.hyperparameter.importances.save_image"):
        figure = Importances.load_outputs(run, inputs, outputs)

    assert list(figure.data[0].x) == ["a × b", "a", "b"]
    assert list(figure.data[0].y) == [0.3, 0.8, 0.2]
    assert list(figure.data[1].y) == [0.9, 0.4, 0.1]

    inputs["hyperparameter_names"] = ["a"]
    with patch("deepcave.plugins.hyperparameter.importances.save_image"):
        filtered_figure = Importances.load_outputs(run, inputs, outputs)
    assert list(filtered_figure.data[0].x) == ["a"]
