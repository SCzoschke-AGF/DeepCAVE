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
import numpy as np
import pytest

from deepcave.plugins.hyperparameter import importances as importances_module
from deepcave.plugins.hyperparameter.importances import Importances
from deepcave.plugins.static import _processing_inputs_key
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.converters.smac3v1 import SMAC3v1Run
from deepcave.runs.objective import Objective
from deepcave.runs.status import Status


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


def test_initial_inputs_are_safe_for_dependency_loading() -> None:
    plugin = Importances()

    updated = plugin.load_dependency_inputs(_dependency_run(), {}, plugin.load_inputs())

    assert updated["objective_id1"]["value"] == 0
    assert updated["objective_id2"]["value"] == -1

    inputs = plugin.load_inputs()
    inputs["objective_id1"]["value"] = 1
    inputs["objective_id2"]["value"] = 0
    inputs["hyperparameter_names"]["value"] = ["a"]
    updated = plugin.load_dependency_inputs(_dependency_run(), {}, inputs)
    assert updated["objective_id1"]["value"] == 1
    assert updated["objective_id2"]["value"] == -1
    assert updated["hyperparameter_names"]["value"] == list("abcde")


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


def test_ped_anova_dependency_inputs_disable_multi_objective_controls() -> None:
    plugin = Importances()
    run = _dependency_run()
    inputs = _dependency_inputs(plugin)
    inputs["method"]["value"] = "ped_anova"
    inputs["objective_id2"]["value"] = 1

    updated = plugin.load_dependency_inputs(run, {}, inputs)

    assert updated["objective_id2"]["value"] == -1
    assert updated["objective_id2"]["disabled"] is True
    assert updated["n_trees"]["disabled"] is True
    assert updated["target_quantile"] == {"value": 0.1, "disabled": False}
    assert updated["interaction_order"]["disabled"] is True
    assert updated["interaction_hp_names"]["disabled"] is True


def test_ped_anova_target_quantile_is_part_of_processing_cache_key() -> None:
    plugin = Importances()
    inputs = {
        "target_quantile": {"value": 0.1},
        "hyperparameter_names": {"value": ["a", "b"]},
    }
    changed_target = deepcopy(inputs)
    changed_target["target_quantile"]["value"] = 0.2

    get_key = Importances._dict_as_key.__wrapped__
    key = get_key(plugin, inputs, remove_filters=True)

    assert key != get_key(plugin, changed_target, remove_filters=True)


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


@pytest.mark.parametrize("method", ["global", "local", "ped_anova"])
def test_importances_ignore_unsuccessful_and_invalid_trials(method: str) -> None:
    run = DeepCAVERun(
        "valid-trials",
        configspace=ConfigSpace.ConfigurationSpace(space={"x": (0.0, 1.0), "y": (0.0, 1.0)}),
        objectives=[Objective("loss", lower=0.0, upper=1.0)],
    )
    rng = np.random.RandomState(0)
    for x, y in rng.uniform(size=(30, 2)):
        run.add(config={"x": x, "y": y}, costs=0.7 * x + 0.3 * y, budget=0, seed=0)
    inputs = {
        "objective_id1": 0,
        "objective_id2": -1,
        "method": method,
        "n_trees": 4,
        "target_quantile": 0.2,
    }
    expected = Importances.process(run, inputs)
    dirty = deepcopy(run)
    excluded = [(status, 0.0) for status in Status if status != Status.SUCCESS] + [
        (Status.SUCCESS, value) for value in (None, np.nan, np.inf, -np.inf)
    ]
    for index, (status, value) in enumerate(excluded):
        dirty.add(
            config={"x": (index + 1) / 100, "y": 0.5},
            costs=[value],
            budget=0,
            seed=0,
            status=status,
        )
    history = dirty.history
    actual = Importances.process(dirty, inputs)
    assert actual == expected
    assert dirty.history is history
    assert len(dirty.history) == 30 + len(excluded)


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


def test_process_calculates_ped_anova_importances() -> None:
    run = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
    inputs = {
        "objective_id1": 0,
        "objective_id2": -1,
        "method": "ped_anova",
        "n_trees": None,
        "target_quantile": 0.1,
        "interaction_order": 1,
        "interaction_hp_names": [],
    }

    outputs = Importances.process(run, inputs)
    importances = outputs[0]

    assert set(importances) == set(run.configspace.keys())
    assert sum(importance for importance, _ in importances.values()) == pytest.approx(1.0)
    assert all(variance == 0.0 for _, variance in importances.values())
    json.dumps(outputs)


def test_process_rejects_multi_objective_ped_anova() -> None:
    run = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
    inputs = {
        "objective_id1": 0,
        "objective_id2": 0,
        "method": "ped_anova",
        "n_trees": None,
        "target_quantile": 0.1,
        "interaction_order": 1,
        "interaction_hp_names": [],
    }

    with pytest.raises(RuntimeError, match="single-objective"):
        Importances.process(run, inputs)


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


def test_global_plot_aligns_effects_across_budgets_and_ignores_old_parameter_selection() -> None:
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
        figure = Importances.load_outputs(run, inputs, outputs)["graph"]["figure"]

    assert list(figure.data[0].y) == ["a × b", "a", "b"]
    assert list(figure.data[0].x) == [0.3, 0.8, 0.2]
    assert list(figure.data[1].x) == [0.9, 0.4, 0.1]

    inputs["hyperparameter_names"] = ["a"]
    with patch("deepcave.plugins.hyperparameter.importances.save_image"):
        filtered_figure = Importances.load_outputs(run, inputs, outputs)["graph"]["figure"]
    assert list(filtered_figure.data[0].y) == ["a × b", "a", "b"]


def test_screening_then_selected_exact_pairs_reuse_fitted_forest(tmp_path, monkeypatch):
    run = DeepCAVERun(
        "screening",
        configspace=ConfigSpace.ConfigurationSpace(space={name: (0.0, 1.0) for name in "abcde"}),
        objectives=[Objective("loss", lower=-1.0, upper=1.0)],
    )
    for values in np.random.RandomState(0).uniform(size=(70, 5)):
        run.add(
            config=dict(zip("abcde", values)),
            costs=(values[0] - 0.5) * (values[1] - 0.5) + 0.1 * values[2],
            budget=0,
            seed=0,
        )
    original_cache = importances_module.get_cached_fanova
    monkeypatch.setattr(
        importances_module,
        "get_cached_fanova",
        lambda run, objective, budget, trees, directory: original_cache(
            run, objective, budget, trees, tmp_path
        ),
    )
    fits = []
    original_calculate = importances_module.GlobalEvaluator.calculate

    def calculate(self, *args, **kwargs):
        fits.append(1)
        return original_calculate(self, *args, **kwargs)

    monkeypatch.setattr(importances_module.GlobalEvaluator, "calculate", calculate)
    monkeypatch.setattr(importances_module, "save_image", lambda *_: None)
    inputs = {
        "objective_id1": 0,
        "objective_id2": -1,
        "method": "global",
        "n_trees": 3,
        "interaction_order": 2,
        "interaction_mode": "screening",
        "screen_top_k": 0,
    }
    first = Importances.process(run, inputs)
    assert len(first["_screening"][0]) == 10
    assert all(record["order"] == 1 for record in first[0])
    assert all(row["exact"] is None for row in first["_screening"][0])
    additional = first["_screening"][0][-1]["members"]
    inputs.update(screen_top_k=1, exact_pairs=[json.dumps(additional)])
    second = Importances.process(run, inputs)
    assert fits == [1]
    exact_pairs = {tuple(row["members"]) for row in second[0] if row["order"] == 2}
    assert exact_pairs == {tuple(additional), tuple(second["_screening"][0][0]["members"])}
    assert sum(row["exact"] is not None for row in second["_screening"][0]) == len(exact_pairs)
    json.dumps(second)
    rendered = Importances.load_outputs(
        run,
        {**inputs, "hyperparameter_names": list("abcde"), "budget_ids": [0], "n_hps": 20},
        second,
    )
    assert rendered["screening_panel"]["style"] == {}
    assert len(rendered["screening"]["data"]) == 10
    assert len(rendered["graph"]["figure"].data[0].x) == 5 + len(exact_pairs)
    # Clearing a pending selection must not discard exact calculations on the same forest.
    third = Importances.process(run, {**inputs, "screen_top_k": 0, "exact_pairs": []})
    assert {tuple(row["members"]) for row in third[0] if row["order"] == 2} == exact_pairs
    assert fits == [1]


@pytest.mark.parametrize(
    "exact,expected_status",
    [
        (0.044465, "Exact"),
        (0.0, "Exact"),
        (None, "Refine recommended"),
        (np.nan, "Refine recommended"),
        (np.inf, "Refine recommended"),
    ],
)
def test_cached_exact_result_overrides_screening_status(exact, expected_status, monkeypatch):
    row = {"members": ["a", "b"], "exact": exact, "status": "Refine recommended"}
    outputs = {"0": [], "_screening": {"0": [row]}}
    inputs = {"budget_ids": [0], "hyperparameter_names": ["a", "b"]}
    monkeypatch.setattr(Importances, "_importance_figure", lambda *_: None)

    rendered = Importances.load_outputs(MagicMock(), inputs, outputs)

    assert rendered["screening"]["data"][0]["status"] == expected_status
    assert row["status"] == "Refine recommended"
    assert rendered["screening"]["data"][0]["id"] == json.dumps([0, "a", "b"])


def test_table_click_adds_pair_once_using_row_id_instead_of_page_position():
    plugin = Importances()
    plugin.previous_inputs = _dependency_inputs(plugin)
    pair = json.dumps(["c", "e"])
    event = {
        "screening_cell": {
            "column_id": "pair",
            "row": 0,
            "row_id": json.dumps([2, "c", "e"]),
        }
    }

    updated = plugin.load_input_updates(event)

    assert updated == {"exact_pairs": {"value": [pair]}}
    plugin.previous_inputs["exact_pairs"].update(updated["exact_pairs"])
    assert plugin.load_input_updates(event) == updated
    event["screening_cell"]["column_id"] = "score"
    assert plugin.load_input_updates(event) == {}


def test_completed_pairs_leave_picker_without_changing_applied_requests():
    plugin = Importances()
    run = _dependency_run()
    run.id = "study"
    inputs = _dependency_inputs(plugin)
    inputs["run"] = {"value": run.id}
    completed, pending = json.dumps(["a", "b"]), json.dumps(["c", "d"])
    inputs["exact_pairs"]["value"] = [completed, pending]
    inputs["pending_exact_pairs"]["value"] = [completed, pending]
    plugin.raw_outputs = {
        run.id: {
            "_screening_context": {"objective_id1": 0, "n_trees": 10},
            "_screening": {"0": [{"members": ["a", "b"], "exact": 0.0}]},
        }
    }

    updated = plugin.load_dependency_inputs(run, deepcopy(inputs), inputs)

    assert completed not in {
        option["value"] for option in updated["pending_exact_pairs"]["options"]
    }
    assert updated["pending_exact_pairs"]["value"] == [pending]
    assert updated["exact_pairs"]["value"] == [completed, pending]
    previous = deepcopy(inputs)
    for key, value in updated.items():
        inputs[key].update(value)
    assert _processing_inputs_key(plugin, inputs) == _processing_inputs_key(plugin, previous)

    plugin.previous_inputs = deepcopy(inputs)
    event = {"screening_cell": {"column_id": "pair", "row_id": json.dumps([0, "a", "b"])}}
    assert plugin.load_input_updates(event)["exact_pairs"]["value"] == [completed, pending]
    # Removing a pending chip preserves completed requests and removes only unprocessed work.
    inputs["pending_exact_pairs"]["value"] = []
    removed = plugin.load_dependency_inputs(run, plugin.previous_inputs, inputs)
    assert removed["exact_pairs"]["value"] == [completed]
    assert removed["pending_exact_pairs"]["value"] == []


@pytest.mark.parametrize("change", ["objective", "trees", "run", "partial_budget"])
def test_picker_keeps_pairs_available_when_exact_result_does_not_cover_context(change):
    plugin = Importances()
    run = _dependency_run()
    run.id = "study"
    inputs = _dependency_inputs(plugin)
    plugin.raw_outputs = {
        run.id: {
            "_screening_context": {"objective_id1": 0, "n_trees": 10},
            "_screening": {"0": [{"members": ["a", "b"], "exact": 0.1}]},
        }
    }
    if change == "objective":
        inputs["objective_id1"]["value"] = 1
    elif change == "trees":
        inputs["n_trees"]["value"] = 16
    elif change == "run":
        run.id = "another-study"
    else:
        plugin.raw_outputs[run.id]["_screening"]["1"] = [{"members": ["a", "b"], "exact": None}]

    updated = plugin.load_dependency_inputs(run, {}, inputs)

    assert json.dumps(["a", "b"]) in {
        option["value"] for option in updated["pending_exact_pairs"]["options"]
    }


def test_screening_controls_offer_all_pairs_without_main_effect_preselection():
    plugin = Importances()
    inputs = _dependency_inputs(plugin)
    inputs["method"]["value"] = "global"
    inputs["interaction_order"]["value"] = 2
    inputs["exact_pairs"]["value"] = [json.dumps(["a", "b"]), '["a", "missing"]']
    updated = plugin.load_dependency_inputs(_dependency_run(), {}, inputs)
    assert len(updated["exact_pairs"]["options"]) == 10
    assert updated["exact_pairs"]["value"] == [json.dumps(["a", "b"])]
    assert inputs["interaction_mode"]["value"] == "screening"


@pytest.mark.parametrize(
    "field,value",
    [("screen_top_k", 2), ("interaction_mode", "selected"), ("exact_pairs", ['["a", "b"]'])],
)
def test_screening_settings_are_part_of_processing_cache_key(field, value):
    plugin = Importances()
    first = plugin.load_inputs()
    changed = deepcopy(first)
    changed[field]["value"] = value
    get_key = Importances._dict_as_key.__wrapped__
    assert get_key(plugin, first, remove_filters=True) != get_key(
        plugin, changed, remove_filters=True
    )
