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

import unittest

import ConfigSpace
import pytest

from deepcave.runs import AbstractRun, check_equality
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.converters.smac3v1 import SMAC3v1Run
from deepcave.runs.converters.smac3v2 import SMAC3v2Run
from deepcave.runs.exceptions import NotMergeableError
from deepcave.runs.group import Group
from deepcave.runs.objective import Objective


def _make_run(name, hyperparameter, value, model_family="xgboost"):
    configspace = ConfigSpace.ConfigurationSpace()
    configspace.add(hyperparameter)
    run = DeepCAVERun(
        name,
        configspace=configspace,
        objectives=Objective("score", lower=0.0, upper=1.0, optimize="upper"),
        meta={"model_family": model_family},
    )
    run.add(costs=0.5, config={hyperparameter.name: value}, seed=0, budget=0.0)
    return run


class TestRun(unittest.TestCase):
    def setUp(self) -> None:
        # Initiate run here
        self.run1: AbstractRun = SMAC3v1Run.from_path(
            "logs/SMAC3v1/outlier-detection/SMAC-pendigits-015-0-0"
        )
        self.run2: AbstractRun = SMAC3v1Run.from_path(
            "logs/SMAC3v1/outlier-detection/DEHB-pendigits-015-0-25"
        )
        self.run3: AbstractRun = SMAC3v1Run.from_path(
            "logs/SMAC3v1/outlier-detection/SMAC-pendigits-015-0-50"
        )
        self.run1_v2: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_1")
        self.run2_v2: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_2")
        self.run3_v2: AbstractRun = SMAC3v2Run.from_path("logs/SMAC3v2/mlp/run_3")

    def test_group_v1(self) -> None:
        group1 = Group("blub", runs=[self.run1, self.run2])
        check_equality([self.run1, group1])

        group2 = Group("blub", runs=[self.run1, self.run2, self.run3])
        check_equality([self.run1, self.run2, self.run3, group2])

    def test_group_v2(self) -> None:
        group1 = Group("blub", runs=[self.run1_v2, self.run2_v2])
        check_equality([self.run1_v2, group1])

        group2 = Group("blub", runs=[self.run1_v2, self.run2_v2, self.run3_v2])
        check_equality([self.run1_v2, self.run2_v2, self.run3_v2, group2])


def test_relaxed_group_combines_numeric_search_space_ranges() -> None:
    run1 = _make_run(
        "run1",
        ConfigSpace.Float("learning_rate", bounds=(0.005, 0.3), log=True),
        0.2,
    )
    run2 = _make_run(
        "run2",
        ConfigSpace.Float("learning_rate", bounds=(0.003, 0.05), log=True),
        0.01,
    )

    with pytest.raises(NotMergeableError, match="Configspace of runs are not equal"):
        Group("strict", [run1, run2])

    group = Group("relaxed", [run1, run2], require_configspace_congruence=False)

    hyperparameter = group.configspace["learning_rate"]
    assert hyperparameter.lower == pytest.approx(0.003)
    assert hyperparameter.upper == pytest.approx(0.3)
    assert hyperparameter.log is True
    assert group.meta["require_configspace_congruence"] is False
    assert len(group.history) == 2
    values = sorted(group.get_config(index)["learning_rate"] for index in group.configs)
    assert values == pytest.approx([0.01, 0.2])


def test_relaxed_group_still_requires_matching_parameter_types() -> None:
    run1 = _make_run("run1", ConfigSpace.Float("depth", bounds=(1.0, 10.0)), 4.0)
    run2 = _make_run("run2", ConfigSpace.Integer("depth", bounds=(1, 10)), 4)

    with pytest.raises(NotMergeableError, match="same type"):
        Group("relaxed", [run1, run2], require_configspace_congruence=False)


def test_relaxed_group_uses_optuna_model_family_when_available() -> None:
    run1 = _make_run(
        "run1",
        ConfigSpace.Float("rate", bounds=(0.1, 1.0)),
        0.5,
        model_family="xgboost",
    )
    run2 = _make_run(
        "run2",
        ConfigSpace.Float("rate", bounds=(0.2, 1.0)),
        0.5,
        model_family="catboost",
    )

    with pytest.raises(NotMergeableError, match="Model families of runs are not equal"):
        Group("relaxed", [run1, run2], require_configspace_congruence=False)


if __name__ == "__main__":
    unittest.main()
