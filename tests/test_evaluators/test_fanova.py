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

from deepcave.evaluators.fanova import fANOVA as Evaluator
from deepcave.runs import AbstractRun
from deepcave.runs.converters.smac3v1 import SMAC3v1Run


class TestFanova(unittest.TestCase):
    def setUp(self):
        # Initiate run here
        self.run: AbstractRun = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
        self.hp_names = list(self.run.configspace.keys())
        self.evaluator = Evaluator(self.run)

    def test(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(objective, budget, seed=42)
        importances2 = self.evaluator.get_importances(self.hp_names)

        # Different seed: Different results
        assert importances["batch_size"][1] != importances2["batch_size"][1]

    def test_seed(self):
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)

        # Calculate
        self.evaluator.calculate(objective, budget, seed=0)
        importances = self.evaluator.get_importances(self.hp_names)

        self.evaluator.calculate(objective, budget, seed=0)
        importances2 = self.evaluator.get_importances(self.hp_names)

        # Same seed: Same results
        assert importances["batch_size"][1] == importances2["batch_size"][1]

    def test_pairwise_importances_after_main_effects(self) -> None:
        budget = self.run.get_budget(0)
        objective = self.run.get_objective(0)
        selected_hp_names = self.hp_names[:2]

        self.evaluator.calculate(objective, budget, n_trees=4, seed=0)
        main_importances = self.evaluator.get_importances(self.hp_names, depth=1)
        pairwise_importances = self.evaluator.get_importances(selected_hp_names, depth=2)
        cached_pairwise_importances = self.evaluator.get_importances(selected_hp_names, depth=2)

        assert set(main_importances) == set(self.hp_names)
        assert set(pairwise_importances) == {
            selected_hp_names[0],
            selected_hp_names[1],
            tuple(selected_hp_names),
        }
        assert pairwise_importances == cached_pairwise_importances
        assert (
            pairwise_importances[tuple(selected_hp_names)][0]
            <= pairwise_importances[tuple(selected_hp_names)][2]
        )


if __name__ == "__main__":
    unittest.main()
