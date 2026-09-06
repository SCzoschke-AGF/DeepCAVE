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

from typing import Any, Dict

import inspect
import pickle

import numpy as np
import pytest

from deepcave.evaluators.ped_anova import PEDANOVA
from deepcave.runs.converters.optuna import OptunaRun
from deepcave.runs.converters.smac3v1 import SMAC3v1Run


def test_ped_anova_is_normalized_and_deterministic() -> None:
    run = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
    hp_names = list(run.configspace.keys())
    evaluator = PEDANOVA(run)
    objective = run.get_objective(0)
    budget = run.get_budget(0)

    evaluator.calculate(objective, budget, target_quantile=0.1)
    first = evaluator.get_importances(hp_names)
    evaluator.calculate(objective, budget, target_quantile=0.1)
    second = evaluator.get_importances(hp_names)

    assert first == second
    assert sum(importance for importance, _ in first.values()) == pytest.approx(1.0)
    assert all(importance >= 0.0 for importance, _ in first.values())
    assert all(variance == 0.0 for _, variance in first.values())


@pytest.mark.parametrize("target_quantile", [0.0, 1.0, -0.1, 1.1])
def test_ped_anova_rejects_invalid_target_quantile(target_quantile: float) -> None:
    run = SMAC3v1Run.from_path("logs/SMAC3v1/mlp/run_1")
    evaluator = PEDANOVA(run)

    with pytest.raises(RuntimeError, match="target_quantile"):
        evaluator.calculate(
            run.get_objective(0),
            run.get_budget(0),
            target_quantile=target_quantile,
        )


def test_ped_anova_matches_optuna_4_9(tmp_path: Any) -> None:
    optuna = pytest.importorskip("optuna")
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )
    from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances
    from optuna.trial import create_trial

    if "target_quantile" not in inspect.signature(PedAnovaImportanceEvaluator).parameters:
        pytest.skip("Parity reference requires Optuna >= 4.7.")

    rng = np.random.RandomState(7)
    n_trials = 160
    distributions = {
        "x_float": FloatDistribution(-5.0, 5.0),
        "x_log": FloatDistribution(1e-3, 10.0, log=True),
        "x_int": IntDistribution(1, 9),
        "x_cat": CategoricalDistribution(("a", "b", "c")),
    }
    parameters: Dict[str, np.ndarray] = {
        "x_float": rng.uniform(-5.0, 5.0, n_trials),
        "x_log": np.exp(rng.uniform(np.log(1e-3), np.log(10.0), n_trials)),
        "x_int": rng.randint(1, 10, n_trials),
        "x_cat": rng.choice(np.array(["a", "b", "c"], dtype=object), n_trials),
    }
    values = (
        (parameters["x_float"].astype(float) - 1.25) ** 2
        + 0.35 * np.log10(parameters["x_log"].astype(float) / 0.2) ** 2
        + 0.08 * (parameters["x_int"].astype(float) - 5.0) ** 2
        + np.where(
            parameters["x_cat"] == "a",
            0.0,
            np.where(parameters["x_cat"] == "b", 0.4, 1.2),
        )
    )

    study = optuna.create_study(direction="minimize")
    for index, value in enumerate(values):
        params = {
            name: (column[index].item() if hasattr(column[index], "item") else column[index])
            for name, column in parameters.items()
        }
        study.add_trial(
            create_trial(
                params=params,
                distributions=distributions,
                value=float(value),
            )
        )

    run_path = tmp_path / "optuna-run"
    run_path.mkdir()
    with (run_path / "optuna_study.pkl").open("wb") as file:
        pickle.dump(study, file)

    run = OptunaRun.from_path(run_path)
    evaluator = PEDANOVA(run)
    evaluator.calculate(
        run.get_objective(0),
        run.get_budget(0),
        target_quantile=0.1,
    )
    actual = {name: importance for name, (importance, _) in evaluator.get_importances().items()}
    expected = get_param_importances(
        study,
        evaluator=PedAnovaImportanceEvaluator(
            target_quantile=0.1,
            region_quantile=1.0,
        ),
        params=list(distributions),
        normalize=True,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
