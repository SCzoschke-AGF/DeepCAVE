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

from typing import Any

import pickle
from pathlib import Path

import optuna
import pytest
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial

from deepcave.runs import Status
from deepcave.runs.converters.optuna import OptunaRun


def _write_study(path: Path, study: Any) -> Path:
    path.mkdir()
    with (path / "optuna_study.pkl").open("wb") as file:
        pickle.dump(study, file)
    return path


def test_converter_handles_live_trial_states(tmp_path: Path) -> None:
    study = optuna.create_study(directions=["minimize", "maximize"])
    distribution = {"x": FloatDistribution(0.0, 1.0)}
    study.add_trial(create_trial(params={"x": 0.1}, distributions=distribution, values=[0.4, 0.8]))
    study.add_trial(
        create_trial(
            state=TrialState.FAIL,
            params={"x": 0.2},
            distributions=distribution,
        )
    )
    study.add_trial(
        create_trial(
            state=TrialState.PRUNED,
            params={"x": 0.3},
            distributions=distribution,
        )
    )
    study.add_trial(
        create_trial(
            state=TrialState.RUNNING,
            params={"x": 0.4},
            distributions=distribution,
        )
    )
    study.add_trial(create_trial(state=TrialState.WAITING))

    run = OptunaRun.from_path(_write_study(tmp_path / "run", study))

    assert len(run.history) == 3
    assert [trial.status for trial in run.history] == [
        Status.SUCCESS,
        Status.FAILED,
        Status.PRUNED,
    ]
    assert run.history[0].costs[:2] == [0.4, 0.8]
    assert run.history[1].costs == [None, None, None]
    assert run.history[2].costs == [None, None, None]


def test_converter_uses_original_study_name_instead_of_snapshot_directory(
    tmp_path: Path,
) -> None:
    study_name = "xgboost_data:original-study-name_with_full_details"
    study = optuna.create_study(study_name=study_name, direction="minimize")
    study.add_trial(
        create_trial(
            params={"x": 0.1},
            distributions={"x": FloatDistribution(0.0, 1.0)},
            value=0.4,
        )
    )

    run = OptunaRun.from_path(_write_study(tmp_path / "technical-slug-a1b2c3", study))

    assert run.name == study_name
    assert run.meta["optuna_study_name"] == study_name
    assert run.meta["optuna_study_id"] == study._study_id


def test_converter_handles_missing_timestamps(tmp_path: Path) -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        create_trial(
            params={"x": 0.1},
            distributions={"x": FloatDistribution(0.0, 1.0)},
            value=0.4,
        )
    )
    trial = study.get_trials(deepcopy=False)[0]
    trial.datetime_start = None
    trial.datetime_complete = None

    run = OptunaRun.from_path(_write_study(tmp_path / "run", study))

    assert run.history[0].start_time == 0.0
    assert run.history[0].end_time == 0.0
    assert run.history[0].costs == [0.4, None]


def test_converter_preserves_duplicate_configurations_and_trial_numbers(tmp_path: Path) -> None:
    study = optuna.create_study(direction="maximize")
    distribution = {"x": FloatDistribution(0.0, 1.0)}
    for value in (0.4, 0.6):
        study.add_trial(
            create_trial(
                params={"x": 0.5},
                distributions=distribution,
                value=value,
            )
        )

    run = OptunaRun.from_path(_write_study(tmp_path / "run", study))

    assert len(run.history) == 2
    assert [trial.costs[0] for trial in run.history] == pytest.approx([0.4, 0.6])
    assert [trial.additional["optuna_trial_number"] for trial in run.history] == [0, 1]


def test_converter_preserves_trial_user_attrs(tmp_path: Path) -> None:
    study = optuna.create_study(direction="maximize")
    study.add_trial(
        create_trial(
            params={"x": 0.5},
            distributions={"x": FloatDistribution(0.0, 1.0)},
            value=0.8,
            user_attrs={
                "average_precision": 0.8,
                "fold_average_precision": [0.81, 0.76, 0.84, 0.79, 0.82],
            },
        )
    )

    run = OptunaRun.from_path(_write_study(tmp_path / "run", study))

    assert run.history[0].additional == {
        "optuna_trial_number": 0,
        "user_attrs": {
            "average_precision": 0.8,
            "fold_average_precision": [0.81, 0.76, 0.84, 0.79, 0.82],
        },
    }


def test_converter_promotes_only_complete_finite_scalar_user_attrs(tmp_path: Path) -> None:
    study = optuna.create_study(direction="minimize")
    distribution = {"x": FloatDistribution(0.0, 1.0)}
    study.add_trial(
        create_trial(
            params={"x": 0.25},
            distributions=distribution,
            value=0.4,
            user_attrs={
                "score": 0.8,
                "incomplete": 1.0,
                "flag": True,
                "non_finite": float("inf"),
                "fold_scores": [0.7, 0.9],
            },
        )
    )
    study.add_trial(
        create_trial(
            params={"x": 0.75},
            distributions=distribution,
            value=0.2,
            user_attrs={
                "score": 0.9,
                "flag": False,
                "non_finite": 0.5,
                "fold_scores": [0.8, 1.0],
            },
        )
    )

    run = OptunaRun.from_path(_write_study(tmp_path / "run", study))

    assert run.get_objective_names() == ["Objective0", "user_attr:score", "Time"]
    assert run.meta["user_attr_objectives"] == {"score": 1}
    assert run.history[0].costs[:2] == pytest.approx([0.4, 0.8])
    assert run.history[1].costs[:2] == pytest.approx([0.2, 0.9])


def test_converter_retains_optuna_step_as_parameter_metadata(tmp_path: Path) -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(
        create_trial(
            params={"n_estimators": 150},
            distributions={"n_estimators": IntDistribution(100, 600, step=50)},
            value=0.4,
        )
    )

    run = OptunaRun.from_path(_write_study(tmp_path / "run", study))

    assert run.configspace["n_estimators"].meta["step"] == 50


def test_converter_retains_model_family(tmp_path: Path) -> None:
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("model_family", "xgboost")
    study.add_trial(
        create_trial(
            params={"x": 0.1},
            distributions={"x": FloatDistribution(0.0, 1.0)},
            value=0.4,
        )
    )

    run = OptunaRun.from_path(_write_study(tmp_path / "run", study))

    assert run.meta["model_family"] == "xgboost"


def test_converter_retains_sampler_name(tmp_path: Path) -> None:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())
    study.add_trial(create_trial(value=0.4))

    run = OptunaRun.from_path(_write_study(tmp_path / "run", study))

    assert run.meta["optuna_sampler"] == "RandomSampler"


def test_converter_rejects_study_without_completed_trial(tmp_path: Path) -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(create_trial(state=TrialState.WAITING))

    with pytest.raises(ValueError, match="does not contain a completed trial"):
        OptunaRun.from_path(_write_study(tmp_path / "run", study))
