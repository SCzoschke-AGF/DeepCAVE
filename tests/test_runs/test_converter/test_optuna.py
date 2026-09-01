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
from optuna.distributions import FloatDistribution
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


def test_converter_rejects_study_without_completed_trial(tmp_path: Path) -> None:
    study = optuna.create_study(direction="minimize")
    study.add_trial(create_trial(state=TrialState.WAITING))

    with pytest.raises(ValueError, match="does not contain a completed trial"):
        OptunaRun.from_path(_write_study(tmp_path / "run", study))
