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
# OptunaRun

This module provides utilities to create an Optuna run.

## Classes
    - OptunaRun: Define an Optuna run object.
"""

from typing import Any, List, Union

import math
import pickle
from pathlib import Path

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Uniform,
)
from ConfigSpace.hyperparameters import Hyperparameter

from deepcave.runs import Status
from deepcave.runs.objective import Objective
from deepcave.runs.run import Run
from deepcave.utils.hash import file_to_hash
from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


def _is_finite_numeric_user_attr(value: Any) -> bool:
    """Return whether an Optuna user attribute can serve as a scalar objective."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _complete_numeric_user_attrs(trials: List[Any]) -> List[str]:
    """Return scalar numeric user attributes available on every completed trial."""
    if not trials:
        return []
    names = set(trials[0].user_attrs)
    for trial in trials[1:]:
        names.intersection_update(trial.user_attrs)
    return sorted(
        name
        for name in names
        if all(_is_finite_numeric_user_attr(trial.user_attrs.get(name)) for trial in trials)
    )


class OptunaRun(Run):
    """
    Define an Optuna run object.

    Properties
    ----------
    path : Path
        The path to the run.
    """

    prefix = "Optuna"
    _initial_order = 2

    @staticmethod
    def _get_pickle_file(path: Path) -> Path:
        """
        Get the path to the pickle file from the directory path.

        Parameters
        ----------
        path : Path
            The path to the directory containing the pickle file.

        Returns
        -------
        Path
            The path to the pickle file.
        """
        pickle_files = list(path.glob("*.pkl"))
        if len(pickle_files) != 1:
            raise RuntimeError(f"There should be exactly one pickle file in '{path}'")
        else:
            return pickle_files[0]

    @property
    def hash(self) -> str:
        """
        Hash of the current run.

        If the hash changes, the cache has to be cleared.
        This ensures that the cache always holds the latest results of the run.

        Returns
        -------
        str
            The hash of the run.
        """
        if self.path is None:
            return ""

        pickle_file = OptunaRun._get_pickle_file(self.path)
        return file_to_hash(pickle_file)

    @property
    def latest_change(self) -> Union[float, int]:
        """
        Get the timestamp of the latest change.

        Returns
        -------
        Union[float, int]
            The latest change.
        """
        if self.path is None:
            return 0

        pickle_file = OptunaRun._get_pickle_file(self.path)
        return Path(pickle_file).stat().st_mtime

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "OptunaRun":
        """
        Based on working_dir/run_name/*, return a new trials object.

        Parameters
        ----------
        path : Union[Path, str]
            The path to base the trial object on.

        Returns
        -------
        The Optuna run.

        Raises
        ------
        RuntimeError
            Instances are not supported.
        """
        path = Path(path)

        try:
            from optuna.distributions import (
                CategoricalDistribution,
                FloatDistribution,
                IntDistribution,
                distribution_to_json,
            )
            from optuna.search_space import IntersectionSearchSpace
            from optuna.study import StudyDirection
            from optuna.trial import TrialState
        except ImportError:
            raise ImportError(
                "The Optuna package is required to load Optuna runs. "
                "Please install it via `pip install deepcave[optuna]`"
            )

        # Load the optuna study from the file path
        pickle_file_path = OptunaRun._get_pickle_file(path)
        with open(pickle_file_path, "rb") as f:
            optuna_study = pickle.load(f)

        # Read configspace
        optuna_space = IntersectionSearchSpace(include_pruned=True).calculate(study=optuna_study)
        configspace = ConfigurationSpace()

        hyperparameters: List[Hyperparameter] = []

        for hp_name, hp in optuna_space.items():
            if isinstance(hp, FloatDistribution) or isinstance(hp, IntDistribution):
                if hp.step is not None and hp.step != 1:
                    logger.warning(
                        "ConfigSpace does not enforce Optuna's step during sampling. "
                        f'Step={hp.step} is retained as metadata for hyperparameter "{hp_name}".'
                    )

            if isinstance(hp, FloatDistribution):
                hyperparameters.append(
                    Float(
                        name=hp_name,
                        bounds=(hp.low, hp.high),
                        distribution=Uniform(),
                        default=None,
                        log=hp.log,
                        meta={"step": hp.step},
                    )
                )
            elif isinstance(hp, IntDistribution):
                hyperparameters.append(
                    Integer(
                        name=hp_name,
                        bounds=(hp.low, hp.high),
                        distribution=Uniform(),
                        default=None,
                        log=hp.log,
                        meta={"step": hp.step},
                    )
                )
            elif isinstance(hp, CategoricalDistribution):
                hyperparameters.append(
                    Categorical(
                        name=hp_name,
                        default=hp.choices[0],
                        items=hp.choices,
                    )
                )
            else:
                raise ValueError(
                    (
                        "The hyperparameters in the Optuna study must be of type "
                        "`FloatDistribution`, `IntDistribution` or `CategoricalDistribution`, "
                        f"but a hyperparameter of type {type(hp)} was given."
                    )
                )
        configspace.add(hyperparameters)

        supported_trials = [
            trial
            for trial in optuna_study.trials
            if trial.state in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)
        ]
        completed_trials = [
            trial for trial in supported_trials if trial.state == TrialState.COMPLETE
        ]
        if not completed_trials:
            raise ValueError("The Optuna study does not contain a completed trial yet.")
        user_attr_objective_names = _complete_numeric_user_attrs(completed_trials)

        n_objectives = len(optuna_study.directions)
        obj_list = list()
        for i in range(n_objectives):
            if optuna_study.metric_names is not None:
                metric_name = optuna_study.metric_names[i]
            else:
                metric_name = f"Objective{i}"
            optimize = "lower" if optuna_study.directions[i] == StudyDirection.MINIMIZE else "upper"

            obj_list.append(
                Objective(
                    name=metric_name,
                    lower=None,
                    upper=None,
                    optimize=optimize,
                )
            )

        user_attr_objective_offset = len(obj_list)
        obj_list.extend(
            Objective(f"user_attr:{name}", optimize="upper") for name in user_attr_objective_names
        )
        obj_list.append(Objective("Time"))

        # Let's create a new run object
        study_name = optuna_study.study_name
        meta = {
            "optuna_study_name": study_name,
            "optuna_study_id": getattr(
                optuna_study,
                "_deepcave_study_id",
                # Older detached snapshots no longer contain the original ID.
                None if hasattr(optuna_study, "_deepcave_sampler_name") else optuna_study._study_id,
            ),
            "optuna_sampler": getattr(
                optuna_study, "_deepcave_sampler_name", type(optuna_study.sampler).__name__
            ),
        }
        # Preserve the audit before intersection conversion can hide dynamic parameters.
        names = set().union(*(trial.distributions for trial in completed_trials))
        changing = [
            name
            for name in sorted(names)
            if any(name not in trial.distributions for trial in completed_trials)
            or len(
                {
                    distribution_to_json(trial.distributions[name])
                    for trial in completed_trials
                    if name in trial.distributions
                }
            )
            != 1
        ]
        meta["optuna_space_error"] = (
            "KDE scoring does not support conditional or changing Optuna distributions: "
            + ", ".join(changing)
            if changing
            else None
        )
        model_family = optuna_study.user_attrs.get("model_family")
        if model_family is not None:
            meta["model_family"] = model_family
        meta["user_attr_objectives"] = {
            name: user_attr_objective_offset + index
            for index, name in enumerate(user_attr_objective_names)
        }

        run = OptunaRun(name=study_name, configspace=configspace, objectives=obj_list, meta=meta)

        # The path has to be set manually
        run._path = path

        start_times = [
            trial.datetime_start for trial in supported_trials if trial.datetime_start is not None
        ]
        first_starttime = min(start_times) if len(start_times) > 0 else None

        for trial in supported_trials:
            try:
                config = Configuration(configspace, trial.params)
            except ValueError:
                raise ValueError(
                    f"Could not convert the configuration of trial {trial.number} to "
                    f"a ConfigSpace configuration.\nThis might be due to the "
                    f"configuration space containing conditionals or dynamic "
                    f"hyperparameter value ranges, which are currently not supported."
                )

            if trial.state == TrialState.COMPLETE:
                status = Status.SUCCESS
            elif trial.state == TrialState.FAIL:
                status = Status.FAILED
            else:
                status = Status.PRUNED

            starttime = 0.0
            if first_starttime is not None and trial.datetime_start is not None:
                starttime = (trial.datetime_start - first_starttime).total_seconds()

            endtime = starttime
            if first_starttime is not None and trial.datetime_complete is not None:
                endtime = (trial.datetime_complete - first_starttime).total_seconds()

            if status == Status.SUCCESS:
                if trial.values is None or len(trial.values) != n_objectives:
                    raise ValueError(
                        f"Completed trial {trial.number} does not provide {n_objectives} "
                        "objective values."
                    )
                costs = list(trial.values)
                costs.extend(float(trial.user_attrs[name]) for name in user_attr_objective_names)
                if trial.datetime_start is not None and trial.datetime_complete is not None:
                    time = float(endtime - starttime)
                else:
                    time = None
            else:
                costs = [None] * (n_objectives + len(user_attr_objective_names))
                time = None

            run.add(
                costs=costs + [time],
                config=config,
                budget=0.0,
                seed=-1,
                start_time=starttime,
                end_time=endtime,
                status=status,
                origin=None,
                additional={
                    "optuna_trial_number": trial.number,
                    "user_attrs": dict(trial.user_attrs),
                },
                allow_duplicate_trials=True,
            )

        return run

    @classmethod
    def is_valid_run(cls, path_name: str) -> bool:
        """
        Check whether the path name belongs to a valid optuna run.

        Parameters
        ----------
        path_name: str
            The path to check.

        Returns
        -------
        bool
            True if path is valid run.
            False otherwise.
        """
        path = Path(path_name)
        pickle_files = list(path.glob("*.pkl"))
        if len(pickle_files) != 1:
            return False
        return True
