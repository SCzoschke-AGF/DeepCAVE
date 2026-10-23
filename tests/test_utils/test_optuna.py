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

import pickle
from pathlib import Path
from unittest.mock import patch

import optuna
import pytest
from optuna.storages import InMemoryStorage

from deepcave.runs.converters.optuna import OptunaRun
from deepcave.utils.optuna import (
    SNAPSHOT_FILENAME,
    create_optuna_snapshot,
    get_optuna_snapshot_path,
    list_optuna_studies,
    resolve_optuna_database,
)


def _create_database(path: Path, study_names: list[str]) -> None:
    storage = f"sqlite:///{path}"
    for index, study_name in enumerate(study_names):
        study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction="minimize",
        )
        study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)
        if index == 0:
            running_trial = study.ask()
            running_trial.suggest_float("x", 0.0, 1.0)


def test_resolve_database_and_reject_storage_uri(tmp_path: Path) -> None:
    database = tmp_path / "optuna.db"
    database.touch()

    assert resolve_optuna_database("optuna.db", tmp_path) == database
    assert resolve_optuna_database(str(database), tmp_path) == database

    with pytest.raises(ValueError, match="not a storage URI"):
        resolve_optuna_database(f"sqlite:///{database}", tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_optuna_database("missing.db", tmp_path)

    invalid_database = tmp_path / "invalid.db"
    invalid_database.write_text("not a SQLite database", encoding="utf-8")
    with pytest.raises(Exception):
        list_optuna_studies(invalid_database)


def test_list_studies_and_stable_snapshot_path(tmp_path: Path) -> None:
    database = tmp_path / "optuna.db"
    names = ["short", "very:long/study name " * 8]
    _create_database(database, names)

    assert list_optuna_studies(database) == sorted(names)

    first = get_optuna_snapshot_path(database, names[1], tmp_path)
    second = get_optuna_snapshot_path(database, names[1], tmp_path)
    assert first == second
    assert first.parent == tmp_path / "deepcave_optuna"
    slug, digest = first.name.rsplit("-", 1)
    assert len(slug) <= 32
    assert len(digest) == 12
    assert "/" not in first.name and ":" not in first.name


def test_list_studies_does_not_load_trial_history(tmp_path: Path) -> None:
    database = tmp_path / "optuna.db"
    names = ["second", "first"]
    _create_database(database, names)

    with patch("deepcave.utils.optuna._import_optuna", side_effect=AssertionError):
        assert list_optuna_studies(database) == sorted(names)


def test_snapshot_is_detached_and_tracks_skipped_trials(tmp_path: Path) -> None:
    database = tmp_path / "optuna.db"
    _create_database(database, ["other-study"])
    _create_database(database, ["live-study"])

    result = create_optuna_snapshot(database, "live-study", tmp_path)
    snapshot_file = result.run_path / SNAPSHOT_FILENAME

    assert result.total_trials == 2
    assert result.imported_trials == 1
    assert result.skipped_trials == 1
    assert not result.refreshed
    with snapshot_file.open("rb") as file:
        snapshot = pickle.load(file)
    assert isinstance(snapshot._storage, InMemoryStorage)

    database.rename(tmp_path / "optuna.db.away")
    run = OptunaRun.from_path(result.run_path)
    assert run.meta["optuna_study_id"] == 2
    assert run.meta["optuna_study_name"] == "live-study"
    assert len(run.history) == 1

    # Legacy detached snapshots must not export their in-memory ID as the source ID.
    del snapshot._deepcave_study_id
    with snapshot_file.open("wb") as file:
        pickle.dump(snapshot, file)
    legacy_run = OptunaRun.from_path(result.run_path)
    assert legacy_run.meta["optuna_study_id"] is None


@pytest.mark.parametrize("sampler_name", [None, "RandomSampler", "TPESampler", "", 42])
def test_snapshot_reports_original_sampler_metadata(
    tmp_path: Path, sampler_name: str | int | None
) -> None:
    database = tmp_path / "optuna.db"
    study_name = "xgb_random"
    study = optuna.create_study(
        storage=f"sqlite:///{database}",
        study_name=study_name,
        sampler=optuna.samplers.RandomSampler(seed=7),
    )
    if sampler_name is not None:
        study.set_user_attr("optuna_sampler", sampler_name)
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)

    result = create_optuna_snapshot(database, study_name, tmp_path)
    run = OptunaRun.from_path(result.run_path)

    assert run.meta["optuna_study_name"] == study_name
    assert run.meta["optuna_sampler"] == (
        sampler_name if isinstance(sampler_name, str) and sampler_name else "Unknown sampler"
    )


def test_failed_refresh_preserves_previous_snapshot(tmp_path: Path) -> None:
    database = tmp_path / "optuna.db"
    _create_database(database, ["study"])
    result = create_optuna_snapshot(database, "study", tmp_path)
    snapshot_file = result.run_path / SNAPSHOT_FILENAME
    previous_snapshot = snapshot_file.read_bytes()

    with patch.object(OptunaRun, "from_path", side_effect=ValueError("invalid snapshot")):
        with pytest.raises(ValueError, match="invalid snapshot"):
            create_optuna_snapshot(database, "study", tmp_path)

    assert snapshot_file.read_bytes() == previous_snapshot

    with patch("deepcave.utils.optuna.os.replace", side_effect=OSError("publish failed")):
        with pytest.raises(OSError, match="publish failed"):
            create_optuna_snapshot(database, "study", tmp_path)

    assert snapshot_file.read_bytes() == previous_snapshot
    assert list(result.run_path.glob("*.tmp")) == []


def test_snapshot_rejects_unknown_study(tmp_path: Path) -> None:
    database = tmp_path / "optuna.db"
    _create_database(database, ["study"])

    with pytest.raises(ValueError, match='study "missing" was not found'):
        create_optuna_snapshot(database, "missing", tmp_path)
