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

"""Utilities for importing Optuna studies from local SQLite databases."""

from typing import Any, List

import hashlib
import os
import pickle
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from deepcave.runs.converters.optuna import OptunaRun

SNAPSHOT_DIRECTORY = "deepcave_optuna"
SNAPSHOT_FILENAME = "optuna_study.pkl"


@dataclass(frozen=True)
class OptunaSnapshotResult:
    """Describe a successfully created Optuna snapshot."""

    run_path: Path
    total_trials: int
    imported_trials: int
    skipped_trials: int
    refreshed: bool


def _import_optuna() -> Any:
    """Import Optuna lazily so it remains an optional dependency."""
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "The Optuna package is required to import Optuna studies. "
            "Please install it via `pip install deepcave[optuna]`."
        )

    return optuna


def resolve_optuna_database(database: str, working_directory: Path) -> Path:
    """Resolve and validate a local SQLite database path."""
    value = database.strip()
    if value == "":
        raise ValueError("Please enter a SQLite database path.")
    if "://" in value:
        raise ValueError("Enter a local SQLite file path, not a storage URI.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(working_directory) / path

    try:
        path = path.resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"SQLite database does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"SQLite database is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"SQLite database is not readable: {path}")

    return path


def _storage_url(database: Path) -> str:
    """Return an Optuna SQLite storage URL for an absolute path."""
    return f"sqlite:///{database.as_posix()}"


def list_optuna_studies(database: Path) -> List[str]:
    """List study names contained in a local Optuna SQLite database."""
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT DISTINCT study_name FROM studies ORDER BY study_name"
        ).fetchall()

    return [study_name for (study_name,) in rows]


def get_optuna_snapshot_path(database: Path, study_name: str, working_directory: Path) -> Path:
    """Return the stable run directory for a database and study name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", study_name).strip("._-")
    slug = slug[:32].rstrip("._-") or "study"
    identifier = f"{database.resolve()}\0{study_name}".encode("utf-8")
    digest = hashlib.sha256(identifier).hexdigest()[:12]
    return Path(working_directory).resolve() / SNAPSHOT_DIRECTORY / f"{slug}-{digest}"


def create_optuna_snapshot(
    database: Path,
    study_name: str,
    working_directory: Path,
) -> OptunaSnapshotResult:
    """Create and atomically publish a self-contained Optuna study snapshot."""
    optuna = _import_optuna()
    from optuna.storages import InMemoryStorage
    from optuna.trial import TrialState

    storage_url = _storage_url(database)
    study_names = list_optuna_studies(database)
    if study_name not in study_names:
        raise ValueError(f'Optuna study "{study_name}" was not found in {database}.')

    memory = InMemoryStorage()
    optuna.copy_study(
        from_study_name=study_name,
        from_storage=storage_url,
        to_storage=memory,
        to_study_name=study_name,
    )
    snapshot = optuna.load_study(storage=memory, study_name=study_name)
    # Storage does not retain the original sampler. Do not report load_study's default.
    sampler_name = snapshot.user_attrs.get("optuna_sampler")
    snapshot._deepcave_sampler_name = (
        sampler_name.strip()
        if isinstance(sampler_name, str) and sampler_name.strip()
        else "Unknown sampler"
    )

    total_trials = len(snapshot.trials)
    imported_trials = sum(
        trial.state in (TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED)
        for trial in snapshot.trials
    )
    skipped_trials = total_trials - imported_trials

    run_path = get_optuna_snapshot_path(database, study_name, working_directory)
    snapshot_path = run_path / SNAPSHOT_FILENAME
    refreshed = snapshot_path.is_file()
    if run_path.is_dir():
        unexpected_pickles = [
            path for path in run_path.glob("*.pkl") if path.name != SNAPSHOT_FILENAME
        ]
        if unexpected_pickles:
            raise ValueError(
                f"The snapshot directory contains an unexpected Pickle file: "
                f"{unexpected_pickles[0]}"
            )
    snapshot_bytes = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)

    snapshot_root = run_path.parent
    snapshot_root.mkdir(parents=True, exist_ok=True)

    # Validate exactly the bytes that will be published without touching an old snapshot.
    with tempfile.TemporaryDirectory(prefix=".optuna-validation-", dir=snapshot_root) as tmp:
        validation_path = Path(tmp) / SNAPSHOT_FILENAME
        with validation_path.open("wb") as file:
            file.write(snapshot_bytes)
            file.flush()
            os.fsync(file.fileno())
        OptunaRun.from_path(Path(tmp))

    run_path.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".optuna_study.",
            suffix=".tmp",
            dir=run_path,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(snapshot_bytes)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, snapshot_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return OptunaSnapshotResult(
        run_path=run_path,
        total_trials=total_trials,
        imported_trials=imported_trials,
        skipped_trials=skipped_trials,
        refreshed=refreshed,
    )
