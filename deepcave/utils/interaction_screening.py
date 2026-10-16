"""Persist fitted fANOVA models so screening and selected exact pairs share one forest."""

from typing import Tuple

import hashlib
import json
import os
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.fanova import fANOVA
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective


def get_cached_fanova(
    run: AbstractRun, objective: Objective, budget: float, n_trees: int, directory: Path
) -> Tuple[fANOVA, Path]:
    """Key the model by its actual encoded data, configuration space and forest settings."""
    frame = run.get_encoded_data(objective, budget, specific=True, include_combined_cost=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(run.configspace.to_serialized_dict(), sort_keys=True).encode())
    digest.update(
        np.ascontiguousarray(
            frame[[*run.configspace.keys(), COMBINED_COST_NAME]].to_numpy(dtype=float)
        ).tobytes()
    )
    digest.update(f"interaction-screening-v1:{n_trees}:seed=0".encode())
    path = directory / f"{digest.hexdigest()}.pkl"
    if path.is_file():
        with path.open("rb") as stream:
            return pickle.load(stream), path
    evaluator = fANOVA(run)
    evaluator.calculate(objective, budget, n_trees=n_trees, seed=0)
    return evaluator, path


def save_cached_fanova(evaluator: fANOVA, path: Path) -> None:
    """Atomically retain the forest together with cached marginals and screening estimates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(evaluator, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
