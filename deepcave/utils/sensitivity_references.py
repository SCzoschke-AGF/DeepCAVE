"""Find the original selection trials for sensitivity configurations in local Optuna storage."""

from typing import Any, Dict, Optional, Sequence, Tuple

import hashlib
import json
import math
import sqlite3
from functools import lru_cache
from itertools import groupby
from pathlib import Path

from deepcave.evaluators.sensitivity import canonical, resolve_attribute

REFERENCE_CACHE_KEY = "sensitivity_reference_trials"
DATASET_KEYS = ("dataset_sha256", "dataset_hash", "data_hash", "dataset_id", "dataset")
MODEL_KEYS = ("model_family", "model_type", "model")
SEED_PARAMETERS = {
    "model": ("model_seed", "random_state", "random_seed", "seed"),
    "data": ("data_seed", "split_seed", "cv_split_seed", "data_random_state"),
}


def dataset_identity(attributes: Dict) -> Optional[Tuple[str, str]]:
    """Prefer a dataset hash, then an explicit dataset ID or name."""
    for name in DATASET_KEYS:
        value = attributes.get(name)
        if value is not None and value != "":
            kind = "hash" if name in DATASET_KEYS[:3] else name
            return kind, canonical(value)
    return None


def reference_key(group: Dict) -> str:
    """Keep a reference stable across sensitivity modes, conditions and objective changes."""
    row = group["rows"][0]
    attributes = {**row.get("study_attrs", {}), **row.get("user_attrs", {})}
    sources = sorted({(row["storage"], row["study_name"]) for row in group["rows"]})
    identity = [sources, group["parameters"], dataset_identity(attributes)]
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def reference_label(trial: Dict) -> str:
    """Identify a matching trial without requiring the user to open its study."""
    values = ", ".join(f"{value:.6g}" if value is not None else "—" for value in trial["values"])
    return (
        f"{trial['study_name']} · Trial {trial['trial_number']} · Objective: {values} · "
        f"Model seed: {trial['model_seed']} · Data seed: {trial['data_seed']}"
    )


def reference_objective(trial: Dict, group: Dict) -> Optional[float]:
    """Resolve the selected objective without substituting another metric."""
    objective = group["context"]["objective"][0]
    if objective.startswith("user_attr:"):
        value = trial.get("user_attrs", {}).get(objective[len("user_attr:") :])
    elif "metrics" in trial:
        value = trial["metrics"].get(objective)
    else:
        # References saved before metric names were retained contain native values only.
        index = group["rows"][0]["metric_info"][objective]["index"]
        values = trial.get("values", [])
        value = values[index] if index < len(values) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _trial_objectives(connection: sqlite3.Connection, trial_id: int, study_id: int) -> Dict:
    """Read the objective metadata for one known trial."""
    values = [
        value if kind == "FINITE" else None
        for value, kind in connection.execute(
            "SELECT value, value_type FROM trial_values WHERE trial_id = ? ORDER BY objective",
            (trial_id,),
        )
    ]
    names_row = connection.execute(
        "SELECT value_json FROM study_system_attributes WHERE study_id = ? AND key = 'study:metric_names'",
        (study_id,),
    ).fetchone()
    names = json.loads(names_row[0]) if names_row else None
    attributes = {}
    for name, encoded in connection.execute(
        "SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?", (trial_id,)
    ):
        value = json.loads(encoded)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            attributes[name] = value
    return {
        "values": values,
        "metrics": {
            "metric:" + (names[index] if names else f"Objective{index}"): value
            for index, value in enumerate(values)
        },
        "user_attrs": attributes,
    }


def refresh_reference_objectives(trial: Dict) -> Dict:
    """Load missing metrics for an assigned trial by its exact identity, without a search."""
    database = Path(trial["database"]).resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT t.study_id FROM trials t JOIN studies s ON s.study_id = t.study_id "
            "WHERE t.trial_id = ? AND t.number = ? AND s.study_name = ?",
            (trial["trial_id"], trial["trial_number"], trial["study_name"]),
        ).fetchone()
        if row is None:
            raise ValueError(f"Reference trial {trial['trial_number']} was not found in its study.")
        return {**trial, **_trial_objectives(connection, trial["trial_id"], row[0])}


def find_reference_trials(database: Path, groups: Sequence[Dict], mappings: Dict) -> Dict[str, Any]:
    """Read matching completed trials from other studies without modifying the Optuna database."""
    from optuna.distributions import json_to_distribution

    configs = {reference_key(group): group for group in groups}
    matches: Dict[str, list] = {key: [] for key in configs}
    ignored = {
        name
        for kind, mapping in mappings.items()
        if mapping
        for name in (*SEED_PARAMETERS[kind], mapping.split(":", 1)[-1])
    }

    def parameter_key(parameters: Dict) -> str:
        return canonical({name: value for name, value in parameters.items() if name not in ignored})

    targets: Dict[tuple, list] = {}
    missing_dataset = 0
    for key, group in configs.items():
        row = group["rows"][0]
        attributes = {**row.get("study_attrs", {}), **row.get("user_attrs", {})}
        dataset = dataset_identity(attributes)
        if dataset is None:
            missing_dataset += 1
            continue
        model = next((attributes[name] for name in MODEL_KEYS if attributes.get(name)), None)
        sources = {row["study_name"] for row in group["rows"]}
        targets.setdefault((dataset, parameter_key(group["parameters"])), []).append(
            (key, sources, model)
        )

    result = {
        "matches": matches,
        "missing_dataset": missing_dataset,
        "configurations": len(configs),
    }
    if not targets:
        return result

    attribute_names = set(DATASET_KEYS + MODEL_KEYS) | ignored
    placeholders = ",".join("?" for _ in attribute_names)
    database = database.resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        studies = dict(connection.execute("SELECT study_id, study_name FROM studies"))
        study_attrs: Dict[int, Dict] = {}
        trial_attrs: Dict[int, Dict] = {}
        for table, field, attributes in (
            ("study_user_attributes", "study_id", study_attrs),
            ("trial_user_attributes", "trial_id", trial_attrs),
        ):
            for owner, name, value in connection.execute(
                f"SELECT {field}, key, value_json FROM {table} WHERE key IN ({placeholders})",
                tuple(attribute_names),
            ):
                attributes.setdefault(owner, {})[name] = json.loads(value)

        @lru_cache(maxsize=None)
        def distribution(encoded: str) -> Any:
            return json_to_distribution(encoded)

        rows = connection.execute(
            "SELECT t.trial_id, t.number, t.study_id, p.param_name, "
            "p.param_value, p.distribution_json FROM trials t "
            "LEFT JOIN trial_params p ON p.trial_id = t.trial_id "
            "WHERE t.state = 'COMPLETE' ORDER BY t.trial_id"
        )
        for trial_id, parameter_rows in groupby(rows, key=lambda row: row[0]):
            records = list(parameter_rows)
            _, number, study_id, *_ = records[0]
            study, attrs = study_attrs.get(study_id, {}), trial_attrs.get(trial_id, {})
            attributes = {**study, **attrs}
            dataset = dataset_identity(attributes)
            if dataset is None:
                continue
            parameters = {
                name: distribution(encoded).to_external_repr(value)
                for _, _, _, name, value, encoded in records
                if name is not None and name not in ignored
            }
            candidates = targets.get((dataset, parameter_key(parameters)), [])
            model = next((attributes[name] for name in MODEL_KEYS if attributes.get(name)), None)
            eligible = [
                key
                for key, sources, target_model in candidates
                if studies[study_id] not in sources
                and (
                    model is None
                    or target_model is None
                    or canonical(model) == canonical(target_model)
                )
            ]
            if not eligible:
                continue
            seeds = {}
            for kind, mapping in mappings.items():
                name = mapping.split(":", 1)[-1] if mapping else ""
                seeds[kind] = resolve_attribute(mapping, study, attrs)
                if seeds[kind] is None:
                    seeds[kind] = attrs.get(name, study.get(name))
            trial = {
                "id": hashlib.sha256(canonical([str(database), trial_id]).encode()).hexdigest(),
                "database": str(database),
                "study_name": studies[study_id],
                "trial_id": trial_id,
                "trial_number": number,
                **_trial_objectives(connection, trial_id, study_id),
                "model_seed": seeds.get("model"),
                "data_seed": seeds.get("data"),
            }
            for key in eligible:
                matches[key].append(trial)
    for trials in matches.values():
        trials.sort(key=lambda trial: (trial["study_name"], trial["trial_number"]))
    return result
