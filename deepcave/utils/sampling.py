"""Read the General sampler mapping and build trial-filtered analytical run views."""

from typing import Any, Dict, List, Optional, Sequence

import json
import math
from copy import copy, deepcopy

from deepcave.runs import AbstractRun
from deepcave.runs.group import Group
from deepcave.utils.hash import string_to_hash

SAMPLING_METHODS_CACHE_KEY = "optuna_sampling_methods"
MISSING_METHOD = "missing"


def method_token(value: Any) -> str:
    """Keep missing attributes separate from literal method names and preserve types."""
    if value is None:
        return MISSING_METHOD
    return "value:" + json.dumps(value, ensure_ascii=False, sort_keys=True)


def mapped_method(attribute: str, study_attrs: Dict, trial_attrs: Dict) -> Any:
    """Resolve General's existing study:/trial: mapping without assuming a field name."""
    source, name = attribute.split(":", 1)
    return (study_attrs if source == "study" else trial_attrs).get(name)


def trial_attributes(run: AbstractRun) -> Dict[int, Dict]:
    """Use full imported attributes even for snapshots with scalar-only candidate metadata."""
    return {
        trial.additional["optuna_trial_number"]: trial.additional.get("user_attrs", {})
        for trial in run.history
        if "optuna_trial_number" in trial.additional
    }


def method_options(runs: Sequence[AbstractRun], attribute: Optional[str]) -> List[Dict]:
    """Offer distinct mapped values from the loaded source studies, including missing values."""
    if not attribute:
        return []
    values = {}
    for run in runs:
        if run.prefix != "Optuna":
            continue
        study = run.meta.get("optuna_user_attrs", {})
        attrs = trial_attributes(run)
        records = run.meta.get("optuna_candidates")
        trial_attrs = (
            [attrs.get(row["trial_number"], row.get("user_attrs", {})) for row in records]
            if records is not None
            else list(attrs.values())
        )
        for trial in trial_attrs:
            value = mapped_method(attribute, study, trial)
            token = method_token(value)
            values[token] = {
                "label": "Missing metadata" if value is None else str(value),
                "value": token,
            }
    return sorted(
        values.values(), key=lambda item: (item["value"] == MISSING_METHOD, item["label"])
    )


def sampling_signature(attribute: Optional[str], methods: Sequence[str]) -> Optional[str]:
    """Fingerprint the effective filter independently of selection order."""
    if not attribute or not methods:
        return None
    return string_to_hash(json.dumps([attribute, sorted(set(methods))]))


def filtered_run(run: AbstractRun, attribute: Optional[str], methods: Sequence[str]) -> AbstractRun:
    """Filter before analysis, preserving original runs, trial numbers and configuration IDs."""
    signature = sampling_signature(attribute, methods)
    if signature is None:
        return run
    if not isinstance(run, Group) and run.prefix != "Optuna":
        return run
    if isinstance(run, Group) and not any(source.prefix == "Optuna" for source in run.runs):
        return run
    assert attribute is not None
    selected = set(methods)
    view = copy(run)
    view.meta = {**run.meta, "objectives": deepcopy(run.meta["objectives"])}
    view._sampling_filter_key = signature
    if isinstance(run, Group):
        view.runs = [filtered_run(source, attribute, methods) for source in run.runs]
        view.history = []
        keys: Dict[tuple, int] = {}
        view._original_config_mapping = {}
        for source_id, source in enumerate(view.runs):
            for trial in source.history:
                original_id = trial.config_id
                trial = copy(trial)
                trial.config_id = run.get_new_config_id(source_id, original_id)
                key = trial.get_key()
                if key in keys:
                    view.history[keys[key]] = trial
                else:
                    keys[key] = len(view.history)
                    view.history.append(trial)
                view._original_config_mapping[trial.config_id] = (source_id, original_id)
    else:
        study = run.meta.get("optuna_user_attrs", {})
        view.history = [
            trial
            for trial in run.history
            if method_token(mapped_method(attribute, study, trial.additional.get("user_attrs", {})))
            in selected
        ]
        attrs = trial_attributes(run)
        view.meta["optuna_candidates"] = [
            row
            for row in run.meta.get("optuna_candidates", [])
            if method_token(
                mapped_method(
                    attribute, study, attrs.get(row["trial_number"], row.get("user_attrs", {}))
                )
            )
            in selected
        ]
    config_ids = {trial.config_id for trial in view.history}
    view.configs = {key: value for key, value in run.configs.items() if key in config_ids}
    view.config_id_mapping = {
        key: value for key, value in run.config_id_mapping.items() if value in config_ids
    }
    view.origins = {key: value for key, value in run.origins.items() if key in config_ids}
    view.models = {key: value for key, value in run.models.items() if key in config_ids}
    view.trial_keys = {trial.get_key(): index for index, trial in enumerate(view.history)}
    view._highest_budget = {}
    for trial in view.history:
        view._update_highest_budget(trial.config_id, trial.budget, trial.status)
    for key, field in (("budgets", "budget"), ("seeds", "seed")):
        view.meta[key] = sorted({getattr(trial, field) for trial in view.history}) or run.meta[key]
    for index, objective in enumerate(view.meta["objectives"]):
        values = [
            trial.costs[index]
            for trial in view.history
            if trial.costs[index] is not None and math.isfinite(trial.costs[index])
        ]
        if values:
            if not objective["lock_lower"]:
                objective["lower"] = min(values)
            if not objective["lock_upper"]:
                objective["upper"] = max(values)
    if isinstance(view, Group):
        view.budgets, view.seeds = view.meta["budgets"], view.meta["seeds"]
    return view
