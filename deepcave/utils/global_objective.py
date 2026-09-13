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

"""Shared global-main-objective identifiers and run helpers."""

from typing import Any, Dict, List, Optional

NATIVE_OBJECTIVE_PREFIX = "objective:"
USER_ATTR_OBJECTIVE_PREFIX = "user_attr:"
DEFAULT_MAIN_OBJECTIVE = f"{NATIVE_OBJECTIVE_PREFIX}0"


def _user_attr_objectives(run: Any) -> Dict[str, int]:
    """Return the user-attribute objective mapping stored by compatible converters."""
    mapping = getattr(run, "meta", {}).get("user_attr_objectives", {})
    if not isinstance(mapping, dict):
        return {}
    return {str(name): int(objective_id) for name, objective_id in mapping.items()}


def _exclude_time_objective(run: Any, objective_id: int, objective: Any) -> bool:
    """Return whether an Optuna converter's synthetic time objective should be omitted."""
    if objective.name.lower() != "time":
        return False
    if getattr(run, "prefix", None) == "Optuna":
        return True
    members = getattr(run, "runs", None) or []
    return bool(members) and all(getattr(member, "prefix", None) == "Optuna" for member in members)


def main_objective_options(run: Any) -> List[Dict[str, Any]]:
    """Build stable global objective options for a run or group."""
    user_attrs_by_id = {
        objective_id: name for name, objective_id in _user_attr_objectives(run).items()
    }
    options = []
    for objective_id, objective in enumerate(run.get_objectives()):
        if _exclude_time_objective(run, objective_id, objective):
            continue
        if objective_id in user_attrs_by_id:
            name = user_attrs_by_id[objective_id]
            options.append(
                {
                    "label": f"User attr · {name}",
                    "value": f"{USER_ATTR_OBJECTIVE_PREFIX}{name}",
                }
            )
        else:
            options.append(
                {
                    "label": f"Objective · {objective.name}",
                    "value": f"{NATIVE_OBJECTIVE_PREFIX}{objective_id}",
                }
            )
    return options


def resolve_main_objective_id(run: Any, value: Any) -> Optional[int]:
    """Resolve a stable global objective value to a run-local objective index."""
    if isinstance(value, int):
        return value if value in run.get_objective_ids() else None
    if not isinstance(value, str):
        return None
    if value.startswith(USER_ATTR_OBJECTIVE_PREFIX):
        return _user_attr_objectives(run).get(value[len(USER_ATTR_OBJECTIVE_PREFIX) :])
    if value.startswith(NATIVE_OBJECTIVE_PREFIX):
        try:
            objective_id = int(value[len(NATIVE_OBJECTIVE_PREFIX) :])
        except ValueError:
            return None
        return objective_id if objective_id in run.get_objective_ids() else None
    try:
        objective_id = int(value)
    except ValueError:
        return None
    return objective_id if objective_id in run.get_objective_ids() else None


def is_user_attr_objective(value: Any) -> bool:
    """Return whether a global objective value identifies a trial user attribute."""
    return isinstance(value, str) and value.startswith(USER_ATTR_OBJECTIVE_PREFIX)


def apply_main_objective_direction(run: Any, value: Any, direction: Any) -> None:
    """Apply the chosen direction to a derived user-attribute objective in memory."""
    if not is_user_attr_objective(value) or direction not in ("upper", "lower"):
        return
    objective_id = resolve_main_objective_id(run, value)
    objectives = getattr(run, "meta", {}).get("objectives", [])
    if objective_id is not None and objective_id < len(objectives):
        objectives[objective_id]["optimize"] = direction
    for member in getattr(run, "runs", []) or []:
        apply_main_objective_direction(member, value, direction)
