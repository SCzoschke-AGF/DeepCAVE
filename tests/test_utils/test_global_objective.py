from types import SimpleNamespace

from deepcave.runs.objective import Objective
from deepcave.utils.global_objective import (
    apply_main_objective_direction,
    is_user_attr_objective,
    main_objective_options,
    resolve_main_objective_id,
)


class DummyRun:
    """Small run double exposing the objective API used by the global selector."""

    prefix = "Optuna"

    def __init__(self) -> None:
        self.meta = {
            "objectives": [
                Objective("Loss", optimize="lower").to_json(),
                Objective("user_attr:accuracy", optimize="upper").to_json(),
                Objective("Time", optimize="lower").to_json(),
            ],
            "user_attr_objectives": {"accuracy": 1},
        }
        self.runs = []

    def get_objectives(self):
        return [Objective.from_json(objective) for objective in self.meta["objectives"]]

    def get_objective_ids(self):
        return list(range(len(self.meta["objectives"])))


def test_options_use_stable_keys_and_hide_optuna_time() -> None:
    run = DummyRun()

    assert main_objective_options(run) == [
        {"label": "Objective · Loss", "value": "objective:0"},
        {"label": "User attr · accuracy", "value": "user_attr:accuracy"},
    ]


def test_stable_keys_resolve_to_run_local_objective_ids() -> None:
    run = DummyRun()

    assert resolve_main_objective_id(run, "objective:0") == 0
    assert resolve_main_objective_id(run, "user_attr:accuracy") == 1
    assert resolve_main_objective_id(run, 2) == 2
    assert resolve_main_objective_id(run, "2") == 2
    assert resolve_main_objective_id(run, "objective:99") is None
    assert resolve_main_objective_id(run, "user_attr:missing") is None
    assert resolve_main_objective_id(run, "not-an-objective") is None


def test_direction_is_applied_only_to_derived_user_attr_objectives() -> None:
    run = DummyRun()
    member = DummyRun()
    run.runs = [member]

    apply_main_objective_direction(run, "user_attr:accuracy", "lower")

    assert run.meta["objectives"][1]["optimize"] == "lower"
    assert member.meta["objectives"][1]["optimize"] == "lower"

    apply_main_objective_direction(run, "objective:0", "upper")
    assert run.meta["objectives"][0]["optimize"] == "lower"

    apply_main_objective_direction(run, "user_attr:accuracy", "sideways")
    assert run.meta["objectives"][1]["optimize"] == "lower"


def test_user_attr_key_detection_is_strict() -> None:
    assert is_user_attr_objective("user_attr:accuracy") is True
    assert is_user_attr_objective("objective:0") is False
    assert is_user_attr_objective(SimpleNamespace()) is False
