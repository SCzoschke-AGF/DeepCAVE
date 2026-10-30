import concurrent.futures
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ConfigSpace import ConfigurationSpace

from deepcave.runs.converters.optuna import OptunaRun
from deepcave.runs.group import Group
from deepcave.runs.handler import (
    OPTUNA_SAMPLER_ATTRIBUTE_CACHE_KEY,
    RUN_ABBREV_AS_NAME_CACHE_KEY,
    RUN_ABBREV_ATTRIBUTES_CACHE_KEY,
    RunHandler,
)
from deepcave.runs.objective import Objective
from deepcave.utils.cache import Cache
from deepcave.utils.run_caches import RunCaches


def _handler(active_run_id=None):
    handler = object.__new__(RunHandler)
    handler.c = Cache(defaults={"active_run_id": active_run_id}, write_file=False)
    handler._update_lock = threading.RLock()
    handler.runs = {}
    handler.groups = {}
    return handler


def test_active_run_defaults_to_first_individual_run_before_groups():
    handler = _handler()
    handler.runs = {"/run-a": SimpleNamespace(id="run-a")}
    handler.groups = {"group": SimpleNamespace(id="group-a")}

    assert handler.get_active_run_id() == "run-a"
    assert handler.c.get("active_run_id") == "run-a"


def test_active_run_preserves_a_valid_group_and_does_not_replace_stale_ids():
    handler = _handler("group-a")
    handler.runs = {"/run-a": SimpleNamespace(id="run-a")}
    handler.groups = {"group": SimpleNamespace(id="group-a")}

    assert handler.get_active_run_id() == "group-a"

    handler.groups = {}
    assert handler.get_active_run_id() == "group-a"
    assert handler.c.get("active_run_id") == "group-a"


def test_active_run_does_not_fall_back_when_selection_is_unavailable():
    handler = _handler("removed")
    handler.groups = {"group": SimpleNamespace(id="group-a")}

    assert handler.get_active_run_id() == "removed"


def test_active_run_is_none_without_loaded_runs():
    handler = _handler()

    assert handler.get_active_run_id() is None
    assert handler.c.get("active_run_id") is None


def test_active_run_stays_selected_when_no_runs_are_loaded():
    handler = _handler("removed")

    assert handler.get_active_run_id() == "removed"
    assert handler.c.get("active_run_id") == "removed"


def test_run_name_prefers_loaded_run_display_name():
    handler = _handler()
    handler.runs = {"/runs/technical-slug-a1b2c3": SimpleNamespace(name="original Optuna study")}

    assert handler.get_run_name("/runs/technical-slug-a1b2c3") == "original Optuna study"
    assert handler.get_run_name("/runs/not-loaded") == "not-loaded"


def test_active_run_setter_accepts_only_loaded_runs():
    handler = _handler()
    handler.runs = {"/run-a": SimpleNamespace(id="run-a")}

    handler.set_active_run_id("run-a")
    assert handler.get_active_run_id() == "run-a"

    with pytest.raises(RuntimeError, match="Run not found"):
        handler.set_active_run_id("removed")


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "enabled,attributes,expected",
    [
        (False, ["model", "dataset"], "original study"),
        (True, ["model", "missing", "empty", "unset", "dataset"], "xgb_0"),
        (True, ["dataset", "model"], "0_xgb"),
        (True, [], "original study"),
        (True, ["missing", "empty", "unset"], "original study"),
    ],
)
def test_run_abbrev_name_applies_to_fresh_and_cached_runs(cached, enabled, attributes, expected):
    handler = _handler()
    handler.c.set(RUN_ABBREV_AS_NAME_CACHE_KEY, value=enabled)
    handler.c.set(RUN_ABBREV_ATTRIBUTES_CACHE_KEY, value=attributes)
    meta = {
        "optuna_study_name": "original study",
        "optuna_user_attrs": {"model": "xgb", "dataset": 0, "empty": "", "unset": None},
    }
    run = SimpleNamespace(
        name="original study", prefix="Optuna", meta=meta, get_meta=lambda: meta.copy()
    )
    handler.runs = {"/run-a": run} if cached else {}
    handler.rc = SimpleNamespace(update=lambda _run: False)
    handler.logger = SimpleNamespace(debug=Mock())
    handler.available_run_classes = [SimpleNamespace(from_path=lambda _path: run)]

    assert handler.update_run("/run-a") is run
    assert run.name == expected
    assert run.meta["optuna_study_name"] == "original study"


def test_run_abbrev_name_toggle_preserves_identity_groups_and_cached_results(tmp_path):
    run = OptunaRun(
        name="original study",
        configspace=ConfigurationSpace(),
        objectives=Objective("score", lower=0.0, upper=1.0),
        meta={
            "optuna_study_name": "original study",
            "optuna_user_attrs": {"model": "xgb", "dataset": "iris"},
        },
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "study.pkl").write_bytes(b"unchanged snapshot")
    run.path = snapshot
    group = Group("Comparison", [run])
    other = SimpleNamespace(prefix="DeepCAVE", name="manual name")
    handler = _handler(run.id)
    handler.runs = {str(snapshot): run, "other": other}
    handler.groups = {"Comparison": group}
    handler.c.set(RUN_ABBREV_ATTRIBUTES_CACHE_KEY, value=["model", "dataset"])
    handler.rc = RunCaches(SimpleNamespace(CACHE_DIR=tmp_path / "cache", DEBUG=False))
    handler.rc.update(run)
    handler.rc.set(run, "analysis", "inputs", value={"result": 42})
    identity = (run.id, run.hash, group.id, group.hash)

    handler.set_optuna_run_abbrev_as_name(True)

    assert run.name == handler.get_run_name(snapshot) == "xgb_iris"
    assert group.run_names == ["xgb_iris"]
    assert handler.groups["Comparison"] is group
    assert other.name == "manual name"
    assert handler.c.get(RUN_ABBREV_AS_NAME_CACHE_KEY) is True
    assert identity == (run.id, run.hash, group.id, group.hash)
    assert handler.get_active_run_id() == run.id
    assert handler.rc.update(run) is False
    assert handler.rc.get(run, "analysis", "inputs") == {"result": 42}

    handler.set_optuna_run_abbrev_as_name(False)

    assert run.name == "original study"
    assert group.run_names == ["original study"]
    assert handler.c.get(RUN_ABBREV_AS_NAME_CACHE_KEY) is False
    assert run.meta["optuna_study_name"] == "original study"


def test_concurrent_updates_are_serialized():
    class TrackingRunCaches:
        def __init__(self):
            self.active_updates = 0
            self.max_active_updates = 0
            self.lock = threading.Lock()

        def update(self, _):
            with self.lock:
                self.active_updates += 1
                self.max_active_updates = max(self.max_active_updates, self.active_updates)

            time.sleep(0.01)

            with self.lock:
                self.active_updates -= 1
            return False

    handler = _handler()
    handler.runs = {"/run-a": SimpleNamespace(id="run-a")}
    handler.rc = TrackingRunCaches()
    worker_count = 6
    start = threading.Barrier(worker_count)

    def update():
        start.wait()
        handler.update()

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(update) for _ in range(worker_count)]
        for future in futures:
            future.result()

    assert handler.rc.max_active_updates == 1


def test_group_configspace_congruence_is_forwarded_and_persisted(monkeypatch):
    created_groups = []

    class FakeGroup:
        def __init__(self, name, runs, require_configspace_congruence=True):
            self.id = name
            created_groups.append((name, runs, require_configspace_congruence))

    handler = _handler()
    handler.runs = {
        "/run-a": SimpleNamespace(id="run-a"),
        "/run-b": SimpleNamespace(id="run-b"),
    }
    handler.rc = SimpleNamespace(update=lambda _group: False)
    monkeypatch.setattr("deepcave.runs.handler.Group", FakeGroup)

    handler.update_groups(
        {"comparison": ["/run-a", "/run-b"]},
        {"comparison": False},
    )

    assert created_groups[0][2] is False
    assert handler.c.get("groups_configspace_congruence") == {"comparison": False}

    created_groups.clear()
    handler.update_groups()
    assert created_groups[0][2] is False


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "attribute,expected",
    [
        (None, "TPESampler"),
        ("study:sampler", "random"),
        ("trial:sampler", ["random", "tpe", {"name": "custom"}]),
        ("trial:missing", []),
        ("study:missing", None),
    ],
)
def test_optuna_sampler_mapping_applies_to_fresh_and_cached_runs(cached, attribute, expected):
    handler = _handler()
    handler.c.set(OPTUNA_SAMPLER_ATTRIBUTE_CACHE_KEY, value=attribute)
    run = SimpleNamespace(
        id="run-a",
        name="Run A",
        prefix="Optuna",
        meta={"optuna_sampler": "TPESampler", "optuna_user_attrs": {"sampler": "random"}},
        history=[
            SimpleNamespace(additional={"user_attrs": attrs})
            for attrs in [
                {"sampler": "random"},
                {"sampler": "tpe"},
                {"sampler": "random"},
                {},
                {"sampler": {"name": "custom"}},
                {"sampler": {"name": "custom"}},
            ]
        ],
    )
    handler.runs = {"/run-a": run} if cached else {}
    handler.rc = SimpleNamespace(update=lambda _run: False)
    handler.logger = SimpleNamespace(debug=Mock())
    handler.available_run_classes = [SimpleNamespace(from_path=lambda _path: run)]

    assert handler.update_run("/run-a") is run
    assert run.meta["optuna_sampler"] == expected
    assert run.meta["optuna_user_attrs"]["sampler"] == "random"


def test_sampler_setting_updates_loaded_runs_and_can_be_cleared():
    handler = _handler()
    handler.c.set("groups", value={})
    optuna_run = SimpleNamespace(
        id="optuna",
        prefix="Optuna",
        meta={"optuna_sampler": "TPESampler", "optuna_user_attrs": {"sampler": "random"}},
    )
    other_run = SimpleNamespace(id="other", prefix="DeepCAVE", meta={"sampler": "manual"})
    handler.runs = {"optuna": optuna_run, "other": other_run}
    handler.rc = SimpleNamespace(update=lambda _run: False)

    handler.set_optuna_sampler_attribute("study:sampler")
    assert optuna_run.meta["optuna_sampler"] == "random"
    assert handler.c.get(OPTUNA_SAMPLER_ATTRIBUTE_CACHE_KEY) == "study:sampler"
    handler.set_optuna_sampler_attribute(None)
    assert optuna_run.meta["optuna_sampler"] == "TPESampler"
    assert handler.c.get(OPTUNA_SAMPLER_ATTRIBUTE_CACHE_KEY) is None
    assert other_run.meta == {"sampler": "manual"}
