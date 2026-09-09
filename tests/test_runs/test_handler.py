import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pytest

from deepcave.runs.handler import RunHandler
from deepcave.utils.cache import Cache


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


def test_active_run_setter_accepts_only_loaded_runs():
    handler = _handler()
    handler.runs = {"/run-a": SimpleNamespace(id="run-a")}

    handler.set_active_run_id("run-a")
    assert handler.get_active_run_id() == "run-a"

    with pytest.raises(RuntimeError, match="Run not found"):
        handler.set_active_run_id("removed")


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
