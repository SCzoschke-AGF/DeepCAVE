import importlib
import json
import re
from types import SimpleNamespace

from dash import Patch, html, no_update

import deepcave

_temporary_globals = []
for _name in ("app", "c", "notification", "queue", "rc", "run_handler"):
    if not hasattr(deepcave, _name):
        setattr(deepcave, _name, SimpleNamespace())
        _temporary_globals.append(_name)

main_module = importlib.import_module("deepcave.layouts.main")
for _name in _temporary_globals:
    delattr(deepcave, _name)


class _FakeApp:
    def __init__(self):
        self.callbacks = []
        self.clientside_callbacks = []

    def callback(self, *_args, **_kwargs):
        def decorator(callback):
            self.callbacks.append(callback)
            return callback

        return decorator

    def clientside_callback(self, *args, **_kwargs):
        self.clientside_callbacks.append(args)


class _Plugin:
    name = "Analysis"

    def __init__(self, plugin_id):
        self.plugin_id = plugin_id
        self.calls = 0

    def check_run_compatibility(self, _run):
        return True

    def __call__(self):
        self.calls += 1
        return [html.Div(self.plugin_id)]


def _state():
    return {
        "run_id": None,
        "data_revision": None,
        "mounted": [],
        "statuses": {},
        "invalidation": {},
        "dirty": False,
    }


def test_active_analysis_is_mounted_first_and_warm_pages_are_not_remounted(monkeypatch):
    fake_app = _FakeApp()
    run = SimpleNamespace(id="run-a", latest_change=3.0, history=[1, 2], get_objectives=lambda: [])
    monkeypatch.setattr(main_module, "app", fake_app)
    monkeypatch.setattr(
        main_module,
        "run_handler",
        SimpleNamespace(set_active_run_id=lambda _run_id: None, get_run=lambda _run_id: run),
    )
    layout = object.__new__(main_module.MainLayout)
    plugins = {plugin_id: _Plugin(plugin_id) for plugin_id in main_module.PERSISTENT_ANALYSIS_IDS}
    layout.plugins = plugins
    layout.register_callbacks()
    mount = next(
        callback
        for callback in fake_app.callbacks
        if callback.__name__ == "mount_persistent_analysis"
    )

    first_children, first_state = mount(
        "/plugins/one_dimensional_analysis", 0, 0, [], 0, 0, "run-a", _state()
    )
    assert len(first_children) == 1
    assert first_state["mounted"] == ["one_dimensional_analysis"]
    assert plugins["one_dimensional_analysis"].calls == 1
    assert plugins["two_dimensional_analysis"].calls == 0

    second_children, second_state = mount(
        "/plugins/two_dimensional_analysis", 0, 0, [], 0, 0, "run-a", first_state
    )
    assert isinstance(second_children, Patch)
    assert second_state["mounted"] == [
        "one_dimensional_analysis",
        "two_dimensional_analysis",
    ]
    assert plugins["one_dimensional_analysis"].calls == 1
    assert plugins["two_dimensional_analysis"].calls == 1

    children, state = mount(
        "/plugins/one_dimensional_analysis", 0, 0, [], 0, 0, "run-a", second_state
    )
    assert children is no_update
    assert state is no_update
    assert plugins["one_dimensional_analysis"].calls == 1


def test_coverage_mounts_without_workers_and_keeps_a_warm_page(monkeypatch):
    fake_app = _FakeApp()
    run = SimpleNamespace(id="run-a", latest_change=1, history=[1], get_objectives=lambda: [])
    plugin = _Plugin("search_space_coverage")
    monkeypatch.setattr(main_module, "app", fake_app)
    monkeypatch.setattr(main_module, "queue", SimpleNamespace(ready=lambda: False))
    monkeypatch.setattr(
        main_module,
        "run_handler",
        SimpleNamespace(set_active_run_id=lambda _: None, get_run=lambda _: run),
    )
    layout = object.__new__(main_module.MainLayout)
    layout.plugins = {"search_space_coverage": plugin}
    layout.register_callbacks()
    mount = next(
        callback
        for callback in fake_app.callbacks
        if callback.__name__ == "mount_persistent_analysis"
    )
    children, state = mount("/plugins/search_space_coverage", 0, 0, [], 0, 0, run.id, _state())
    assert len(children) == 1
    assert state["mounted"] == ["search_space_coverage"]
    assert mount("/plugins/search_space_coverage", 0, 0, [], 0, 0, run.id, state) == (
        no_update,
        no_update,
    )
    assert plugin.calls == 1


def test_coverage_overview_round_trip_uses_the_same_persistent_routes_in_browser(monkeypatch):
    fake_app = _FakeApp()
    run = SimpleNamespace(id="run-a", latest_change=1, history=[1], get_objectives=lambda: [])
    coverage = _Plugin("search_space_coverage")
    overview = _Plugin("run_overview")
    monkeypatch.setattr(main_module, "app", fake_app)
    monkeypatch.setattr(main_module, "queue", SimpleNamespace(ready=lambda: True))
    monkeypatch.setattr(
        main_module,
        "run_handler",
        SimpleNamespace(
            set_active_run_id=lambda _: None,
            get_run=lambda _: run,
        ),
    )
    layout = object.__new__(main_module.MainLayout)
    layout.plugins = {"search_space_coverage": coverage, "run_overview": overview}
    layout.register_callbacks()
    mount = next(c for c in fake_app.callbacks if c.__name__ == "mount_persistent_analysis")
    display = next(c for c in fake_app.callbacks if c.__name__ == "display_page")

    _, state = mount("/plugins/search_space_coverage", 0, 0, [], 0, 0, run.id, _state())
    assert display("/plugins/run_overview", 0, run.id)[0].children == "run_overview"
    assert mount("/plugins/search_space_coverage", 0, 0, [], 0, 0, run.id, state) == (
        no_update,
        no_update,
    )
    assert display("/plugins/search_space_coverage", 0, run.id) is no_update
    assert coverage.calls == overview.calls == 1

    visibility = next(
        args[0]
        for args in fake_app.clientside_callbacks
        if isinstance(args[0], str) and "const persistentIds" in args[0]
    )
    match = re.search(r"const persistentIds = new Set\((\[[^]]*\])\);", visibility)
    assert match is not None
    routes = json.loads(match.group(1))
    assert routes == list(main_module.PERSISTENT_ANALYSIS_IDS)
    assert "search_space_coverage" in routes
    assert "run_overview" not in routes


def test_same_run_data_invalidation_replaces_warm_analysis_context(monkeypatch):
    fake_app = _FakeApp()
    run = SimpleNamespace(id="run-a", latest_change=3.0, history=[1], get_objectives=lambda: [])
    plugin = _Plugin("two_dimensional_analysis")
    monkeypatch.setattr(main_module, "app", fake_app)
    monkeypatch.setattr(
        main_module,
        "run_handler",
        SimpleNamespace(set_active_run_id=lambda _run_id: None, get_run=lambda _run_id: run),
    )
    layout = object.__new__(main_module.MainLayout)
    layout.plugins = {"two_dimensional_analysis": plugin}
    layout.register_callbacks()
    mount = next(
        callback
        for callback in fake_app.callbacks
        if callback.__name__ == "mount_persistent_analysis"
    )

    _, mounted = mount("/plugins/two_dimensional_analysis", 0, 0, [], 0, 0, "run-a", _state())
    children, refreshed = mount(
        "/plugins/two_dimensional_analysis", 0, 1, [], 0, 0, "run-a", mounted
    )

    assert isinstance(children, list)
    assert refreshed["mounted"] == ["two_dimensional_analysis"]
    assert plugin.calls == 2
