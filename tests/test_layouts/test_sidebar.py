import importlib
from types import SimpleNamespace

from dash import dcc

import deepcave


def test_configurations_plugin_is_not_exposed_in_ui():
    plugin_ids = [
        plugin.id for plugins in deepcave.config.PLUGINS.values() for plugin in plugins
    ]

    assert "configurations" not in plugin_ids


def _walk_components(component):
    """Yield a Dash component and all descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from _walk_components(child)


def test_global_run_selector_survives_page_navigation(monkeypatch):
    # Sidebar imports these runtime-only globals. They are deliberately absent in API/test mode.
    monkeypatch.setattr(deepcave, "app", SimpleNamespace(), raising=False)
    monkeypatch.setattr(deepcave, "queue", SimpleNamespace(), raising=False)
    sidebar_module = importlib.import_module("deepcave.layouts.sidebar")
    sidebar_class = sidebar_module.SidebarLayout
    monkeypatch.setattr(
        sidebar_class,
        "_get_global_run_state",
        staticmethod(lambda: ([{"label": "Run B", "value": "run-b"}], "run-b")),
    )
    monkeypatch.setattr(
        sidebar_class,
        "_get_global_objective_options",
        staticmethod(
            lambda _run_id: [
                {"label": "Objective · Accuracy", "value": "objective:0"},
                {"label": "User attr · latency", "value": "user_attr:latency"},
            ]
        ),
    )

    sidebar = object.__new__(sidebar_class)
    layout = sidebar()
    components = list(_walk_components(layout))
    selector = next(
        component
        for component in components
        if getattr(component, "id", None) == "global-run-selector"
    )
    general_link = next(
        component
        for component in components
        if getattr(component, "id", None) == "general-navigation-link"
    )

    assert selector.value == "run-b"
    assert selector.persistence is True
    assert selector.persistence_type == "local"
    assert isinstance(general_link, dcc.Link)
    objective_selector = next(
        component
        for component in components
        if getattr(component, "id", None) == "global-main-objective-selector"
    )
    direction = next(
        component
        for component in components
        if getattr(component, "id", None) == "global-objective-direction"
    )

    assert objective_selector.value == "objective:0"
    assert objective_selector.persistence is True
    assert objective_selector.persistence_type == "local"
    assert direction.persistence is True
    assert direction.persistence_type == "local"


def test_global_objective_controls_are_available_without_a_runtime(monkeypatch):
    monkeypatch.setattr(deepcave, "app", SimpleNamespace(), raising=False)
    monkeypatch.setattr(deepcave, "queue", SimpleNamespace(), raising=False)
    sidebar_class = importlib.import_module("deepcave.layouts.sidebar").SidebarLayout
    monkeypatch.setattr(
        sidebar_class,
        "_get_global_run_state",
        staticmethod(lambda: ([{"label": "Run B", "value": "run-b"}], "run-b")),
    )

    sidebar = object.__new__(sidebar_class)
    components = list(_walk_components(sidebar()))

    assert any(
        getattr(component, "id", None) == "global-main-objective-selector"
        for component in components
    )


def test_main_objective_options_are_not_refreshed_by_the_periodic_timer(monkeypatch):
    callbacks = []

    class FakeApp:
        @staticmethod
        def callback(*dependencies, **_kwargs):
            def decorator(callback):
                callbacks.append((callback, dependencies))
                return callback

            return decorator

        @staticmethod
        def clientside_callback(*_args, **_kwargs):
            return None

    sidebar_module = importlib.import_module("deepcave.layouts.sidebar")
    monkeypatch.setattr(sidebar_module, "app", FakeApp())

    sidebar = object.__new__(sidebar_module.SidebarLayout)
    sidebar.register_callbacks()

    _, dependencies = next(
        entry for entry in callbacks if entry[0].__name__ == "refresh_global_main_objective"
    )

    def flatten(items):
        for item in items:
            if isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    inputs = [
        dependency
        for dependency in flatten(dependencies)
        if dependency.__class__.__name__ == "Input"
    ]
    assert [(item.component_id, item.component_property) for item in inputs] == [
        ("global-run-selector", "value")
    ]
