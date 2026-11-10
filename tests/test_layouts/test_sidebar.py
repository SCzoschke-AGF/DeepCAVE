import importlib
from types import SimpleNamespace

from dash import dcc, no_update

import deepcave
from deepcave.utils.cache import Cache
from deepcave.utils.sampling import SAMPLING_METHODS_CACHE_KEY, method_token


def test_configurations_plugin_is_not_exposed_in_ui():
    plugin_ids = [
        plugin.id for plugins in deepcave.config.PLUGINS.values() for plugin in plugins
    ]

    assert "configurations" not in plugin_ids


def test_budget_analysis_is_not_exposed_in_ui():
    assert "Budget Analysis" not in deepcave.config.PLUGINS
    plugin_ids = [
        plugin.id for plugins in deepcave.config.PLUGINS.values() for plugin in plugins
    ]
    assert "budget_correlation" not in plugin_ids


def test_multi_metrics_comparison_replaces_pareto_front_in_ui():
    plugin_ids = [
        plugin.id for plugins in deepcave.config.PLUGINS.values() for plugin in plugins
    ]

    assert "multi_metrics_comparison" in plugin_ids
    assert "pareto_front" not in plugin_ids


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
                {"label": "Accuracy", "value": "objective:0"},
                {"label": "latency", "value": "user_attr:latency"},
            ]
        ),
    )

    monkeypatch.setattr(sidebar_class, "register_callbacks", lambda self: None)
    sidebar = sidebar_class(deepcave.config.PLUGINS)
    layout = sidebar()
    components = list(_walk_components(layout))
    selector = next(
        component
        for component in components
        if getattr(component, "id", None) == "global-run-selector"
    )
    assert selector.value == "run-b"
    assert selector.persistence is True
    assert selector.persistence_type == "local"
    assert not any(
        getattr(component, "id", None) == "general-navigation-link"
        for component in components
    )
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

    monkeypatch.setattr(sidebar_class, "register_callbacks", lambda self: None)
    sidebar = sidebar_class(deepcave.config.PLUGINS)
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


def test_sampling_filter_restores_selection_uses_mapping_and_triggers_analysis_refresh(monkeypatch):
    callbacks = {}

    class FakeApp:
        @staticmethod
        def callback(*_args, **_kwargs):
            def register(function):
                callbacks[function.__name__] = function
                return function

            return register

        @staticmethod
        def clientside_callback(*_args, **_kwargs):
            pass

    module = importlib.import_module("deepcave.layouts.sidebar")
    monkeypatch.setattr(module, "app", FakeApp())
    cache = Cache(defaults={"optuna_sampler_attribute": "trial:sampling_method"})
    random, bo = method_token("random"), method_token("bo")
    options = [{"label": "random", "value": random}, {"label": "bo", "value": bo}]
    cache.set(SAMPLING_METHODS_CACHE_KEY, value=[random])
    handler = SimpleNamespace(
        get_sampling_method_options=lambda: options,
        set_sampling_methods=lambda methods: cache.set(
            SAMPLING_METHODS_CACHE_KEY, value=methods or []
        ),
        set_active_run_id=lambda run_id: None,
    )
    monkeypatch.setattr(deepcave, "c", cache, raising=False)
    monkeypatch.setattr(deepcave, "run_handler", handler, raising=False)
    sidebar = object.__new__(module.SidebarLayout)
    sidebar.register_callbacks()
    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id="global-update"))
    result = callbacks["sampling_methods"](0, "trial:sampling_method", [], [], None)
    assert result[0] == options and result[1] == [random] and not result[2]
    assert result[3] == {"attribute": "trial:sampling_method", "methods": [random]}
    assert (
        callbacks["sampling_methods"](1, "trial:sampling_method", [random], options, result[3])
        == (no_update,) * 5
    )
    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id="global-sampling-methods"))
    chosen = callbacks["sampling_methods"](1, "trial:sampling_method", [bo], options, result[3])
    assert cache.get(SAMPLING_METHODS_CACHE_KEY) == [bo]
    assert callbacks["select_global_run"]("study", chosen[3], 4) == 5
    cleared = callbacks["sampling_methods"](1, "trial:sampling_method", [], options, chosen[3])
    assert cleared[3]["methods"] == []


def test_sampling_filter_layout_is_available_without_a_mapping(monkeypatch):
    monkeypatch.setattr(deepcave, "app", SimpleNamespace(), raising=False)
    monkeypatch.setattr(deepcave, "queue", SimpleNamespace(), raising=False)
    module = importlib.import_module("deepcave.layouts.sidebar")
    monkeypatch.setattr(module.SidebarLayout, "register_callbacks", lambda self: None)
    monkeypatch.setattr(
        module.SidebarLayout, "_get_global_run_state", staticmethod(lambda: ([], None))
    )
    monkeypatch.setattr(
        module.SidebarLayout, "_get_global_objective_options", staticmethod(lambda run_id: [])
    )
    components = list(_walk_components(module.SidebarLayout({})()))
    filters = next(c for c in components if getattr(c, "id", None) == "global-trial-filters")
    assert filters.open is False
    assert filters.children[0].children == "Filters"
    dropdown = next(c for c in components if getattr(c, "id", None) == "global-sampling-methods")
    assert dropdown in list(_walk_components(filters))
    assert dropdown.multi and dropdown.disabled and dropdown.placeholder == "All methods"
