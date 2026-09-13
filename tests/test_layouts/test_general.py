import importlib
from types import SimpleNamespace

import deepcave


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


def test_optuna_study_dropdown_uses_full_study_names(monkeypatch, tmp_path):
    callbacks = []

    def register_callback(*_args, **_kwargs):
        def decorator(callback):
            callbacks.append(callback)
            return callback

        return decorator

    monkeypatch.setattr(
        deepcave,
        "app",
        SimpleNamespace(callback=register_callback),
        raising=False,
    )
    for name in ("c", "notification", "queue", "rc", "run_handler"):
        monkeypatch.setattr(deepcave, name, SimpleNamespace(), raising=False)

    general_module = importlib.import_module("deepcave.layouts.general")
    monkeypatch.setattr(
        general_module,
        "resolve_optuna_database",
        lambda _value, _working_dir: tmp_path / "optuna.db",
    )
    long_study_name = "xgboost_data:4897daef3195_average_precision_v1_sampler:random_" * 2
    monkeypatch.setattr(
        general_module,
        "list_optuna_studies",
        lambda _database: [long_study_name],
    )

    layout = object.__new__(general_module.GeneralLayout)
    layout._callback_discover_optuna_studies()
    options, selected_study, _message, _class_name = callbacks[0]("optuna.db", str(tmp_path), None)

    assert options == [{"label": long_study_name, "value": long_study_name}]
    assert selected_study == long_study_name


def test_group_layout_restores_relaxed_configspace_setting(monkeypatch):
    callbacks = []

    def register_callback(*_args, **_kwargs):
        def decorator(callback):
            callbacks.append(callback)
            return callback

        return decorator

    general_module = importlib.import_module("deepcave.layouts.general")
    monkeypatch.setattr(
        general_module,
        "app",
        SimpleNamespace(callback=register_callback),
    )
    monkeypatch.setattr(
        general_module,
        "run_handler",
        SimpleNamespace(
            get_selected_groups=lambda: {"comparison": ["/run-a", "/run-b"]},
            get_selected_run_paths=lambda: ["/run-a", "/run-b"],
            get_run_name=lambda path: path.removeprefix("/"),
            get_groups_configspace_congruence=lambda: {"comparison": False},
        ),
    )

    layout = object.__new__(general_module.GeneralLayout)
    layout._callback_display_groups()
    children, _clicks = callbacks[0](None, None, None, None)
    components = [component for child in children for component in _walk_components(child)]
    checkbox = next(
        component
        for component in components
        if getattr(component, "id", None)
        == {"type": "group-require-configspace-congruence", "index": 0}
    )

    assert checkbox.value is False
