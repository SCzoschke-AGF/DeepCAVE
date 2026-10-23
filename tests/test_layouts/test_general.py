import importlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dash import no_update
from dash.exceptions import PreventUpdate

import deepcave
from deepcave.runs import Status
from deepcave.utils.cache import Cache


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
    monkeypatch.setattr(general_module, "c", Cache())
    monkeypatch.setattr(
        general_module,
        "app",
        SimpleNamespace(callback=register_callback),
    )
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
    options, selected_study, message, class_name, discovery, placeholder, disabled = callbacks[0](
        "optuna.db", str(tmp_path), None, None
    )

    assert options == [{"label": long_study_name, "value": long_study_name}]
    assert selected_study is None
    assert message == ""
    assert class_name == "d-none"
    assert discovery["studies"] == [long_study_name]
    assert placeholder == "Select a study (1 available)"
    assert disabled is False


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
    children, _clicks, hydration = callbacks[0](None, None, None, None)
    components = [component for child in children for component in _walk_components(child)]
    checkbox = next(
        component
        for component in components
        if getattr(component, "id", None)
        == {"type": "group-require-configspace-congruence", "index": 0}
    )

    assert checkbox.value is False
    assert hydration["congruence"] == {"comparison": False}


def test_initial_run_management_callback_preserves_plugin_inputs(monkeypatch):
    callbacks = []
    writes = []

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
        "c",
        SimpleNamespace(set=lambda *keys, **kwargs: writes.append((keys, kwargs))),
    )
    monkeypatch.setattr(
        general_module,
        "run_handler",
        SimpleNamespace(get_selected_run_paths=lambda: ["/run-a"]),
    )

    layout = object.__new__(general_module.GeneralLayout)
    layout._callback_manage_run()
    callbacks[0]([], [], None, [], [], None, None, 0)

    assert writes == []


def test_general_page_only_exposes_optuna_loading(general_callbacks):
    general_module, general_layout, _callbacks = general_callbacks
    layout = general_layout()

    card = next(component for component in layout if isinstance(component, general_module.dbc.Card))
    card_ids = {getattr(component, "id", None) for component in _walk_components(card)}
    assert {
        "general-optuna-database-input",
        "general-optuna-browse-button",
        "general-optuna-study-dropdown",
        "general-optuna-load-button",
        "general-optuna-database-status",
    } <= card_ids
    all_components = [component for child in layout for component in _walk_components(child)]
    all_ids = {getattr(component, "id", None) for component in all_components}
    assert "general-working-directory-input" not in all_ids
    assert "general-available-runs-container" not in all_ids
    assert "general-converter-label" not in all_ids
    assert not any(
        isinstance(component, general_module.html.Details) for component in all_components
    )
    working_directory = next(
        component
        for component in all_components
        if getattr(component, "id", None) == "general-working-directory"
    )
    assert isinstance(working_directory, general_module.dcc.Store)
    assert working_directory.data == str(Path.cwd())
    rows = [
        component
        for component in _walk_components(card)
        if getattr(component, "className", None) == "general-optuna-row"
    ]
    assert [[child.id for child in row.children] for row in rows] == [
        ["general-optuna-database-input", "general-optuna-browse-button"],
        ["general-optuna-study-dropdown", "general-optuna-load-button"],
    ]


@pytest.fixture
def general_callbacks(monkeypatch):
    for name in ("app", "c", "notification", "queue", "rc", "run_handler"):
        monkeypatch.setattr(deepcave, name, SimpleNamespace(), raising=False)
    module = importlib.import_module("deepcave.layouts.general")
    callbacks = []

    def register_callback(*_args, **_kwargs):
        def decorator(callback):
            callbacks.append(callback)
            return callback

        return decorator

    monkeypatch.setattr(module, "app", SimpleNamespace(callback=register_callback))
    monkeypatch.setattr(module, "c", Cache())
    monkeypatch.setattr(
        module, "run_handler", SimpleNamespace(get_working_directory=lambda: Path.cwd())
    )
    layout = object.__new__(module.GeneralLayout)
    return module, layout, callbacks


def _database(path, studies):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE studies (study_name TEXT)")
        connection.executemany("INSERT INTO studies VALUES (?)", [(name,) for name in studies])
    return path


def test_database_and_studies_are_restored_after_reload(general_callbacks, monkeypatch, tmp_path):
    module, layout, callbacks = general_callbacks
    database = _database(tmp_path / "optuna.db", ["first", "second"])
    layout._callback_discover_optuna_studies()
    discover = callbacks[0]
    discover(database.name, str(tmp_path), None, None)

    # Initial discovery with an empty input must not erase the remembered database.
    discover("", str(tmp_path), None, None)
    layout._callback_browse_optuna_database()
    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=None))
    opened, _directory, _entries, restored, error = callbacks[1](None, None, None, None, [], "")
    assert opened is False
    assert restored == str(database)
    assert error == ""
    options, selected, _message, _class, _discovery, _placeholder, disabled = discover(
        restored, str(tmp_path), None, None
    )
    assert options == [
        {"label": "first", "value": "first"},
        {"label": "second", "value": "second"},
    ]
    assert selected is None
    assert disabled is False


def test_database_restore_without_a_saved_path(general_callbacks, monkeypatch):
    module, layout, callbacks = general_callbacks
    layout._callback_browse_optuna_database()
    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=None))
    assert callbacks[0](None, None, None, None, [], "") == (
        False,
        no_update,
        no_update,
        "",
        "",
    )


def test_selected_run_table_uses_real_metadata_and_preserves_remove_targets(
    general_callbacks, monkeypatch
):
    module, layout, callbacks = general_callbacks
    paths = ["/data/run-a", "/data/run-b", "/data/unavailable"]
    long_name = "xgboost_full_study_name_" * 8
    names = {paths[0]: long_name, paths[1]: "legacy run", paths[2]: "unavailable run"}
    selected = paths.copy()
    monkeypatch.setattr(
        module,
        "run_handler",
        SimpleNamespace(
            runs={
                paths[0]: SimpleNamespace(
                    get_meta=lambda: {"optuna_sampler": "TPESampler"},
                    history=[
                        SimpleNamespace(status=status)
                        for status in (
                            Status.SUCCESS,
                            Status.FAILED,
                            Status.SUCCESS,
                            Status.PRUNED,
                        )
                    ],
                    configspace={"x": None, "y": None, "z": None},
                ),
                paths[1]: SimpleNamespace(get_meta=lambda: {}, history=[], configspace={}),
            },
            get_run_name=lambda path: names[path],
            get_selected_run_paths=lambda: selected.copy(),
            remove_run=lambda path: selected.remove(path),
        ),
    )
    monkeypatch.setattr(module, "c", SimpleNamespace(set=Mock()))
    layout._callback_display_selected_runs()
    table, count = callbacks[0](paths)
    rows = table.children[1].children
    assert count == "3 runs"
    assert rows[0].children[0].children[0].children == long_name
    assert len(rows[0].children[0].children) == 1
    assert [header.children for header in table.children[0].children.children] == [
        "Run Name",
        "Successful Trials",
        "Hyperparameters",
        "Sampler",
        "Actions",
    ]
    assert [cell.children for cell in rows[0].children[1:4]] == [2, 3, "TPESampler"]
    assert [cell.children for cell in rows[1].children[1:4]] == [0, 0, "Unknown"]
    assert [cell.children for cell in rows[2].children[1:4]] == ["—", "—", "—"]
    for index, row in enumerate(rows):
        button, store = row.children[4].children
        assert button.id == {"type": "general-dynamic-remove-run", "index": index}
        assert store.data == paths[index]

    layout._callback_manage_run()
    result = callbacks[1]([], [None, 1, None], None, [], paths, None, None, 0)
    assert result[2] == [paths[0], paths[2]]


@pytest.mark.parametrize("paths, count", [([], "0 runs"), (None, "0 runs"), (["/run"], "1 run")])
def test_selected_run_table_empty_state_and_count(general_callbacks, monkeypatch, paths, count):
    module, layout, callbacks = general_callbacks
    monkeypatch.setattr(
        module, "run_handler", SimpleNamespace(runs={}, get_run_name=lambda _path: "Run")
    )
    layout._callback_display_selected_runs()
    table, label = callbacks[0](paths)
    assert label == count
    if not paths:
        empty_cell = table.children[1].children[0].children
        assert empty_cell.children == "No runs selected."
        assert empty_cell.colSpan == 5


def test_database_change_resets_study_even_if_both_databases_contain_it(
    general_callbacks, tmp_path
):
    _module, layout, callbacks = general_callbacks
    layout._callback_discover_optuna_studies()
    discover = callbacks[0]
    first = _database(tmp_path / "first.db", ["shared", "first-only"])
    second = _database(tmp_path / "second.sqlite3", ["shared"])

    discovery = discover(str(first), str(tmp_path), None, None)[4]
    assert discover(str(first), str(tmp_path), "shared", discovery)[1] == "shared"
    changed = discover(str(second), str(tmp_path), "shared", discovery)
    assert changed[0] == [{"label": "shared", "value": "shared"}]
    assert changed[1] is None
    assert changed[6] is False
    # A textual path change also clears a selection when it resolves to the same file.
    assert discover(first.name, str(tmp_path), "shared", discovery)[1] is None


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("missing", "does not exist"),
        ("directory", "not a file"),
        ("invalid", "not a database"),
        ("no-schema", "no such table"),
        ("empty", "No Optuna studies"),
        ("unreadable", "not readable"),
        ("uri", "not a storage URI"),
    ],
)
def test_database_errors_are_shown_next_to_the_selector(
    general_callbacks, monkeypatch, tmp_path, kind, expected
):
    _module, layout, callbacks = general_callbacks
    layout._callback_discover_optuna_studies()
    database = tmp_path / "test.db"
    if kind == "directory":
        database.mkdir()
    elif kind == "invalid":
        database.write_text("invalid SQLite contents")
    elif kind == "no-schema":
        database.touch()
    elif kind == "empty":
        _database(database, [])
    elif kind == "unreadable":
        database.touch()
        monkeypatch.setattr("deepcave.utils.optuna.os.access", lambda *_args: False)
    elif kind == "uri":
        database = f"sqlite:///{database}"

    options, study, message, class_name, discovery, _placeholder, disabled = callbacks[0](
        str(database), str(tmp_path), "old-study", None
    )
    assert options == []
    assert study is None
    assert expected in message
    assert "text-danger" in class_name or "text-warning" in class_name
    assert discovery["error"] == message
    assert disabled is True


def test_empty_database_input_clears_discovery_without_an_initial_error(
    general_callbacks, tmp_path
):
    _module, layout, callbacks = general_callbacks
    layout._callback_discover_optuna_studies()
    result = callbacks[0]("", str(tmp_path), "old-study", {"database": "old.db"})
    assert result == ([], None, "", "d-none", None, "Select a study", True)


def test_browser_navigates_and_selects_sqlite_without_changing_the_working_directory(
    general_callbacks, monkeypatch, tmp_path
):
    module, layout, callbacks = general_callbacks
    handler = Mock(get_working_directory=Mock(return_value=tmp_path))
    monkeypatch.setattr(module, "run_handler", handler)
    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("a.db", "b.sqlite", "c.sqlite3", "upper.DB", "skip.txt", "a.db-wal"):
        (tmp_path / name).touch()
    database = _database(folder / "studies.sqlite3", ["a", "b"])
    layout._callback_browse_optuna_database()
    browse = callbacks[0]

    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id="general-optuna-browse-button"))
    opened, directory, entries, value, error = browse(1, None, None, None, [], "")
    assert opened is True
    assert directory == str(tmp_path)
    assert [entry.children[1] for entry in entries] == [
        "data",
        "a.db",
        "b.sqlite",
        "c.sqlite3",
        "upper.DB",
    ]
    assert value is no_update
    assert error == ""

    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=entries[0].id))
    navigated = browse(1, None, None, directory, [1, 0, 0, 0, 0], "")
    assert navigated[1] == str(folder)
    file_entry = navigated[2][0]
    assert file_entry.children[1] == database.name

    monkeypatch.setattr(
        module, "ctx", SimpleNamespace(triggered_id="general-optuna-browser-parent")
    )
    assert browse(1, 1, None, str(folder), [0], "")[1] == str(tmp_path)

    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=file_entry.id))
    selected = browse(1, 1, None, str(folder), [1], "")
    assert selected[0] is False
    assert selected[3] == str(database)
    assert selected[4] == ""

    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id="general-optuna-browse-button"))
    assert browse(2, 1, None, str(tmp_path), [], str(database))[1] == str(folder)
    handler.set_working_directory.assert_not_called()
    handler.remove_run.assert_not_called()


@pytest.mark.parametrize("action", ["manual", "cancel", "wildcard", "unreadable"])
def test_browser_handles_manual_navigation_cancellation_and_permissions(
    general_callbacks, monkeypatch, tmp_path, action
):
    module, layout, callbacks = general_callbacks
    monkeypatch.setattr(
        module, "run_handler", SimpleNamespace(get_working_directory=lambda: tmp_path)
    )
    layout._callback_browse_optuna_database()
    browse = callbacks[0]
    if action == "manual":
        trigger = "general-optuna-browser-directory"
    elif action == "cancel":
        trigger = "general-optuna-browser-close"
    elif action == "wildcard":
        trigger = {"type": "general-optuna-browser-entry", "path": str(tmp_path)}
    else:
        trigger = "general-optuna-browser-directory"

        def deny_listing(_path):
            raise PermissionError("Directory is not accessible")

        monkeypatch.setattr(Path, "iterdir", deny_listing)
    monkeypatch.setattr(module, "ctx", SimpleNamespace(triggered_id=trigger))
    if action == "wildcard":
        with pytest.raises(PreventUpdate):
            browse(1, None, None, str(tmp_path), [0], "keep.db")
        return
    result = browse(1, None, None, str(tmp_path), [], "keep.db")
    assert result[3] is no_update
    if action == "cancel":
        assert result[0] is False
    elif action == "unreadable":
        assert result[0] is True
        assert result[2] == []
        assert "not accessible" in result[4]
    else:
        assert result[1] == str(tmp_path)


def test_load_button_rejects_stale_discovery_and_preserves_refresh(general_callbacks, tmp_path):
    module, layout, callbacks = general_callbacks
    layout._callback_update_optuna_button()
    update = callbacks[0]
    database = _database(tmp_path / "study.db", ["study"])
    other = _database(tmp_path / "other.db", ["study"])
    discovery = {"database": str(database), "input": str(database), "studies": ["study"]}
    assert update(str(other), "study", str(tmp_path), 0, discovery) == ("Load Study", True)
    assert update(database.name, "study", str(tmp_path), 0, discovery) == ("Load Study", True)
    assert update(str(database), None, str(tmp_path), 0, discovery) == ("Load Study", True)
    assert update(str(database), "study", str(tmp_path), 0, discovery) == ("Load Study", False)

    snapshot = module.get_optuna_snapshot_path(database, "study", tmp_path)
    snapshot.mkdir(parents=True)
    (snapshot / module.SNAPSHOT_FILENAME).touch()
    assert update(str(database), "study", str(tmp_path), 1, discovery) == ("Refresh Study", False)


def test_loading_multiple_studies_preserves_existing_runs_and_database_selection(
    general_callbacks, monkeypatch, tmp_path
):
    module, layout, callbacks = general_callbacks
    layout._callback_manage_run()
    load = callbacks[0]
    database_dir = tmp_path / "data"
    database_dir.mkdir()
    database = _database(database_dir / "studies.db", ["first", "second"])
    selected_runs = ["/existing-run"]
    groups = {"existing-group": ["/existing-run"]}

    def add_run(path):
        selected_runs.append(path)
        return True

    handler = SimpleNamespace(
        get_working_directory=lambda: tmp_path,
        get_selected_run_paths=lambda: selected_runs.copy(),
        add_run=add_run,
        update_groups=Mock(),
        groups=groups,
    )
    monkeypatch.setattr(module, "run_handler", handler)
    monkeypatch.setattr(module, "c", SimpleNamespace(set=Mock()))
    monkeypatch.setattr(module, "notification", SimpleNamespace(update=Mock()))
    monkeypatch.setattr(
        module,
        "queue",
        SimpleNamespace(
            has_active_jobs_for_run=lambda _run_id: False, delete_inactive_jobs_for_run=Mock()
        ),
    )
    snapshot = Mock(
        return_value=SimpleNamespace(refreshed=False, imported_trials=1, skipped_trials=0)
    )
    monkeypatch.setattr(module, "create_optuna_snapshot", snapshot)

    first = load([], [], 1, [], selected_runs.copy(), str(database), "first", 0)
    second = load([], [], 1, [], selected_runs.copy(), str(database), "second", first[3])
    assert second[2] == [
        "/existing-run",
        str(module.get_optuna_snapshot_path(database, "first", tmp_path)),
        str(module.get_optuna_snapshot_path(database, "second", tmp_path)),
    ]
    assert second[3] == 2
    assert groups == {"existing-group": ["/existing-run"]}
    assert [call.args for call in snapshot.call_args_list] == [
        (database, "first", tmp_path),
        (database, "second", tmp_path),
    ]
