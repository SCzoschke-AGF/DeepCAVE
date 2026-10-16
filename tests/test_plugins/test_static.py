from typing import Any

from types import SimpleNamespace

import deepcave
import pytest
from dash.exceptions import PreventUpdate

from deepcave.plugins import static
from deepcave.plugins.static import (
    PluginState,
    StaticPlugin,
    _append_render_key,
    _applied_inputs_key,
    _job_is_active,
    _manage_callback_blocking,
    _processing_inputs_key,
    _process_button_disabled,
    _process_button_was_triggered,
    _delivery_attempt_due,
    _result_delivery_complete,
    _should_refresh_outputs,
    _should_update_status_content,
)


def test_initial_page_load_refreshes_cached_outputs() -> None:
    assert _should_refresh_outputs(False, False, None) is True
    assert _should_refresh_outputs(False, False, "update-interval-output") is False
    assert _should_refresh_outputs(
        False, False, "update-interval-output", delivery_pending=True
    ) is True


def test_initial_page_load_synchronizes_unchanged_button_content() -> None:
    assert _should_update_status_content(PluginState.READY, PluginState.READY, None) is True
    assert (
        _should_update_status_content(
            PluginState.READY, PluginState.READY, "importances-update-interval"
        )
        is False
    )


def test_cached_result_does_not_disable_recalculation() -> None:
    assert _process_button_disabled(PluginState.READY) is False
    assert _process_button_disabled(PluginState.NEEDS_PROCESSING) is False
    assert _process_button_disabled(PluginState.FAILED) is False
    assert _process_button_disabled(PluginState.PROCESSING) is True


def test_ready_button_waits_until_result_reached_browser_page() -> None:
    assert _result_delivery_complete(PluginState.READY, "current", None) is False
    assert _result_delivery_complete(PluginState.READY, "current", "old") is False
    assert _result_delivery_complete(PluginState.PROCESSING, "current", "current") is False
    assert _result_delivery_complete(PluginState.READY, "current", "current") is True


def test_browser_delivery_clears_spinner_even_if_ready_status_was_already_sent(monkeypatch):
    callbacks = []

    def register_callback(_output, _inputs):
        def register(callback):
            callbacks.append(callback)
            return callback

        return register

    monkeypatch.setattr(
        deepcave, "app", SimpleNamespace(callback=register_callback), raising=False
    )
    context = SimpleNamespace(triggered_id="analysis-update-interval")
    monkeypatch.setattr(static, "ctx", context)
    plugin = object.__new__(StaticPlugin)
    plugin.id = "analysis"
    plugin.button_caption = "Apply analysis"
    plugin._setup()
    plugin._state = PluginState.PROCESSING
    StaticPlugin._callback_loop_update_status_label.__wrapped__(plugin)
    update_status = callbacks[0]

    children, disabled = update_status(1, None)
    assert len(children) == 2
    assert disabled is True

    # A READY response may be superseded before the browser applies it, while the
    # shared server state already records READY and the browser still shows the spinner.
    plugin._state = PluginState.READY
    plugin._expected_render_key = "result"
    update_status(2, None)
    assert plugin._previous_state == PluginState.READY

    context.triggered_id = "analysis-rendered-inputs-key"
    children, disabled = update_status(2, "result")
    assert len(children) == 1
    assert children[0].children == "Apply analysis"
    assert disabled is False

    context.triggered_id = "analysis-update-interval"
    children, disabled = update_status(3, "result")
    assert len(children) == 1
    assert children[0].children == "Apply analysis"
    assert disabled is False


def test_render_key_keeps_single_list_like_output_atomic() -> None:
    component_children = ["first child", "second child"]

    assert _append_render_key(component_children, 1, "key") == [component_children, "key"]
    assert _append_render_key(["left", "right"], 2, "key") == ["left", "right", "key"]


def test_cached_render_failure_allows_recalculation(monkeypatch) -> None:
    callbacks = []

    def register_callback(_output, _inputs):
        def register(callback):
            callbacks.append(callback)
            return callback

        return register

    monkeypatch.setattr(deepcave, "app", SimpleNamespace(callback=register_callback), raising=False)
    monkeypatch.setattr(deepcave, "c", SimpleNamespace(get=lambda *_args: None), raising=False)
    monkeypatch.setattr(
        deepcave, "rc", SimpleNamespace(get=lambda *_args: {"legacy": True}), raising=False
    )
    monkeypatch.setattr(deepcave, "run_handler", SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        deepcave,
        "queue",
        SimpleNamespace(
            get_finished_jobs=lambda: [],
            is_running=lambda _id: False,
            is_pending=lambda _id: False,
            has_failed=lambda _id: False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        static, "ctx", SimpleNamespace(triggered_id="analysis-update-interval-output")
    )
    plugin = object.__new__(StaticPlugin)
    plugin.id = "analysis"
    plugin.inputs = []
    plugin.outputs = [("content", "children")]
    plugin._setup()
    plugin._list_to_dict = lambda _values, input=True: {}
    plugin._clean_inputs = lambda _inputs: {}
    plugin.get_selected_runs = lambda _inputs: [SimpleNamespace(id="run")]
    plugin.check_runs_compatibility = lambda _runs: None

    def render(*_args):
        raise KeyError("silhouette")

    plugin._process_raw_outputs = render
    StaticPlugin._callback_inputs_changed.__wrapped__(plugin)

    with pytest.raises(KeyError, match="silhouette"):
        callbacks[0](None, 0, "run", None, "page")

    assert plugin._state == PluginState.FAILED
    assert plugin._expected_render_key is None
    assert _process_button_disabled(plugin._state) is False
    assert plugin._blocked is False


def test_result_delivery_is_not_duplicated_while_first_response_is_in_flight() -> None:
    attempts = {("page", "result"): 5.0}

    assert _delivery_attempt_due("page", "result", None, attempts, 6.0) is False
    assert _delivery_attempt_due("page", "result", None, attempts, 16.0) is True
    assert _delivery_attempt_due("page", "result", "result", attempts, 16.0) is False


def test_interval_waits_for_browser_delivery_without_blocking_other_pages(monkeypatch) -> None:
    callbacks = []

    def register_callback(_output, _inputs):
        def register(callback):
            callbacks.append(callback)
            return callback

        return register

    monkeypatch.setattr(
        deepcave, "app", SimpleNamespace(callback=register_callback), raising=False
    )
    monkeypatch.setattr(static, "monotonic", lambda: 6.0)
    plugin = object.__new__(StaticPlugin)
    plugin.id = "analysis"
    plugin._setup()
    plugin._state = PluginState.READY
    plugin._expected_render_key = "result"
    plugin._render_attempts = {("page", "result"): 5.0}
    StaticPlugin._callback_loop_trigger_main_loop.__wrapped__(plugin)
    poll = callbacks[0]

    # The main callback has returned, but its response is still in transit. A new
    # interval trigger would make Dash discard that response as superseded.
    assert plugin._blocked is False
    with pytest.raises(PreventUpdate):
        poll(1, 0, None, "page")
    with pytest.raises(PreventUpdate):
        poll(2, 0, "previous-result", "page")

    assert poll(1, 0, None, "other-page") == 1
    assert poll(3, 0, "result", "page") == 1

    # A lost response can still be retried, and running jobs continue to be polled.
    monkeypatch.setattr(static, "monotonic", lambda: 16.0)
    assert poll(4, 0, None, "page") == 1
    monkeypatch.setattr(static, "monotonic", lambda: 6.0)
    plugin._state = PluginState.PROCESSING
    assert poll(5, 0, None, "page") == 1
    plugin._state = PluginState.READY
    plugin._render_attempts.clear()
    assert poll(6, 0, None, "page") == 1


def test_only_the_button_callback_starts_processing() -> None:
    button_id = "analysis-update-button"

    assert _process_button_was_triggered(1, button_id, button_id) is True
    assert _process_button_was_triggered(1, "analysis-update-interval-output", button_id) is False
    assert _process_button_was_triggered(None, button_id, button_id) is False


def test_applied_inputs_key_tracks_run_and_processing_values_only() -> None:
    plugin = object.__new__(StaticPlugin)
    plugin.inputs = [
        ("cluster_count", "value", False, str),
        ("cluster_count", "options", False, None),
    ]
    applied = {
        "run": {"value": "run-a"},
        "cluster_count": {"value": "4", "options": [1, 2, 3, 4]},
    }
    transient = {
        "run": {"value": "run-a"},
        "cluster_count": {"value": "4", "options": []},
    }
    changed_cluster = {
        "run": {"value": "run-a"},
        "cluster_count": {"value": "1", "options": [1, 2, 3, 4]},
    }
    changed_run = {
        "run": {"value": "run-b"},
        "cluster_count": {"value": "4", "options": [1, 2, 3, 4]},
    }

    assert _applied_inputs_key(plugin, applied) == _applied_inputs_key(plugin, transient)
    assert _applied_inputs_key(plugin, applied) != _applied_inputs_key(plugin, changed_cluster)
    assert _applied_inputs_key(plugin, applied) != _applied_inputs_key(plugin, changed_run)


def test_submitted_job_stays_active_between_queue_registries() -> None:
    class QueueWithoutVisibleJob:
        @staticmethod
        def is_running(_job_id: str) -> bool:
            return False

        @staticmethod
        def is_pending(_job_id: str) -> bool:
            return False

    queue = QueueWithoutVisibleJob()

    assert _job_is_active(queue, "job-1", {"job-1"}) is True
    assert _job_is_active(queue, "job-1", set()) is False


def test_processing_key_ignores_transient_ui_metadata() -> None:
    plugin = object.__new__(StaticPlugin)
    plugin.inputs = [
        ("top_percent", "value", False, float),
        ("top_percent", "options", False, None),
        ("top_percent", "disabled", False, None),
    ]
    first = {
        "top_percent": {"value": 5, "options": [1, 5, 10], "disabled": False},
        "run": {"value": "run-a"},
    }
    transient = {
        "top_percent": {"value": 5, "options": [], "disabled": True},
        "run": {"value": "run-b"},
    }
    changed = {
        "top_percent": {"value": 10, "options": [1, 5, 10], "disabled": False},
        "run": {"value": "run-a"},
    }

    assert _processing_inputs_key(plugin, first) == _processing_inputs_key(plugin, transient)
    assert _processing_inputs_key(plugin, first) != _processing_inputs_key(plugin, changed)
    old_key = _processing_inputs_key(plugin, first)
    plugin.processing_version = 1
    assert _processing_inputs_key(plugin, first) != old_key


def test_callback_blocking_is_released_after_prevent_update() -> None:
    plugin = object.__new__(StaticPlugin)
    plugin._blocked = False

    @_manage_callback_blocking(plugin)
    def callback() -> Any:
        assert plugin._blocked is True
        raise PreventUpdate

    with pytest.raises(PreventUpdate):
        callback()

    assert plugin._blocked is False


def test_callback_blocking_is_released_after_return() -> None:
    plugin = object.__new__(StaticPlugin)
    plugin._blocked = False

    @_manage_callback_blocking(plugin)
    def callback() -> str:
        assert plugin._blocked is True
        return "done"

    assert callback() == "done"
    assert plugin._blocked is False
