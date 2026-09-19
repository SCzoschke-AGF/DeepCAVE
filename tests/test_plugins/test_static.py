from typing import Any

import pytest
from dash.exceptions import PreventUpdate

from deepcave.plugins.static import (
    PluginState,
    StaticPlugin,
    _manage_callback_blocking,
    _process_button_disabled,
    _should_refresh_outputs,
    _should_update_status_content,
)


def test_initial_page_load_refreshes_cached_outputs() -> None:
    assert _should_refresh_outputs(False, False, None) is True
    assert _should_refresh_outputs(False, False, "update-interval-output") is False


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
