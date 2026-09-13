from typing import Any

import pytest
from dash.exceptions import PreventUpdate

from deepcave.plugins.static import StaticPlugin, _manage_callback_blocking


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
