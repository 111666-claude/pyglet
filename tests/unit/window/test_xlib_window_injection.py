"""Platform-window level test with injected X11 native calls.

This drives the *public* BaseWindow/XlibWindow methods but replaces the
Xlib/XShape modules with fakes, proving the full
public API -> controller -> platform driver -> native call path without a
display or GL context.  Skipped on platforms where the X11 backend cannot be
imported.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

xlib_window_mod = pytest.importorskip("pyglet.window.xlib")
XlibWindow = xlib_window_mod.XlibWindow

from pyglet.window import WindowStateException  # noqa: E402
from pyglet.window.state import (  # noqa: E402
    StateCapabilities,
    WindowStateController,
    X11NativeState,
)


def make_bare_xlib_window(passthrough_supported: bool = True):
    window = XlibWindow.__new__(XlibWindow)

    fake_xlib = SimpleNamespace(
        XCreateRegion=MagicMock(return_value=1234),
        XDestroyRegion=MagicMock(),
    )
    fake_xshape = SimpleNamespace(
        ShapeInput=2,
        ShapeSet=0,
        XShapeCombineRegion=MagicMock(),
        XShapeCombineMask=MagicMock(),
    )
    wm_state = MagicMock()
    driver = X11NativeState(
        fake_xlib, fake_xshape,
        display=9, window=111, view=222, change_wm_state=wm_state,
    )
    window._window_state = WindowStateController(
        driver,
        capabilities=StateCapabilities(
            alpha_framebuffer=True,
            mouse_passthrough=passthrough_supported,
        ),
    )
    window._window_state.realize()  # all-off baseline -> no native calls yet
    return window, fake_xlib, fake_xshape, wm_state


def test_public_set_mouse_passthrough_shapes_both_x_windows():
    win, fake_xlib, fake_xshape, _ = make_bare_xlib_window()

    win.set_mouse_passthrough(True)

    assert win.mouse_passthrough is True
    fake_xlib.XCreateRegion.assert_called_once()
    shaped_xids = [c.args[1] for c in fake_xshape.XShapeCombineRegion.call_args_list]
    assert shaped_xids == [111, 222]

    win.set_mouse_passthrough(False)
    masked_xids = [c.args[1] for c in fake_xshape.XShapeCombineMask.call_args_list]
    assert masked_xids == [111, 222]


def test_public_click_through_alias():
    win, _, fake_xshape, _ = make_bare_xlib_window()
    win.click_through = True
    assert win.click_through is True
    assert win.mouse_passthrough is True
    assert fake_xshape.XShapeCombineRegion.called


def test_unsupported_shape_raises_via_public_api():
    win, _, _, _ = make_bare_xlib_window(passthrough_supported=False)
    with pytest.raises(WindowStateException):
        win.set_mouse_passthrough(True)
    assert win.mouse_passthrough is False


def test_topmost_uses_ewmh_callback_through_public_api():
    win, _, _, wm_state = make_bare_xlib_window()
    win.set_topmost(True)
    wm_state.assert_called_once_with("_NET_WM_STATE_ABOVE", True)
    win.topmost = False
    wm_state.assert_called_with("_NET_WM_STATE_ABOVE", False)


def test_close_teardown_clears_input_shape():
    win, _, fake_xshape, _ = make_bare_xlib_window()
    win.set_mouse_passthrough(True)
    fake_xshape.XShapeCombineMask.reset_mock()

    # Platform close() calls the controller teardown before destroying XIDs.
    win._window_state.teardown()

    assert fake_xshape.XShapeCombineMask.called
    assert win._window_state.closed is True
