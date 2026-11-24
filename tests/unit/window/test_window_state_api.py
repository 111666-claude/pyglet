"""Tests for constructor style resolution and the BaseWindow state API.

No OpenGL context or native window is created: the state controller is
injected into a bare BaseWindow instance.
"""
from __future__ import annotations

import pytest

from pyglet.window import BaseWindow, WindowStateException
from pyglet.window.state import (
    RecordingNativeState,
    StateCapabilities,
    WindowState,
    WindowStateController,
)


class TestResolveWindowState:
    def test_default_style_is_plain_opaque_window(self):
        state = BaseWindow.resolve_window_state(BaseWindow.WINDOW_STYLE_DEFAULT)
        assert state == WindowState(False, False, False)

    def test_transparent_style_enables_transparency_only(self):
        state = BaseWindow.resolve_window_state(BaseWindow.WINDOW_STYLE_TRANSPARENT)
        assert state.transparent is True
        assert state.mouse_passthrough is False
        assert state.topmost is False

    def test_overlay_style_enables_all_three(self):
        state = BaseWindow.resolve_window_state(BaseWindow.WINDOW_STYLE_OVERLAY)
        assert state == WindowState(True, True, True)

    def test_explicit_false_overrides_style(self):
        state = BaseWindow.resolve_window_state(
            BaseWindow.WINDOW_STYLE_OVERLAY,
            transparent=False, mouse_passthrough=False, topmost=False,
        )
        assert state == WindowState(False, False, False)

    def test_explicit_true_decouples_features_from_style(self):
        # The headline feature: an ordinary style can be click-through, and
        # transparency can be requested without click-through.
        state = BaseWindow.resolve_window_state(
            BaseWindow.WINDOW_STYLE_DEFAULT,
            transparent=True, mouse_passthrough=True,
        )
        assert state.transparent is True
        assert state.mouse_passthrough is True
        assert state.topmost is False

    def test_none_style_with_explicit_values(self):
        state = BaseWindow.resolve_window_state(
            None, mouse_passthrough=True,
        )
        assert state == WindowState(False, True, False)


@pytest.fixture
def bare_window():
    """A BaseWindow instance with an injected recording state controller."""
    window = BaseWindow.__new__(BaseWindow)
    native = RecordingNativeState()
    window._window_state = WindowStateController(
        native,
        capabilities=StateCapabilities(alpha_framebuffer=True),
    )
    window._window_state.realize()
    return window, native


class TestBaseWindowStateAPI:
    def test_set_mouse_passthrough_method(self, bare_window):
        window, native = bare_window
        window.set_mouse_passthrough(True)
        assert window.mouse_passthrough is True
        assert native.calls == [("mouse_passthrough", True)]

        window.set_mouse_passthrough(False)
        assert window.mouse_passthrough is False
        assert ("mouse_passthrough", False) in native.calls

    def test_click_through_alias(self, bare_window):
        window, _ = bare_window
        window.set_click_through(True)
        assert window.click_through is True
        assert window.mouse_passthrough is True
        window.click_through = False
        assert window.click_through is False

    def test_transparent_property_setter(self, bare_window):
        window, native = bare_window
        window.transparent = True
        assert window.transparent is True
        window.set_transparent(False)
        assert window.transparent is False

    def test_topmost_property(self, bare_window):
        window, _ = bare_window
        window.topmost = True
        assert window.topmost is True
        window.set_topmost(False)
        assert window.topmost is False

    def test_transparent_supported_reflects_capability(self):
        window = BaseWindow.__new__(BaseWindow)
        window._window_state = WindowStateController(
            capabilities=StateCapabilities(alpha_framebuffer=False),
        )
        assert window.transparent_supported is False
        with pytest.raises(WindowStateException):
            window.set_transparent(True)

    def test_properties_exist_on_class(self):
        # Guard the public surface expected by applications/tests.
        for name in ("transparent", "mouse_passthrough", "click_through", "topmost"):
            assert isinstance(getattr(BaseWindow, name), property)
        assert callable(BaseWindow.set_transparent)
        assert callable(BaseWindow.set_mouse_passthrough)
        assert callable(BaseWindow.set_click_through)
        assert callable(BaseWindow.set_topmost)
