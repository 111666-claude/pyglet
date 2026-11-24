"""Tests for the platform native-state drivers with injected native calls.

Every native library / object (Win32 user32/dwmapi, the Cocoa NSWindow, the
Xlib/XShape modules) is replaced with a fake, so the exact state-switch
call sequences for all three platforms can be verified on any OS.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pyglet.window.state import (
    CocoaNativeState,
    Win32NativeState,
    WindowState,
    X11NativeState,
)


# ---------------------------------------------------------------------- #
# Win32
# ---------------------------------------------------------------------- #
WIN32_CONST = SimpleNamespace(
    GWL_EXSTYLE=-20,
    WS_EX_TRANSPARENT=0x20,
    WS_EX_LAYERED=0x80000,
    LWA_ALPHA=0x2,
    HWND_TOPMOST=-1,
    HWND_NOTOPMOST=-2,
    SWP_NOSIZE=0x1,
    SWP_NOMOVE=0x2,
    SWP_NOACTIVATE=0x10,
    DWM_BB_ENABLE=0x1,
    DWM_BB_BLURREGION=0x2,
)


class FakeBlurBehind:
    def __init__(self):
        self.dwFlags = 0
        self.hRgnBlur = None
        self.fEnable = False


def make_win32_driver(base_ex_style=0x100):  # arbitrary WS_EX_ style bit
    user32 = MagicMock(name="user32")
    dwmapi = MagicMock(name="dwmapi")
    gdi32 = MagicMock(name="gdi32")
    gdi32.CreateRectRgn.return_value = 0xDEADBEEF
    driver = Win32NativeState(
        user32, dwmapi, gdi32, WIN32_CONST,
        blurbehind_type=FakeBlurBehind, byref=lambda obj: obj,
        hwnd=111, view_hwnd=222, base_ex_style=base_ex_style,
    )
    return driver, user32, dwmapi, gdi32


class TestWin32Driver:
    def test_overlay_initial_sync_sets_layered_transparent_topmost(self):
        d, user32, dwmapi, gdi32 = make_win32_driver()
        d.sync(WindowState(True, True, True))

        expected_ex = 0x100 | WIN32_CONST.WS_EX_LAYERED | WIN32_CONST.WS_EX_TRANSPARENT
        user32.SetWindowLongW.assert_any_call(111, WIN32_CONST.GWL_EXSTYLE, expected_ex)
        # Child view window receives the hit-test-through bit.
        user32.SetWindowLongW.assert_any_call(222, WIN32_CONST.GWL_EXSTYLE,
                                              WIN32_CONST.WS_EX_TRANSPARENT)
        # DWM blur-behind enabled + layered alpha attributes preserved.
        dwmapi.DwmEnableBlurBehindWindow.assert_called_once()
        user32.SetLayeredWindowAttributes.assert_called_once_with(111, 0, 255, WIN32_CONST.LWA_ALPHA)
        gdi32.DeleteObject.assert_called_once_with(0xDEADBEEF)
        # Topmost z-order.
        args, kwargs = user32.SetWindowPos.call_args
        assert args[1] == WIN32_CONST.HWND_TOPMOST

    def test_passthrough_off_keeps_layered_for_transparency_and_clears_child(self):
        d, user32, _, _ = make_win32_driver()
        d.sync(WindowState(True, True, True))
        user32.reset_mock()

        d.sync(WindowState(True, False, True))

        ex = d.ex_style
        assert ex & WIN32_CONST.WS_EX_LAYERED
        assert not (ex & WIN32_CONST.WS_EX_TRANSPARENT)
        user32.SetWindowLongW.assert_any_call(111, WIN32_CONST.GWL_EXSTYLE, ex)
        user32.SetWindowLongW.assert_any_call(222, WIN32_CONST.GWL_EXSTYLE, 0)

    def test_transparent_off_while_passthrough_on_keeps_layered(self):
        d, user32, dwmapi, _ = make_win32_driver()
        d.sync(WindowState(True, True, False))
        dwmapi.reset_mock()

        d.sync(WindowState(False, True, False))

        assert d.ex_style & WIN32_CONST.WS_EX_LAYERED
        # Blur-behind gets a disabled structure.
        bb = dwmapi.DwmEnableBlurBehindWindow.call_args.args[1]
        assert bb.fEnable is False

    def test_everything_off_restores_base_style_and_notopmost(self):
        base = 0x100
        d, user32, _, _ = make_win32_driver(base_ex_style=base)
        d.sync(WindowState(True, True, True))
        d.sync(WindowState(False, False, False))

        assert d.ex_style == base
        last_ex_call = [c for c in user32.SetWindowLongW.call_args_list
                        if c.args[0] == 111][-1]
        assert last_ex_call.args[2] == base
        last_pos = user32.SetWindowPos.call_args
        assert last_pos.args[1] == WIN32_CONST.HWND_NOTOPMOST

    def test_idempotent_sync_makes_no_extra_calls(self):
        d, user32, dwmapi, _ = make_win32_driver()
        state = WindowState(True, True, False)
        d.sync(state)
        calls_after_first = user32.SetWindowLongW.call_count
        dwm_count = dwmapi.DwmEnableBlurBehindWindow.call_count

        d.sync(state)
        assert user32.SetWindowLongW.call_count == calls_after_first
        assert dwmapi.DwmEnableBlurBehindWindow.call_count == dwm_count

    def test_attach_resets_diff_history_for_new_hwnd(self):
        d, user32, _, _ = make_win32_driver()
        d.sync(WindowState(topmost=True))
        user32.SetWindowPos.reset_mock()

        # Fullscreen recreation gives a new HWND; attach forces re-application.
        d.attach(333, 444, 0x100)
        d.sync(WindowState(topmost=True))
        assert user32.SetWindowPos.call_args.args[0] == 333


# ---------------------------------------------------------------------- #
# Cocoa
# ---------------------------------------------------------------------- #
class FakeNSWindow:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name):
        def recorder(*args):
            self.calls.append((name, args))
        return recorder

    def methods(self):
        return [name for name, _ in self.calls]

    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)


class TestCocoaDriver:
    def make_driver(self):
        window = FakeNSWindow()
        clear = object()
        opaque = object()
        driver = CocoaNativeState(
            window, normal_level=0, status_level=25,
            clear_color=lambda: clear, opaque_color=lambda: opaque,
        )
        return driver, window, clear, opaque

    def test_transparent_on_configures_non_opaque_clear_shadowless(self):
        d, w, clear, _ = self.make_driver()
        d.sync(WindowState(transparent=True))

        assert ("setOpaque_", (False,)) in w.calls
        assert ("setBackgroundColor_", (clear,)) in w.calls
        assert ("setHasShadow_", (False,)) in w.calls

    def test_transparent_off_restores_opaque_background_and_shadow(self):
        d, w, _, opaque = self.make_driver()
        d.sync(WindowState(transparent=True))
        d.sync(WindowState(transparent=False))

        assert ("setOpaque_", (True,)) in w.calls
        assert ("setBackgroundColor_", (opaque,)) in w.calls
        assert ("setHasShadow_", (True,)) in w.calls

    def test_passthrough_calls_setIgnoresMouseEvents(self):
        d, w, _, _ = self.make_driver()
        d.sync(WindowState(mouse_passthrough=True))
        assert ("setIgnoresMouseEvents_", (True,)) in w.calls

        d.sync(WindowState(mouse_passthrough=False))
        assert ("setIgnoresMouseEvents_", (False,)) in w.calls

    def test_topmost_uses_status_level(self):
        d, w, _, _ = self.make_driver()
        d.sync(WindowState(topmost=True))
        assert ("setLevel_", (25,)) in w.calls
        d.sync(WindowState(topmost=False))
        assert ("setLevel_", (0,)) in w.calls

    def test_unchanged_aspects_are_not_reapplied(self):
        d, w, _, _ = self.make_driver()
        d.sync(WindowState(transparent=True, mouse_passthrough=True))
        n_opaque = w.count("setOpaque_")
        n_ignore = w.count("setIgnoresMouseEvents_")
        # topmost stayed at the all-off baseline -> no level call yet.
        assert w.count("setLevel_") == 0

        # Only topmost changes: transparency/ignore calls must not repeat.
        d.sync(WindowState(transparent=True, mouse_passthrough=True, topmost=True))
        assert w.count("setOpaque_") == n_opaque
        assert w.count("setIgnoresMouseEvents_") == n_ignore
        assert [args for name, args in w.calls if name == "setLevel_"] == [(25,)]

    def test_plain_off_state_produces_no_native_calls(self):
        d, w, _, _ = self.make_driver()
        d.sync(WindowState())
        assert w.calls == []

    def test_attach_to_fresh_window_reapplies_everything(self):
        d, w1, _, _ = self.make_driver()
        d.sync(WindowState(mouse_passthrough=True))
        w2 = FakeNSWindow()
        d.attach(w2)
        d.sync(WindowState(mouse_passthrough=True))
        assert ("setIgnoresMouseEvents_", (True,)) in w2.calls


# ---------------------------------------------------------------------- #
# X11
# ---------------------------------------------------------------------- #
def make_x11_driver(window=100, view=200, display=object()):
    xlib = SimpleNamespace(
        XCreateRegion=MagicMock(return_value=999),
        XDestroyRegion=MagicMock(),
    )
    xshape = SimpleNamespace(
        ShapeInput=2,
        ShapeSet=0,
        XShapeCombineRegion=MagicMock(),
        XShapeCombineMask=MagicMock(),
    )
    wm_states = MagicMock()
    driver = X11NativeState(
        xlib, xshape, display=display, window=window, view=view,
        change_wm_state=wm_states,
    )
    return driver, xlib, xshape, wm_states


class TestX11Driver:
    def test_passthrough_on_uses_empty_input_region_on_window_and_view(self):
        d, xlib, xshape, _ = make_x11_driver()
        d.sync(WindowState(mouse_passthrough=True))

        xlib.XCreateRegion.assert_called_once_with()
        region = xlib.XCreateRegion.return_value
        # Both the top window and the covering child view are shaped.
        assert xshape.XShapeCombineRegion.call_count == 2
        for call, xid in zip(xshape.XShapeCombineRegion.call_args_list, (100, 200)):
            args = call.args
            assert args[1] == xid
            assert args[2] == xshape.ShapeInput
            assert args[5] == region
            assert args[6] == xshape.ShapeSet
        xlib.XDestroyRegion.assert_called_once_with(region)

    def test_passthrough_off_resets_mask_on_both_windows(self):
        d, _, xshape, _ = make_x11_driver()
        d.sync(WindowState(mouse_passthrough=True))
        d.sync(WindowState(mouse_passthrough=False))

        assert xshape.XShapeCombineMask.call_count == 2
        for call, xid in zip(xshape.XShapeCombineMask.call_args_list, (100, 200)):
            args = call.args
            assert args[1] == xid
            assert args[2] == xshape.ShapeInput
            assert args[3] == 0  # x offset
            assert args[4] == 0  # y offset
            assert args[5] == 0  # Pixmap None
            assert args[6] == xshape.ShapeSet

    def test_topmost_uses_ewmh_state_callback(self):
        d, _, _, wm_state = make_x11_driver()
        d.sync(WindowState(topmost=True))
        wm_state.assert_called_once_with("_NET_WM_STATE_ABOVE", True)
        d.sync(WindowState(topmost=False))
        wm_state.assert_called_with("_NET_WM_STATE_ABOVE", False)

    def test_idempotent_sync_does_not_reshape(self):
        d, _, xshape, wm_state = make_x11_driver()
        state = WindowState(mouse_passthrough=True)
        d.sync(state)
        region_calls = xshape.XShapeCombineRegion.call_count
        # topmost stayed at the all-off baseline -> no WM calls.
        assert wm_state.call_count == 0

        d.sync(state)
        assert xshape.XShapeCombineRegion.call_count == region_calls
        assert wm_state.call_count == 0

    def test_plain_off_state_produces_no_native_calls(self):
        d, xlib, xshape, wm_state = make_x11_driver()
        d.sync(WindowState())
        xlib.XCreateRegion.assert_not_called()
        xshape.XShapeCombineRegion.assert_not_called()
        xshape.XShapeCombineMask.assert_not_called()
        wm_state.assert_not_called()

    def test_missing_handles_makes_no_shape_calls(self):
        d, _, xshape, _ = make_x11_driver(window=0, view=0)
        d.sync(WindowState(mouse_passthrough=True))
        xshape.XShapeCombineRegion.assert_not_called()

    def test_attach_retargets_new_xids(self):
        d, _, xshape, _ = make_x11_driver()
        d.sync(WindowState(mouse_passthrough=True))
        d.attach(display=object(), window=300, view=400)
        d.sync(WindowState(mouse_passthrough=True))
        shaped_xids = [c.args[1] for c in xshape.XShapeCombineRegion.call_args_list]
        assert 300 in shaped_xids and 400 in shaped_xids
