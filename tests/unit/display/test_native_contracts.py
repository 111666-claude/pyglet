"""Contract tests isolating every native API.

Two levels are covered:

1. Pure backend mappers (``xrandr_screen_info`` / ``win32_screen_info`` /
   ``cocoa_screen_info``) are fed fake native primitives and exercised
   through the same four scenarios on every platform shape:
   primary/secondary, duplicate names, disconnect/reconnect and an empty
   display list.

2. The *real* Win32 and Cocoa backend modules are imported on Linux with
   their native libraries replaced by in-memory stubs, and their actual
   ``get_screens`` orchestration is driven through the four scenarios.
   The X11 geometry/native pairing helper is tested directly, with an
   optional live X server smoke test that skips when no display exists.
"""
from __future__ import annotations

import ctypes
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pyglet.display.identity import (
    ScreenInfo,
    ScreenRegistry,
    cocoa_screen_info,
    win32_screen_info,
    xrandr_screen_info,
)


# ---------------------------------------------------------------------------
# Shared fake native records + per-backend mapping
# ---------------------------------------------------------------------------

def x11_record(stable, name=None, *, primary_id=100, output_id=1, crtc=10,
                x=0, y=0, w=1920, h=1080, hz=60.0, connected=True, ordinal=0):
    h_total, v_total = 2200, 1125
    dotclock = int(hz * h_total * v_total)
    return dict(
        output_id=output_id, name=name or stable, connected=connected,
        crtc_id=crtc, x=x, y=y, width=w, height=h,
        primary_output_id=primary_id if stable == "PRIMARY" else primary_id,
        dot_clock_hz=dotclock, h_total=h_total, v_total=v_total,
        ordinal=ordinal,
    )


def win32_record(stable, name="Monitor", *, handle=100, primary=False,
                 x=0, y=0, w=1920, h=1080, frequency=60, ordinal=0):
    return dict(
        hmonitor=handle, device_name=stable, is_primary=primary,
        x=x, y=y, width=w, height=h, dm_display_frequency=frequency,
        friendly_name=name, ordinal=ordinal,
    )


def cocoa_record(stable, name="Monitor", *, display_id=100, main=False,
                 x=0, y=0, w=1920, h=1080, rate=60.0, ordinal=0):
    return dict(
        display_id=display_id, x=x, y=y, width=w, height=h,
        is_main=main, refresh_rate_hz=rate, localized_name=name,
        ordinal=ordinal,
    )


BACKENDS = {
    "x11": (xrandr_screen_info, x11_record),
    "win32": (win32_screen_info, win32_record),
    "cocoa": (cocoa_screen_info, cocoa_record),
}


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["x11", "win32", "cocoa"])
class TestPerBackendContract:
    def _info(self, backend, stable, **kwargs):
        mapper, record = BACKENDS[backend]
        return mapper(**record(stable, **kwargs))

    def test_primary_and_secondary(self, backend):
        primary_kw = dict(primary=True) if backend != "x11" else {}
        if backend == "x11":
            a = self._info(backend, "DP-1", output_id=101,
                           primary_id=101, x=0, y=0)
            b = self._info(backend, "HDMI-2", output_id=102,
                            primary_id=101, x=1920, y=0)
        elif backend == "win32":
            a = self._info(backend, r"\\.\DISPLAY1", handle=11, primary=True)
            b = self._info(backend, r"\\.\DISPLAY2", handle=22,
                           primary=False, x=1920)
        else:
            a = self._info(backend, "10", display_id=10, main=True)
            b = self._info(backend, "20", display_id=20, main=False, x=1920)

        registry = ScreenRegistry(backend)
        ids = registry.synchronize([a, b])
        assert ids[0].is_primary is True
        assert ids[1].is_primary is False
        assert registry.primary.key == ids[0].key
        assert ids[0].key != ids[1].key

    def test_duplicate_names_disambiguated(self, backend):
        if backend == "x11":
            # RandR connector names are normally unique, but two outputs can
            # still report the same monitor/EDID name; feed identical names
            # with distinct output ids.
            a = self._info(backend, "OUT-A", name="DELL Twin", output_id=1)
            b = self._info(backend, "OUT-B", name="DELL Twin", output_id=2,
                            x=1920)
        elif backend == "win32":
            a = self._info(backend, r"\\.\DISPLAY1", name="DELL Twin", handle=1)
            b = self._info(backend, r"\\.\DISPLAY2", name="DELL Twin",
                           handle=2, x=1920)
        else:
            a = self._info(backend, "10", name="DELL Twin", display_id=10)
            b = self._info(backend, "20", name="DELL Twin", display_id=20,
                           x=1920)

        ids = ScreenRegistry(backend).synchronize([a, b])
        assert ids[0].key != ids[1].key
        assert sorted(i.label for i in ids) == ["DELL Twin", "DELL Twin #2"]

    def test_disconnect_reconnect_keeps_key(self, backend):
        if backend == "x11":
            a = self._info(backend, "DP-1", output_id=101, primary_id=101)
            b = self._info(backend, "DP-2", output_id=102, primary_id=101,
                           x=1920)
            b_reconnect = self._info(backend, "DP-2", output_id=102,
                                     primary_id=101, x=1920)
        elif backend == "win32":
            a = self._info(backend, r"\\.\DISPLAY1", handle=11, primary=True)
            b = self._info(backend, r"\\.\DISPLAY2", handle=22, x=1920)
            # HMONITOR handle changes after replug.
            b_reconnect = self._info(backend, r"\\.\DISPLAY2", handle=999,
                                     x=1920)
        else:
            a = self._info(backend, "10", display_id=10, main=True)
            b = self._info(backend, "20", display_id=20, x=1920)
            b_reconnect = self._info(backend, "20", display_id=20, x=1920)

        registry = ScreenRegistry(backend)
        key_b = registry.synchronize([a, b])[1].key
        registry.synchronize([a])
        assert registry.resolve(key_b) is None
        restored = registry.synchronize([a, b_reconnect])
        assert restored[1].key == key_b

    def test_empty_display_list(self, backend):
        a = self._info(backend, "OUT-1")
        registry = ScreenRegistry(backend)
        key = registry.synchronize([a])[0].key

        assert registry.synchronize([]) == []
        assert registry.primary is None
        assert registry.resolve(key) is None

        restored = registry.synchronize([a])
        assert restored[0].key == key

    def test_invalid_native_rate_is_none(self, backend):
        if backend == "win32":
            info = self._info(backend, r"\\.\DISPLAY1", frequency=0)
            assert info.refresh_rate is None
            info = self._info(backend, r"\\.\DISPLAY1", frequency=1)
            assert info.refresh_rate is None
        elif backend == "cocoa":
            info = self._info(backend, "10", rate=0)
            assert info.refresh_rate is None
        else:
            info = xrandr_screen_info(
                output_id=1, name="DP-1", connected=True, crtc_id=2,
                x=0, y=0, width=1920, height=1080, primary_output_id=1,
                dot_clock_hz=0, h_total=0, v_total=0,
            )
            assert info.refresh_rate is None

    def test_backend_specific_filtering(self, backend):
        if backend == "x11":
            disconnected = xrandr_screen_info(
                output_id=9, name="DP-9", connected=False, crtc_id=0,
                x=0, y=0, width=0, height=0, primary_output_id=1,
                dot_clock_hz=None, h_total=None, v_total=None,
            )
            assert disconnected is None
            inactive = xrandr_screen_info(
                output_id=9, name="DP-9", connected=True, crtc_id=0,
                x=0, y=0, width=0, height=0, primary_output_id=1,
                dot_clock_hz=None, h_total=None, v_total=None,
            )
            assert inactive is None


# ---------------------------------------------------------------------------
# Base Display/Screen behaviour (window screen-selection surface)
# ---------------------------------------------------------------------------

class _ScriptedDisplay:
    """Minimal Display stand-in exposing the default-screen contract."""


@pytest.mark.unit
class TestBaseDisplayScreenContract:
    def _display(self, screen_values):
        from pyglet.display.base import Display

        class FakeDisplay(Display):
            def get_screens(self):
                return list(screen_values)

        # Bypass Display.__init__ so scripted displays do not register in the
        # global pyglet.display._displays WeakSet and leak into get_display()
        # (which pyglet.window uses to create its shadow window).
        display = FakeDisplay.__new__(FakeDisplay)
        display.screen_registry = ScreenRegistry("test")
        return display

    def test_screen_constructs_without_identity_backwards_compatible(self):
        from pyglet.display.base import Screen
        display = SimpleNamespace(_backend="test")
        screen = Screen(display, 0, 0, 1280, 720)
        assert screen.width == 1280 and screen.height == 720
        assert screen.x == 0 and screen.y == 0
        assert isinstance(screen.stable_id, str) and screen.stable_id
        assert screen.get_display_id() == screen.display_id
        assert screen.get_monitor_name() == "Unknown"
        assert screen.refresh_rate is None
        assert screen.is_primary is True  # (0, 0) fallback

    def test_get_default_screen_prefers_primary_flag(self):
        from pyglet.display.base import Screen
        display = SimpleNamespace(_backend="test")
        origin = Screen(display, 0, 0, 100, 100)
        other_primary = Screen(
            display, 500, 500, 100, 100,
            identity=_identity(primary=True),
        )
        origin.identity = _identity(primary=False)
        d = self._display([origin, other_primary])
        assert d.get_default_screen() is other_primary

    def test_get_default_screen_origin_heuristic(self):
        from pyglet.display.base import Screen
        display = SimpleNamespace(_backend="test")
        left = Screen(display, -1920, 0, 100, 100, identity=_identity(False))
        origin = Screen(display, 0, 0, 100, 100, identity=_identity(False))
        d = self._display([left, origin])
        assert d.get_default_screen() is origin

    def test_get_default_screen_empty_returns_none(self):
        d = self._display([])
        assert d.get_default_screen() is None

    def test_get_screen_by_stable_id_roundtrip(self):
        from pyglet.display.base import Screen
        d = self._display([])
        info = ScreenInfo(backend="test", stable_native_id="DP-7",
                          name="Neo", is_primary=True)
        identity = d.screen_registry.synchronize([info])[0]
        screen = Screen(d, 0, 0, 100, 100, identity=identity)

        # Wire the scripted enumeration to return this screen.
        d.get_screens = lambda: [screen]
        assert d.get_screen_by_stable_id(identity.key) is screen
        assert d.get_screen_by_stable_id("test:out:DP-7") is screen
        assert d.get_screen_by_stable_id("missing") is None

    def test_window_creation_on_empty_display_raises_predictably(self, monkeypatch):
        import pyglet

        # Avoid GL/shadow-window creation so the guard is testable headless.
        monkeypatch.setitem(pyglet.options, "shadow_window", False)
        import pyglet.window
        from pyglet.window import NoSuchDisplayException

        class EmptyDisplay:
            def get_default_screen(self):
                return None

            def get_screens(self):
                return []

        with pytest.raises(NoSuchDisplayException):
            pyglet.window.Window(display=EmptyDisplay())


def _identity(primary):
    from pyglet.display.identity import ScreenIdentity
    return ScreenIdentity(key=f"k-{primary}", ordinal=1, label="L",
                          is_primary=primary)


# ---------------------------------------------------------------------------
# X11 pairing helper (pure, real module import)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestXlibPairing:
    xlib = pytest.importorskip("pyglet.display.xlib")

    def _screen(self, x, y, w, h):
        return SimpleNamespace(x=x, y=y, width=w, height=h)

    def test_exact_geometry_pairing(self):
        infos = [
            ScreenInfo(stable_native_id="A", x=0, y=0, width=1920, height=1080),
            ScreenInfo(stable_native_id="B", x=1920, y=0, width=1280, height=1024),
        ]
        screens = [self._screen(1920, 0, 1280, 1024),
                   self._screen(0, 0, 1920, 1080)]
        paired = self.xlib.pair_infos_to_screens(screens, infos)
        assert [p.stable_native_id for p in paired] == ["B", "A"]

    def test_geometry_fallback_for_unmatched(self):
        screens = [self._screen(0, 0, 800, 600)]
        paired = self.xlib.pair_infos_to_screens(screens, [])
        assert len(paired) == 1
        assert (paired[0].width, paired[0].height) == (800, 600)
        assert paired[0].stable_native_id is None

    def test_same_resolution_second_pass(self):
        infos = [ScreenInfo(stable_native_id="A", x=999, y=999,
                            width=1920, height=1080)]
        screens = [self._screen(0, 0, 1920, 1080)]
        paired = self.xlib.pair_infos_to_screens(screens, infos)
        assert paired[0].stable_native_id == "A"

    def test_live_xserver_smoke(self):
        """Real-native smoke; skips when no X server is reachable."""
        try:
            display = self.xlib.XlibDisplay()
        except Exception:
            pytest.skip("No reachable X11 display")
        screens = display.get_screens()
        assert len(screens) >= 1
        assert all(isinstance(s.stable_id, str) for s in screens)
        default = display.get_default_screen()
        assert default is not None
        assert default.stable_id == screens[0].stable_id or \
            any(s.is_primary for s in screens)


# ---------------------------------------------------------------------------
# Real Win32 backend with stubbed native libraries
# ---------------------------------------------------------------------------

@pytest.fixture
def stubbed_win32(monkeypatch):
    user32 = MagicMock(name="_user32")
    win32_pkg = types.ModuleType("pyglet.libs.win32")
    win32_pkg._user32 = user32
    win32_pkg._shcore = MagicMock()
    win32_pkg._gdi32 = MagicMock()
    monkeypatch.setitem(sys.modules, "pyglet.libs.win32", win32_pkg)

    constants = types.ModuleType("pyglet.libs.win32.constants")
    for name in [
        "CDS_FULLSCREEN", "DISP_CHANGE_SUCCESSFUL", "ENUM_CURRENT_SETTINGS",
        "WINDOWS_8_1_OR_GREATER", "WINDOWS_VISTA_OR_GREATER",
        "WINDOWS_10_CREATORS_UPDATE_OR_GREATER", "USER_DEFAULT_SCREEN_DPI",
        "LOGPIXELSX", "LOGPIXELSY",
    ]:
        setattr(constants, name, 0)
    constants.MONITORINFOF_PRIMARY = 1
    monkeypatch.setitem(sys.modules, "pyglet.libs.win32.constants", constants)

    class DEVMODE(ctypes.Structure):
        _fields_ = [
            ("dmSize", ctypes.c_uint32),
            ("dmDisplayFrequency", ctypes.c_uint32),
            ("dmPelsWidth", ctypes.c_uint32),
            ("dmPelsHeight", ctypes.c_uint32),
            ("dmBitsPerPel", ctypes.c_uint32),
            ("dmDisplayFixedOutput", ctypes.c_uint32),
        ]

    class MONITORINFOEX(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_uint32),
            ("dwFlags", ctypes.c_uint32),
            ("szDevice", ctypes.c_wchar * 32),
        ]

    types_mod = types.ModuleType("pyglet.libs.win32.types")
    types_mod.DEVMODE = DEVMODE
    types_mod.MONITORINFOEX = MONITORINFOEX
    types_mod.MONITORENUMPROC = lambda f: f
    types_mod.UINT = ctypes.c_uint32
    types_mod.DWORD = ctypes.c_uint32
    types_mod.sizeof = ctypes.sizeof
    types_mod.byref = lambda obj: obj
    types_mod.DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = 1
    types_mod.PROCESS_PER_MONITOR_DPI_AWARE = 2
    monkeypatch.setitem(sys.modules, "pyglet.libs.win32.types", types_mod)

    ctx = types.ModuleType("pyglet.libs.win32.context_managers")
    ctx.device_context = MagicMock()
    monkeypatch.setitem(sys.modules,
                        "pyglet.libs.win32.context_managers", ctx)

    sys.modules.pop("pyglet.display.win32", None)
    module = __import__("pyglet.display.win32", fromlist=["Win32Display"])

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    state = {"records": []}

    def enum_monitors(_hdc, _hdc2, proc, _data):
        for rec in state["records"]:
            rect = RECT(rec["x"], rec["y"],
                        rec["x"] + rec["w"], rec["y"] + rec["h"])
            proc(rec["handle"], None, ctypes.pointer(rect), 0)
        return 1

    user32.EnumDisplayMonitors.side_effect = enum_monitors

    def get_monitor_info(handle, info):
        rec = next(r for r in state["records"] if r["handle"] == handle)
        info.dwFlags = 1 if rec["primary"] else 0
        info.szDevice = rec["device"]
        return 1

    user32.GetMonitorInfoW.side_effect = get_monitor_info

    def enum_settings(device, _mode_num, mode):
        rec = next(r for r in state["records"] if r["device"] == device)
        mode.dmDisplayFrequency = rec["frequency"]
        return 1

    user32.EnumDisplaySettingsW.side_effect = enum_settings

    def enum_devices(device, _index, dev, _flags):
        rec = next(r for r in state["records"] if r["device"] == device)
        dev.DeviceString = rec["friendly"]
        return 1

    user32.EnumDisplayDevicesW.side_effect = enum_devices

    try:
        yield SimpleNamespace(module=module, state=state)
    finally:
        sys.modules.pop("pyglet.display.win32", None)


def _win_rec(handle, device, *, primary=False, x=0, w=1920, freq=60,
             friendly="Dell Twin"):
    return dict(handle=handle, device=device, primary=primary, x=x, y=0,
                w=w, h=1080, frequency=freq, friendly=friendly)


@pytest.mark.unit
class TestRealWin32BackendContract:
    def test_primary_duplicate_names_and_rate(self, stubbed_win32):
        env = stubbed_win32
        env.state["records"] = [
            _win_rec(111, r"\\.\DISPLAY1", primary=True, freq=60),
            _win_rec(222, r"\\.\DISPLAY2", x=1920, w=2560, freq=144),
        ]
        display = env.module.Win32Display()
        screens = display.get_screens()

        assert len(screens) == 2
        assert screens[0].is_primary is True
        assert screens[1].is_primary is False
        assert screens[0].refresh_rate == 60.0
        assert screens[1].refresh_rate == 144.0
        assert screens[1].label == "Dell Twin #2"
        # Win32 re-enumerates on every get_screens() (historical behavior),
        # so compare identity rather than object identity across calls.
        assert display.get_default_screen().stable_id == screens[0].stable_id

    def test_reconnect_changed_hmonitor(self, stubbed_win32):
        env = stubbed_win32
        env.state["records"] = [
            _win_rec(111, r"\\.\DISPLAY1", primary=True),
            _win_rec(222, r"\\.\DISPLAY2", x=1920),
        ]
        display = env.module.Win32Display()
        key2 = display.get_screens()[1].stable_id

        # Unplug second monitor.
        env.state["records"] = [_win_rec(111, r"\\.\DISPLAY1", primary=True)]
        assert display.get_screens()[0].is_primary
        assert display.get_screen_by_stable_id(key2) is None

        # Reconnect: new HMONITOR value, same device name.
        env.state["records"] = [
            _win_rec(111, r"\\.\DISPLAY1", primary=True),
            _win_rec(999, r"\\.\DISPLAY2", x=1920),
        ]
        screens = display.get_screens()
        assert screens[1].stable_id == key2
        assert screens[1]._handle == 999
        looked_up = display.get_screen_by_stable_id(key2)
        assert looked_up is not None
        assert looked_up.stable_id == key2
        assert looked_up._handle == 999

    def test_empty_display_list(self, stubbed_win32):
        env = stubbed_win32
        display = env.module.Win32Display()
        env.state["records"] = [_win_rec(111, r"\\.\DISPLAY1", primary=True)]
        key = display.get_screens()[0].stable_id

        env.state["records"] = []
        assert display.get_screens() == []
        assert display.get_default_screen() is None
        assert display.get_screen_by_stable_id(key) is None


# ---------------------------------------------------------------------------
# Real Cocoa backend with stubbed native libraries
# ---------------------------------------------------------------------------

@pytest.fixture
def stubbed_cocoa(monkeypatch):
    darwin = types.ModuleType("pyglet.libs.darwin")
    darwin.NSDeviceResolution = "NSDeviceResolution"
    monkeypatch.setitem(sys.modules, "pyglet.libs.darwin", darwin)

    cocoapy = types.ModuleType("pyglet.libs.darwin.cocoapy")
    quartz = MagicMock(name="quartz")
    cocoapy.quartz = quartz
    cocoapy.cf = MagicMock(name="cf")
    cocoapy.CGDirectDisplayID = ctypes.c_uint32
    cocoapy.ObjCClass = lambda _name: MagicMock(name=_name)
    cocoapy.get_NSString = lambda value: value
    cocoapy.cfstring_to_string = lambda _ptr: "PPPPPPPP"
    cocoapy.cfarray_to_list = lambda _ref: []
    monkeypatch.setitem(sys.modules,
                        "pyglet.libs.darwin.cocoapy", cocoapy)

    sys.modules.pop("pyglet.display.cocoa", None)
    module = __import__("pyglet.display.cocoa", fromlist=["CocoaDisplay"])
    monkeypatch.setattr(module, "byref", lambda obj: obj)

    state = {"records": []}

    def active_list(_max_count, array, count):
        for index, rec in enumerate(state["records"]):
            array[index] = rec["id"]
        count.value = len(state["records"])
        return 0

    quartz.CGGetActiveDisplayList.side_effect = active_list
    quartz.CGMainDisplayID.side_effect = lambda: next(
        (r["id"] for r in state["records"] if r["main"]), 0,
    )
    quartz.CGDisplayBounds.side_effect = lambda did: _bounds(did, state)
    quartz.CGDisplayCopyDisplayMode.return_value = 12345

    rate_calls = {"index": 0}

    def rate_for(_mode):
        # CocoaScreenMode reads the rate once per screen, in enumeration order.
        index = rate_calls["index"]
        rate_calls["index"] += 1
        records = state["records"]
        return records[min(index, len(records) - 1)]["rate"] if records else None

    quartz.CGDisplayModeGetRefreshRate.side_effect = rate_for
    quartz.CGDisplayModeGetWidth.return_value = 1920
    quartz.CGDisplayModeGetHeight.return_value = 1080
    quartz.CGDisplayModeCopyPixelEncoding.return_value = 1
    # No matching NSScreen -> friendly name stays None.
    module.NSScreen.screens.return_value.count.return_value = 0

    try:
        yield SimpleNamespace(module=module, state=state, quartz=quartz)
    finally:
        sys.modules.pop("pyglet.display.cocoa", None)


def _bounds(display_id, state):
    rec = next(r for r in state["records"] if r["id"] == display_id)
    return SimpleNamespace(
        origin=SimpleNamespace(x=rec["x"], y=rec["y"]),
        size=SimpleNamespace(width=rec["w"], height=rec["h"]),
    )


def _coc_rec(display_id, *, main=False, x=0, rate=60.0):
    return dict(id=display_id, main=main, x=x, y=0, w=1920, h=1080, rate=rate)


@pytest.mark.unit
class TestRealCocoaBackendContract:
    def test_primary_rates_and_unknown_name(self, stubbed_cocoa):
        env = stubbed_cocoa
        env.state["records"] = [
            _coc_rec(10, main=True, rate=60.0),
            _coc_rec(20, x=1920, rate=0.0),
        ]
        display = env.module.CocoaDisplay()
        screens = display.get_screens()

        assert screens[0].is_primary is True
        assert screens[0].refresh_rate == 60.0
        assert screens[1].is_primary is False
        assert screens[1].refresh_rate is None  # CoreGraphics 0 -> unknown
        assert display.get_default_screen().stable_id == screens[0].stable_id

    def test_reconnect_same_display_id(self, stubbed_cocoa):
        env = stubbed_cocoa
        env.state["records"] = [_coc_rec(10, main=True), _coc_rec(20, x=1920)]
        display = env.module.CocoaDisplay()
        key2 = display.get_screens()[1].stable_id

        env.state["records"] = [_coc_rec(10, main=True)]
        assert display.get_screen_by_stable_id(key2) is None

        env.state["records"] = [_coc_rec(10, main=True), _coc_rec(20, x=1920)]
        screens = display.get_screens()
        assert screens[1].stable_id == key2

    def test_empty_display_list(self, stubbed_cocoa):
        env = stubbed_cocoa
        display = env.module.CocoaDisplay()
        env.state["records"] = [_coc_rec(10, main=True)]
        key = display.get_screens()[0].stable_id

        env.state["records"] = []
        assert display.get_screens() == []
        assert display.get_default_screen() is None
        assert display.get_screen_by_stable_id(key) is None
