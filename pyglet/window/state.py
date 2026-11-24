"""Independent window state (transparency / mouse pass-through / topmost).

This module deliberately has **no** platform or pyglet imports at module load
time beyond the lightweight window exception types.  All native calls are
hidden behind small driver objects whose dependencies are passed in, which
makes the state transitions testable with plain fakes on any operating
system.

The :class:`WindowStateController` is the single source of truth for the
desired state.  Platform windows create one controller during construction,
attach a native driver in ``_create`` and drive the lifecycle as follows::

    controller = WindowStateController(driver, initial=..., capabilities=...)
    # ... native handle created ...
    controller.realize()      # push the full desired state to a fresh handle
    # ... fullscreen toggle / _recreate destroys the handle ...
    controller.invalidate()   # drop "applied" bookkeeping, keep desired state
    # ... new handle created ...
    controller.realize()      # state is fully restored
    # ... close ...
    controller.teardown()      # best-effort native reset before destroy

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from pyglet.window.exceptions import WindowStateException

#: Aspect name for per-pixel transparent rendering.
TRANSPARENT = "transparent"
#: Aspect name for mouse click-through (input passes to windows below).
MOUSE_PASSTHROUGH = "mouse_passthrough"
#: Aspect name for always-on-top / above-other-windows behaviour.
TOPMOST = "topmost"

#: All managed aspects, in the order they are first applied.
ASPECTS: tuple[str, ...] = (TRANSPARENT, MOUSE_PASSTHROUGH, TOPMOST)

#: Human readable descriptions used in unsupported-state errors.
_ASPECT_DESCRIPTIONS = {
    TRANSPARENT: "per-pixel window transparency",
    MOUSE_PASSTHROUGH: "mouse pass-through (click-through)",
    TOPMOST: "topmost window level",
}


@dataclass(frozen=True)
class WindowState:
    """Immutable snapshot of the independently switchable window state."""

    transparent: bool = False
    mouse_passthrough: bool = False
    topmost: bool = False

    def changed(self, **changes: bool) -> WindowState:
        """Return a copy with the given aspects replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class StateCapabilities:
    """Which optional state features the native window/framebuffer supports.

    Attributes:
        alpha_framebuffer:
            ``True`` when the GL framebuffer provides alpha bits, which are
            required for per-pixel transparency.
        mouse_passthrough:
            ``True`` when the platform session supports click-through
            (e.g. the X11 SHAPE extension is available).
    """

    alpha_framebuffer: bool = True
    mouse_passthrough: bool = True


@runtime_checkable
class NativeStateDriver(Protocol):
    """Protocol implemented by platform native-state drivers.

    ``sync`` must make the native window match ``state``.  Implementations
    are expected to be idempotent and to diff against the previously synced
    state so redundant native calls are avoided.
    """

    def sync(self, state: WindowState) -> None:
        ...


class RecordingNativeState:
    """Injectable fake :class:`NativeStateDriver` for unit tests.

    It records aspect-level transitions as ``(aspect, value)`` tuples in
    :attr:`calls` and keeps the last synced state in :attr:`synced`.

    Example::

        native = RecordingNativeState()
        controller = WindowStateController(native, realized=True)
        controller.set_mouse_passthrough(True)
        assert native.calls == [("mouse_passthrough", True)]
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.sync_calls: list[WindowState] = []
        self.synced: WindowState | None = None

    def sync(self, state: WindowState) -> None:
        self.sync_calls.append(state)
        previous = self.synced or WindowState()
        for aspect in ASPECTS:
            value = getattr(state, aspect)
            if value != getattr(previous, aspect):
                self.calls.append((aspect, value))
        self.synced = state

    def values(self, aspect: str) -> list[bool]:
        """Return just the values recorded for ``aspect``."""
        return [value for name, value in self.calls if name == aspect]

    def count(self, aspect: str) -> int:
        """Return the number of recorded transitions for ``aspect``."""
        return sum(1 for name, _ in self.calls if name == aspect)

    def reset_log(self) -> None:
        """Clear the call log without affecting the synced state."""
        self.calls.clear()
        self.sync_calls.clear()

    def reset(self) -> None:
        """Clear the diff baseline (called when a native handle is recreated)."""
        self.synced = None


class WindowStateController:
    """State authority + lifecycle guard for the switchable window aspects.

    Args:
        driver:
            Injectable native driver.  May be ``None`` until the platform
            window creates its native handle; assign one later via
            :attr:`driver`.
        initial:
            Desired initial state.  Defaults to every aspect disabled, which
            preserves the behaviour of an ordinary window.
        capabilities:
            Platform/framebuffer capabilities used to reject unsupported
            combinations with a clear :class:`WindowStateException`.
    """

    def __init__(
        self,
        driver: NativeStateDriver | None = None,
        *,
        initial: WindowState | None = None,
        capabilities: StateCapabilities | None = None,
    ) -> None:
        self._driver = driver
        self._desired = initial or WindowState()
        self._capabilities = capabilities or StateCapabilities()
        self._realized = False
        self._closed = False

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def driver(self) -> NativeStateDriver | None:
        """The injected native driver, if any."""
        return self._driver

    @driver.setter
    def driver(self, value: NativeStateDriver | None) -> None:
        self._driver = value

    @property
    def capabilities(self) -> StateCapabilities:
        return self._capabilities

    @capabilities.setter
    def capabilities(self, value: StateCapabilities) -> None:
        self._capabilities = value

    @property
    def state(self) -> WindowState:
        """The current *desired* state (independent of native realisation)."""
        return self._desired

    @property
    def realized(self) -> bool:
        """True after :meth:`realize` and before :meth:`invalidate`/teardown."""
        return self._realized

    @property
    def closed(self) -> bool:
        return self._closed

    def get(self, aspect: str) -> bool:
        return bool(getattr(self._desired, aspect))

    # ------------------------------------------------------------------ #
    # State changes
    # ------------------------------------------------------------------ #
    def set(self, aspect: str, enabled: bool) -> bool:
        """Switch one aspect on/off.

        Returns ``True`` when the desired state changed, ``False`` for a
        redundant call (in which case no native call is made).

        Raises:
            ValueError: for an unknown aspect name.
            WindowStateException: when enabling an unsupported combination.
        """
        if aspect not in ASPECTS:
            msg = f"Unknown window state aspect {aspect!r}; expected one of {ASPECTS}."
            raise ValueError(msg)
        if self._closed:
            msg = "Cannot change window state after the window has been closed."
            raise WindowStateException(msg)

        enabled = bool(enabled)
        self._check_supported(aspect, enabled)

        if getattr(self._desired, aspect) == enabled:
            return False

        self._desired = replace(self._desired, **{aspect: enabled})
        self._sync()
        return True

    def update(self, **aspects: bool) -> bool:
        """Set several aspects at once.

        All requested combinations are validated before any native call is
        made, so an unsupported batch fails atomically.
        """
        unknown = [name for name in aspects if name not in ASPECTS]
        if unknown:
            msg = f"Unknown window state aspect(s) {unknown!r}; expected ones of {ASPECTS}."
            raise ValueError(msg)
        if self._closed:
            msg = "Cannot change window state after the window has been closed."
            raise WindowStateException(msg)

        values = {name: bool(value) for name, value in aspects.items()}
        for name, value in values.items():
            self._check_supported(name, value)

        changed = any(getattr(self._desired, name) != value for name, value in values.items())
        if not changed:
            return False

        self._desired = replace(self._desired, **values)
        self._sync()
        return True

    def _check_supported(self, aspect: str, enabled: bool) -> None:
        if not enabled:
            return
        if aspect == TRANSPARENT and not self._capabilities.alpha_framebuffer:
            msg = (
                "Per-pixel transparency cannot be enabled at runtime because this "
                "window's framebuffer has no alpha channel. Create the window with "
                "transparent=True (or style='transparent'/'overlay'), or supply a "
                "GL config with alpha_size >= 8."
            )
            raise WindowStateException(msg)
        if aspect == MOUSE_PASSTHROUGH and not self._capabilities.mouse_passthrough:
            msg = (
                "Mouse pass-through is not supported by this native window/session "
                "(the X11 SHAPE extension was not found)."
            )
            raise WindowStateException(msg)

    # ------------------------------------------------------------------ #
    # Native lifecycle
    # ------------------------------------------------------------------ #
    def _sync(self) -> None:
        if self._realized and self._driver is not None:
            self._driver.sync(self._desired)

    def realize(self) -> None:
        """Push the complete desired state to a freshly created native window.

        Must be called at the end of every platform ``_create`` call,
        including recreation after a fullscreen toggle.
        """
        if self._closed:
            msg = "Cannot realize window state after the controller was closed."
            raise WindowStateException(msg)
        # Validate the initial desired state as well (it bypasses set()).
        for aspect in ASPECTS:
            self._check_supported(aspect, getattr(self._desired, aspect))
        self._realized = True
        if self._driver is not None:
            self._driver.sync(self._desired)

    def invalidate(self) -> None:
        """Forget native applied-state without touching the desired state.

        Call just before destroying a native handle that will be recreated
        (fullscreen toggle / ``_recreate``).  Drivers exposing a ``reset()``
        method have their diff cache cleared, so the next :meth:`realize`
        re-applies every aspect to the new handle.
        """
        self._realized = False
        reset = getattr(self._driver, "reset", None)
        if callable(reset):
            reset()

    def teardown(self) -> None:
        """Best-effort reset of native state before/while destroying a window.

        A single all-off state is synced to the driver so injected tests can
        assert that closing the window clears click-through/topmost/
        transparency state.  Never raises: close paths must complete.
        """
        if self._realized and self._driver is not None:
            try:
                self._driver.sync(WindowState())
            except Exception:  # noqa: BLE001 - teardown must never mask close
                pass
        self._realized = False
        self._closed = True


# ---------------------------------------------------------------------- #
# Platform native drivers
#
# These classes only perform native calls.  Every platform specific object
# (user32/dwmapi/..., the NSWindow, xlib/...) is passed in, so the exact
# call sequences can be unit tested on any operating system with fakes.
# ---------------------------------------------------------------------- #
class Win32NativeState:
    """Applies :class:`WindowState` through injected Win32 libraries.

    Args:
        user32: object providing ``SetWindowLongW``, ``SetWindowPos`` and
            ``SetLayeredWindowAttributes`` (typically
            ``pyglet.libs.win32._user32``).
        dwmapi: DWM library providing ``DwmEnableBlurBehindWindow``.
        gdi32: GDI library providing region create/delete functions.
        const: object exposing the required Win32 constants.
        blurbehind_type: ctypes ``DWM_BLURBEHIND`` structure class.
        byref: the ``ctypes.byref`` callable (identity in tests).
    """

    def __init__(
        self,
        user32: Any,
        dwmapi: Any | None = None,
        gdi32: Any | None = None,
        const: Any | None = None,
        *,
        blurbehind_type: Any | None = None,
        byref: Any | None = None,
        hwnd: int = 0,
        view_hwnd: int = 0,
        base_ex_style: int = 0,
        on_ex_style: Any | None = None,
    ) -> None:
        self.user32 = user32
        self.dwmapi = dwmapi
        self.gdi32 = gdi32
        self.const = const
        self.blurbehind_type = blurbehind_type
        self.byref = byref or (lambda obj: obj)
        self.hwnd = hwnd
        self.view_hwnd = view_hwnd
        # Style-derived extended bits (caption/border/...) without the
        # transparency/pass-through bits managed here.
        self.base_ex_style = base_ex_style
        self.ex_style = base_ex_style
        # Optional callback receiving the final extended style, used to keep
        # the owning window's cached ``_ex_ws_style`` in sync.
        self.on_ex_style = on_ex_style
        self._synced: WindowState | None = None

    def attach(self, hwnd: int, view_hwnd: int = 0, base_ex_style: int | None = None) -> None:
        """Bind to a (possibly new) native handle and clear diff history."""
        self.hwnd = hwnd
        self.view_hwnd = view_hwnd
        if base_ex_style is not None:
            self.base_ex_style = base_ex_style
            self.ex_style = base_ex_style
        self.reset()

    def reset(self) -> None:
        """Force every native call on the next :meth:`sync`."""
        self._synced = None

    def _extended_style(self, state: WindowState) -> int:
        ex_style = self.base_ex_style
        if state.transparent or state.mouse_passthrough:
            ex_style |= self.const.WS_EX_LAYERED
        if state.mouse_passthrough:
            ex_style |= self.const.WS_EX_TRANSPARENT
        return ex_style

    def sync(self, state: WindowState) -> None:
        c = self.const
        # A freshly created native window already matches the all-off
        # baseline, so an ordinary window causes zero native calls.
        previous = self._synced or WindowState()

        ex_style = self._extended_style(state)
        if ex_style != self.ex_style:
            self.user32.SetWindowLongW(self.hwnd, c.GWL_EXSTYLE, ex_style)
            self.ex_style = ex_style
            if self.on_ex_style is not None:
                self.on_ex_style(ex_style)

        # The child view window covers the client area; give it the
        # hit-test-through bit as well so clicks cannot land on it.
        if previous.mouse_passthrough != state.mouse_passthrough and self.view_hwnd:
            view_ex_style = c.WS_EX_TRANSPARENT if state.mouse_passthrough else 0
            self.user32.SetWindowLongW(self.view_hwnd, c.GWL_EXSTYLE, view_ex_style)

        if previous.transparent != state.transparent:
            self._apply_blur_behind(state.transparent)
            if state.transparent:
                # Alpha 255 + LWA_ALPHA keeps the whole window at full
                # opacity while DWM honours per-pixel framebuffer alpha.
                self.user32.SetLayeredWindowAttributes(
                    self.hwnd, 0, 255, c.LWA_ALPHA,
                )

        if previous.topmost != state.topmost:
            insert_after = c.HWND_TOPMOST if state.topmost else c.HWND_NOTOPMOST
            flags = c.SWP_NOMOVE | c.SWP_NOSIZE | c.SWP_NOACTIVATE
            self.user32.SetWindowPos(self.hwnd, insert_after, 0, 0, 0, 0, flags)

        self._synced = state

    def _apply_blur_behind(self, enabled: bool) -> None:
        if self.dwmapi is None or self.blurbehind_type is None:
            return

        bb = self.blurbehind_type()
        if enabled:
            # An empty region disables the actual blur while the
            # DWM_BB_ENABLE composition path still honours GL alpha.
            region = self.gdi32.CreateRectRgn(0, 0, -1, -1)
            bb.dwFlags = self.const.DWM_BB_ENABLE | self.const.DWM_BB_BLURREGION
            bb.hRgnBlur = region
            bb.fEnable = True
            self.dwmapi.DwmEnableBlurBehindWindow(self.hwnd, self.byref(bb))
            self.gdi32.DeleteObject(region)
        else:
            bb.dwFlags = self.const.DWM_BB_ENABLE
            bb.fEnable = False
            self.dwmapi.DwmEnableBlurBehindWindow(self.hwnd, self.byref(bb))


class CocoaNativeState:
    """Applies :class:`WindowState` to an injected ``NSWindow``-like object."""

    def __init__(
        self,
        nswindow: Any | None = None,
        *,
        normal_level: int = 0,
        status_level: int = 25,
        clear_color: Any | None = None,
        opaque_color: Any | None = None,
    ) -> None:
        self.nswindow = nswindow
        self.normal_level = normal_level
        self.status_level = status_level
        # Zero-arg callables returning the real NSColor objects.
        self.clear_color = clear_color
        self.opaque_color = opaque_color
        self._synced: WindowState | None = None

    def attach(self, nswindow: Any) -> None:
        self.nswindow = nswindow
        self.reset()

    def reset(self) -> None:
        self._synced = None

    def sync(self, state: WindowState) -> None:
        window = self.nswindow
        if window is None:
            return
        # Baseline all-off matches a freshly created NSWindow, so an ordinary
        # window produces no state-related native calls on first sync.
        previous = self._synced or WindowState()

        if previous.transparent != state.transparent:
            window.setOpaque_(not state.transparent)
            if state.transparent:
                if self.clear_color is not None:
                    window.setBackgroundColor_(self.clear_color())
                window.setHasShadow_(False)
            else:
                if self.opaque_color is not None:
                    window.setBackgroundColor_(self.opaque_color())
                window.setHasShadow_(True)

        if previous.mouse_passthrough != state.mouse_passthrough:
            window.setIgnoresMouseEvents_(bool(state.mouse_passthrough))

        if previous.topmost != state.topmost:
            level = self.status_level if state.topmost else self.normal_level
            window.setLevel_(level)

        self._synced = state


class X11NativeState:
    """Applies :class:`WindowState` through injected Xlib/XShape modules.

    Args:
        xlib: ``pyglet.libs.x11.xlib`` or a fake with region calls.
        xsync: ``pyglet.libs.x11.xsync`` (XSync + XShape bindings) or fake.
        change_wm_state: optional ``call(atom_name: str, add: bool)`` used
            to add/remove EWMH states such as ``_NET_WM_STATE_ABOVE``.
    """

    def __init__(
        self,
        xlib: Any,
        xsync: Any,
        *,
        display: Any = None,
        window: int = 0,
        view: int = 0,
        change_wm_state: Any | None = None,
    ) -> None:
        self.xlib = xlib
        self.xsync = xsync
        self.display = display
        self.window = window
        self.view = view
        self.change_wm_state = change_wm_state
        self._synced: WindowState | None = None

    def attach(self, display: Any, window: int, view: int) -> None:
        self.display = display
        self.window = window
        self.view = view
        self.reset()

    def reset(self) -> None:
        self._synced = None

    def sync(self, state: WindowState) -> None:
        # Baseline all-off matches a freshly created X window.
        previous = self._synced or WindowState()

        if previous.mouse_passthrough != state.mouse_passthrough:
            self._apply_input_shape(state.mouse_passthrough)

        if previous.topmost != state.topmost:
            if self.change_wm_state is not None:
                self.change_wm_state("_NET_WM_STATE_ABOVE", state.topmost)

        # Per-pixel transparency on X11 is established at window creation
        # by selecting a 32-bit ARGB visual; there is no standard dynamic
        # compositor switch.  The flag is still tracked so lifecycle and
        # capability errors behave consistently across platforms.
        self._synced = state

    def _shape_targets(self) -> list[int]:
        return [xid for xid in (self.window, self.view) if xid]

    def _apply_input_shape(self, enabled: bool) -> None:
        targets = self._shape_targets()
        if not targets or self.display is None:
            return

        if enabled:
            region = self.xlib.XCreateRegion()
            for xid in targets:
                self.xsync.XShapeCombineRegion(
                    self.display, xid, self.xsync.ShapeInput,
                    0, 0, region, self.xsync.ShapeSet,
                )
            self.xlib.XDestroyRegion(region)
        else:
            # Pixmap None (0) + ShapeSet restores the default input region.
            for xid in targets:
                self.xsync.XShapeCombineMask(
                    self.display, xid, self.xsync.ShapeInput,
                    0, 0, 0, self.xsync.ShapeSet,
                )


__all__ = [
    "ASPECTS",
    "MOUSE_PASSTHROUGH",
    "TOPMOST",
    "TRANSPARENT",
    "CocoaNativeState",
    "RecordingNativeState",
    "StateCapabilities",
    "Win32NativeState",
    "WindowState",
    "WindowStateController",
    "X11NativeState",
]
