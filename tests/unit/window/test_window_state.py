"""Tests for the injectable window-state controller.

These tests never create a native window: a RecordingNativeState fake stands
in for the platform native calls.  They cover independent switching,
idempotency, unsupported combinations and the full create/recreate/close
lifecycle.
"""
from __future__ import annotations

import pytest

from pyglet.window import WindowStateException
from pyglet.window.state import (
    MOUSE_PASSTHROUGH,
    TOPMOST,
    TRANSPARENT,
    RecordingNativeState,
    StateCapabilities,
    WindowState,
    WindowStateController,
)


@pytest.fixture
def native():
    return RecordingNativeState()


@pytest.fixture
def controller(native):
    return WindowStateController(native)


class TestInitialState:
    def test_defaults_are_all_off(self, controller):
        assert controller.state == WindowState(False, False, False)
        assert controller.get(TRANSPARENT) is False
        assert controller.get(MOUSE_PASSTHROUGH) is False
        assert controller.get(TOPMOST) is False

    def test_initial_state_is_kept(self, native):
        state = WindowState(transparent=True, mouse_passthrough=True)
        controller = WindowStateController(native, initial=state)
        assert controller.state is state
        # Not realized yet -> no native calls.
        assert native.calls == []

    def test_not_realized_by_default(self, controller):
        assert controller.realized is False


class TestIndependentSwitching:
    def test_enable_passthrough_before_realize_makes_no_native_call(self, controller, native):
        changed = controller.set(MOUSE_PASSTHROUGH, True)
        assert changed is True
        assert controller.get(MOUSE_PASSTHROUGH) is True
        assert native.calls == []

    def test_set_after_realize_calls_native(self, native):
        controller = WindowStateController(native)
        controller.realize()
        native.reset_log()

        assert controller.set(MOUSE_PASSTHROUGH, True) is True
        assert native.calls == [(MOUSE_PASSTHROUGH, True)]

    def test_redundant_set_makes_no_native_call(self, native):
        controller = WindowStateController(
            native, initial=WindowState(mouse_passthrough=True),
        )
        controller.realize()
        native.reset_log()

        assert controller.set(MOUSE_PASSTHROUGH, True) is False
        assert native.calls == []

    def test_each_aspect_switches_independently(self, native):
        controller = WindowStateController(native)
        controller.realize()
        native.reset_log()

        controller.set(TRANSPARENT, True)
        controller.set(MOUSE_PASSTHROUGH, True)
        controller.set(TOPMOST, True)
        assert native.calls == [
            (TRANSPARENT, True),
            (MOUSE_PASSTHROUGH, True),
            (TOPMOST, True),
        ]

        # Turning one off does not disturb the others.
        native.reset_log()
        controller.set(MOUSE_PASSTHROUGH, False)
        assert native.calls == [(MOUSE_PASSTHROUGH, False)]
        assert controller.state == WindowState(True, False, True)

    def test_update_batch(self, native):
        controller = WindowStateController(native)
        controller.realize()
        native.reset_log()

        changed = controller.update(transparent=True, topmost=True)
        assert changed is True
        assert ("transparent", True) in native.calls
        assert ("topmost", True) in native.calls

        assert controller.update(transparent=True, topmost=True) is False

    def test_unknown_aspect_raises(self, controller):
        with pytest.raises(ValueError):
            controller.set("does_not_exist", True)
        with pytest.raises(ValueError):
            controller.update(nope=True)

    def test_truthy_values_are_coerced(self, controller):
        controller.set(MOUSE_PASSTHROUGH, 1)
        assert controller.get(MOUSE_PASSTHROUGH) is True
        controller.set(MOUSE_PASSTHROUGH, 0)
        assert controller.get(MOUSE_PASSTHROUGH) is False


class TestUnsupportedCombinations:
    def test_transparency_without_alpha_raises_clear_error(self, native):
        caps = StateCapabilities(alpha_framebuffer=False)
        controller = WindowStateController(native, capabilities=caps)

        with pytest.raises(WindowStateException, match="alpha"):
            controller.set(TRANSPARENT, True)

        # Desired state and native side are untouched.
        assert controller.get(TRANSPARENT) is False
        assert native.calls == []

    def test_passthrough_without_shape_support_raises(self, native):
        caps = StateCapabilities(mouse_passthrough=False)
        controller = WindowStateController(native, capabilities=caps)

        with pytest.raises(WindowStateException, match="SHAPE|pass-through"):
            controller.set(MOUSE_PASSTHROUGH, True)
        assert controller.get(MOUSE_PASSTHROUGH) is False

    def test_disabling_never_requires_capability(self, native):
        # Start from a permissive, realized state with both aspects on, then
        # downgrade capabilities; turning aspects off must still be allowed.
        caps = StateCapabilities(alpha_framebuffer=False, mouse_passthrough=False)
        controller = WindowStateController(
            native, initial=WindowState(transparent=True, mouse_passthrough=True),
        )
        controller.realize()
        controller.capabilities = caps
        assert controller.set(TRANSPARENT, False) is True
        assert controller.set(MOUSE_PASSTHROUGH, False) is True

    def test_batch_is_atomic_on_unsupported(self, native):
        caps = StateCapabilities(alpha_framebuffer=False)
        controller = WindowStateController(native, capabilities=caps)
        with pytest.raises(WindowStateException):
            controller.update(topmost=True, transparent=True)
        # topmost must not have been partially applied.
        assert controller.get(TOPMOST) is False
        assert native.calls == []

    def test_realize_validates_initial_unsupported_state(self, native):
        caps = StateCapabilities(mouse_passthrough=False)
        controller = WindowStateController(
            native,
            initial=WindowState(mouse_passthrough=True),
            capabilities=caps,
        )
        with pytest.raises(WindowStateException):
            controller.realize()
        assert controller.realized is False


class TestLifecycle:
    def test_realize_pushes_full_desired_state(self, native):
        controller = WindowStateController(
            native,
            initial=WindowState(transparent=True, mouse_passthrough=True),
        )
        controller.realize()
        assert native.synced == WindowState(True, True, False)
        assert native.calls == [
            (TRANSPARENT, True),
            (MOUSE_PASSTHROUGH, True),
        ]

    def test_invalidate_then_realize_reapplies_all_state(self, native):
        controller = WindowStateController(
            native, initial=WindowState(mouse_passthrough=True, topmost=True),
        )
        controller.realize()
        native.reset_log()

        # Simulate fullscreen _recreate destroying the native handle.
        controller.invalidate()
        assert controller.realized is False
        # invalidate() resets the driver's diff baseline as well.
        assert native.synced is None

        # Toggles while unrealized are remembered but not emitted.
        controller.set(TOPMOST, False)
        assert native.calls == []

        # New native handle: the complete desired state is pushed again.
        controller.realize()
        assert native.sync_calls[-1] == WindowState(False, True, False)
        assert (MOUSE_PASSTHROUGH, True) in native.calls

    def test_teardown_resets_native_state_and_closes(self, native):
        controller = WindowStateController(
            native, initial=WindowState(mouse_passthrough=True, topmost=True),
        )
        controller.realize()
        native.reset_log()

        controller.teardown()
        # The native driver receives an all-off sync.
        assert native.synced == WindowState()
        assert (MOUSE_PASSTHROUGH, False) in native.calls
        assert (TOPMOST, False) in native.calls
        assert controller.closed is True

        with pytest.raises(WindowStateException):
            controller.set(MOUSE_PASSTHROUGH, True)

    def test_teardown_swallows_driver_errors(self, native):
        class BrokenDriver:
            def sync(self, state):
                raise OSError("native handle already gone")

        controller = WindowStateController(native)
        controller.realize()
        # Native side dies before the close reset; teardown must not raise.
        controller.driver = BrokenDriver()
        controller.teardown()
        assert controller.closed is True

    def test_teardown_without_realize_is_safe(self, controller):
        controller.teardown()
        assert controller.closed is True

    def test_driver_can_be_injected_after_construction(self, native):
        controller = WindowStateController()
        controller.driver = native
        controller.set(TOPMOST, True)  # not realized yet
        controller.realize()
        assert native.synced == WindowState(topmost=True)
