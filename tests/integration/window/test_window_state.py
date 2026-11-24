"""Integration tests for independent transparency / pass-through state.

These create real (invisible) windows and therefore need a working display
server.  They skip automatically when no display is available.  The
platform-specific *visual* acceptance (actual see-through and click-through)
is covered by the manual steps in
``doc/programming_guide/window_transparency.rst`` and the example in
``examples/window/transparent_overlay.py``.
"""
from __future__ import annotations

import os

import pytest

import pyglet
from pyglet.window import WindowStateException

pytestmark = pytest.mark.integration


@pytest.fixture
def window_factory():
    created = []

    def _create(*args, **kwargs):
        kwargs.setdefault("visible", False)
        try:
            win = pyglet.window.Window(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - environment without display/GL
            pytest.skip(f"No usable native window/display: {exc!r}")
        created.append(win)
        return win

    yield _create

    for win in created:
        try:
            win.close()
        except Exception:  # noqa: BLE001
            pass


def test_plain_window_defaults(window_factory):
    win = window_factory(320, 240)
    assert win.transparent is False
    assert win.mouse_passthrough is False
    assert win.topmost is False
    assert win.transparent_supported is False


def test_enabling_transparency_without_alpha_is_clear_error(window_factory):
    win = window_factory(320, 240)
    with pytest.raises(WindowStateException):
        win.set_transparent(True)
    # Failed request must not flip the state.
    assert win.transparent is False


def test_overlay_style_resolves_bundled_state(window_factory):
    win = window_factory(240, 180, style="overlay")
    assert win.transparent is True
    assert win.mouse_passthrough is True
    assert win.topmost is True
    assert win.transparent_supported is True


def test_independent_opaque_click_through(window_factory):
    win = window_factory(240, 180, mouse_passthrough=True)
    assert win.transparent is False
    assert win.mouse_passthrough is True
    win.set_mouse_passthrough(False)
    assert win.mouse_passthrough is False


def test_state_survives_recreate(window_factory):
    win = window_factory(200, 160, style="overlay")
    win.set_mouse_passthrough(False)

    # This is what set_fullscreen() invokes internally.
    win._recreate(["fullscreen"])

    assert win.transparent is True
    assert win.mouse_passthrough is False
    assert win.topmost is True
    # The fresh native window must be realized with the desired state.
    assert win._window_state.realized is True


def test_close_is_idempotent_and_marks_controller_closed(window_factory):
    win = window_factory(160, 120, mouse_passthrough=True)
    win.close()
    assert win._window_state.closed is True
    win.close()  # must not raise


def test_transparent_and_normal_windows_coexist(window_factory):
    overlay = window_factory(180, 140, style="overlay")
    normal = window_factory(220, 180)

    # Independent state.
    assert overlay.mouse_passthrough is True
    assert normal.mouse_passthrough is False

    # The normal window's driver never shaped/flagged anything on creation.
    normal_driver = normal._window_state.driver
    assert normal_driver is not None
    assert normal_driver._synced == normal._window_state.state

    # Toggling one does not mutate the other's desired state.
    overlay.set_mouse_passthrough(False)
    assert overlay.mouse_passthrough is False
    assert normal.mouse_passthrough is False

    # Closing the overlay leaves the normal window usable.
    overlay.close()
    assert normal._window is not None
    normal.dispatch_events()
    normal.set_caption("still alive")
