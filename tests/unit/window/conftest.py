"""Pytest configuration for window state unit tests.

These tests exercise the platform-independent state controller and injected
native drivers, so they must not require a display server or GL context.
Disable pyglet's implicit 1x1 "shadow" window before any test module imports
``pyglet.window``; otherwise importing the package creates a real native
window and fails on truly headless machines (CI uses xvfb, but this keeps
the injected tests runnable anywhere).
"""
import pyglet

pyglet.options["shadow_window"] = False
