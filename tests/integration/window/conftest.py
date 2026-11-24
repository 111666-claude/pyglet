"""Disable the shadow GL window before window modules are imported.

The state integration tests create their own (invisible) windows and skip
when no display/GL context is usable, so the implicit shadow window is
unnecessary and would turn a headless run into a collection error.
"""
import pyglet

pyglet.options["shadow_window"] = False
