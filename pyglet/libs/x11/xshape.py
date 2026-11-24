"""Minimal bindings for the X11 SHAPE extension (libXext).

Only the calls needed for input-region / click-through handling are bound.
The library and functions are resolved lazily/compatibly so importing this
module never fails on platforms without X11; use :func:`shape_available` to
query support for a display.
"""

from __future__ import annotations

import ctypes
from ctypes import byref, c_int

import pyglet.lib

__all__ = [
    "ShapeBounding",
    "ShapeClip",
    "ShapeInput",
    "ShapeSet",
    "ShapeUnion",
    "ShapeIntersect",
    "ShapeSubtract",
    "ShapeInvert",
    "XShapeCombineMask",
    "XShapeCombineRegion",
    "XShapeQueryExtension",
    "shape_available",
]

# Shape kind
ShapeBounding = 0
ShapeClip = 1
ShapeInput = 2

# Shape operation
ShapeSet = 0
ShapeUnion = 1
ShapeIntersect = 2
ShapeSubtract = 3
ShapeInvert = 4

try:
    _lib = pyglet.lib.load_library("Xext")
except (ImportError, OSError):  # pragma: no cover - exercised only without Xext
    _lib = None


if _lib is not None:
    XShapeQueryExtension = _lib.XShapeQueryExtension
    XShapeQueryExtension.argtypes = [ctypes.c_void_p,
                                     ctypes.POINTER(c_int),
                                     ctypes.POINTER(c_int)]
    XShapeQueryExtension.restype = c_int

    XShapeCombineRegion = _lib.XShapeCombineRegion
    XShapeCombineRegion.argtypes = [
        ctypes.c_void_p,  # Display *
        ctypes.c_void_p,  # destination Window
        c_int,            # shape_kind
        c_int, c_int,     # x/y offset
        ctypes.c_void_p,  # Region
        c_int,            # shape operation
    ]
    XShapeCombineRegion.restype = None

    XShapeCombineMask = _lib.XShapeCombineMask
    XShapeCombineMask.argtypes = [
        ctypes.c_void_p,  # Display *
        ctypes.c_void_p,  # destination Window
        c_int,            # shape_kind
        c_int, c_int,     # x/y offset
        ctypes.c_ulong,   # Pixmap (None == 0)
        c_int,            # shape operation
    ]
    XShapeCombineMask.restype = None
else:  # pragma: no cover - systems without libXext
    def XShapeQueryExtension(*_args, **_kwargs):  # noqa: D401
        return 0

    def XShapeCombineRegion(*_args, **_kwargs):
        return None

    def XShapeCombineMask(*_args, **_kwargs):
        return None


def shape_available(display) -> bool:
    """Return True if the SHAPE extension is available on ``display``."""
    if _lib is None:
        return False
    event_base = c_int()
    error_base = c_int()
    try:
        return bool(XShapeQueryExtension(
            ctypes.cast(display, ctypes.c_void_p),
            byref(event_base), byref(error_base),
        ))
    except (ctypes.ArgumentError, TypeError):
        # display may already be an integer/void pointer value.
        try:
            return bool(XShapeQueryExtension(
                display, byref(event_base), byref(error_base),
            ))
        except Exception:  # noqa: BLE001 - capability probe must not raise
            return False
