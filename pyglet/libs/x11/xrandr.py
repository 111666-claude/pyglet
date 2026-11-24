"""Minimal ctypes wrappers for the XRandR extension.

Only the calls/structures needed to enumerate active outputs and their
current mode are bound.  The structure layout matches Xrandr.h and the
bindings mirror the style of :mod:`pyglet.libs.x11.xinerama`.
"""
from __future__ import annotations

import warnings
from ctypes import (
    Structure,
    byref,
    c_char_p,
    c_int,
    c_uint,
    c_ulong,
    c_ushort,
    POINTER,
)

import pyglet
from pyglet.libs.x11.xlib import Display, Time, Window

try:
    _lib = pyglet.lib.load_library("Xrandr")
except ImportError:
    if pyglet.options.get("debug_lib"):
        warnings.warn("Xrandr could not be loaded.", stacklevel=2)
    raise

# X11 XIDs are unsigned long.
RRCrtc = c_ulong
RROutput = c_ulong
RRMode = c_ulong
Connection = c_ushort
SubpixelOrder = c_ushort
XRRModeFlags = c_ulong

#: XRROutputInfo.connection value for a connected output.
RR_CONNECTED = 0
#: XRROutputInfo.connection value for a disconnected output.
RR_DISCONNECTED = 1


class XRRModeInfo(Structure):
    _fields_ = [
        ("id", RRMode),
        ("width", c_uint),
        ("height", c_uint),
        # Note: RandR dotClock is expressed in Hz (unlike XF86VidMode's
        # dotclock, which is in kHz).
        ("dotClock", c_ulong),
        ("hSyncStart", c_uint),
        ("hSyncEnd", c_uint),
        ("hTotal", c_uint),
        ("hSkew", c_uint),
        ("vSyncStart", c_uint),
        ("vSyncEnd", c_uint),
        ("vTotal", c_uint),
        ("name", c_char_p),
        ("nameLength", c_uint),
        ("modeFlags", XRRModeFlags),
    ]


class XRRScreenResources(Structure):
    _fields_ = [
        ("timestamp", Time),
        ("configTimestamp", Time),
        ("ncrtc", c_int),
        ("crtcs", POINTER(RRCrtc)),
        ("noutput", c_int),
        ("outputs", POINTER(RROutput)),
        ("nmode", c_int),
        ("modes", POINTER(XRRModeInfo)),
    ]


class XRROutputInfo(Structure):
    _fields_ = [
        ("timestamp", Time),
        ("crtc", RRCrtc),
        ("name", c_char_p),
        ("nameLen", c_int),
        ("mm_width", c_ulong),
        ("mm_height", c_ulong),
        ("connection", Connection),
        ("subpixel_order", SubpixelOrder),
        ("ncrtc", c_int),
        ("crtcs", POINTER(RRCrtc)),
        ("nclone", c_int),
        ("clones", POINTER(RROutput)),
        ("nmode", c_int),
        ("npreferred", c_int),
        ("modes", POINTER(RRMode)),
    ]


class XRRCrtcInfo(Structure):
    _fields_ = [
        ("timestamp", Time),
        ("x", c_int),
        ("y", c_int),
        ("width", c_uint),
        ("height", c_uint),
        ("mode", RRMode),
        ("rotation", c_int),
        ("noutput", c_int),
        ("outputs", POINTER(RROutput)),
        ("rotations", c_ushort),
        ("npossible", c_int),
        ("possible", POINTER(RROutput)),
    ]


XRRQueryVersion = _lib.XRRQueryVersion
XRRQueryVersion.argtypes = [POINTER(Display), POINTER(c_int), POINTER(c_int)]
XRRQueryVersion.restype = c_int

XRRGetScreenResources = _lib.XRRGetScreenResources
XRRGetScreenResources.argtypes = [POINTER(Display), Window]
XRRGetScreenResources.restype = POINTER(XRRScreenResources)

XRRGetScreenResourcesCurrent = _lib.XRRGetScreenResourcesCurrent
XRRGetScreenResourcesCurrent.argtypes = [POINTER(Display), Window]
XRRGetScreenResourcesCurrent.restype = POINTER(XRRScreenResources)

XRRFreeScreenResources = _lib.XRRFreeScreenResources
XRRFreeScreenResources.argtypes = [POINTER(XRRScreenResources)]
XRRFreeScreenResources.restype = None

XRRGetOutputPrimary = _lib.XRRGetOutputPrimary
XRRGetOutputPrimary.argtypes = [POINTER(Display), Window]
XRRGetOutputPrimary.restype = RROutput

XRRGetOutputInfo = _lib.XRRGetOutputInfo
XRRGetOutputInfo.argtypes = [POINTER(Display), POINTER(XRRScreenResources), RROutput]
XRRGetOutputInfo.restype = POINTER(XRROutputInfo)

XRRFreeOutputInfo = _lib.XRRFreeOutputInfo
XRRFreeOutputInfo.argtypes = [POINTER(XRROutputInfo)]
XRRFreeOutputInfo.restype = None

XRRGetCrtcInfo = _lib.XRRGetCrtcInfo
XRRGetCrtcInfo.argtypes = [POINTER(Display), POINTER(XRRScreenResources), RRCrtc]
XRRGetCrtcInfo.restype = POINTER(XRRCrtcInfo)

XRRFreeCrtcInfo = _lib.XRRFreeCrtcInfo
XRRFreeCrtcInfo.argtypes = [POINTER(XRRCrtcInfo)]
XRRFreeCrtcInfo.restype = None


def query_version(display) -> tuple[int, int] | None:
    """Return the RandR ``(major, minor)`` version, or ``None`` on failure."""
    major, minor = c_int(), c_int()
    if XRRQueryVersion(display, byref(major), byref(minor)):
        return major.value, minor.value
    return None


__all__ = [
    "RRCrtc", "RROutput", "RRMode",
    "RR_CONNECTED", "RR_DISCONNECTED",
    "XRRModeInfo", "XRRScreenResources", "XRROutputInfo", "XRRCrtcInfo",
    "XRRQueryVersion",
    "XRRGetScreenResources", "XRRGetScreenResourcesCurrent",
    "XRRFreeScreenResources", "XRRGetOutputPrimary",
    "XRRGetOutputInfo", "XRRFreeOutputInfo",
    "XRRGetCrtcInfo", "XRRFreeCrtcInfo",
    "query_version",
]
