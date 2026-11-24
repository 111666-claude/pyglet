"""Backend independent screen identity and refresh-rate helpers.

This module contains **no native API calls**.  The X11, Win32 and Cocoa
backends translate their own structures into :class:`ScreenInfo` records
through the small ``*_screen_info`` helpers below; all identity and
fallback policy then lives here where it can be unit tested with fakes.

Identity policy
---------------
A monitor may be reachable through several identifiers of different
stability.  When :class:`ScreenRegistry` compares a freshly enumerated
snapshot with previous ones it uses, strongest first:

1. a hardware identity (EDID manufacturer/product/serial),
2. a stable operating-system output/device path (X11/RandR connector
   name such as ``DP-1``, the Win32 device name/path, ...),
3. the native identifier (the Cocoa ``CGDirectDisplayID`` or the X11
   RandR output XID; Win32 ``HMONITOR`` values are explicitly *not*
   treated as stable),
4. an unambiguous name + resolution fingerprint,
5. a position/ordinal fallback.

Two screens with the same monitor name are never collapsed into one
identity: the native identifiers separate them and their labels gain a
stable ``#2``/``#3`` suffix.  When a screen is disconnected its identity
is retained by the registry and marked as absent; reconnecting the same
display reattaches the original key.  If two indistinguishable same-name
screens could match either stored identity, the registry deliberately
does *not* guess and allocates a fresh identity instead.

Refresh rates are always exposed in **Hertz**.  The conversion helpers
name the source unit explicitly because the two X11 APIs disagree:
``XF86VidMode`` reports the dot clock in kilohertz while RandR reports it
in hertz.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

UNKNOWN_MONITOR_NAME = "Unknown"
"""Fallback monitor name when the platform cannot provide one."""


# ---------------------------------------------------------------------------
# Refresh-rate normalization
# ---------------------------------------------------------------------------

def normalize_refresh_rate(value: Any) -> Optional[float]:
    """Normalize a refresh rate that is already expressed in Hz.

    Non-numeric, non-finite and non-positive values are invalid and become
    ``None`` so callers always observe either a positive Hz value or no
    value at all -- never a sentinel such as ``0`` or ``1``.
    """
    if isinstance(value, bool):
        # ``True`` would otherwise be accepted as 1 Hz.
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0.0:
        return None
    return rate


def refresh_rate_from_hz(value: Any) -> Optional[float]:
    """Validate a refresh rate already measured in Hertz."""
    return normalize_refresh_rate(value)


def refresh_rate_from_win32(dm_display_frequency: Any) -> Optional[float]:
    """Convert a Win32 ``DEVMODE.dmDisplayFrequency`` value.

    Microsoft documents ``0`` and ``1`` as the display hardware default
    rather than real frequencies, so both map to ``None``.
    """
    try:
        frequency = int(dm_display_frequency)
    except (TypeError, ValueError):
        return None
    if frequency <= 1:
        return None
    return float(frequency)


def refresh_rate_from_cocoa(cg_refresh_rate: Any) -> Optional[float]:
    """Convert ``CGDisplayModeGetRefreshRate`` output.

    CoreGraphics returns ``0`` for displays without a fixed refresh rate
    (notably some built-in panels), which is treated as "unknown".
    """
    return normalize_refresh_rate(cg_refresh_rate)


def refresh_rate_from_dotclock(dotclock: Any, htotal: Any, vtotal: Any,
                               *, dotclock_unit: str) -> Optional[float]:
    """Calculate a vertical refresh rate from a pixel clock and totals.

    ``dotclock_unit`` is either ``"Hz"`` (RandR ``XRRModeInfo.dotClock``)
    or ``"kHz"`` (``XF86VidModeModeInfo.dotclock``).  Mixing the two units
    produces a thousand-fold error, so the unit is an explicit argument.
    Zero or negative timings return ``None`` instead of raising.
    """
    try:
        clock = float(dotclock)
        h_total = int(htotal)
        v_total = int(vtotal)
    except (TypeError, ValueError):
        return None

    if dotclock_unit.lower() in ("khz", "kilohertz"):
        clock_hz = clock * 1000.0
    elif dotclock_unit.lower() in ("hz", "hertz"):
        clock_hz = clock
    else:
        raise ValueError(f"dotclock_unit must be 'Hz' or 'kHz', not {dotclock_unit!r}")

    if clock_hz <= 0 or h_total <= 0 or v_total <= 0:
        return None

    rate = clock_hz / (h_total * v_total)
    return normalize_refresh_rate(rate)


def refresh_rate_from_xf86vm(dotclock_khz: Any, htotal: Any, vtotal: Any) -> Optional[float]:
    """Refresh rate (Hz) from an XF86VidMode dot clock measured in kHz."""
    return refresh_rate_from_dotclock(dotclock_khz, htotal, vtotal, dotclock_unit="kHz")


def refresh_rate_from_xrandr(dotclock_hz: Any, htotal: Any, vtotal: Any) -> Optional[float]:
    """Refresh rate (Hz) from a RandR mode dot clock measured in Hz."""
    return refresh_rate_from_dotclock(dotclock_hz, htotal, vtotal, dotclock_unit="Hz")


# ---------------------------------------------------------------------------
# Native data contract
# ---------------------------------------------------------------------------

def _clean_name(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        try:
            if name is None:
                return None
            name = str(name)
        except Exception:  # pragma: no cover - defensive
            return None
    name = name.strip()
    if not name or name.lower() == UNKNOWN_MONITOR_NAME.lower():
        return None
    return name


def _stable_hash(*parts: Any) -> str:
    """Deterministic short hash (unlike the salted builtin ``hash``)."""
    material = "|".join(str(part) for part in parts)
    return hashlib.sha1(material.encode("utf-8", "replace")).hexdigest()[:12]


@dataclass
class ScreenInfo:
    """Normalized description of one active monitor.

    Backends build these from their native structures; the registry turns
    them into stable :class:`ScreenIdentity` objects.
    """

    backend: str = "generic"
    ordinal: int = 0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    is_primary: bool = False
    refresh_rate: Optional[float] = None

    #: Native identifier (XID / CGDirectDisplayID / HMONITOR value / ...).
    native_id: Any = None
    #: Whether :attr:`native_id` is stable enough to derive a saved key.
    native_id_is_stable: bool = False
    #: Stable output/device identifier (connector name, device path/name).
    stable_native_id: Optional[str] = None
    #: Strongest identity available: EDID serial/manufacturer/product.
    hardware_id: Optional[str] = None
    #: Raw monitor/friendly/connector name (may be duplicated).
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.name = _clean_name(self.name)
        self.refresh_rate = normalize_refresh_rate(self.refresh_rate)
        try:
            self.x = int(self.x)
            self.y = int(self.y)
            self.width = int(self.width)
            self.height = int(self.height)
            self.ordinal = int(self.ordinal)
        except (TypeError, ValueError):
            # Geometry stays zeroed if a backend feeds garbage; callers
            # decide whether zero-sized entries are filtered.
            pass

    @property
    def valid_geometry(self) -> bool:
        return self.width > 0 and self.height > 0

    @property
    def fingerprint(self) -> str:
        """Name + resolution fingerprint used for reconnect matching."""
        name = self.name or UNKNOWN_MONITOR_NAME
        return f"{name.lower()}|{self.width}x{self.height}"

    @property
    def geometry_token(self) -> str:
        return f"{self.x},{self.y},{self.width}x{self.height}"


def assign_primary_fallback(infos: Sequence[ScreenInfo]) -> Sequence[ScreenInfo]:
    """Ensure exactly one record is flagged primary when the OS does not.

    If the native API reports no primary screen, the screen whose top-left
    corner is ``(0, 0)`` is chosen (matching pyglet's historical behavior);
    otherwise the first entry is flagged.  Already flagged records are left
    untouched.
    """
    if not infos:
        return infos
    if any(info.is_primary for info in infos):
        return infos

    chosen = next((info for info in infos if info.x == 0 and info.y == 0), infos[0])
    chosen.is_primary = True
    return infos


# ---------------------------------------------------------------------------
# Backend record mappers (pure: safe to test on any operating system)
# ---------------------------------------------------------------------------

def xrandr_screen_info(*, output_id: int, name: Optional[str], connected: bool,
                       crtc_id: int, x: int, y: int, width: int, height: int,
                       primary_output_id: int, dot_clock_hz: Any,
                       h_total: int, v_total: int, ordinal: int = 0,
                       backend: str = "x11") -> Optional[ScreenInfo]:
    """Build a :class:`ScreenInfo` from extracted RandR primitives.

    Returns ``None`` for outputs that are disconnected, not attached to a
    CRTC or have no active mode/geometry, mirroring the screens pyglet has
    historically enumerated.
    """
    if not connected or not crtc_id or width <= 0 or height <= 0:
        return None

    connector = _clean_name(name)
    return ScreenInfo(
        backend=backend,
        ordinal=ordinal,
        x=x, y=y, width=width, height=height,
        is_primary=bool(primary_output_id) and output_id == primary_output_id,
        refresh_rate=refresh_rate_from_xrandr(dot_clock_hz, h_total, v_total),
        native_id=output_id,
        # RandR output XIDs are stable for the life of the X server; the
        # connector name is the cross-session anchor.
        native_id_is_stable=True,
        stable_native_id=connector,
        name=connector,
    )


def win32_screen_info(*, hmonitor: Any, device_name: Optional[str],
                      is_primary: bool, x: int, y: int, width: int, height: int,
                      dm_display_frequency: Any, friendly_name: Optional[str] = None,
                      ordinal: int = 0, backend: str = "win32") -> ScreenInfo:
    """Build a :class:`ScreenInfo` from Win32 monitor/mode primitives.

    ``device_name`` is the GDI name such as ``\\\\.\\DISPLAY1`` and is used
    as the stable output token.  ``hmonitor`` is an ``HMONITOR`` handle and
    is intentionally *not* considered stable across hotplug cycles.
    """
    device = _clean_name(device_name)
    friendly = _clean_name(friendly_name)
    return ScreenInfo(
        backend=backend,
        ordinal=ordinal,
        x=x, y=y, width=width, height=height,
        is_primary=bool(is_primary),
        refresh_rate=refresh_rate_from_win32(dm_display_frequency),
        native_id=hmonitor,
        native_id_is_stable=False,
        stable_native_id=device,
        # Prefer the friendly name; keep the GDI device name available as
        # the display label fallback via ``stable_native_id``.
        name=friendly or device,
    )


def cocoa_screen_info(*, display_id: Any, x: int, y: int, width: int, height: int,
                      is_main: bool, refresh_rate_hz: Any,
                      localized_name: Optional[str] = None,
                      ordinal: int = 0, backend: str = "cocoa") -> ScreenInfo:
    """Build a :class:`ScreenInfo` from CoreGraphics/AppKit primitives."""
    return ScreenInfo(
        backend=backend,
        ordinal=ordinal,
        x=x, y=y, width=width, height=height,
        is_primary=bool(is_main),
        refresh_rate=refresh_rate_from_cocoa(refresh_rate_hz),
        native_id=display_id,
        # CGDirectDisplayID identifies a display for the window-server
        # session and is preserved across unplug/replug of the same panel.
        native_id_is_stable=True,
        stable_native_id=str(display_id) if display_id is not None else None,
        name=_clean_name(localized_name),
    )


def fallback_screen_info(*, x: int, y: int, width: int, height: int,
                         ordinal: int = 0, backend: str = "generic",
                         name: Optional[str] = None) -> ScreenInfo:
    """Build an identity from geometry alone when no native API exists."""
    return ScreenInfo(
        backend=backend,
        ordinal=ordinal,
        x=x, y=y, width=width, height=height,
        is_primary=(x == 0 and y == 0),
        native_id=_stable_hash(backend, ordinal, x, y, width, height),
        native_id_is_stable=False,
        name=name,
    )


# ---------------------------------------------------------------------------
# Identity objects and registry
# ---------------------------------------------------------------------------

@dataclass
class ScreenIdentity:
    """Stable identity and queryable facts for one screen."""

    key: str
    """Saveable identity key; resolvable through :class:`ScreenRegistry`."""

    ordinal: int
    """Persistent 1-based number assigned on first sight."""

    label: str
    """Unique human-readable label (duplicate names get ``#2`` suffixes)."""

    name: str = UNKNOWN_MONITOR_NAME
    """Raw monitor name, or :data:`UNKNOWN_MONITOR_NAME`."""

    native_id: Any = None
    is_primary: bool = False
    refresh_rate: Optional[float] = None
    connected: bool = True

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    hardware_id: Optional[str] = None
    stable_native_id: Optional[str] = None

    def update_from(self, info: ScreenInfo) -> None:
        self.name = info.name or UNKNOWN_MONITOR_NAME
        self.native_id = info.native_id
        self.is_primary = info.is_primary
        self.refresh_rate = info.refresh_rate
        self.x, self.y = info.x, info.y
        self.width, self.height = info.width, info.height
        self.hardware_id = info.hardware_id
        self.stable_native_id = info.stable_native_id


def default_key(info: ScreenInfo) -> str:
    """Deterministic saveable key derived from the strongest token."""
    prefix = info.backend or "screen"
    if info.hardware_id:
        return f"{prefix}:hw:{info.hardware_id}"
    if info.stable_native_id:
        return f"{prefix}:out:{info.stable_native_id}"
    if info.native_id_is_stable and info.native_id is not None:
        return f"{prefix}:id:{info.native_id}"
    if info.valid_geometry:
        return f"{prefix}:fp:{_stable_hash(info.fingerprint)}"
    return f"{prefix}:geom:{_stable_hash(info.geometry_token)}"


def build_identity(info: ScreenInfo, ordinal: int = 1,
                   key: Optional[str] = None) -> ScreenIdentity:
    """Build a standalone identity without a :class:`ScreenRegistry`."""
    identity = ScreenIdentity(
        key=key or default_key(info),
        ordinal=ordinal,
        label=info.name or UNKNOWN_MONITOR_NAME,
    )
    identity.update_from(info)
    return identity


class ScreenRegistry:
    """Track screens across enumerations, hotplug and reconnect.

    The registry is intentionally stateful but never calls native APIs
    itself; feed it lists of :class:`ScreenInfo` via :meth:`synchronize`.
    """

    def __init__(self, backend: str = "screen") -> None:
        self.backend = backend
        self._entries: Dict[str, ScreenIdentity] = {}
        self._next_ordinal = 1
        #: Keys of the latest snapshot in enumeration order.
        self._current_keys: List[str] = []

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._current_keys)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._entries

    @property
    def screens(self) -> List[ScreenIdentity]:
        """Identities from the latest snapshot (enumeration order)."""
        return [self._entries[key] for key in self._current_keys if key in self._entries]

    @property
    def primary(self) -> Optional[ScreenIdentity]:
        """The currently connected primary screen, if any."""
        for key in self._current_keys:
            entry = self._entries.get(key)
            if entry is not None and entry.is_primary:
                return entry
        return None

    def get(self, key: str) -> Optional[ScreenIdentity]:
        """Return an identity even if it is currently disconnected."""
        return self._entries.get(key)

    def resolve(self, key: str) -> Optional[ScreenIdentity]:
        """Resolve a saved key to a *currently connected* identity.

        Returns ``None`` when the key is unknown or its screen is unplugged,
        allowing the caller to fall back to the default screen.
        """
        entry = self._entries.get(key)
        if entry is not None and entry.connected and key in self._current_keys:
            return entry
        return None

    # -- core --------------------------------------------------------------

    def synchronize(self, infos: Optional[Sequence[ScreenInfo]]) -> List[ScreenIdentity]:
        """Replace the current snapshot and reconcile identities.

        Returns identities aligned one-to-one and in the same order as
        ``infos``.  Passing an empty sequence represents an empty display
        list: all known identities become disconnected and an empty list is
        returned.
        """
        infos = list(infos or [])
        assign_primary_fallback(infos)

        matched_key: List[Optional[str]] = [None] * len(infos)
        used_keys: set = set()

        for _score, index, key in self._rank_matches(infos):
            if matched_key[index] is not None or key in used_keys:
                continue
            matched_key[index] = key
            used_keys.add(key)

        current_keys: List[str] = []
        for index, info in enumerate(infos):
            key = matched_key[index]
            if key is not None:
                entry = self._entries[key]
                entry.connected = True
                entry.update_from(info)
            else:
                key = self._create_key(info)
                key = self._allocate_key(key)
                entry = ScreenIdentity(
                    key=key,
                    ordinal=self._next_ordinal,
                    label=info.name or UNKNOWN_MONITOR_NAME,
                )
                self._next_ordinal += 1
                entry.update_from(info)
                self._entries[key] = entry
            current_keys.append(key)

        current_set = set(current_keys)
        for key, entry in self._entries.items():
            entry.connected = key in current_set

        self._current_keys = current_keys
        self._update_labels()
        return [self._entries[key] for key in current_keys]

    # -- matching ----------------------------------------------------------

    def _rank_matches(self, infos: Sequence[ScreenInfo]):
        """Yield ``(score, info_index, key)`` candidates, strongest first."""
        fingerprint_counts: Dict[str, int] = {}
        for entry in self._entries.values():
            fp = self._entry_fingerprint(entry)
            if fp is not None:
                fingerprint_counts[fp] = fingerprint_counts.get(fp, 0) + 1
        info_fp_counts: Dict[str, int] = {}
        for info in infos:
            info_fp_counts[info.fingerprint] = info_fp_counts.get(info.fingerprint, 0) + 1

        candidates = []
        for index, info in enumerate(infos):
            for key, entry in self._entries.items():
                score = self._match_score(
                    info, entry, fingerprint_counts, info_fp_counts,
                )
                if score > 0:
                    candidates.append((score, index, key))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return candidates

    @staticmethod
    def _entry_fingerprint(entry: ScreenIdentity) -> Optional[str]:
        if not entry.width or not entry.height:
            return None
        name = (entry.name or UNKNOWN_MONITOR_NAME).lower()
        return f"{name}|{entry.width}x{entry.height}"

    def _match_score(self, info: ScreenInfo, entry: ScreenIdentity,
                     entry_fp_counts: Dict[str, int],
                     info_fp_counts: Dict[str, int]) -> int:
        # 4: hardware identity always wins, even across port changes.
        if info.hardware_id and entry.hardware_id:
            # Equal -> strongest match; unequal -> never bind these.
            return 4 if info.hardware_id == entry.hardware_id else 0

        # 3: stable output/device token.
        if info.stable_native_id and entry.stable_native_id:
            if info.stable_native_id == entry.stable_native_id:
                return 3
            # Both known but different -> never match on a weaker signal.
            return 0

        # 2: raw native identifier.
        if info.native_id is not None and entry.native_id is not None:
            if info.native_id == entry.native_id:
                return 2

        # 1: fingerprint, only when unambiguous on both sides.  This avoids
        # swapping two identical same-name monitors.
        fp = info.fingerprint
        if (info.valid_geometry and entry.width and entry.height
                and entry_fp_counts.get(fp, 0) == 1
                and info_fp_counts.get(fp, 0) == 1
                and self._entry_fingerprint(entry) == fp):
            return 1

        return 0

    # -- key/label allocation ---------------------------------------------

    def _create_key(self, info: ScreenInfo) -> str:
        return default_key(info)

    def _allocate_key(self, base: str) -> str:
        if base not in self._entries:
            return base
        suffix = 2
        while f"{base}-{suffix}" in self._entries:
            suffix += 1
        return f"{base}-{suffix}"

    def _update_labels(self) -> None:
        groups: Dict[str, List[ScreenIdentity]] = {}
        for key in self._current_keys:
            entry = self._entries[key]
            groups.setdefault(entry.name, []).append(entry)

        for name, entries in groups.items():
            entries.sort(key=lambda e: e.ordinal)
            if len(entries) == 1:
                entries[0].label = name
                continue
            for position, entry in enumerate(entries):
                if position == 0:
                    entry.label = name
                else:
                    entry.label = f"{name} #{position + 1}"


__all__ = [
    "UNKNOWN_MONITOR_NAME",
    "ScreenInfo",
    "ScreenIdentity",
    "ScreenRegistry",
    "default_key",
    "build_identity",
    "assign_primary_fallback",
    "normalize_refresh_rate",
    "refresh_rate_from_hz",
    "refresh_rate_from_win32",
    "refresh_rate_from_cocoa",
    "refresh_rate_from_dotclock",
    "refresh_rate_from_xf86vm",
    "refresh_rate_from_xrandr",
    "xrandr_screen_info",
    "win32_screen_info",
    "cocoa_screen_info",
    "fallback_screen_info",
]
