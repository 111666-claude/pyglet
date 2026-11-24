"""Contract tests for refresh-rate normalization and units.

The key guarantee is that every public refresh value is expressed in Hz
(or ``None`` when the native value is missing/invalid) and that the two
X11 dot-clock units (XF86VidMode kHz vs RandR Hz) are never mixed.
"""
import math

import pytest

from pyglet.display.identity import (
    normalize_refresh_rate,
    refresh_rate_from_cocoa,
    refresh_rate_from_dotclock,
    refresh_rate_from_win32,
    refresh_rate_from_xf86vm,
    refresh_rate_from_xrandr,
)


# A 148.5 MHz modeline with 2200 x 1125 totals is exactly 60 Hz.
_HZ_DOTCLOCK = 148_500_000
_KHZ_DOTCLOCK = 148_500
_HTOTAL = 2200
_VTOTAL = 1125


@pytest.mark.unit
class TestNormalizeHz:
    @pytest.mark.parametrize("value,expected", [
        (60, 60.0),
        (59.94, 59.94),
        ("144", 144.0),
        (120.0, 120.0),
    ])
    def test_valid_values(self, value, expected):
        assert normalize_refresh_rate(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [
        0, -1, 0.0, None, "", "abc", float("nan"), float("inf"), True, False,
    ])
    def test_invalid_values_become_none(self, value):
        assert normalize_refresh_rate(value) is None


@pytest.mark.unit
class TestWin32Rate:
    @pytest.mark.parametrize("frequency", [0, 1])
    def test_documented_default_sentinels_are_unknown(self, frequency):
        # dmDisplayFrequency 0/1 mean "hardware default", not 0/1 Hz.
        assert refresh_rate_from_win32(frequency) is None

    @pytest.mark.parametrize("frequency,expected", [
        (60, 60.0), (144, 144.0), (240, 240.0), ("120", 120.0),
    ])
    def test_real_frequencies(self, frequency, expected):
        assert refresh_rate_from_win32(frequency) == expected

    def test_garbage(self):
        assert refresh_rate_from_win32(None) is None
        assert refresh_rate_from_win32("n/a") is None


@pytest.mark.unit
class TestCocoaRate:
    def test_zero_is_unknown(self):
        # CGDisplayModeGetRefreshRate returns 0 for some built-in panels.
        assert refresh_rate_from_cocoa(0) is None
        assert refresh_rate_from_cocoa(0.0) is None

    def test_fractional_hz_preserved(self):
        assert refresh_rate_from_cocoa(59.94005) == pytest.approx(59.94005)


@pytest.mark.unit
class TestX11DotClockUnits:
    def test_xrandr_dotclock_is_hz(self):
        rate = refresh_rate_from_xrandr(_HZ_DOTCLOCK, _HTOTAL, _VTOTAL)
        assert rate == pytest.approx(60.0)

    def test_xf86vm_dotclock_is_khz(self):
        rate = refresh_rate_from_xf86vm(_KHZ_DOTCLOCK, _HTOTAL, _VTOTAL)
        assert rate == pytest.approx(60.0)

    def test_explicit_unit_argument(self):
        assert refresh_rate_from_dotclock(
            _HZ_DOTCLOCK, _HTOTAL, _VTOTAL, dotclock_unit="Hz",
        ) == pytest.approx(60.0)
        assert refresh_rate_from_dotclock(
            _KHZ_DOTCLOCK, _HTOTAL, _VTOTAL, dotclock_unit="kHz",
        ) == pytest.approx(60.0)

    def test_units_are_not_interchangeable(self):
        # Feeding the Hz-valued RandR clock into the kHz helper multiplies
        # by 1000 -> a clearly wrong ~60 kHz "rate"; the helpers exist so
        # callers cannot make this confusion silently.
        wrong = refresh_rate_from_xf86vm(_HZ_DOTCLOCK, _HTOTAL, _VTOTAL)
        assert wrong == pytest.approx(60_000.0)
        correct = refresh_rate_from_xrandr(_HZ_DOTCLOCK, _HTOTAL, _VTOTAL)
        assert math.isclose(wrong, correct * 1000.0)

    @pytest.mark.parametrize("clock,h,v", [
        (0, 100, 100),
        (1000, 0, 100),
        (1000, 100, 0),
        (-1, 100, 100),
        (None, 100, 100),
        ("x", 100, 100),
    ])
    def test_invalid_timings_return_none(self, clock, h, v):
        assert refresh_rate_from_xrandr(clock, h, v) is None
        assert refresh_rate_from_xf86vm(clock, h, v) is None

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValueError):
            refresh_rate_from_dotclock(1000, 100, 100, dotclock_unit="MHz")
