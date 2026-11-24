"""Tests for the Linux evdev backend's SYN_DROPPED recovery.

The kernel emits an ``EV_SYN/SYN_DROPPED`` marker when its event ring buffer
overran: every event between the marker and the following ``SYN_REPORT`` is
stale and must be discarded, after which the authoritative device state has to
be re-read through ``EVIOCGKEY`` / ``EVIOCGABS``. These tests drive a fully
simulated device (no ``/dev/input`` node or hardware required) to verify that
contract, including repeated drops, disconnects, short reads, unsupported
ioctls and the public input API built on top of the controls.
"""
import errno

import ctypes
import warnings

import pytest

from pyglet.input.base import Button, AbsoluteAxis, RelativeAxis, Joystick, Controller
from pyglet.input.controller import Relation
from tests.annotations import require_platform, Platform

pytestmark = require_platform(Platform.LINUX)

from pyglet.input.linux import evdev
from pyglet.input.linux.evdev import (
    InputEvent, Timeval, InputABSInfo,
)
from pyglet.input.linux.evdev_constants import (
    EV_SYN, EV_KEY, EV_REL, EV_ABS, EV_MSC,
    SYN_REPORT, SYN_DROPPED,
    BTN_SOUTH, BTN_EAST, ABS_X, ABS_HAT0X, ABS_HAT0Y, REL_X, KEY_MAX,
)


def make_event(ev_type, code, value):
    return InputEvent(Timeval(0, 0), ev_type, code, value)


def make_syn(code=SYN_REPORT):
    return make_event(EV_SYN, code, 0)


class FakeEvdevDevice(evdev.EvdevDevice):
    """An EvdevDevice wired to scripted reads and scripted ioctl state."""

    def __init__(self, controls_spec, monkeypatch):
        # Bypass EvdevDevice.__init__, which would open a real /dev/input node.
        self.name = 'Fake evdev device'
        self._filename = '/dev/input/eventFAKE'
        self._fileno = 42
        self._event_size = ctypes.sizeof(InputEvent)
        self._event_buffer = (InputEvent * 64)()
        self._resync_pending = False
        self.closed = False
        self._read_batches = []
        self._read_errors = []

        self.controls = []
        self.control_map = {}
        # "kernel" state used by the simulated EVIOCG* ioctls.
        self.key_state = {}
        self.abs_state = {}
        self.abs_should_fail = {}      # code -> errno for EVIOCGABS
        self.key_should_fail = None    # errno for EVIOCGKEY

        for ev_type, code, name, raw_name in controls_spec:
            if ev_type == EV_KEY:
                control = Button(name, raw_name)
                self.key_state[code] = 0
            elif ev_type == EV_ABS:
                control = AbsoluteAxis(name, -32768, 32767, raw_name)
                self.abs_state[code] = 0
            else:
                control = RelativeAxis(name, raw_name)
            control.event_type = ev_type
            control.event_code = code
            control._value = 0
            self.control_map[(ev_type, code)] = control
            self.controls.append(control)

        monkeypatch.setattr(evdev, '_readv', self._fake_readv)
        monkeypatch.setattr(evdev, 'EVIOCGKEY', self._fake_gkey)
        monkeypatch.setattr(evdev, 'EVIOCGABS', self._fake_gabs)

    # Scripted transport -------------------------------------------------

    def queue_batch(self, events):
        self._read_batches.append(list(events))

    def queue_read_error(self, error):
        self._read_errors.append(error)

    def _fake_readv(self, fileno, buffer):
        if self._read_errors:
            raise self._read_errors.pop(0)
        if not self._read_batches:
            raise OSError(errno.EAGAIN, 'resource temporarily unavailable')
        batch = self._read_batches.pop(0)
        for i, event in enumerate(batch):
            buffer[i] = event
        return len(batch) * self._event_size

    def close(self):
        # Avoid the real close(): no platform event loop or open fd exists.
        self.closed = True
        self._fileno = None

    # Scripted ioctls -----------------------------------------------------

    def _fake_gkey(self, fileno):
        if self.key_should_fail is not None:
            err = self.key_should_fail
            self.key_should_fail = None
            raise OSError(err, 'simulated ioctl failure')
        buffer = (ctypes.c_byte * (KEY_MAX // 8 + 1))()
        for code, pressed in self.key_state.items():
            if pressed:
                buffer[code // 8] |= 1 << (code % 8)
        return buffer

    def _fake_gabs(self, fileno, code):
        if code in self.abs_should_fail:
            raise OSError(self.abs_should_fail[code], 'simulated ioctl failure')
        if code not in self.abs_state:
            raise OSError(errno.EINVAL, 'invalid argument')
        return InputABSInfo(self.abs_state[code], -32768, 32767, 0, 0)


@pytest.fixture
def device(monkeypatch):
    spec = [
        (EV_KEY, BTN_SOUTH, 'a', 'BTN_SOUTH'),
        (EV_KEY, BTN_EAST, 'b', 'BTN_EAST'),
        (EV_ABS, ABS_X, 'x', 'ABS_X'),
        (EV_ABS, ABS_HAT0X, 'hat_x', 'ABS_HAT0X'),
        (EV_ABS, ABS_HAT0Y, 'hat_y', 'ABS_HAT0Y'),
        (EV_REL, REL_X, 'x', 'REL_X'),
    ]
    return FakeEvdevDevice(spec, monkeypatch)


# ---------------------------------------------------------------------------
# Normal operation (regression guard for the pre-existing behaviour)
# ---------------------------------------------------------------------------

def test_normal_events_are_applied_in_order(device):
    button = device.control_map[(EV_KEY, BTN_SOUTH)]
    axis = device.control_map[(EV_ABS, ABS_X)]
    changes = []
    button.push_handlers(on_change=lambda value: changes.append(('btn', value)))
    axis.push_handlers(on_change=lambda value: changes.append(('abs', value)))

    device.queue_batch([
        make_event(EV_KEY, BTN_SOUTH, 1),
        make_event(EV_ABS, ABS_X, 100),
        make_syn(),
        make_event(EV_KEY, BTN_SOUTH, 0),
        make_syn(),
    ])
    device.select()

    assert button.value is False
    assert axis.value == 100
    assert changes == [('btn', True), ('abs', 100), ('btn', False)]
    assert device.closed is False


def test_unknown_event_codes_are_ignored(device):
    device.queue_batch([make_event(EV_MSC, 0, 5), make_syn()])
    device.select()  # must not raise KeyError
    assert device.closed is False


# ---------------------------------------------------------------------------
# SYN_DROPPED recovery
# ---------------------------------------------------------------------------

def test_syn_dropped_discards_stale_events_and_resyncs_state(device):
    button_a = device.control_map[(EV_KEY, BTN_SOUTH)]
    button_b = device.control_map[(EV_KEY, BTN_EAST)]
    axis_x = device.control_map[(EV_ABS, ABS_X)]
    hat_x = device.control_map[(EV_ABS, ABS_HAT0X)]

    # Starting from a known, pressed state.
    button_a._value = 1
    device.key_state[BTN_SOUTH] = 1

    # Simulated kernel truth at the sync boundary: A released, B pressed,
    # the axis and the hat moved.
    device.key_state[BTN_SOUTH] = 0
    device.key_state[BTN_EAST] = 1
    device.abs_state[ABS_X] = 512
    device.abs_state[ABS_HAT0X] = -1
    device.abs_state[ABS_HAT0Y] = 1

    press_events = []
    release_events = []
    button_a.push_handlers(on_press=lambda: press_events.append('a-press'),
                           on_release=lambda: release_events.append('a-release'))
    button_b.push_handlers(on_press=lambda: press_events.append('b-press'),
                           on_release=lambda: release_events.append('b-release'))

    device.queue_batch([
        # Stale frame from inside the dropped interval - must be ignored.
        make_event(EV_KEY, BTN_SOUTH, 1),
        make_event(EV_ABS, ABS_X, -9000),
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_event(EV_KEY, BTN_EAST, 0),
        make_event(EV_ABS, ABS_X, -9001),
        make_syn(),
    ])
    device.select()

    assert button_a.value is False          # matches kernel, not stale event
    assert button_b.value is True
    assert axis_x.value == 512              # from EVIOCGABS, not -9001
    assert hat_x.value == -1
    assert device.control_map[(EV_ABS, ABS_HAT0Y)].value == 1
    assert press_events == ['b-press']      # only real transitions
    assert release_events == ['a-release']
    assert device._resync_pending is False


def test_legacy_blind_loop_diverges_while_recovery_does_not(device):
    """Regression pin for the original bug: blindly applying the events of a
    dropped frame leaves the controls permanently out of sync with the kernel.
    """
    button = device.control_map[(EV_KEY, BTN_SOUTH)]
    axis = device.control_map[(EV_ABS, ABS_X)]
    button._value = 1                       # physically held before the drop
    device.key_state[BTN_SOUTH] = 1         # kernel truth: still held
    device.abs_state[ABS_X] = 512

    stale = [
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_event(EV_KEY, BTN_SOUTH, 0),   # stale release
        make_event(EV_ABS, ABS_X, -9001),   # stale axis position
        make_syn(),
    ]

    # Reproduction of the pre-fix select() loop on the same event stream.
    legacy_button = Button('a', 'BTN_SOUTH')
    legacy_button.event_type, legacy_button.event_code = EV_KEY, BTN_SOUTH
    legacy_button._value = 1
    legacy_axis = AbsoluteAxis('x', -32768, 32767, 'ABS_X')
    legacy_axis.event_type, legacy_axis.event_code = EV_ABS, ABS_X
    legacy_axis._value = 0
    legacy_map = {(EV_KEY, BTN_SOUTH): legacy_button, (EV_ABS, ABS_X): legacy_axis}
    for event in stale:
        try:
            legacy_map[(event.type, event.code)].value = event.value
        except KeyError:
            pass
    assert legacy_button.value is False     # diverged: kernel says held
    assert legacy_axis.value == -9001       # diverged: kernel says 512

    # The fixed backend instead re-reads authoritative state at the boundary.
    device.queue_batch(stale)
    device.select()
    assert button.value is True
    assert axis.value == 512


def test_recovery_is_not_reported_as_repeated_keypress(device):
    # A button that is already pressed must not re-fire on_press just because
    # the state was re-read during recovery.
    button = device.control_map[(EV_KEY, BTN_SOUTH)]
    button._value = 1
    device.key_state[BTN_SOUTH] = 1

    presses = []
    button.push_handlers(on_press=lambda: presses.append(1))

    device.queue_batch([
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_event(EV_KEY, BTN_SOUTH, 0),   # stale release, discarded
        make_syn(),
    ])
    device.select()

    assert button.value is True
    assert presses == []
    assert device._resync_pending is False


def test_first_observed_button_state_does_not_emit_release(device):
    # Buttons with no prior event hold an internal value of None; adopting the
    # kernel's "not pressed" state during recovery must not fire on_release.
    button = device.control_map[(EV_KEY, BTN_EAST)]
    button._value = None
    releases = []
    button.push_handlers(on_release=lambda: releases.append(1))
    device.key_state[BTN_EAST] = 0

    device._set_control_value(button, False)
    assert button._value == 0
    assert releases == []

    # And a pressed first state is adopted silently as well.
    button._value = None
    presses = []
    button.push_handlers(on_press=lambda: presses.append(1))
    device._set_control_value(button, True)
    assert button.value is True
    assert presses == []


def test_repeated_syn_dropped_recovers_each_time(device):
    button = device.control_map[(EV_KEY, BTN_SOUTH)]
    axis = device.control_map[(EV_ABS, ABS_X)]

    # First drop: button ends up pressed.
    device.key_state[BTN_SOUTH] = 1
    device.abs_state[ABS_X] = 10
    device.queue_batch([
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_event(EV_ABS, ABS_X, -1),
        make_syn(),
    ])
    device.select()
    assert button.value is True
    assert axis.value == 10

    # Second drop, in a later read: button released again, axis moves.
    device.key_state[BTN_SOUTH] = 0
    device.abs_state[ABS_X] = 20
    device.queue_batch([
        make_event(EV_ABS, ABS_X, -2),
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_syn(),
    ])
    device.select()
    assert button.value is False
    assert axis.value == 20
    assert device._resync_pending is False


def test_resync_pending_survives_across_reads(device):
    # If the 64-event buffer fills mid-frame, SYN_REPORT arrives on the next
    # read; stale events on that second read must still be discarded.
    axis = device.control_map[(EV_ABS, ABS_X)]
    device.abs_state[ABS_X] = 777

    device.queue_batch([make_event(EV_SYN, SYN_DROPPED, 0)])
    device.select()
    assert device._resync_pending is True

    device.queue_batch([
        make_event(EV_ABS, ABS_X, -1234),   # still inside the dropped gap
        make_syn(),
    ])
    device.select()
    assert axis.value == 777
    assert device._resync_pending is False

    # Normal events flow again afterwards.
    device.queue_batch([make_event(EV_ABS, ABS_X, 42), make_syn()])
    device.select()
    assert axis.value == 42


def test_relative_axes_are_left_untouched_during_resync(device):
    rel = device.control_map[(EV_REL, REL_X)]
    rel._value = 5
    device.queue_batch([
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_event(EV_REL, REL_X, 999),    # stale, discarded
        make_syn(),
    ])
    device.select()
    assert rel.value == 5                   # no queryable state, no synthetic move


# ---------------------------------------------------------------------------
# Disconnect / transport error handling
# ---------------------------------------------------------------------------

def test_read_error_eagain_is_retried_not_fatal(device):
    device.queue_read_error(OSError(errno.EAGAIN, 'try again'))
    device.select()
    assert device.closed is False


def test_fatal_read_error_closes_device(device):
    device.queue_read_error(OSError(errno.EIO, 'input/output error'))
    device.select()
    assert device.closed is True
    assert device._fileno is None


def test_zero_length_read_closes_device(device):
    device.queue_batch([])
    device.select()
    assert device.closed is True


def test_disconnect_during_resync_closes_device(device):
    device.key_should_fail = errno.ENODEV
    device.queue_batch([
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_syn(),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        device.select()
    assert device.closed is True


def test_disconnect_during_abs_resync_closes_device(device):
    device.abs_should_fail[ABS_X] = errno.EIO
    device.queue_batch([
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_syn(),
    ])
    device.select()
    assert device.closed is True


# ---------------------------------------------------------------------------
# Short reads and unsupported ioctls
# ---------------------------------------------------------------------------

def test_short_read_only_processes_complete_events(device, monkeypatch):
    button = device.control_map[(EV_KEY, BTN_SOUTH)]
    axis = device.control_map[(EV_ABS, ABS_X)]

    def short_read(fileno, buffer):
        buffer[0] = make_event(EV_KEY, BTN_SOUTH, 1)
        buffer[1] = make_event(EV_ABS, ABS_X, 33)
        # Two full events plus half of another: the partial record must be
        # ignored rather than interpreted as garbage.
        return device._event_size * 2 + device._event_size // 2

    monkeypatch.setattr(evdev, '_readv', short_read)
    device.select()
    assert button.value is True
    assert axis.value == 33
    assert device.closed is False


def test_unsupported_key_ioctl_keeps_device_alive(device):
    device.key_should_fail = errno.ENOTTY
    axis = device.control_map[(EV_ABS, ABS_X)]
    device.abs_state[ABS_X] = 9
    with pytest.warns(UserWarning, match='resync key state'):
        device.queue_batch([
            make_event(EV_SYN, SYN_DROPPED, 0),
            make_syn(),
        ])
        device.select()
    assert device.closed is False
    assert axis.value == 9                 # ABS recovery still happened


def test_unsupported_abs_ioctl_keeps_other_state_working(device):
    device.abs_should_fail[ABS_X] = errno.ENOTTY
    button = device.control_map[(EV_KEY, BTN_SOUTH)]
    other_axis = device.control_map[(EV_ABS, ABS_HAT0X)]
    device.key_state[BTN_SOUTH] = 1
    device.abs_state[ABS_HAT0X] = 1
    with pytest.warns(UserWarning, match='ABS_X'):
        device.queue_batch([
            make_event(EV_SYN, SYN_DROPPED, 0),
            make_syn(),
        ])
        device.select()
    assert device.closed is False
    assert button.value is True
    assert other_axis.value == 1
    assert device.control_map[(EV_ABS, ABS_X)].value == 0


# ---------------------------------------------------------------------------
# Public input API (Controller / Joystick) on top of recovery
# ---------------------------------------------------------------------------

def test_controller_state_tracks_kernel_after_drop(device):
    mapping = {
        'guid': '00000000000000000000000000000000',
        'name': 'Fake evdev device',
        'a': Relation('button', 0),
        'b': Relation('button', 1),
        'leftx': Relation('axis', 0),
    }
    controller = Controller(device, mapping)

    device.key_state[BTN_SOUTH] = 1
    device.abs_state[ABS_X] = 32767
    device.queue_batch([
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_event(EV_KEY, BTN_SOUTH, 0),   # stale
        make_event(EV_ABS, ABS_X, -32768),  # stale
        make_syn(),
    ])
    device.select()

    assert controller.a is True
    assert controller.leftx == pytest.approx(1.0, abs=1e-6)

    button_presses = []
    controller.push_handlers(on_button_press=lambda c, name: button_presses.append(name))
    # A subsequent genuine release after recovery must still dispatch.
    device.key_state[BTN_SOUTH] = 0
    device.queue_batch([make_event(EV_KEY, BTN_SOUTH, 0), make_syn()])
    device.select()
    assert controller.a is False
    assert button_presses == []            # release, not a press


def test_joystick_buttons_and_axes_recover(device):
    joystick = Joystick(device)
    device.key_state[BTN_SOUTH] = 1
    device.abs_state[ABS_X] = 32767
    device.queue_batch([
        make_event(EV_SYN, SYN_DROPPED, 0),
        make_syn(),
    ])
    device.select()

    assert joystick.buttons[0] is True
    assert joystick.x == pytest.approx(1.0, abs=1e-6)
