"""Repeatable acceptance harness for independent transparency and click-through.

This window exercises the two independently switchable aspects:

* **transparent rendering**  (per-pixel alpha framebuffer + compositor state)
* **mouse pass-through**      (a.k.a. click-through / ``WS_EX_TRANSPARENT`` /
  ``setIgnoresMouseEvents_`` / X11 SHAPE input region)

plus the related topmost state.

Run interactively::

    python examples/window/state_acceptance.py

Hotkeys (while the window can receive keyboard input):

* ``T``  toggle transparent rendering
* ``M``  toggle mouse pass-through
* ``A``  toggle always-on-top
* ``F``  toggle fullscreen (state must survive the switch)
* ``Q`` / ``Esc`` quit

Because a click-through window cannot be clicked/focused normally, enabling
pass-through with ``M`` automatically reverts it after 6 seconds, unless you
pass ``--sticky``.

For a hands-off, repeatable platform check use demo mode::

    python examples/window/state_acceptance.py --demo

which cycles through opaque -> transparent/interactive ->
transparent/click-through -> ordinary window and prints what to verify in
each phase.  See ``doc/programming_guide/window_transparency.rst`` for the
per-platform acceptance checklist.
"""
from __future__ import annotations

import random
import sys

import pyglet
from pyglet import shapes
from pyglet.text import Label

DEMO_PHASES = [
    {
        "name": "1/4 OPAQUE + INTERACTIVE",
        "transparent": False, "passthrough": False, "topmost": True,
        "check": "Window background is opaque; hovering/keys work.",
    },
    {
        "name": "2/4 TRANSPARENT + INTERACTIVE",
        "transparent": True, "passthrough": False, "topmost": True,
        "check": "Desktop shows through clear corners; window still receives clicks.",
    },
    {
        "name": "3/4 TRANSPARENT + CLICK-THROUGH",
        "transparent": True, "passthrough": True, "topmost": True,
        "check": "Whole window is click-through; clicks reach the app below.",
    },
    {
        "name": "4/4 ORDINARY WINDOW (all off)",
        "transparent": False, "passthrough": False, "topmost": False,
        "check": "Fully normal window: opaque, input, normal stacking.",
    },
]

demo_mode = "--demo" in sys.argv
sticky = "--sticky" in sys.argv

window = pyglet.window.Window(
    720, 460,
    caption="Transparency / Click-through acceptance",
    transparent=True,
    mouse_passthrough=False,
    topmost=True,
)

batch = pyglet.graphics.Batch()
# A translucent panel and an opaque-ish circle make alpha blending obvious.
panel = shapes.Rectangle(40, 120, 640, 260, color=(30, 60, 90, 180), batch=batch)
circle = shapes.Circle(360, 250, 80, color=(255, 200, 40, 255), batch=batch)
ring = shapes.Arc(360, 250, 100, color=(255, 255, 255, 220), thickness=4, batch=batch)

status = Label(
    "", x=360, y=420, anchor_x="center", font_size=16,
    color=(255, 255, 255, 255), batch=batch,
)
hint = Label(
    "T transparent | M pass-through (auto-revert 6s) | A topmost | F fullscreen | Q quit",
    x=360, y=30, anchor_x="center", font_size=11,
    color=(230, 230, 230, 255), batch=batch,
)
check_label = Label(
    "", x=360, y=392, anchor_x="center", font_size=12,
    color=(180, 230, 255, 255), multiline=True, width=660, batch=batch,
)

_revert_event = None


def refresh_labels(check: str = "") -> None:
    status.text = (
        f"transparent={window.transparent}  "
        f"mouse_passthrough={window.mouse_passthrough}  "
        f"topmost={window.topmost}  fullscreen={window.fullscreen}"
    )
    check_label.text = check


def set_passthrough(enabled: bool, auto_revert: bool = False) -> None:
    global _revert_event
    window.set_mouse_passthrough(enabled)
    if _revert_event is not None:
        pyglet.clock.unschedule(_revert_event)
        _revert_event = None
    if enabled and auto_revert and not sticky:
        def revert(dt):
            window.set_mouse_passthrough(False)
            refresh_labels("Pass-through auto-reverted; window is interactive again.")
        _revert_event = revert
        pyglet.clock.schedule_once(revert, 6.0)
    refresh_labels()


@window.event
def on_draw():
    # Clear with alpha 0 so untouched pixels are fully transparent.
    pyglet.gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    window.clear()
    batch.draw()


@window.event
def on_mouse_press(x, y, button, modifiers):
    # Visible feedback that input is arriving.
    circle.color = (
        random.randint(80, 255),
        random.randint(80, 255),
        40, 255,
    )
    refresh_labels("Mouse press received by pyglet window.")


@window.event
def on_key_press(symbol, modifiers):
    from pyglet.window import key

    if symbol in (key.Q, key.ESCAPE):
        window.close()
    elif symbol == key.T:
        window.set_transparent(not window.transparent)
        refresh_labels()
    elif symbol == key.M:
        set_passthrough(not window.mouse_passthrough, auto_revert=True)
    elif symbol == key.A:
        window.set_topmost(not window.topmost)
        refresh_labels()
    elif symbol == key.F:
        window.set_fullscreen(not window.fullscreen)


# ---------------------------------------------------------------------- #
# Demo mode: scripted, repeatable cycle
# ---------------------------------------------------------------------- #
if demo_mode:
    phase_index = 0

    def apply_phase(index):
        phase = DEMO_PHASES[index]
        window.set_transparent(phase["transparent"])
        window.set_mouse_passthrough(phase["passthrough"])
        window.set_topmost(phase["topmost"])
        refresh_labels(phase["check"])
        print(f"[acceptance] {phase['name']}: {phase['check']}")

    def cycle(dt):
        global phase_index
        apply_phase(phase_index)
        phase_index = (phase_index + 1) % len(DEMO_PHASES)

    # First phase immediately, then rotate every 5 seconds.
    apply_phase(0)
    phase_index = 1
    pyglet.clock.schedule_interval(cycle, 5.0)
    print("[acceptance] demo mode: cycling phases every 5s. Close the window to stop.")
else:
    refresh_labels("Ready. Press M to test click-through (auto-reverts in 6s).")


pyglet.app.run()
