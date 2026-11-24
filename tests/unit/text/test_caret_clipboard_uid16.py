"""Tests for Caret clipboard interactions (copy / cut / paste).

These tests cover cross-line selections, the common desktop shortcuts on
both Control (Windows/Linux) and Command (macOS), empty selections, empty
clipboards, newline normalisation, unfocused carets and the "one physical key
press must not produce duplicate input" behaviour.

A real :class:`~pyglet.text.document.UnformattedDocument` backs a mocked
layout, so document/caret/selection state is asserted exactly while all GL
and clock access is stubbed.  The clipboard is a small in-memory object that
can be replaced with any compatible provider.
"""
from __future__ import annotations

from unittest.mock import Mock, NonCallableMock

import pytest

from pyglet.graphics.shader import ShaderProgram
from pyglet.graphics.vertexdomain import IndexedVertexList
from pyglet.text import caret as caret_module
from pyglet.text import layout as layout_module
from pyglet.text.caret import Caret
from pyglet.text.document import UnformattedDocument
from pyglet.window import key


class _SliceTupleList(list):
    """Mimics ctypes colour sequences: slices compare as tuples."""

    def __getitem__(self, item):
        if isinstance(item, slice):
            return tuple(super().__getitem__(item))
        return super().__getitem__(item)


class FakeClipboard:
    """In-memory replaceable clipboard speaking the Window protocol."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.get_count = 0
        self.set_count = 0

    def get_clipboard_text(self) -> str:
        self.get_count += 1
        return self.text

    def set_clipboard_text(self, text: str) -> None:
        self.set_count += 1
        self.text = text


class SimpleTextClipboard:
    """Clipboard double that only exposes a plain ``.text`` attribute."""

    def __init__(self, text: str = "") -> None:
        self.text = text


@pytest.fixture(autouse=True)
def _stub_clock():
    clock = Mock(spec=caret_module.clock)
    original = caret_module.clock
    caret_module.clock = clock
    yield clock
    caret_module.clock = original


def _make_caret(text: str = "one\ntwo\nthree",
                clipboard: object | None = None,
                *,
                use_window: bool = True,
                active: bool = True):
    document = UnformattedDocument(text)

    layout = NonCallableMock(spec=layout_module.TextLayout)
    layout.foreground_decoration_group = NonCallableMock()
    layout.attach_mock(Mock(), "push_handlers")
    layout.set_selection = Mock()
    layout.document = document

    program = NonCallableMock(spec=ShaderProgram)
    layout.foreground_decoration_group.attach_mock(program, "program")

    def _vertex_list(count, mode, batch=None, group=None, colors=None, visible=None):
        vertex_list = NonCallableMock(spec=IndexedVertexList)
        vertex_list.colors = _SliceTupleList(colors[1])
        vertex_list.visible = (1, 1)
        return vertex_list

    program.vertex_list = _vertex_list

    window = FakeClipboard() if use_window and clipboard is None else clipboard
    caret = Caret(layout, window=window if use_window else None,
                  clipboard=None if use_window else clipboard,
                  active=active)
    # Geometry is mocked; state transitions never need real line metrics.
    caret._update = Mock()  # noqa: SLF001
    return document, layout, window, caret


def _select(caret: Caret, start: int, end: int) -> None:
    caret._mark = start  # noqa: SLF001
    caret.position = end


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modifier", [key.MOD_CTRL, key.MOD_COMMAND, key.MOD_ACCEL])
def test_copy_shortcut_copies_cross_line_selection(modifier):
    document, _layout, clipboard, caret = _make_caret()
    _select(caret, 0, len("one\ntwo\n"))

    handled = caret.on_key_press(key.C, modifier)

    assert handled
    assert clipboard.text == "one\ntwo\n"
    # Copy never mutates the document.
    assert document.text == "one\ntwo\nthree"
    # Standard desktop behaviour keeps the selection and caret.
    assert caret.mark == 0
    assert caret.position == len("one\ntwo\n")
    assert clipboard.set_count == 1


def test_copy_direct_method_returns_text_and_preserves_document():
    document, _layout, clipboard, caret = _make_caret("abcDEF")
    _select(caret, 3, 6)

    result = caret.copy()

    assert result == "DEF"
    assert clipboard.text == "DEF"
    assert document.text == "abcDEF"


def test_copy_motion_interface_still_supported():
    document, _layout, clipboard, caret = _make_caret("abcDEF")
    _select(caret, 3, 6)

    assert caret.on_text_motion(key.MOTION_COPY)
    assert clipboard.text == "DEF"
    assert document.text == "abcDEF"


def test_copy_without_selection_is_noop_and_preserves_clipboard():
    document, _layout, clipboard, caret = _make_caret("abc", FakeClipboard("KEEP"))
    caret.position = 1

    assert caret.copy() is None
    assert clipboard.text == "KEEP"
    assert clipboard.set_count == 0
    assert document.text == "abc"

    # A collapsed mark == position selection is also empty.
    caret._mark = 1  # noqa: SLF001
    assert caret.copy() is None
    assert clipboard.set_count == 0


# ---------------------------------------------------------------------------
# Cut
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modifier", [key.MOD_CTRL, key.MOD_COMMAND])
def test_cut_shortcut_removes_only_selection(modifier):
    document, _layout, clipboard, caret = _make_caret("abcDEFghi")
    _select(caret, 3, 6)

    assert caret.on_key_press(key.X, modifier)
    assert clipboard.text == "DEF"
    assert document.text == "abcghi"
    # Caret lands at the deletion point with no lingering selection.
    assert caret.position == 3
    assert caret.mark is None


def test_cut_direct_method_returns_removed_text():
    document, _layout, clipboard, caret = _make_caret("abcDEF")
    _select(caret, 3, 6)

    assert caret.cut() == "DEF"
    assert document.text == "abc"
    assert caret.mark is None


def test_cut_with_empty_selection_deletes_nothing():
    document, _layout, clipboard, caret = _make_caret("abc", FakeClipboard("KEEP"))
    caret.position = 1

    assert caret.cut() is None
    assert document.text == "abc"
    assert clipboard.text == "KEEP"
    assert clipboard.set_count == 0


# ---------------------------------------------------------------------------
# Paste
# ---------------------------------------------------------------------------

def test_paste_replaces_selection_and_resets_caret_state():
    document, _layout, clipboard, caret = _make_caret("abcDEFghi", FakeClipboard("XYZ"))
    _select(caret, 3, 6)

    caret.on_key_press(key.V, key.MOD_CTRL)

    assert document.text == "abcXYZghi"
    assert caret.position == 6
    assert caret.mark is None


def test_paste_at_collapsed_caret_inserts_in_place():
    document, _layout, clipboard, caret = _make_caret("ab", FakeClipboard("Z"))
    caret.position = 1

    caret.paste()

    assert document.text == "aZb"
    assert caret.position == 2
    assert caret.mark is None


def test_paste_empty_clipboard_is_noop():
    document, _layout, clipboard, caret = _make_caret("abc", FakeClipboard(""))
    caret.position = 1

    result = caret.paste()

    assert result == ""
    assert document.text == "abc"
    assert caret.position == 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("line1\nline2", "line1\nline2"),
        ("line1\rline2", "line1\nline2"),
        ("line1\r\nline2", "line1\nline2"),
        ("a\r\nb\rc\nd", "a\nb\nc\nd"),
    ],
)
def test_paste_normalises_newline_text(raw, expected):
    document, _layout, _clipboard, caret = _make_caret("", FakeClipboard(raw))

    caret.paste()

    assert document.text == expected
    assert caret.position == len(expected)


def test_paste_multiline_then_cross_line_selection_roundtrip():
    document, _layout, clipboard, caret = _make_caret("", FakeClipboard("x\ny\nz"))
    caret.paste()
    assert document.text == "x\ny\nz"

    caret.select_all()
    assert (caret.mark, caret.position) == (0, 5)
    caret.copy()
    assert clipboard.text == "x\ny\nz"


def test_paste_accepts_explicit_text_argument():
    document, _layout, _clipboard, caret = _make_caret("ab")
    caret.position = 1

    caret.paste("Q")

    assert document.text == "aQb"
    assert caret.position == 2


# ---------------------------------------------------------------------------
# No duplicate input per physical key press
# ---------------------------------------------------------------------------

def test_ctrl_c_followed_by_motion_copies_once():
    document, _layout, clipboard, caret = _make_caret("abc")
    caret.select_all()

    caret.on_key_press(key.C, key.MOD_CTRL)
    caret.on_text_motion(key.MOTION_COPY)  # paired platform event

    assert clipboard.text == "abc"
    assert clipboard.set_count == 1
    assert document.text == "abc"


def test_ctrl_x_followed_by_motion_cuts_once():
    document, _layout, clipboard, caret = _make_caret("abcDEF")
    _select(caret, 3, 6)

    caret.on_key_press(key.X, key.MOD_CTRL)
    caret.on_text_motion(key.MOTION_CUT)

    assert document.text == "abc"
    assert clipboard.set_count == 1


def test_ctrl_v_followed_by_motion_pastes_once():
    document, _layout, clipboard, caret = _make_caret("ab", FakeClipboard("Z"))
    caret.position = 1

    caret.on_key_press(key.V, key.MOD_CTRL)
    caret.on_text_motion(key.MOTION_PASTE)

    assert document.text == "aZb"
    assert clipboard.get_count == 1


def test_ctrl_v_followed_by_control_text_does_not_duplicate():
    document, _layout, _clipboard, caret = _make_caret("ab", FakeClipboard("Z"))
    caret.position = 1

    caret.on_key_press(key.V, key.MOD_CTRL)
    # Some platform/keyboard layouts also deliver the raw control character.
    caret.on_text("\x16")

    assert document.text == "aZb"


@pytest.mark.parametrize("control_char", ["\x01", "\x03", "\x16", "\x18", "\x00", "\x7f"])
def test_on_text_ignores_shortcut_control_characters(control_char):
    document, _layout, _clipboard, caret = _make_caret("ab")
    caret.position = len("ab")

    caret.on_text(control_char)

    assert document.text == "ab"


def test_on_text_still_inserts_tab_newline_and_printables():
    document, _layout, _clipboard, caret = _make_caret("")
    caret.on_text("a\tb\nc")
    assert document.text == "a\tb\nc"


# ---------------------------------------------------------------------------
# Unfocused / inactive controls
# ---------------------------------------------------------------------------

def test_deactivated_caret_ignores_shortcuts():
    document, _layout, clipboard, caret = _make_caret("abc", FakeClipboard("ZZ"))
    caret.select_all()

    caret.on_deactivate()

    assert caret.on_key_press(key.C, key.MOD_CTRL) is None
    assert clipboard.text == "ZZ"
    assert clipboard.set_count == 0

    assert caret.on_text_motion(key.MOTION_CUT) is None
    assert document.text == "abc"


def test_focused_false_constructor_ignores_shortcuts():
    document, _layout, clipboard, caret = _make_caret(
        "abc", FakeClipboard("ZZ"), active=False,
    )
    caret._mark = 0  # noqa: SLF001
    caret._position = 3

    assert caret.focused is False
    assert caret.active is False
    assert caret.on_key_press(key.X, key.MOD_CTRL) is None
    assert document.text == "abc"
    assert clipboard.set_count == 0


def test_refocusing_reenables_shortcuts():
    document, _layout, clipboard, caret = _make_caret("abc")
    caret.select_all()
    caret.active = False
    assert caret.on_key_press(key.C, key.MOD_CTRL) is None

    caret.on_activate()
    assert caret.on_key_press(key.C, key.MOD_CTRL)
    assert clipboard.text == "abc"


# ---------------------------------------------------------------------------
# Replaceable clipboard providers
# ---------------------------------------------------------------------------

def test_clipboard_keyword_with_plain_text_object():
    clipboard = SimpleTextClipboard("HELLO")
    document, _layout, _win, caret = _make_caret(
        "", clipboard=clipboard, use_window=False,
    )

    caret.paste()
    assert document.text == "HELLO"

    caret.select_all()
    caret.cut()
    assert clipboard.text == "HELLO"
    assert document.text == ""


def test_window_attribute_can_be_replaced_after_construction():
    document = UnformattedDocument("abc")
    layout = NonCallableMock(spec=layout_module.TextLayout)
    layout.foreground_decoration_group = NonCallableMock()
    layout.attach_mock(Mock(), "push_handlers")
    layout.set_selection = Mock()
    layout.document = document
    program = NonCallableMock(spec=ShaderProgram)
    layout.foreground_decoration_group.attach_mock(program, "program")

    def _vertex_list(count, mode, batch=None, group=None, colors=None, visible=None):
        vertex_list = NonCallableMock(spec=IndexedVertexList)
        vertex_list.colors = _SliceTupleList(colors[1])
        vertex_list.visible = (1, 1)
        return vertex_list

    program.vertex_list = _vertex_list

    caret = Caret(layout)
    caret._update = Mock()  # noqa: SLF001
    replacement = FakeClipboard("Q")
    caret.window = replacement

    assert caret.window is replacement
    caret.position = 1
    caret.paste()
    assert document.text == "aQbc"


def test_mock_window_clipboard_protocol():
    window = Mock()
    window.get_clipboard_text.return_value = "M"
    document, _layout, _win, caret = _make_caret("abc", window)
    caret.position = 0

    caret.paste()
    assert document.text == "Mabc"
    window.get_clipboard_text.assert_called()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def test_clipboard_events_are_dispatched_once():
    _document, _layout, _clipboard, caret = _make_caret("abcDEF")
    _select(caret, 3, 6)

    copy_handler = Mock()
    cut_handler = Mock()
    paste_handler = Mock()
    caret.push_handlers(
        on_clipboard_copy=copy_handler,
        on_clipboard_cut=cut_handler,
        on_clipboard_paste=paste_handler,
    )

    caret.copy()
    caret.cut()
    caret.paste("Z")

    copy_handler.assert_called_once_with("DEF")
    cut_handler.assert_called_once_with("DEF")
    paste_handler.assert_called_once_with("Z")


# ---------------------------------------------------------------------------
# Backwards compatible keyboard navigation
# ---------------------------------------------------------------------------

def test_plain_and_unrelated_modifier_keys_are_unhandled():
    _document, _layout, _clipboard, caret = _make_caret("abc")

    assert caret.on_key_press(key.C, 0) is None
    assert caret.on_key_press(key.Z, key.MOD_CTRL) is None
    assert caret.on_key_press(key.LEFT, key.MOD_SHIFT) is None


def test_arrow_navigation_still_moves_caret():
    document, _layout, _clipboard, caret = _make_caret("abc")
    caret.position = 2

    caret.on_text_motion(key.MOTION_LEFT)
    assert caret.position == 1

    caret.on_text_motion(key.MOTION_RIGHT)
    assert caret.position == 2


def test_select_all_shortcut_selects_entire_multiline_document():
    document, _layout, _clipboard, caret = _make_caret("a\nbb\nccc")

    assert caret.on_key_press(key.A, key.MOD_COMMAND)
    assert caret.mark == 0
    assert caret.position == len("a\nbb\nccc") == 8
