from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from pyglet.text import caret
from pyglet.text.document import UnformattedDocument
from pyglet.text.layout.base import _GlyphBox, _Line
from pyglet.text.layout.incremental import IncrementalTextLayout
from pyglet.window import key


class FakeGlyph:
    """Minimal glyph with a controllable, possibly fractional advance."""

    def __init__(self, advance, width=1):
        self.advance = advance
        self.width = width
        self.vertices = (0, 0, width, 1)
        self.tex_coords = (0, 0, 1, 0, 1, 1, 0, 1)


def make_glyph_box(advances, kerns=None):
    if kerns is None:
        kerns = [0] * len(advances)
    glyphs = [(kern, FakeGlyph(advance)) for kern, advance in zip(kerns, advances)]
    font = SimpleNamespace(ascent=10, descent=-2)
    raw_advance = sum(advance + kern for advance, kern in zip(advances, kerns))
    return _GlyphBox(object(), font, glyphs, raw_advance)


def make_line_layout(text, advances, kerns=None):
    """Build only the coordinate-related state of an incremental layout."""
    document = UnformattedDocument(text)
    layout = object.__new__(IncrementalTextLayout)
    layout._document = document

    line = _Line(0)
    line.x = 0
    line.y = 0
    line.add_box(make_glyph_box(advances, kerns))
    layout.lines = [line]

    layout._translate_x = 0
    layout._translate_y = 0
    layout._x = 0
    layout._y = 0
    layout._z = 0
    layout._anchor_left = 0
    layout._anchor_bottom = 0

    # Caret integration points.  Coordinate methods remain the real methods.
    layout.push_handlers = Mock()
    layout._batch = Mock()
    layout.foreground_decoration_group = Mock()
    layout.foreground_decoration_group.program.vertex_list.return_value = MagicMock()
    layout.get_line_from_point = Mock(return_value=0)
    layout.ensure_line_visible = Mock()
    layout.ensure_x_visible = Mock()
    layout._get_content_height = Mock(return_value=12)
    layout._selection_start = 0
    layout._selection_end = 0

    def set_selection(start, end):
        layout._selection_start = start
        layout._selection_end = end

    layout.set_selection = Mock(side_effect=set_selection)
    return layout


@pytest.fixture(autouse=True)
def disable_caret_blink_clock(monkeypatch):
    # PERIOD == 0 keeps the real clock proxy untouched while preventing a
    # recurring blink timer from outliving these tests.
    monkeypatch.setattr(caret.Caret, "PERIOD", 0)


@pytest.fixture
def caret_layout():
    advances = [10] * len("alpha beta")
    layout = make_line_layout("alpha beta", advances)
    text_caret = caret.Caret(layout)
    return text_caret, layout


def double_click(text_caret, x, y=0):
    text_caret.on_mouse_press(x, y, 1, 0)
    text_caret.on_mouse_press(x, y, 1, 0)


@pytest.mark.parametrize(
    ("text", "position", "expected"),
    [
        # Word interiors, starts, and ends all select only the word.
        ("alpha beta", 0, (0, 5)),
        ("alpha beta", 2, (0, 5)),
        ("alpha beta", 5, (0, 5)),
        ("alpha beta", 6, (6, 10)),
        ("This paragraph has text", 8, (5, 14)),
        ("This paragraph has text", 14, (5, 14)),
        # Repeated whitespace is one selectable run.
        ("alpha  beta", 5, (0, 5)),
        ("alpha  beta", 6, (5, 7)),
        ("alpha  beta", 7, (7, 11)),
        # Repeated punctuation is also one run.
        ("foo...bar", 3, (0, 3)),
        ("foo...bar", 4, (3, 6)),
        ("foo...bar", 6, (6, 9)),
        # Newlines do not become part of a word selection, and horizontal
        # whitespace does not cross a line boundary.
        ("alpha\nbeta", 5, (0, 5)),
        ("alpha\nbeta", 6, (6, 10)),
        ("alpha\nbeta", 8, (6, 10)),
        ("a  \nb", 2, (1, 3)),
        ("a  \nb", 3, (1, 3)),
        ("a\n\nb", 2, (1, 3)),
        # Degenerate input stays at a valid empty range.
        ("", 0, (0, 0)),
    ],
)
def test_word_bounds_are_consistent(text, position, expected):
    assert caret.Caret._get_word_bounds(text, position) == expected


def test_word_bounds_clamp_positions_outside_document():
    assert caret.Caret._get_word_bounds("abc", -10) == (0, 3)
    assert caret.Caret._get_word_bounds("abc", 10) == (0, 3)
    assert caret.Caret._get_word_bounds("", 4) == (0, 0)


def test_double_click_selects_first_word_without_following_space(caret_layout):
    text_caret, layout = caret_layout

    double_click(text_caret, 24)

    assert text_caret.mark == 0
    assert text_caret.position == 5
    assert layout.document.text[text_caret.mark:text_caret.position] == "alpha"
    assert (layout._selection_start, layout._selection_end) == (0, 5)


def test_double_click_at_word_end_does_not_select_space(caret_layout):
    text_caret, layout = caret_layout

    # x=49 maps to caret position 5, immediately after "alpha".
    assert layout.get_position_on_line(0, 49) == 5
    double_click(text_caret, 49)

    assert layout.document.text[text_caret.mark:text_caret.position] == "alpha"
    assert (layout._selection_start, layout._selection_end) == (0, 5)


def test_double_click_event_with_fractional_widths_matches_caret_pixels():
    layout = make_line_layout("alpha beta", [10.4] * len("alpha beta"))
    text_caret = caret.Caret(layout)

    double_click(text_caret, 24)

    assert layout.document.text[text_caret.mark:text_caret.position] == "alpha"
    assigned = text_caret._list.position.__setitem__.call_args.args[1]
    assert assigned[0] == 50
    assert assigned[3] == 50
    assert layout.get_point_from_position(text_caret.position)[0] == 50


def test_drag_and_keyboard_selection_remain_consistent(caret_layout):
    text_caret, layout = caret_layout

    # Press in "alpha", then drag two positions to the right.
    text_caret.on_mouse_press(24, 0, 1, 0)
    assert text_caret.position == 2
    assert text_caret.mark is None

    text_caret.on_mouse_drag(44, 0, 20, 0, 1, 0)
    assert (text_caret.mark, text_caret.position) == (2, 4)
    assert layout.document.text[2:4] == "ph"

    # Shift-right extends the active selection.
    text_caret.on_text_motion_select(key.MOTION_RIGHT)
    assert (text_caret.mark, text_caret.position) == (2, 5)
    assert layout._selection_end == 5

    # An unmodified left press/motion moves to the selection's start and
    # clears the selection, as before.
    text_caret.on_text_motion(key.MOTION_LEFT)
    assert text_caret.position == 2
    assert text_caret.mark is None
    assert (layout._selection_start, layout._selection_end) == (0, 0)


def test_keyboard_word_motions_keep_existing_behavior(caret_layout):
    text_caret, _layout = caret_layout

    text_caret.position = 0
    text_caret.on_text_motion(key.MOTION_NEXT_WORD)
    assert text_caret.position == 6

    text_caret.on_text_motion(key.MOTION_PREVIOUS_WORD)
    assert text_caret.position == 0


def test_typing_replaces_selection_and_backspace_still_edits(caret_layout):
    text_caret, layout = caret_layout

    text_caret.on_mouse_press(24, 0, 1, 0)
    text_caret.on_mouse_drag(44, 0, 20, 0, 1, 0)
    assert layout.document.text[2:4] == "ph"

    text_caret.on_text("X")
    assert layout.document.text == "alXa beta"
    assert text_caret.position == 3
    assert text_caret.mark is None

    text_caret.on_text_motion(key.MOTION_BACKSPACE)
    assert layout.document.text == "ala beta"
    assert text_caret.position == 2


def test_fractional_glyph_coordinates_round_trip_through_same_pixels():
    box = make_glyph_box([7.4, 8.4])

    # Each fractional advance is rounded once, as glyph placement does.
    assert box.advance == 15
    assert box.get_point_in_box(0) == 0
    assert box.get_point_in_box(1) == 7
    assert box.get_point_in_box(2) == 15

    # The first glyph's rounded midpoint is 3.5 and the second's is 11.
    assert box.get_position_in_box(3) == 0
    assert box.get_position_in_box(4) == 1
    assert box.get_position_in_box(10) == 1
    assert box.get_position_in_box(11) == 2


def test_layout_point_and_hit_test_use_rounded_glyph_advances():
    layout = make_line_layout("ab", [7.4, 8.4])

    assert layout.get_point_from_position(0)[0] == 0
    assert layout.get_point_from_position(1)[0] == 7
    assert layout.get_point_from_position(2)[0] == 15

    assert layout.get_position_on_line(0, 0) == 0
    assert layout.get_position_on_line(0, 3) == 0
    assert layout.get_position_on_line(0, 4) == 1
    assert layout.get_position_on_line(0, 10) == 1
    assert layout.get_position_on_line(0, 11) == 2
    assert layout.get_position_on_line(0, 15) == 2
    assert layout.get_position_on_line(0, 999) == 2


def test_caret_draws_at_rounded_layout_coordinate():
    layout = make_line_layout("ab", [7.4, 8.4])
    text_caret = caret.Caret(layout)

    text_caret.position = 1

    assert text_caret._position == 1
    assigned = text_caret._list.position.__setitem__.call_args.args[1]
    assert assigned[0] == 7
    assert assigned[3] == 7  # both ends of the caret line share x


def test_glyph_placement_vertices_match_rounded_caret_points():
    advances = [7.4, 8.4]
    box = make_glyph_box(advances)

    baseline_ranges = Mock()
    baseline_ranges.ranges.return_value = iter([(0, 2, 0)])
    color_ranges = Mock()
    color_ranges.ranges.return_value = iter([(0, 2, (255, 255, 255, 255))])
    decoration_ranges = Mock()
    decoration_ranges.ranges.return_value = iter(())
    context = SimpleNamespace(
        baseline_iter=baseline_ranges,
        colors_iter=color_ranges,
        decoration_iter=decoration_ranges,
        add_list=Mock(),
    )

    program = Mock()
    layout = SimpleNamespace(
        group_cache={},
        group_class=Mock(),
        group=Mock(),
        program=program,
        batch=Mock(),
        _parse_distance=Mock(return_value=0),
    )

    box.place(
        layout,
        0,
        0, 0, 0,
        0, 0,
        0,
        True,
        0, 0,
        context,
    )

    vertices = program.vertex_list_indexed.call_args.kwargs["position"][1]
    first_glyph_xs = [vertices[index] for index in (0, 3, 6, 9)]
    second_glyph_xs = [vertices[index] for index in (12, 15, 18, 21)]

    assert first_glyph_xs == [0, 1, 1, 0]
    assert second_glyph_xs == [7, 8, 8, 7]
    assert box.get_point_in_box(1) == second_glyph_xs[0]
