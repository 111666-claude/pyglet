"""Window related exceptions.

Kept in a dependency-free module so that state helpers and tests can import
the exception types without pulling in the full window implementation.
"""


class WindowException(Exception):
    """The root exception for all window-related errors."""


class WindowStateException(WindowException):
    """Raised when a requested window state or state combination is unsupported.

    Examples include enabling per-pixel transparency on a window whose
    framebuffer has no alpha channel, or enabling mouse click-through on a
    platform session that does not provide the required native extension.
    """


__all__ = ["WindowException", "WindowStateException"]
