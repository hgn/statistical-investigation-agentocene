"""Multi-line, in-place progress display for the gatherer.

TTY-aware per the house rules: uses carriage control (ANSI cursor-up + line
clear) only when stderr is a terminal; otherwise falls back to plain periodic
log lines so piped/CI output stays readable. Respects NO_COLOR.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO


def fmt_dur(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class ProgressPrinter:
    """Renders a block of status lines, overwriting the previous block in place
    on a TTY. On a non-TTY it emits the first line as a plain log entry."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._out = stream or sys.stderr
        self.tty = self._out.isatty()
        self._color = self.tty and os.environ.get("NO_COLOR") is None
        self._prev = 0

    def _dim(self, text: str) -> str:
        return f"\x1b[2m{text}\x1b[0m" if self._color else text

    def render(self, lines: list[str]) -> None:
        if not self.tty:
            if lines:
                self._out.write(lines[0].strip() + "\n")
                self._out.flush()
            return
        if self._prev:
            self._out.write(f"\x1b[{self._prev}A")  # move cursor up
        for ln in lines:
            self._out.write("\x1b[2K" + ln + "\n")  # clear line, write, newline
        self._prev = len(lines)
        self._out.flush()

    def finalize(self) -> None:
        """Leave the last rendered block in place and drop to a fresh line."""
        if self.tty and self._prev:
            self._out.write("\n")
            self._out.flush()
        self._prev = 0
