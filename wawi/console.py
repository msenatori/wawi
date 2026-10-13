"""Plain-text turn printer for the terminal.

Why this exists: with the Rich TUI off (``ui.show_tui: false``) every trace of a
turn went to ``wawi.log`` and the console stayed completely silent. You could
talk to Wawi and watch nothing happen. This module is the one surface that
prints what was ASKED and what was ANSWERED, in the terminal you launched it
from, so a spoken answer also leaves something you can read, copy and scroll
back to.

It deliberately does not use Rich Live: the TUI owns that, and two live regions
on one stdout fight each other. These are ordinary appended lines, which is what
survives being redirected into a file by the background launcher
(``scripts/wawictl``) and read later with ``tail -f``.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO

# ANSI is used only when stdout is a real terminal. Under `wawictl start`
# stdout is a file, and escape codes there turn a readable transcript into
# noise that `tail` renders as garbage.
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


class TurnPrinter:
    """Print each question/answer pair as it completes.

    Args:
        enabled: False silences it entirely (the TUI is on, or the user asked
            for no console output).
        stream: Where to write. Defaults to stdout at call time, not at import
            time, so a redirected stream is honoured.
    """

    def __init__(self, enabled: bool = True, stream: TextIO | None = None) -> None:
        self._enabled = enabled
        self._stream = stream
        self._color: bool | None = None

    # -- internals ---------------------------------------------------------

    @property
    def _out(self) -> TextIO:
        return self._stream if self._stream is not None else sys.stdout

    def _use_color(self) -> bool:
        if self._color is None:
            try:
                self._color = bool(self._out.isatty())
            except Exception:
                self._color = False
        return self._color

    def _paint(self, text: str, *codes: str) -> str:
        if not self._use_color():
            return text
        return f"{''.join(codes)}{text}{_RESET}"

    def _write(self, line: str) -> None:
        if not self._enabled:
            return
        try:
            self._out.write(line + "\n")
            # Unbuffered on purpose: redirected to a file this is the only
            # thing that makes `tail -f` show the answer as it lands instead
            # of in 4 KB bursts.
            self._out.flush()
        except Exception:
            # A broken pipe (the terminal closed) must never take a turn down.
            pass

    # -- public API --------------------------------------------------------

    def banner(self, text: str) -> None:
        """One startup line telling the user how to talk to it."""
        self._write(self._paint(text, _DIM))

    def question(self, text: str) -> None:
        """Print the transcribed question."""
        if not text:
            return
        stamp = time.strftime("%H:%M:%S")
        self._write("")
        self._write(f"{self._paint(stamp, _DIM)} {self._paint('› ' + text, _BOLD)}")

    def answer(self, text: str) -> None:
        """Print the answer the assistant is about to speak."""
        if not text:
            return
        self._write(self._paint(f"  {text}", _CYAN))

    def note(self, text: str) -> None:
        """Print a dim aside (an error, a dropped turn)."""
        self._write(self._paint(f"  {text}", _DIM))
