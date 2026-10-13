"""The terminal transcript.

This module exists because a turn used to leave NO trace on the console when
the Rich TUI was off: everything went to ``wawi.log``. So the properties
worth pinning are the ones that make the printed answer trustworthy — it is
written even when nothing can be spoken, it is flushed immediately (the
background launcher redirects it into a file that ``tail -f`` follows), and it
carries no ANSI escapes into that file.
"""

from __future__ import annotations

import io

from wawi.console import TurnPrinter


class _Stream(io.StringIO):
    """A StringIO that reports itself as a pipe, and counts flushes."""

    def __init__(self, tty: bool = False) -> None:
        super().__init__()
        self._tty = tty
        self.flushes = 0

    def isatty(self) -> bool:
        return self._tty

    def flush(self) -> None:  # noqa: D102
        self.flushes += 1
        super().flush()


class TestOutput:
    def test_question_and_answer_both_reach_the_stream(self):
        out = _Stream()
        printer = TurnPrinter(stream=out)
        printer.question("cómo se escribe sol en inglés")
        printer.answer("Sun.")
        text = out.getvalue()
        assert "cómo se escribe sol en inglés" in text
        assert "Sun." in text

    def test_nothing_is_printed_when_disabled(self):
        out = _Stream()
        printer = TurnPrinter(enabled=False, stream=out)
        printer.question("hola")
        printer.answer("chau")
        assert out.getvalue() == ""

    def test_empty_text_prints_no_blank_line(self):
        out = _Stream()
        printer = TurnPrinter(stream=out)
        printer.question("")
        printer.answer("")
        assert out.getvalue() == ""

    def test_each_write_is_flushed(self):
        # Redirected into a file by scripts/wawictl, buffering is the
        # difference between `tail -f` showing the answer now and showing it
        # in a 4 KB burst several answers later.
        out = _Stream()
        printer = TurnPrinter(stream=out)
        printer.answer("Sun.")
        assert out.flushes >= 1


class TestColour:
    def test_no_ansi_escapes_when_the_stream_is_not_a_terminal(self):
        out = _Stream(tty=False)
        printer = TurnPrinter(stream=out)
        printer.question("hola")
        printer.answer("chau")
        assert "\033[" not in out.getvalue()

    def test_ansi_is_used_on_a_real_terminal(self):
        out = _Stream(tty=True)
        printer = TurnPrinter(stream=out)
        printer.answer("chau")
        assert "\033[" in out.getvalue()


class TestFailureIsNeverFatal:
    def test_a_broken_stream_does_not_raise(self):
        # The user closed the terminal mid-answer. Losing the transcript is
        # acceptable; losing the turn is not.
        class _Broken(io.StringIO):
            def write(self, s):  # noqa: ANN001, D102
                raise BrokenPipeError("terminal closed")

        printer = TurnPrinter(stream=_Broken())
        printer.question("hola")
        printer.answer("chau")
        printer.note("nada")
