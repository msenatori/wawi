"""Full-screen Rich Live TUI observer for Wawi — v3.

Renders a full-screen alternate-buffer terminal UI with three zones:
  - Header (1 line): compact WAWI label + arc-reactor glyph + state label
  - Body (fills remaining height): scrolling transcript of turns (auto-follow tail)
  - Footer (2 lines): status dot, VU meter / spinner / waveform, per-turn and
    session metrics

Architecture:
  - One rich.Live(screen=True) instance drives all rendering at 12 fps.
  - The worker thread only mutates plain Python fields (GIL-safe on CPython).
  - _FullScreenRenderable.__rich__() computes layout fresh each frame.
  - Transcript is a list of turn dicts; body auto-follows the tail.
  - No locks needed — plain attribute assignment is atomic enough for display.

Public interface (UNCHANGED from v2 — orchestrator call sites must not change):
  TuiObserver(enabled)
  .start()
  .stop()
  .__call__(old, new)          — state observer
  .set_metrics(turn, session)
  .set_level_source(source)
  .set_texts(user_text=, response=)

Threading note:
  Rich Live auto_refresh runs on an internal daemon thread. Worker thread writes
  plain fields. GIL + attribute atomicity is sufficient for display reads.

When enabled=False: all methods are no-ops.
"""

from __future__ import annotations

import itertools
import logging
import re
import time
from typing import TYPE_CHECKING, Callable

from wawi.state import State

if TYPE_CHECKING:
    from wawi.orchestrator import SessionMetrics, TurnMetrics

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State display maps
# ---------------------------------------------------------------------------

# Human-readable state labels (rioplatense Spanish, REQ-F1-07)
_LABELS: dict[State, str] = {
    State.IDLE: "en espera",
    State.LISTENING: "escuchando",
    State.THINKING: "pensando",
    State.SPEAKING: "hablando",
}

# Rich markup style per state
_STYLES: dict[State, str] = {
    State.IDLE: "dim",
    State.LISTENING: "bold green",
    State.THINKING: "bold yellow",
    State.SPEAKING: "bold blue",
}

# State dot colors for header/footer
_DOT_STYLES: dict[State, str] = {
    State.IDLE: "dim",
    State.LISTENING: "green",
    State.THINKING: "yellow",
    State.SPEAKING: "blue",
}

# ---------------------------------------------------------------------------
# Animation constants
# ---------------------------------------------------------------------------

# VU meter block characters: index 0 = silence, 8 = full
_VU_CHARS = " ▁▂▃▄▅▆▇█"

# Spinner frames (braille dots)
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Arc reactor animation sequences per state (driven by time.monotonic)
_REACTOR_IDLE = ["·", "◦", "○", "◉", "○", "◦"]
_REACTOR_THINKING = ["◜", "◝", "◞", "◟"]
_REACTOR_SPEAKING = [
    "▁▂▃▄▅", "▂▃▄▅▄", "▃▄▅▄▃", "▄▅▄▃▂",
    "▅▄▃▂▁", "▄▃▂▁▂", "▃▂▁▂▃", "▂▁▂▃▄",
]

# Reactor animation speeds (seconds per frame)
_REACTOR_IDLE_SPEED = 0.4
_REACTOR_THINK_SPEED = 0.12
_REACTOR_SPEAK_SPEED = 0.09

# Minimum body height to avoid crashing on tiny terminals
_MIN_BODY_LINES = 2

# Header height in lines (must match what _build_header produces)
_HEADER_LINES = 1

# Footer height in lines (must match what _build_footer produces)
_FOOTER_LINES = 2

# VU meter bar width (characters)
_VU_WIDTH = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vu_bar(level: float, width: int = _VU_WIDTH) -> str:
    """Build a VU meter string of *width* block characters from *level* (0..1)."""
    level = max(0.0, min(1.0, level))
    filled = round(level * width)
    bar = ""
    for i in range(width):
        if i < filled:
            intensity = int((i + 1) / width * (len(_VU_CHARS) - 1))
            bar += _VU_CHARS[intensity]
        else:
            bar += " "
    return bar


def _fmt_tokens(n: int | None) -> str:
    """Format token count as e.g. '1.8k' or '—'."""
    if n is None:
        return "—"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _fmt_cost(usd: float | None) -> str:
    """Format cost as e.g. '$0.012' or '—'."""
    if usd is None:
        return "—"
    return f"${usd:.3f}"


def _reactor_frame(frames: list[str], speed: float) -> str:
    """Pick the current animation frame based on wall time and speed."""
    idx = int(time.monotonic() / speed) % len(frames)
    return frames[idx]


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap *text* to fit within *width* characters, returning a list of lines.

    Simple word-wrap: breaks on spaces, never splits mid-word unless a single
    word exceeds width (in which case it hard-wraps at width).
    """
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for word in words:
            if not current:
                # First word on the line — may be longer than width
                if len(word) > width:
                    # Hard-wrap the oversized word
                    while len(word) > width:
                        lines.append(word[:width])
                        word = word[width:]
                    current = word
                else:
                    current = word
            elif len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                lines.append(current)
                if len(word) > width:
                    while len(word) > width:
                        lines.append(word[:width])
                        word = word[width:]
                    current = word
                else:
                    current = word
        if current:
            lines.append(current)
    return lines if lines else [""]


# Matches a fenced code block: ```[optional lang]\n<body>\n``` (or EOF).
_FENCED_BLOCK = re.compile(r"```([^\n`]*)\n(.*?)(?:```|$)", re.DOTALL)


def _count_syntax_lines(syntax: "Syntax", body_width: int) -> int:
    """Return the number of terminal lines a Syntax renderable occupies at body_width."""
    from rich.console import Console

    c = Console(width=body_width, highlight=False, no_color=True, force_terminal=False)
    rendered = c.render_lines(syntax, c.options.update_width(body_width), pad=False)
    return max(1, len(rendered))


class _Turn:
    """One conversation turn (user utterance + Wawi response + optional metrics)."""

    __slots__ = ("user", "response", "metrics")

    def __init__(self, user: str | None) -> None:
        self.user: str | None = user
        self.response: str | None = None
        self.metrics: "TurnMetrics | None" = None


# ---------------------------------------------------------------------------
# Renderable
# ---------------------------------------------------------------------------


class _FullScreenRenderable:
    """Rich renderable producing the full-screen layout on every __rich__() call.

    Three zones (header / body / footer) are stacked vertically. Body fills
    the remaining terminal height and auto-follows the transcript tail.
    """

    def __init__(self, observer: "TuiObserver") -> None:
        self._obs = observer
        self._spinner = itertools.cycle(_SPINNER_FRAMES)
        self._last_spinner_tick = 0.0
        self._spinner_frame = _SPINNER_FRAMES[0]

    # ------------------------------------------------------------------
    # Spinner helper
    # ------------------------------------------------------------------

    def _tick_spinner(self) -> str:
        now = time.monotonic()
        if now - self._last_spinner_tick >= 0.08:
            self._spinner_frame = next(self._spinner)
            self._last_spinner_tick = now
        return self._spinner_frame

    # ------------------------------------------------------------------
    # Zone builders
    # ------------------------------------------------------------------

    def _build_header(self, state: State, console_width: int) -> "Text":
        """1-line header: WAWI label + reactor glyph + state label."""
        from rich.text import Text

        label = _LABELS.get(state, state.value)
        style = _STYLES.get(state, "")

        if state == State.IDLE:
            glyph = _reactor_frame(_REACTOR_IDLE, _REACTOR_IDLE_SPEED)
        elif state == State.THINKING:
            glyph = _reactor_frame(_REACTOR_THINKING, _REACTOR_THINK_SPEED)
        elif state == State.SPEAKING:
            glyph = _reactor_frame(_REACTOR_SPEAKING, _REACTOR_SPEAK_SPEED)
        else:  # LISTENING
            glyph = "◉"

        header = Text(no_wrap=True)
        header.append("  WAWI ", style="bold white")
        header.append(glyph + " ", style=style)
        header.append(label, style=style)
        # Fill remaining width with a dim rule
        filled = f"  WAWI  {glyph}  {label}"
        remaining = max(0, console_width - len(filled) - 4)
        header.append("  " + "─" * remaining, style="dim")
        return header

    def _build_footer(
        self,
        state: State,
        turn_metrics: "TurnMetrics | None",
        session_metrics: "SessionMetrics | None",
        level_source: Callable[[], float] | None,
        console_width: int,
    ) -> "Text":
        """2-line footer: animation + per-turn latency/cost + session totals."""
        from rich.text import Text

        style = _STYLES.get(state, "")
        label = _LABELS.get(state, state.value)

        # Line 1: state dot + animation widget
        line1 = Text(no_wrap=True)
        dot_style = _DOT_STYLES.get(state, "dim")
        line1.append("  ● ", style=dot_style)
        line1.append(label + "  ", style=style)

        if state == State.LISTENING:
            level = level_source() if level_source is not None else 0.0
            vu = _vu_bar(level)
            line1.append("[", style="dim")
            line1.append(vu, style="green")
            line1.append("]", style="dim")
        elif state == State.THINKING:
            spin = self._tick_spinner()
            reactor = _reactor_frame(_REACTOR_THINKING, _REACTOR_THINK_SPEED)
            line1.append(f"{reactor} {spin} procesando…", style="yellow")
        elif state == State.SPEAKING:
            wave = _reactor_frame(_REACTOR_SPEAKING, _REACTOR_SPEAK_SPEED)
            line1.append(wave, style="blue")

        # Line 2: metrics
        line2 = Text(no_wrap=True)
        line2.append("  ", style="default")

        if turn_metrics is not None:
            tok_total: int | None = None
            if turn_metrics.tokens_in is not None or turn_metrics.tokens_out is not None:
                tok_total = (turn_metrics.tokens_in or 0) + (turn_metrics.tokens_out or 0)
            line2.append(
                f"stt {turn_metrics.stt_s:.1f}s  ia {turn_metrics.brain_s:.1f}s"
                f"  tot {turn_metrics.total_s:.1f}s",
                style="default",
            )
            line2.append("  │  ", style="dim")
            line2.append(
                f"{_fmt_cost(turn_metrics.cost_usd)} · {_fmt_tokens(tok_total)} tok",
                style="default",
            )
        else:
            line2.append("stt —  ia —  tot —  │  — · — tok", style="dim")

        if session_metrics is not None:
            sess_tok = session_metrics.total_tokens_in + session_metrics.total_tokens_out
            line2.append("  │  ", style="dim")
            line2.append(
                f"sesión {session_metrics.turn_count}"
                f" · {_fmt_cost(session_metrics.total_cost_usd or None)}"
                f" · {_fmt_tokens(sess_tok or None)}",
                style="dim",
            )
        else:
            line2.append("  │  sesión —", style="dim")

        footer = Text(no_wrap=True)
        footer.append_text(line1)
        footer.append("\n")
        footer.append_text(line2)
        return footer

    def _render_turns_tail(
        self,
        turns: list["_Turn"],
        body_height: int,
        body_width: int,
    ) -> "Text | Group":
        """Render transcript turns fitting within *body_height* lines, newest at bottom.

        Each element of all_lines is either:
          - (str, str)         — one terminal line: (content, style_hint)
          - (Syntax, int)      — a code block sentinel: (Syntax_obj, line_count)

        The tail is computed by accumulating from the end until body_height is
        reached, so multi-line code blocks are counted correctly.
        """
        from rich.console import Group
        from rich.text import Text

        if not turns:
            idle_msg = Text(no_wrap=True)
            idle_msg.append(
                "\n  (en espera — presioná la tecla para hablar)",
                style="dim",
            )
            return idle_msg

        # Collect rendered elements per turn (oldest to newest).
        # Elements are either (str, str) for a single line or (Syntax, int) sentinel.
        all_elements: list = []  # list[tuple[str, str] | tuple[Syntax, int]]

        for turn in turns:
            # User utterance
            if turn.user is not None:
                prefix = "  Vos: "
                content_width = max(1, body_width - len(prefix))
                wrapped = _wrap_text(turn.user, content_width)
                first = True
                for line in wrapped:
                    if first:
                        all_elements.append((prefix + line, "user_first"))
                        first = False
                    else:
                        all_elements.append((" " * len(prefix) + line, "user_cont"))

            # Wawi response — may contain fenced code blocks
            if turn.response is not None:
                prefix = "  Wawi: "
                prefix_width = len(prefix)
                indent = " " * prefix_width
                content_width = max(1, body_width - prefix_width)

                # Check whether the response contains any fenced code blocks.
                has_code = bool(_FENCED_BLOCK.search(turn.response))

                if not has_code:
                    # Plain prose — existing word-wrap path (regression-safe).
                    wrapped = _wrap_text(turn.response, content_width)
                    first = True
                    for line in wrapped:
                        if first:
                            all_elements.append((prefix + line, "resp_first"))
                            first = False
                        else:
                            all_elements.append((indent + line, "resp_cont"))
                else:
                    # Response has code blocks — split into prose/code segments.
                    from rich.syntax import Syntax

                    segments: list[tuple[str, str | None]] = []
                    cursor = 0
                    for m in _FENCED_BLOCK.finditer(turn.response):
                        if m.start() > cursor:
                            segments.append((turn.response[cursor : m.start()], None))
                        lang = m.group(1).strip() or "text"
                        segments.append((m.group(2), lang))
                        cursor = m.end()
                    if cursor < len(turn.response):
                        segments.append((turn.response[cursor:], None))

                    first_line = True
                    for seg_text, lang in segments:
                        if lang is None:
                            # Prose segment
                            wrapped = _wrap_text(seg_text.strip("\n"), content_width)
                            for line in wrapped:
                                if first_line:
                                    all_elements.append((prefix + line, "resp_first"))
                                    first_line = False
                                else:
                                    all_elements.append((indent + line, "resp_cont"))
                        else:
                            # Code block — emit Wawi prefix line if this is first
                            if first_line:
                                all_elements.append((prefix, "resp_first"))
                                first_line = False
                            code = seg_text.rstrip("\n")
                            syntax = Syntax(code, lang, theme="monokai", word_wrap=True)
                            n_lines = _count_syntax_lines(syntax, body_width)
                            # Sentinel tuple: (Syntax_obj, line_count)
                            all_elements.append((syntax, n_lines))

            # Per-turn metrics (optional, single line)
            if turn.metrics is not None:
                m = turn.metrics
                tok = None
                if m.tokens_in is not None or m.tokens_out is not None:
                    tok = (m.tokens_in or 0) + (m.tokens_out or 0)
                metrics_str = (
                    f"  ↳ {_fmt_cost(m.cost_usd)} · {_fmt_tokens(tok)} tok"
                    f" · stt {m.stt_s:.1f}s ia {m.brain_s:.1f}s"
                )
                all_elements.append((metrics_str, "metrics"))

            # Separator between turns
            all_elements.append(("  " + "╌" * max(0, body_width - 4), "sep"))

        # Take the tail that fits body_height lines (counting Syntax blocks correctly).
        # Walk from the end, accumulating line counts, until we hit body_height.
        tail: list = []
        lines_used = 0
        for elem in reversed(all_elements):
            if isinstance(elem[0], str):
                elem_lines = 1
            else:
                # Syntax sentinel: (Syntax_obj, line_count)
                elem_lines = elem[1]
            if lines_used + elem_lines > body_height:
                break
            tail.append(elem)
            lines_used += elem_lines
        tail.reverse()

        # Build output: a Group of Text and Syntax renderables stacked top-to-bottom.
        # We accumulate plain-text chunks into a Text, then flush when a Syntax block
        # arrives (Group requires separate renderables per code block).
        renderables = []
        current_text: Text | None = None

        def _flush_text() -> None:
            nonlocal current_text
            if current_text is not None and len(current_text) > 0:
                renderables.append(current_text)
                current_text = None

        def _ensure_text() -> Text:
            nonlocal current_text
            if current_text is None:
                current_text = Text(no_wrap=True)
            return current_text

        for i, elem in enumerate(tail):
            is_last = i == len(tail) - 1
            if isinstance(elem[0], str):
                content, hint = elem
                t = _ensure_text()
                if hint == "user_first":
                    prefix_end = content.index(": ") + 2 if ": " in content else 0
                    t.append(content[:prefix_end], style="bold cyan")
                    t.append(content[prefix_end:], style="cyan")
                elif hint == "user_cont":
                    t.append(content, style="cyan")
                elif hint == "resp_first":
                    prefix_end = content.index(": ") + 2 if ": " in content else 0
                    t.append(content[:prefix_end], style="bold green")
                    t.append(content[prefix_end:], style="default")
                elif hint == "resp_cont":
                    t.append(content, style="default")
                elif hint == "metrics":
                    t.append(content, style="dim")
                else:  # sep
                    t.append(content, style="dim")
                if not is_last:
                    t.append("\n")
            else:
                # Syntax sentinel — flush pending text, then append Syntax.
                syntax_obj, _n = elem
                _flush_text()
                renderables.append(syntax_obj)
                # Add a newline-Text between the Syntax and what follows, if needed.
                if not is_last:
                    current_text = Text(no_wrap=True)
                    current_text.append("\n")
                    _flush_text()

        _flush_text()

        if not renderables:
            return Text(no_wrap=True)
        if len(renderables) == 1 and isinstance(renderables[0], Text):
            return renderables[0]
        return Group(*renderables)

    # ------------------------------------------------------------------
    # Rich protocol
    # ------------------------------------------------------------------

    def __rich__(self):  # type: ignore[return]
        from rich.console import Console, Group
        from rich.rule import Rule
        from rich.text import Text

        obs = self._obs
        state = obs._current_state

        # Probe terminal size from the live console if available
        console = obs._console
        if console is not None:
            term_width = console.width
            term_height = console.height
        else:
            term_width = 80
            term_height = 24

        # Compute body height — clamp to minimum to avoid crashes
        body_height = max(_MIN_BODY_LINES, term_height - _HEADER_LINES - _FOOTER_LINES - 2)
        body_width = max(20, term_width - 2)

        header = self._build_header(state, term_width)

        body = self._render_turns_tail(
            obs._turns,
            body_height=body_height,
            body_width=body_width,
        )

        footer = self._build_footer(
            state,
            obs._turn_metrics,
            obs._session_metrics,
            obs._level_source,
            term_width,
        )

        # Stack header + divider + body + divider + footer
        return Group(
            header,
            Rule(style="dim"),
            body,
            Rule(style="dim"),
            footer,
        )

    def _get_renderable(self):
        """Return the renderable for testing without Live alt-screen."""
        return self.__rich__()


# ---------------------------------------------------------------------------
# TuiObserver — public class
# ---------------------------------------------------------------------------


class TuiObserver:
    """Full-screen animated Rich Live TUI for Wawi.

    Args:
        enabled: If False, all methods are no-ops (show_tui=False in config).
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._live = None  # rich.live.Live instance, created on start()
        self._renderable: _FullScreenRenderable | None = None
        self._console = None  # rich.console.Console used by Live

        self._current_state: State = State.IDLE

        # Metrics stored by orchestrator worker thread
        self._turn_metrics: "TurnMetrics | None" = None
        self._session_metrics: "SessionMetrics | None" = None

        # Callable returning current audio level (0..1); wired by orchestrator
        self._level_source: Callable[[], float] | None = None

        # Transcript: list of _Turn objects, newest at the end
        self._turns: list[_Turn] = []

    # ------------------------------------------------------------------
    # Internal helper for renderable access (used by tests)
    # ------------------------------------------------------------------

    def _get_renderable(self):
        """Return the current frame renderable. Usable without Live alt-screen."""
        if self._renderable is None:
            # Create a temporary renderable for testing
            self._renderable = _FullScreenRenderable(self)
        return self._renderable._get_renderable()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enter the Rich Live full-screen context and begin auto-refreshing.

        Must be called from the main thread. No-op when disabled.
        """
        if not self.enabled:
            return

        from rich.console import Console
        from rich.live import Live

        self._console = Console()
        self._renderable = _FullScreenRenderable(self)
        self._live = Live(
            self._renderable,
            console=self._console,
            refresh_per_second=12,
            screen=True,  # alternate screen buffer — full terminal
        )
        self._live.start()

    def stop(self) -> None:
        """Exit the Rich Live context. Safe even if start() was never called."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                log.warning("Error stopping Rich Live", exc_info=True)
            self._live = None
            self._renderable = None
            self._console = None

    # ------------------------------------------------------------------
    # Observer callback (state transitions from worker thread)
    # ------------------------------------------------------------------

    def __call__(self, old: State, new: State) -> None:  # noqa: ARG002
        """State observer: update current state for the next render frame."""
        if not self.enabled:
            return
        self._current_state = new

    # ------------------------------------------------------------------
    # Metrics update (called from worker thread after each real turn)
    # ------------------------------------------------------------------

    def set_metrics(self, turn: "TurnMetrics", session: "SessionMetrics") -> None:
        """Store per-turn and session metrics for display on the next frame.

        Also attaches the turn metrics to the most recent pending turn in the
        transcript so per-turn metrics appear inline in the body.
        """
        if not self.enabled:
            return
        self._turn_metrics = turn
        self._session_metrics = session

        # Attach to the latest turn that has no metrics yet
        for t in reversed(self._turns):
            if t.metrics is None:
                t.metrics = turn
                break

    # ------------------------------------------------------------------
    # Level source wiring (called from orchestrator __init__)
    # ------------------------------------------------------------------

    def set_level_source(self, source: Callable[[], float]) -> None:
        """Set the callable used to read current audio level for the VU meter.

        Args:
            source: Zero-argument callable returning float in 0..~1.0 range.
        """
        if not self.enabled:
            return
        self._level_source = source

    # ------------------------------------------------------------------
    # Text / transcript accumulation (called from worker thread per turn)
    # ------------------------------------------------------------------

    def set_texts(self, user_text: str | None = None, response: str | None = None) -> None:
        """Accumulate transcript turns for on-screen display.

        Calling with user_text=X opens a new turn (appends to _turns).
        Calling with response=Y fills the response on the latest pending turn.
        Both can be called independently; guards handle missing context.
        """
        if not self.enabled:
            return

        if user_text is not None:
            # New utterance → new turn
            self._turns.append(_Turn(user=user_text))

        if response is not None:
            # Fill response on the latest turn that has no response yet
            for t in reversed(self._turns):
                if t.response is None:
                    t.response = response
                    return
            # No pending turn — create one with user=None as fallback
            fallback = _Turn(user=None)
            fallback.response = response
            self._turns.append(fallback)
