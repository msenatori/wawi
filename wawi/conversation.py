"""Rolling conversation memory and a readable transcript.

Why this exists
---------------
Every brain turn used to start from nothing. ``ask_sync()`` took a single
string, so the model could not see anything said thirty seconds earlier -- it
would answer, verbatim, "no tengo contexto de la conversación anterior donde lo
llamaste la lista". For a voice assistant that is not a rough edge, it is the
whole product missing: people refer back constantly ("el 3", "ese", "lo que
dijiste recién") and speech makes them do it *more* than typing, because
repeating a repository name out loud is exactly what they are trying to avoid.

What it is (and is not)
-----------------------
A bounded, in-memory ring of recent turns rendered into the prompt. It is NOT
a database, NOT persisted across restarts, and NOT a semantic memory. That is a
deliberate floor, not an oversight:

- Bounded, because every remembered turn is re-sent (and re-paid for) on every
  later turn. The window trades money and latency for continuity, so it is
  small and explicit rather than unlimited and surprising.
- In-memory FOR THE MODEL, because the window is what gets re-sent. What the
  model can see and what is written down are separate decisions.

The readable log on disk (``log_path``) is a third thing again, and the one to
be careful with: it is a running record of everything said near the machine,
so it is opt-in through ``conversation_log`` in config.yaml, appends rather
than truncates, and never contains anything the user did not say to Wawi on
purpose. Turning it off leaves no file behind.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Turns kept for the model. Each one costs input tokens on every later turn,
# so this buys continuity by the sentence, not by the hour. Six covers "el 3"
# style references comfortably without the prompt growing without bound.
DEFAULT_MAX_TURNS = 6

# A single remembered turn is truncated to this many characters. A long answer
# still gets spoken in full; only its memory is clipped, because the useful
# part for a follow-up is the topic, not every word.
_MAX_REMEMBERED_CHARS = 600


@dataclass(frozen=True)
class Turn:
    """One exchange: what the user said and what the assistant answered."""

    user: str
    assistant: str
    tools: tuple[str, ...] = ()

    def render(self, assistant_name: str) -> str:
        """Format for inclusion in the prompt."""
        said = _clip(self.user)
        answered = _clip(self.assistant)
        return f"Usuario: {said}\n{assistant_name}: {answered}"


def _clip(text: str, limit: int = _MAX_REMEMBERED_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass
class Conversation:
    """Thread-safe rolling window of recent turns.

    Written from the pipeline worker thread and read from the UI thread, so
    every access is guarded. The lock is held only for list operations, never
    across a model call.
    """

    max_turns: int = DEFAULT_MAX_TURNS
    log_path: Path | None = None
    assistant_name: str = "assistant"
    _turns: list[Turn] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def __iter__(self) -> Iterator[Turn]:
        return iter(self.turns())

    def turns(self) -> list[Turn]:
        """Return a snapshot, oldest first."""
        with self._lock:
            return list(self._turns)

    def add(self, user: str, assistant: str, tools: Iterable[str] = ()) -> None:
        """Record one completed exchange, dropping the oldest if full.

        Turns with nothing on either side are ignored: an empty transcription
        or a dropped answer is not something to remember, and keeping it would
        waste a slot in the window.
        """
        user = (user or "").strip()
        assistant = (assistant or "").strip()
        if not user and not assistant:
            return
        turn = Turn(user, assistant, tuple(tools))
        with self._lock:
            self._turns.append(turn)
            if len(self._turns) > self.max_turns:
                del self._turns[: len(self._turns) - self.max_turns]
        self._append_to_log(turn)

    def _append_to_log(self, turn: Turn) -> None:
        """Append one exchange to the readable log, if one is configured.

        Never raises: a full disk or a bad path must not cost the user their
        answer, so a failure is logged once and the turn continues.
        """
        if self.log_path is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tools = f"  [{', '.join(turn.tools)}]" if turn.tools else ""
        block = (
            f"\n[{stamp}]\n"
            f"  vos    > {turn.user}\n"
            f"  {self.assistant_name:6.6s} < {turn.assistant}{tools}\n"
        )
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(block)
        except Exception:
            log.warning("Could not append to %s", self.log_path, exc_info=True)

    def clear(self) -> None:
        """Forget everything. Wired to the user saying so, never automatic."""
        with self._lock:
            count = len(self._turns)
            self._turns.clear()
        log.info("Conversation memory cleared (%d turns dropped)", count)

    def as_prompt_context(self, assistant_name: str) -> str:
        """Render the window as a prompt preamble, or "" when empty.

        The framing matters: the model is told this is *earlier in the same
        conversation* and that references like "el 3" may point into it.
        Without that, models tend to treat a transcript as reference material
        and answer about it rather than continuing it.
        """
        turns = self.turns()
        if not turns:
            return ""
        body = "\n".join(t.render(assistant_name) for t in turns)
        return (
            "Conversación hasta ahora (lo más reciente al final). El usuario "
            "puede referirse a cualquier cosa dicha acá, incluso por número o "
            'con "eso" o "el de antes":\n'
            f"{body}\n\n"
            "Mensaje nuevo del usuario:\n"
        )

    def transcript(self, assistant_name: str = "assistant") -> str:
        """Full readable log of the window, for display or for saving."""
        lines: list[str] = []
        for turn in self.turns():
            lines.append(f"Usuario: {turn.user}")
            if turn.tools:
                lines.append(f"  [herramientas: {', '.join(turn.tools)}]")
            lines.append(f"{assistant_name}: {turn.assistant}")
        return "\n".join(lines)
