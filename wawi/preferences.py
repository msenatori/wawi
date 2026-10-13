"""Durable preferences the user teaches the assistant, in their own words.

The gap this fills
------------------
Conversation memory (:mod:`wawi.conversation`) is a rolling window: say "sé
más breve" and six turns later it is gone. But "sé más breve" is not a fact
about *this* conversation, it is a standing instruction about how to answer
every one. Those two things look similar and behave completely differently, so
they are stored separately.

This is the useful half of self-improvement. The assistant gets better at
answering without a single line of its code changing: the user corrects it in
speech, the correction persists, and it lands in the system prompt on every
later turn. It is reviewable (a plain text file), reversible (delete a line),
and it cannot break the program -- the worst a bad preference can do is make an
answer worse, which is exactly the blast radius you want for something the
assistant writes about itself.

Why so few, and so short
------------------------
Preferences are re-sent on every turn, so they are paid for forever. A cap
keeps that bounded, and it also keeps them *effective*: thirty competing style
instructions do not produce a well-behaved assistant, they produce one that
ignores all of them. Newest wins on conflict, because the most recent
correction is the one the user is currently frustrated about.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Preferences kept. Small on purpose — see the module docstring.
MAX_PREFERENCES = 12

# Longer than this and it is not a preference, it is a task.
MAX_CHARS = 200


@dataclass(frozen=True)
class Preference:
    """One standing instruction, with when it was given."""

    text: str
    added: str = ""

    def to_line(self) -> str:
        return f"{self.added}\t{self.text}" if self.added else self.text

    @classmethod
    def from_line(cls, line: str) -> "Preference | None":
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            return None
        if "\t" in line:
            added, _, text = line.partition("\t")
            return cls(text.strip(), added.strip())
        return cls(line.strip())


class Preferences:
    """Thread-safe, file-backed list of standing instructions.

    Persisted as plain text, one per line, so the user can open the file and
    delete something without going through the assistant. That matters: a
    preference the assistant misheard should not need the assistant's
    cooperation to remove.
    """

    def __init__(self, path: Path | None = None, max_items: int = MAX_PREFERENCES) -> None:
        self.path = Path(path) if path else None
        self.max_items = max_items
        self._lock = threading.Lock()
        self._items: list[Preference] = []
        self.load()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def load(self) -> None:
        """Read the file if it exists. A malformed file is never fatal."""
        if self.path is None or not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            log.warning("Could not read preferences from %s", self.path, exc_info=True)
            return
        parsed = [p for p in (Preference.from_line(ln) for ln in lines) if p]
        with self._lock:
            self._items = parsed[-self.max_items :]
        log.info("Loaded %d preferences from %s", len(parsed), self.path)

    def items(self) -> list[Preference]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def add(self, text: str) -> Preference | None:
        """Store a standing instruction. Returns it, or None if unusable.

        A duplicate is moved to the end rather than stored twice: repeating
        yourself should reinforce a preference, not consume two of the slots.
        """
        text = " ".join((text or "").split())
        if not text:
            return None
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS - 1].rstrip() + "…"

        pref = Preference(text, datetime.now().strftime("%Y-%m-%d"))
        with self._lock:
            lowered = text.lower()
            self._items = [p for p in self._items if p.text.lower() != lowered]
            self._items.append(pref)
            if len(self._items) > self.max_items:
                dropped = self._items[: len(self._items) - self.max_items]
                del self._items[: len(self._items) - self.max_items]
                for d in dropped:
                    log.info("Preference dropped (list full): %r", d.text)
        self.save()
        log.info("Preference learned: %r", text)
        return pref

    def remove(self, index: int) -> Preference | None:
        """Forget preference *index* (1-based, as the user hears it listed)."""
        with self._lock:
            if not 1 <= index <= len(self._items):
                return None
            pref = self._items.pop(index - 1)
        self.save()
        log.info("Preference forgotten: %r", pref.text)
        return pref

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
        self.save()
        log.info("All preferences cleared")

    def save(self) -> None:
        """Persist. Never raises: losing a preference must not lose the turn."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            body = "\n".join(
                ["# Preferencias que le enseñaste. Editá o borrá líneas a mano."]
                + [p.to_line() for p in self.items()]
            )
            self.path.write_text(body + "\n", encoding="utf-8")
        except Exception:
            log.warning("Could not save preferences to %s", self.path, exc_info=True)

    def as_prompt_block(self) -> str:
        """Render for the system prompt, or "" when there is nothing."""
        items = self.items()
        if not items:
            return ""
        body = "\n".join(f"- {p.text}" for p in items)
        return (
            "El usuario te enseñó estas preferencias. Respetalas en TODAS tus "
            "respuestas; si dos se contradicen, gana la última:\n"
            f"{body}"
        )

    def spoken_list(self) -> str:
        """Numbered list to read out loud when asked what it has learned."""
        items = self.items()
        if not items:
            return "Todavía no me enseñaste ninguna preferencia."
        return "; ".join(f"{i}: {p.text}" for i, p in enumerate(items, 1))
