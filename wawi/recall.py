"""A small durable rolling cache of what was said, so a restart is not amnesia.

The gap this fills
------------------
:mod:`wawi.conversation` is a six-turn ring that says so in its own docstring:
in memory, for the model, gone on exit. The SDK session id in the orchestrator
is the same story with a better cache — it holds the real history, but it lives
in an attribute and is dropped on any failed turn. So every restart, every
crash, every ``ctrl-c`` puts the user back at a machine that has never met them.
That is what this fixes, and it fixes only that.

Why the block is FROZEN for the whole process
---------------------------------------------
This lands in the system prompt, and on the fast path (:class:`wawi.brain.
Engine`) the system prompt is part of the connection signature: changing it
tears down the live session and reconnects. If the block were re-rendered after
every turn it would reconnect after every turn -- paying the ~1.3s the CLI takes
to start AND throwing away the in-session conversation, which is the very thing
this module exists to protect. So the snapshot read at startup is what the model
sees for the whole run, and it never changes underneath the connection.

That is also the honest semantics. The block means "before this session". Inside
the session the live conversation already covers continuity, so re-injecting a
rolling copy of it would say the same thing twice and, the moment the window
slid, say it differently the second time.

Why the raw tail and not distilled facts
----------------------------------------
Distilling "el proyecto se llama wawi" out of a transcript needs a model. A
second call per turn costs money and latency on a product whose whole complaint
is latency, and a ``remember_fact`` tool costs a fourth schema on every turn of
every user (see :mod:`wawi.tools` on why that list is three entries long). So
this keeps the raw tail -- clipped much harder than the in-session window, since
what survives a restart usefully is the topic, not the wording -- and the prompt
says out loud that it was transcribed from speech and may be wrong. Framing it
as unreliable is cheaper than making it reliable, and it is the difference
between context and a misheard sentence quoted back as fact.

Privacy
-------
This is a transcript of someone's home written to a file. It is opt-in through
``recall_file`` in config.yaml exactly like ``conversation_log``: an empty value
disables it and leaves nothing behind. :meth:`Recall.clear` deletes the file
rather than blanking it, because a wipe the user asked for should not leave the
old bytes on disk to be recovered.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# Exchanges carried across a restart. Small on purpose: every one of them is
# re-sent inside the system prompt on every turn of the next session, and the
# point is to walk back in knowing what you were talking about, not to give the
# model a diary. Configurable through `recall_turns`.
DEFAULT_MAX_ENTRIES = 6

# Each side of a remembered exchange is clipped to this. Much shorter than the
# 600 the in-session window allows, because a day later the useful part is the
# subject, not the sentence -- and this text is paid for on every turn of the
# whole next run, not just the next six.
MAX_CHARS_PER_SIDE = 160

# Field separator on disk. Speech has no tabs in it and every field is
# whitespace-collapsed before it is written, so a value can never smuggle one in
# and shift the columns.
_SEP = "\t"

_HEADER = (
    "# Lo último que hablaron, para no arrancar de cero. Borrá el archivo "
    "cuando quieras: se regenera vacío."
)


@dataclass(frozen=True)
class Exchange:
    """One remembered turn: what was asked, what was answered, and when."""

    user: str
    assistant: str
    when: str = ""

    def to_line(self) -> str:
        return _SEP.join((self.when, self.user, self.assistant))

    @classmethod
    def from_line(cls, line: str) -> "Exchange | None":
        """Parse one stored line, or None when it is not one.

        A line that does not parse is skipped rather than raised on: the file is
        appended to by a process that gets killed, so a torn last line is an
        expected state, not a corrupt store.
        """
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            return None
        parts = line.split(_SEP)
        if len(parts) != 3:
            return None
        when, user, assistant = (p.strip() for p in parts)
        if not user and not assistant:
            return None
        return cls(user, assistant, when)

    def render(self, assistant_name: str) -> str:
        return f"Usuario: {self.user}\n{assistant_name}: {self.assistant}"


def _clip(text: str, limit: int = MAX_CHARS_PER_SIDE) -> str:
    """Collapse whitespace and cut to *limit*, ellipsis included when cut."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class Recall:
    """Thread-safe, file-backed rolling cache of recent exchanges.

    Written from the pipeline worker thread and read from wherever the prompt is
    assembled, so every access is guarded. The lock is never held across a model
    call -- only across list operations and the write.
    """

    def __init__(
        self, path: Path | None = None, max_entries: int = DEFAULT_MAX_ENTRIES
    ) -> None:
        self.path = Path(path) if path else None
        # A cap of zero is "remember nothing", which is a legitimate answer and
        # must not become a negative slice that silently keeps everything.
        self.max_entries = max(0, int(max_entries))
        self._lock = threading.Lock()
        self._items: list[Exchange] = []
        self.load()
        # What the model is told, frozen here for the life of the process. See
        # the module docstring: re-rendering this per turn reconnects the live
        # SDK session every turn.
        self._carried_over: list[Exchange] = list(self._items)

    @property
    def enabled(self) -> bool:
        return self.path is not None and self.max_entries > 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def entries(self) -> list[Exchange]:
        """Snapshot of what will be written, oldest first."""
        with self._lock:
            return list(self._items)

    def carried_over(self) -> list[Exchange]:
        """Snapshot of what came from previous sessions, oldest first."""
        return list(self._carried_over)

    def load(self) -> None:
        """Read the store. A missing, unreadable or torn file means no memory.

        Nothing here may raise. This runs during startup, and a store half
        written by a process that AppKit killed must cost the user their memory,
        never their assistant.
        """
        if not self.enabled:
            return
        assert self.path is not None
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            return
        except Exception:
            log.warning(
                "Could not read the rolling memory at %s — starting with none",
                self.path,
                exc_info=True,
            )
            return
        parsed = [e for e in (Exchange.from_line(ln) for ln in lines) if e]
        dropped = len(lines) - len(parsed)
        with self._lock:
            self._items = parsed[-self.max_entries :]
        if dropped:
            log.info("Rolling memory: skipped %d unreadable line(s)", dropped)
        log.info("Rolling memory: %d exchange(s) carried over", len(self._items))

    def add(self, user: str, assistant: str) -> Exchange | None:
        """Record one exchange, evicting the oldest, and persist.

        An exchange with nothing on either side is ignored: a silent capture or
        a dropped answer is not worth one of the few slots that survive a
        restart.
        """
        if not self.enabled:
            return None
        user, assistant = _clip(user), _clip(assistant)
        if not user and not assistant:
            return None
        entry = Exchange(user, assistant, datetime.now().strftime("%Y-%m-%d %H:%M"))
        with self._lock:
            self._items.append(entry)
            excess = len(self._items) - self.max_entries
            if excess > 0:
                del self._items[:excess]
        self.save()
        return entry

    def clear(self) -> None:
        """Forget everything, on disk too. Wired to the user asking, never automatic.

        The file is deleted rather than emptied. Someone wiping a transcript of
        their own house is not asking for a shorter file, and leaving the old
        bytes in place for an undelete to find would be answering a different
        question than the one they asked.
        """
        with self._lock:
            count = len(self._items)
            self._items.clear()
            self._carried_over = []
        if self.path is not None:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                log.warning("Could not delete %s", self.path, exc_info=True)
        log.info("Rolling memory cleared (%d exchange(s) dropped)", count)

    def save(self) -> None:
        """Persist atomically. Never raises: losing memory must not lose the turn.

        Written to a sibling temp file and renamed over the target, because the
        writer is a process that is routinely killed with ctrl-c mid-run. A
        plain truncate-and-write leaves a half file behind when that happens;
        :func:`os.replace` is atomic on the same filesystem, so the next start
        reads either the whole old store or the whole new one.
        """
        if not self.enabled:
            return
        assert self.path is not None
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            body = "\n".join([_HEADER] + [e.to_line() for e in self.entries()])
            tmp.write_text(body + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            log.warning(
                "Could not save the rolling memory to %s", self.path, exc_info=True
            )
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                log.debug("Could not remove %s", tmp, exc_info=True)

    def as_prompt_block(self, assistant_name: str) -> str:
        """The carried-over exchanges as prompt body, or "" when there are none.

        Body only: the framing sentence that tells the model what this is lives
        in :mod:`wawi.prompts`, with the rest of the language-dependent prompt
        text, so it is written once per language instead of once per store.
        """
        entries = self.carried_over()
        if not entries:
            return ""
        return "\n".join(e.render(assistant_name) for e in entries)
