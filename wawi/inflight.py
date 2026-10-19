"""What was cut off: the turn or the task that was running when Wawi died.

Everything else Wawi knows about the work in progress lives in memory —
:class:`wawi.task.Task`, the conversational session id — and a crash, a
``kill -9`` or a restart takes it with it. The user then has to notice, and
remember, and say it all again. This file is the minimum that survives: what
was being done, and the SDK session it was being done in.

Two slots, because both can be live at once: a task runs for minutes while
ordinary turns ("¿cómo va?") come and go next to it. A turn finishing must not
erase the task that is still running.

The lifecycle is the whole contract:

* written when a turn starts its brain call, and when a task is dispatched;
* cleared when that turn ends — answered or failed — and when that task
  reaches DONE, FAILED or a user stop;
* NOT cleared by a shutdown with a task still running: that task was cut, it
  did not finish;
* the last completed conversational session is kept separately from both
  interrupted-work slots, so an idle restart can continue the conversation.

So a clean exit with nothing in flight leaves nothing to offer. :meth:`Inflight.take`
clears the interrupted-work slots as it reads them, so they are offered exactly
once, while leaving a completed conversation session available for the next turn.

Same failure rule as :mod:`wawi.recall`: a store that cannot be read or
written costs the recovery, never the turn. A corrupt file is logged and
treated as no file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KIND_TURN = "turn"
KIND_TASK = "task"

#: Cap on the text kept per record. It is spoken back as part of a question,
#: and a whole dictated paragraph is not something to read out on startup.
_TEXT_CHARS = 600


@dataclass(frozen=True)
class Record:
    """One piece of interrupted work.

    ``text`` is the user's words for a turn and the goal for a task: in both
    cases what has to be sent again to redo it.
    """

    kind: str
    text: str
    ts: float
    session_id: str | None = None
    task_id: str = ""

    def to_dict(self) -> dict:
        body = {"kind": self.kind, "text": self.text, "ts": self.ts}
        if self.session_id:
            body["session_id"] = self.session_id
        if self.task_id:
            body["id"] = self.task_id
        return body

    @classmethod
    def from_dict(cls, raw: object, kind: str) -> "Record | None":
        """Parse one slot, or None when it is not a well-formed record."""
        if not isinstance(raw, dict):
            return None
        text = raw.get("text")
        ts = raw.get("ts")
        session_id = raw.get("session_id")
        if not isinstance(text, str) or not text.strip():
            return None
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            return None
        if session_id is not None and not isinstance(session_id, str):
            return None
        return cls(
            kind=kind,
            text=text.strip(),
            ts=float(ts),
            session_id=session_id or None,
            task_id=str(raw.get("id", "") or ""),
        )


class Inflight:
    """The interrupted-work file. ``Inflight(None)`` is a store that keeps nothing.

    Thread-safe: the turn thread and the executor's thread both write to it.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._slots: dict[str, Any] = {}
        if self.path is not None:
            session = (self._read() or {}).get("session")
            if isinstance(session, str) and session:
                self._slots["session"] = session

    @property
    def enabled(self) -> bool:
        return self.path is not None

    # -- writers ---------------------------------------------------------

    def mark_turn(self, user_text: str, session_id: str | None) -> None:
        """A turn is about to ask the brain. Cleared by :meth:`clear_turn`."""
        self._put(KIND_TURN, Record(KIND_TURN, _clip(user_text), time.time(), session_id))

    def clear_turn(self) -> None:
        self._drop(KIND_TURN)

    def conversation_session(self) -> str | None:
        """The last completed conversational session, if one is persisted."""
        with self._lock:
            return self._slots.get("session")

    def remember_session(self, session_id: str | None) -> None:
        """Persist (or forget) the session independently of interrupted work."""
        if self.path is None:
            return
        with self._lock:
            if session_id:
                self._slots["session"] = session_id
            else:
                self._slots.pop("session", None)
            self._save_locked()

    def mark_task(self, task_id: str, goal: str, session_id: str | None = None) -> None:
        """A task was dispatched. Cleared by :meth:`clear_task`."""
        self._put(
            KIND_TASK,
            Record(KIND_TASK, _clip(goal), time.time(), session_id, task_id=task_id),
        )

    def set_task_session(self, task_id: str, session_id: str) -> None:
        """Attach the executor's session once the SDK reports it.

        Ignored for a task that is no longer the one on file — a late report
        from a finished run must not resurrect it.
        """
        if not session_id:
            return
        with self._lock:
            slot = self._slots.get(KIND_TASK)
            if slot is None or slot.get("id") != task_id:
                return
            if slot.get("session_id") == session_id:
                return
            slot["session_id"] = session_id
            self._save_locked()

    def clear_task(self, task_id: str) -> None:
        """Forget *task_id*. A newer task on file is left alone."""
        with self._lock:
            slot = self._slots.get(KIND_TASK)
            if slot is None or slot.get("id") != task_id:
                return
            del self._slots[KIND_TASK]
            self._save_locked()

    # -- reader ----------------------------------------------------------

    def take(self, max_age_s: float) -> Record | None:
        """Read what the last run left, clear its work slots, and return one to offer.

        Work slots are cleared on read: asking about the same interruption
        on every start until someone answers is worse than asking once. When
        both slots survived, the task wins — a turn that overlapped a running
        task was almost always a question ABOUT that task.
        """
        if self.path is None:
            return None
        with self._lock:
            raw = self._read()
            session = raw.get("session") if raw is not None else None
            self._slots = {"session": session} if isinstance(session, str) and session else {}
            self._save_locked()
        if raw is None:
            return None
        for kind in (KIND_TASK, KIND_TURN):
            if kind not in raw:
                continue
            record = Record.from_dict(raw[kind], kind)
            if record is None:
                log.warning("Ignoring a malformed %s record in %s", kind, self.path)
                continue
            age = time.time() - record.ts
            if max_age_s <= 0 or age > max_age_s:
                log.info(
                    "Ignoring an interrupted %s from %.1fh ago (older than %.1fh)",
                    kind, age / 3600, max_age_s / 3600,
                )
                continue
            return record
        return None

    # -- internals -------------------------------------------------------

    def _put(self, kind: str, record: Record) -> None:
        if self.path is None or not record.text:
            return
        with self._lock:
            self._slots[kind] = record.to_dict()
            self._save_locked()

    def _drop(self, kind: str) -> None:
        if self.path is None:
            return
        with self._lock:
            if self._slots.pop(kind, None) is None:
                return
            self._save_locked()

    def _read(self) -> dict | None:
        assert self.path is not None
        try:
            body = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception:
            log.warning("Could not read %s; ignoring it", self.path, exc_info=True)
            return None
        try:
            raw = json.loads(body)
        except ValueError:
            log.warning("%s is corrupt; ignoring it", self.path)
            return None
        if not isinstance(raw, dict):
            log.warning("%s is not a JSON object; ignoring it", self.path)
            return None
        return raw

    def _save_locked(self) -> None:
        """Persist atomically, or remove the file when nothing is in flight.

        tmp + :func:`os.replace`, as in :meth:`wawi.recall.Recall.save`: this
        is written precisely around the moments the process is killed, so a
        half-written file is the expected case, not the edge case.
        """
        assert self.path is not None
        if not self._slots:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                log.warning("Could not delete %s", self.path, exc_info=True)
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self._slots, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            log.warning("Could not save %s", self.path, exc_info=True)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                log.debug("Could not remove %s", tmp, exc_info=True)


def _clip(text: str) -> str:
    return " ".join((text or "").split())[:_TEXT_CHARS]
