"""What the owner taught her about her initiative, in ``initiative_local.json``.

Every act kind has a **level**: ``off`` < ``quiet`` (done silently) <
``offer`` (a bubble with a link) < ``say`` / ``auto`` (spoken, or done
directly). Each kind declares where it starts and how high it may go
(:data:`wawi.initiative.core.KINDS`). Then the owner moves it, without ever
being asked:

* **accepted** (the link was clicked, or the owner did it by voice) three
  times in a row: one step up, up to the kind's ceiling;
* **ignored** (the bubble went away untouched) counts one, **dismissed** (the
  bubble clicked away) counts two; three points with no yes in between: one
  step down, ``say``/``auto`` → ``offer`` → ``quiet`` → ``off``.

The owner's own word wins over all of it: a kind switched off from the menu
or by voice ("no me avises más de eso") stays off until switched on again,
which also forgets a learned ``off``. The master switch, the calendar opt-in,
the morning summary's day and the last time the owner was seen live here too.
Counters only: no text of any act is stored.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LEVELS = ("off", "quiet", "offer", "say", "auto")
RANK = {"off": 0, "quiet": 1, "offer": 2, "say": 3, "auto": 3}
#: Accepted this many times in a row: one step up.
PROMOTE_AFTER = 3
#: This many "no" points (ignored 1, dismissed 2) with no yes: one step down.
DEMOTE_AT = 3
OUTCOMES = ("accepted", "ignored", "dismissed")


def step_up(level: str, top: str) -> str:
    """One level up towards *top* (never past it)."""
    if RANK.get(level, 0) >= RANK.get(top, 0):
        return level
    if level == "off":
        return "quiet"
    if level == "quiet":
        return "offer"
    return top


def step_down(level: str) -> str:
    return {"say": "offer", "auto": "offer", "offer": "quiet", "quiet": "off"}.get(level, "off")


class LearningStore:
    """Thread-safe counters and switches, saved on every change (rare)."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.enabled: bool | None = None
        self.calendar = False
        self.morning_day = ""
        self.last_active: float | None = None
        self.kinds: dict[str, dict[str, Any]] = {}
        self._load()

    # -- persistence ----------------------------------------------------------
    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            log.warning("initiative: ignoring unreadable %s", self.path)
            return
        if not isinstance(data, dict):
            return
        if isinstance(data.get("enabled"), bool):
            self.enabled = data["enabled"]
        self.calendar = bool(data.get("calendar", False))
        self.morning_day = str(data.get("morning_day") or "")
        last = data.get("last_active")
        self.last_active = float(last) if isinstance(last, (int, float)) else None
        kinds = data.get("kinds")
        if isinstance(kinds, dict):
            for name, row in kinds.items():
                if isinstance(row, dict):
                    self.kinds[str(name)] = self._clean(row)

    @staticmethod
    def _clean(row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in ("shown", *OUTCOMES, "streak_yes", "streak_no"):
            try:
                out[key] = max(0, int(row.get(key, 0)))
            except (TypeError, ValueError):
                out[key] = 0
        out["level"] = row.get("level") if row.get("level") in LEVELS else None
        out["owner_off"] = bool(row.get("owner_off", False))
        return out

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            data = {"enabled": self.enabled, "calendar": self.calendar,
                    "morning_day": self.morning_day, "last_active": self.last_active,
                    "kinds": json.loads(json.dumps(self.kinds))}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("initiative: could not save %s: %s", self.path, exc)

    # -- levels ---------------------------------------------------------------
    def row(self, kind: str) -> dict[str, Any]:
        with self._lock:
            r = self.kinds.get(kind)
            if r is None:
                r = self.kinds[kind] = self._clean({})
            return r

    def level(self, kind: str, default: str) -> str:
        """The level *kind* acts at now: the owner's off, else what was learned."""
        with self._lock:
            r = self.kinds.get(kind)
            if r is None:
                return default
            if r["owner_off"]:
                return "off"
            return r["level"] or default

    def set_owner(self, kind: str, on: bool, default: str) -> None:
        """The menu or the voice switched *kind*. On also forgets a learned off."""
        with self._lock:
            r = self.row(kind)
            r["owner_off"] = not on
            if on and RANK.get(r["level"] or default, 0) < RANK.get(default, 0):
                r["level"] = None  # learned down below where it starts: start over
            r["streak_no"] = r["streak_yes"] = 0
        self.save()

    def shown(self, kind: str) -> None:
        with self._lock:
            self.row(kind)["shown"] += 1
        self.save()

    def record(self, kind: str, outcome: str, *, default: str, top: str) -> str | None:
        """Count *outcome* for *kind* and maybe move its level. Returns the new
        level when it moved, else None."""
        if outcome not in OUTCOMES:
            return None
        moved = None
        with self._lock:
            r = self.row(kind)
            r[outcome] += 1
            current = r["level"] or default
            if outcome == "accepted":
                r["streak_yes"] += 1
                r["streak_no"] = 0
                if r["streak_yes"] >= PROMOTE_AFTER:
                    r["streak_yes"] = 0
                    up = step_up(current, top)
                    if up != current:
                        r["level"] = moved = up
            else:
                r["streak_yes"] = 0
                r["streak_no"] += 2 if outcome == "dismissed" else 1
                if r["streak_no"] >= DEMOTE_AT:
                    r["streak_no"] = 0
                    down = step_down(current)
                    if down != current:
                        r["level"] = moved = down
        if moved:
            log.info("initiative: %s now %s (learned)", kind, moved)
        self.save()
        return moved

    def forget(self) -> None:
        with self._lock:
            self.kinds.clear()
        self.save()
