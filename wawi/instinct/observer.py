"""The activity observer: how the owner's day goes, in numbers only.

Every ~30 s one :class:`Sample` is taken -- and this is ALL it contains:

* which app is in front: its bundle id and name (Quartz's window list gives
  the owner process of the front window; ``NSRunningApplication`` its bundle
  id). Window titles are never requested, nothing on screen is read;
* seconds since the last key press and since the last mouse event
  (``CGEventSourceSecondsSinceLastEventType`` -- a single number, it cannot
  say which key);
* whether the screen is locked or the display asleep.

Samples are folded at once into :class:`ActivityStore`, a set of compact
per-day aggregates (seconds active per hour, seconds per app category per
hour, app switches per hour, session starts and ends, breaks, focus
blocks). The sample itself is dropped: there is no raw log. Days older than
``keep_days`` (8 weeks by default) are forgotten. The bundle id is mapped to
a category on the spot and never stored.

Pure apart from :func:`probe`: the clock comes in with every sample, so the
tests run whole synthetic weeks in milliseconds.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from wawi.instinct.categories import FOCUS, categorize

log = logging.getLogger(__name__)

#: A "day" starts at 04:00, so a session that runs past midnight belongs to
#: the day it started in, and "the start of the day" is the morning.
DAY_SHIFT_H = 4
#: Away at least this long = a break (seconds).
BREAK_S = 5 * 60.0
#: Away at least this long = the next activity starts a new session.
SESSION_GAP_S = 3 * 3600.0
#: A focus block counts from this long (seconds).
FOCUS_MIN_S = 25 * 60.0
#: Time outside focus apps that ends a focus block (seconds).
FOCUS_BREAK_S = 3 * 60.0
#: At most this many categories per day: the rest go to "other".
MAX_CATS = 16
#: At most this many breaks / sessions / focus blocks per day.
MAX_ITEMS = 40
#: Window for the app-switching rate (seconds).
SWITCH_WINDOW_S = 10 * 60.0
SAVE_EVERY_S = 300.0


@dataclass(frozen=True)
class Sample:
    """One look at the machine. Nothing else about the owner exists here."""

    wall: float
    bundle: str | None = None
    app: str | None = None
    key_idle: float = 1e9
    mouse_idle: float = 1e9
    locked: bool = False


def probe() -> Sample:
    """A live sample (macOS). Missing pieces read as "unknown", never raise."""
    wall = time.time()
    bundle = app = None
    key_idle = mouse_idle = 1e9
    locked = False
    try:
        import Quartz  # type: ignore[import]

        src = Quartz.kCGEventSourceStateCombinedSessionState
        since = Quartz.CGEventSourceSecondsSinceLastEventType
        key_idle = float(since(src, Quartz.kCGEventKeyDown))
        mouse_idle = min(
            float(since(src, Quartz.kCGEventMouseMoved)),
            float(since(src, Quartz.kCGEventLeftMouseDown)),
            float(since(src, Quartz.kCGEventScrollWheel)),
        )
        session = Quartz.CGSessionCopyCurrentDictionary() or {}
        locked = bool(session.get("CGSSessionScreenIsLocked", False))
        try:
            locked = locked or bool(Quartz.CGDisplayIsAsleep(Quartz.CGMainDisplayID()))
        except Exception:
            pass
        # Owner name and pid of the front normal window: thread-safe, and no
        # Screen Recording permission (titles would need it; never asked).
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        ) or []
        front = next((w for w in windows if w.get("kCGWindowLayer") == 0), None)
        if front is not None:
            app = str(front.get("kCGWindowOwnerName") or "") or None
            pid = front.get("kCGWindowOwnerPID")
            if pid:
                try:
                    from AppKit import NSRunningApplication  # type: ignore[import]

                    running = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
                    if running is not None:
                        bundle = str(running.bundleIdentifier() or "") or None
                except Exception:
                    pass
    except Exception:
        log.debug("instinct: probe failed", exc_info=True)
    return Sample(wall, bundle, app, key_idle, mouse_idle, locked)


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #


def day_key(wall: float) -> str:
    lt = time.localtime(wall - DAY_SHIFT_H * 3600)
    return f"{lt.tm_year:04d}-{lt.tm_mon:02d}-{lt.tm_mday:02d}"


def day_minute(wall: float) -> int:
    """Minutes since the day started (04:00), 0..1439."""
    lt = time.localtime(wall)
    return (lt.tm_hour * 60 + lt.tm_min - DAY_SHIFT_H * 60) % 1440


def clock_of(dm: float) -> tuple[int, int]:
    """A day minute back to (hour, minute) on the clock."""
    m = int(round(dm + DAY_SHIFT_H * 60)) % 1440
    return m // 60, m % 60


def hhmm(dm: float) -> str:
    h, m = clock_of(dm)
    return f"{h}:{m:02d}"


def _zeros() -> list[int]:
    return [0] * 24


def new_day(wall: float) -> dict[str, Any]:
    return {"wd": time.localtime(wall - DAY_SHIFT_H * 3600).tm_wday,
            "first": None, "last": None,
            "act": _zeros(), "idle": _zeros(), "keys": _zeros(), "mouse": _zeros(),
            "sw": _zeros(), "cats": {}, "breaks": [], "sessions": [], "focus": [],
            "voice": 0, "touch": 0}


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


class ActivityStore:
    """Rolling per-day aggregates, persisted as one small JSON file."""

    def __init__(self, path: Path | None = None, keep_days: int = 56) -> None:
        self.path = path
        self.keep_days = int(keep_days)
        self.days: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.dirty = False

    def day(self, wall: float) -> dict[str, Any]:
        key = day_key(wall)
        with self.lock:
            d = self.days.get(key)
            if d is None:
                d = self.days[key] = new_day(wall)
                self.rotate(wall)
            return d

    def rotate(self, wall: float) -> None:
        """Forget days older than ``keep_days``."""
        y, m, d = (int(x) for x in day_key(wall).split("-"))
        oldest = (date(y, m, d) - timedelta(days=self.keep_days - 1)).isoformat()
        with self.lock:
            for key in [k for k in self.days if k < oldest]:
                del self.days[key]
                self.dirty = True

    def recent(self, n: int | None = None) -> list[tuple[str, dict[str, Any]]]:
        with self.lock:
            items = sorted(self.days.items())
        return items[-n:] if n else items

    def wipe(self) -> None:
        with self.lock:
            self.days.clear()
            self.dirty = False
        if self.path is not None:
            for p in (self.path, self.path.with_name(self.path.name + ".tmp")):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    log.warning("instinct: could not delete %s", p, exc_info=True)

    # -- persistence -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {"version": 1, "days": json.loads(json.dumps(self.days))}

    def save(self) -> None:
        if self.path is None:
            return
        data = self.to_dict()
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                           encoding="utf-8")
            os.replace(tmp, self.path)
            self.dirty = False
        except OSError:
            log.warning("instinct: could not save %s", self.path, exc_info=True)

    def load(self, wall: float | None = None) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("instinct: %s unreadable; starting fresh", self.path)
            return
        days = raw.get("days") if isinstance(raw, dict) else None
        if not isinstance(days, dict):
            return
        clean: dict[str, dict[str, Any]] = {}
        for key, d in days.items():
            if isinstance(d, dict) and len(str(key)) == 10 and _valid_day(d):
                clean[str(key)] = d
        with self.lock:
            self.days = clean
        self.rotate(wall if wall is not None else time.time())


def _valid_day(d: dict[str, Any]) -> bool:
    base = new_day(0)
    for k, v in base.items():
        if k not in d:
            d[k] = v
        elif isinstance(v, list) and k in ("act", "idle", "keys", "mouse", "sw"):
            if not (isinstance(d[k], list) and len(d[k]) == 24):
                return False
    return isinstance(d.get("cats"), dict)


# --------------------------------------------------------------------------- #
# The observer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Context:
    """The owner right now, as the instinct engine sees it."""

    wall: float
    active: bool = False
    idle_s: float = 0.0
    category: str = "unknown"
    typing: bool = False
    locked: bool = False
    #: Seconds into the current focus block (0 = none going on).
    focus_s: float = 0.0
    #: (ended at, length in seconds) of the last focus block, or None.
    focus_ended: tuple[float, float] | None = None
    #: (came back at, after how long away) the last time they returned.
    returned: tuple[float, float] | None = None
    #: First activity of today (wall), or None.
    day_started: float | None = None
    #: Seconds since the last break (or since the session started).
    since_break_s: float = 0.0
    #: Today's breaks as (began at, seconds away).
    breaks_today: tuple[tuple[float, float], ...] = ()
    #: App switches in the last ten minutes.
    switches: int = 0
    #: Seconds since the owner touched or talked to her.
    since_contact_s: float = float("inf")
    #: Active minutes today.
    active_min_today: float = 0.0


class ActivityObserver:
    """Folds samples into the store and keeps the live picture of "now"."""

    def __init__(
        self,
        store: ActivityStore,
        *,
        sample_s: float = 30.0,
        idle_after_s: float = 120.0,
        overrides: dict[str, str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.sample_s = float(sample_s)
        self.idle_after_s = float(idle_after_s)
        self.overrides = dict(overrides or {})
        self._clock = clock
        self._lock = threading.RLock()
        self._last_wall: float | None = None
        self._last_cat: str | None = None
        self._active = False
        self._last_active: float | None = None
        self._last_key_idle = 1e9
        self._locked = False
        self._session_start: float | None = None
        self._since_break_from: float | None = None
        self._returned: tuple[float, float] | None = None
        self._switches: deque[float] = deque(maxlen=400)
        self._focus_start: float | None = None
        self._focus_last: float | None = None
        self._focus_off = 0.0
        self._focus_cats: dict[str, float] = {}
        self._focus_ended: tuple[float, float] | None = None
        self._breaks: list[tuple[float, float]] = []
        self._contact: float | None = None
        self._day_first: tuple[str, float] | None = None
        self._saved_at = clock()

    # -- hooks -----------------------------------------------------------------
    def note(self, kind: str, wall: float | None = None) -> None:
        """The owner talked to her ("talk") or touched / petted her."""
        wall = self._clock() if wall is None else wall
        with self._lock:
            self._contact = wall
            d = self.store.day(wall)
            if kind == "talk":
                d["voice"] = int(d.get("voice", 0)) + 1
            elif kind in ("touch", "pet"):
                d["touch"] = int(d.get("touch", 0)) + 1
            self.store.dirty = True

    # -- samples -----------------------------------------------------------------
    def ingest(self, s: Sample) -> None:
        with self._lock, self.store.lock:
            self._ingest(s)
        if self.store.dirty and s.wall - self._saved_at >= SAVE_EVERY_S:
            self._saved_at = s.wall
            self.store.save()

    def _ingest(self, s: Sample) -> None:
        wall = s.wall
        dt = self.sample_s if self._last_wall is None else \
            max(0.0, min(wall - self._last_wall, 2 * self.sample_s))
        if self._last_wall is not None and \
                wall - self._last_wall > max(3 * self.sample_s, self.idle_after_s):
            # No samples for a while: Wawi was not running, or the Mac slept.
            # Either way the owner was not seen working; whatever was going on
            # (a focus block, a session) ended at the last sample.
            if self._focus_start is not None:
                self._end_focus(self._last_wall, self.store.day(self._last_wall))
            self._active = False
        day = self.store.day(wall)
        hour = time.localtime(wall).tm_hour
        dm = day_minute(wall)
        idle_input = min(s.key_idle, s.mouse_idle)
        active = not s.locked and idle_input < self.idle_after_s
        cat = categorize(s.bundle, s.app, self.overrides) if active else "idle"
        self._locked = s.locked
        self._last_key_idle = s.key_idle

        if active:
            day["act"][hour] += int(round(dt))
            cats = day["cats"]
            if cat not in cats and len(cats) >= MAX_CATS:
                cat_key = "other"
            else:
                cat_key = cat
            cats.setdefault(cat_key, _zeros())[hour] += int(round(dt))
            # Input "intensity": how many samples saw a key / the mouse since
            # the previous one. Counts, never content.
            if s.key_idle < max(dt, 1.0):
                day["keys"][hour] += 1
            if s.mouse_idle < max(dt, 1.0):
                day["mouse"][hour] += 1
            if self._last_cat not in (None, "idle") and cat != self._last_cat:
                day["sw"][hour] += 1
                self._switches.append(wall)
            if not self._active:
                self._came_back(wall, day, dm)
            if day["first"] is None:
                day["first"] = dm
                self._day_first = (day_key(wall), wall)
            day["last"] = dm
            if day["sessions"]:
                day["sessions"][-1][1] = dm
            self._last_active = wall
        else:
            day["idle"][hour] += int(round(dt))
        self._focus(wall, active, cat, dt, day)
        self._active = active
        self._last_cat = cat
        self._last_wall = wall
        self.store.dirty = True

    def _came_back(self, wall: float, day: dict[str, Any], dm: int) -> None:
        last = self._last_active
        gap = float("inf") if last is None else wall - last
        new_day = last is not None and day_key(last) != day_key(wall)
        if last is None or new_day or gap >= SESSION_GAP_S:
            if len(day["sessions"]) < MAX_ITEMS:
                day["sessions"].append([dm, dm])
            self._session_start = wall
            self._since_break_from = wall
            self._breaks = [b for b in self._breaks if day_key(b[0]) == day_key(wall)]
        elif gap >= BREAK_S:
            if len(day["breaks"]) < MAX_ITEMS:
                day["breaks"].append([day_minute(last), int(round(gap / 60.0))])
            self._since_break_from = wall
            self._breaks.append((last, gap))
            del self._breaks[:-MAX_ITEMS]
        if last is not None and gap >= BREAK_S:
            self._returned = (wall, gap)
        elif last is None:
            self._returned = (wall, float("inf"))

    def _focus(self, wall: float, active: bool, cat: str, dt: float,
               day: dict[str, Any]) -> None:
        if active and cat in FOCUS:
            if self._focus_start is None:
                self._focus_start = wall
                self._focus_off = 0.0
                self._focus_cats = {}
            self._focus_last = wall
            self._focus_cats[cat] = self._focus_cats.get(cat, 0.0) + dt
            return
        if self._focus_start is None:
            return
        if active:
            self._focus_off += dt
            if self._focus_off < FOCUS_BREAK_S:
                return
        elif self._focus_last is not None and wall - self._focus_last < BREAK_S:
            return
        self._end_focus(wall, day)

    def _end_focus(self, wall: float, day: dict[str, Any]) -> None:
        start, last = self._focus_start, self._focus_last or self._focus_start
        self._focus_start = None
        if start is None or last is None:
            return
        length = last - start
        if length >= FOCUS_MIN_S:
            main = max(self._focus_cats.items(), key=lambda kv: kv[1])[0] \
                if self._focus_cats else "code"
            if len(day["focus"]) < MAX_ITEMS:
                day["focus"].append([day_minute(start), int(round(length / 60.0)), main])
            self._focus_ended = (wall, length)

    # -- the live picture ------------------------------------------------------------
    def context(self, wall: float | None = None) -> Context:
        wall = self._clock() if wall is None else wall
        with self._lock:
            day = self.store.days.get(day_key(wall))
            active_min = 0.0 if day is None else sum(day["act"]) / 60.0
            first = None
            if self._day_first is not None and self._day_first[0] == day_key(wall):
                first = self._day_first[1]
            idle_s = 0.0 if self._last_active is None else max(0.0, wall - self._last_active)
            switches = sum(1 for t in self._switches if wall - t <= SWITCH_WINDOW_S)
            since_break = 0.0 if self._since_break_from is None else wall - self._since_break_from
            today = day_key(wall)
            return Context(
                wall=wall,
                active=self._active,
                idle_s=idle_s if not self._active else 0.0,
                category=self._last_cat or "unknown",
                typing=self._last_key_idle < 5.0,
                locked=self._locked,
                focus_s=0.0 if self._focus_start is None or self._focus_last is None
                else self._focus_last - self._focus_start,
                focus_ended=self._focus_ended,
                returned=self._returned,
                day_started=first,
                since_break_s=since_break if self._active else 0.0,
                breaks_today=tuple(b for b in self._breaks if day_key(b[0]) == today),
                switches=switches,
                since_contact_s=float("inf") if self._contact is None else wall - self._contact,
                active_min_today=active_min,
            )

    def wipe(self) -> None:
        """Forget everything, live picture included."""
        with self._lock:
            self.store.wipe()
            self._returned = None
            self._focus_ended = None
            self._breaks = []
            self._switches.clear()
            self._focus_start = None
