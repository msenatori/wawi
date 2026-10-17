"""Time of day and how long the user has been at it.

Day parts (``clock.morning``, ``clock.lunch``, ``clock.evening``,
``clock.late_night``) fire once when the clock crosses them, and only while
someone is at the Mac: a "¡buen día!" to an empty room is noise. If the user
is away at the crossing, the greeting waits for them up to ``grace_min``.

``break.suggest`` fires after ``long_session_h`` of continuous activity. A
"break" is ``break_idle_min`` without any input. Idle time comes from
``CGEventSourceSecondsSinceLastEventType`` — a single number of seconds since
the last input of any kind. It never sees which keys or what was typed, and
needs no Accessibility or Input Monitoring permission.
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dtime, timedelta
from typing import Any, Callable

from wawi.reactions import Event
from wawi.sensors.base import Sensor

log = logging.getLogger(__name__)

DEFAULT_DAYPARTS = {
    "morning": "08:00",
    "lunch": "13:00",
    "evening": "19:00",
    "late_night": "00:30",
}


def seconds_idle() -> float | None:
    """Seconds since the last keyboard/mouse/trackpad event, or None."""
    try:
        import Quartz  # type: ignore[import]

        return float(Quartz.CGEventSourceSecondsSinceLastEventType(
            Quartz.kCGEventSourceStateCombinedSessionState, Quartz.kCGAnyInputEventType
        ))
    except Exception:
        return None


def _parse_hhmm(text: Any) -> dtime | None:
    try:
        hh, mm = str(text).strip().split(":", 1)
        return dtime(int(hh), int(mm))
    except (ValueError, TypeError):
        return None


def _duration_es(seconds: float) -> str:
    minutes = int(seconds // 60)
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h} h {m} min"
    if h:
        return "1 hora" if h == 1 else f"{h} horas"
    return f"{m} min"


class ClockSensor(Sensor):
    name = "clock"
    default_interval = 60.0

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        now: Callable[[], datetime] = datetime.now,
        idle: Callable[[], float | None] = seconds_idle,
    ) -> None:
        super().__init__(config)
        c = self.config
        self._now = now
        self._idle = idle
        raw = c.get("dayparts", DEFAULT_DAYPARTS)
        if raw is False or raw is None:
            raw = {}
        self.dayparts = {k: t for k, t in ((str(k), _parse_hhmm(v)) for k, v in dict(raw).items()) if t}
        self.grace = timedelta(minutes=float(c.get("grace_min", 90)))
        self.active_idle_s = float(c.get("active_idle_min", 10)) * 60
        self.long_session_s = float(c.get("long_session_h", 2)) * 3600
        self.break_idle_s = float(c.get("break_idle_min", 5)) * 60
        self.remind_s = float(c.get("remind_every_min", 60)) * 60
        self._prev: datetime | None = None
        self._pending: dict[str, datetime] = {}  # daypart -> when it was due
        self._active_since: datetime | None = None
        self._last_suggest: datetime | None = None
        self.session_s: float = 0.0

    def poll(self) -> list[Event]:
        now = self._now()
        idle = self._idle()
        events: list[Event] = []
        active = idle is None or idle < self.active_idle_s

        # -- day parts -------------------------------------------------------
        if self._prev is not None:
            for name, at in self.dayparts.items():
                due = datetime.combine(now.date(), at)
                if self._prev < due <= now:
                    self._pending[name] = due
        for name, due in list(self._pending.items()):
            if now - due > self.grace:
                del self._pending[name]  # they never came back; skip it today
            elif active:
                del self._pending[name]
                events.append(Event(
                    f"clock.{name}", "info", f"clock:{name}:{due.date().isoformat()}",
                    name, {"part": name, "hour": due.strftime("%H:%M")},
                ))
        self._prev = now

        # -- long session ----------------------------------------------------
        if idle is not None:
            if idle >= self.break_idle_s:
                # A real break: the session restarts when they come back.
                self._active_since = None
                self._last_suggest = None
                self.session_s = 0.0
            else:
                if self._active_since is None:
                    self._active_since = now - timedelta(seconds=idle)
                self.session_s = (now - self._active_since).total_seconds()
                if self.session_s >= self.long_session_s and (
                    self._last_suggest is None
                    or (now - self._last_suggest).total_seconds() >= self.remind_s
                ):
                    self._last_suggest = now
                    events.append(Event(
                        "break.suggest", "notice", "break",
                        "¿Una pausa?",
                        {"duration": _duration_es(self.session_s), "minutes": int(self.session_s // 60)},
                    ))
        return events

    def summary(self) -> str | None:
        if self.session_s >= 3600:
            return f"llevás {_duration_es(self.session_s)} sin pausa"
        return None
