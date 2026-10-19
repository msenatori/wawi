"""Today's calendar events for the morning summary, from macOS Calendar (EventKit).

Off by default. Only the menu's "Agenda en el resumen" turns it on, and only
then is Calendar access requested (the macOS prompt). Needs the optional
``pyobjc-framework-EventKit``; without it, or without permission, the
summary simply has no "Agenda" section. Read locally, never sent anywhere.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

log = logging.getLogger(__name__)

#: EKAuthorizationStatus: 3 = authorized / full access (macOS 14+).
_ALLOWED = (3,)


def available() -> bool:
    try:
        import EventKit  # type: ignore[import]  # noqa: F401
    except Exception:
        return False
    return True


def request_access(done: Callable[[bool], None] | None = None) -> bool:
    """Ask macOS for Calendar access (the prompt shows once). False when
    EventKit is not installed. *done(granted)* is called on some thread."""
    try:
        import EventKit  # type: ignore[import]
    except Exception:
        log.info("initiative: no EventKit (pip install pyobjc-framework-EventKit); "
                 "the summary goes without the calendar")
        return False
    store = EventKit.EKEventStore.alloc().init()

    def handler(granted, _error):  # noqa: ANN001
        log.info("initiative: calendar access %s", "granted" if granted else "denied")
        if done is not None:
            done(bool(granted))

    try:
        if hasattr(store, "requestFullAccessToEventsWithCompletion_"):
            store.requestFullAccessToEventsWithCompletion_(handler)
        else:
            store.requestAccessToEntityType_completion_(EventKit.EKEntityTypeEvent, handler)
    except Exception:
        log.warning("initiative: could not ask for calendar access", exc_info=True)
        return False
    return True


def today_events(now: float, *, limit: int = 8) -> list[tuple[str, str]] | None:
    """[(HH:MM or "todo el día", title)] for today, or None when there is no
    calendar to read (no EventKit, no permission)."""
    try:
        import EventKit  # type: ignore[import]
        from Foundation import NSDate  # type: ignore[import]
    except Exception:
        return None
    try:
        status = int(EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent))
        if status not in _ALLOWED:
            return None
        store = EventKit.EKEventStore.alloc().init()
        day = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
        start = NSDate.dateWithTimeIntervalSince1970_(day.timestamp())
        end = NSDate.dateWithTimeIntervalSince1970_((day + timedelta(days=1)).timestamp())
        pred = store.predicateForEventsWithStartDate_endDate_calendars_(start, end, None)
        events = sorted(store.eventsMatchingPredicate_(pred) or [],
                        key=lambda e: e.startDate().timeIntervalSince1970())
        out = []
        for ev in events[:limit]:
            if ev.isAllDay():
                when = "todo el día"
            else:
                when = datetime.fromtimestamp(ev.startDate().timeIntervalSince1970()).strftime("%H:%M")
            out.append((when, str(ev.title() or "(sin título)")))
        return out
    except Exception:
        log.debug("initiative: calendar read failed", exc_info=True)
        return None
