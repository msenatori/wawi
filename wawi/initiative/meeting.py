"""Is the owner in a meeting right now? Local, cheap, and never nosy.

Three signals, each only looked at when it can possibly be true:

* **Google Meet in a browser.** Only browsers that are ALREADY running are
  asked (``NSWorkspace`` first, then one ``osascript`` per running browser,
  guarded by ``application id … is running`` so a browser that quit in
  between is never launched). The script hands back only the URLs that
  contain ``meet.google.com/``; Python keeps the COUNT of those that look
  like a call (``xxx-xxxx-xxx``) and drops the strings at once. No URL, no
  meeting code and no tab title is ever logged or stored.
* **Zoom in a call.** ``zoom.us`` running AND its ``CptHost`` process, which
  only exists during a call (one ``ps``, only while Zoom runs).
* **A meeting app in front** (FaceTime, Teams, Webex): the window owner name
  from Quartz, no title -- the same signal the reflexes already use.

Teams and Slack huddles in the background have no cheap local tell, so they
count only while in front. The camera ("in use") would be a fourth signal,
but needs CoreMediaIO, which Wawi does not ship.

The state is sticky: it starts on the first positive look and ends only
after ``exit_after_s`` without any signal, so a tab switch or a slow
``osascript`` never flaps it. :func:`in_meeting` is the process-wide flag
every speaker consults (reflexes' quiet mode, the reaction voice, the
curiosities, the dance, her barks); it is only raised while the owner keeps
the "callarme en reuniones" initiative on.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from wawi.initiative.core import Act, Ctx, Moment, register

log = logging.getLogger(__name__)

#: Browsers that can be asked for their tab URLs, by bundle id. All of them
#: answer the same AppleScript (windows -> tabs -> URL).
BROWSERS: dict[str, str] = {
    "com.google.Chrome": "Chrome",
    "com.google.Chrome.beta": "Chrome Beta",
    "com.apple.Safari": "Safari",
    "company.thebrowser.Browser": "Arc",
    "com.brave.Browser": "Brave",
    "com.microsoft.edgemac": "Edge",
}
ZOOM_BUNDLE = "us.zoom.xos"
#: Zoom processes that only exist during a call.
ZOOM_CALL_PROCESSES = frozenset({"CptHost"})
#: Front-window owner names that mean a call (lowercase).
FRONT_APPS = frozenset({"facetime", "microsoft teams", "microsoft teams (work or school)",
                        "webex", "cisco webex meetings", "slack huddle"})
#: A Meet call, not the landing page: meet.google.com/abc-defg-hij.
MEET_CALL = re.compile(r"^https://meet\.google\.com/[a-z]{3,4}-[a-z]{4}-[a-z]{3,4}(?:[/?#]|$)")
#: A browser that refused (no Automation permission) is not asked again for this long.
REFUSED_BACKOFF_S = 1800.0
OSASCRIPT_TIMEOUT_S = 4.0

# Only the URLs that mention Meet leave the browser; one per line. Only the
# tab in FRONT of each window: a call left open in a background tab (three
# were, live) is not a meeting, and kept her quiet for hours. Asked as one
# list (a nested "repeat with … in (URL of every tab …)" fails with -1700).
_TABS_SCRIPT = """
if application id "{bundle}" is running then
  tell application id "{bundle}"
    set found to {{}}
    set fronts to URL of {tab} of every window
    repeat with u in fronts
      set s to u as text
      if s contains "meet.google.com/" then set end of found to s
    end repeat
  end tell
  set AppleScript's text item delimiters to linefeed
  return found as text
end if
return ""
"""

# --------------------------------------------------------------------------- #
# The process-wide flag
# --------------------------------------------------------------------------- #

_IN_MEETING = threading.Event()


def in_meeting() -> bool:
    """True while the owner is in a meeting (and lets her go quiet for it)."""
    return _IN_MEETING.is_set()


def set_in_meeting(on: bool) -> None:
    if on:
        _IN_MEETING.set()
    else:
        _IN_MEETING.clear()


# --------------------------------------------------------------------------- #
# Probes (macOS); each one degrades to "nothing seen", never raises
# --------------------------------------------------------------------------- #


def running_bundles() -> set[str]:
    """Bundle ids of the running apps (NSWorkspace; cheap, no permission)."""
    try:
        from AppKit import NSWorkspace  # type: ignore[import]

        apps = NSWorkspace.sharedWorkspace().runningApplications() or []
        return {str(a.bundleIdentifier()) for a in apps if a.bundleIdentifier()}
    except Exception:
        return set()


def osascript(script: str, timeout: float = OSASCRIPT_TIMEOUT_S) -> tuple[int, str, str]:
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                         timeout=timeout, check=False)
    return out.returncode, out.stdout, out.stderr


def _processes() -> set[str]:
    from wawi.sensors.services import list_processes

    return list_processes()


def _front_app() -> str | None:
    from wawi.reflexes import frontmost_window

    front = frontmost_window()
    return front[0] if front else None


def count_meet_calls(text: str) -> int:
    """How many lines of *text* are Meet calls. The URLs go no further."""
    return sum(1 for line in (text or "").splitlines() if MEET_CALL.match(line.strip()))


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MeetingEnd:
    """A meeting that just ended: when, and how long it lasted (seconds)."""

    at: float
    lasted_s: float
    source: str


class MeetingDetector:
    """Sticky meeting state from the three signals. ``check`` every ~25 s."""

    def __init__(
        self,
        *,
        browsers: bool = True,
        zoom: bool = True,
        front_apps: bool = True,
        exit_after_s: float = 60.0,
        bundles: Callable[[], set[str]] = running_bundles,
        script: Callable[[str], tuple[int, str, str]] = osascript,
        processes: Callable[[], set[str]] = _processes,
        front: Callable[[], str | None] = _front_app,
    ) -> None:
        self.browsers = bool(browsers)
        self.zoom = bool(zoom)
        self.front_apps = bool(front_apps)
        self.exit_after_s = float(exit_after_s)
        self._bundles = bundles
        self._script = script
        self._processes = processes
        self._front = front
        self.active = False
        self.source = ""
        self.started_at: float | None = None
        self._last_seen = 0.0
        self._refused: dict[str, float] = {}
        self.ended: MeetingEnd | None = None

    def signals(self, now: float) -> list[str]:
        """Which signals say "meeting" right now (cheapest first)."""
        found: list[str] = []
        if self.front_apps:
            try:
                name = self._front()
            except Exception:
                name = None
            if name and name.strip().lower() in FRONT_APPS:
                found.append("app")
        bundles = self._bundles() if (self.zoom or self.browsers) else set()
        if self.zoom and ZOOM_BUNDLE in bundles:
            try:
                if ZOOM_CALL_PROCESSES & self._processes():
                    found.append("zoom")
            except Exception:
                log.debug("initiative: process list failed", exc_info=True)
        if self.browsers and not found:
            for bundle, name in BROWSERS.items():
                if bundle not in bundles or now < self._refused.get(bundle, 0.0):
                    continue
                if self._meet_in(bundle, name, now):
                    found.append("meet")
                    break
        return found

    def _meet_in(self, bundle: str, name: str, now: float) -> bool:
        try:
            tab = "current tab" if bundle == "com.apple.Safari" else "active tab"
            code, out, err = self._script(_TABS_SCRIPT.format(bundle=bundle, tab=tab))
        except Exception as exc:  # timeout, no osascript
            log.debug("initiative: asking %s for tabs failed: %s", name, exc)
            return False
        if code != 0:
            # Most often -1743: the owner has not allowed Wawi to ask this
            # browser (System Settings > Privacy > Automation). Once per backoff.
            log.info("initiative: %s did not answer the Meet check (%s); retry in %d min",
                     name, (err or "").strip()[-60:] or code, REFUSED_BACKOFF_S // 60)
            self._refused[bundle] = now + REFUSED_BACKOFF_S
            return False
        return count_meet_calls(out) > 0

    def check(self, now: float | None = None) -> bool:
        """Look once and update the sticky state. Returns :attr:`active`."""
        now = time.time() if now is None else now
        found = self.signals(now)
        if found:
            self._last_seen = now
            if not self.active:
                self.active, self.source, self.started_at = True, found[0], now
                log.info("initiative: en reunión (%s): me callo", found[0])
        elif self.active and now - self._last_seen >= self.exit_after_s:
            started = self.started_at if self.started_at is not None else self._last_seen
            self.ended = MeetingEnd(now, max(0.0, self._last_seen - started), self.source)
            self.active, self.started_at = False, None
            log.info("initiative: terminó la reunión (%.0f min)", self.ended.lasted_s / 60.0)
        return self.active

    def take_end(self) -> MeetingEnd | None:
        """The meeting that ended since the last call, once."""
        end, self.ended = self.ended, None
        return end


# --------------------------------------------------------------------------- #
# The watcher: quiet mode while it lasts, "¿anoto lo que quedó?" after
# --------------------------------------------------------------------------- #

#: The offer after a meeting.
NOTES_TEXT = "¿Anoto lo que quedó de la reunión?"
NOTES_LINK = "Tomar nota…"


class MeetingWatcher:
    kinds = ("meeting_quiet", "meeting_notes")

    def __init__(self, engine: Any, detector: MeetingDetector) -> None:
        self.engine = engine
        self.detector = detector

    def sense(self, now: float) -> None:
        """Before every look: is the owner in a meeting? Raise or drop the
        process-wide quiet flag (only while "meeting_quiet" is on)."""
        if all(self.engine.level(k) == "off" for k in self.kinds):
            self.engine.meeting_active = False
            set_in_meeting(False)
            return
        active = self.detector.check(now)
        self.engine.meeting_active = active
        set_in_meeting(active and self.engine.level("meeting_quiet") != "off")

    def poll(self, ctx: Ctx) -> list[Moment]:
        end = self.detector.take_end()
        if end is None or end.lasted_s < self.engine.settings.meeting_min_min * 60.0:
            return []
        return [Moment("meeting_notes", f"meeting_notes:{end.at:.0f}",
                       {"lasted_min": round(end.lasted_s / 60.0)})]

    def act(self, moment: Moment, ctx: Ctx) -> Act | None:
        hooks = self.engine.hooks
        if hooks.start_note is None:
            return None
        # Ten minutes: after that the meeting is not fresh any more.
        return Act("meeting_notes", NOTES_TEXT, link=NOTES_LINK, accept=hooks.start_note,
                   done_if=hooks.notes_active, ttl_s=600.0, show_s=45.0, key=moment.key)


@register("meeting", order=10)
def _meeting_watcher(engine: Any) -> MeetingWatcher:
    s = engine.settings
    return MeetingWatcher(engine, MeetingDetector(
        browsers=s.meeting_browsers, zoom=s.meeting_zoom, front_apps=s.meeting_front_apps,
        exit_after_s=s.meeting_exit_s))
