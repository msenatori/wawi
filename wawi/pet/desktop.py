"""Reading the desktop for her body: window frames, the caret, typing.

The only module in :mod:`wawi.pet` that talks to macOS, and only to READ:

* **Window frames** — ``CGWindowListCopyWindowInfo`` with on-screen-only and
  no desktop elements. Bounds, owner and layer need no Screen Recording
  permission (titles would; we never ask for them).
* **The caret** — Accessibility: the focused element's
  ``AXSelectedTextRange`` and ``AXBoundsForRange``. Needs the Accessibility
  permission the app already uses; without it, None (callers fall back to
  the front window, then the mouse).
* **Typing** — ``CGEventSourceSecondsSinceLastEventType`` for key-down: how
  long since ANY key was pressed. No keystroke is ever read; this is not a
  keylogger and cannot become one.

Window lists and the caret are sampled on a background thread at most
:data:`HZ` times a second, so the 30 fps render tick only ever reads the
latest tuple (an atomic attribute swap under the GIL).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

from wawi.pet import world as W

log = logging.getLogger(__name__)

#: Samples per second, at most.
HZ = 2.0

try:  # pragma: no cover - macOS only
    import Quartz  # type: ignore[import]

    _HAS_QUARTZ = True
except Exception:  # pragma: no cover
    Quartz = None  # type: ignore[assignment]
    _HAS_QUARTZ = False


def primary_height() -> float:
    """Height of the primary display (the one with the menu bar), which is
    what flips Quartz's y-down into AppKit's y-up. Thread-safe."""
    if not _HAS_QUARTZ:
        return 0.0
    try:
        return float(Quartz.CGDisplayBounds(Quartz.CGMainDisplayID()).size.height)
    except Exception:
        return 0.0


def read_windows(own_pid: int | None = None) -> list[W.Win]:
    """On-screen ordinary windows, front to back, in AppKit coordinates."""
    if not _HAS_QUARTZ:
        return []
    try:
        opts = (Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements)
        raw = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
        return W.parse_windows([dict(d) for d in raw],
                               os.getpid() if own_pid is None else own_pid,
                               primary_height())
    except Exception:
        log.debug("Could not read the window list", exc_info=True)
        return []


def key_idle_seconds() -> float:
    """Seconds since the last key-down anywhere (never WHICH key). inf when
    unknown. Cheap: safe to call every frame."""
    if not _HAS_QUARTZ:
        return math.inf
    try:
        return float(Quartz.CGEventSourceSecondsSinceLastEventType(
            Quartz.kCGEventSourceStateCombinedSessionState, Quartz.kCGEventKeyDown))
    except Exception:
        return math.inf


def read_caret() -> W.Rect | None:
    """The text caret of the focused element, in AppKit coordinates, or None.

    Accessibility only; any failure (no permission, an app that does not
    expose ranges, a zero-size rect) is None. Never raises.
    """
    try:  # pragma: no cover - needs a real focused text field
        import ApplicationServices as AS  # type: ignore[import]

        system = AS.AXUIElementCreateSystemWide()
        err, focused = AS.AXUIElementCopyAttributeValue(system, "AXFocusedUIElement", None)
        if err != 0 or focused is None:
            return None
        err, rng = AS.AXUIElementCopyAttributeValue(focused, "AXSelectedTextRange", None)
        if err != 0 or rng is None:
            return None
        rect = _bounds_for(AS, focused, rng)
        if rect is None:
            # An empty range has no bounds in many apps: ask for the
            # character just before the caret instead.
            ok, r = AS.AXValueGetValue(rng, AS.kAXValueCFRangeType, None)
            if ok and r.location > 0:
                prev = AS.AXValueCreate(AS.kAXValueCFRangeType, (r.location - 1, 1))
                rect = _bounds_for(AS, focused, prev)
                if rect is not None:
                    x, y, w, h = rect
                    rect = (x + w, y, 1.0, h)
        if rect is None:
            return None
        x, y, w, h = rect
        if h <= 0 or h > 400:
            return None
        return W.cg_to_appkit((x, y, max(1.0, w), h), primary_height())
    except Exception:
        return None


def _bounds_for(AS: Any, element: Any, rng: Any) -> W.Rect | None:  # pragma: no cover
    err, value = AS.AXUIElementCopyParameterizedAttributeValue(
        element, "AXBoundsForRange", rng, None)
    if err != 0 or value is None:
        return None
    ok, r = AS.AXValueGetValue(value, AS.kAXValueCGRectType, None)
    if not ok:
        return None
    rect = (float(r.origin.x), float(r.origin.y), float(r.size.width), float(r.size.height))
    return rect if rect[3] > 0 else None


class DesktopSampler:
    """Samples window frames (and, when asked, the caret) off the main thread.

    ``snapshot()`` returns ``(version, windows, caret)``; *version* bumps
    whenever the window list changed, so the body only recomputes its ledges
    then. ``want_caret`` is set by the main thread when the caret matters
    (watching you type), so Accessibility is not queried for nothing.
    """

    def __init__(self, hz: float = HZ) -> None:
        self.period = 1.0 / max(0.2, hz)
        self.want_caret = False
        self._snap: tuple[int, tuple[W.Win, ...], W.Rect | None] = (0, (), None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pid = os.getpid()

    @property
    def available(self) -> bool:
        return _HAS_QUARTZ

    def start(self) -> None:
        if self._thread is not None or not _HAS_QUARTZ:
            return
        self._thread = threading.Thread(target=self._loop, name="wawi-desktop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> tuple[int, tuple[W.Win, ...], W.Rect | None]:
        return self._snap

    def _loop(self) -> None:  # pragma: no cover - thread around macOS calls
        version, last = 0, ()
        while not self._stop.wait(self.period):
            try:
                wins = tuple(read_windows(self._pid))
                caret = read_caret() if self.want_caret else None
                if wins != last:
                    version += 1
                    last = wins
                self._snap = (version, wins, caret)
            except Exception:
                log.debug("Desktop sample failed", exc_info=True)
