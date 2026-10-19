"""What Wawi is busy with right now, so the mascot can SHOW it.

When the brain runs a tool (fetching a page, taking a screenshot, writing a
note…) she should visibly be doing it: in the spaceship, behind a camera, at
her notebook. Each tool declares its own animation where it is defined (the
``"animation"`` field of its spec in :mod:`wawi.tools`; see
:func:`wawi.tools.animation_for`), and anything that does not — a connector,
a future tool — gets the generic :data:`DEFAULT`.

Tools run on the brain's thread (the SDK's event loop) and the mascot lives
on the AppKit main thread, so this is a small locked board: the tool side
calls :func:`begin` / :func:`end` (or the :func:`doing` context manager, which
ends even when the tool raises), and the mascot polls :func:`now` every
frame. A tool that is over in 50 ms still shows for :data:`MIN_SHOW` seconds.

Longer work in the background — a delegated task (wawi/task.py), an Orca
worker — sets a flag with :func:`set_busy` for as long as it runs; she shows
it more quietly while she is otherwise idle.

Work she starts ON HER OWN (fixing herself, looking up curiosities) is
quieter still: :func:`musing` (or :func:`set_musing`) only raises a tiny
"…" over her head (wawi/mascot.badge), and she keeps doing whatever she was
doing. It is not "busy": nothing waits for it, and nothing pauses for it.

Listeners (:func:`subscribe`) hear ``("start" | "end", tool, animation)``;
they are called outside the lock and may never break a tool.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

log = logging.getLogger(__name__)

#: A tool shows at least this long, however quick it was.
MIN_SHOW = 1.2
#: The animation of a tool that declares none (see wawi/mascot.WORK_CLIPS).
DEFAULT = "busy"
#: The animation of background work.
BACKGROUND = "background"

Listener = Callable[[str, str, str], None]


class Board:
    """The locked board itself (one per process: :data:`BOARD`; tests make
    their own with a fake *clock*)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 min_show: float = MIN_SHOW) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self.min_show = float(min_show)
        self._next = 0
        #: token → (tool, animation, started)
        self._active: dict[int, tuple[str, str, float]] = {}
        #: (animation, shown until) of tools already over.
        self._linger: list[tuple[str, float]] = []
        #: source → animation, for background work.
        self._background: dict[str, str] = {}
        #: source → how many runs of it are under way, for her own work.
        self._musing: dict[str, int] = {}
        self._listeners: list[Listener] = []

    # -- the tool side (any thread) -------------------------------------------
    def begin(self, tool: str, animation: str | None = None) -> int:
        """A tool started. Returns the token to :meth:`end` it with."""
        anim = animation or DEFAULT
        with self._lock:
            self._next += 1
            token = self._next
            self._active[token] = (str(tool), anim, self._clock())
        self._tell("start", str(tool), anim)
        return token

    def end(self, token: int) -> None:
        """The tool with *token* is over (twice, or an unknown token: nothing)."""
        with self._lock:
            got = self._active.pop(token, None)
            if got is None:
                return
            tool, anim, started = got
            until = started + self.min_show
            if until > self._clock():
                self._linger.append((anim, until))
        self._tell("end", tool, anim)

    @contextmanager
    def doing(self, tool: str, animation: str | None = None) -> Iterator[int]:
        """``with doing("fetch_page", "ship"): ...`` — ends even on errors."""
        token = self.begin(tool, animation)
        try:
            yield token
        finally:
            self.end(token)

    def pulse(self, tool: str, animation: str | None = None,
              seconds: float | None = None) -> None:
        """A tool we only hear START (a connector's, through the SDK hook):
        show it for *seconds* (default :attr:`min_show` x 2)."""
        anim = animation or DEFAULT
        with self._lock:
            self._linger.append((anim, self._clock() + (seconds or 2 * self.min_show)))
        self._tell("start", str(tool), anim)

    def set_busy(self, source: str, on: bool, animation: str | None = None) -> None:
        """Background work by *source* (``"task"``, ``"orca"``…) started or
        stopped. Any thread."""
        with self._lock:
            if on:
                self._background[str(source)] = animation or BACKGROUND
            else:
                self._background.pop(str(source), None)

    def set_musing(self, source: str, on: bool) -> None:
        """Work she started on her own, by *source*, started or stopped.
        Counted, so two overlapping runs of one source end with the last."""
        source = str(source)
        with self._lock:
            n = self._musing.get(source, 0) + (1 if on else -1)
            if n > 0:
                self._musing[source] = n
            else:
                self._musing.pop(source, None)

    @contextmanager
    def musing_while(self, source: str) -> Iterator[None]:
        """``with musing_while("selffix"): ...`` — ends even on errors."""
        self.set_musing(source, True)
        try:
            yield
        finally:
            self.set_musing(source, False)

    def musing(self) -> bool:
        """Is she working on something of her own right now? Any thread."""
        with self._lock:
            return bool(self._musing)

    # -- the mascot side (main thread) ---------------------------------------
    def now(self) -> tuple[str | None, str | None]:
        """``(tool animation, background animation)`` to show right now.

        The tool is the most recently started one still running, else the
        latest one still inside its minimum time. Background work is any
        source still busy (the most recently set)."""
        t = self._clock()
        with self._lock:
            self._linger = [(a, u) for a, u in self._linger if u > t]
            tool = None
            if self._active:
                tool = self._active[max(self._active)][1]
            elif self._linger:
                tool = self._linger[-1][0]
            background = next(reversed(self._background.values()), None) \
                if self._background else None
        return tool, background

    def busy(self) -> bool:
        """Anything at all running (a tool, or background work)."""
        tool, background = self.now()
        return tool is not None or background is not None

    # -- listeners ------------------------------------------------------------
    def subscribe(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _tell(self, event: str, tool: str, animation: str) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(event, tool, animation)
            except Exception:
                log.debug("working listener failed", exc_info=True)

    def clear(self) -> None:
        """Forget everything (tests, shutdown)."""
        with self._lock:
            self._active.clear()
            self._linger.clear()
            self._background.clear()
            self._musing.clear()


BOARD = Board()

begin = BOARD.begin
end = BOARD.end
doing = BOARD.doing
pulse = BOARD.pulse
set_busy = BOARD.set_busy
set_musing = BOARD.set_musing
musing_while = BOARD.musing_while
musing = BOARD.musing
now = BOARD.now
busy = BOARD.busy
subscribe = BOARD.subscribe
unsubscribe = BOARD.unsubscribe
