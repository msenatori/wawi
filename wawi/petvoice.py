"""Wawi saying a reaction's bubble out loud, outside any conversation.

A reaction with ``speak=True`` ("¡Se cayó example.com!") reaches the body,
which hands the text to :class:`PetVoice`. Saying it is the easy part; the
hard parts are all about NOT saying it at the wrong moment:

- **Never over a turn.** If you are talking to her, she is thinking, or she
  is already answering, the sentence waits in a short queue until the
  pipeline is idle. Past ``stale_s`` it is dropped: news about a site that
  went down two minutes ago is no longer worth interrupting for, and the
  bubble already showed it.
- **Never hearing herself.** The actual speaking is done by the orchestrator
  (``speak_now``) through the same path as an answer: listener paused (or
  barge-in armed under the same rules), echo text set, echo tail cleared.
- **Quiet mode.** Meetings, full-screen apps and quiet hours (the same
  detector the reflexes use) mean silence, and an unknown state counts as
  quiet: a voice out of nowhere in a meeting costs more than a missed line.
- **Rate.** At most ``max_per_hour`` spoken reactions per hour, whatever
  their source (reflexes or the mind).

Everything here is clock-driven and callback-driven so tests run it without
a voice, a microphone or real time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

DEFAULT_PER_HOUR = 4
DEFAULT_STALE_S = 60.0
#: Pending lines kept; older ones are dropped first (they would be stale soon).
MAX_QUEUE = 3


@dataclass
class _Line:
    text: str
    at: float


class PetVoice:
    """Callable handed to ``panel.set_pet_speaker``.

    *speak_now(text) -> bool* says it right now if the pipeline is idle and
    returns True, or returns False without saying anything when a turn is in
    flight (the line is retried). *quiet() -> bool | None* is the quiet-mode
    check (None = unknown = quiet).
    """

    def __init__(
        self,
        speak_now: Callable[[str], bool],
        *,
        quiet: Callable[[], bool | None],
        max_per_hour: int = DEFAULT_PER_HOUR,
        stale_s: float = DEFAULT_STALE_S,
        retry_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        start_thread: bool = True,
    ) -> None:
        self._speak_now = speak_now
        self._quiet = quiet
        self.max_per_hour = max(0, int(max_per_hour))
        self.stale_s = max(1.0, float(stale_s))
        self._retry_s = retry_s
        self._clock = clock
        self._start_thread = start_thread
        self._queue: deque[_Line] = deque()
        self._spoken: deque[float] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- intake (any thread) -------------------------------------------------
    def __call__(self, text: str) -> None:
        self.submit(text)

    def submit(self, text: str) -> bool:
        text = " ".join(str(text or "").split())
        if not text or self.max_per_hour == 0:
            return False
        with self._lock:
            if any(line.text == text for line in self._queue):
                return False  # the same news twice is one sentence
            self._queue.append(_Line(text, self._clock()))
            while len(self._queue) > MAX_QUEUE:
                dropped = self._queue.popleft()
                log.info("Pet voice: dropped %r (queue full)", dropped.text)
        self._wake.set()
        if self._start_thread:
            self._ensure_thread()
        return True

    def pending(self) -> list[str]:
        with self._lock:
            return [line.text for line in self._queue]

    # -- the decision, one step at a time (tests drive this directly) --------
    def step(self) -> str:
        """Try the oldest pending line once.

        Returns what happened: ``"empty"``, ``"stale"``, ``"quiet"``,
        ``"rate"``, ``"busy"`` (kept for a retry) or ``"spoken"``.
        """
        now = self._clock()
        with self._lock:
            while self._queue and now - self._queue[0].at > self.stale_s:
                line = self._queue.popleft()
                log.info("Pet voice: dropped stale %r", line.text)
                if not self._queue:
                    return "stale"
            if not self._queue:
                return "empty"
            line = self._queue[0]
        try:
            quiet = self._quiet()
        except Exception:
            quiet = None
        if quiet is not False:
            self._discard(line)
            log.info("Pet voice: quiet mode, not saying %r", line.text)
            return "quiet"
        if not self._allowed(now):
            self._discard(line)
            log.info("Pet voice: over %d/hour, not saying %r", self.max_per_hour, line.text)
            return "rate"
        try:
            spoke = bool(self._speak_now(line.text))
        except Exception:
            log.warning("Pet voice: speaking failed", exc_info=True)
            spoke = True  # do not retry a line that broke the voice
        if not spoke:
            return "busy"
        self._discard(line)
        with self._lock:
            self._spoken.append(now)
        return "spoken"

    def _discard(self, line: _Line) -> None:
        with self._lock:
            try:
                self._queue.remove(line)
            except ValueError:
                pass

    def _allowed(self, now: float) -> bool:
        with self._lock:
            while self._spoken and now - self._spoken[0] >= 3600.0:
                self._spoken.popleft()
            return len(self._spoken) < self.max_per_hour

    # -- worker ----------------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="wawi-pet-voice", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            result = self.step()
            if result == "busy":
                self._stop.wait(self._retry_s)
            elif result in ("empty",):
                self._wake.wait(30.0)
                self._wake.clear()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


def build(cfg: Any, speak_now: Callable[[str], bool]) -> PetVoice | None:
    """From ``reflexes:`` settings; None when ``speak_reactions: false``."""
    rx = dict(getattr(cfg, "reflexes", None) or {})
    if rx.get("speak_reactions", True) is False:
        return None
    try:
        per_hour = int(rx.get("speak_per_hour", DEFAULT_PER_HOUR))
    except (TypeError, ValueError):
        per_hour = DEFAULT_PER_HOUR
    try:
        stale = float(rx.get("speak_stale_s", DEFAULT_STALE_S))
    except (TypeError, ValueError):
        stale = DEFAULT_STALE_S
    return PetVoice(speak_now, quiet=_quiet_from(rx), max_per_hour=per_hour, stale_s=stale)


def _quiet_from(rx: dict[str, Any]) -> Callable[[], bool | None]:
    """The reflex engine's live answer, or our own detector when none runs."""
    from wawi import reflexes

    own: list[Any] = []

    def check() -> bool | None:
        state = reflexes.quiet_state()
        if state is None:
            if not own:
                own.append(reflexes.QuietDetector(rx))
            state = own[0].check()
        return bool(state.quiet)

    return check
