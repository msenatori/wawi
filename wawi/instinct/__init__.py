"""Wawi's instincts: she watches how your day goes and now and then acts on it.

Four small parts, all local and numpy-free stats (the optional local model
only adds words):

* :mod:`~wawi.instinct.observer` -- every ~30 s, which KIND of app is in
  front and whether you are at the keyboard (timing only, never content),
  folded into per-day aggregates; 8 weeks rolling, in ``instinct_state.json``.
* :mod:`~wawi.instinct.patterns` -- your routines: when you start, lunch,
  break, focus, stop; which app shows up when.
* :mod:`~wawi.instinct.engine` -- every couple of minutes, maybe one small
  act of her own (greet you back, curl up while you focus, bring the ball).
  The existing learner (``wawi.learn``) learns which ones you welcome.
* :mod:`~wawi.instinct.selfmodel` -- what she knows, for the mind's bubbles.

Privacy: app names / bundle ids are turned into a category on the spot
(unknown apps keep their lowercased name), input is "seconds since the last
key / mouse event". No window titles, no screen content, no keystrokes.
Nothing here leaves the machine.

Public surface, safe to call when nothing is installed:

* :func:`build` / :func:`install` / :func:`get`, :meth:`Instinct.start` / ``stop``.
* :func:`note` -- the owner talked to / touched her (the body calls it).
* :func:`self_notes` -- the self-model for the mind (empty when off).
* :func:`patterns_text` / :func:`forget` -- the voice tool.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from wawi.instinct.engine import InstinctEngine
from wawi.instinct.observer import ActivityObserver, ActivityStore, Sample, probe
from wawi.instinct.patterns import PatternMiner
from wawi.instinct.settings import InstinctSettings

log = logging.getLogger(__name__)

__all__ = ["Instinct", "InstinctSettings", "build", "forget", "get", "install", "note",
           "patterns_text", "self_notes"]

_INSTINCT: "Instinct | None" = None


class Instinct:
    """The observer, the miner and the engine, and the thread that runs them."""

    def __init__(
        self,
        settings: InstinctSettings,
        *,
        probe_fn: Callable[[], Sample] = probe,
        clock: Callable[[], float] = time.time,
        engine: InstinctEngine | None = None,
        subscribe: bool = True,
    ) -> None:
        self.settings = settings
        self.store = ActivityStore(settings.state_file, keep_days=settings.keep_days)
        self.observer = ActivityObserver(self.store, sample_s=settings.sample_s,
                                         idle_after_s=settings.idle_after_s,
                                         overrides=settings.categories, clock=clock)
        self.miner = PatternMiner(self.store)
        self.engine = engine or InstinctEngine(settings, observer=self.observer,
                                               miner=self.miner, clock=clock)
        self._probe = probe_fn
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscribe = subscribe
        self._subscribed = False

    def load(self) -> None:
        now = self._clock()
        self.store.load(now)
        self.observer.seed(now)

    def start(self) -> None:
        if self._subscribe and not self._subscribed:
            self.engine.bus.subscribe(self.engine.on_event)
            self._subscribed = True
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="wawi-instinct", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.settings.observe:
            self.store.save()

    def step(self, now: float | None = None) -> str | None:
        """One sample (if observing) and maybe one decision. The thread's body."""
        if self.settings.observe:
            try:
                sample = self._probe()
                self.observer.ingest(sample)
            except Exception:
                log.debug("instinct: sample failed", exc_info=True)
        return self.engine.maybe_decide(now)

    def _loop(self) -> None:
        # A first look right away, then on the sampling clock. Decisions are
        # rate-limited inside the engine.
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                log.warning("instinct: step failed", exc_info=True)
            self._stop.wait(self.settings.sample_s)

    def forget(self) -> None:
        self.observer.wipe()
        self.miner.invalidate()
        self.engine.reset()


def get() -> Instinct | None:
    return _INSTINCT


def install(instinct: Instinct | None) -> None:
    global _INSTINCT
    _INSTINCT = instinct


def build(cfg: Any) -> Instinct | None:
    """A loaded (not started) Instinct for *cfg*'s ``instinct:`` section, or
    None when it is off. Never raises."""
    try:
        settings = InstinctSettings.from_dict(getattr(cfg, "instinct", None) or {})
        if not settings.enabled:
            return None
        inst = Instinct(settings)
        if settings.observe:
            inst.load()
        log.info("instinct: on (observe %s, %d instincts)", settings.observe,
                 len(settings.instincts))
        return inst
    except Exception:
        log.warning("instinct: could not start; Wawi runs without instincts", exc_info=True)
        return None


def note(kind: str) -> None:
    """The owner talked to her ("talk") or touched / petted her."""
    inst = _INSTINCT
    if inst is None or not inst.settings.observe:
        return
    try:
        inst.observer.note(kind)
    except Exception:
        log.debug("instinct: note failed", exc_info=True)


def self_notes(now: float | None = None) -> dict[str, Any]:
    """The self-model for the mind: {"notes", "facts", "lines"} (empty when off)."""
    inst = _INSTINCT
    if inst is None or not inst.settings.observe:
        return {"notes": [], "facts": [], "lines": []}
    from wawi.instinct.selfmodel import self_notes as _notes

    return _notes(inst.observer, inst.miner, now)


def patterns_text(now: float | None = None) -> str:
    from wawi.instinct.tool import summary

    return summary(now)


def forget() -> str:
    from wawi.instinct.tool import forget_all

    return forget_all()
