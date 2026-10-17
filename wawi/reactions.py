"""The contract between what Wawi notices and what Wawi does.

Three parts of her life meet here, and none of them imports the others:

- **Sensors** (``wawi/sensors/``) watch the machine and the world and emit
  :class:`Event` objects: a website went down, the disk is nearly full, rain
  is coming.
- **The mind** (``wawi/mind/``) and the sensors' own reflex rules turn events
  into :class:`Reaction` objects: put on the firefighter helmet and run,
  bark, say something in a bubble.
- **The body** (``wawi/mascot.py`` / ``wawi/panel.py`` / ``wawi/pet/``) drains
  the :class:`ReactionBus` on the main thread and acts them out.

Everything crosses threads through the bus, so a sensor polling the network
never touches AppKit, and the animator never blocks on I/O.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any

#: Severities, lowest to highest. Only ``urgent`` may break quiet mode
#: (meetings, full screen, quiet hours); see the reflex layer's budget.
SEVERITIES = ("info", "notice", "warning", "urgent")


@dataclass(frozen=True)
class Event:
    """Something a sensor noticed.

    *kind* is a dotted name owned by the sensor (``web.down``, ``web.up``,
    ``disk.low``, ``weather.rain``, ``rss.item``, ``service.down``…).
    *key* identifies the thing it is about (the URL, the volume, the feed
    item id) so repeats can be collapsed and a later ``web.up`` can cancel
    an ongoing ``web.down`` reaction. *data* is plain JSON-able detail.
    """

    kind: str
    severity: str = "info"
    key: str = ""
    title: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Reaction:
    """Something for Wawi to act out.

    - *behaviour*: name of an animation clip / behaviour the body knows
      (``run``, ``dig``, ``bark``, ``spaceship``, ``newspaper``…). Unknown
      names must degrade to something generic, never crash.
    - *costume*: an outfit worn while the reaction lasts (``firefighter``,
      ``umbrella``, ``sunglasses``, ``detective``…), or None.
    - *bubble*: short text for a speech bubble (never spoken aloud unless
      *speak* is True), or None.
    - *speak*: also say *bubble* with the voice.
    - *sound*: a short non-speech sound (``bark``, ``howl``, ``whine``) or None.
    - *priority*: 0 (background) … 3 (urgent); higher pre-empts lower.
    - *duration*: seconds to keep the costume/behaviour going; 0 = one shot.
    - *until_key*: keep going until a reaction with ``cancel_key`` equal to
      this arrives (e.g. firefighter until the site is back up).
    - *cancel_key*: stop any reaction whose ``until_key`` matches.
    - *source*: the Event that caused it, for logging and the mind.
    """

    behaviour: str | None = None
    costume: str | None = None
    bubble: str | None = None
    speak: bool = False
    sound: str | None = None
    priority: int = 1
    duration: float = 0.0
    until_key: str | None = None
    cancel_key: str | None = None
    source: Event | None = None


class ReactionBus:
    """Thread-safe priority queue of reactions, plus a raw event feed.

    Producers (any thread) call :meth:`emit` for events and :meth:`push`
    for reactions. The body calls :meth:`pop` on the main thread every
    frame; it never blocks. Event listeners (the reflexes, the mind)
    subscribe with :meth:`subscribe` and are called on the emitting thread,
    so they must be quick and must not touch AppKit.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heap: list[tuple[int, int, Reaction]] = []
        self._seq = itertools.count()
        self._listeners: list[Any] = []

    # -- reactions -------------------------------------------------------
    def push(self, reaction: Reaction) -> None:
        with self._lock:
            heapq.heappush(self._heap, (-reaction.priority, next(self._seq), reaction))

    def pop(self) -> Reaction | None:
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)[2]

    def pending(self) -> int:
        with self._lock:
            return len(self._heap)

    # -- events ----------------------------------------------------------
    def subscribe(self, listener: Any) -> None:
        """*listener(event)* is called for every emitted event."""
        with self._lock:
            self._listeners.append(listener)

    def emit(self, event: Event) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # one bad listener must not silence the rest
                import logging

                logging.getLogger(__name__).warning(
                    "Reaction listener failed on %s", event.kind, exc_info=True
                )


#: The process-wide bus. Tests build their own ReactionBus.
BUS = ReactionBus()
