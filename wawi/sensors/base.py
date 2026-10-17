"""The sensor plugin contract and the small state helpers every sensor shares.

A sensor is polled from the runner's background thread. It must never touch
AppKit and must bound every I/O call with a timeout: one hung poll delays
every other sensor on the same thread.

Sensors are **edge-triggered**: ``poll()`` returns events only when something
*changed* (the site went down, the disk crossed a threshold), never one per
poll. :class:`Debouncer` and :class:`Levels` are the two shapes of "changed"
the built-ins need; plugins are welcome to use them.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from wawi.reactions import Event

log = logging.getLogger(__name__)


class Sensor:
    """Base class for sensors. Subclass and override :meth:`poll`.

    - ``name``: unique id, also the config section name.
    - ``interval``: seconds between polls (``interval`` in its config wins).
    - ``config``: the sensor's own config mapping.
    - ``setup()`` runs once on the runner thread before the first poll, so a
      slow setup (a geocoding request) never delays app startup.
    - ``summary()`` returns a short Spanish clause for ``wawi.sensors.status()``
      ("todas tus webs andan"), or None when it has nothing to say.
    """

    name: str = "sensor"
    default_interval: float = 60.0
    #: Lower bound so a config typo (interval: 0) cannot spin the CPU.
    min_interval: float = 5.0

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        try:
            interval = float(self.config.get("interval", self.default_interval))
        except (TypeError, ValueError):
            interval = self.default_interval
        self.interval = max(self.min_interval, interval)

    def setup(self) -> None:  # pragma: no cover - default no-op
        pass

    def poll(self) -> list[Event]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def summary(self) -> str | None:
        return None

    def inherit(self, old: "Sensor") -> None:
        """Take over what *old* (same name, previous config) had learned.

        Called when the runner swaps in a reconfigured sensor (a URL added by
        voice). The default keeps nothing; sensors with up/down state
        override it, because a fresh Debouncer would treat a site that is
        down right now as a silent baseline and the costume its outage put on
        would never be cancelled.
        """


class Debouncer:
    """Up/down state per key that flips only after N consecutive readings.

    One failed request is a blip, not an outage; barking at blips is exactly
    the kind of thing that gets a mascot turned off. ``update`` returns
    ``"down"`` or ``"up"`` on the poll where the state actually flips, else
    None.

    *alert_initial*: when the very first settled state is "down", report it
    (a website that is down at startup matters). When False the first state is
    a silent baseline (a dev server that simply isn't running yet is not an
    outage).
    """

    def __init__(self, fail_after: int = 2, recover_after: int = 1, *, alert_initial: bool = True) -> None:
        self.fail_after = max(1, int(fail_after))
        self.recover_after = max(1, int(recover_after))
        self.alert_initial = alert_initial
        self._state: dict[str, str] = {}
        self._bad: dict[str, int] = {}
        self._good: dict[str, int] = {}
        # Keys whose "down" was a silent baseline: their recovery is silent too
        # (a service starting for the first time did not "come back").
        self._silent: set[str] = set()

    def state(self, key: str) -> str | None:
        return self._state.get(key)

    def states(self) -> dict[str, str]:
        return dict(self._state)

    def update(self, key: str, bad: bool) -> str | None:
        prev = self._state.get(key)
        if bad:
            self._good[key] = 0
            self._bad[key] = self._bad.get(key, 0) + 1
            if prev != "down" and self._bad[key] >= self.fail_after:
                self._state[key] = "down"
                if prev is None and not self.alert_initial:
                    self._silent.add(key)
                    return None
                return "down"
        else:
            self._bad[key] = 0
            self._good[key] = self._good.get(key, 0) + 1
            if prev is None:
                # First healthy reading is the baseline, never an "up" event.
                self._state[key] = "up"
                return None
            if prev == "down" and self._good[key] >= self.recover_after:
                self._state[key] = "up"
                if key in self._silent:
                    self._silent.discard(key)
                    return None
                return "up"
        return None

    def copy(self) -> "Debouncer":
        new = Debouncer(self.fail_after, self.recover_after, alert_initial=self.alert_initial)
        new._state, new._bad, new._good = dict(self._state), dict(self._bad), dict(self._good)
        new._silent = set(self._silent)
        return new

    def forget(self, keep: Iterable[str]) -> None:
        """Drop keys no longer configured, so state cannot grow forever."""
        keep = set(keep)
        for table in (self._state, self._bad, self._good):
            for key in [k for k in table if k not in keep]:
                del table[key]
        self._silent &= keep


class Levels:
    """Ordered severity levels for one numeric reading, with hysteresis.

    *classify(value) -> int* maps a reading (any value: a number, a tuple)
    to a level (0 = fine). Going UP a
    level is immediate; going DOWN requires *relaxed(value) -> int* to agree,
    where *relaxed* uses thresholds shifted by a margin. Without the margin a
    disk hovering at 9.9 % / 10.1 % free would announce itself every poll.
    """

    def __init__(self, classify: Callable[[Any], int], relaxed: Callable[[Any], int]) -> None:
        self._classify = classify
        self._relaxed = relaxed
        self._level: dict[str, int] = {}

    def level(self, key: str) -> int | None:
        return self._level.get(key)

    def update(self, key: str, value: Any) -> tuple[int | None, int]:
        """Return (previous_level, new_level). Equal values mean no change."""
        prev = self._level.get(key)
        raw = self._classify(value)
        if prev is None or raw > prev:
            new = raw
        else:
            # Only step down as far as the relaxed thresholds allow.
            new = max(raw, min(prev, self._relaxed(value)))
        self._level[key] = new
        return prev, new


def human_bytes(n: float) -> str:
    """'12,3 GB' — Rioplatense decimal comma, because the bubble is Spanish."""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(max(0.0, n))
    unit = units[0]
    for unit in units:
        if value < 1000 or unit == units[-1]:
            break
        value /= 1000.0
    text = f"{value:.0f}" if value >= 100 or unit == "B" else f"{value:.1f}"
    return f"{text.replace('.', ',')} {unit}"


def as_list(value: Any) -> list[Any]:
    """Config lists may be written as a single scalar."""
    if value is None or value is False:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
