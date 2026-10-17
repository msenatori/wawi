"""Background scheduler for sensors.

One daemon thread polls every sensor when it is due. Why one thread and not
a pool: sensors are few, each poll is short and bounded by its own timeouts,
and a single thread makes the ordering, the logs and the shutdown trivially
predictable. A sensor that keeps failing backs off exponentially instead of
hammering a dead endpoint (or the log).

The scheduling core (:meth:`SensorRunner.tick`) takes the time as an argument
so tests drive it with a fake clock and never sleep.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from wawi.reactions import BUS, Event, ReactionBus
from wawi.sensors.base import Sensor

log = logging.getLogger(__name__)

#: Longest the thread ever sleeps, so periodic hooks (reflex batching) and
#: stop() stay responsive even when every sensor polls hourly.
MAX_SLEEP_S = 30.0


@dataclass
class _Slot:
    sensor: Sensor
    next_due: float
    failures: int = 0
    ready: bool = False  # setup() done
    dead: bool = False  # setup() failed for good
    last_error: str = ""


class SensorRunner:
    def __init__(
        self,
        sensors: list[Sensor],
        bus: ReactionBus = BUS,
        *,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        jitter: float = 0.1,
        max_backoff: float = 3600.0,
        hooks: list[Callable[[], Any]] | None = None,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._rng = rng or random.Random()
        self._jitter = max(0.0, float(jitter))
        self._max_backoff = float(max_backoff)
        self._hooks = list(hooks or [])
        now = clock()
        # Stagger first polls a little so every sensor does not fire its
        # network request in the same instant at launch.
        self._slots = [
            _Slot(s, now + (self._rng.uniform(0, 2.0) if self._jitter else 0.0)) for s in sensors
        ]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Guards the slot LIST (not the sensors): voice tools reconfigure it
        # from a turn's thread while the runner thread is iterating it.
        self._slots_lock = threading.RLock()
        # Set when the slots change, so the loop wakes up and polls a newly
        # added sensor now instead of after its current (up to 30 s) nap.
        self._changed = threading.Event()

    # -- introspection ----------------------------------------------------
    @property
    def sensors(self) -> list[Sensor]:
        with self._slots_lock:
            return [slot.sensor for slot in self._slots]

    def get(self, name: str) -> Sensor | None:
        with self._slots_lock:
            return next((s.sensor for s in self._slots if s.sensor.name == name), None)

    # -- live reconfiguration ----------------------------------------------
    def replace(self, sensor: Sensor) -> None:
        """Install *sensor*, replacing the one with the same name (or adding it).

        The newcomer inherits what the old one had learned (``inherit``), so a
        site that is down right now does not lose its "down" state, which is
        what eventually cancels the firefighter costume. It polls soon rather
        than at the old schedule: someone just asked for it.
        """
        now = self._clock()
        old: Sensor | None = None
        with self._slots_lock:
            for i, slot in enumerate(self._slots):
                if slot.sensor.name == sensor.name:
                    old = slot.sensor
                    self._slots[i] = _Slot(sensor, now + 1.0)
                    break
            else:
                self._slots.append(_Slot(sensor, now + 1.0))
        if old is not None and old is not sensor:
            try:
                sensor.inherit(old)
            except Exception:
                log.debug("Sensor %s could not inherit state", sensor.name, exc_info=True)
            _close(old)
        log.info("Sensor %s %s", sensor.name, "reconfigured" if old else "added")
        self._changed.set()

    def remove(self, name: str) -> bool:
        with self._slots_lock:
            gone = [s for s in self._slots if s.sensor.name == name]
            self._slots = [s for s in self._slots if s.sensor.name != name]
        for slot in gone:
            _close(slot.sensor)
        if gone:
            log.info("Sensor %s removed", name)
            self._changed.set()
        return bool(gone)

    def add_hook(self, hook: Callable[[], Any]) -> None:
        self._hooks.append(hook)

    # -- scheduling core --------------------------------------------------
    def tick(self, now: float | None = None) -> int:
        """Poll every due sensor once. Returns how many were polled."""
        now = self._clock() if now is None else now
        ran = 0
        with self._slots_lock:
            slots = list(self._slots)
        for slot in slots:
            if slot.dead or self._stop.is_set() or now < slot.next_due:
                continue
            self._run_slot(slot, now)
            ran += 1
        for hook in self._hooks:
            try:
                hook()
            except Exception:
                log.warning("Sensor runner hook failed", exc_info=True)
        return ran

    def next_due(self) -> float | None:
        with self._slots_lock:
            dues = [s.next_due for s in self._slots if not s.dead]
        return min(dues) if dues else None

    def _run_slot(self, slot: _Slot, now: float) -> None:
        sensor = slot.sensor
        try:
            if not slot.ready:
                sensor.setup()
                slot.ready = True
            events = sensor.poll() or []
        except Exception as exc:
            slot.failures += 1
            slot.last_error = f"{type(exc).__name__}: {exc}"
            delay = min(self._max_backoff, sensor.interval * (2 ** slot.failures))
            slot.next_due = now + delay
            # First failure loud, the rest quiet: a laptop offline for an hour
            # should not write a traceback every few minutes.
            level = logging.WARNING if slot.failures == 1 else logging.DEBUG
            log.log(level, "Sensor %s failed (%d in a row), retrying in %.0fs: %s",
                    sensor.name, slot.failures, delay, slot.last_error,
                    exc_info=slot.failures == 1 and not isinstance(exc, OSError))
            return
        if slot.failures:
            log.info("Sensor %s recovered after %d failures", sensor.name, slot.failures)
        slot.failures = 0
        slot.last_error = ""
        slot.next_due = now + sensor.interval * (1.0 + self._rng.uniform(-self._jitter, self._jitter))
        for event in events:
            if not isinstance(event, Event):
                log.warning("Sensor %s returned a non-Event: %r", sensor.name, event)
                continue
            log.info("Sensor event %s (%s) %s", event.kind, event.severity, event.key)
            self._bus.emit(event)

    # -- thread -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="wawi-sensors", daemon=True)
        self._thread.start()
        log.info("Sensors running: %s", ", ".join(s.sensor.name for s in self._slots) or "none")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # pragma: no cover - tick already isolates sensors
                log.exception("Sensor runner tick failed")
            due = self.next_due()
            wait = MAX_SLEEP_S if due is None else min(MAX_SLEEP_S, max(0.5, due - self._clock()))
            if self._changed.wait(wait):
                self._changed.clear()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._changed.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        with self._slots_lock:
            slots = list(self._slots)
        for slot in slots:
            _close(slot.sensor)


def _close(sensor: Sensor) -> None:
    try:
        sensor.close()
    except Exception:
        log.debug("Sensor %s close failed", sensor.name, exc_info=True)
