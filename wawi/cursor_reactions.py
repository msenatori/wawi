"""Pure, clock-injected detection of a cursor approaching Wawi's head."""

from __future__ import annotations

import math
import time
from collections.abc import Callable


class CursorReactions:
    INNER = 80.0
    OUTER = 260.0
    FAST = 700.0  # points per second toward the head
    MIN_APPROACH = 25.0
    STILL_AFTER = 1.6
    STILL_SPEED = 25.0
    BARK_COOLDOWN = 120.0
    GROWL_COOLDOWN = 60.0
    WAG_COOLDOWN = 4.0
    EAR_COOLDOWN = 8.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._last: tuple[float, float] | None = None
        self._entered = False
        self._quiet_from: float | None = None
        self._last_reaction: dict[str, float] = {}
        self._rushes: list[float] = []

    def sample(self, offset: tuple[float, float] | None, *, idle: bool,
               asleep: bool = False, guarded: bool = False,
               happy: bool = True, night: bool = False) -> str | None:
        """Return a one-shot name, never from mere hover or a cursor on her."""
        now = self.clock()
        if not idle or offset is None:
            self._last = None
            self._entered = False
            self._quiet_from = None
            return None
        distance = math.hypot(*offset)
        previous = self._last
        self._last = (now, distance)
        if distance < self.INNER or distance > self.OUTER:
            self._entered = False
            self._quiet_from = None
            return None
        if previous is None:
            return None
        dt = now - previous[0]
        if dt <= 0 or dt > 0.5:
            return None
        speed = (previous[1] - distance) / dt
        entering = not self._entered and previous[1] > self.OUTER
        self._entered = True
        if abs(speed) <= self.STILL_SPEED:
            if self._quiet_from is None:
                self._quiet_from = now
        else:
            self._quiet_from = None

        reaction: str | None = None
        if entering and speed >= self.MIN_APPROACH:
            if asleep:
                if speed >= self.FAST:
                    self._rushes = [t for t in self._rushes if now - t < 8.0]
                    self._rushes.append(now)
                    reaction = "bark" if len(self._rushes) >= 2 else "ear_twitch"
                else:
                    reaction = "ear_twitch"
            elif guarded or night or not happy:
                reaction = "growl"
            elif speed >= self.FAST:
                reaction = "bark"
            else:
                reaction = "wag"
        elif self._quiet_from is not None and now - self._quiet_from >= self.STILL_AFTER:
            reaction = "ear_twitch" if asleep else "wag"

        cooldown = {"bark": self.BARK_COOLDOWN, "growl": self.GROWL_COOLDOWN,
                    "wag": self.WAG_COOLDOWN, "ear_twitch": self.EAR_COOLDOWN}
        if reaction is not None and now - self._last_reaction.get(reaction, -1e12) >= cooldown[reaction]:
            self._last_reaction[reaction] = now
            return reaction
        return None
