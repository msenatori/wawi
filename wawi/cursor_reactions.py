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
    #: Fast approaches this close together are "rushing" her (seconds).
    RUSH_WINDOW = 8.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._last: tuple[float, float] | None = None
        self._entered = False
        self._quiet_from: float | None = None
        self._last_reaction: dict[str, float] = {}
        self._rushes: list[float] = []

    def sample(self, offset: tuple[float, float] | None, *, idle: bool,
               asleep: bool = False, guarded: bool = False,
               happy: bool = True, night: bool = False,
               wake_cooldown: float | None = None) -> str | None:
        """Return a one-shot name, never from mere hover or a cursor on her.

        Awake, she wags (barks at a fast one); she growls only guarding her
        food, at night, or when out of sorts (*happy* False) AND rushed — a
        thirsty dog is not a grumpy one. Asleep, an ear twitch; with
        *wake_cooldown* an approach wakes her for real ("wake") at most that
        often; two rushes bark her awake. With her body on, every answer
        goes through her dog model (PetLife.cursor_reaction), which
        habituates and never lets a cursor wake her."""
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
            self._rushes = [t for t in self._rushes if now - t < self.RUSH_WINDOW]
            if speed >= self.FAST:
                self._rushes.append(now)
            rushed = len(self._rushes) >= 2
            if asleep:
                if rushed:
                    reaction = "bark"
                elif wake_cooldown is not None \
                        and now - self._last_reaction.get("wake", -1e12) >= wake_cooldown:
                    self._last_reaction["wake"] = now
                    return "wake"
                else:
                    reaction = "ear_twitch"
            elif guarded or night or (not happy and rushed):
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
