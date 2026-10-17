""""Viene a mirar lo que escribís": when she comes over to watch you type.

Ignored for a while (no voice turn, no touch) and you are typing away: she
walks over, sits next to the caret and watches, now and then with a side-eye.
She leaves after a bit, when you stop typing for a while, or when you touch
her (then she is happy).

Typing is detected WITHOUT a keylogger: the only input is "seconds since the
last key-down anywhere" (``CGEventSourceSecondsSinceLastEventType``), which
says nothing about which key. Pure: the clock comes in with every call.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class TypingDetector:
    """Sustained typing from "seconds since the last key press" samples."""

    #: A key within this many seconds: typing right now.
    ACTIVE = 1.5
    #: A pause longer than this ends a typing stretch.
    BREAK = 4.0

    since: float | None = None

    def feed(self, now: float, key_idle: float) -> None:
        if key_idle <= self.ACTIVE:
            if self.since is None:
                self.since = now - key_idle
        elif key_idle > self.BREAK:
            self.since = None

    def typing_for(self, now: float) -> float:
        return 0.0 if self.since is None else max(0.0, now - self.since)


@dataclass
class WatchTrigger:
    """Decides when she goes to watch, and when she leaves.

    Args (all seconds):
        lonely_after: ignored at least this long before she comes over.
        sustained: typing at least this long, so a stray key is not "typing".
        stay: how long she watches, drawn from this range.
        bored_after: she leaves once you have not typed for this long.
        cooldown: after leaving, she will not come back for this long.
    """

    lonely_after: float = 30 * 60.0
    sustained: float = 6.0
    stay: tuple[float, float] = (90.0, 180.0)
    bored_after: float = 45.0
    cooldown: float = 20 * 60.0
    rng: random.Random = field(default_factory=random.Random)

    typing: TypingDetector = field(default_factory=TypingDetector)
    watching: bool = False
    _until: float = 0.0
    _left_at: float = -1e12

    def should_go(self, now: float, ignored_for: float, key_idle: float) -> bool:
        """Feed a sample; True when she should go and watch now."""
        self.typing.feed(now, key_idle)
        if self.watching:
            return False
        if now - self._left_at < self.cooldown:
            return False
        if ignored_for < self.lonely_after:
            return False
        return self.typing.typing_for(now) >= self.sustained

    def started(self, now: float) -> None:
        self.watching = True
        self._until = now + self.rng.uniform(*self.stay)

    def should_leave(self, now: float, key_idle: float) -> bool:
        """Feed a sample while watching; True when she has seen enough."""
        self.typing.feed(now, key_idle)
        if not self.watching:
            return False
        return now >= self._until or key_idle >= self.bored_after

    def left(self, now: float) -> None:
        self.watching = False
        self._left_at = now

    def touched(self, now: float) -> bool:
        """You touched her: if she was watching, she leaves (happy). The
        cooldown starts afresh — you were there after all."""
        if not self.watching:
            return False
        self.left(now)
        return True
