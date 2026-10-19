"""``pet.liveliness``: the old one-knob setting, now an alias for a
personality (wawi/pet/personality.py, files in ``personalities/``).

* ``calm`` (the default) → ``perra_tranquila``: a grown dog at your desk,
  mostly asleep, as close to a real one as we can.
* ``normal`` → ``perra_atenta``: the same dog, a little more awake and
  glancing at you more often.
* ``playful`` → ``cachorra``: a young dog: shorter sleeps, plays up to three
  times a day.

``pet.personality`` wins when both are set. Also here: :class:`Presence`,
the owner coming back to the Mac, from "seconds since any input".
"""

from __future__ import annotations

import math

from wawi.pet import personality as _personality

LEVELS = ("calm", "normal", "playful")
DEFAULT = "calm"


def parse(value: object) -> str:
    """A level name from config (Spanish too); anything else is the default."""
    text = str(value or "").strip().lower()
    aliases = {"tranquila": "calm", "calma": "calm", "normal": "normal",
               "atenta": "normal", "inquieta": "playful", "juguetona": "playful"}
    text = aliases.get(text, text)
    return text if text in LEVELS else DEFAULT


def personality_for(level: str) -> str:
    """The built-in personality a liveliness level stands for."""
    return _personality.ALIASES.get(parse(level), _personality.DEFAULT)


class Presence:
    """The owner coming back to the Mac, from "seconds since any input".

    Away means no input for at least *away_after*; back is the first sample
    with input again (a key or the mouse in the last couple of seconds).
    :attr:`away_for` then says how long they were gone (for the greeting).
    Unknown idle (inf, NaN) changes nothing: no Quartz, no greetings.
    """

    BACK = 2.0

    def __init__(self, away_after: float) -> None:
        self.away_after = away_after
        self.away = False
        self.away_for = 0.0
        self._longest = 0.0

    def feed(self, input_idle: float) -> bool:
        """One sample; True exactly once when they are back."""
        if not math.isfinite(input_idle) or not math.isfinite(self.away_after):
            return False
        if input_idle >= self.away_after:
            self.away = True
            self._longest = max(self._longest, input_idle)
            return False
        if self.away and input_idle <= self.BACK:
            self.away = False
            self.away_for, self._longest = self._longest, 0.0
            return True
        return False
