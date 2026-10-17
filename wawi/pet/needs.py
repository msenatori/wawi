"""Her needs, gently decaying over real time, and the mood they add up to.

Five needs, each a *satisfaction* level from 0 (badly wanted) to 100 (fully
met): ``food`` (hunger), ``water`` (thirst), ``energy``, ``fun`` (boredom) and
``affection``. They drop over HOURS, not minutes: a whole day of ignoring her
leaves her hungry and a bit sad, never "dead" — every need has a floor and
nothing here ever punishes. Sleeping (her resting look) restores energy.

The state is a small JSON file (``pet_state.json`` by default, gitignored).
On restart the time she was away is caught up — as rest, since she was not on
screen — capped so a month away is not a disaster either.

Pure: time comes in as arguments, nothing here touches AppKit. A lock makes
every public method safe to call from any thread (the voice brain asks for
:func:`status` from its own).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

NEEDS = ("food", "water", "energy", "fun", "affection")

#: Points lost per hour while she is awake on screen. Tuned so that a
#: working day of being ignored (8-10 h) lands her in "hungry"/"lonely", and
#: a full day in "really hungry" — but never at zero.
DECAY_PER_H: dict[str, float] = {
    "food": 6.0,
    "water": 7.5,
    "energy": 4.0,
    "fun": 6.0,
    "affection": 4.5,
}
#: Asleep, energy comes BACK this fast (per hour) and the rest decay slower:
#: she mostly sleeps, and sleeping is good for her.
SLEEP_ENERGY_PER_H = 12.0
SLEEP_DECAY_FACTOR = 0.6
#: Nothing ever drops below this: hungry, never starving.
FLOOR = 8.0
#: Offline catch-up is capped at this many hours.
CATCH_UP_MAX_H = 72.0

#: Below this a need sets the mood.
LOW = 32.0
#: Everything at or above this: happy.
HIGH = 70.0

MOODS = ("happy", "content", "bored", "hungry", "sleepy", "lonely")

#: What each interaction gives back (points).
FEED = {"food": 55.0, "fun": 4.0, "affection": 4.0}
WATER = {"water": 60.0, "affection": 2.0}
PLAY = {"fun": 40.0, "affection": 6.0, "energy": -6.0}
TOUCH = {"affection": 5.0, "fun": 1.0}
TALK = {"affection": 3.0, "fun": 2.0}
#: Being stroked, per second of strokes.
PET_PER_S = {"affection": 2.5}


@dataclass(frozen=True)
class Snapshot:
    food: float
    water: float
    energy: float
    fun: float
    affection: float

    def get(self, name: str) -> float:
        return float(getattr(self, name))


def mood_of(s: Snapshot) -> str:
    """The mood her needs add up to. The most pressing low need wins
    (hunger and thirst both read "hungry"); with none low, happy when
    everything is high, content otherwise."""
    low = {
        "hungry": min(s.food, s.water),
        "sleepy": s.energy,
        "lonely": s.affection,
        "bored": s.fun,
    }
    pressing = [(v, m) for m, v in low.items() if v < LOW]
    if pressing:
        # Ties go to the order above: food first, then sleep, company, play.
        order = list(low)
        return min(pressing, key=lambda vm: (vm[0], order.index(vm[1])))[1]
    if all(s.get(n) >= HIGH for n in NEEDS):
        return "happy"
    return "content"


#: Mood → multiplier on her idle behaviours (mascot.BEHAVIOUR_WEIGHTS, and
#: acts like ``hungry_bowl`` that only a mood brings in). Utility-AI style:
#: the scheduler still picks at random, the mood just tilts the dice.
MOOD_BIAS: dict[str, dict[str, float]] = {
    "happy": {"wag": 2.0, "roll": 2.5, "tail_chase": 1.6, "run": 1.4,
              "eyeroll": 0.4, "side_eye": 0.4, "squint": 0.4},
    "content": {},
    "bored": {"dig": 2.5, "tail_chase": 2.2, "run": 1.6, "sniff": 1.5, "shrug": 1.8,
              "volume_knob": 1.6, "wag": 0.6, "ball_bring": 1.5},
    "hungry": {"hungry_bowl": 3.0, "sniff": 2.5, "stare": 2.0, "dig": 1.3,
               "roll": 0.3, "tail_chase": 0.4, "run": 0.4, "wag": 0.5},
    "sleepy": {"yawn": 3.5, "stretch": 2.0, "run": 0.2, "tail_chase": 0.2, "dig": 0.4,
               "roll": 0.5},
    "lonely": {"stare": 2.5, "side_eye": 2.0, "squint": 1.5, "tilt": 1.5, "wag": 0.5,
               "roll": 0.4},
}


def mood_bias(mood: str) -> dict[str, float]:
    return dict(MOOD_BIAS.get(mood, {}))


#: Short Spanish words for each need when it is low, and when it is fine.
_LOW_WORDS = {
    "food": "tiene hambre",
    "water": "tiene sed",
    "energy": "está cansada",
    "fun": "está aburrida",
    "affection": "te extraña",
}
_MOOD_WORDS = {
    "happy": "Está feliz",
    "content": "Está tranquila",
    "bored": "Está aburrida",
    "hungry": "Tiene hambre",
    "sleepy": "Tiene sueño",
    "lonely": "Se siente solita",
}


def describe(s: Snapshot) -> str:
    """A short Spanish summary for the voice brain and the menu.

    >>> describe(Snapshot(90, 90, 90, 90, 90))
    'Está feliz: panza llena, con agua, descansada y mimada.'
    """
    mood = mood_of(s)
    lows = [n for n in NEEDS if s.get(n) < LOW]
    if not lows:
        if mood == "happy":
            return "Está feliz: panza llena, con agua, descansada y mimada."
        mid = [n for n in NEEDS if s.get(n) < HIGH]
        hints = {"food": "un snack no le vendría mal", "water": "un poco de agua tampoco",
                 "energy": "una siesta le vendría bien", "fun": "jugaría un rato",
                 "affection": "unos mimos la alegrarían"}
        tail = ", ".join(hints[n] for n in mid[:2])
        return f"Está tranquila{': ' + tail if tail else ''}."
    words = [_LOW_WORDS[n] for n in lows]
    if len(words) == 1:
        body = words[0]
    else:
        body = ", ".join(words[:-1]) + " y " + words[-1]
    return f"{_MOOD_WORDS[mood]}: {body}."


class Needs:
    """The five needs, their decay, and what interactions give back.

    Thread-safe. All times are wall-clock seconds (``time.time()``), passed
    in, so tests run a day in a microsecond.
    """

    def __init__(self, values: dict[str, float] | None = None, now: float | None = None,
                 requests: dict[str, dict[str, float]] | None = None) -> None:
        self._lock = threading.Lock()
        self._v = {n: 80.0 for n in NEEDS}
        for n, v in (values or {}).items():
            if n in self._v:
                self._v[n] = _clamp(float(v))
        self.updated_at = time.time() if now is None else float(now)
        self.requests = {name: dict((requests or {}).get(name, {})) for name in ("food", "water")}
        #: Bumped on every interaction, so the store saves soon after one.
        self.changes = 0

    # -- reading -------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(**self._v)

    def mood(self) -> str:
        return mood_of(self.snapshot())

    def status(self) -> str:
        return describe(self.snapshot())

    def levels(self) -> dict[str, Any]:
        """For the voice brain: the mood and how much she WANTS each thing,
        0..1 (1 = badly): ``{"mood", "hunger", "thirst", "energy",
        "boredom", "loneliness"}`` — energy is how rested she is, 0..1."""
        s = self.snapshot()
        return {
            "mood": mood_of(s),
            "hunger": round(1.0 - s.food / 100.0, 2),
            "thirst": round(1.0 - s.water / 100.0, 2),
            "energy": round(s.energy / 100.0, 2),
            "boredom": round(1.0 - s.fun / 100.0, 2),
            "loneliness": round(1.0 - s.affection / 100.0, 2),
            "summary": describe(s),
        }

    def as_dict(self) -> dict[str, float]:
        with self._lock:
            return {n: round(v, 2) for n, v in self._v.items()}

    def request_state(self, name: str) -> dict[str, float]:
        with self._lock:
            return dict(self.requests[name])

    def set_request_state(self, name: str, **values: float) -> None:
        with self._lock:
            self.requests[name].update(values)
            self.changes += 1

    def requests_dict(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {name: dict(state) for name, state in self.requests.items()}

    # -- time ------------------------------------------------------------------
    def advance(self, now: float, asleep: bool = False) -> None:
        """Decay (and, asleep, rest) up to *now*. A clock going backwards
        (the machine's time was changed) is ignored, never "un-decays"."""
        with self._lock:
            dt = float(now) - self.updated_at
            self.updated_at = max(self.updated_at, float(now))
            if dt <= 0.0:
                return
            self._decay(min(dt, CATCH_UP_MAX_H * 3600.0) / 3600.0, asleep)

    def _decay(self, hours: float, asleep: bool) -> None:
        factor = SLEEP_DECAY_FACTOR if asleep else 1.0
        for n in NEEDS:
            if n == "energy" and asleep:
                self._v[n] = _clamp(self._v[n] + SLEEP_ENERGY_PER_H * hours)
                continue
            v = self._v[n]
            if v <= FLOOR:
                continue  # already at the floor: decay never pushes it lower
            self._v[n] = max(FLOOR, v - DECAY_PER_H[n] * factor * hours)

    # -- interactions ----------------------------------------------------------
    def give(self, gains: dict[str, float], scale: float = 1.0) -> None:
        with self._lock:
            for n, g in gains.items():
                if n in self._v:
                    self._v[n] = _clamp(self._v[n] + g * scale)
            self.changes += 1

    def feed(self) -> None:
        self.give(FEED)

    def water(self) -> None:
        self.give(WATER)

    def play(self) -> None:
        self.give(PLAY)

    def touched(self) -> None:
        self.give(TOUCH)

    def talked(self) -> None:
        self.give(TALK)

    def petted(self, seconds: float) -> None:
        if seconds > 0:
            self.give(PET_PER_S, seconds)


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


class NeedsStore:
    """Loads and saves :class:`Needs` as JSON, with offline catch-up.

    The file is written atomically (temp file + rename), at most every
    :attr:`SAVE_EVERY` seconds unless something changed, and never raises:
    a full disk costs her memory of breakfast, not the app.
    """

    SAVE_EVERY = 300.0
    VERSION = 1

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self._saved_changes = -1
        self._saved_at = 0.0

    def load(self, now: float | None = None) -> Needs:
        """The saved needs, caught up to *now* — as rest — or fresh ones."""
        now = time.time() if now is None else float(now)
        if self.path is None or not self.path.is_file():
            return Needs(now=now)
        try:
            data = json.loads(self.path.read_text())
            values = {n: float(v) for n, v in (data.get("needs") or {}).items()}
            saved_at = float(data.get("saved_at", now))
        except Exception:
            log.warning("Could not read %s; Wawi starts with fresh needs", self.path)
            return Needs(now=now)
        needs = Needs(values, now=min(saved_at, now), requests=data.get("requests"))
        # Away = resting: she was not on screen, so she slept.
        needs.advance(now, asleep=True)
        return needs

    def save(self, needs: Needs, now: float | None = None, force: bool = False) -> bool:
        if self.path is None:
            return False
        now = time.time() if now is None else float(now)
        if not force and needs.changes == self._saved_changes \
                and now - self._saved_at < self.SAVE_EVERY:
            return False
        data = {"version": self.VERSION, "saved_at": now, "needs": needs.as_dict(),
                "requests": needs.requests_dict()}
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=1))
            os.replace(tmp, self.path)
        except Exception:
            log.debug("Could not save %s", self.path, exc_info=True)
            return False
        self._saved_changes = needs.changes
        self._saved_at = now
        return True
