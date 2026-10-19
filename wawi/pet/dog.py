"""How a real dog spends her day at your desk: the state machine behind her
idle life.

The owner's brief: "as close to a real (female) dog as possible — I don't
mind her calm". So she mostly rests, and nothing comes in bursts. Each state
draws ITS OWN duration on entry (one hazard per state, no global timers), so
a long sleep stays a long sleep:

    DEEP ──► DOZE ──► (head lift) ──► lie back down ──► DOZE ──► DEEP …
                                  └─► get up ──► REST ──► ACTIVE ──► REST …
    REST ──► lie down (sigh) ──► DOZE ──► DEEP …          PLAY only on a trigger

* **DEEP** — deep sleep, a lognormal bout (median ~20 min). Only an ear
  flick for small things (rationed); strong stimuli wake her.
* **DOZE** — light sleep, on the way down and between deep bouts. A small
  stimulus gets a head lift and a glance, then back down.
* **REST** — awake, lying with her head up (the idle look). A glance at the
  owner every few minutes, a yawn or a scratch every 10–20 min at most.
* **ACTIVE** — up and about for a few minutes: sniffing, looking around,
  maybe walking to a spot near the owner.
* **PLAY** — a few seconds of zoomies, at most once a day (by default), only
  after a long rest or the owner coming back; never during work.

Overlays (a greeting, a request) are cues on top of a state. What she does
comes out as :class:`Cue` objects for the caller (wawi/pet/life.py) to play
on the animator, in order. Every number comes from a :class:`Personality`
(wawi/pet/personality.py; files in ``personalities/``).

Pure: the clock and the dice come in, nothing here touches AppKit.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

from wawi.pet.personality import Personality

log = logging.getLogger(__name__)

DEEP, DOZE, REST, ACTIVE, PLAY = "deep", "doze", "rest", "active", "play"
STATES = (DEEP, DOZE, REST, ACTIVE, PLAY)
SLEEP = frozenset({DEEP, DOZE})

#: What she may do while resting (rule 7: one every 10–20 min at most).
REST_POOL = ("yawn", "scratch", "stretch")
#: Up and about: ordinary dog things, nothing ironic, no tricks.
ACTIVE_POOL = ("sniff", "look_around", "look_left", "look_right", "scratch", "stretch",
               "yawn", "wag", "tilt", "shake", "dig", "play_chew", "stare")
#: Play: the energetic ones. Only ever from PLAY (rule 12).
PLAY_POOL = ("zoomies", "tail_chase", "play_ball", "run", "roll", "jump_tries",
             "bird_chase", "butterfly", "catch_leaf")

#: Small stimuli of any kind get one reaction per this many seconds at most.
SMALL_GAP = 60.0
#: Seconds per minute / hour, for reading the personality.
MIN, HOUR = 60.0, 3600.0
#: Sleep/awake transitions logged at INFO at most this often (else DEBUG),
#: except the ones a person caused (voice, touch…), which always are.
LOG_GAP = 300.0
#: Reasons a person (or something important) woke her: always logged.
LOUD_REASONS = frozenset({"voice", "touch", "owner_back", "command", "drop", "act",
                          "request", "celebrate", "work"})


@dataclass(frozen=True)
class Cue:
    """One thing for the animator to show. Kinds:

    ``lie_down`` (a sigh and down to sleep), ``head_lift`` (*seconds*, still
    lying), ``ear`` (an ear flick, asleep), ``get_up`` (*names*: the clips
    after standing, e.g. ``wake`` = stretch + yawn, ``shake``), ``glance``
    (*seconds*, toward the owner), ``gesture`` (*names*, a string of clips:
    ``"wag@6"`` = wag for ~6 s), ``behave`` (pick one from *names*),
    ``play`` (pick one from *names*), ``relocate`` (walk to a spot near the
    owner)."""

    kind: str
    names: tuple[str, ...] = ()
    seconds: float = 0.0
    reason: str | None = None


@dataclass
class Sense:
    """What the caller knows about now."""

    #: Local hour (0–24, fractional).
    hour: float = 12.0
    #: The pet.rhythm night flag.
    night: bool = False
    #: Someone is at the Mac (any input recently).
    owner_here: bool = True
    #: They are working (typing recently): never play then.
    working: bool = False
    #: The animator can show something now (idle, not in a turn, an act, a
    #: walk, notes, a dance, a game…). While not, awake states wait.
    free: bool = True


def _u(rng: random.Random, span: tuple[float, float]) -> float:
    return rng.uniform(span[0], span[1])


def rhythm(p: Personality, hour: float, night: bool, owner_here: bool = True
           ) -> tuple[float, float, float]:
    """Multipliers for now: (chance to get up, sleep bout length, rest length).

    Rule 11: more up in the morning and evening, deeper sleep after lunch,
    mostly asleep at night; alone she gets up less and rests shorter."""
    active, sleep, rest = 1.0, 1.0, 1.0
    if night:
        active, sleep, rest = p.rhythm_night_active, p.rhythm_night_sleep, p.rhythm_night_rest
    else:
        for hours, a, s in ((p.rhythm_morning_hours, p.rhythm_morning_active,
                             p.rhythm_morning_sleep),
                            (p.rhythm_siesta_hours, p.rhythm_siesta_active,
                             p.rhythm_siesta_sleep),
                            (p.rhythm_evening_hours, p.rhythm_evening_active,
                             p.rhythm_evening_sleep)):
            if hours[0] <= hour < hours[1]:
                active, sleep = a, s
                break
    if not owner_here:
        active *= p.rhythm_alone_active
        rest *= p.rhythm_alone_rest
    return active, sleep, rest


def greeting_level(p: Personality, away_s: float) -> int:
    """Rule 10: 0 nothing, 1 a look and a short wag, 2 she gets up and wags,
    3 the full welcome."""
    if not math.isfinite(away_s) or away_s < p.greeting_look_after_min * MIN:
        return 0
    if away_s < p.greeting_get_up_after_min * MIN:
        return 1
    if away_s < p.greeting_full_after_min * MIN:
        return 2
    return 3


class Habituation:
    """Rule 6: per-stimulus novelty. Each exposure multiplies it by *drop*
    (~0.5); it recovers toward 1 with time constant *recover* seconds. A
    stimulus that never stops (the cursor moving about all day) stays at
    zero novelty."""

    def __init__(self, drop: float = 0.5, recover: float = 600.0,
                 threshold: float = 0.3) -> None:
        self.drop = drop
        self.recover = max(1.0, recover)
        self.threshold = threshold
        self._v: dict[str, tuple[float, float]] = {}

    def novelty(self, kind: str, now: float) -> float:
        v0, t0 = self._v.get(kind, (1.0, now))
        return 1.0 - (1.0 - v0) * math.exp(-max(0.0, now - t0) / self.recover)

    def expose(self, kind: str, now: float) -> float:
        """Record one exposure; the novelty it had (before this one)."""
        n = self.novelty(kind, now)
        self._v[kind] = (n * self.drop, now)
        return n

    def notable(self, kind: str, now: float) -> bool:
        """Expose and say whether it was novel enough to react to."""
        return self.expose(kind, now) >= self.threshold


class Request:
    """Rule 13: asking for water, food or play. At most one episode per
    1–3 h; in it, a few tries a few minutes apart (look at the owner, at the
    bowl, back); answered, it stops; ignored, she sighs and lies down."""

    def __init__(self, p: Personality, rng: random.Random, now: float = 0.0) -> None:
        self.p, self.rng = p, rng
        self.need: str | None = None
        #: Its bubble was shown (once per episode).
        self.said = False
        self.tries_left = 0
        self.next_try = 0.0
        self.next_allowed = now

    def can_start(self, now: float) -> bool:
        return self.need is None and now >= self.next_allowed

    def start(self, now: float, need: str) -> None:
        lo, hi = (int(round(x)) for x in self.p.requests_tries)
        self.need = need
        self.said = False
        self.tries_left = self.rng.randint(max(1, lo), max(1, lo, hi))
        self.next_try = now

    def due(self, now: float) -> bool:
        return self.need is not None and self.tries_left > 0 and now >= self.next_try

    def tried(self, now: float) -> None:
        self.tries_left -= 1
        self.next_try = now + _u(self.rng, self.p.requests_between_min) * MIN

    def gave_up(self, now: float) -> bool:
        """True once, when the last try went unanswered long enough."""
        if self.need is None or self.tries_left > 0 or now < self.next_try:
            return False
        self._end(now)
        return True

    def answered(self, now: float, need: str | None = None) -> bool:
        if self.need is None or (need is not None and need != self.need):
            return False
        self._end(now)
        return True

    def _end(self, now: float) -> None:
        self.need = None
        self.tries_left = 0
        self.next_allowed = now + _u(self.rng, self.p.requests_every_h) * HOUR


@dataclass
class _Clock:
    glance: float = math.inf
    gesture: float = math.inf
    thing: float = math.inf


class Dog:
    """The state machine. See the module docstring."""

    def __init__(self, p: Personality, rng: random.Random | None = None,
                 now: float = 0.0, state: str = REST) -> None:
        self.p = p
        self.rng = rng or random.Random()
        self.habit = Habituation(p.habituation_drop, p.habituation_recover_min * MIN,
                                 p.habituation_threshold)
        self.state = state
        self.since = now
        self.until = now
        #: For DOZE: what comes after it ("deep", or "wake" = the head lift
        #: that ends a sleep).
        self.after = "deep"
        #: When she was last up and about (ACTIVE, PLAY, a greeting).
        self.last_up = now
        self.plays: list[float] = []
        self.last_ear = -math.inf
        self.last_small = -math.inf
        self.clock = _Clock()
        self.time_in: dict[str, float] = {s: 0.0 for s in STATES}
        self._logged_at = -math.inf
        self.reason: str | None = None
        self._enter(now, state, self._duration(state, now, Sense()), "start", log_it=False)

    # -- reading ------------------------------------------------------------------
    @property
    def asleep(self) -> bool:
        return self.state in SLEEP

    @property
    def deep(self) -> bool:
        return self.state == DEEP

    def budget(self, now: float) -> dict[str, float]:
        """Seconds spent in each state so far (the current one included)."""
        out = dict(self.time_in)
        out[self.state] += max(0.0, now - self.since)
        return out

    # -- the clock ----------------------------------------------------------------
    def tick(self, now: float, sense: Sense) -> list[Cue]:
        """Advance to *now*; the cues to play, in order."""
        if not sense.free:
            if not self.asleep:
                # Busy (a turn, a walk, an act): the awake clock waits, so
                # she does not drop off mid-conversation nor owe a burst.
                self.until = max(self.until, now + 1.0)
                for name in ("glance", "gesture", "thing"):
                    setattr(self.clock, name, max(getattr(self.clock, name), now + 1.0))
            return []
        if now >= self.until:
            return self._next(now, sense)
        if self.state == REST:
            if now >= self.clock.gesture:
                self.clock.gesture = now + _u(self.rng, self.p.rest_gesture_every_min) * MIN
                self.clock.glance = max(self.clock.glance, now + 30.0)
                return [Cue("behave", REST_POOL)]
            if now >= self.clock.glance:
                self.clock.glance = now + self._glance_gap(sense)
                return [Cue("glance", seconds=_u(self.rng, self.p.rest_glance_s))]
        elif self.state == ACTIVE and now >= self.clock.thing:
            self.clock.thing = now + _u(self.rng, self.p.active_something_every_s)
            return [Cue("behave", ACTIVE_POOL)]
        return []

    def _glance_gap(self, sense: Sense) -> float:
        gap = _u(self.rng, self.p.rest_glance_every_min) * MIN
        return gap if sense.owner_here else gap * 3.0

    def _duration(self, state: str, now: float, sense: Sense) -> float:
        p, rng = self.p, self.rng
        active, sleep, rest = rhythm(p, sense.hour, sense.night, sense.owner_here)
        if state == DEEP:
            bout = p.sleep_deep_median_min * math.exp(p.sleep_deep_sigma * rng.gauss(0.0, 1.0))
            bout = min(p.sleep_deep_max_min, max(p.sleep_deep_min_min, bout))
            return bout * sleep * MIN
        if state == DOZE:
            if self.after == "deep" and self.state in (REST, ACTIVE, PLAY):
                d = _u(rng, p.sleep_doze_in_min) * MIN
                if sense.owner_here and now - self.last_up < 5 * MIN:
                    d *= p.sleep_latency_after_activity
                return d
            return _u(rng, p.sleep_doze_between_min) * MIN
        if state == REST:
            span = p.rest_after_active_min if self.state in (ACTIVE, PLAY) else p.rest_minutes
            return _u(rng, span) * MIN * rest
        if state == ACTIVE:
            return _u(rng, p.active_minutes) * MIN
        return _u(rng, p.play_seconds)

    def _enter(self, now: float, state: str, seconds: float, reason: str,
               after: str = "deep", log_it: bool = True) -> None:
        self.time_in[self.state] += max(0.0, now - self.since)
        was_asleep = self.asleep
        if self.state in (ACTIVE, PLAY):
            self.last_up = now
        self.state, self.since, self.until, self.after = state, now, now + seconds, after
        self.reason = reason
        if state in (ACTIVE, PLAY):
            self.last_up = now
        if state == REST:
            self.clock = _Clock(glance=now + _u(self.rng, self.p.rest_glance_every_min) * MIN,
                                gesture=now + _u(self.rng, self.p.rest_gesture_every_min) * MIN)
        elif state == ACTIVE:
            self.clock = _Clock(thing=now + self.rng.uniform(3.0, 12.0))
        else:
            self.clock = _Clock()
        if log_it and was_asleep != self.asleep:
            self._log(now, reason)

    def _log(self, now: float, reason: str) -> None:
        loud = reason in LOUD_REASONS
        level = logging.INFO if loud or now - self._logged_at >= LOG_GAP else logging.DEBUG
        if level == logging.INFO:
            self._logged_at = now
        if self.asleep:
            log.log(level, "Fell asleep (%s)", reason)
        else:
            log.log(level, "Woke up (%s)", reason)

    def _next(self, now: float, sense: Sense) -> list[Cue]:
        p, rng = self.p, self.rng
        state = self.state
        if state == DEEP:
            # Never straight up out of deep sleep: through a light doze first.
            self.after = "wake"
            self._enter(now, DOZE, self._duration(DOZE, now, sense), "deep_over", after="wake")
            return []
        if state == DOZE:
            if self.after == "deep":
                self._enter(now, DEEP, self._duration(DEEP, now, sense), "deeper")
                return []
            return self._wake_up(now, sense)
        if state == REST:
            active, _sleep, _rest = rhythm(p, sense.hour, sense.night, sense.owner_here)
            if rng.random() < min(1.0, p.active_chance * active):
                self._enter(now, ACTIVE, self._duration(ACTIVE, now, sense), "up")
                if rng.random() < p.wake_walk:
                    return [Cue("relocate", reason="up")]
                return []
            return self.settle(now, "tired", sense)
        # ACTIVE or PLAY over: back to resting.
        self._enter(now, REST, self._duration(REST, now, sense), "settle")
        return []

    def _wake_up(self, now: float, sense: Sense) -> list[Cue]:
        """The end of a sleep (rule 4): she lifts her head and looks; half
        the time (more after lunch, at night, alone) she lies back down."""
        p, rng = self.p, self.rng
        cues = [Cue("head_lift", seconds=_u(rng, p.wake_head_lift_s))]
        active, _s, _r = rhythm(p, sense.hour, sense.night, sense.owner_here)
        back = p.sleep_lie_back_down
        back = min(0.95, back + (1.0 - back) * max(0.0, 1.0 - active))
        if rng.random() < back:
            self.after = "deep"
            self._enter(now, DOZE, _u(rng, p.sleep_doze_between_min) * MIN * 0.5,
                        "lay_back", after="deep")
            return cues
        self._enter(now, REST, self._duration(REST, now, sense), "self")
        names = []
        if rng.random() < p.wake_stretch:
            names.append("wake")
        if rng.random() < p.wake_shake:
            names.append("shake")
        cues.append(Cue("get_up", tuple(names), reason="self"))
        if now - self.last_up >= p.play_after_rest_min * MIN:
            play = self.maybe_play(now, sense, "long_rest")
            if play is not None:
                return cues + [play]
        if rng.random() < p.wake_walk:
            cues.append(Cue("relocate", reason="woke"))
        return cues

    # -- going to sleep -------------------------------------------------------------
    def settle(self, now: float, reason: str, sense: Sense | None = None) -> list[Cue]:
        """Lie down (a sigh), doze, then sleep (rule 3)."""
        if self.asleep:
            return []
        self.after = "deep"
        self._enter(now, DOZE, self._duration(DOZE, now, sense or Sense()), reason,
                    after="deep")
        return [Cue("lie_down", reason=reason)]

    # -- stimuli ----------------------------------------------------------------------
    def strong(self, now: float, reason: str, sense: Sense | None = None) -> list[Cue]:
        """Something that really wakes a dog (rule 5): the owner's voice, a
        click or petting, being picked up and dropped, a command, an
        important event. Straight to attentive; nothing to play (the
        animator already shows it)."""
        sense = sense or Sense()
        if self.asleep:
            self._enter(now, REST, self._duration(REST, now, sense), reason)
        elif self.state == REST:
            self.until = max(self.until, now + _u(self.rng, self.p.rest_minutes) * MIN * 0.5)
        return []

    def notice(self, now: float, kind: str, sense: Sense | None = None) -> str | None:
        """A small stimulus (*kind*: ``cursor``, ``app``, ``typing``…).
        Habituated (rule 6). Returns what she does: ``ear`` (deep asleep: an
        ear flick, rationed), ``lift`` (dozing: head up and a glance, then
        back down), ``look`` (awake), or None."""
        novel = self.habit.notable(kind, now)
        if self.state == DEEP:
            if novel and now - self.last_ear >= self.p.habituation_ear_flick_every_s:
                self.last_ear = now
                return "ear"
            return None
        # Different things at once (the cursor, typing, an app) are one
        # moment for a dog: one reaction, not three in a row.
        if not novel or now - self.last_small < SMALL_GAP:
            return None
        self.last_small = now
        return "lift" if self.state == DOZE else "look"

    def owner_back(self, now: float, away_s: float, sense: Sense | None = None) -> list[Cue]:
        """The owner is back at the Mac after *away_s* seconds (rule 10)."""
        sense = sense or Sense()
        level = greeting_level(self.p, away_s)
        if level == 0:
            return []
        rng, p = self.rng, self.p
        log.info("Owner back after %.0f min: greeting %d", away_s / MIN, level)
        if level == 1:
            if self.state == DEEP:
                return []
            if self.state == DOZE:
                return [Cue("head_lift", seconds=_u(rng, p.wake_head_lift_s))]
            return [Cue("gesture", ("glance", f"wag@{p.greeting_look_wag_s:g}"))]
        cues: list[Cue] = []
        if self.asleep:
            self._enter(now, REST, self._duration(REST, now, sense), "owner_back")
            cues.append(Cue("get_up", ("wake",), reason="owner_back"))
        elif self.state == REST:
            self.until = max(self.until, now + _u(rng, p.rest_minutes) * MIN)
        self.last_up = now
        if level == 2:
            cues.append(Cue("gesture", ("glance", f"wag@{_u(rng, p.greeting_get_up_wag_s):.1f}")))
        else:
            total = _u(rng, p.greeting_full_s)
            cues.append(Cue("gesture", ("glance", f"wag@{total * 0.45:.1f}", "hop",
                                        f"wag@{total * 0.45:.1f}", "shake")))
        play = self.maybe_play(now, sense, "owner_back")
        if play is not None:
            cues.append(play)
        return cues

    def maybe_play(self, now: float, sense: Sense, why: str) -> Cue | None:
        """Rule 12: play only on a trigger, never during work or at night,
        at most play_per_day a day with a long cooldown; likelier at night's
        doorstep (the evening)."""
        p = self.p
        if sense.night or sense.working or not sense.owner_here or p.play_per_day <= 0:
            return None
        self.plays = [t for t in self.plays if now - t < 24 * HOUR]
        if len(self.plays) >= p.play_per_day:
            return None
        if self.plays and now - self.plays[-1] < p.play_cooldown_h * HOUR:
            return None
        chance = p.play_chance
        lo, hi = p.rhythm_evening_hours
        if lo <= sense.hour < hi:
            chance *= p.play_evening_boost
        if self.rng.random() >= min(1.0, chance):
            return None
        self.plays.append(now)
        self._enter(now, PLAY, self._duration(PLAY, now, sense), why)
        log.info("Play time (%s)", why)
        return Cue("play", PLAY_POOL, reason=why)
