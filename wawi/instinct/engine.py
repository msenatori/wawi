"""The instinct engine: now and then, Wawi does something on her own.

Every couple of minutes (cheap: a few comparisons) the engine looks at two
things and decides whether to act -- and most of the time it decides not to:

* **her self-model** -- mood and needs (``wawi.pet.api.needs()``), how long
  since the owner touched or talked to her, what the sensors said lately;
* **the owner's context** -- from the observer and the pattern miner: just
  came back? deep in a focus block? a break is overdue by their own rhythm?
  later than they usually go? hopping between apps?

Each instinct is a *drive*: a trigger, an urgency in [0, 1] and a reaction.
The urgency is multiplied by what the owner taught her (the learner's
Thompson draw for ``instinct:<name>``, x0.5..x2) and rolled against a die,
so a welcomed instinct becomes more frequent and an unwelcome one rarer,
never certain and never extinct. On top: a per-instinct cooldown, a minimum
gap between any two instincts, the reflex layer's quiet mode (no bubble in
full screen or quiet hours, nothing at all in a meeting) and its hourly
bubble budget, shared with the sensors' alerts.

The instincts (cooldowns in :data:`wawi.instinct.settings.INSTINCTS`):

==============  ==========================================================
welcome         came back after 45+ min away: "¡Volviste!" and happy; the
                first return of the day is a quiet coffee (the morning
                sensor already says good morning).
focus_company   20+ min into a focus block: curls up nearby, no bubble.
focus_done      a 50+ min focus block just ended: celebrates it.
break_ball      worked longer than the owner's own break rhythm (default
                100 min): brings the ball. Not when a break nudge just came.
late_night      active past 23:30 and later than usual: yawns, "¿Seguimos?".
                Much rarer for a known night owl.
frenzy          a different app on most looks for 10 minutes: an ironic
                side-eye, rare.
hungry          her food is low and the owner is around: asks with the bowl.
lunch           inside the learned lunch window and no long break yet:
                "¿Comiste?".
noticed         the owner does the usual thing at the usual hour: a small
                "como siempre" -- she shows she knows them.
==============  ==========================================================

Reactions use new frame names (welcome, curl_focus, lunch_ask, late_night,
coffee, think_bubble, pattern_notebook); the body falls back to clips she
already has when the frames are missing (``wawi.pet.acting.FALLBACK_CLIPS``).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from wawi.instinct.observer import ActivityObserver, Context, day_minute
from wawi.instinct.categories import word
from wawi.instinct.patterns import PatternMiner, _duration, at_hour
from wawi.instinct.settings import InstinctSettings
from wawi.reactions import BUS, Event, Reaction, ReactionBus

log = logging.getLogger(__name__)

#: Seconds a bubble from an instinct stays (the body clips it anyway).
BUBBLE_S = 6.0
#: Events from the sensors that already cover an instinct: if one came this
#: recently (seconds), the instinct keeps quiet instead of nagging twice.
COVERED_BY = {
    "break_ball": ("break.suggest", 3600.0),
    "lunch": ("clock.lunch", 3 * 3600.0),
    "late_night": ("clock.late_night", 2 * 3600.0),
}
#: App switches in the observer's ten-minute window that read as a frenzy.
#: With a sample every 30 s there are 20 looks in ten minutes: this is "the
#: front app was different on most of them".
FRENZY_SWITCHES = 14
#: Instincts with no bubble: allowed in quiet mode (never in a meeting).
SILENT = frozenset({"focus_company"})


@dataclass(frozen=True)
class Drive:
    name: str
    urgency: float
    reaction: Reaction


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _default_needs() -> dict[str, Any] | None:
    try:
        from wawi.pet import api

        return api.needs()
    except Exception:
        return None


def _default_quiet() -> Any:
    try:
        from wawi import reflexes

        return reflexes.quiet_state()
    except Exception:
        return None


def _default_budget(spend: bool) -> bool | None:
    try:
        from wawi import reflexes

        return reflexes.budget_allows("info", spend=spend)
    except Exception:
        return None


def _default_learner() -> Any:
    try:
        from wawi import learn

        return learn.get()
    except Exception:
        return None


class InstinctEngine:
    def __init__(
        self,
        settings: InstinctSettings | None = None,
        *,
        observer: ActivityObserver | None = None,
        miner: PatternMiner | None = None,
        bus: ReactionBus = BUS,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
        needs: Callable[[], dict[str, Any] | None] = _default_needs,
        quiet: Callable[[], Any] = _default_quiet,
        budget: Callable[[bool], bool | None] = _default_budget,
        learner: Callable[[], Any] = _default_learner,
    ) -> None:
        self.settings = settings or InstinctSettings()
        self.observer = observer
        self.miner = miner
        self.bus = bus
        self._clock = clock
        self.rng = rng or random.Random()
        self._needs = needs
        self._quiet = quiet
        self._budget = budget
        self._learner = learner
        self._lock = threading.Lock()
        self.last_fired: dict[str, float] = {}
        self.last_any = -1e18
        self._decided_at = -1e18
        self._events: deque[tuple[str, float]] = deque(maxlen=100)
        self.history: deque[tuple[float, str, str | None]] = deque(maxlen=50)

    # -- inputs ------------------------------------------------------------------
    def on_event(self, event: Event) -> None:
        """Bus listener: remembers what the sensors just said (quick)."""
        kind = str(getattr(event, "kind", ""))
        if kind and not kind.startswith("instinct."):
            with self._lock:
                self._events.append((kind, float(getattr(event, "at", 0.0) or self._clock())))

    def _recently(self, kind: str, within: float, now: float) -> bool:
        with self._lock:
            return any(k == kind and now - t <= within for k, t in self._events)

    def reset(self) -> None:
        with self._lock:
            self.last_fired.clear()
            self.last_any = -1e18
            self.history.clear()

    # -- the decision -----------------------------------------------------------------
    def maybe_decide(self, now: float | None = None) -> str | None:
        now = self._clock() if now is None else now
        if now - self._decided_at < self.settings.decide_s:
            return None
        return self.decide(now)

    def decide(self, now: float | None = None) -> str | None:
        """Look, weigh, maybe act. Returns the instinct acted on, or None."""
        now = self._clock() if now is None else now
        self._decided_at = now
        s = self.settings
        if not s.enabled:
            return None
        ctx = self.observer.context(now) if (self.observer is not None and s.observe) else None
        quiet = self._quiet_state()
        meeting = bool(getattr(quiet, "meeting", False)) or (
            ctx is not None and ctx.active and ctx.category == "meeting")
        if meeting and s.quiet_meetings:
            return None
        quiet_now = bool(getattr(quiet, "quiet", False)) if quiet is not None else False
        if now - self.last_any < s.min_gap_min * 60.0:
            return None
        needs = self._safe_needs()
        drives = self.drives(ctx, needs, now)
        lr = self._learner_obj()
        for d in sorted(drives, key=lambda d: -d.urgency):
            if quiet_now and d.name not in SILENT:
                continue
            factor = 1.0
            if lr is not None and hasattr(lr, "instinct_factor"):
                try:
                    factor = float(lr.instinct_factor(d.name, bool(d.reaction.bubble)))
                except Exception:
                    factor = 1.0
            p = _clamp(d.urgency * factor)
            if self.rng.random() >= p:
                continue
            if d.reaction.bubble and self._budget(False) is False:
                continue
            self._fire(d, now, lr)
            return d.name
        return None

    def _fire(self, d: Drive, now: float, lr: Any) -> None:
        r = d.reaction
        if r.bubble:
            self._budget(True)
        self.last_fired[d.name] = now
        self.last_any = now
        self.history.append((now, d.name, r.bubble))
        log.info("instinct: %s (%s)", d.name, r.bubble or r.behaviour)
        self.bus.push(r)
        if lr is not None and hasattr(lr, "instinct_shown"):
            try:
                lr.instinct_shown(d.name, bubble=bool(r.bubble))
            except Exception:
                log.debug("instinct: learner hook failed", exc_info=True)

    def _ready(self, name: str, now: float) -> bool:
        s = self.settings
        if name not in s.instincts:
            return False
        last = self.last_fired.get(name)
        return last is None or now - last >= s.cooldowns.get(name, 60.0) * 60.0

    def _quiet_state(self) -> Any:
        try:
            q = self._quiet()
        except Exception:
            return None
        if isinstance(q, bool):
            return type("Q", (), {"quiet": q, "meeting": False})()
        return q

    def _safe_needs(self) -> dict[str, Any]:
        try:
            n = self._needs()
        except Exception:
            n = None
        return n if isinstance(n, dict) else {}

    def _learner_obj(self) -> Any:
        try:
            return self._learner()
        except Exception:
            return None

    # -- drives ----------------------------------------------------------------------
    def drives(self, ctx: Context | None, needs: dict[str, Any], now: float) -> list[Drive]:
        """Every instinct that wants to act now, with its urgency."""
        out: list[Drive] = []
        for name in self.settings.instincts:
            if not self._ready(name, now):
                continue
            covered = COVERED_BY.get(name)
            if covered and self._recently(covered[0], covered[1], now):
                continue
            fn = getattr(self, f"_drive_{name}", None)
            if fn is None:
                continue
            try:
                d = fn(ctx, needs, now)
            except Exception:
                log.debug("instinct: drive %s failed", name, exc_info=True)
                d = None
            if d is not None and d.urgency > 0:
                out.append(d)
        return out

    def _reaction(self, name: str, behaviour: str | None, bubble: str | None = None,
                  *, priority: int = 1, duration: float = 0.0) -> Reaction:
        source = Event(kind=f"instinct.{name}", severity="info", key=f"instinct:{name}",
                       title=bubble or "")
        return Reaction(behaviour=behaviour, bubble=bubble,
                        speak=bool(bubble) and self.settings.speak,
                        priority=priority, duration=duration, source=source)

    @staticmethod
    def _level(needs: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return _clamp(float(needs.get(key, default)))
        except (TypeError, ValueError):
            return default

    def _pattern(self, key: str, now: float) -> Any:
        if self.miner is None:
            return None
        try:
            return self.miner.get(key, now)
        except Exception:
            return None

    def _drive_welcome(self, ctx, needs, now) -> Drive | None:
        if ctx is None or not ctx.active or ctx.returned is None:
            return None
        at, gap = ctx.returned
        if now - at > 180.0 or gap < 45 * 60.0:
            return None
        first = gap == float("inf") or (ctx.day_started is not None and abs(ctx.day_started - at) < 1)
        if first:
            # The morning sensor already says good morning; she just has her coffee.
            bubble = None if self._recently("clock.morning", 3 * 3600.0, now) else "¡Buen día!"
            return Drive("welcome", 0.7, self._reaction("welcome", "coffee", bubble))
        lonely = self._level(needs, "loneliness", 0.3)
        bubble = "¡Volviste! Te extrañé." if gap >= 3 * 3600 or lonely >= 0.6 else \
            self.rng.choice(("¡Volviste!", "¡Ahí estás!", "¡Volviste! ¿Todo bien?"))
        return Drive("welcome", 0.5 + 0.4 * lonely, self._reaction("welcome", "welcome", bubble))

    def _drive_focus_company(self, ctx, needs, now) -> Drive | None:
        if ctx is None or not ctx.active or ctx.focus_s < 20 * 60.0:
            return None
        return Drive("focus_company", 0.5,
                     self._reaction("focus_company", "curl_focus", None, priority=0, duration=90.0))

    def _drive_focus_done(self, ctx, needs, now) -> Drive | None:
        if ctx is None or ctx.focus_ended is None:
            return None
        ended, length = ctx.focus_ended
        if now - ended > 300.0 or length < 50 * 60.0:
            return None
        bubble = f"¡Qué bloque! {_duration(length / 60.0)} de corrido."
        return Drive("focus_done", 0.7, self._reaction("focus_done", "cheer", bubble))

    def _drive_break_ball(self, ctx, needs, now) -> Drive | None:
        if ctx is None or not ctx.active:
            return None
        every = 100.0
        p = self._pattern("breaks", now)
        if p is not None and p.confidence >= 0.3:
            every = max(40.0, min(180.0, float(p.data.get("every", every))))
        worked = ctx.since_break_s / 60.0
        if worked < every:
            return None
        bored = self._level(needs, "boredom", 0.3)
        urgency = _clamp(0.3 + 0.3 * (worked / every - 1.0) + 0.3 * bored, 0.0, 0.9)
        if worked >= 90:
            bubble = f"Llevás {_duration(worked)} sin parar. ¿Pausa?"
        else:
            bubble = "¿Pausa? Te traje la pelota."
        return Drive("break_ball", urgency, self._reaction("break_ball", "ball_bring", bubble))

    def _drive_late_night(self, ctx, needs, now) -> Drive | None:
        if ctx is None or not ctx.active:
            return None
        lt = time.localtime(now)
        if not (lt.tm_hour < 5 or (lt.tm_hour == 23 and lt.tm_min >= 30)):
            return None
        urgency = 0.5 + 0.3 * (1.0 - self._level(needs, "energy", 0.5))
        end = self._pattern("end", now)
        if end is not None and end.confidence >= 0.3 and \
                day_minute(now) < float(end.data.get("minute", 0)) + 60:
            return None  # this IS their usual hour
        owl = self._pattern("late", now)
        if owl is not None and owl.confidence >= 0.5:
            urgency *= 0.3
        hour = "la 1" if lt.tm_hour == 1 else f"las {lt.tm_hour}"
        if lt.tm_hour == 23:
            bubble = "¿Seguimos? Ya casi es medianoche."
        else:
            bubble = f"¿Seguimos? Son {hour}."
        return Drive("late_night", urgency, self._reaction("late_night", "late_night", bubble))

    def _drive_frenzy(self, ctx, needs, now) -> Drive | None:
        if ctx is None or not ctx.active or ctx.switches < FRENZY_SWITCHES:
            return None
        bubble = "¿Todo bien? Saltás de app en app." if self.rng.random() < 0.5 else None
        return Drive("frenzy", 0.25, self._reaction("frenzy", "side_eye", bubble, priority=0))

    def _drive_hungry(self, ctx, needs, now) -> Drive | None:
        hunger = self._level(needs, "hunger", 0.0)
        if hunger < 0.7:
            return None
        if ctx is not None and (not ctx.active or ctx.focus_s >= 20 * 60.0):
            return None
        bubble = self.rng.choice(("Che, ¿y mi comida? Pregunto sin presionar.",
                                  "Mi plato está vacío. Lo dejo ahí."))
        return Drive("hungry", _clamp((hunger - 0.6) * 2.0),
                     self._reaction("hungry", "hungry_bowl", bubble))

    def _drive_lunch(self, ctx, needs, now) -> Drive | None:
        if ctx is None or not ctx.active or self.miner is None:
            return None
        exp = next((e for e in self.miner.expectations(ctx, now) if e.key == "lunch"), None)
        if exp is None or exp.in_min > 15:
            return None
        bubble = self.rng.choice(("¿Comiste?", "¿Y el almuerzo? ¿Comiste?"))
        return Drive("lunch", _clamp(0.2 + 0.6 * exp.confidence),
                     self._reaction("lunch", "lunch_ask", bubble))

    def _drive_noticed(self, ctx, needs, now) -> Drive | None:
        if ctx is None or not ctx.active or self.miner is None:
            return None
        cat = ctx.category
        p = self._pattern(f"app:{cat}", now)
        if p is None or p.confidence < 0.5:
            return None
        if int(p.data.get("hour", -1)) != time.localtime(now).tm_hour:
            return None
        name = word(cat)
        name = name[0].upper() + name[1:]
        bubble = f"{name} {at_hour(p.data['hour'])}, como siempre."
        return Drive("noticed", 0.3, self._reaction("noticed", "think_bubble", bubble, priority=0))
