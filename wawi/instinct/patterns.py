"""The pattern miner: the owner's routines, from the store, with plain stats.

No model: medians, spreads and "on how many of the days it could have
happened did it happen". Each finding is a :class:`Pattern` with a Spanish
sentence and a confidence in [0, 1]:

* **start** -- the usual start of the day (weekdays and weekends apart).
* **end** -- the usual end of the day.
* **lunch** -- the usual long midday break (11:30-15:30, 20+ minutes).
* **breaks** -- the usual stretch between breaks ("una pausa cada ~70 min").
* **focus** -- focus blocks: long runs in the editor / terminal with few
  app switches; how long and when they usually happen.
* **app:<category>** -- an app that shows up at the same hour on weekdays
  ("Slack a las 10").
* **late** -- late-night work (activity between 00:00 and 04:00).
* **hours** -- active hours on a normal weekday.
* **switching** -- how much the owner hops between apps, per active hour.

Confidence is support (days where it happened / days where it could have)
scaled down when there are few days and when the times are scattered, so a
routine seen twice is a hunch and one seen fifteen times is a fact.
:func:`expectations` turns them into "what is likely now / next".
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from wawi.instinct.categories import FOCUS, word
from wawi.instinct.observer import ActivityStore, Context, day_key, day_minute, hhmm

#: Below this many days nothing is a pattern yet.
MIN_DAYS = 3
#: A day counts as worked when it had at least this many active minutes.
WORKED_MIN = 60.0
#: Categories never reported as a "recurring app" (they are the job itself).
_NOT_RECURRING = FOCUS | {"browser", "files", "unknown", "idle", "other"}
#: Lunch: a break that starts in this window (day minutes) and lasts this long.
LUNCH_FROM, LUNCH_TO = (11 * 60 + 30) - 240, (15 * 60 + 30) - 240
LUNCH_MIN = 20


@dataclass(frozen=True)
class Pattern:
    key: str
    text: str
    confidence: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Expectation:
    key: str
    text: str
    confidence: float
    #: Minutes until it is due (negative = overdue / happening now).
    in_min: float = 0.0


def confidence(hits: int, total: int, spread_min: float | None = None,
               tolerance_min: float = 45.0) -> float:
    """Support across days, damped by few days and by scatter."""
    if total <= 0 or hits <= 0:
        return 0.0
    support = hits / total
    volume = min(1.0, hits / 10.0) ** 0.5
    tight = 1.0 if spread_min is None else 1.0 / (1.0 + max(0.0, spread_min) / tolerance_min)
    return round(max(0.0, min(1.0, support * volume * tight)), 2)


def _spread(values: list[float]) -> float:
    """Half the interquartile range (robust "± minutes")."""
    if len(values) < 4:
        return statistics.pstdev(values) if len(values) > 1 else 0.0
    q = statistics.quantiles(values, n=4)
    return (q[2] - q[0]) / 2.0


def about(dm: float) -> str:
    """A usual time, said the way people say it: to the nearest 10 minutes."""
    return hhmm(round(dm / 10.0) * 10)


def _duration(minutes: float) -> str:
    minutes = int(round(minutes))
    h, m = divmod(minutes, 60)
    if h and m >= 10:
        return f"{h} h {m:02d}"
    if h:
        return "una hora" if h == 1 else f"{h} horas"
    return f"{m} minutos"


def at_hour(h: int) -> str:
    """"a las 10" / "a la 1" (Spanish clock hours)."""
    return "a la 1" if int(h) == 1 else f"a las {int(h)}"


def _worked(days: Iterable[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    return [(k, d) for k, d in days if sum(d.get("act", ())) / 60.0 >= WORKED_MIN
            and d.get("first") is not None]


class PatternMiner:
    """Reads the store; caches its findings for a few minutes."""

    def __init__(self, store: ActivityStore, *, cache_s: float = 600.0) -> None:
        self.store = store
        self.cache_s = cache_s
        self._cached: tuple[float, str, list[Pattern]] | None = None

    def invalidate(self) -> None:
        self._cached = None

    # -- patterns -------------------------------------------------------------
    def patterns(self, now: float | None = None, min_conf: float = 0.25) -> list[Pattern]:
        """Every routine found, most confident first. Today is left out (it
        is not over), so "usual" means the days before."""
        now = time.time() if now is None else now
        today = day_key(now)
        if self._cached and now - self._cached[0] < self.cache_s and self._cached[1] == today:
            found = self._cached[2]
        else:
            found = self._mine([(k, d) for k, d in self.store.recent() if k < today])
            self._cached = (now, today, found)
        return [p for p in found if p.confidence >= min_conf]

    def get(self, key: str, now: float | None = None) -> Pattern | None:
        return next((p for p in self.patterns(now, 0.0) if p.key == key), None)

    def _mine(self, days: list[tuple[str, dict[str, Any]]]) -> list[Pattern]:
        out: list[Pattern] = []
        worked = _worked(days)
        if len(worked) < MIN_DAYS:
            return out
        weekdays = [(k, d) for k, d in worked if d.get("wd", 0) < 5]
        weekend = [(k, d) for k, d in worked if d.get("wd", 0) >= 5]
        for fn in (self._start, self._end, self._lunch, self._breaks, self._focus,
                   self._recurring, self._late, self._hours, self._switching):
            try:
                out += fn(worked, weekdays, weekend, days)
            except (ValueError, ZeroDivisionError, statistics.StatisticsError):
                continue
        return sorted(out, key=lambda p: -p.confidence)

    def _start(self, worked, weekdays, weekend, days) -> list[Pattern]:
        out = []
        for label, group, key in (("los días de semana", weekdays, "start"),
                                  ("los fines de semana", weekend, "start_weekend")):
            if len(group) < MIN_DAYS:
                continue
            firsts = [float(d["first"]) for _, d in group]
            med, spread = statistics.median(firsts), _spread(firsts)
            hits = sum(1 for f in firsts if abs(f - med) <= 45)
            conf = confidence(hits, len(group), spread)
            out.append(Pattern(key, f"{label.capitalize()} arrancás tipo {about(med)}",
                               conf, {"minute": med, "spread": spread}))
        return out

    def _end(self, worked, weekdays, weekend, days) -> list[Pattern]:
        group = weekdays if len(weekdays) >= MIN_DAYS else worked
        lasts = [float(d["last"]) for _, d in group if d.get("last") is not None]
        if len(lasts) < MIN_DAYS:
            return []
        med, spread = statistics.median(lasts), _spread(lasts)
        hits = sum(1 for v in lasts if abs(v - med) <= 60)
        return [Pattern("end", f"Solés cortar tipo {about(med)}",
                        confidence(hits, len(lasts), spread, 60.0),
                        {"minute": med, "spread": spread})]

    def _lunch(self, worked, weekdays, weekend, days) -> list[Pattern]:
        group = weekdays if len(weekdays) >= MIN_DAYS else worked
        starts, lengths = [], []
        for _, d in group:
            cands = [b for b in d.get("breaks", ())
                     if LUNCH_FROM <= b[0] <= LUNCH_TO and b[1] >= LUNCH_MIN]
            if cands:
                b = max(cands, key=lambda b: b[1])
                starts.append(float(b[0]))
                lengths.append(float(b[1]))
        if len(starts) < MIN_DAYS:
            return []
        med, spread = statistics.median(starts), _spread(starts)
        dur = statistics.median(lengths)
        return [Pattern("lunch", f"Almorzás tipo {about(med)}, más o menos {_duration(dur)}",
                        confidence(len(starts), len(group), spread),
                        {"minute": med, "length": dur, "spread": spread})]

    def _breaks(self, worked, weekdays, weekend, days) -> list[Pattern]:
        stretches: list[float] = []
        support = 0
        for _, d in worked:
            marks = sorted(float(b[0]) for b in d.get("breaks", ()))
            if not marks:
                continue
            support += 1
            start = float(d["first"])
            for b in sorted(d.get("breaks", ()), key=lambda b: b[0]):
                if b[0] - start >= 10:
                    stretches.append(b[0] - start)
                start = b[0] + b[1]
        if len(stretches) < MIN_DAYS * 2:
            return []
        med = statistics.median(stretches)
        spread = _spread(stretches)
        return [Pattern("breaks", f"Hacés una pausa cada {_duration(med)} más o menos",
                        confidence(support, len(worked), spread, med or 60.0),
                        {"every": med, "spread": spread})]

    def _focus(self, worked, weekdays, weekend, days) -> list[Pattern]:
        blocks = [(float(b[0]), float(b[1]), str(b[2])) for _, d in worked
                  for b in d.get("focus", ())]
        with_focus = sum(1 for _, d in worked if d.get("focus"))
        if with_focus < MIN_DAYS:
            return []
        length = statistics.median([b[1] for b in blocks])
        hours = [int(hhmm(b[0]).split(":")[0]) for b in blocks]
        top = statistics.mode(hours)
        at_top = sum(1 for h in hours if abs(h - top) <= 1)
        cat = statistics.mode([b[2] for b in blocks])
        text = (f"Te concentrás en bloques de {_duration(length)} en {word(cat)}, "
                f"sobre todo {at_hour(top)}")
        return [Pattern("focus", text, confidence(with_focus, len(worked)),
                        {"length": length, "hour": top, "share_at_hour": at_top / len(blocks),
                         "category": cat})]

    def _recurring(self, worked, weekdays, weekend, days) -> list[Pattern]:
        group = weekdays if len(weekdays) >= MIN_DAYS else worked
        best: dict[str, tuple[int, int]] = {}
        for _, d in group:
            for cat, hours in d.get("cats", {}).items():
                if cat in _NOT_RECURRING:
                    continue
                for h, secs in enumerate(hours):
                    if secs >= 5 * 60:
                        k = f"{cat}|{h}"
                        best[k] = (best.get(k, (0, 0))[0] + 1, h)
        out = []
        seen: set[str] = set()
        for k, (hits, h) in sorted(best.items(), key=lambda kv: -kv[1][0]):
            cat = k.split("|")[0]
            if cat in seen or hits < MIN_DAYS:
                continue
            conf = confidence(hits, len(group))
            if conf < 0.3:
                continue
            seen.add(cat)
            when = "los días de semana" if group is weekdays else "casi todos los días"
            out.append(Pattern(f"app:{cat}", f"{when.capitalize()} {at_hour(h)} abrís {word(cat)}",
                               conf, {"category": cat, "hour": h}))
            if len(out) >= 3:
                break
        return out

    def _late(self, worked, weekdays, weekend, days) -> list[Pattern]:
        # Hours 0-3 of a day record are the night that ends it (the day key
        # is shifted to 04:00, so they belong to the evening before).
        late_days = {k for k, d in days if sum(d["act"][0:4]) >= 15 * 60}
        if len(late_days) < 2:
            return []
        recent = days[-14:]
        hits = sum(1 for k, _ in recent if k in late_days)
        conf = confidence(hits, max(len(recent), 1))
        return [Pattern("late", f"Trasnochás seguido: {hits} de los últimos {len(recent)} días "
                                f"seguiste después de medianoche",
                        max(conf, 0.3) if hits >= 3 else conf, {"nights": hits})]

    def _hours(self, worked, weekdays, weekend, days) -> list[Pattern]:
        group = weekdays if len(weekdays) >= MIN_DAYS else worked
        mins = [sum(d["act"]) / 60.0 for _, d in group]
        med = statistics.median(mins)
        return [Pattern("hours", f"Un día normal estás {_duration(med)} en la compu",
                        confidence(len(group), len(group), _spread(mins), 90.0),
                        {"minutes": med})]

    def _switching(self, worked, weekdays, weekend, days) -> list[Pattern]:
        rates = []
        for _, d in worked:
            hours = sum(d["act"]) / 3600.0
            if hours >= 1:
                rates.append(sum(d["sw"]) / hours)
        if len(rates) < MIN_DAYS:
            return []
        med = statistics.median(rates)
        if med >= 40:
            text = f"Saltás mucho de app en app: unas {int(med)} veces por hora"
        elif med < 2:
            text = "Sos de quedarte en una cosa: casi no cambiás de app"
        elif med <= 10:
            text = f"Sos de quedarte en una cosa: cambiás de app unas {int(med)} veces por hora"
        else:
            text = f"Cambiás de app unas {int(med)} veces por hora"
        return [Pattern("switching", text, confidence(len(rates), len(worked), None) * 0.8,
                        {"per_hour": med})]

    # -- expectations -------------------------------------------------------------
    def expectations(self, ctx: Context | None = None, now: float | None = None
                     ) -> list[Expectation]:
        """What is likely now or next, from the patterns and the live context."""
        now = (ctx.wall if ctx is not None else time.time()) if now is None else now
        dm = day_minute(now)
        lt = time.localtime(now)
        weekday = time.localtime(now - 4 * 3600).tm_wday < 5
        out: list[Expectation] = []
        pats = {p.key: p for p in self.patterns(now, 0.0)}

        lunch = pats.get("lunch")
        if lunch and lunch.confidence >= 0.3 and weekday:
            start = lunch.data["minute"]
            delta = start - dm
            if -60 <= delta <= 45:
                had = ctx is not None and any(
                    day_minute(b[0]) >= start - 60 and b[1] >= 15 * 60 for b in ctx.breaks_today)
                if not had:
                    out.append(Expectation("lunch", f"Por esta hora solés almorzar ({about(start)})",
                                           lunch.confidence, delta))

        breaks = pats.get("breaks")
        if breaks and ctx is not None and ctx.active:
            every = max(30.0, float(breaks.data["every"]))
            worked = ctx.since_break_s / 60.0
            if worked >= every * 0.9:
                out.append(Expectation("break", f"Llevás {_duration(worked)} sin pausa; "
                                       f"solés cortar cada {_duration(every)}",
                                       breaks.confidence, every - worked))

        end = pats.get("end")
        if end and end.confidence >= 0.3:
            delta = end.data["minute"] - dm
            if -30 <= delta <= 60:
                out.append(Expectation("end", f"Solés cortar tipo {about(end.data['minute'])}",
                                       end.confidence, delta))
            elif delta < -90 and ctx is not None and ctx.active:
                out.append(Expectation("late", "Hoy seguís más tarde que de costumbre",
                                       end.confidence, delta))

        for key, p in pats.items():
            if key.startswith("app:") and weekday and p.confidence >= 0.4:
                h = p.data["hour"]
                if h == lt.tm_hour or h == (lt.tm_hour + 1) % 24:
                    in_min = (h - lt.tm_hour) * 60 - lt.tm_min
                    out.append(Expectation(key, f"{at_hour(h).capitalize()} solés abrir {word(p.data['category'])}",
                                           p.confidence, in_min))

        focus = pats.get("focus")
        if focus and focus.confidence >= 0.3 and focus.data["hour"] == lt.tm_hour:
            out.append(Expectation("focus", "Es tu hora de concentrarte", focus.confidence, 0.0))
        return sorted(out, key=lambda e: -e.confidence)
