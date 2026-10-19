"""Reflexes: fixed, declarative rules that turn sensor Events into Reactions.

The mind (``wawi/mind/``) can think about an event; a reflex just *reacts*,
instantly and for free: the website fell over, so Wawi puts on the
firefighter helmet and runs. Rules are data (:data:`DEFAULT_RULES`) and
every one can be changed or switched off from ``config.yaml`` → ``reflexes:``.

Everything a reflex wants to do goes through an **interruption budget**,
because a pet that barks at every headline gets closed:

- at most N reactions per hour per severity;
- ``batch`` rules (news) are collected and delivered as one bubble at most
  every ``batch_minutes`` ("Tenés 4 noticias nuevas");
- **quiet mode** — a full-screen app in front, a meeting, or configured quiet
  hours — holds back everything except ``urgent``; urgent still shows
  (costume/behaviour + bubble) but is never spoken, and makes no sound during
  a meeting or quiet hours (a bark at 3 a.m. is worse than a late notice);
- a reaction that *cancels* an ongoing one (``web.up`` ending the firefighter
  run) is always delivered, silently if need be, so a costume can never get
  stuck on because the "all clear" was over budget;
- a **triage hook** (by default what Wawi learned from her owner,
  ``wawi.learn.triage``) may reshape info/notice events: "now" gives a
  batched kind (a headline the owner cares about) its own bubble, "later"
  batches a kind the owner keeps brushing away, "ignore" drops it. Urgent and
  warning events, and stateful rules (until/cancel keys), never ask.
"""

from __future__ import annotations

import logging
import re
import string
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Any, Callable

from wawi.reactions import BUS, SEVERITIES, Event, Reaction, ReactionBus

log = logging.getLogger(__name__)

SEVERITY_PRIORITY = {"info": 0, "notice": 1, "warning": 2, "urgent": 3}
DEFAULT_BUDGET = {"info": 3, "notice": 6, "warning": 10, "urgent": 30}
MAX_BUBBLE = 140

#: kind -> rule. Fields: behaviour, costume, sound, bubble (template), speak,
#: priority (default from severity), duration, until_key / cancel_key
#: (templates), batch, batch_bubble (template; gets {count} and {title}).
DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "web.down": {
        "behaviour": "run", "costume": "firefighter", "sound": "bark",
        "bubble": "¡Se cayó {host}!", "priority": 3, "until_key": "web:{url}",
    },
    "web.up": {
        "behaviour": "celebrate", "bubble": "{host} volvió", "priority": 2,
        "duration": 4, "cancel_key": "web:{url}",
    },
    "disk.low": {
        "behaviour": "dig",
        "bubble": "Me estoy quedando sin lugar para mis huesos ({free} libres)",
        "priority": 2, "duration": 8,
    },
    "disk.critical": {
        "behaviour": "alarm", "costume": "alarm", "sound": "howl",
        "bubble": "¡El disco está casi lleno! Quedan {free}", "priority": 3,
        "until_key": "disk:{volume}",
    },
    "disk.ok": {"cancel_key": "disk:{volume}"},
    "battery.low": {
        "behaviour": "sad", "sound": "whine",
        "bubble": "Me estoy quedando sin pilas ({percent}%)", "priority": 2,
        "duration": 6, "until_key": "battery",
    },
    "battery.critical": {
        "behaviour": "sad", "sound": "whine",
        "bubble": "¡Batería al {percent}%! Enchufame", "priority": 3,
        "until_key": "battery",
    },
    "battery.charging": {
        "behaviour": "celebrate", "bubble": "¡Gracias por enchufarme!",
        "priority": 0, "duration": 3, "cancel_key": "battery",
    },
    "memory.high": {
        "behaviour": "doctor", "costume": "doctor",
        "bubble": "Tengo la cabeza llena: queda poca memoria ({free_pct}% libre)",
        "priority": 2, "duration": 8,
    },
    "cpu.hot": {
        "behaviour": "doctor", "costume": "doctor",
        "bubble": "Uf, la compu está recalentada", "priority": 2,
        "until_key": "cpu.hot",
    },
    "cpu.cool": {"cancel_key": "cpu.hot"},
    "service.down": {
        "behaviour": "detective", "costume": "detective", "sound": "bark",
        "bubble": "Se cayó {name}", "priority": 2, "until_key": "{key}",
    },
    "service.up": {
        "behaviour": "celebrate", "bubble": "{name} volvió", "priority": 1,
        "duration": 4, "cancel_key": "{key}",
    },
    "weather.rain_soon": {
        "costume": "umbrella", "bubble": "Va a llover a las {hour}",
        "priority": 1, "duration": 1800,
    },
    "weather.hot": {
        "costume": "sunglasses", "bubble": "Hoy hace calor: {max} °C",
        "priority": 0, "duration": 1800,
    },
    "weather.cold": {
        "costume": "scarf", "bubble": "Abrigate: hoy hace {min} °C",
        "priority": 0, "duration": 1800,
    },
    "weather.sunny": {
        "behaviour": "celebrate", "costume": "sunglasses",
        "bubble": "¡Lindo día! {temp} °C y sol", "priority": 0, "duration": 600,
    },
    "rss.item": {
        "behaviour": "newspaper", "bubble": "{title}", "priority": 0,
        "duration": 6, "batch": True, "batch_bubble": "Tenés {count} noticias nuevas",
    },
    "break.suggest": {
        "behaviour": "ball_bring", "bubble": "¿Una pausa? Llevás {duration} sin parar",
        "priority": 1, "duration": 8,
    },
    "clock.morning": {"behaviour": "celebrate", "bubble": "¡Buen día!", "priority": 0, "duration": 3},
    "clock.lunch": {"bubble": "¿Almorzamos?", "priority": 0, "duration": 4},
    "clock.late_night": {
        "behaviour": "sad", "bubble": "Ya es re tarde… ¿y si vamos a dormir?",
        "priority": 0, "duration": 5,
    },
}

#: Names this module may emit, for the body to support (unknown names must
#: degrade gracefully there anyway).
BEHAVIOURS = sorted({r["behaviour"] for r in DEFAULT_RULES.values() if r.get("behaviour")})
COSTUMES = sorted({r["costume"] for r in DEFAULT_RULES.values() if r.get("costume")})
SOUNDS = sorted({r["sound"] for r in DEFAULT_RULES.values() if r.get("sound")})

_RULE_FIELDS = {
    "behaviour", "costume", "sound", "bubble", "speak", "priority", "duration",
    "until_key", "cancel_key", "batch", "batch_bubble",
}
#: Bubble for a batch of events the triage held back from a non-batch rule.
LATER_BUBBLE = "Te guardé {count} avisos para cuando quieras"
#: Triage verdicts (same words as wawi.mind's triage).
NOW, LATER, IGNORE = "now", "later", "ignore"

#: Top-level keys of ``reflexes:`` that are settings, not rules.
_SETTINGS = {
    "enabled", "rules", "budget", "batch_minutes", "batch_gather_s", "dedupe_minutes",
    "quiet_hours", "quiet_fullscreen", "quiet_meetings", "meeting_apps", "meeting_processes",
    # Spoken reactions (wawi/petvoice.py) and the morning greeting.
    "speak_reactions", "speak_per_hour", "speak_stale_s", "speak_morning",
}


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #


class _SafeFormatter(string.Formatter):
    """str.format for user-written templates that can never crash or snoop.

    Only plain ``{name}`` fields: no attribute access (``{x.__class__}``), no
    indexing, no positional fields. A missing field renders empty.
    """

    def get_field(self, field_name: str, args: Any, kwargs: Any) -> Any:
        if not field_name.isidentifier():
            raise KeyError(field_name)
        value = kwargs.get(field_name)
        return ("" if value is None else value), field_name

    def format_field(self, value: Any, format_spec: str) -> str:
        try:
            return super().format_field(value, format_spec)
        except (ValueError, TypeError):
            return str(value)


_FMT = _SafeFormatter()


def render(template: Any, fields: dict[str, Any]) -> str | None:
    """Fill *template* from *fields*. Never raises; None for empty output."""
    if template is None or template is False:
        return None
    text = str(template)
    try:
        out = _FMT.vformat(text, (), fields)
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        # Malformed or forbidden fields: drop them, keep the words.
        out = re.sub(r"\{[^{}]*\}", "", text)
    out = " ".join(out.split())
    if len(out) > MAX_BUBBLE:
        out = out[: MAX_BUBBLE - 1].rstrip() + "…"
    return out or None


def event_fields(event: Event) -> dict[str, Any]:
    fields = dict(event.data or {})
    fields.setdefault("kind", event.kind)
    fields.setdefault("key", event.key)
    fields.setdefault("title", event.title)
    fields.setdefault("severity", event.severity)
    return fields


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def build_rules(cfg: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Defaults overlaid with ``reflexes.rules`` and dotted top-level keys.

    An override merges field by field (``web.down: {sound: null}`` silences
    only the bark); ``false`` or null as the whole value disables the kind.
    """
    cfg = cfg or {}
    rules = {k: dict(v) for k, v in DEFAULT_RULES.items()}
    overrides: dict[str, Any] = {}
    if isinstance(cfg.get("rules"), dict):
        overrides.update(cfg["rules"])
    overrides.update({k: v for k, v in cfg.items() if k not in _SETTINGS and "." in str(k)})
    # "Buen día" out loud is opt-in: a voice from nowhere at 8 a.m. is a lot.
    if "clock.morning" in rules and "speak" not in (overrides.get("clock.morning") or {}):
        rules["clock.morning"]["speak"] = bool(cfg.get("speak_morning", False))
    for kind, override in overrides.items():
        kind = str(kind)
        if override is None or override is False:
            rules.pop(kind, None)
            continue
        if not isinstance(override, dict):
            log.warning("reflexes.%s must be a mapping; ignored", kind)
            continue
        rule = rules.setdefault(kind, {})
        for field_name, value in override.items():
            if field_name in _RULE_FIELDS:
                rule[field_name] = value
            else:
                log.warning("reflexes.%s.%s is not a rule field; ignored", kind, field_name)
    return rules


# --------------------------------------------------------------------------- #
# Quiet mode
# --------------------------------------------------------------------------- #

DEFAULT_MEETING_APPS = (
    "zoom.us", "Microsoft Teams", "Microsoft Teams (work or school)", "Webex",
    "Cisco Webex Meetings", "FaceTime", "Slack Huddle",
)
#: Processes that only exist DURING a call. Zoom's CptHost is the reliable
#: one. (Microphone-in-use is useless here: Wawi itself always holds the mic.)
DEFAULT_MEETING_PROCESSES = ("CptHost",)


@dataclass(frozen=True)
class QuietState:
    quiet: bool = False
    meeting: bool = False
    reason: str = ""


def frontmost_window() -> tuple[str, bool] | None:
    """(owner app name, covers a whole display) of the front normal window.

    Quartz window list: thread-safe, and owner names and bounds need no
    Screen Recording permission (window *titles* would, so they are not read).
    """
    try:
        import Quartz  # type: ignore[import]

        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        ) or []
        front = next((w for w in windows if w.get("kCGWindowLayer") == 0), None)
        if front is None:
            return None
        b = front.get("kCGWindowBounds") or {}
        err, ids, _n = Quartz.CGGetActiveDisplayList(16, None, None)
        full = False
        for did in ids or ():
            r = Quartz.CGDisplayBounds(did)
            if (
                abs(b.get("X", -1) - r.origin.x) < 1 and abs(b.get("Y", -1) - r.origin.y) < 1
                and abs(b.get("Width", 0) - r.size.width) < 1
                and abs(b.get("Height", 0) - r.size.height) < 1
            ):
                full = True
        return str(front.get("kCGWindowOwnerName") or ""), full
    except Exception:
        return None


def _initiative_meeting() -> bool:
    try:
        from wawi.initiative.meeting import in_meeting

        return in_meeting()
    except Exception:
        return False


def _hhmm(text: str) -> dtime | None:
    try:
        h, m = str(text).strip().split(":")
        return dtime(int(h), int(m))
    except (ValueError, TypeError):
        return None


def parse_quiet_hours(raw: Any) -> list[tuple[dtime, dtime]]:
    """'23:00-08:00' or a list of them; ranges may wrap midnight."""
    spans = []
    for item in raw if isinstance(raw, (list, tuple)) else ([raw] if raw else []):
        start, _, end = str(item).partition("-")
        a, b = _hhmm(start), _hhmm(end)
        if a and b:
            spans.append((a, b))
        else:
            log.warning("reflexes.quiet_hours: %r is not HH:MM-HH:MM", item)
    return spans


def in_spans(t: dtime, spans: list[tuple[dtime, dtime]]) -> bool:
    for a, b in spans:
        if a <= b and a <= t < b:
            return True
        if a > b and (t >= a or t < b):
            return True
    return False


class QuietDetector:
    """Is now a bad moment to interrupt? Cached for a few seconds."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        now: Callable[[], datetime] = datetime.now,
        frontmost: Callable[[], tuple[str, bool] | None] = frontmost_window,
        processes: Callable[[], set[str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        cache_s: float = 15.0,
    ) -> None:
        cfg = cfg or {}
        self.spans = parse_quiet_hours(cfg.get("quiet_hours"))
        self.fullscreen = bool(cfg.get("quiet_fullscreen", True))
        self.meetings = bool(cfg.get("quiet_meetings", True))
        self.meeting_apps = {str(a).lower() for a in cfg.get("meeting_apps", DEFAULT_MEETING_APPS) or ()}
        self.meeting_processes = {str(p) for p in cfg.get("meeting_processes", DEFAULT_MEETING_PROCESSES) or ()}
        self._now = now
        self._frontmost = frontmost
        if processes is None:
            from wawi.sensors.services import list_processes

            processes = list_processes
        self._processes = processes
        self._clock = clock
        self._cache_s = cache_s
        self._cached: tuple[float, QuietState] | None = None

    def check(self) -> QuietState:
        t = self._clock()
        if self._cached and t - self._cached[0] < self._cache_s:
            return self._cached[1]
        state = self._compute()
        self._cached = (t, state)
        return state

    def _compute(self) -> QuietState:
        if _initiative_meeting():
            # The initiative's detector (wawi/initiative/meeting.py) sees what
            # the front window cannot: a Meet tab behind, Zoom in the back.
            return QuietState(True, True, "meeting")
        meeting = False
        front = None
        if self.meetings or self.fullscreen:
            front = self._frontmost()
        if self.meetings:
            if front and front[0].lower() in self.meeting_apps:
                meeting = True
            elif self.meeting_processes and (self.meeting_processes & self._processes()):
                meeting = True
        if meeting:
            return QuietState(True, True, "meeting")
        if self.fullscreen and front and front[1]:
            return QuietState(True, False, "fullscreen")
        if self.spans and in_spans(self._now().time(), self.spans):
            return QuietState(True, False, "quiet_hours")
        return QuietState()


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


class Budget:
    """Sliding one-hour window of deliveries per severity."""

    def __init__(self, limits: dict[str, Any] | None = None, window_s: float = 3600.0) -> None:
        self.limits = dict(DEFAULT_BUDGET)
        for sev, value in (limits or {}).items():
            if sev in SEVERITY_PRIORITY:
                self.limits[sev] = None if value is None else max(0, int(value))
        self.window_s = window_s
        self._spent: dict[str, deque[float]] = {s: deque() for s in SEVERITIES}

    def allow(self, severity: str, now: float) -> bool:
        limit = self.limits.get(severity, self.limits["info"])
        if limit is None:
            return True
        spent = self._spent.setdefault(severity, deque())
        while spent and now - spent[0] >= self.window_s:
            spent.popleft()
        return len(spent) < limit

    def spend(self, severity: str, now: float) -> None:
        self._spent.setdefault(severity, deque()).append(now)


# --------------------------------------------------------------------------- #
# The reflex engine
# --------------------------------------------------------------------------- #


@dataclass
class _Batch:
    rule: dict[str, Any]
    items: list[Event]
    first_at: float
    count: int = 0


#: The attached engine, so other parts of Wawi (the pet's voice, the mind) can
#: ask "is now a bad moment?" with the same answer the reflexes use.
_ACTIVE: "Reflexes | None" = None


def quiet_state() -> QuietState | None:
    """The live quiet state, or None while no reflex engine is attached."""
    engine = _ACTIVE
    if engine is None:
        return None
    try:
        return engine._quiet.check()
    except Exception:
        log.debug("quiet check failed", exc_info=True)
        return None


def is_quiet() -> bool | None:
    """True/False, or None when unknown (callers treat unknown as quiet for speech)."""
    state = quiet_state()
    return None if state is None else bool(state.quiet)


class Reflexes:
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        *,
        bus: ReactionBus = BUS,
        clock: Callable[[], float] = time.time,
        quiet: QuietDetector | None = None,
        triage: Callable[[Event], str | None] | None = None,
    ) -> None:
        cfg = dict(cfg or {})
        #: event -> "now" / "later" / "ignore" / None (None = the rules decide).
        self._triage = triage if triage is not None else _learned_triage
        self.rules = build_rules(cfg)
        self._bus = bus
        self._clock = clock
        self._quiet = quiet or QuietDetector(cfg)
        self.budget = Budget(cfg.get("budget") if isinstance(cfg.get("budget"), dict) else None)
        self.batch_s = float(cfg.get("batch_minutes", 20)) * 60
        self.gather_s = float(cfg.get("batch_gather_s", 60))
        self.dedupe_s = float(cfg.get("dedupe_minutes", 10)) * 60
        self._lock = threading.Lock()
        self._active = False
        self._subscribed = False
        self._batches: dict[str, _Batch] = {}
        self._last_batch_delivery: dict[str, float] = {}
        self._last_by_key: dict[str, tuple[str, float]] = {}
        #: kind -> fn(event) -> bubble text or None, for bubbles that need more
        #: than a template (the morning greeting reads the other sensors).
        self._composers: dict[str, Callable[[Event], str | None]] = {}

    def set_composer(self, kind: str, fn: Callable[[Event], str | None] | None) -> None:
        if fn is None:
            self._composers.pop(kind, None)
        else:
            self._composers[kind] = fn

    @property
    def quiet(self) -> QuietDetector:
        return self._quiet

    # -- wiring -------------------------------------------------------------
    def attach(self) -> None:
        # ReactionBus has no unsubscribe; detach() just mutes this listener.
        if not self._subscribed:
            self._bus.subscribe(self.on_event)
            self._subscribed = True
        self._active = True
        global _ACTIVE
        _ACTIVE = self

    def detach(self) -> None:
        self._active = False
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None

    # -- entry points -------------------------------------------------------
    def on_event(self, event: Event) -> None:
        if not self._active:
            return
        # Asked outside the lock: a triage may take a moment, and it must
        # never hold up tick() or another sensor's event.
        verdict = self._ask_triage(event)
        with self._lock:
            self._handle(event, self._clock(), verdict)

    def tick(self) -> None:
        """Deliver due batches. Called periodically by the sensor runner."""
        if not self._active:
            return
        with self._lock:
            self._flush(self._clock())

    # -- core ---------------------------------------------------------------
    def _push(self, reaction: Reaction) -> None:
        log.info("Reflex: %s / %s / %s", reaction.behaviour, reaction.costume, reaction.bubble)
        self._bus.push(reaction)

    def _ask_triage(self, event: Event) -> str | None:
        if event.severity not in ("info", "notice"):
            return None
        rule = self.rules.get(event.kind)
        if rule is None or rule.get("until_key") or rule.get("cancel_key"):
            return None
        try:
            verdict = self._triage(event)
        except Exception:
            log.debug("Reflex triage failed for %s", event.kind, exc_info=True)
            return None
        return verdict if verdict in (NOW, LATER, IGNORE) else None

    def _handle(self, event: Event, now: float, verdict: str | None = None) -> None:
        rule = self.rules.get(event.kind)
        if rule is None:
            return
        severity = event.severity if event.severity in SEVERITY_PRIORITY else "info"
        fields = event_fields(event)

        # The same thing reported twice in a row (a sensor and the mind, or a
        # flapping restart) is one interruption, not two.
        if event.key:
            last = self._last_by_key.get(event.key)
            if last and last[0] == event.kind and now - last[1] < self.dedupe_s:
                return
            self._last_by_key[event.key] = (event.kind, now)
            if len(self._last_by_key) > 2000:
                oldest = sorted(self._last_by_key.items(), key=lambda kv: kv[1][1])[:1000]
                for k, _ in oldest:
                    del self._last_by_key[k]

        cancel_key = render(rule.get("cancel_key"), fields)
        visible = any(rule.get(f) for f in ("behaviour", "costume", "bubble", "sound"))
        if not visible:
            if cancel_key:
                self._push(Reaction(cancel_key=cancel_key, priority=0, source=event))
            return

        if verdict == IGNORE:
            log.info("Reflex for %s dropped (learned: not interesting)", event.kind)
            return
        batched = bool(rule.get("batch"))
        if verdict == NOW:
            batched = False  # this one is worth a bubble of its own
        elif verdict == LATER and not batched:
            batched = True
            rule = {**rule, "batch": True,
                    "batch_bubble": rule.get("batch_bubble") or LATER_BUBBLE}
        if batched:
            batch = self._batches.get(event.kind)
            if batch is None:
                batch = self._batches[event.kind] = _Batch(rule, [], now)
            batch.items.append(event)
            del batch.items[:-50]  # bound memory; the count still says "many"
            batch.count += 1
            self._flush(now)
            return

        quiet = self._quiet.check()
        if (quiet.quiet and severity != "urgent") or not self.budget.allow(severity, now):
            log.info("Reflex for %s held back (%s)", event.kind, quiet.reason or "budget")
            if cancel_key:
                self._push(Reaction(cancel_key=cancel_key, priority=0, source=event))
            return
        self.budget.spend(severity, now)
        bubble = None
        composer = self._composers.get(event.kind)
        if composer is not None:
            try:
                text = composer(event)
                # Through render() only for its whitespace folding and clip.
                bubble = render(text.replace("{", "(").replace("}", ")"), {}) if text else None
            except Exception:
                log.debug("composer for %s failed", event.kind, exc_info=True)
        self._push(self._reaction(rule, fields, severity, quiet, event, cancel_key, bubble))

    def _reaction(
        self, rule: dict[str, Any], fields: dict[str, Any], severity: str,
        quiet: QuietState, event: Event, cancel_key: str | None, bubble: str | None = None,
    ) -> Reaction:
        try:
            priority = int(rule.get("priority", SEVERITY_PRIORITY[severity]))
        except (TypeError, ValueError):
            priority = SEVERITY_PRIORITY[severity]
        try:
            duration = max(0.0, float(rule.get("duration", 0) or 0))
        except (TypeError, ValueError):
            duration = 0.0
        sound = rule.get("sound") or None
        if quiet.meeting or quiet.reason == "quiet_hours":
            sound = None  # never make noise over someone's call, or their sleep
        return Reaction(
            behaviour=rule.get("behaviour") or None,
            costume=rule.get("costume") or None,
            bubble=bubble if bubble is not None else render(rule.get("bubble"), fields),
            speak=bool(rule.get("speak", False)) and not quiet.quiet,
            sound=sound,
            priority=max(0, min(3, priority)),
            duration=duration,
            until_key=render(rule.get("until_key"), fields),
            cancel_key=cancel_key,
            source=event,
        )

    def _flush(self, now: float) -> None:
        for kind, batch in list(self._batches.items()):
            if not batch.items:
                continue
            if now - batch.first_at < self.gather_s:
                continue  # let a burst from one poll arrive first
            last = self._last_batch_delivery.get(kind)
            if last is not None and now - last < self.batch_s:
                continue
            severity = max(
                (e.severity for e in batch.items if e.severity in SEVERITY_PRIORITY),
                key=lambda s: SEVERITY_PRIORITY[s], default="info",
            )
            quiet = self._quiet.check()
            if (quiet.quiet and severity != "urgent") or not self.budget.allow(severity, now):
                continue  # keep collecting; deliver when the moment is right
            count = max(batch.count, len(batch.items))
            newest = batch.items[-1]
            fields = event_fields(newest)
            if count == 1:
                bubble = render(batch.rule.get("bubble"), fields)
            else:
                fields = {**fields, "count": count}
                bubble = render(batch.rule.get("batch_bubble") or batch.rule.get("bubble"), fields)
            self.budget.spend(severity, now)
            self._last_batch_delivery[kind] = now
            del self._batches[kind]
            self._push(self._reaction(batch.rule, fields, severity, quiet, newest, None, bubble))


def _learned_triage(event: Event) -> str | None:
    """The default triage: what Wawi learned from her owner (wawi/learn),
    or None when learning is off or not installed."""
    from wawi import learn

    return learn.triage(event)


def budget_allows(severity: str = "info", spend: bool = False,
                  now: float | None = None) -> bool | None:
    """Does the reflexes' hourly bubble budget allow one more *severity*
    bubble now? With *spend*, also count it. None while no engine is attached.

    For bubbles that do not come from a sensor event (wawi/instinct), so they
    share one budget with the alerts instead of piling on top of them.
    """
    engine = _ACTIVE
    if engine is None:
        return None
    try:
        when = engine._clock() if now is None else now
        with engine._lock:
            if not engine.budget.allow(severity, when):
                return False
            if spend:
                engine.budget.spend(severity, when)
        return True
    except Exception:
        log.debug("budget check failed", exc_info=True)
        return None
