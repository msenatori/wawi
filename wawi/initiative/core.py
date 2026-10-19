"""The initiative loop: moments in, acts out, through a budget and what she learned.

A **watcher** looks at one local signal every tick and returns
:class:`Moment` objects ("the meeting just ended", "an Orca agent finished",
"the owner just arrived this morning"). For each moment the same watcher
builds an :class:`Act`: a bubble line, maybe a clickable link, what the
click does, what to prepare silently. The engine then decides HOW, from the
act kind's level (:mod:`wawi.initiative.learning`):

=========  ===============================================================
``quiet``  only the silent part (prepare a board page, set a state).
``offer``  the silent part plus a bubble with a link; no voice.
``say``    the offer, and the line said aloud (rare: one per hour).
``auto``   the click is done for the owner ("te abrí el resumen").
=========  ===============================================================

Everything visible goes through :class:`Budget`: never during a turn, a
note, a game or a meeting; at most ~3 offers and ~1 spoken line per hour;
nothing spoken at night or in quiet mode; and only while the owner is at the
keyboard (an act waits, up to its ``ttl_s``, for a good moment). One bubble
at a time; when it goes away the outcome (accepted / ignored / dismissed)
teaches the level.

Adding a moment = one watcher class with ``kinds``, ``poll(ctx)`` and
``act(moment, ctx)``, registered with :func:`register`, plus its row in
:data:`KINDS`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from wawi.initiative.learning import RANK, LearningStore

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Kinds, moments, acts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Kind:
    name: str
    #: Spanish, for the menu and the voice tool ("te aviso cuando…").
    title: str
    #: Where it starts.
    default: str = "offer"
    #: How high learning may take it ("say", "auto", or the default itself).
    top: str = "offer"


#: Every act kind the initiative knows. A new moment adds its row here.
KINDS: dict[str, Kind] = {
    "meeting_quiet": Kind("meeting_quiet", "me callo cuando estás en una reunión", "quiet", "quiet"),
    "meeting_notes": Kind("meeting_notes", "te ofrezco anotar lo que quedó de una reunión"),
    "orca_done": Kind("orca_done", "te aviso cuando un agente de Orca termina", top="say"),
    "orca_waiting": Kind("orca_waiting", "te aviso cuando un agente de Orca te espera", top="say"),
    "morning": Kind("morning", "te dejo el resumen de la mañana", top="auto"),
}


@dataclass(frozen=True)
class Moment:
    kind: str
    #: Identifies this occurrence (one act per key).
    key: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Act:
    kind: str
    #: The bubble's line.
    text: str
    #: The clickable last line ("Tomar nota…"), or "" for none.
    link: str = ""
    #: What a click on the link (or level ``auto``) does. Called on a thread.
    accept: Callable[[], None] | None = None
    #: The silent part, done at any level but off. Called on the loop thread.
    prepare: Callable[[], None] | None = None
    #: Polled while the bubble is up: the owner did it some other way (by voice).
    done_if: Callable[[], bool] | None = None
    #: Said aloud at level ``say`` (default: *text*).
    say: str = ""
    #: The bubble at level ``auto`` (the act was done for the owner).
    auto_text: str = ""
    #: She walks over to where the owner works first.
    walk: bool = False
    #: Seconds it may wait for a good moment before it is dropped.
    ttl_s: float = 1800.0
    #: Seconds the bubble stays.
    show_s: float = 30.0
    key: str = ""
    #: Say it even at level offer: the owner was away when it happened.
    speak_on_return: bool = False
    created: float = 0.0


@dataclass(frozen=True)
class Ctx:
    """The owner and Wawi right now, as the watchers and the budget see them."""

    now: float
    #: Seconds since the last key / mouse event (1e9 = unknown or locked).
    idle_s: float = 0.0
    #: "turn", "notes", "game" or None.
    busy: str | None = None
    meeting: bool = False
    #: Reflexes' quiet mode (full screen, quiet hours) or unknown.
    quiet: bool = False
    #: Night by her rhythm (wawi/pet/rhythm).
    night: bool = False
    #: The owner just arrived: first activity of the (4 a.m.) day, or back
    #: after ``away_h`` hours. Never because Wawi started.
    arrived: bool = False
    #: When the owner was last seen before this look (persisted), or None.
    last_active: float | None = None

    @property
    def present(self) -> bool:
        return self.idle_s < PRESENT_IDLE_S


#: At the keyboard = an input event within this many seconds.
PRESENT_IDLE_S = 120.0
#: Away long enough that a finished agent is worth saying when they are back.
AWAY_S = 300.0


class Watcher(Protocol):
    """``poll`` / ``act`` run on the loop thread every tick; an optional
    ``sense(now)`` runs before the tick's :class:`Ctx` is built (the meeting
    watcher uses it to set :attr:`Initiative.meeting_active`)."""

    kinds: tuple[str, ...]

    def poll(self, ctx: Ctx) -> list[Moment]: ...

    def act(self, moment: Moment, ctx: Ctx) -> Act | None: ...


#: name -> (order, factory(engine) -> Watcher or None). Lower orders poll
#: first: the morning page reads what the Orca watcher saw this tick.
REGISTRY: dict[str, tuple[int, Callable[["Initiative"], Watcher | None]]] = {}


def register(name: str, order: int = 50) -> Callable[[Callable[["Initiative"], Watcher | None]], Any]:
    def deco(factory: Callable[["Initiative"], Watcher | None]) -> Any:
        REGISTRY[name] = (order, factory)
        return factory
    return deco


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InitiativeSettings:
    enabled: bool = True
    tick_s: float = 25.0
    offers_per_hour: int = 3
    says_per_hour: int = 1
    local_file: str = "initiative_local.json"
    #: Meeting signals and how long without one ends it (seconds).
    meeting_browsers: bool = True
    meeting_zoom: bool = True
    meeting_front_apps: bool = True
    meeting_exit_s: float = 60.0
    #: A meeting shorter than this gets no "¿anoto lo que quedó?" (minutes).
    meeting_min_min: float = 5.0
    orca: bool = True
    orca_every_s: float = 30.0
    #: The morning window (local hours) and what counts as "back" (hours away).
    morning_from_h: float = 6.0
    morning_until_h: float = 12.0
    away_h: float = 4.0
    night_start: float = 21.0
    night_end: float = 7.0

    @classmethod
    def from_dict(cls, raw: Any) -> "InitiativeSettings":
        d = cls()
        raw = raw if isinstance(raw, dict) else {}
        meeting = raw.get("meeting") if isinstance(raw.get("meeting"), dict) else {}
        morning = raw.get("morning") if isinstance(raw.get("morning"), dict) else {}

        def num(src: dict, key: str, default: float, lo: float, hi: float) -> float:
            try:
                return min(hi, max(lo, float(src.get(key, default))))
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=bool(raw.get("enabled", d.enabled)),
            tick_s=num(raw, "tick_s", d.tick_s, 5.0, 600.0),
            offers_per_hour=int(num(raw, "offers_per_hour", d.offers_per_hour, 0, 30)),
            says_per_hour=int(num(raw, "says_per_hour", d.says_per_hour, 0, 10)),
            local_file=str(raw.get("local_file", d.local_file) or ""),
            meeting_browsers=bool(meeting.get("browsers", d.meeting_browsers)),
            meeting_zoom=bool(meeting.get("zoom", d.meeting_zoom)),
            meeting_front_apps=bool(meeting.get("front_apps", d.meeting_front_apps)),
            meeting_exit_s=num(meeting, "exit_after_s", d.meeting_exit_s, 0.0, 3600.0),
            meeting_min_min=num(meeting, "min_minutes", d.meeting_min_min, 0.0, 600.0),
            orca=bool(raw.get("orca", d.orca)),
            orca_every_s=num(raw, "orca_every_s", d.orca_every_s, 5.0, 3600.0),
            morning_from_h=num(morning, "from_hour", d.morning_from_h, 0.0, 24.0),
            morning_until_h=num(morning, "until_hour", d.morning_until_h, 0.0, 24.0),
            away_h=num(morning, "away_hours", d.away_h, 0.5, 24.0),
            night_start=num(raw, "night_start", d.night_start, 0.0, 24.0),
            night_end=num(raw, "night_end", d.night_end, 0.0, 24.0),
        )


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


class Budget:
    """When a visible act may happen, and how many per hour."""

    def __init__(self, offers_per_hour: int = 3, says_per_hour: int = 1,
                 window_s: float = 3600.0) -> None:
        self.limits = {"offer": int(offers_per_hour), "say": int(says_per_hour)}
        self.window_s = window_s
        self._spent: dict[str, deque[float]] = {"offer": deque(), "say": deque()}

    def blocked(self, ctx: Ctx) -> str | None:
        """Why nothing visible may happen now, or None."""
        if ctx.busy:
            return ctx.busy
        if ctx.meeting:
            return "meeting"
        if not ctx.present:
            return "away"
        return None

    def _left(self, what: str, now: float) -> int:
        spent = self._spent[what]
        while spent and now - spent[0] >= self.window_s:
            spent.popleft()
        return self.limits[what] - len(spent)

    def allows(self, level: str, ctx: Ctx) -> bool:
        if level == "quiet":
            return True
        if self.blocked(ctx) is not None:
            return False
        if level == "say":
            if ctx.night or ctx.quiet:
                return False
            return self._left("say", ctx.now) > 0 and self._left("offer", ctx.now) > 0
        return self._left("offer", ctx.now) > 0

    def spend(self, level: str, now: float) -> None:
        if level in ("offer", "say", "auto"):
            self._spent["offer"].append(now)
        if level == "say":
            self._spent["say"].append(now)


# --------------------------------------------------------------------------- #
# Hooks into the rest of Wawi
# --------------------------------------------------------------------------- #


@dataclass
class Hooks:
    """What the orchestrator lends the initiative. All optional."""

    #: "turn" / "notes" / "game" / None.
    busy: Callable[[], str | None] = lambda: None
    start_note: Callable[[], None] | None = None
    notes_active: Callable[[], bool] = lambda: False
    #: The board (wawi/board.Board) for pages.
    board: Any = None
    #: Says a line through the reaction voice (PetVoice: queue, quiet, echo).
    speak: Callable[[str], None] | None = None
    #: Titles of the issues she filed (wawi/selfimprove).
    issues: Callable[[], list[str]] = lambda: []
    #: Where notes go (for the morning's "última nota").
    notes_dir: Any = None


def _input_idle() -> float:
    """Seconds since the last key or mouse event; 1e9 when locked/unknown."""
    try:
        import Quartz  # type: ignore[import]

        session = Quartz.CGSessionCopyCurrentDictionary() or {}
        if session.get("CGSSessionScreenIsLocked", False):
            return 1e9
        src = Quartz.kCGEventSourceStateCombinedSessionState
        since = Quartz.CGEventSourceSecondsSinceLastEventType
        return float(min(since(src, Quartz.kCGEventKeyDown),
                         since(src, Quartz.kCGEventMouseMoved),
                         since(src, Quartz.kCGEventLeftMouseDown),
                         since(src, Quartz.kCGEventScrollWheel)))
    except Exception:
        return 1e9


def _reflex_quiet() -> bool:
    try:
        from wawi import reflexes

        state = reflexes.quiet_state()
        return bool(state.quiet) if state is not None else False
    except Exception:
        return False


def _walk_over(kind: str) -> None:
    """She trots over to where the owner works (the caret, else the cursor):
    the ``mail`` behaviour moves her "to_user" (wawi/pet/acting.MOVES)."""
    try:
        from wawi.reactions import BUS, Event, Reaction

        BUS.push(Reaction(behaviour="mail", priority=1,
                          source=Event(kind=f"initiative.{kind}", key=f"initiative:{kind}")))
    except Exception:
        log.debug("initiative: walk over failed", exc_info=True)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


@dataclass
class _Shown:
    act: Act
    level: str
    until: float


class Initiative:
    """The loop thread, the pending acts, the one bubble, the controls."""

    def __init__(
        self,
        settings: InitiativeSettings | None = None,
        *,
        store: LearningStore | None = None,
        hooks: Hooks | None = None,
        clock: Callable[[], float] = time.time,
        idle: Callable[[], float] = _input_idle,
        quiet: Callable[[], bool] = _reflex_quiet,
        walk: Callable[[str], None] = _walk_over,
        background: bool = True,
        watchers: list[Watcher] | None = None,
    ) -> None:
        self.settings = settings or InitiativeSettings()
        self.store = store or LearningStore(None)
        self.hooks = hooks or Hooks()
        self.budget = Budget(self.settings.offers_per_hour, self.settings.says_per_hour)
        self._clock = clock
        self._idle = idle
        self._quiet = quiet
        self._walk = walk
        self._background = background
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.pending: list[Act] = []
        self.current: _Shown | None = None
        self.seen_keys: deque[str] = deque(maxlen=200)
        self.last_kind: tuple[str, float] | None = None
        self.history: deque[tuple[float, str, str]] = deque(maxlen=50)
        self._saved_active = 0.0
        #: Set by the meeting watcher's ``sense`` (before each look).
        self.meeting_active = False
        if watchers is None:
            watchers = []
            for name, (_order, factory) in sorted(REGISTRY.items(), key=lambda kv: kv[1][0]):
                try:
                    w = factory(self)
                except Exception:
                    log.warning("initiative: watcher %s could not start", name, exc_info=True)
                    w = None
                if w is not None:
                    watchers.append(w)
        self.watchers = watchers

    # -- switches ----------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.settings.enabled if self.store.enabled is None else self.store.enabled

    def set_enabled(self, on: bool) -> None:
        self.store.enabled = bool(on)
        self.store.save()
        if not on:
            self.dismiss_all()
            from wawi.initiative.meeting import set_in_meeting

            set_in_meeting(False)
        log.info("initiative: %s", "on" if on else "off")

    def level(self, kind: str) -> str:
        k = KINDS.get(kind)
        if k is None or not self.enabled:
            return "off"
        return self.store.level(kind, k.default)

    def set_kind(self, kind: str, on: bool) -> bool:
        k = KINDS.get(kind)
        if k is None:
            return False
        self.store.set_owner(kind, on, k.default)
        log.info("initiative: %s %s by the owner", kind, "on" if on else "off")
        if not on:
            with self._lock:
                self.pending = [a for a in self.pending if a.kind != kind]
                if self.current is not None and self.current.act.kind == kind:
                    self.current = None
            if kind == "meeting_quiet":
                from wawi.initiative.meeting import set_in_meeting

                set_in_meeting(False)
        return True

    def recent_kind(self, within_s: float = 1800.0) -> str | None:
        """The kind of the last act she showed, if recent (for "eso")."""
        last = self.last_kind
        if last is None or self._clock() - last[1] > within_s:
            return None
        return last[0]

    # -- the loop ----------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="wawi-initiative", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.store.save()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                log.warning("initiative: step failed", exc_info=True)
            self._stop.wait(self.settings.tick_s)

    def context(self, now: float) -> Ctx:
        """One look at the owner. Updates the persisted "last seen"."""
        try:
            idle = float(self._idle())
        except Exception:
            idle = 1e9
        try:
            busy = self.hooks.busy()
        except Exception:
            busy = None
        meeting = self.meeting_active
        from wawi.pet.rhythm import profile

        night = profile(now, night_start=self.settings.night_start,
                        night_end=self.settings.night_end)[4]
        last = self.store.last_active
        arrived = False
        if idle < PRESENT_IDLE_S:
            from wawi.instinct.observer import day_key

            arrived = last is None or day_key(last) != day_key(now) or \
                now - last >= self.settings.away_h * 3600.0
            seen = now - min(idle, now)
            self.store.last_active = seen
            if arrived or now - self._saved_active >= 300.0:
                self._saved_active = now
                self.store.save()
        return Ctx(now=now, idle_s=idle, busy=busy, meeting=meeting, quiet=self._safe_quiet(),
                   night=night, arrived=arrived, last_active=last)

    def _safe_quiet(self) -> bool:
        try:
            return bool(self._quiet())
        except Exception:
            return False

    def step(self, now: float | None = None) -> Ctx | None:
        """One tick: sense, collect moments, settle the bubble, deliver."""
        now = self._clock() if now is None else now
        if not self.enabled:
            return None
        for w in self.watchers:
            sense = getattr(w, "sense", None)
            if sense is not None:
                try:
                    sense(now)
                except Exception:
                    log.debug("initiative: %s sense failed", type(w).__name__, exc_info=True)
        ctx = self.context(now)
        for w in self.watchers:
            try:
                moments = w.poll(ctx)
            except Exception:
                log.debug("initiative: %s poll failed", type(w).__name__, exc_info=True)
                continue
            for m in moments:
                if m.key and m.key in self.seen_keys:
                    continue
                if m.key:
                    self.seen_keys.append(m.key)
                if self.level(m.kind) == "off":
                    log.info("initiative: %s (off)", m.kind)
                    continue
                try:
                    act = w.act(m, ctx)
                except Exception:
                    log.debug("initiative: %s act failed", m.kind, exc_info=True)
                    act = None
                if act is not None:
                    act.key = act.key or m.key
                    self.enqueue(act, ctx)
        self._settle(ctx)
        self._deliver(ctx)
        return ctx

    # -- acts --------------------------------------------------------------------
    def enqueue(self, act: Act, ctx: Ctx) -> None:
        act.created = ctx.now
        if not act.key:
            act.key = f"{act.kind}:{ctx.now:.0f}"
        if ctx.idle_s >= AWAY_S:
            act.speak_on_return = True
        level = self.level(act.kind)
        if level != "off" and act.prepare is not None:
            try:
                act.prepare()
            except Exception:
                log.warning("initiative: preparing %s failed", act.kind, exc_info=True)
        if level in ("off", "quiet"):
            self._note(ctx.now, act.kind, level)
            return
        with self._lock:
            self.pending = [a for a in self.pending if a.key != act.key] + [act]

    def _note(self, now: float, kind: str, what: str) -> None:
        self.history.append((now, kind, what))
        log.info("initiative: %s → %s", kind, what)

    def _settle(self, ctx: Ctx) -> None:
        with self._lock:
            shown = self.current
        if shown is None:
            return
        done = False
        if shown.act.done_if is not None:
            try:
                done = bool(shown.act.done_if())
            except Exception:
                done = False
        if done:
            self._finish("accepted")
        elif ctx.now >= shown.until:
            self._finish("ignored")

    def _finish(self, outcome: str) -> None:
        with self._lock:
            shown, self.current = self.current, None
        if shown is None or shown.level == "auto":
            return
        k = KINDS.get(shown.act.kind)
        if k is None:
            return
        self._note(self._clock(), shown.act.kind, outcome)
        self.store.record(shown.act.kind, outcome, default=k.default, top=k.top)

    def _deliver(self, ctx: Ctx) -> None:
        with self._lock:
            self.pending = [a for a in self.pending if ctx.now - a.created <= a.ttl_s]
            if self.current is not None or not self.pending:
                return
            act = self.pending[0]
        level = self.level(act.kind)
        if level in ("off", "quiet"):
            with self._lock:
                self.pending.remove(act)
            return
        if act.speak_on_return and level == "offer":
            level = "say"
        if level == "say" and not self.budget.allows("say", ctx):
            level = "offer"  # spoken is rarer; the bubble may still go
        if not self.budget.allows(level, ctx):
            return  # wait for a better moment (or the act's ttl)
        with self._lock:
            if act in self.pending:
                self.pending.remove(act)
        self.budget.spend(level, ctx.now)
        self.last_kind = (act.kind, ctx.now)
        self.store.shown(act.kind)
        if act.walk:
            self._walk(act.kind)
        if level == "auto":
            self._run(act.accept)
            text = act.auto_text or act.text
            with self._lock:
                self.current = _Shown(Act(act.kind, text, key=act.key), "auto",
                                      ctx.now + min(act.show_s, 12.0))
            self._note(ctx.now, act.kind, "auto")
            return
        with self._lock:
            self.current = _Shown(act, level, ctx.now + act.show_s)
        self._note(ctx.now, act.kind, level)
        if level == "say" and self.hooks.speak is not None:
            line = act.say or act.text
            self._spawn(self.hooks.speak, line)

    def _run(self, fn: Callable[[], None] | None) -> None:
        if fn is not None:
            self._spawn(fn)

    def _spawn(self, fn: Callable[..., None], *args: Any) -> None:
        def safe() -> None:
            try:
                fn(*args)
            except Exception:
                log.warning("initiative: an act failed", exc_info=True)

        if not self._background:
            safe()
            return
        threading.Thread(target=safe, name="wawi-initiative-act", daemon=True).start()

    # -- the bubble (main thread, from the panel) ------------------------------------
    def bubble_text(self, now: float | None = None) -> str | None:
        now = self._clock() if now is None else now
        with self._lock:
            shown = self.current
        if shown is None or now >= shown.until:
            return None
        act = shown.act
        return f"{act.text}\n{act.link}" if act.link else act.text

    def bubble_link(self, text: str | None) -> str | None:
        """The link line of *text* when it is her bubble right now."""
        with self._lock:
            shown = self.current
        if shown is None or not shown.act.link or not text:
            return None
        return shown.act.link if text.endswith("\n" + shown.act.link) else None

    def owns(self, text: str | None) -> bool:
        with self._lock:
            shown = self.current
        return bool(text) and shown is not None and text in (
            shown.act.text, f"{shown.act.text}\n{shown.act.link}")

    def click(self, on_link: bool) -> None:
        """Her bubble was clicked: on the link = accepted, else dismissed."""
        with self._lock:
            shown = self.current
        if shown is None:
            return
        if on_link and shown.act.link:
            self._run(shown.act.accept)
            self._finish("accepted")
        else:
            self._finish("dismissed")

    def dismiss_all(self) -> None:
        with self._lock:
            self.pending.clear()
            self.current = None

    # -- the voice tool / menu -------------------------------------------------------
    def describe(self) -> list[tuple[str, str, str]]:
        """(kind, Spanish title, level) for every kind."""
        return [(k.name, k.title, self.level(k.name)) for k in KINDS.values()]
