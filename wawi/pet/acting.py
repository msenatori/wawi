"""Acting out a :class:`~wawi.reactions.Reaction`: what to show, and which
reaction gets the stage.

Two pure pieces:

* :func:`plan_for` turns a reaction into a :class:`Plan` — the clip to play,
  the costume loop to wear, how her body moves (patrol, a trip by spaceship,
  a walk to you), the sound and the bubble. Unknown behaviours degrade to a
  generic bark or head tilt; unknown costumes are ignored. Never raises.
* :class:`Director` decides which reaction is on stage: a higher priority
  pre-empts (a long-running one is suspended and resumed afterwards), a
  lower one waits in a short queue, ``cancel_key`` ends any reaction whose
  ``until_key`` matches — wherever it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from wawi.reactions import Reaction

log = logging.getLogger(__name__)

#: Behaviour name → act clip (mascot.ACT_CLIPS / mascot.CLIPS). Names not
#: here are tried as clip names as they are.
BEHAVIOUR_CLIPS: dict[str, str] = {
    "run": "run",
    "bark": "bark",
    "howl": "howl",
    "alarm": "alarm",
    "dig": "dig",
    "newspaper": "newspaper",
    "mail": "mail",
    "ball_bring": "ball_bring",
    "detective": "detective",
    "doctor": "doctor",
    "celebrate": "party",
    "party": "party",
    "cheer": "cheer",
    "sad": "sad",
    "spaceship": "ship_board",
    "walk_to": "",
    "eat": "eat",
    "drink": "drink",
    "play": "play",
}
#: Behaviours whose art may not be there yet (the instinct frames): the
#: first clip she has wins, before the generic bark -- a bark would be the
#: wrong answer to "curl up quietly while they focus".
FALLBACK_CLIPS: dict[str, tuple[str, ...]] = {
    "welcome": ("welcome_a", "cheer", "wag"),
    "curl_focus": ("sleep", "lie_down", "idle"),
    "lunch_ask": ("hungry_bowl", "tilt"),
    "late_night": ("moon_yawn", "yawn"),
    "moon_yawn": ("yawn",),
    "coffee": ("wake", "stretch", "yawn"),
    "think_bubble": ("tilt", "look_around"),
    "pattern_notebook": ("tilt", "look_around"),
    # The games' own art (wawi/games/core.py ART picks these too).
    "giggle": ("tickle", "wag"),
    "smug_win": ("party", "smug"),
    "think_paw_chin": ("tilt",),
    "cover_eyes": ("wag",),
    "rps_rock": ("paw_up", "tilt"),
    "rps_paper": ("paw_up", "tilt"),
    "rps_scissors": ("paw_up", "tilt"),
}
#: What an unknown behaviour becomes, first one she can do.
GENERIC = ("bark", "tilt", "wag")
#: Behaviour → how her body moves for it.
MOVES: dict[str, str] = {
    "run": "patrol",
    "walk_to": "walk_to",
    "newspaper": "to_user",
    "mail": "to_user",
    "ball_bring": "to_user",
    "spaceship": "ship",
}
#: Behaviours that are about a costume as much as a clip: with no behaviour
#: given, the costume alone is worn this long (seconds).
COSTUME_ONLY_HOLD = 20.0
#: A reaction kept "until" something stops actively after this long and
#: just wears its costume, quietly, until it is cancelled.
ACTIVE_FOR = 45.0
#: …and is dropped after this long no matter what (a sensor that never sends
#: its cancel must not leave her in a helmet for a week).
HOLD_MAX = 2 * 3600.0
#: A trip by spaceship with neither a duration nor an until_key.
SHIP_DEFAULT = 25.0

#: Bubble limits.
BUBBLE_MAX = 80
BUBBLE_MIN_S = 3.0
BUBBLE_MAX_S = 8.0


def bubble_text(text: str | None, limit: int = BUBBLE_MAX) -> str | None:
    """*text* tidied for a speech bubble: one line, at most *limit* chars,
    cut on a word boundary with an ellipsis. None for nothing to say."""
    if not text:
        return None
    t = " ".join(str(text).split())
    if not t:
        return None
    if len(t) <= limit:
        return t
    cut = t[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:-–—") + "…"


def bubble_seconds(text: str) -> float:
    """How long a bubble stays: long enough to read, never long."""
    return max(BUBBLE_MIN_S, min(BUBBLE_MAX_S, 1.5 + len(text) / 14.0))


def clip_for_behaviour(name: str | None, clips: Iterable[str]) -> str | None:
    """The clip that acts out behaviour *name*, given the clips she has.

    Known names map through :data:`BEHAVIOUR_CLIPS`; an unknown name is tried
    as a clip name; failing both, a generic bark (or tilt). None only when
    she has none of those either, or *name* is empty.
    """
    if not name:
        return None
    have = set(clips)
    key = str(name).strip().lower()
    want = BEHAVIOUR_CLIPS.get(key, key)
    if want == "" and key in BEHAVIOUR_CLIPS:
        return None  # pure movement (walk_to): no clip of its own
    if want in have:
        return want
    for alt in FALLBACK_CLIPS.get(key, ()):
        if alt in have:
            return alt
    for g in GENERIC:
        if g in have:
            return g
    return None


@dataclass(frozen=True)
class Plan:
    """What acting out one reaction means."""

    clip: str | None = None
    costume: str | None = None
    move: str | None = None
    sound: str | None = None
    bubble: str | None = None
    speak: bool = False
    #: Seconds the reaction lasts at least (0 = as long as its clip/move).
    hold: float = 0.0
    #: Kept until a matching cancel_key (capped at HOLD_MAX).
    until: bool = False

    @property
    def empty(self) -> bool:
        return not (self.clip or self.costume or self.move or self.sound or self.bubble)


def plan_for(r: Reaction, clips: Iterable[str], costumes: Iterable[str] = ()) -> Plan:
    """The :class:`Plan` for reaction *r*.

    *clips* are the clip names she can play; *costumes* the costume names she
    has frames for (see ``mascot.costume_frames``). Never raises.
    """
    try:
        have = set(clips)
        name = (r.behaviour or "").strip().lower() or None
        costume = (r.costume or "").strip().lower() or None
        if costume is not None and costume not in set(costumes):
            costume = None  # a costume she does not have is not worn
        clip = clip_for_behaviour(name, have) if name else None
        move = MOVES.get(name or "")
        hold = max(0.0, float(r.duration or 0.0))
        until = bool(r.until_key)
        if name is None and costume is not None and hold <= 0.0 and not until:
            hold = COSTUME_ONLY_HOLD
        if move == "ship" and hold <= 0.0 and not until:
            hold = SHIP_DEFAULT
        return Plan(
            clip=clip,
            costume=costume,
            move=move,
            sound=(r.sound or "").strip().lower() or None,
            bubble=bubble_text(r.bubble),
            speak=bool(r.speak and r.bubble),
            hold=hold,
            until=until,
        )
    except Exception:
        log.debug("Could not plan reaction %r", r, exc_info=True)
        return Plan()


@dataclass
class Act:
    """A reaction on (or waiting for) the stage."""

    reaction: Reaction
    plan: Plan
    created: float
    started: float | None = None
    #: Seconds it has actually been on stage (suspensions do not count).
    elapsed: float = 0.0
    cancelled: bool = False

    @property
    def priority(self) -> int:
        return int(self.reaction.priority)

    @property
    def long_running(self) -> bool:
        return self.plan.until or self.plan.hold > 5.0

    def over(self) -> bool:
        """Its time is up (one-shots end when their clip does, elsewhere)."""
        if self.cancelled:
            return True
        if self.plan.until:
            return self.elapsed >= HOLD_MAX
        if self.plan.hold > 0.0:
            return self.elapsed >= self.plan.hold
        return False


@dataclass
class Director:
    """Which reaction is on stage. Pure: time comes in with each call.

    * :meth:`offer` a reaction. Higher priority than what is on stage takes
      the stage (a long-running one is suspended, a short one dropped);
      equal priority takes it from a long-running one; anything else queues.
    * ``cancel_key`` ends every act whose ``until_key`` matches — on stage,
      suspended or queued — then the cancelling reaction itself plays, if it
      has anything to show.
    * :meth:`finish` when the act on stage is done; the next one comes on
      (suspended first, then the queue by priority; stale low-priority
      one-shots are dropped).
    """

    QUEUE_MAX = 8
    #: Queued one-shots of priority <= 1 older than this are old news.
    STALE = 120.0

    active: Act | None = None
    suspended: list[Act] = field(default_factory=list)
    queue: list[Act] = field(default_factory=list)
    #: Acts that ended by cancellation while on stage, for the caller to
    #: play their way out (the spaceship comes back).
    ended: list[Act] = field(default_factory=list)

    def offer(self, r: Reaction, plan: Plan, now: float) -> Act | None:
        """Take reaction *r* (planned as *plan*). Returns the act now on
        stage if it changed, else None."""
        changed = False
        if r.cancel_key:
            changed = self.cancel(r.cancel_key)
        if plan.empty:
            return self.active if changed else None
        act = Act(r, plan, created=now)
        cur = self.active
        if cur is None:
            self._start(act, now)
            return act
        if act.priority > cur.priority or (act.priority == cur.priority and cur.long_running
                                           and not act.long_running):
            if cur.long_running:
                self.suspended.append(cur)
            self._start(act, now)
            return act
        self.queue.append(act)
        if len(self.queue) > self.QUEUE_MAX:
            # Drop the least important, oldest first.
            worst = min(self.queue, key=lambda a: (a.priority, -a.created))
            self.queue.remove(worst)
        return self.active if changed else None

    def cancel(self, key: str) -> bool:
        """End every act kept until *key*. True if the one on stage ended."""
        hit = False
        cur = self.active
        if cur is not None and cur.reaction.until_key == key:
            cur.cancelled = True
            self.ended.append(cur)
            self.active = None
            hit = True
        self.suspended = [a for a in self.suspended if a.reaction.until_key != key]
        self.queue = [a for a in self.queue if a.reaction.until_key != key]
        if hit:
            self._next(None)
        return hit

    def tick(self, dt: float) -> None:
        if self.active is not None:
            self.active.elapsed += max(0.0, dt)

    def finish(self, now: float) -> Act | None:
        """The act on stage is done. Returns the next one (or None)."""
        self.active = None
        return self._next(now)

    def _start(self, act: Act, now: float | None) -> None:
        if act.started is None:
            act.started = now if now is not None else act.created
        self.active = act

    def _next(self, now: float | None) -> Act | None:
        if self.suspended:
            self._start(self.suspended.pop(), now)
            return self.active
        if now is not None:
            self.queue = [a for a in self.queue
                          if a.priority >= 2 or a.long_running or now - a.created <= self.STALE]
        if self.queue:
            best = max(self.queue, key=lambda a: (a.priority, -a.created))
            self.queue.remove(best)
            self._start(best, now)
            return best
        return None


# ---------------------------------------------------------------------------
# Sounds
# ---------------------------------------------------------------------------

SOUND_DIR = Path(__file__).resolve().parent.parent / "assets" / "sounds"
SOUND_EXTS = (".aiff", ".aif", ".wav", ".caf", ".m4a", ".mp3")
_sound_missing: set[str] = set()


def sound_path(name: str | None, directory: Path | None = None) -> Path | None:
    """The file for bark/howl/whine…, or None (logged once). Barks are never
    synthesised with the voice: no file, no sound."""
    if not name:
        return None
    clean = "".join(c for c in str(name).lower() if c.isalnum() or c in "_-")
    if not clean:
        return None
    base = directory or SOUND_DIR
    for ext in SOUND_EXTS:
        p = base / f"{clean}{ext}"
        if p.is_file():
            return p
    if clean not in _sound_missing:
        _sound_missing.add(clean)
        log.info("No sound file for %r in %s; staying quiet", clean, base)
    return None
