"""Her life, frame by frame: the glue between needs, the desktop, reactions
and the animator.

The panel calls :meth:`PetLife.tick` on the main thread every frame with an
:class:`Env` (what is on screen, where the caret is, how long since a key was
pressed…) and applies the :class:`Out` it gets back (where her window goes,
the spaceship offset, the bubble, a sound). Nothing here imports AppKit, so
the tests drive whole days of her life with a fake clock and synthetic
windows.

Her idle life is a real dog's (wawi/pet/dog.py): she mostly rests — long
sleeps, light dozes between them, a while awake with her head up glancing at
you, now and then a few minutes up and about — with every number from her
personality (``pet.personality``, files in ``personalities/``). Small things
(the cursor, an app switch, typing) get a habituated glance or an ear flick;
only strong ones (your voice, a click, a drop, a command, an event) wake her.
She greets you by how long you were away, asks for water or food rarely and
politely, plays only on a trigger, and makes a small silent glance about the
app in front (wawi/pet/context.py — its name only, never titles or content).
She moves to a spot near you only when she gets up. Ignored for a while,
she comes over to watch you type. Reactions from the sensors and the mind
take the stage by priority.
Wandering and the trips that move her stop entirely in full-screen apps and
under Reduce Motion; sitting by the panel (the panel is visible) she stays
there and acts in place.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Sequence

from wawi import mascot as _m
from wawi.pet import api as _api
from wawi.pet import world as W
from wawi.pet import context as _context
from wawi.pet import dog as _dog
from wawi.pet import liveliness as _live
from wawi.pet import personality as _personality
from wawi.pet.acting import (
    ACTIVE_FOR,
    Act,
    Director,
    Plan,
    bubble_seconds,
    plan_for,
    sound_path,
)
from wawi.pet.needs import Needs, NeedsStore, mood_bias
from wawi.pet.settings import PetSettings
from wawi.pet.rhythm import profile as rhythm_profile
from wawi.pet.watch import WatchTrigger
from wawi.reactions import Reaction

log = logging.getLogger(__name__)

Point = tuple[float, float]

#: Interactions are reactions too, from the menu or the voice brain.
INTERACTIONS: dict[str, Reaction] = {
    "feed": Reaction(behaviour="eat", priority=2),
    "water": Reaction(behaviour="drink", priority=2),
    "play": Reaction(behaviour="play", priority=2),
}
#: How far an emergency patrol runs each way from where it starts (points).
PATROL_SPAN = 170.0
#: Spaceship: climbing speed out of view, and how long the way back takes.
SHIP_UP = 400.0
SHIP_SIDE = -150.0
SHIP_BACK = 2.6
#: While watching you type, a side-eye every this many seconds.
SIDE_EYE_GAP = (14.0, 32.0)
#: How often she checks she is not sitting on the front window's text (how
#: often she may then move off it is the personality's desktop.cover_every_min:
#: each of those moves can mean the ladder).
COVER_CHECK = 20.0
#: No input for this long: nobody is at the Mac (she gets up less).
OWNER_AWAY = 300.0
#: A key this recently: the owner is working (never play then).
WORKING = 60.0
#: Typing again after this long without a key: a small stimulus.
TYPING_PAUSE = 60.0
#: A cue waiting longer than this for her to be free is dropped (stale).
CUE_STALE = 20.0
#: What she asks for, and how: the clip that looks at the "target", and the
#: quiet bubble of the first try.
REQUESTS = {
    "food": ("hungry_bowl", "¿Comemos algo?"),
    "water": ("hungry_bowl", "¿Un poco de agua?"),
    "play": ("ball_bring", "¿Jugamos un rato?"),
}
#: Commands that answer a request.
ANSWERS = {"feed": "food", "water": "water", "play": "play"}
#: Desktop zoomies: how far each way from her spot (points) and how fast.
ZOOM_SPAN = 160.0
ZOOM_SPEED = 320.0
#: The parachute off a window's edge: only with at least this drop below.
CHUTE_MIN_DROP = 120.0
#: A key pressed this recently (seconds): the user is typing, and her idle
#: activities wait (wawi/mascot.ACTIVITIES).
TYPING_HOLD = 20.0
#: The ladder on demand (``WAWI_ACTIVITY=escalera``, ``trick("escalera")``):
#: not an animator activity but a climb onto a real window.
LADDER_TRICK = "ladder"
#: ``WAWI_ACTIVITY=parachute`` (or a list, or ``all``) makes her do those
#: activities in turn, this many seconds apart: to see each one on demand.
DEMO_ENV = "WAWI_ACTIVITY"
DEMO_GAP = 25.0


@dataclass
class Env:
    """What the panel knows this frame. Coordinates: AppKit global points."""

    now: float
    wall: float
    attached: bool = True
    position: Point | None = None
    size: tuple[float, float] = (124.0, 110.0)
    screens: Sequence[W.Screen] = ()
    windows: Sequence[W.Win] = ()
    world_version: int = 0
    cursor: Point | None = None
    caret: W.Rect | None = None
    key_idle: float = math.inf
    reduce_motion: bool = False
    dragging: bool = False
    quiet: bool = False
    in_game: bool = False
    #: Seconds since ANY input (key, mouse, trackpad); inf when unknown.
    input_idle: float = math.inf
    #: In a meeting (wawi/initiative/meeting.py), and quiet mode known to be on
    #: (``quiet`` above also counts "unknown" as quiet, for speech).
    meeting: bool = False
    hushed: bool = False


@dataclass
class Out:
    """What the panel should do this frame."""

    #: Her ground point (standalone only); None = leave the window alone.
    ground: Point | None = None
    #: Extra offset of her window (the spaceship), in any mode.
    fly: Point = (0.0, 0.0)
    #: Away (in space): do not draw her.
    hidden: bool = False
    bubble: str | None = None
    #: A sound file to play now (once).
    sound: str | None = None
    #: Text to say aloud now (once), for reactions with ``speak``.
    speak: str | None = None
    #: The caret position is wanted (it costs an Accessibility call).
    want_caret: bool = False
    #: Her ladder, if one is standing (or folding away): world.LadderView.
    ladder: W.LadderView | None = None


@dataclass
class _Run:
    """The act on stage, and where it is in its little script."""

    act: Act
    phase: str = "begin"
    t: float = 0.0
    replans: int = 0
    calm: bool = False
    #: Where the ship starts its way back from (its offset when it turned).
    ship_from: Point | None = None


@dataclass
class _Goal:
    point: Point
    ledge: W.Ledge | None
    perch: bool
    purpose: str
    replans: int = 0


class PetLife:
    """See the module docstring. Main thread only, except what goes through
    :mod:`wawi.pet.api`."""

    def __init__(
        self,
        settings: PetSettings,
        anim: _m.MascotAnimator,
        rng: random.Random | None = None,
        store: NeedsStore | None = None,
        wall: float | None = None,
        learner: Any = None,
    ) -> None:
        self.settings = settings
        #: What she learns from the owner (wawi/learn). None = the installed
        #: one, if any, looked up when needed.
        self.learner = learner
        self._now = 0.0
        self.anim = anim
        self.rng = rng or random.Random()
        self.store = store if store is not None else NeedsStore(settings.state_file)
        self.needs: Needs = self.store.load(wall)
        self.director = Director()
        #: Who she is: every number of her dog life (wawi/pet/personality.py).
        self.personality = _personality.load(
            getattr(settings, "personality", "") or _personality.DEFAULT)
        p = self.personality
        # The dog model decides when she sleeps, wakes and does things; the
        # animator's own timers (sleep_after, WAKE_GAP, random picks) are off.
        anim.driven = True
        self.watch = WatchTrigger(lonely_after=settings.lonely_after or math.inf,
                                  rng=self.rng)
        self.presence = _live.Presence(p.greeting_look_after_min * 60.0)
        self.context = _context.ContextGestures(
            (p.context_every_min[0] * 60.0, p.context_every_min[1] * 60.0), rng=self.rng)
        #: The dog (wawi/pet/dog.py) and her requests; made on the first tick
        #: (they need the clock).
        self.dog: _dog.Dog | None = None
        self.requests: _dog.Request | None = None
        self.sense = _dog.Sense()
        self._cues: list[tuple[_dog.Cue, float]] = []
        self._anim_asleep = anim.asleep
        self._key_idle_was = math.inf
        self.body: W.Body | None = None
        self.ledges: list[W.Ledge] = []
        self._world_version = -1
        self._run: _Run | None = None
        self._goal: _Goal | None = None
        self._watch_phase: str | None = None
        self._side_eye_at = 0.0
        self._next_wander: float | None = None
        self._next_cover_check = 0.0
        self._last_cover_move = -1e12
        self._last_key = anim.key
        self.ignored_for = 0.0
        self.mood = self.needs.mood()
        self._mood_at = -1e12
        self._bubble: str | None = None
        self._bubble_until = 0.0
        self._fly = (0.0, 0.0)
        self._hidden = False
        self._pending: list[Reaction] = []
        self._sound: str | None = None
        self._speak: str | None = None
        self._was_attached: bool | None = None
        #: The last frame's Env (activities picked by the animator act on it).
        self._env: Env | None = None
        #: Floating down off a window under the parachute.
        self._chute = False
        self._demo = _demo_list()
        self._demo_i = 0
        self._next_demo: float | None = None
        self._night = False
        #: When she last told you it is late (wall clock); at most every 2 h.
        self._late_said_at = 0.0

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        _api.attach(self.needs.status, self.needs.levels,
                    self._ladder_destination_available, self._at_window)
        self.anim.on_pick = self._picked
        # Her window can now be lifted (jumps, the parachute): see tick().
        self.anim.can_lift = True
        log.info("Personality: %s", self.personality.name)
        if self._demo:
            log.info("%s: she will do %s in turn", DEMO_ENV, ", ".join(self._demo))

    def stop(self, wall: float | None = None) -> None:
        self.store.save(self.needs, wall, force=True)
        _api.attach(None)
        lr = self._learner()
        if lr is not None:
            try:
                lr.save()
            except Exception:
                log.debug("Could not save what she learned", exc_info=True)

    # -- learning (wawi/learn) --------------------------------------------------
    def _learner(self) -> Any:
        if self.learner is not None:
            return self.learner
        from wawi import learn

        return learn.get()

    def _learn(self, method: str, *args: Any) -> Any:
        """Call the learner, if any. Never raises: learning is a bonus."""
        if method == "signal" and args:
            _instinct_note(str(args[0]))
        lr = self._learner()
        if lr is None:
            return None
        try:
            return getattr(lr, method)(*args)
        except Exception:
            log.debug("learn.%s failed", method, exc_info=True)
            return None

    def _picked(self, name: str) -> None:
        self._learn("behaviour_started", name, self._now)
        # Two activities are bigger on the desktop: her body runs the
        # zoomies across her ledge, and jumps off a window's edge with the
        # parachute. Anywhere else the animator's clip plays them in place.
        if name == "zoomies":
            self._desktop_zoomies()
        elif name == "parachute":
            self._desktop_chute()

    def _learned_bias(self, mood: str, env: Env) -> dict[str, float]:
        bias = mood_bias(mood)
        lr = self._learner()
        if lr is None:
            return bias
        try:
            lr.observe(mood=mood, needs=asdict(self.needs.snapshot()),
                       ignored_for=self.ignored_for, key_idle=env.key_idle, wall=env.wall)
            arms = list(_m.BEHAVIOUR_WEIGHTS) + [
                n for n, w in bias.items() if w > 0 and n not in _m.BEHAVIOUR_WEIGHTS]
            return lr.bias(bias, arms)
        except Exception:
            log.debug("learned bias failed", exc_info=True)
            return bias

    def _shown(self, reaction: Reaction, text: str | None, now: float) -> None:
        """A reaction's bubble is up: if a sensor event caused it, the
        learner watches what the owner does next."""
        source = getattr(reaction, "source", None)
        if text and self.settings.bubbles and getattr(source, "kind", None):
            self._learn("alert_shown", source, text, now)

    def dismiss_bubble(self, now: float) -> bool:
        """A click on her bubble: it goes away (and she takes the hint)."""
        if not self._bubble or now >= self._bubble_until:
            return False
        self._bubble_until = now
        self._learn("signal", "bubble_click", now)
        return True

    # -- inputs ----------------------------------------------------------------
    def react(self, reaction: Reaction) -> None:
        """A reaction from the bus (main thread). Planned on the next tick."""
        self._pending.append(reaction)

    def command(self, name: str) -> bool:
        """feed / water / play, from the menu or the voice brain; and
        ``trick:<activity>`` (wawi.pet.api.trick) to see one on demand."""
        if name.startswith("trick:"):
            trick = name[len("trick:"):]
            if trick == LADDER_TRICK:
                return self._desktop_ladder()
            if trick == "ladder_down":
                return self._desktop_ladder_down()
            return self.anim.perform(trick)
        r = INTERACTIONS.get(name)
        if r is None:
            return False
        getattr(self.needs, name)()
        if self.requests is not None and self.requests.answered(self._now, ANSWERS.get(name)):
            log.info("Request answered (%s)", ANSWERS.get(name))
        self._strong("command")
        if name in ("feed", "water"):
            need = "food" if name == "feed" else "water"
            self.needs.set_request_state(need, stage=0.0,
                                         due=self.needs.updated_at + self.rng.uniform(
                                             *self.settings.request_interval))
        self._contact()
        self._pending.append(r)
        self._learn("signal", name, self._now)
        return True

    def touched(self, now: float) -> bool:
        """A click on her. True if the body handled it (do not also pass it
        to the animator)."""
        self.needs.touched()
        self._contact()
        self._strong("touch")
        self._learn("signal", "touch", now)
        if self._watch_phase in ("going", "watching"):
            self._end_watch(now, happy=True)
            return True
        return False

    def petted(self, seconds: float) -> None:
        self.needs.petted(seconds)
        self._contact()
        self._strong("touch")
        self._learn("signal", "pet", self._now)

    def _contact(self) -> None:
        self.ignored_for = 0.0

    def _strong(self, reason: str) -> None:
        """Something that really wakes a dog (wawi/pet/dog.py, rule 5)."""
        if self.dog is not None:
            self.dog.strong(self._now, reason, self.sense)
            if not self.dog.asleep:
                self._cues = [(c, t) for c, t in self._cues
                              if c.kind not in ("lie_down", "head_lift", "ear")]

    # -- the tick ----------------------------------------------------------------
    def tick(self, dt: float, env: Env) -> Out:
        out = Out()
        anim = self.anim
        dt = max(0.0, float(dt))
        self._now = env.now
        self._env = env
        # Muting is not rest: she lives on, she just does not hear.
        resting = anim.asleep
        self.needs.advance(env.wall, asleep=resting)
        (sleep, anim.rhythm_idle_gap, anim.rhythm_wake_gap,
         anim.rhythm_weights, self._night) = rhythm_profile(
            env.wall, enabled=self.settings.rhythm_enabled,
            night_start=self.settings.night_start, night_end=self.settings.night_end)
        anim.rhythm_night = self._night
        if env.now - self._mood_at >= 5.0:
            self._mood_at = env.now
            self.mood = self.needs.mood()
            anim.bias = self._learned_bias(self.mood, env)
        # The animator's own "doze off after" is off (anim.driven): the dog
        # model says when. Kept up to date for anything that reads it.
        anim.sleep_after = sleep
        if anim.key != self._last_key:
            if anim.key == "listening":
                self.needs.talked()
                self._contact()
                self._strong("voice")
                self._learn("signal", "talk", env.now)
            self._last_key = anim.key
        if anim.key == "idle":
            self.ignored_for += dt
        else:
            self.ignored_for = 0.0
        self.store.save(self.needs, env.wall)

        for name in _api.drain():
            self.command(name)

        standalone = not env.attached and env.position is not None
        if standalone != self._was_attached:
            self._was_attached = standalone
            if not standalone:
                self._leave_world(env)
        calm = env.reduce_motion or (standalone and W.fullscreen_active(env.windows, env.screens))
        if standalone:
            self._sync_world(env)

        for r in self._pending:
            plan = plan_for(r, self._clip_names(), self._costumes(r))
            if r.priority <= 0 and self.dog is not None and self.dog.asleep \
                    and not (plan.move or plan.sound or plan.costume):
                # A background thought (an instinct's side-eye, keeping her
                # company): not worth waking a sleeping dog. Its bubble, if
                # any, still shows (silent, like a curiosity).
                self._say(plan.bubble, env.now)
                continue
            if plan.bubble and not (plan.clip or plan.costume or plan.move or plan.sound):
                # Just something to say: a bubble never waits for the stage
                # (she can talk with her helmet on).
                if r.cancel_key:
                    self.director.cancel(r.cancel_key)
                self._say(plan.bubble, env.now)
                self._shown(r, plan.bubble, env.now)
                if plan.speak:
                    self._speak = plan.bubble
                continue
            self.director.offer(r, plan, env.now)
        self._pending.clear()

        self._drive(dt, env, standalone, calm)
        self._dog_tick(env, standalone, calm)
        self._mark_late(env)
        self._learn("tick", env.now)
        if standalone and not env.dragging:
            if not calm:
                self._watching(env)
                self._wander(env)
            self._move(dt, env, calm)
        if self.body is not None and standalone:
            out.ground = (self.body.x, self.body.shown_y)
            out.ladder = self.body.ladder_view()
            led = self.body.ledge
            anim.idle_clip = "sit_edge" if (led is not None and led.wid is not None
                                            and self.body.mode == "rest") else "idle"
        else:
            anim.idle_clip = "idle"
            if anim.locomotion not in (None, "fly"):
                anim.set_locomotion(None)

        # Her idle activities wait while you type, while she is busy with a
        # reaction or a trip, and in full-screen apps (wawi/mascot.ACTIVITIES).
        anim.hold_activities = bool(
            env.key_idle < TYPING_HOLD or calm or self._run is not None
            or self._watch_phase is not None or self.director.active is not None
            or self._chute or (self.body is not None and standalone and self.body.busy))
        self._demo_tick(env)
        # A jump or the parachute lifts her window off the ground.
        out.fly = (self._fly[0], self._fly[1] + anim.lift)
        out.hidden = self._hidden
        if self._bubble and env.now < self._bubble_until and not self._hidden:
            out.bubble = self._bubble
        out.sound, self._sound = self._sound, None
        out.speak, self._speak = self._speak, None
        out.want_caret = standalone and (
            self._watch_phase is not None
            or self.ignored_for >= max(0.0, self.settings.lonely_after - 60.0) > 0.0
            or (self._run is not None and self._run.act.plan.move == "to_user"))
        return out

    # -- helpers -----------------------------------------------------------------
    def _clip_names(self) -> set[str]:
        a = self.anim
        return set(a._clips) | set(a._acts)

    def _costumes(self, r: Reaction) -> list[str]:
        c = (r.costume or "").strip().lower()
        return [c] if c and _m.costume_frames(c, self.anim.frames) else []

    def _say(self, text: str | None, now: float) -> None:
        if text and self.settings.bubbles:
            self._bubble = text
            self._bubble_until = now + bubble_seconds(text)

    def _mark_late(self, env: Env) -> None:
        """Night and you are still at it: a yawn and a quiet "ya es tarde",
        at most once every two hours, never spoken aloud."""
        if not self._night or env.quiet or env.in_game or self.anim.asleep \
                or self.anim.key != "idle" or self.anim.acting \
                or env.key_idle is None or env.key_idle > 120.0 \
                or env.wall - self._late_said_at < 2 * 3600.0 \
                or self._bubble and env.now < self._bubble_until:
            return
        pose = "moon_yawn" if "moon_yawn" in self._clip_names() else "yawn"
        if self.anim.act(pose):
            self._late_said_at = env.wall
            self._say(self.rng.choice((
                "Ya es tarde… ¿vamos a dormir?",
                "Uaaa… es re tarde, zzz",
                "¿Todavía despierto? Yo ya tengo sueño",
            )), env.now)

    # -- the dog: sleep, waking, noticing, greeting, asking --------------------------
    @property
    def deep_asleep(self) -> bool:
        """In deep sleep (not dozing): curiosities and the like wait."""
        return self.dog is not None and self.dog.deep

    def _sense(self, env: Env, calm: bool) -> _dog.Sense:
        anim = self.anim
        local = datetime.fromtimestamp(env.wall)
        idle = env.input_idle
        free = (anim.key == "idle" and not anim.acting and anim.locomotion is None
                and not anim.working and not anim.background and not anim.taking_notes
                and not anim.dancing and not anim.petting and self._run is None
                and self.director.active is None and self._watch_phase is None
                and not self._chute and not env.in_game and not env.dragging
                and not (self.body is not None and self.body.busy))
        return _dog.Sense(
            hour=local.hour + local.minute / 60.0,
            night=self._night,
            owner_here=not (math.isfinite(idle) and idle >= OWNER_AWAY),
            working=env.key_idle < WORKING,
            free=free)

    def _dog_tick(self, env: Env, standalone: bool, calm: bool) -> None:
        """Run the dog model (wawi/pet/dog.py) and show what it says."""
        anim, now = self.anim, env.now
        self.sense = sense = self._sense(env, calm)
        if self.dog is None:
            self.dog = _dog.Dog(self.personality, self.rng, now,
                                state=_dog.DOZE if anim.asleep else _dog.REST)
            self.requests = _dog.Request(self.personality, self.rng, now)
            self._anim_asleep = anim.asleep
        dog = self.dog
        # The animator woke up without the dog asking: a voice turn, a touch,
        # a drop, an act, background work… a strong stimulus.
        if self._anim_asleep and not anim.asleep:
            reason = anim.wake_reason or "other"
            if reason != "dog" and dog.asleep:
                self._strong(reason)
        anim.wake_reason = None
        cues: list[_dog.Cue] = []
        if self.presence.feed(env.input_idle):
            # Whoever comes back is not "ignoring" her.
            anim.ignored_for = 0.0
            if anim.key == "idle" and not env.in_game:
                cues += dog.owner_back(now, self.presence.away_for, sense)
        cues += dog.tick(now, sense)
        self._typing_stimulus(env, sense)
        for cue in cues:
            self._cues.append((cue, now))
        self._play_cues(env, standalone, calm)
        self._request(env, sense)
        self._context_glance(env, calm, sense)
        # Keep the animator where the dog is (her life is the one source).
        if not self._cues and anim.oneshot_free:
            if dog.asleep and not anim.asleep:
                anim.lie_down()
            elif not dog.asleep and anim.asleep:
                anim.get_up("dog", ())
        self._anim_asleep = anim.asleep

    def _typing_stimulus(self, env: Env, sense: _dog.Sense) -> None:
        """Typing again after a pause: a small, habituated stimulus."""
        was, self._key_idle_was = self._key_idle_was, env.key_idle
        if not (env.key_idle < 2.0 and was >= TYPING_PAUSE) or self.dog is None:
            return
        response = self.dog.notice(env.now, "typing", sense)
        self._small_response(response, env.now, toward=self._toward_owner(env))

    def _small_response(self, response: str | None, now: float,
                        toward: float | None = None) -> bool:
        """Show the dog's answer to a small stimulus (see Dog.notice)."""
        if response == "ear":
            return self.anim.ear_flick()
        if response == "lift":
            return self.anim.head_lift(self.rng.uniform(*self.personality.wake_head_lift_s))
        if response == "look":
            return self.anim.gesture(("glance",), toward=toward)
        return False

    def _toward_owner(self, env: Env) -> float | None:
        """x offset from her to the owner's attention: the cursor, else the
        front window's middle (+ right)."""
        x = self.body.x if self.body is not None else (
            env.position[0] if env.position is not None else None)
        if x is None:
            return None
        if env.cursor is not None:
            return env.cursor[0] - x
        front = W.frontmost(env.windows)
        if front is not None:
            return front.rect[0] + front.rect[2] / 2.0 - x
        return None

    def _play_cues(self, env: Env, standalone: bool, calm: bool) -> None:
        """Play the dog's cues in order, each when she is free (a cue left
        waiting too long is old news and dropped)."""
        anim, now = self.anim, env.now
        while self._cues:
            cue, at = self._cues[0]
            if now - at > CUE_STALE:
                self._cues.pop(0)
                continue
            if not (self.sense.free and anim.oneshot_free):
                return
            self._cues.pop(0)
            toward = self._toward_owner(env)
            kind = cue.kind
            if kind == "lie_down":
                if not anim.lie_down():
                    self._cues.insert(0, (cue, at))
                    return
            elif kind == "head_lift":
                anim.head_lift(cue.seconds)
            elif kind == "ear":
                anim.ear_flick()
            elif kind == "get_up":
                anim.get_up("dog", cue.names, toward=toward)
            elif kind == "glance":
                anim.gesture((f"glance@{cue.seconds:.1f}",), toward=toward)
            elif kind == "gesture":
                anim.gesture(cue.names, toward=toward)
            elif kind in ("behave", "play"):
                anim.pick(cue.names)
            elif kind == "relocate":
                if standalone and not calm:
                    self._relocate(env)
            if not anim.oneshot_free:
                return  # the next one waits for this one to end

    def cursor_reaction(self, reaction: str | None, now: float) -> str | None:
        """The cursor came near (wawi/cursor_reactions.py): what she shows,
        through the dog. Habituated; asleep, at most an ear flick (deep) or
        a head lift (dozing), never woken by it."""
        if reaction is None or self.dog is None:
            return reaction
        response = self.dog.notice(now, "cursor", self.sense)
        if response is None:
            return None
        if response == "ear":
            return "ear_twitch"
        if response == "lift":
            self._small_response("lift", now)
            return None
        if self.anim.asleep:
            return None
        return reaction if reaction in ("wag", "bark", "growl") else None

    def _context_glance(self, env: Env, calm: bool, sense: _dog.Sense) -> None:
        """A silent glance about the app in front (wawi/pet/context.py): only
        resting and awake, habituated, rare. Only its name is looked at, and
        only its category is logged."""
        anim = self.anim
        front = W.frontmost(env.windows)
        owner = front.owner if front is not None else None
        resting = self.dog is not None and self.dog.state == _dog.REST
        allowed = resting and sense.free and anim.oneshot_free and not (
            calm or env.reduce_motion or env.meeting or env.hushed or self._night
            or W.fullscreen_active(env.windows, env.screens))
        category = self.context.sample(env.now, owner, allowed)
        if category is None or self.dog is None:
            return
        if self.dog.notice(env.now, "app", sense) != "look":
            return
        names = _context.GESTURES[category]
        if category == "code" and self.rng.random() < self.personality.context_laptop_chance:
            names = _context.LAPTOP
        if anim.gesture(names, toward=self._toward_owner(env), seconds=4.0):
            log.info("Context glance: %s", category)

    # -- asking for water, food, play (wawi/pet/dog.py, rule 13) ----------------
    def _due_need(self, env: Env) -> str | None:
        """What her needs' clock says she would ask for now, if anything."""
        for name in ("food", "water"):
            state = self.needs.request_state(name)
            due = state.get("due")
            if due is None:
                self.needs.set_request_state(
                    name, stage=0.0,
                    due=env.wall + self.rng.uniform(*self.settings.request_interval))
                continue
            if env.wall >= due:
                return name
        if self.needs.snapshot().fun < 20.0:
            return "play"
        return None

    def _may_ask(self, env: Env) -> bool:
        return (self.settings.requests_enabled and not self._night and not env.quiet
                and not env.in_game and not env.meeting and self.dog is not None
                and self.dog.state == _dog.REST and self.requests is not None
                and not (self._bubble and env.now < self._bubble_until))

    def _request(self, env: Env, sense: _dog.Sense) -> None:
        """Start an episode when one is due, try now and then, give up."""
        req, now = self.requests, env.now
        if req is None or self.dog is None:
            return
        if req.need is None:
            if req.can_start(now) and sense.free and self._may_ask(env):
                need = self._due_need(env)
                if need is not None:
                    req.start(now, need)
                    log.info("Asking for %s", need)
            return
        need = req.need
        if req.gave_up(now):
            # Nobody answered: a sigh, and she lies down.
            log.info("Request ignored (%s): she lies down", need)
            self._need_answered(need, env)
            for cue in self.dog.settle(now, "ignored", sense):
                self._cues.append((cue, now))
            return
        if req.due(now) and sense.free and self.anim.oneshot_free and not self.anim.asleep \
                and not self._cues:
            self._try_request(env, need)

    def _try_request(self, env: Env, need: str) -> bool:
        """One try: look at you, at the bowl (or the ball), back at you."""
        req = self.requests
        target, bubble = REQUESTS[need]
        toward = self._toward_owner(env)
        if req is None or not self.anim.gesture(("glance", target, "glance"), toward=toward):
            return False
        req.tried(env.now)
        if not req.said:
            req.said = True  # the bubble once per episode, never spoken
            self._say(bubble, env.now)
        return True

    def _need_answered(self, need: str, env: Env) -> None:
        if need in ("food", "water"):
            self.needs.set_request_state(need, stage=0.0, due=env.wall + self.rng.uniform(
                *self.settings.request_interval))

    def ask_need(self, now: float) -> bool:
        """Hungry or thirsty and the cursor comes over: she may start asking
        (the same rare, polite episode as on her own). False when she did
        not (the caller wags instead)."""
        env, req = self._env, self.requests
        if env is None or req is None or self.mood != "hungry" \
                or not req.can_start(now) or not self._may_ask(env) \
                or self.anim.asleep or not self.anim.oneshot_free:
            return False
        snap = self.needs.snapshot()
        need = "food" if snap.food <= snap.water else "water"
        req.start(now, need)
        log.info("Asking for %s", need)
        return self._try_request(env, need)

    # -- the world -----------------------------------------------------------------
    def _sync_world(self, env: Env) -> None:
        if env.world_version != self._world_version or not self.ledges:
            self._world_version = env.world_version
            self.ledges = W.ledges(env.screens, env.windows, env.size)
        if self.body is None:
            x, y = env.position  # type: ignore[misc]
            self.body = W.Body(x, y)
            if not self.settings.gravity_on_drop:
                self.body.mode = "perch"
        if env.dragging:
            x, y = env.position  # type: ignore[misc]
            self.body.place(x, y)
            self._goal = None
            self._chute = False
            if self._watch_phase is not None:
                self._end_watch(env.now, happy=False)
            if not self.settings.gravity_on_drop:
                self.body.mode = "perch"

    def _leave_world(self, env: Env) -> None:
        """Back by the panel: every desktop trip ends where it is."""
        if self._watch_phase is not None:
            self._end_watch(env.now, happy=False)
        self.body = None
        self._goal = None
        self.anim.set_locomotion(None)

    def _move(self, dt: float, env: Env, calm: bool) -> None:
        body = self.body
        if body is None:
            return
        anim = self.anim
        # Her ladder, when the art is there; else she climbs the old way.
        body.ladders = _m.ladder_ready(anim.frames, anim.props)
        # A voice turn: she stops where she is to listen (falling still falls).
        body.paused = anim.key != "idle" or calm
        if calm and body.mode == "fall":
            # Reduce Motion / full screen: no tumbling across the screen.
            led = W.landing(self.ledges, body.x, body.y)
            if led is not None:
                body.y, body.mode, body.ledge = led.y, "rest", led
        event = body.update(dt, self.ledges)
        if event == "landed" and self._goal is not None:
            self._replan(env)
        elif event == "arrived" and self._goal is not None:
            goal, self._goal = self._goal, None
            self._arrived(goal, env)
        elif event == "fell" and self._goal is not None:
            pass  # re-planned on landing
        if event == "arrived" and body.ledge is not None and body.mode == "rest":
            # Got to the end of where she can stand: that is a wall, so she
            # turns round and faces back in instead of staring into it.
            led = body.ledge
            if body.x - led.x0 <= 2.0 and led.x1 - led.x0 > 8.0:
                body.facing_right = True
            elif led.x1 - body.x <= 2.0 and led.x1 - led.x0 > 8.0:
                body.facing_right = False
        if self._chute:
            if event == "landed":
                self._chute = body.chute = False
                anim.act("chute_end")  # landed: fold the parachute
            elif event == "arrived" and W.support(self.ledges, body.x, body.y) is not None:
                self._chute = body.chute = False  # nothing to jump off after all
        # Keep her on a screen whatever happens (a monitor was unplugged).
        leg = body.legs[0] if body.legs else None
        crossing = leg is not None and leg.kind in ("shift", "walk_off")
        # Not while she hops the gap between monitors or steps off an edge,
        # and not while a ledge holds her: pulling her back there is what
        # left her walking on the spot at a screen's side.
        if body.mode in ("rest", "walk", "trot") and env.screens and not crossing \
                and W.support(self.ledges, body.x, body.y) is None:
            x, y = W.clamp_to_screens(env.screens, body.x, body.y, env.size)
            if (x, y) != (body.x, body.y) and body.mode != "fall":
                body.x = x
                if y > body.y + 0.5:
                    body.y = y
        loco = {"walk": "walk", "trot": "trot", "climb": "climb", "fall": "fall",
                "land": "land"}.get(body.mode)
        lad = body.ladder
        if body.carrying_ladder:
            loco = "ladder_carry"
        elif lad is not None and lad.phase in ("set", "up", "fold", "held"):
            if body.mode == "climb":
                loco = "ladder"
            elif body.mode == "ladder_set":
                loco = "ladder_set"
            elif body.mode == "step_off":
                loco = "ladder_step"
            elif body.mode == "ladder_fold" and lad.phase == "held":
                loco = "ladder_held"  # folded small by her paws, then poof
            elif body.mode == "ladder_fold":
                loco = "ladder_pull" if body.y > lad.bottom + 1.0 else "ladder_stow"
        if loco is not None and not anim.has_locomotion(loco):
            loco = {"ladder": "climb", "ladder_carry": "walk", "ladder_step": "walk"}.get(loco)
        if loco == "fall" and body.chute and anim.has_locomotion("chute"):
            loco = "chute"
        run = self._run
        if loco == "trot" and run is not None and run.phase == "patrol" \
                and run.act.plan.costume:
            loco = None  # the costume's own run cycle is the act's clip
        if self._fly != (0.0, 0.0) and anim.locomotion == "fly":
            loco = "fly"
        anim.set_locomotion(loco)
        anim.facing_right = body.facing_right

    def _go(self, target: Point, ledge: W.Ledge | None, env: Env, purpose: str,
            perch: bool = False) -> bool:
        body = self.body
        if body is None:
            return False
        legs = W.plan_path(body, target, ledge, self.ledges, env.size, perch=perch)
        if not legs:
            return False
        self.anim.wake_up(purpose)
        body.go(legs)
        self._goal = _Goal(target, ledge, perch, purpose)
        return True

    def _replan(self, env: Env) -> None:
        goal = self._goal
        body = self.body
        if goal is None or body is None:
            return
        goal.replans += 1
        if goal.replans > 3:
            self._goal = None
            self._arrived(goal, env)
            return
        legs = W.plan_path(body, goal.point, goal.ledge, self.ledges, env.size,
                           perch=goal.perch)
        body.go(legs)

    def _arrived(self, goal: _Goal, env: Env) -> None:
        if goal.purpose == "watch" and self._watch_phase == "going":
            self._watch_phase = "watching"
            self.watch.started(env.now)
            self.anim.act("watch", loop=True)
            self._side_eye_at = env.now + self.rng.uniform(*SIDE_EYE_GAP)
        elif goal.purpose == "act" and self._run is not None and self._run.phase == "going":
            self._run.phase = "clip"
            self._run.t = 0.0
        elif goal.purpose == "zoomies":
            self.anim.act("zoomies_end")  # skid, and pant

    # -- wandering --------------------------------------------------------------------
    def _free_to_move(self, body: W.Body) -> bool:
        return (self._run is None and self._watch_phase is None and not body.busy
                and self.anim.key == "idle" and body.mode in ("rest",)
                and self.anim.behaviour not in _m.ACTIVITIES and not self._chute)

    def _wander(self, env: Env) -> None:
        """Sitting on the front window's text while awake: she moves off it
        (at most every desktop.cover_every_min). Never while she sleeps: a
        dog does not sleepwalk. Her other moves come from the dog getting
        up (:meth:`_relocate`)."""
        body = self.body
        if body is None or not self.settings.wander or not self._free_to_move(body):
            return
        if self.dog is None or self.dog.asleep or self.anim.asleep:
            return
        if env.now >= self._next_cover_check:
            self._next_cover_check = env.now + COVER_CHECK
            front = W.frontmost(env.windows)
            gap = self.personality.desktop_cover_every_min * 60.0
            if front is not None and env.now - self._last_cover_move >= gap:
                if W.intersects(W.her_rect(body.x, body.y, env.size), front.content):
                    if self._wander_now(env):
                        self._last_cover_move = env.now

    def _relocate(self, env: Env) -> bool:
        """The dog got up and wants another spot, near the owner. Rationed by
        pet.wander_every_min (and off with pet.wander false)."""
        body = self.body
        if body is None or not self.settings.wander or not self._free_to_move(body):
            return False
        if self._next_wander is not None and env.now < self._next_wander:
            return False
        if not self._wander_now(env):
            return False
        self._next_wander = env.now + self.rng.uniform(*self.settings.wander_every)
        return True

    def _wander_now(self, env: Env) -> bool:
        body = self.body
        if body is None:
            return False
        front = W.frontmost(env.windows)
        avoid = W.avoid_rects(front, env.cursor, env.caret)
        near = env.cursor
        if near is None and front is not None:
            near = (front.rect[0] + front.rect[2] / 2.0, front.rect[1])
        p = self.personality
        spot = W.choose_spot(self.ledges, body.pos, env.size, avoid, self.rng, env.screens,
                             top_weight=p.desktop_top_weight,
                             same_level_weight=p.desktop_same_level_weight,
                             near=near, near_weight=p.desktop_near_owner)
        if spot is None:
            return False
        ledge, x = spot
        return self._go((x, ledge.y), ledge, env, "wander")

    # -- idle activities on the desktop ------------------------------------------------
    def _standalone_body(self) -> W.Body | None:
        """Her body, when she is alone on the desktop, standing still and
        free to go (else None: the activity plays in place)."""
        env, body = self._env, self.body
        if env is None or body is None or env.attached or env.position is None:
            return None
        if env.reduce_motion or env.dragging or body.busy or body.mode != "rest" \
                or body.ledge is None:
            return None
        return body

    def _desktop_zoomies(self) -> bool:
        """The zoomies with her whole body: a mad dash to one end of her
        ledge, to the other, back home; then she skids and pants."""
        body = self._standalone_body()
        if body is None:
            return False
        led, x = body.ledge, body.x
        a, b = led.clamp(x - ZOOM_SPAN), led.clamp(x + ZOOM_SPAN)  # type: ignore[union-attr]
        if b - a < 60.0:
            return False
        self.anim.hand_over()
        body.go([W.Leg("walk", a, ZOOM_SPEED), W.Leg("walk", b, ZOOM_SPEED),
                 W.Leg("walk", x, ZOOM_SPEED)])
        self._goal = _Goal((x, led.y), led, False, "zoomies")  # type: ignore[union-attr]
        return True

    def _desktop_chute(self) -> bool:
        """Sitting on a window's top edge: walk to its end and jump off with
        the parachute; her body floats down to whatever is below."""
        body = self._standalone_body()
        if body is None or self.env_size is None:
            return False
        led = body.ledge
        if led is None or led.wid is None:
            return False
        w = self.env_size[0]
        left = abs(body.x - led.x0) <= abs(body.x - led.x1)
        edge = led.x0 - w * 0.9 if left else led.x1 + w * 0.9
        below = W.landing(self.ledges, edge, led.y - 3.0)
        if below is None or led.y - below.y < CHUTE_MIN_DROP:
            return False
        self.anim.hand_over()
        body.go([W.Leg("walk", led.clamp(edge)), W.Leg("walk_off", edge)])
        body.chute = True
        self._chute = True
        return True

    def _desktop_ladder(self) -> bool:
        """The ladder on demand: only up onto a real window's top edge."""
        body = self._standalone_body()
        env = self._env
        if body is None or env is None:
            return False
        here = body.ledge
        ups = [led for led in self.ledges
               if led.wid is not None and led.y > body.y + 40.0
               and here is not None and here.x0 - 1 <= led.x0 - env.size[0] * 0.4 <= here.x1 + 1]
        if ups:
            led = min(ups, key=lambda e: abs(e.x0 - body.x))
            x = led.clamp(led.x0 + env.size[0])
            return self._go((x, led.y), led, env, "wander")
        return False

    def _ladder_destination_available(self) -> bool:
        body, env = self._standalone_body(), self._env
        if body is None or env is None or body.ledge is None:
            return False
        here = body.ledge
        return any(led.wid is not None and led.y > body.y + 40.0
                   and here.x0 - 1 <= led.x0 - env.size[0] * .4 <= here.x1 + 1
                   for led in self.ledges)

    def _at_window(self) -> bool:
        body = self.body
        return bool(body and body.ledge and body.ledge.wid is not None and not body.busy)

    def _desktop_ladder_down(self) -> bool:
        """Descend from a window edge using the same ladder art, if possible."""
        body, env = self._standalone_body(), self._env
        if body is None or env is None or body.ledge is None or body.ledge.wid is None:
            return False
        below = [led for led in self.ledges if led.y < body.y - 40
                 and led.x0 <= body.x <= led.x1]
        if not below or not body.ladders:
            return False
        ground = max(below, key=lambda led: led.y)
        body.go([W.ladder_leg(ground.y, env.size, body.facing_right),
                 W.Leg("climb", ground.y), W.Leg("ladder_fold")])
        return True

    @property
    def env_size(self) -> tuple[float, float] | None:
        return self._env.size if self._env is not None else None

    def _demo_tick(self, env: Env) -> None:
        """WAWI_ACTIVITY: one activity after another, to see them all."""
        if not self._demo or self.anim.key != "idle":
            return
        if self._next_demo is None:
            self._next_demo = env.now + 5.0
        if env.now < self._next_demo or self._run is not None or self._chute \
                or self.anim.acting or self.anim.behaviour in _m.ACTIVITIES \
                or (self.body is not None and self.body.busy):
            return
        name = self._demo[self._demo_i % len(self._demo)]
        self._demo_i += 1
        self._next_demo = env.now + DEMO_GAP
        if name == LADDER_TRICK:
            log.info("%s: %s%s", DEMO_ENV, name,
                     "" if self._desktop_ladder() else " (not now: by the panel or busy)")
        elif self.anim.perform(name):
            log.info("%s: %s", DEMO_ENV, name)
        else:
            log.info("%s: she cannot do %s (art missing?)", DEMO_ENV, name)

    # -- watching you type ------------------------------------------------------------
    def _watching(self, env: Env) -> None:
        body = self.body
        if body is None or self.settings.lonely_after <= 0.0:
            return
        now = env.now
        if self._watch_phase is None:
            # Only an awake dog comes over: never out of her sleep.
            if self._run is not None or body.busy or self.anim.key != "idle" \
                    or self.dog is None or self.dog.asleep or self.anim.asleep:
                self.watch.typing.feed(now, env.key_idle)
                return
            if not self.watch.should_go(now, self.ignored_for, env.key_idle):
                return
            spot = self._caret_spot(env)
            if spot is None:
                self.watch.left(now)  # nowhere to sit: try again after the cooldown
                return
            self._watch_phase = "going"
            if not self._go(spot, None, env, "watch", perch=True):
                self._watch_phase = None
                self.watch.left(now)
            return
        if self._watch_phase == "watching":
            if self.watch.should_leave(now, env.key_idle):
                self._end_watch(now, happy=False)
                return
            if not self.anim.acting:
                self.anim.act("watch", loop=True)
            if now >= self._side_eye_at:
                self._side_eye_at = now + self.rng.uniform(*SIDE_EYE_GAP)
                self.anim.act(self.rng.choice(("watch_side_eye", "watch_eyeroll")))

    def _caret_spot(self, env: Env) -> Point | None:
        caret = env.caret
        if caret is None:
            front = W.frontmost(env.windows)
            if front is not None:
                x, y, w, h = front.content
                caret = (x + w * 0.6, y + min(h * 0.5, 160.0), 2.0, 16.0)
            elif env.cursor is not None:
                caret = (env.cursor[0], env.cursor[1], 2.0, 16.0)
            else:
                return None
        avoid = W.avoid_rects(None, env.cursor, None)
        return W.caret_spot(caret, env.size, env.screens, avoid)

    def _end_watch(self, now: float, happy: bool) -> None:
        phase = self._watch_phase
        self._watch_phase = None
        self.watch.left(now)
        if self.anim.act_name in ("watch", "watch_side_eye", "watch_eyeroll"):
            self.anim.stop_act()
        if self._goal is not None and self._goal.purpose == "watch":
            self._goal = None
            if self.body is not None:
                self.body.stop()
        if self.body is not None and phase in ("watching", "going"):
            self.body.leave_perch()  # and gravity takes her down
        if happy:
            self.anim.act("petted")

    # -- reactions on stage -----------------------------------------------------------
    def _drive(self, dt: float, env: Env, standalone: bool, calm: bool) -> None:
        run = self._run
        active = self.director.active
        if run is not None and run.act is not active:
            # Cancelled or pre-empted: play the way out (the ship comes back).
            if run.phase in ("ship_out", "away"):
                run.phase, run.t = "ship_in", 0.0
                run.act.cancelled = True
            elif run.phase not in ("ship_in", "ship_return_start", "ship_return"):
                self._end_run(env, keep=not run.act.cancelled)
                run = None
        if run is None:
            if active is None or self.anim.key != "idle":
                return
            if self._watch_phase is not None:
                self._end_watch(env.now, happy=False)
            run = self._run = _Run(active, calm=calm)
            if active.elapsed > 0.0:
                run.phase = "resume"
        if self.anim.key == "idle" and run.act is active:
            self.director.tick(dt)
        run.t += dt
        self._step(run, env, standalone, calm)

    def _step(self, run: _Run, env: Env, standalone: bool, calm: bool) -> None:
        act, plan, anim = run.act, run.act.plan, self.anim
        if run.phase == "begin":
            self._say(plan.bubble, env.now)
            self._shown(act.reaction, plan.bubble, env.now)
            if plan.speak:
                self._speak = plan.bubble
            if plan.sound and self.settings.sounds:
                path = sound_path(plan.sound)
                self._sound = str(path) if path else None
            if plan.move == "ship":
                run.phase = "ship_board"
                if not (anim.act("ship_board")):
                    run.phase = "ship_out"
                run.t = 0.0
                return
            if plan.move in ("to_user", "walk_to") and standalone and not calm \
                    and self.body is not None:
                target = self._act_target(plan, act, env)
                if target is not None and self._go(target[0], target[1], env, "act",
                                                   perch=target[2]):
                    run.phase, run.t = "going", 0.0
                    return
            run.phase, run.t = "clip", 0.0
        if run.phase == "going":
            if self._goal is None or self._goal.purpose != "act":
                run.phase, run.t = "clip", 0.0  # got there, or gave up
            return
        if run.phase == "clip":
            started = False
            if plan.clip and plan.move != "patrol":
                started = anim.act(plan.clip)
            run.phase, run.t = ("clip_wait" if started else "loop_start"), 0.0
            return
        if run.phase == "clip_wait":
            if not anim.acting:
                run.phase, run.t = "loop_start", 0.0
            return
        if run.phase in ("loop_start", "resume"):
            if plan.move == "patrol" and not calm:
                self._start_patrol(run, env, standalone)
                run.phase = "patrol"
            elif plan.costume:
                clip = _m.costume_clip(plan.costume, anim.frames)
                if clip is not None:
                    anim.act(clip, loop=True)
                run.phase = "wear"
            elif plan.move == "ship":
                run.phase = "away"
            elif plan.clip and (plan.hold > 0 or plan.until):
                run.phase = "repeat"
            else:
                run.phase = "done"
            run.t = 0.0
        if run.phase == "repeat":
            # A timed bark: keep at it for the duration (an "until" one only
            # for a while — then she has made her point).
            if act.over() or (plan.until and act.elapsed >= ACTIVE_FOR):
                run.phase = "done"
            elif not anim.acting and not anim.act(plan.clip or ""):
                run.phase = "done"
        if run.phase in ("patrol", "wear"):
            if run.phase == "patrol" and plan.until and act.elapsed >= ACTIVE_FOR:
                # Long emergency: stop running about, keep the costume on.
                self._calm_patrol(run)
            timed = plan.until or plan.hold > 0.0
            if act.over():
                run.phase = "done"
            elif not timed and run.phase == "wear" and not anim.acting:
                run.phase = "done"
            elif not timed and run.phase == "patrol" and run.t > 8.0:
                run.phase = "done"
        # -- the spaceship ------------------------------------------------------------
        if run.phase == "ship_board":
            if not anim.acting:
                run.phase, run.t = "ship_out", 0.0
            return
        if run.phase == "ship_out":
            if calm:
                run.phase, run.t = "away_still", 0.0
                return
            anim.set_locomotion("fly")
            anim.facing_right = False
            self._fly = (SHIP_SIDE * run.t, SHIP_UP * run.t)
            top = self._space_height(env)
            if self._fly[1] >= top:
                self._hidden = True
                anim.set_locomotion(None)
                run.phase, run.t = "away", 0.0
            return
        if run.phase in ("away", "away_still"):
            if act.over() or act.cancelled:
                run.phase, run.t = ("ship_in" if run.phase == "away" else "ship_return_start"), 0.0
            return
        if run.phase == "ship_in":
            if self._fly == (0.0, 0.0) and run.ship_from is None:
                run.phase, run.t = "ship_return_start", 0.0
                return
            if run.ship_from is None:
                run.ship_from = self._fly
                run.t = 0.0
            self._hidden = False
            anim.set_locomotion("fly")
            anim.facing_right = True
            k = min(1.0, run.t / SHIP_BACK)
            k = k * k * (3 - 2 * k)
            fx, fy = run.ship_from
            self._fly = (fx * (1 - k), fy * (1 - k))
            if k >= 1.0:
                self._fly = (0.0, 0.0)
                anim.set_locomotion(None)
                run.phase, run.t = "ship_return_start", 0.0
            return
        if run.phase == "ship_return_start":
            self._hidden = False
            self._fly = (0.0, 0.0)
            anim.set_locomotion(None)
            run.phase = "ship_return" if anim.act("ship_return") else "done"
            return
        if run.phase == "ship_return":
            if not anim.acting:
                run.phase = "done"
            else:
                return
        if run.phase == "done":
            self._end_run(env, keep=False)

    def _space_height(self, env: Env) -> float:
        """How far up she must fly to be out of view."""
        if env.position is None or not env.screens:
            return 900.0
        x, y = env.position
        i = W.screen_index(env.screens, x, y)
        fx, fy, fw, fh = env.screens[i if i is not None else 0].frame
        return max(200.0, fy + fh - y + 40.0)

    def _act_target(self, plan: Plan, act: Act, env: Env) -> tuple[Point, W.Ledge | None, bool] | None:
        body = self.body
        if body is None:
            return None
        if plan.move == "walk_to":
            data: dict[str, Any] = dict(getattr(act.reaction.source, "data", None) or {})
            try:
                x, y = float(data["x"]), float(data["y"])
                led = W.support(self.ledges, x, y) or W.landing(self.ledges, x, y)
                if led is not None:
                    return ((led.clamp(x), led.y), led, False)
            except (KeyError, TypeError, ValueError):
                pass
            avoid = W.avoid_rects(W.frontmost(env.windows), env.cursor, env.caret)
            spot = W.choose_spot(self.ledges, body.pos, env.size, avoid, self.rng, env.screens)
            return ((spot[1], spot[0].y), spot[0], False) if spot else None
        spot = self._caret_spot(env)
        return (spot, None, True) if spot is not None else None

    def _start_patrol(self, run: _Run, env: Env, standalone: bool) -> None:
        anim, plan = self.anim, run.act.plan
        costume = _m.costume_clip(plan.costume, anim.frames, running=True) if plan.costume \
            else None
        if standalone and self.body is not None and self.body.ledge is not None:
            x = self.body.x
            self.body.start_patrol(x - PATROL_SPAN, x + PATROL_SPAN)
            if costume is not None:
                anim.act(costume, loop=True)
        else:
            # By the panel: the dash (window moves a little), on a loop.
            clip = _m.CLIPS["run"]
            if costume is not None:
                frames = [s.frame for s in costume.steps]
                clip = _m.Clip("run", tuple(s._replace(frame=frames[i % len(frames)])
                                            for i, s in enumerate(clip.steps)),
                               loop=True, xfade=clip.xfade)
            anim.act(clip, loop=True)

    def _calm_patrol(self, run: _Run) -> None:
        if self.body is not None:
            self.body.stop()
        run.phase = "wear"
        clip = _m.costume_clip(run.act.plan.costume, self.anim.frames) \
            if run.act.plan.costume else None
        if clip is not None:
            self.anim.act(clip, loop=True)
        else:
            self.anim.stop_act()

    def _end_run(self, env: Env, keep: bool) -> None:
        """The act on stage is over (*keep*: it was only suspended)."""
        run = self._run
        self._run = None
        self.anim.stop_act()
        self._fly = (0.0, 0.0)
        self._hidden = False
        if self.anim.locomotion == "fly":
            self.anim.set_locomotion(None)
        if self.body is not None:
            if self.body.patrol is not None:
                self.body.stop()
            if self.body.mode == "perch" and run is not None \
                    and run.act.plan.move in ("to_user",):
                self.body.leave_perch()
        if self._goal is not None and self._goal.purpose == "act":
            self._goal = None
            if self.body is not None:
                self.body.stop()
        if not keep and run is not None and self.director.active is run.act:
            self.director.finish(env.now)


def _demo_list() -> list[str]:
    """The activities named in ``WAWI_ACTIVITY`` (``all``, or a comma list,
    Spanish names too), or [] when it is not set."""
    raw = os.environ.get(DEMO_ENV, "").strip().lower()
    if not raw:
        return []
    if raw in ("all", "todas", "todo", "1"):
        return list(_m.ACTIVITIES)
    return [_api.TRICK_ALIASES.get(n.strip(), n.strip()) for n in raw.split(",") if n.strip()]


def _instinct_note(kind: str) -> None:
    """Voice turns and touches feed the owner's day (wawi/instinct). Never raises."""
    if kind not in ("talk", "touch", "pet"):
        return
    try:
        from wawi import instinct

        instinct.note(kind)
    except Exception:
        log.debug("instinct note failed", exc_info=True)
