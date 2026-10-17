"""Her life, frame by frame: the glue between needs, the desktop, reactions
and the animator.

The panel calls :meth:`PetLife.tick` on the main thread every frame with an
:class:`Env` (what is on screen, where the caret is, how long since a key was
pressed…) and applies the :class:`Out` it gets back (where her window goes,
the spaceship offset, the bubble, a sound). Nothing here imports AppKit, so
the tests drive whole days of her life with a fake clock and synthetic
windows.

What she does on her own is deliberately rare and calm: she mostly sleeps
(the animator's resting look). Every so often she gets up and wanders to
another spot — a window's top edge, the other end of the screen — and goes
back to sleep there. Ignored for a long while, she comes over to watch you
type. Reactions from the sensors and the mind take the stage by priority.
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
from typing import Any, Sequence

from wawi import mascot as _m
from wawi.pet import api as _api
from wawi.pet import world as W
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
#: How often she checks she is not sitting on the front window's text.
COVER_CHECK = 20.0
COVER_MOVE_GAP = 180.0
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
        self.watch = WatchTrigger(lonely_after=settings.lonely_after or math.inf,
                                  rng=self.rng)
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
        self._learn("signal", "touch", now)
        if self._watch_phase in ("going", "watching"):
            self._end_watch(now, happy=True)
            return True
        return False

    def petted(self, seconds: float) -> None:
        self.needs.petted(seconds)
        self._contact()
        self._learn("signal", "pet", self._now)

    def _contact(self) -> None:
        self.ignored_for = 0.0

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
        (anim.sleep_after, anim.rhythm_idle_gap, anim.rhythm_wake_gap,
         anim.rhythm_weights, self._night) = rhythm_profile(
            env.wall, enabled=self.settings.rhythm_enabled,
            night_start=self.settings.night_start, night_end=self.settings.night_end)
        anim.rhythm_night = self._night
        if env.now - self._mood_at >= 5.0:
            self._mood_at = env.now
            self.mood = self.needs.mood()
            anim.bias = self._learned_bias(self.mood, env)
        if anim.key != self._last_key:
            if anim.key == "listening":
                self.needs.talked()
                self._contact()
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
        self._request_needs(env)
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

    def _request_needs(self, env: Env) -> None:
        """A food and water clock each; pose first, then one quiet bubble."""
        if not self.settings.requests_enabled or self._night \
                or self.anim.key != "idle" or self.anim.acting or env.quiet or env.in_game \
                or self._run is not None or self.director.active is not None \
                or self._watch_phase is not None or self._chute \
                or self.anim.behaviour in _m.ACTIVITIES \
                or (self.body is not None and self.body.busy) \
                or self._bubble and env.now < self._bubble_until:
            return
        for name, pose, bubble in (("food", "hungry_bowl", "¿Comemos algo?"),
                                   ("water", "sniff", "¿Un poco de agua?")):
            state = self.needs.request_state(name)
            due = state.get("due")
            if due is None:
                self.needs.set_request_state(
                    name, stage=0.0,
                    due=env.wall + self.rng.uniform(*self.settings.request_interval))
                continue
            if env.wall < due:
                continue
            stage = int(state.get("stage", 0))
            if stage == 0:
                if self.anim.act(pose):
                    self.needs.set_request_state(name, stage=1.0, due=env.wall + 3600.0)
                    return
            elif stage == 1:
                self._say(bubble, env.now)
                self.needs.set_request_state(name, stage=2.0,
                                             due=env.wall + self.rng.uniform(
                                                 *self.settings.request_interval))
                return
            else:
                # A single reminder every few hours, never spoken aloud.
                self._say(bubble, env.now)
                self.needs.set_request_state(name, due=env.wall + self.rng.uniform(
                                             *self.settings.request_interval))
                return

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
        if body.mode in ("rest", "walk", "trot") and env.screens:
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
        self.anim.wake_up()
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
    def _wander(self, env: Env) -> None:
        body = self.body
        if body is None or not self.settings.wander:
            return
        if self._next_wander is None:
            self._next_wander = env.now + self.rng.uniform(*self.settings.wander_every)
        free = (self._run is None and self._watch_phase is None and not body.busy
                and self.anim.key == "idle" and body.mode in ("rest",)
                and self.anim.behaviour not in _m.ACTIVITIES and not self._chute)
        if not free:
            return
        if env.now >= self._next_wander:
            self._next_wander = env.now + self.rng.uniform(*self.settings.wander_every)
            self._wander_now(env)
            return
        if env.now >= self._next_cover_check:
            self._next_cover_check = env.now + COVER_CHECK
            front = W.frontmost(env.windows)
            if front is not None and env.now - self._last_cover_move >= COVER_MOVE_GAP:
                if W.intersects(W.her_rect(body.x, body.y, env.size), front.content):
                    if self._wander_now(env):
                        self._last_cover_move = env.now

    def _wander_now(self, env: Env) -> bool:
        body = self.body
        if body is None:
            return False
        led = body.ledge
        avoid = W.avoid_rects(W.frontmost(env.windows), env.cursor, env.caret)
        spot = W.choose_spot(self.ledges, body.pos, env.size, avoid, self.rng, env.screens)
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
            if self._run is not None or body.busy or self.anim.key != "idle":
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
