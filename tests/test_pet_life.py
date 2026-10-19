"""Her life frame by frame: PetLife driving a real MascotAnimator with a fake
clock and synthetic desktops. What these check is what the panel shows."""

from __future__ import annotations

import random

import pytest

from wawi import mascot
from wawi.pet import api
from wawi.pet import world as W
from wawi.pet.life import Env, PetLife
from wawi.pet.needs import NeedsStore
from wawi.pet.settings import PetSettings
from wawi.reactions import Event, Reaction
from wawi.state import State

DT = 1 / 30
SIZE = (124.0, 110.0)
SCREEN = W.Screen((0.0, 0.0, 1440.0, 900.0), (0.0, 60.0, 1440.0, 815.0))
T0 = 1_800_000_000.0


class Rig:
    """An animator, a body and a desktop, ticked together like the panel does."""

    def __init__(self, tmp_path, attached=False, windows=(), frames=mascot.FRAME_NAMES,
                 **settings):
        settings.setdefault("wander", False)
        settings.setdefault("lonely_after_min", 0)
        self.settings = PetSettings.from_dict(settings | {"state_file": str(tmp_path / "s.json")})
        self.anim = mascot.MascotAnimator(available=frames, rng=random.Random(2))
        self.life = PetLife(self.settings, self.anim, rng=random.Random(5),
                            store=NeedsStore(tmp_path / "s.json"), wall=T0)
        self.t = 0.0
        self.attached = attached
        self.windows = list(windows)
        self.version = 1
        self.pos = (700.0, 60.0)
        self.caret = None
        self.cursor = (100.0, 800.0)
        self.key_idle = 1e9
        self.reduce = False
        self.dragging = False
        self.state = State.IDLE
        self.muted = False
        self.out = None

    def set_windows(self, windows):
        self.windows = list(windows)
        self.version += 1

    def env(self):
        return Env(now=self.t, wall=T0 + self.t, attached=self.attached,
                   position=None if self.attached else self.pos, size=SIZE,
                   screens=[SCREEN], windows=self.windows, world_version=self.version,
                   cursor=self.cursor, caret=self.caret, key_idle=self.key_idle,
                   reduce_motion=self.reduce, dragging=self.dragging)

    def step(self):
        self.t += DT
        self.out = self.life.tick(DT, self.env())
        if self.out.ground is not None:
            self.pos = self.out.ground
        self.anim.update(DT, self.state, muted=self.muted, reduce_motion=self.reduce)
        return self.out

    def run(self, seconds, until=None):
        seen = []
        for _ in range(int(seconds / DT)):
            out = self.step()
            seen.append((self.anim.frame, self.anim.locomotion, self.anim.act_name, out))
            if until is not None and until(self):
                break
        return seen


@pytest.fixture(autouse=True)
def _api_off():
    api.attach(None)
    yield
    api.attach(None)


class TestFeeding:
    def test_feeding_shows_the_bowl_eating_then_full(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.needs.give({"food": -60.0})
        before = rig.life.needs.snapshot().food
        assert rig.life.command("feed")
        seen = rig.run(8.0)
        frames = [f for f, *_ in seen]
        assert "eat_a" in frames and "eat_b" in frames and "full" in frames
        assert frames.index("full") > frames.index("eat_a")
        assert rig.life.needs.snapshot().food > before + 40
        assert rig.anim.acting is False  # and it ends on its own

    def test_the_prop_follows_the_step(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.command("water")
        props = []
        for _ in range(int(5 / DT)):
            rig.step()
            props.append(rig.anim.prop)
        assert "prop_bowl_water" in props
        assert props[-1] is None

    def test_api_commands_from_other_threads_reach_her(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.start()
        assert api.play() is True
        rig.run(1.0)
        assert rig.anim.act_name == "play"
        assert api.needs()["mood"] in ("happy", "content")
        rig.life.stop(T0 + 2)
        assert api.feed() is False


class TestReactions:
    def test_unknown_behaviour_barks(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.react(Reaction(behaviour="moonwalk"))
        rig.run(0.2)
        assert rig.anim.act_name == "bark"

    def test_bubble_only_shows_at_once_even_mid_emergency(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.react(Reaction(behaviour="run", costume="firefighter", priority=2,
                                until_key="web:a"))
        rig.run(1.0)
        rig.life.react(Reaction(bubble="Ojo con el disco", priority=0))
        out = rig.step()
        assert out.bubble == "Ojo con el disco"
        assert rig.life.director.active.reaction.until_key == "web:a"  # still on

    def test_bubbles_go_away(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.react(Reaction(bubble="hola"))
        assert rig.step().bubble == "hola"
        rig.run(4.0)
        assert rig.step().bubble is None

    def test_speak_is_handed_over_once(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.react(Reaction(bubble="¡Se cayó la web!", speak=True, behaviour="howl"))
        assert rig.step().speak == "¡Se cayó la web!"
        assert rig.step().speak is None

    def test_firefighter_patrols_until_the_site_is_back(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.react(Reaction(behaviour="run", costume="firefighter", priority=2,
                                until_key="web:a", bubble="¡Se cayó!"))
        seen = rig.run(6.0)
        xs = [o.ground[0] for *_, o in seen]
        assert max(xs) - min(xs) > 200  # running back and forth
        frames = {f for f, *_ in seen}
        assert {"firefighter_a", "firefighter_b"} <= frames
        assert rig.life.body.mode == "trot"
        rig.life.react(Reaction(cancel_key="web:a", behaviour="celebrate", bubble="¡Volvió!"))
        seen = rig.run(5.0)
        assert "happy" in {f for f, *_ in seen} or "jump" in {f for f, *_ in seen}
        assert rig.life.body.patrol is None
        rig.run(3.0)
        assert rig.life.director.active is None and not rig.anim.acting

    def test_a_long_emergency_calms_down_but_keeps_the_helmet(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.react(Reaction(behaviour="run", costume="firefighter", priority=2,
                                until_key="web:a"))
        rig.run(50.0)
        assert rig.life.body.patrol is None
        assert rig.anim.frame == "firefighter_hose"
        assert rig.life.director.active is not None

    def test_detective_preempts_the_firefighter_then_she_goes_back(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.react(Reaction(behaviour="run", costume="firefighter", priority=2,
                                until_key="web:a"))
        rig.run(2.0)
        rig.life.react(Reaction(behaviour="detective", costume="detective", priority=3,
                                duration=15))
        seen = rig.run(5.0)
        assert {"detective", "detective_b"} & {f for f, *_ in seen}
        assert rig.life.body.patrol is None  # stopped running to investigate
        seen = rig.run(13.0)
        assert rig.life.director.active.reaction.until_key == "web:a"
        assert {"firefighter_a", "firefighter_b"} & {f for f, *_ in seen[-30:]}

    def test_reactions_wait_out_a_voice_turn(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.state = State.LISTENING
        rig.life.react(Reaction(behaviour="howl"))
        rig.run(1.0)
        assert rig.anim.frame.startswith("listening") or rig.anim.frame.startswith("megaphone")
        rig.state = State.IDLE
        rig.run(0.5)
        assert rig.anim.frame == "howl"


class TestSpaceship:
    def test_she_flies_off_waits_and_comes_back_when_cancelled(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.react(Reaction(behaviour="spaceship", until_key="fetch:7", priority=1))
        seen = rig.run(8.0)
        assert "ship_board" in {f for f, *_ in seen}
        assert {"ship_fly_a", "ship_fly_b"} & {f for f, *_ in seen}
        out = seen[-1][3]
        assert out.hidden and out.fly[1] > 700  # gone up out of view
        rig.run(30.0)
        assert rig.step().hidden  # still out there: no cancel yet
        rig.life.react(Reaction(cancel_key="fetch:7", behaviour="celebrate",
                                bubble="¡Acá está la página!"))
        seen = rig.run(8.0)
        assert not seen[-1][3].hidden and seen[-1][3].fly == (0.0, 0.0)
        frames = [f for f, *_ in seen]
        assert "ship_return" in frames
        later = frames[frames.index("ship_return"):]
        assert "happy" in later or "jump" in later  # then the celebration

    def test_without_a_cancel_the_trip_has_a_default_length(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.react(Reaction(behaviour="spaceship"))
        rig.run(45.0)
        assert rig.life.director.active is None
        assert rig.step().fly == (0.0, 0.0)

    def test_under_reduce_motion_she_does_not_fly(self, tmp_path):
        rig = Rig(tmp_path)
        rig.reduce = True
        rig.life.react(Reaction(behaviour="spaceship", duration=3))
        seen = rig.run(10.0)
        assert all(o.fly == (0.0, 0.0) and not o.hidden for *_, o in seen)


class TestDesktopLife:
    def test_attached_she_never_moves(self, tmp_path):
        rig = Rig(tmp_path, attached=True, wander=True, wander_every_min=[1, 1])
        seen = rig.run(90.0)
        assert all(o.ground is None for *_, o in seen)

    def test_dropped_mid_air_she_falls_to_the_floor(self, tmp_path):
        rig = Rig(tmp_path)
        rig.pos = (500.0, 400.0)
        seen = rig.run(2.0)
        assert rig.pos[1] == 60.0
        assert "fall" in {loco for _, loco, *_ in seen}

    def test_gravity_on_drop_can_be_turned_off(self, tmp_path):
        rig = Rig(tmp_path, gravity_on_drop=False)
        rig.pos = (500.0, 400.0)
        rig.run(2.0)
        assert rig.pos == (500.0, 400.0)

    def test_window_under_her_closes_she_falls(self, tmp_path):
        w = W.Win(5, (400.0, 60.0, 600.0, 300.0))
        rig = Rig(tmp_path, windows=[w])
        rig.pos = (700.0, 360.0)
        rig.run(1.0)
        assert rig.pos[1] == 360.0
        assert rig.anim.idle_clip == "sit_edge"
        rig.set_windows([])
        rig.run(2.0)
        assert rig.pos[1] == 60.0 and rig.anim.idle_clip == "idle"

    def test_she_moves_only_when_she_gets_up_never_mid_sleep(self, tmp_path):
        # The dog model (wawi/pet/dog.py): no wandering timer of her own; a
        # sleeping dog stays put, and she walks to a new spot on getting up.
        from wawi.pet import dog as D

        rig = Rig(tmp_path, wander=True, wander_every_min=[1, 1])
        rig.cursor = None
        rig.step()
        dog = rig.life.dog
        dog.settle(rig.t, "test")
        dog._enter(rig.t, D.DEEP, 3600.0, "test")
        rig.run(5.0)
        assert rig.anim.asleep
        start = rig.pos
        seen = rig.run(150.0)
        assert rig.pos == start
        assert all(loco is None for _, loco, *_ in seen)
        # She wakes and gets up: off she goes.
        dog.strong(rig.t, "test")
        rig.life._cues.append((D.Cue("relocate"), rig.t))
        seen = rig.run(60.0)
        assert rig.pos != start
        assert {"walk", "climb"} & {loco for _, loco, *_ in seen}

    def test_muted_she_still_moves(self, tmp_path):
        # The microphone off only stops her hearing: her body keeps living.
        from wawi.pet import dog as D

        rig = Rig(tmp_path, wander=True, wander_every_min=[1, 1])
        rig.muted = True
        rig.cursor = None
        rig.step()
        start = rig.pos
        rig.life._cues.append((D.Cue("relocate"), rig.t))
        seen = rig.run(60.0)
        assert rig.pos != start
        assert {"walk", "climb"} & {loco for _, loco, *_ in seen}

    def test_muted_reactions_still_play(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.muted = True
        rig.life.react(Reaction(behaviour="howl"))
        rig.run(0.5)
        assert rig.anim.frame == "howl"

    def test_no_wandering_in_full_screen(self, tmp_path):
        full = W.Win(1, (0.0, 0.0, 1440.0, 900.0))
        rig = Rig(tmp_path, windows=[full], wander=True, wander_every_min=[1, 1])
        rig.pos = (700.0, 60.0)
        rig.run(150.0)
        assert rig.pos == (700.0, 60.0)

    def test_a_voice_turn_stops_her_walk(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.react(Reaction(behaviour="walk_to",
                                source=Event("x", data={"x": 1200.0, "y": 60.0})))
        rig.run(1.0)
        x = rig.pos[0]
        assert x > 700.0
        rig.state = State.LISTENING
        rig.run(2.0)
        assert rig.pos[0] == pytest.approx(x, abs=2.0)
        rig.state = State.IDLE
        rig.run(20.0)
        assert rig.pos[0] == pytest.approx(1200.0)

    def test_she_walks_right_mirrored(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.react(Reaction(behaviour="walk_to",
                                source=Event("x", data={"x": 1200.0, "y": 60.0})))
        rig.run(1.0)
        assert rig.anim.locomotion == "walk" and rig.anim.mirrored


class TestWatchingYouType:
    def _lonely_rig(self, tmp_path):
        rig = Rig(tmp_path, lonely_after_min=1)
        rig.caret = (400.0, 500.0, 2.0, 16.0)
        rig.cursor = (1300.0, 800.0)
        rig.key_idle = 1e9
        rig.run(70.0)  # ignored for over a minute, not typing
        assert rig.life._watch_phase is None
        return rig

    def test_lonely_and_you_type_she_comes_to_watch(self, tmp_path):
        rig = self._lonely_rig(tmp_path)
        rig.key_idle = 0.3
        seen = rig.run(60.0, until=lambda r: r.life._watch_phase == "watching")
        assert rig.life._watch_phase == "watching"
        assert rig.out.want_caret
        x, y = rig.pos
        r = W.her_rect(x, y, SIZE)
        assert not W.intersects(r, W.inflate(rig.caret, 20.0))
        assert x > rig.caret[0]
        assert "climb" in {loco for _, loco, *_ in seen}
        rig.run(3.0)
        assert rig.anim.frame in ("watch", "watch_b", "side_eye", "eyeroll")

    def test_a_touch_makes_her_happy_and_she_leaves(self, tmp_path):
        rig = self._lonely_rig(tmp_path)
        rig.key_idle = 0.3
        rig.run(60.0, until=lambda r: r.life._watch_phase == "watching")
        assert rig.life.touched(rig.t) is True
        seen = rig.run(3.0)
        assert rig.life._watch_phase is None
        assert rig.pos[1] == 60.0  # down from the text line
        assert {"happy", "happy_b"} & {f for f, *_ in seen}

    def test_she_leaves_when_you_stop_typing(self, tmp_path):
        rig = self._lonely_rig(tmp_path)
        rig.key_idle = 0.3
        rig.run(60.0, until=lambda r: r.life._watch_phase == "watching")
        rig.key_idle = 60.0
        rig.run(3.0)
        assert rig.life._watch_phase is None and rig.pos[1] == 60.0

    def test_talking_to_her_means_she_is_not_lonely(self, tmp_path):
        rig = self._lonely_rig(tmp_path)
        rig.state = State.LISTENING
        rig.step()
        rig.state = State.IDLE
        rig.key_idle = 0.3
        rig.run(20.0)
        assert rig.life._watch_phase is None


class TestNeedsInLife:
    def test_a_voice_turn_and_touches_count_as_company(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.needs.give({"affection": -50.0})
        a = rig.life.needs.snapshot().affection
        rig.state = State.LISTENING
        rig.step()
        rig.step()  # the body sees the turn on the frame after it starts
        rig.state = State.IDLE
        rig.life.touched(rig.t)
        assert rig.life.needs.snapshot().affection > a + 5

    def test_mood_sets_the_animator_bias(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.needs.give({"food": -80.0})
        rig.run(6.0)
        assert rig.life.mood == "hungry" and "hungry_bowl" in rig.anim.bias

    def test_needs_are_saved_on_stop(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.life.command("feed")
        rig.life.stop(T0 + 5)
        assert (tmp_path / "s.json").is_file()
