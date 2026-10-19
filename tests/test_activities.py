"""Her idle activities (zoomies, jumps, the parachute, the bird…) and what
she is seen doing while a tool runs (wawi/working.py)."""

from __future__ import annotations

import random

import pytest

from wawi import mascot, tools, working
from wawi.mascot import ACTIVITIES, WORK_CLIPS, MascotAnimator
from wawi.pet import api
from wawi.pet import world as W
from wawi.pet.life import Env, PetLife
from wawi.pet.needs import NeedsStore
from wawi.pet.settings import PetSettings
from wawi.state import State

DT = 1 / 30
ALL_FRAMES = frozenset(mascot.FRAME_NAMES)
ALL_PROPS = frozenset(mascot.PROP_NAMES)
KNOWN = ALL_FRAMES | ALL_PROPS


def anim(seed=1, frames=ALL_FRAMES, props=ALL_PROPS, lift=True):
    a = MascotAnimator(available=frames, rng=random.Random(seed), sleep_after=1e9)
    a.set_props(props)
    a.can_lift = lift
    return a


def run(a, seconds, state=State.IDLE, **kw):
    seen = []
    for _ in range(int(round(seconds / DT))):
        a.update(DT, state, **kw)
        seen.append((a.frame, a.behaviour, a.prop, a.prop_place, a.lift, a.excursion))
    return seen


def play(a, name):
    """Ask for *name* and tick through it; the frames it was on screen."""
    assert a.perform(name)
    shots, started = [], False
    for _ in range(int(40 / DT)):
        a.update(DT, State.IDLE)
        if a.behaviour == name:
            started = True
            shots.append((a.frame, a.prop, a.prop_place, a.lift, a.excursion, a._oneshot_t))
        elif started:
            break
    assert started, name
    return shots


# ---------------------------------------------------------------------------
# The clips
# ---------------------------------------------------------------------------

class TestClips:
    @pytest.mark.parametrize("name", list(ACTIVITIES))
    def test_every_activity_uses_real_frames_props_and_paths(self, name):
        activity = ACTIVITIES[name]
        for step in activity.clip.steps:
            assert step.frame in ALL_FRAMES, (name, step.frame)
            assert step.duration > 0
            if step.prop is not None:
                assert step.prop in ALL_PROPS, (name, step.prop)
            if step.path is not None:
                assert step.path.partition("@")[0] in mascot.PROP_PATHS, (name, step.path)
        assert set(activity.needs) <= KNOWN
        assert activity.weight > 0 and activity.cooldown >= 60.0

    def test_work_and_body_clips_use_real_frames(self):
        clips = list(WORK_CLIPS.values()) + [mascot.BODY_CLIPS["chute"],
                                             mascot.ACT_CLIPS["chute_end"],
                                             mascot.ACT_CLIPS["zoomies_end"]]
        for clip in clips:
            for step in clip.steps:
                assert step.frame in ALL_FRAMES, (clip.name, step.frame)

    def test_the_shipped_art_covers_every_activity(self):
        have = mascot.available_frames() | mascot.available_props()
        for name, activity in ACTIVITIES.items():
            assert set(activity.needs) <= have, name
        for name, clip in WORK_CLIPS.items():
            assert clip.playable(mascot.available_frames()) is not None, name

    def test_new_frames_are_anchored_so_the_shared_box_never_grows(self):
        assert set(mascot.ACTIVITY_FRAMES) <= set(mascot.LOOSE_FRAMES)
        assert set(mascot.WORK_FRAMES) <= set(mascot.LOOSE_FRAMES)
        anchors = mascot.load_anchors()
        for n in mascot.available_frames() & set(mascot.ACTIVITY_FRAMES):
            assert n in anchors, n


# ---------------------------------------------------------------------------
# Picking them
# ---------------------------------------------------------------------------

class TestPicker:
    def test_each_activity_can_be_picked(self):
        seen = set()
        for seed in range(40):
            a = anim(seed)
            for _ in range(60):
                name = a._pick_behaviour()
                a._last_behaviour = name
                seen.add(name)
        assert set(ACTIVITIES) <= seen

    def test_muted_she_keeps_playing(self):
        # Mic off only stops the listening; she keeps living (owner's rule).
        a = anim()
        a._next_behaviour = 0.0
        run(a, 1.0, State.IDLE, muted=True)
        assert a.activities() != []

    @pytest.mark.parametrize("block", ["hold", "reduce", "listening",
                                       "thinking", "speaking"])
    def test_never_in_a_blocked_state(self, block):
        a = anim()
        kw: dict = {}
        state = State.IDLE
        if block == "hold":
            a.hold_activities = True
        elif block == "reduce":
            kw["reduce_motion"] = True
        else:
            state = {"listening": State.LISTENING, "thinking": State.THINKING,
                     "speaking": State.SPEAKING}[block]
        a._next_behaviour = 0.0
        seen = run(a, 120.0, state, **kw)
        assert not {b for _, b, *_ in seen} & set(ACTIVITIES)
        if block in ("hold", "reduce"):
            assert a.activities() == []

    def test_never_while_asleep_unless_she_gets_up_first(self):
        a = MascotAnimator(available=ALL_FRAMES, rng=random.Random(4), sleep_after=1.0)
        a.set_props(ALL_PROPS)
        a.can_lift = True
        run(a, 5.0)
        assert a.asleep
        for _ in range(int(60 / DT)):
            a.update(DT, State.IDLE)
            if a.behaviour in ACTIVITIES:
                assert not a.asleep  # woken first: never "asleep and zooming"

    def test_missing_art_means_never_picked(self):
        a = anim(frames=ALL_FRAMES - {"pant_a"}, props=ALL_PROPS - {"prop_bird_sit"})
        assert "zoomies" not in a.activities()
        assert "bird_chase" not in a.activities()
        assert not a.perform("bird_chase")
        assert "shake" in a.activities()

    def test_lifting_ones_need_her_body(self):
        a = anim(lift=False)
        names = a.activities()
        assert "jump_tries" not in names and "parachute" not in names
        assert "play_chew" in names

    def test_a_cooldown_per_group(self):
        a = anim()
        assert a._start("jump_tries")
        assert "jump_flop" not in a.activities() and "jump_tries" not in a.activities()
        a.t += ACTIVITIES["jump_tries"].cooldown + 1.0
        assert "jump_flop" in a.activities()

    def test_perform_does_it_next_time_she_is_idle(self):
        a = anim()
        assert a.perform("shake")
        run(a, 0.2, State.THINKING)
        assert a.behaviour is None
        run(a, 0.2)
        assert a.behaviour == "shake"
        assert not a.perform("not_a_thing")


# ---------------------------------------------------------------------------
# What they look like
# ---------------------------------------------------------------------------

class TestMotion:
    def test_jump_heights_rise_across_attempts(self):
        clip = ACTIVITIES["jump_tries"].clip
        peaks, acc = [], 0.0
        for step in clip.steps:
            if (step.motion or "").startswith("leap"):
                peaks.append(max(mascot.clip_lift(clip, acc + step.duration * k / 20)
                                 for k in range(21)))
            acc += step.duration
        assert len(peaks) == 4
        assert all(b > a for a, b in zip(peaks, peaks[1:]))
        assert peaks[-1] == pytest.approx(mascot.LEAP_HEIGHTS[-1], rel=0.02)
        assert mascot.clip_lift(clip, clip.duration) == 0.0

    def test_jump_flop_ends_on_her_back(self):
        frames = [f for f, *_ in play(anim(), "jump_flop")]
        assert "flop_back" in frames
        assert frames.index("flop_back") > frames.index("jump_reach")

    def test_the_parachute_goes_up_then_descends_and_lands(self):
        shots = play(anim(), "parachute")
        lifts = [s[3] for s in shots]
        top = max(lifts)
        assert top == pytest.approx(mascot.CHUTE_TOP, rel=0.05)
        floating = [s[3] for s in shots if s[0] in ("chute_hang_a", "chute_hang_b")
                    and s[1] == "prop_chute" and s[2] and s[2][2] >= 1.0]
        assert len(floating) > 30
        assert all(b <= a + 1e-6 for a, b in zip(floating, floating[1:]))  # only down
        assert floating[0] - floating[-1] > 0.8 * top
        assert lifts[-1] == 0.0
        assert shots[-1][0] == "chute_fold"

    def test_the_bird_lands_then_leaves(self):
        shots = play(anim(), "bird_chase")
        bird = [s for s in shots if s[1] and s[1].startswith("prop_bird")]
        assert bird and bird[0][2][2] < 0.5  # it fades in…
        landed = [s for s in bird if s[1] in ("prop_bird_sit", "prop_bird_peck")]
        assert landed and all(s[2][1] == 0.0 for s in landed)  # …stands on the ground…
        off = [s for s in bird if s[1] in mascot._BIRD_FLAP and s[2][1] > 0.1]
        assert off[-1][2][1] > 0.8 and off[-1][2][2] < 0.2  # …and flies off, fading
        after = shots[shots.index(off[-1]) + 1:]
        assert after and all(s[1] is None for s in after)  # gone for good
        assert "pounce" in [s[0] for s in shots]

    @pytest.mark.parametrize("name", ["zoomies", "bird_chase"])
    def test_she_comes_home(self, name):
        clip = ACTIVITIES[name].clip
        assert mascot.clip_excursion(clip, clip.duration) == pytest.approx(0.0, abs=1e-6)
        assert min(mascot.clip_excursion(clip, clip.duration * k / 100)
                   for k in range(101)) < -10.0

    def test_zoomies_runs_both_ways_then_pants(self):
        a = anim()
        shots = play(a, "zoomies")
        exc = [s[4] for s in shots]
        assert min(exc) < -50 and max(exc) > 50
        assert shots[-1][0] in ("pant_a", "pant_b")

    def test_the_bird_stays_put_in_the_world_while_she_creeps(self):
        place = mascot.prop_place("bird_ground", 0.0, 1.0)
        x0 = mascot.prop_draw_rect("prop_bird_sit", (124.0, 110.0), None, None, place, 0.0)[0]
        x1 = mascot.prop_draw_rect("prop_bird_sit", (124.0, 110.0), None, None, place, -20.0)[0]
        assert x1 - x0 == pytest.approx(20.0)

    def test_moving_props_get_room_in_her_window(self):
        rects = mascot.prop_reach_rects((124.0, 110.0), None, ALL_PROPS)
        side, _below, above = mascot.canvas_extent(rects, (124.0, 110.0))
        assert above > 40.0 and side > 10.0


# ---------------------------------------------------------------------------
# On the desktop: her body runs the zoomies and jumps off a window
# ---------------------------------------------------------------------------

SCREEN = W.Screen((0.0, 0.0, 1440.0, 900.0), (0.0, 60.0, 1440.0, 815.0))


class Rig:
    def __init__(self, tmp_path, pos=(700.0, 60.0), windows=(), attached=False):
        settings = PetSettings.from_dict({"wander": False, "lonely_after_min": 0,
                                          "state_file": str(tmp_path / "s.json")})
        self.anim = MascotAnimator(available=ALL_FRAMES, rng=random.Random(2))
        self.anim.set_props(ALL_PROPS)
        self.life = PetLife(settings, self.anim, rng=random.Random(5),
                            store=NeedsStore(tmp_path / "s.json"), wall=1.8e9)
        self.life.start()
        self.t = 0.0
        self.pos = pos
        self.windows = list(windows)
        self.attached = attached
        self.key_idle = 1e9
        self.out = None

    def step(self):
        self.t += DT
        env = Env(now=self.t, wall=1.8e9 + self.t, attached=self.attached,
                  position=None if self.attached else self.pos, size=(124.0, 110.0),
                  screens=[SCREEN], windows=self.windows, world_version=1,
                  cursor=(10.0, 10.0), key_idle=self.key_idle)
        self.out = self.life.tick(DT, env)
        if self.out.ground is not None:
            self.pos = self.out.ground
        self.anim.update(DT, State.IDLE)


@pytest.fixture(autouse=True)
def _clean():
    api.attach(None)
    working.BOARD.clear()
    yield
    api.attach(None)
    working.BOARD.clear()


class TestDesktop:
    def test_off_a_window_with_the_parachute(self, tmp_path):
        win = W.Win(1, (400.0, 300.0, 700.0, 400.0))
        rig = Rig(tmp_path, pos=(700.0, 700.0), windows=[win])
        for _ in range(10):
            rig.step()
        assert rig.life.body.ledge is not None and rig.life.body.ledge.wid == 1
        assert rig.anim.perform("parachute")
        locos, ys, acts = [], [], []
        for _ in range(int(30 / DT)):
            rig.step()
            locos.append(rig.anim.locomotion)
            ys.append(rig.pos[1])
            acts.append(rig.anim.act_name)
            if "chute_end" in acts and rig.anim.act_name is None:
                break
        assert "chute" in locos
        assert "chute_end" in acts
        assert rig.pos[1] == pytest.approx(60.0)  # down on the floor
        # A float, not a fall: about the canopy's speed.
        falling = locos.count("chute") * DT
        assert falling >= (700.0 - 60.0) / W.Body.CHUTE_FALL * 0.9

    def test_zoomies_across_her_ledge(self, tmp_path):
        rig = Rig(tmp_path)
        for _ in range(5):
            rig.step()
        x0 = rig.pos[0]
        assert rig.anim.perform("zoomies")
        xs, acts = [], []
        for _ in range(int(15 / DT)):
            rig.step()
            xs.append(rig.pos[0])
            acts.append(rig.anim.act_name)
        assert min(xs) < x0 - 100 and max(xs) > x0 + 100
        assert rig.pos[0] == pytest.approx(x0)
        assert "zoomies_end" in acts

    def test_a_jump_lifts_her_window(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        rig.step()
        assert rig.anim.perform("jump_tries")
        lifts = []
        for _ in range(int(8 / DT)):
            rig.step()
            lifts.append(rig.out.fly[1])
        assert max(lifts) > 50.0

    def test_typing_holds_them(self, tmp_path):
        rig = Rig(tmp_path)
        rig.key_idle = 2.0
        rig.step()
        rig.step()
        assert rig.anim.hold_activities and rig.anim.activities() == []

    def test_a_trick_from_the_api(self, tmp_path):
        rig = Rig(tmp_path, attached=True)
        api.attach(lambda: "ok")
        assert api.trick("sacudirse")
        for _ in range(5):
            rig.step()
        assert rig.anim.behaviour == "shake"


# ---------------------------------------------------------------------------
# Working: tools show what she is doing
# ---------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class TestBoard:
    def test_her_own_work_is_counted_and_never_busy(self):
        board = working.Board(clock=Clock())
        board.set_musing("selffix", True)
        board.set_musing("selffix", True)
        board.set_musing("selffix", False)
        assert board.musing() is True  # one run still going
        assert board.busy() is False and board.now() == (None, None)
        with board.musing_while("curiosities"):
            board.set_musing("selffix", False)
            assert board.musing() is True
        assert board.musing() is False

    def test_start_and_end_events(self):
        board = working.Board(clock=Clock())
        heard = []
        board.subscribe(lambda *e: heard.append(e))
        with board.doing("fetch_page", "ship"):
            assert board.now() == ("ship", None)
        assert heard == [("start", "fetch_page", "ship"), ("end", "fetch_page", "ship")]

    def test_end_fires_on_errors_too(self):
        board = working.Board(clock=Clock())
        heard = []
        board.subscribe(lambda *e: heard.append(e[0]))
        with pytest.raises(RuntimeError):
            with board.doing("capture_screen", "camera"):
                raise RuntimeError("boom")
        assert heard == ["start", "end"]

    def test_a_quick_tool_still_shows_for_the_minimum_time(self):
        clock = Clock()
        board = working.Board(clock=clock, min_show=1.2)
        board.end(board.begin("copy_to_clipboard", "carry_paper"))
        assert board.now()[0] == "carry_paper"
        clock.t += 1.0
        assert board.now()[0] == "carry_paper"
        clock.t += 0.3
        assert board.now()[0] is None

    def test_unknown_animation_is_the_generic_one(self):
        board = working.Board(clock=Clock())
        board.begin("mcp__x__thing")
        assert board.now()[0] == working.DEFAULT

    def test_background_work(self):
        board = working.Board(clock=Clock())
        board.set_busy("task", True)
        assert board.now() == (None, working.BACKGROUND)
        board.set_busy("task", False)
        assert board.now() == (None, None)

    def test_a_broken_listener_never_breaks_a_tool(self):
        board = working.Board(clock=Clock())
        board.subscribe(lambda *e: 1 / 0)
        with board.doing("x"):
            pass


class TestToolAnimations:
    def test_every_tool_declares_one_or_deliberately_none(self):
        for name in tools._DISPATCH:
            anim_name = tools.animation_for(name)
            if anim_name is None:
                assert tools.has_no_animation(name), f"{name}: no animation declared"
            else:
                assert anim_name in WORK_CLIPS, (name, anim_name)

    def test_the_model_never_sees_it(self, monkeypatch):
        specs = tools.active_tools()
        assert specs and all("animation" not in s for s in specs)
        assert tools.for_model({"name": "x", "animation": "ship"}) == {"name": "x"}

    def test_a_tool_defined_elsewhere_registers_its_own(self, monkeypatch):
        monkeypatch.setattr(tools, "_EXTRA_ANIMATIONS", {})
        tools.register_animation("send_to_orca", "laptop")
        assert tools.animation_for("send_to_orca") == "laptop"
        assert tools.animation_for("never_heard_of_it") is None

    def test_the_brain_publishes_start_and_end_even_when_the_tool_fails(self, monkeypatch):
        from wawi import brain

        heard = []
        working.subscribe(lambda *e: heard.append(e))
        try:
            def boom(name, payload):
                raise RuntimeError("tool exploded")

            monkeypatch.setattr(tools, "dispatch", boom)
            with pytest.raises(RuntimeError):
                brain._run_tool("capture_screen", {}, brain._TurnState())
        finally:
            working.BOARD._listeners.clear()
        assert heard == [("start", "capture_screen", "camera"),
                         ("end", "capture_screen", "camera")]

    def test_self_animating_tools_show_nothing_extra(self, monkeypatch):
        from wawi import brain

        heard = []
        working.subscribe(lambda *e: heard.append(e))
        try:
            monkeypatch.setattr(tools, "dispatch",
                                lambda n, p: tools.ToolResult(True, "ok", {}))
            brain._run_tool("pet_feed", {}, brain._TurnState())
        finally:
            working.BOARD._listeners.clear()
        assert heard == []


class TestWorkClip:
    def test_shown_while_thinking(self):
        a = anim()
        a.working = "camera"
        run(a, 0.5, State.THINKING)
        assert a.frame in ("camera_a", "camera_b")

    @pytest.mark.parametrize("state", [State.LISTENING, State.SPEAKING])
    def test_never_over_listening_or_speaking(self, state):
        a = anim()
        a.working = "camera"
        frames = {f for f, *_ in run(a, 1.0, state)}
        assert not frames & {"camera_a", "camera_b"}

    def test_unknown_or_missing_art_falls_back(self):
        a = anim()
        a.working = "no_such_animation"
        run(a, 0.3, State.THINKING)
        assert a.frame in ("laptop_a", "laptop_b")
        b = anim(frames=ALL_FRAMES - {"laptop_a", "laptop_b"})
        b.working = "laptop"
        run(b, 0.3, State.THINKING)
        assert b.frame == "think_bubble"

    def test_background_work_keeps_her_up_at_her_laptop(self):
        a = MascotAnimator(available=ALL_FRAMES, rng=random.Random(1), sleep_after=2.0)
        a.background = "background"
        frames = {f for f, *_ in run(a, 10.0)}
        assert not a.asleep
        assert {"laptop_a", "laptop_b"} & frames
