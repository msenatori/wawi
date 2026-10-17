"""She climbs like a dog: up a little wooden ladder (wawi/pet/world.py Ladder,
the ladder clips and layout in wawi/mascot.py), and the old cling-to-the-wall
climb when the ladder art is missing."""

from __future__ import annotations

import random

import pytest

from wawi import mascot
from wawi.pet import api
from wawi.pet import world as W
from wawi.pet.life import Env, PetLife
from wawi.pet.needs import NeedsStore
from wawi.pet.settings import PetSettings
from wawi.pet.world import Body, Leg, Screen, Win
from wawi.state import State

DT = 1 / 30
SIZE = (124.0, 110.0)
SCREEN = Screen((0.0, 0.0, 1440.0, 900.0), (0.0, 60.0, 1440.0, 815.0))
T0 = 1_800_000_000.0
#: A window whose top edge (y=360) she can sit on, 600..1000 wide.
WINDOW = Win(5, (600.0, 60.0, 400.0, 300.0))
LADDER_CLIPS = ("ladder_carry", "ladder_set", "ladder", "ladder_step", "ladder_pull",
                "ladder_stow", "ladder_held")


def _ledges(windows=(WINDOW,)):
    return W.ledges([SCREEN], list(windows), SIZE)


def _top(ls):
    return next(led for led in ls if led.wid == 5)


def _walk(body, ls, seconds=60.0):
    """Run her legs to the end; what she did on the way."""
    modes, views, ys = [], [], []
    for _ in range(int(seconds / DT)):
        e = body.update(DT, ls)
        modes.append(body.mode)
        views.append(body.ladder_view())
        ys.append(body.shown_y)
        if e == "arrived" and body.ladder is None:
            break
    return modes, views, ys


# ---------------------------------------------------------------------------
# The body
# ---------------------------------------------------------------------------

class TestLadderPath:
    def test_without_the_art_she_climbs_the_old_way(self):
        ls = _ledges()
        body = Body(200.0, 60.0)
        body.update(DT, ls)
        legs = W.plan_path(body, (800.0, 360.0), _top(ls), ls, SIZE)
        assert [leg.kind for leg in legs] == ["walk", "climb", "shift", "walk"]

    def test_with_the_art_she_sets_climbs_steps_off_and_folds(self):
        ls = _ledges()
        body = Body(200.0, 60.0, ladders=True)
        body.update(DT, ls)
        legs = W.plan_path(body, (800.0, 360.0), _top(ls), ls, SIZE)
        assert [leg.kind for leg in legs] == [
            "walk", "ladder_set", "climb", "shift", "ladder_fold", "walk"]
        body.go(legs)
        modes, views, _ys = _walk(body, ls)
        order = [m for i, m in enumerate(modes) if i == 0 or modes[i - 1] != m]
        for a, b in (("walk", "ladder_set"), ("ladder_set", "climb"), ("climb", "step_off"),
                     ("step_off", "ladder_fold")):
            assert order.index(a) < order.index(b)
        # She ends on the window's top edge, where she was going, ladder gone.
        assert (body.x, body.y) == (800.0, 360.0) and body.ledge.wid == 5
        assert body.ladder is None
        assert any(v is not None and v.poof >= 0.0 for v in views)

    def test_the_ladder_spans_the_climb(self):
        ls = _ledges()
        body = Body(200.0, 60.0, ladders=True)
        body.update(DT, ls)
        body.go(W.plan_path(body, (800.0, 360.0), _top(ls), ls, SIZE))
        while body.mode != "climb":
            body.update(DT, ls)
        lad = body.ladder
        view = body.ladder_view()
        # From the floor she stood on to past the top edge she is going to.
        assert lad.bottom == 60.0 and lad.top == 360.0
        assert view.base == 60.0 and view.lo == 0.0
        assert view.hi >= 300.0 + 0.2 * SIZE[1]
        # Against the window's LEFT side, the wall on her right.
        assert 600.0 - SIZE[0] * 0.5 < lad.x < 600.0 and lad.right
        assert body.facing_right

    def test_she_goes_up_rung_by_rung(self):
        lad = W.Ladder(100.0, 60.0, 360.0, 60.0, True, rung=24.8)
        ys = [lad.step_y(60.0 + 300.0 * i / 600) for i in range(601)]
        assert ys[0] == 60.0 and ys[-1] == 360.0
        assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
        # Pauses at each rung (a grip), then a pull up: many equal heights.
        held = sum(1 for a, b in zip(ys, ys[1:]) if b == a)
        assert held > 100

    def test_mirrored_on_a_right_side(self):
        ls = _ledges()
        body = Body(1300.0, 60.0, ladders=True)
        body.update(DT, ls)
        legs = W.plan_path(body, (900.0, 360.0), _top(ls), ls, SIZE)
        assert legs[1].kind == "ladder_set" and legs[1].right is False
        body.go(legs)
        while body.mode != "climb":
            body.update(DT, ls)
        assert body.ladder.x > 1000.0 and not body.facing_right
        _walk(body, ls)
        assert (body.x, body.y) == (900.0, 360.0) and body.ladder is None

    def test_side_look_never_uses_a_ladder_without_a_window(self):
        ls = _ledges(())
        body = Body(300.0, 60.0, ladders=True)
        body.update(DT, ls)
        body.go(W.up_and_down(body, 140.0, SIZE))
        modes, views, ys = _walk(body, ls)
        assert max(ys) == pytest.approx(200.0)
        assert body.y == 60.0 and body.mode == "rest" and body.ladder is None
        assert all(v is None for v in views)

    def test_off_a_perch_she_climbs_back_down(self):
        ls = _ledges(())
        body = Body(300.0, 60.0, ladders=True)
        body.update(DT, ls)
        body.go(W.plan_path(body, (300.0, 400.0), None, ls, SIZE, perch=True))
        for _ in range(int(30 / DT)):
            if body.update(DT, ls) == "arrived":
                break
        assert body.mode == "perch" and body.ladder is None
        body.leave_perch()
        assert body.ladder is None

    def test_dragged_off_it_the_ladder_poofs(self):
        ls = _ledges()
        body = Body(200.0, 60.0, ladders=True)
        body.update(DT, ls)
        body.go(W.plan_path(body, (800.0, 360.0), _top(ls), ls, SIZE))
        while body.mode != "climb":
            body.update(DT, ls)
        body.place(100.0, 500.0)
        assert body.ladder_view().poof == 0.0
        for _ in range(30):
            body.update(DT, ls)
        assert body.ladder is None


# ---------------------------------------------------------------------------
# The art and the drawing
# ---------------------------------------------------------------------------

ANCHORS = {"_box": (0.0, 0.0, 320.0, 286.0), "prop_ladder": (100.0, 0.0, 100.0, 320.0),
           "prop_ladder_top": (100.0, 0.0, 100.0, 20.0),
           "prop_ladder_foot": (100.0, 0.0, 110.0, 10.0), "prop_poof": (0.0, 0.0, 130.0, 90.0)}


class TestLadderArt:
    def test_ladder_clips_use_real_frames(self):
        for name in LADDER_CLIPS:
            for step in mascot.BODY_CLIPS[name].steps:
                assert step.frame in mascot.FRAME_NAMES
                assert step.frame in mascot.LOOSE_FRAMES  # anchored on their own
        assert set(mascot.LADDER_NEEDS) <= set(mascot.FRAME_NAMES) | set(mascot.PROP_NAMES)
        assert set(mascot.LADDER_PROPS) <= set(mascot.PROP_NAMES)

    def test_the_shipped_art_is_complete(self):
        frames = mascot.available_frames()
        props = mascot.available_props()
        anchors = mascot.load_anchors()
        if not mascot.ladder_ready(frames, props):
            pytest.skip("ladder art not in this checkout")
        for name in mascot.LADDER_FRAMES + mascot.LADDER_PROPS:
            assert name in frames | props
            assert name in anchors
        anim = mascot.MascotAnimator(available=frames)
        for name in LADDER_CLIPS:
            assert anim.has_locomotion(name)
            assert all(s.frame.startswith("ladder_")
                       for s in anim._body_clips[name].steps)

    def test_layout_reaches_however_tall_the_climb(self):
        sprite = SIZE
        rung = mascot.ladder_rung(sprite, ANCHORS)
        assert rung == pytest.approx(124.0 / 320.0 * 320.0 / 5)
        for height in (60.0, 333.0, 900.0):
            pieces = mascot.ladder_layout(0.0, height, sprite, ANCHORS)
            tiles = [(r, c) for n, r, c in pieces if n == "prop_ladder"]
            low = min(max(r[1], c[1]) for r, c in tiles)
            high = max(min(r[1] + r[3], c[1] + c[3]) for r, c in tiles)
            assert low == 0.0
            assert abs(high - height) <= rung / 2 + 0.01
            names = [n for n, *_ in pieces]
            assert "prop_ladder_top" in names and "prop_ladder_foot" in names
        # A taller climb takes more tiles: it is tiled, not stretched.
        n = lambda h: sum(1 for p in mascot.ladder_layout(0.0, h, sprite, ANCHORS)  # noqa: E731
                          if p[0] == "prop_ladder")
        assert n(900.0) > n(333.0) > n(60.0) >= 1

    def test_layout_pulled_up_shows_only_the_top(self):
        pieces = mascot.ladder_layout(250.0, 333.0, SIZE, ANCHORS)
        tiles = [c for n, r, c in pieces if n == "prop_ladder"]
        assert tiles and min(c[1] for c in tiles) >= 240.0
        assert "prop_ladder_foot" not in [n for n, *_ in pieces]

    def test_folded_away_nothing_is_drawn(self):
        assert mascot.ladder_layout(0.0, 0.0, SIZE, ANCHORS) == []
        assert mascot.ladder_layout(120.0, 120.0, SIZE, ANCHORS) == []

    def test_no_art_no_layout(self):
        assert mascot.ladder_layout(0.0, 300.0, SIZE, {}) == []
        assert mascot.poof_rect(0.5, SIZE, {}) is None
        assert not mascot.ladder_ready(mascot.FRAME_NAMES, ())
        assert mascot.ladder_ready(mascot.FRAME_NAMES, mascot.PROP_NAMES)


# ---------------------------------------------------------------------------
# Her life: the animator shows it
# ---------------------------------------------------------------------------

class Rig:
    def __init__(self, tmp_path, props=mascot.PROP_NAMES, frames=mascot.FRAME_NAMES,
                 windows=(WINDOW,)):
        self.settings = PetSettings.from_dict({"wander": False, "lonely_after_min": 0,
                                               "state_file": str(tmp_path / "s.json")})
        self.anim = mascot.MascotAnimator(available=frames, rng=random.Random(2))
        self.anim.set_props(props)
        self.life = PetLife(self.settings, self.anim, rng=random.Random(5),
                            store=NeedsStore(tmp_path / "s.json"), wall=T0)
        self.t = 0.0
        self.windows = list(windows)
        self.pos = (200.0, 60.0)

    def env(self):
        return Env(now=self.t, wall=T0 + self.t, attached=False, position=self.pos,
                   size=SIZE, screens=[SCREEN], windows=self.windows, world_version=1,
                   cursor=(100.0, 800.0), caret=None, key_idle=1e9)

    def run(self, seconds, until=None):
        seen = []
        for _ in range(int(seconds / DT)):
            self.t += DT
            out = self.life.tick(DT, self.env())
            if out.ground is not None:
                self.pos = out.ground
            self.anim.update(DT, State.IDLE)
            seen.append((self.anim.frame, self.anim.locomotion, out))
            if until is not None and until(self):
                break
        return seen


@pytest.fixture(autouse=True)
def _api_off():
    api.attach(None)
    yield
    api.attach(None)


class TestLadderInHerLife:
    def test_abilities_close_brings_her_back_down_the_ladder(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.start()
        rig.run(1.0)
        assert api.window_available()
        assert api.trick("escalera")
        rig.run(60.0, until=lambda r: r.life.body is not None
                and r.life.body.y == 360.0 and not r.life.body.busy)
        assert rig.life.body.ledge.wid == 5
        assert api.trick("ladder_down")
        seen = rig.run(30.0, until=lambda r: r.life.body.y == 60.0
                       and not r.life.body.busy and r.life.body.ladder is None)
        assert "ladder" in {loco for _, loco, _ in seen}
        assert rig.life.body.y == 60.0
        rig.life.stop()

    def test_escalera_takes_her_up_onto_the_window_by_ladder(self, tmp_path):
        rig = Rig(tmp_path)
        rig.life.start()
        rig.run(1.0)
        assert api.trick("escalera")
        seen = rig.run(60.0, until=lambda r: r.life.body is not None
                       and r.life.body.y == 360.0 and r.life.body.ladder is None
                       and not r.life.body.busy)
        locos = [loco for _, loco, _ in seen]
        for name in ("ladder_carry", "ladder_set", "ladder", "ladder_step", "ladder_pull",
                     "ladder_held"):
            assert name in locos, name
        # In that order: carry, set, climb, step off, pull up, sit with it folded.
        firsts = [locos.index(n) for n in ("ladder_carry", "ladder_set", "ladder",
                                           "ladder_step", "ladder_pull", "ladder_held")]
        assert firsts == sorted(firsts)
        assert "climb" not in locos  # never the bare wall
        frames = {f for f, *_ in seen}
        assert {"ladder_climb_a", "ladder_climb_b", "ladder_place", "ladder_proud"} <= frames
        # The ladder was out for the whole climb, from the floor up.
        climbing = [out.ladder for _, loco, out in seen if loco == "ladder"]
        assert climbing and all(v is not None and v.base == 60.0 and v.hi >= 300.0
                                for v in climbing)
        # She ends on the top edge, and the ladder is gone.
        assert rig.pos[1] == 360.0 and rig.life.body.ledge.wid == 5
        assert seen[-1][2].ladder is None or seen[-1][2].ladder.poof >= 0.0
        rig.life.stop()

    @pytest.mark.parametrize("missing", ["prop", "frames"])
    def test_without_the_art_she_clings_on_as_before(self, tmp_path, missing):
        if missing == "prop":
            rig = Rig(tmp_path, props=())  # her ladder frames, but no ladder
        else:
            rig = Rig(tmp_path, frames=tuple(n for n in mascot.FRAME_NAMES
                                             if n not in mascot.LADDER_FRAMES))
        rig.life.start()
        rig.run(1.0)
        assert api.trick("escalera")
        seen = rig.run(60.0, until=lambda r: r.life.body is not None
                       and r.life.body.y == 360.0 and not r.life.body.busy)
        locos = {loco for _, loco, _ in seen}
        assert "climb" in locos and not locos & set(LADDER_CLIPS)
        assert all(out.ladder is None for *_, out in seen)
        assert rig.pos[1] == 360.0
        rig.life.stop()

    def test_with_no_window_she_does_not_deploy_a_ladder(self, tmp_path):
        rig = Rig(tmp_path, windows=())
        rig.life.start()
        rig.run(1.0)
        assert api.trick("escalera")
        seen = rig.run(30.0, until=lambda r: r.life.body is not None
                       and r.life.body.ladder is None and not r.life.body.busy
                       and r.t > 3.0)
        locos = {loco for _, loco, _ in seen}
        assert not locos & set(LADDER_CLIPS)
        assert max(out.ground[1] for *_, out in seen) == pytest.approx(60.0)
        assert rig.pos[1] == 60.0
        rig.life.stop()
