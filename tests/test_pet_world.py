"""Her body on the desktop, with synthetic screens and window lists."""

from __future__ import annotations

import random

import pytest

from wawi.pet import world as W
from wawi.pet.world import Body, Leg, Screen, Win

SIZE = (124.0, 110.0)
#: Closest her centre gets to a screen's side: half her width plus the margin.
SIDE = SIZE[0] / 2 + W.EDGE_MARGIN * SIZE[0]
#: One 1440x900 screen, menu bar 25 pt, Dock 60 pt at the bottom.
SCREEN = Screen((0.0, 0.0, 1440.0, 900.0), (0.0, 60.0, 1440.0, 815.0))
#: A second screen to the right, a bit lower.
RIGHT = Screen((1440.0, -100.0, 1920.0, 1080.0), (1440.0, -100.0, 1920.0, 1055.0))


def win(wid, x, y, w, h, **kw):
    return Win(wid, (float(x), float(y), float(w), float(h)), **kw)


def run(body, ledges, seconds, dt=1 / 30):
    events = []
    for _ in range(int(seconds / dt)):
        e = body.update(dt, ledges)
        if e:
            events.append(e)
    return events


class TestReadingTheDesktop:
    def test_quartz_rects_flip_to_appkit(self):
        assert W.cg_to_appkit((10, 25, 300, 200), 900.0) == (10, 675, 300, 200)

    def test_parse_keeps_ordinary_windows_in_order(self):
        raw = [
            {"kCGWindowLayer": 25, "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1440, "Height": 25},
             "kCGWindowOwnerName": "Window Server", "kCGWindowOwnerPID": 1, "kCGWindowNumber": 1},
            {"kCGWindowLayer": 0, "kCGWindowBounds": {"X": 100, "Y": 100, "Width": 600, "Height": 400},
             "kCGWindowOwnerName": "Code", "kCGWindowOwnerPID": 50, "kCGWindowNumber": 7},
            {"kCGWindowLayer": 0, "kCGWindowBounds": {"X": 50, "Y": 50, "Width": 300, "Height": 200},
             "kCGWindowOwnerName": "Python", "kCGWindowOwnerPID": 999, "kCGWindowNumber": 8},
            {"kCGWindowLayer": 0, "kCGWindowBounds": {"X": 5, "Y": 5, "Width": 40, "Height": 20},
             "kCGWindowOwnerName": "Tooltip", "kCGWindowOwnerPID": 51, "kCGWindowNumber": 9},
            {"kCGWindowLayer": 0, "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 800, "Height": 600},
             "kCGWindowOwnerName": "Ghost", "kCGWindowOwnerPID": 52, "kCGWindowNumber": 10,
             "kCGWindowAlpha": 0.0},
            {"kCGWindowLayer": 0, "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1440, "Height": 80},
             "kCGWindowOwnerName": "Dock", "kCGWindowOwnerPID": 53, "kCGWindowNumber": 11},
            {"kCGWindowLayer": 0, "kCGWindowBounds": "garbage", "kCGWindowNumber": 12},
            {"kCGWindowLayer": 0, "kCGWindowBounds": {"X": 300, "Y": 300, "Width": 500, "Height": 300},
             "kCGWindowOwnerName": "Notes", "kCGWindowOwnerPID": 60, "kCGWindowNumber": 13},
        ]
        wins = W.parse_windows(raw, own_pid=999, primary_height=900.0)
        assert [w.wid for w in wins] == [7, 13]
        assert wins[0].rect == (100, 400, 600, 400)
        assert wins[0].owner == "Code"

    def test_fullscreen_is_the_front_window_filling_a_screen(self):
        assert W.fullscreen_active([win(1, 0, 0, 1440, 900)], [SCREEN])
        assert not W.fullscreen_active([win(1, 0, 60, 1440, 815)], [SCREEN])
        assert not W.fullscreen_active([win(2, 10, 70, 500, 400), win(1, 0, 0, 1440, 900)],
                                       [SCREEN])
        assert not W.fullscreen_active([], [SCREEN])


class TestLedges:
    def test_the_floor_is_above_the_dock(self):
        (floor,) = W.ledges([SCREEN], [], SIZE)
        assert floor.y == 60.0 and floor.wid is None
        assert floor.x0 == SIDE and floor.x1 == 1440.0 - SIDE

    def test_a_window_top_is_a_ledge(self):
        ls = W.ledges([SCREEN], [win(5, 200, 200, 500, 300)], SIZE)
        top = [led for led in ls if led.wid == 5]
        assert len(top) == 1 and top[0].y == 500.0
        # All of her stays on the window: its ends are walls, no hanging over.
        assert top[0].x0 == 262.0 and top[0].x1 == 638.0

    def test_the_part_behind_a_front_window_is_not_walkable(self):
        front = win(1, 400, 350, 400, 300)    # covers x 400..800 around y 500
        back = win(2, 100, 200, 1000, 300)    # top at y 500
        tops = [led for led in W.ledges([SCREEN], [front, back], SIZE) if led.wid == 2]
        assert len(tops) == 2
        assert all(led.x1 <= 400 + 124 * 0.3 - 62 + 1 or led.x0 >= 800 + 62 - 124 * 0.3 - 1
                   for led in tops)

    def test_no_headroom_under_the_menu_bar_no_ledge(self):
        ls = W.ledges([SCREEN], [win(5, 100, 400, 600, 470)], SIZE)  # top at 870
        assert [led for led in ls if led.wid == 5] == []

    def test_a_window_spanning_two_screens_splits(self):
        ls = W.ledges([SCREEN, RIGHT], [win(5, 1200, 300, 600, 200)], SIZE)
        tops = sorted((led for led in ls if led.wid == 5), key=lambda led: led.x0)
        assert [led.screen for led in tops] == [0, 1]
        assert tops[0].x1 <= 1440 - 62 and tops[1].x0 >= 1440 + 62

    def test_support_and_landing(self):
        ls = W.ledges([SCREEN], [win(5, 200, 200, 500, 300)], SIZE)
        assert W.support(ls, 400, 500).wid == 5
        assert W.support(ls, 400, 520) is None
        assert W.landing(ls, 400, 700).wid == 5
        assert W.landing(ls, 1000, 700).wid is None  # nothing under: the floor
        # Between the screens' floors: the nearest one.
        assert W.landing(ls, -50, 300).wid is None


class TestBody:
    def test_she_falls_when_the_window_under_her_goes(self):
        w = win(5, 200, 200, 500, 300)
        ls = W.ledges([SCREEN], [w], SIZE)
        body = Body(400.0, 500.0)
        assert body.update(1 / 30, ls) is None and body.ledge.wid == 5
        gone = W.ledges([SCREEN], [], SIZE)
        assert body.update(1 / 30, gone) == "fell"
        events = run(body, gone, 2.0)
        assert "landed" in events
        assert body.y == 60.0 and body.mode == "rest"

    def test_she_falls_when_the_window_moves(self):
        ls = W.ledges([SCREEN], [win(5, 200, 200, 500, 300)], SIZE)
        body = Body(400.0, 500.0)
        body.update(1 / 30, ls)
        moved = W.ledges([SCREEN], [win(5, 200, 100, 500, 300)], SIZE)  # top now 400
        assert body.update(1 / 30, moved) == "fell"
        run(body, moved, 2.0)
        assert body.y == 400.0  # landed on the same window, lower down

    def test_the_fall_shows_a_landing(self):
        ls = W.ledges([SCREEN], [], SIZE)
        body = Body(400.0, 600.0)
        modes = set()
        for _ in range(60):
            body.update(1 / 30, ls)
            modes.add(body.mode)
        assert {"fall", "land", "rest"} <= modes

    def test_walking_is_calm_and_arrives(self):
        ls = W.ledges([SCREEN], [], SIZE)
        body = Body(300.0, 60.0)
        body.go([Leg("walk", 500.0)])
        xs = []
        for _ in range(30):
            body.update(1 / 30, ls)
            xs.append(body.x)
        assert xs[-1] - 300.0 == pytest.approx(Body.WALK_SPEED, abs=2)
        assert body.facing_right and body.mode == "walk"
        assert "arrived" in run(body, ls, 6.0)
        assert body.x == 500.0 and body.mode == "rest"

    def test_she_never_walks_off_screen(self):
        ls = W.ledges([SCREEN], [], SIZE)
        body = Body(300.0, 60.0)
        body.go([Leg("walk", -500.0)])
        run(body, ls, 12.0)
        assert body.x == SIDE and body.y == 60.0

    def test_patrol_goes_back_and_forth(self):
        ls = W.ledges([SCREEN], [], SIZE)
        body = Body(700.0, 60.0)
        body.start_patrol(600.0, 800.0)
        xs = []
        for _ in range(150):
            body.update(1 / 30, ls)
            xs.append(body.x)
        assert min(xs) == pytest.approx(600.0) and max(xs) == pytest.approx(800.0)
        assert body.mode == "trot"
        body.stop()
        assert body.mode == "rest" and body.patrol is None

    def test_paused_she_stays_put(self):
        ls = W.ledges([SCREEN], [], SIZE)
        body = Body(300.0, 60.0)
        body.go([Leg("walk", 500.0)])
        body.paused = True
        run(body, ls, 1.0)
        assert body.x == 300.0

    def test_up_onto_a_window_climbing_its_side(self):
        w = win(5, 600, 60, 400, 300)
        ls = W.ledges([SCREEN], [w], SIZE)
        target = next(led for led in ls if led.wid == 5)
        body = Body(200.0, 60.0)
        body.update(1 / 30, ls)
        legs = W.plan_path(body, (800.0, 360.0), target, ls, SIZE)
        assert [leg.kind for leg in legs] == ["walk", "climb", "shift", "walk"]
        assert legs[0].to < 600  # climbs the LEFT side, the one nearer to her
        body.go(legs)
        modes = set()
        for _ in range(int(30 / (1 / 30))):
            e = body.update(1 / 30, ls)
            modes.add(body.mode)
            if e == "arrived":
                break
        assert "climb" in modes
        assert (body.x, body.y) == (800.0, 360.0) and body.ledge.wid == 5

    def test_down_is_a_step_off_the_edge(self):
        w = win(5, 600, 60, 400, 300)
        ls = W.ledges([SCREEN], [w], SIZE)
        body = Body(800.0, 360.0)
        body.update(1 / 30, ls)
        legs = W.plan_path(body, (200.0, 60.0), None, ls, SIZE)
        assert [leg.kind for leg in legs] == ["walk", "walk_off"]
        body.go(legs)
        events = run(body, ls, 15.0)
        assert "fell" in events and "landed" in events
        assert body.y == 60.0

    @pytest.mark.parametrize("start,goal", [(1200.0, 2000.0), (2000.0, 600.0)])
    def test_she_crosses_to_the_other_monitor(self, start, goal):
        """Observed: stuck walking on the spot at the right side of a monitor."""
        ls = W.ledges([SCREEN, RIGHT], [], SIZE)
        floors = {led.screen: led for led in ls}
        here, there = (floors[0], floors[1]) if start < 1440 else (floors[1], floors[0])
        body = Body(start, here.y)
        body.update(1 / 30, ls)
        body.go(W.plan_path(body, (goal, there.y), there, ls, SIZE))
        for _ in range(int(40 / (1 / 30))):
            e = body.update(1 / 30, ls)
            if e in ("landed",):
                body.go(W.plan_path(body, (goal, there.y), there, ls, SIZE))
            if e == "arrived":
                break
        assert (body.x, body.y) == (goal, there.y)

    def test_perch_stays_in_the_air(self):
        ls = W.ledges([SCREEN], [], SIZE)
        body = Body(300.0, 60.0)
        legs = W.plan_path(body, (300.0, 400.0), None, ls, SIZE, perch=True)
        assert [leg.kind for leg in legs][-1] == "perch"
        body.go(legs)
        run(body, ls, 20.0)
        assert body.mode == "perch" and body.y == 400.0
        run(body, ls, 2.0)
        assert body.y == 400.0  # no gravity on a perch
        body.leave_perch()
        run(body, ls, 2.0)
        assert body.y == 60.0


class TestChoosingSpots:
    def test_spots_avoid_the_cursor_and_the_front_text(self):
        front = win(1, 100, 100, 700, 500)
        ls = W.ledges([SCREEN], [front], SIZE)
        avoid = W.avoid_rects(front, (1000.0, 80.0), None)
        rng = random.Random(1)
        for _ in range(200):
            spot = W.choose_spot(ls, (300.0, 60.0), SIZE, avoid, rng, [SCREEN], max_dist=2000)
            assert spot is not None
            led, x = spot
            r = W.her_rect(x, led.y, SIZE)
            assert not any(W.intersects(r, a) for a in avoid)
            assert W.on_screen([SCREEN], r)

    def test_the_floor_wins_when_window_tops_weigh_less(self):
        """pet.liveliness: fewer window tops (fewer ladders) when they weigh
        less than the floor and her own level gets a bonus."""
        ls = W.ledges([SCREEN], [win(1, 200, 100, 900, 400)], SIZE)

        def tops(**kw):
            rng = random.Random(3)
            picks = [W.choose_spot(ls, (300.0, 60.0), SIZE, [], rng, [SCREEN],
                                   max_dist=2000, **kw) for _ in range(400)]
            return sum(1 for led, _x in picks if led.wid is not None)

        old = tops()
        fewer = tops(top_weight=0.8, same_level_weight=1.5)
        assert fewer < old * 0.7

    def test_nowhere_free_is_none(self):
        everything = (0.0, 0.0, 5000.0, 5000.0)
        ls = W.ledges([SCREEN], [], SIZE)
        assert W.choose_spot(ls, (300.0, 60.0), SIZE, [everything], random.Random(1)) is None

    def test_she_sits_next_to_the_caret_never_on_the_text(self):
        caret = (500.0, 400.0, 2.0, 16.0)
        x, y = W.caret_spot(caret, SIZE, [SCREEN])
        r = W.her_rect(x, y, SIZE)
        assert not W.intersects(r, W.inflate(caret, 20.0))
        line = (caret[0] - 400, caret[1], 400, caret[3])  # what you just typed
        assert not W.intersects(r, line)
        assert x > caret[0]  # to the right, where the text is not yet

    def test_caret_at_the_right_edge_she_sits_below(self):
        caret = (1400.0, 500.0, 2.0, 16.0)
        x, y = W.caret_spot(caret, SIZE, [SCREEN])
        assert y + SIZE[1] < caret[1]
        assert W.on_screen([SCREEN], W.her_rect(x, y, SIZE))

    def test_caret_with_no_room_anywhere_is_none(self):
        tiny = Screen((0, 0, 200, 150), (0, 0, 200, 150))
        assert W.caret_spot((90.0, 60.0, 2.0, 16.0), SIZE, [tiny]) is None

    def test_clamp_keeps_her_on_the_nearest_screen(self):
        assert W.clamp_to_screens([SCREEN, RIGHT], -300.0, 60.0, SIZE) == (SIDE, 60.0)
        x, y = W.clamp_to_screens([SCREEN, RIGHT], 5000.0, 300.0, SIZE)
        assert x == 1440 + 1920 - SIDE and y == 300.0
