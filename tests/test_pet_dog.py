"""The dog model (wawi/pet/dog.py): a real dog's day at the desk. Pure parts
first (habituation, greetings, requests, the state machine and its time
budget), then whole simulated hours of PetLife driving a real animator."""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import replace
from datetime import datetime

import pytest

from wawi import mascot
from wawi.cursor_reactions import CursorReactions
from wawi.pet import dog as D
from wawi.pet import personality as P
from wawi.pet import world as W
from wawi.pet.life import Env, PetLife
from wawi.pet.needs import NeedsStore
from wawi.pet.settings import PetSettings
from wawi.state import State

CALM = P.load("perra_tranquila")
MIN, HOUR = 60.0, 3600.0
WORKING = D.Sense(hour=11.0, owner_here=True, working=True)


def dog(state=D.REST, seed=1, p=CALM, now=0.0):
    return D.Dog(p, random.Random(seed), now, state=state)


# -- habituation ----------------------------------------------------------------


class TestHabituation:
    def test_halves_per_exposure_and_recovers_in_half_an_hour(self):
        h = D.Habituation(0.5, 10 * MIN, 0.3)
        assert h.expose("cursor", 0.0) == pytest.approx(1.0)
        assert h.expose("cursor", 5.0) == pytest.approx(0.5, abs=0.01)
        assert h.expose("cursor", 10.0) == pytest.approx(0.25, abs=0.02)
        assert h.novelty("cursor", 10.0 + 30 * MIN) > 0.9
        assert h.novelty("typing", 10.0) == 1.0  # per stimulus

    def test_a_cursor_that_never_stops_is_no_news(self):
        h = D.Habituation(0.5, 10 * MIN, 0.3)
        reacted = [t for t in range(0, 3600, 3) if h.notable("cursor", float(t))]
        assert reacted[:2] == [0, 3]
        assert len(reacted) <= 3  # then nothing, for the whole hour

    def test_asleep_small_things_never_wake_her(self):
        d = dog(D.DEEP)
        answers = [d.notice(float(t), "cursor") for t in range(0, 600, 20)]
        assert d.state == D.DEEP
        assert set(answers) <= {"ear", None}
        ears = [t for t, a in zip(range(0, 600, 20), answers) if a == "ear"]
        assert all(b - a >= CALM.habituation_ear_flick_every_s for a, b in zip(ears, ears[1:]))
        e = dog(D.DOZE)
        assert e.notice(0.0, "cursor") == "lift" and e.state == D.DOZE
        assert e.notice(5.0, "typing") is None  # one moment, one reaction

    def test_awake_she_looks_when_it_is_new(self):
        d = dog(D.REST)
        assert d.notice(0.0, "app") == "look"
        assert d.notice(1.0, "app") is None


# -- greeting -----------------------------------------------------------------------


def _wag_seconds(names):
    return sum(float(n.split("@")[1]) for n in names if n.startswith("wag@"))


class TestGreeting:
    @pytest.mark.parametrize("minutes,level", [(1, 0), (4.9, 0), (6, 1), (29, 1), (31, 2),
                                               (119, 2), (121, 3), (600, 3)])
    def test_levels_scale_with_the_absence(self, minutes, level):
        assert D.greeting_level(CALM, minutes * MIN) == level

    def test_short_absence_awake_is_a_look_and_a_short_wag(self):
        cues = dog(D.REST).owner_back(0.0, 10 * MIN)
        assert [c.kind for c in cues] == ["gesture"]
        assert cues[0].names[0] == "glance" and _wag_seconds(cues[0].names) == pytest.approx(3.0)

    def test_short_absence_asleep_is_at_most_a_head_lift(self):
        assert dog(D.DEEP).owner_back(0.0, 10 * MIN) == []
        d = dog(D.DOZE)
        assert [c.kind for c in d.owner_back(0.0, 10 * MIN)] == ["head_lift"]
        assert d.state == D.DOZE

    def test_an_hour_away_she_gets_up_and_wags(self):
        d = dog(D.DEEP)
        cues = d.owner_back(0.0, 60 * MIN, WORKING)
        assert d.state == D.REST
        assert [c.kind for c in cues][:2] == ["get_up", "gesture"]
        assert 5.0 <= _wag_seconds(cues[1].names) <= 10.0

    def test_hours_away_the_full_welcome_then_a_shake(self):
        d = dog(D.DEEP)
        cues = d.owner_back(0.0, 3 * HOUR, WORKING)
        names = cues[1].names
        assert "hop" in names and names[-1] == "shake"
        assert 8.0 <= _wag_seconds(names) <= 20.0


# -- requests -------------------------------------------------------------------------


class TestRequests:
    def test_a_few_tries_minutes_apart_then_she_gives_up(self):
        r = D.Request(CALM, random.Random(3), 0.0)
        assert r.can_start(0.0)
        r.start(0.0, "water")
        tries, t = [], 0.0
        while not r.gave_up(t):
            if r.due(t):
                tries.append(t)
                r.tried(t)
            t += 1.0
        assert 2 <= len(tries) <= 3
        assert all(MIN <= b - a <= 3 * MIN for a, b in zip(tries, tries[1:]))
        assert r.need is None
        assert not r.can_start(t + 59 * MIN)  # not again within the hour
        assert r.can_start(t + 3 * HOUR)

    def test_answered_it_stops(self):
        r = D.Request(CALM, random.Random(3), 0.0)
        r.start(0.0, "food")
        r.tried(0.0)
        assert not r.answered(10.0, "water")  # not what she asked for
        assert r.answered(10.0, "food")
        assert not r.due(1e5) and not r.gave_up(1e5)


# -- the state machine --------------------------------------------------------------


def simulate(d, hours, sense=WORKING, step=1.0, start=0.0):
    t, cues = start, []
    while t < start + hours * HOUR:
        s = replace(sense, hour=(sense.hour + (t - start) / HOUR) % 24)
        cues += [(t, c) for c in d.tick(t, s)]
        t += step
    return t, cues


class TestDogDay:
    def test_deep_sleep_always_passes_through_a_doze(self):
        d = dog(D.DEEP)
        seen = []
        t = 0.0
        while t < 6 * HOUR:
            d.tick(t, WORKING)
            if not seen or seen[-1] != d.state:
                seen.append(d.state)
            t += 1.0
        for a, b in zip(seen, seen[1:]):
            if a == D.DEEP:
                assert b == D.DOZE

    def test_sleep_bouts_are_long(self):
        d = dog(D.DEEP, seed=4)
        bouts = [d._duration(D.DEEP, 0.0, D.Sense(hour=11.0)) / MIN for _ in range(400)]
        assert 15.0 <= min(bouts) and max(bouts) <= 45.0
        assert 17.0 <= sorted(bouts)[200] <= 24.0  # median about 20 min

    def test_time_budget_while_the_owner_works(self):
        """Rule 1, averaged over many 3-hour mornings from 11:00."""
        total: Counter = Counter()
        for seed in range(60):
            d = dog(seed=seed)
            t, _ = simulate(d, 3)
            total.update(d.budget(t))
        whole = sum(total.values())
        share = {k: v / whole for k, v in total.items()}
        assert 0.60 <= share[D.DEEP] + share[D.DOZE] <= 0.70
        assert 0.20 <= share[D.REST] <= 0.30
        assert 0.05 <= share[D.ACTIVE] <= 0.10
        assert share[D.PLAY] < 0.01

    def test_resting_gestures_are_rare(self):
        d = dog(D.REST)
        d.until = 1e9  # a very long rest
        _t, cues = simulate(d, 2)
        gestures = [t for t, c in cues if c.kind == "behave"]
        glances = [t for t, c in cues if c.kind == "glance"]
        assert all(b - a >= 10 * MIN for a, b in zip(gestures, gestures[1:]))
        assert all(b - a >= 3 * MIN - 1 for a, b in zip(glances, glances[1:]))
        assert all(set(c.names) <= set(D.REST_POOL) for _t, c in cues if c.kind == "behave")

    def test_night_is_for_sleeping(self):
        night = D.Sense(hour=23.0, night=True, owner_here=False)
        total: Counter = Counter()
        for seed in range(20):
            d = dog(D.DOZE, seed=seed)
            t, _ = simulate(d, 6, sense=night)
            total.update(d.budget(t))
        whole = sum(total.values())
        assert (total[D.DEEP] + total[D.DOZE]) / whole > 0.85

    def test_more_up_in_the_morning_than_after_lunch(self):
        def active(hour):
            total = 0.0
            for seed in range(30):
                d = dog(seed=seed)
                t, _ = simulate(d, 2, sense=replace(WORKING, hour=hour))
                total += d.budget(t)[D.ACTIVE]
            return total
        assert active(8.0) > active(13.0) * 1.5

    def test_waking_is_a_sequence_not_a_jump(self):
        d = dog(D.DOZE, seed=2)
        d.after = "wake"
        d.until = 0.0
        d.p = replace(CALM, sleep_lie_back_down=0.0, wake_stretch=1.0, wake_shake=1.0)
        cues = d.tick(0.0, WORKING)
        assert [c.kind for c in cues][:2] == ["head_lift", "get_up"]
        assert 3.0 <= cues[0].seconds <= 10.0
        assert cues[1].names == ("wake", "shake")
        assert d.state == D.REST

    def test_half_the_time_she_lies_back_down(self):
        backs = 0
        for seed in range(200):
            d = dog(D.DOZE, seed=seed)
            d.after, d.until = "wake", 0.0
            d.tick(0.0, replace(WORKING, hour=9.0))  # morning: no extra sleepiness
            backs += d.asleep
        assert 0.3 <= backs / 200 <= 0.5

    def test_a_strong_stimulus_wakes_her_straight_up(self):
        d = dog(D.DEEP)
        assert d.strong(5.0, "voice") == []
        assert d.state == D.REST and d.reason == "voice"


class TestPlay:
    def _ready(self, seed=0, p=CALM):
        d = dog(D.DEEP, seed=seed, p=replace(p, play_chance=1.0))
        return d

    def test_never_during_work_nor_at_night(self):
        d = self._ready()
        assert d.maybe_play(0.0, WORKING, "owner_back") is None
        assert d.maybe_play(0.0, D.Sense(hour=23.0, night=True), "owner_back") is None
        assert d.state != D.PLAY

    def test_at_most_once_a_day_for_the_calm_dog(self):
        d = self._ready()
        free = D.Sense(hour=18.0, owner_here=True, working=False)
        assert d.maybe_play(0.0, free, "owner_back") is not None
        assert d.state == D.PLAY
        assert 5.0 <= d.until - d.since <= 20.0
        assert d.maybe_play(3 * HOUR, free, "owner_back") is None
        assert d.maybe_play(25 * HOUR, free, "owner_back") is not None

    def test_the_puppy_plays_up_to_three_times(self):
        d = self._ready(p=P.load("cachorra"))
        free = D.Sense(hour=18.0, owner_here=True, working=False)
        plays = [t for t in range(0, int(20 * HOUR), 600)
                 if d.maybe_play(float(t), free, "long_rest") is not None]
        assert len(plays) == 3
        assert all(b - a >= 2.5 * HOUR for a, b in zip(plays, plays[1:]))

    def test_the_day_never_plays_by_chance(self):
        """No trigger, no play: a whole working day of ticks."""
        for seed in range(10):
            d = dog(seed=seed, p=replace(CALM, play_chance=1.0))
            _t, cues = simulate(d, 10, sense=replace(WORKING, hour=8.0))
            assert not [c for _t, c in cues if c.kind == "play"]


# -- whole hours of her life (PetLife + a real animator) ----------------------------

ELEVEN = datetime(2027, 1, 15, 11, 0).timestamp()
SCREEN = W.Screen((0.0, 0.0, 1440.0, 900.0), (0.0, 60.0, 1440.0, 815.0))
APPS = ("Code", "Safari", "Slack", "Terminal", "Finder")
ENERGETIC = set(D.PLAY_POOL) | {"jump_flop", "parachute", "party", "cheer"}
TRICKS = {"roll", "volume_knob", "eyeroll", "side_eye", "squint", "smug", "shrug"}


class Desk:
    """An owner at work from 11:00: the cursor jitters about, they type in
    bursts and switch apps every few minutes. Ticks like the panel does."""

    DT = 0.25

    def __init__(self, tmp_path, seed=1, personality="perra_tranquila", attached=False):
        self.settings = PetSettings.from_dict({
            "personality": personality, "state_file": str(tmp_path / "s.json"),
            "lonely_after_min": 0})
        self.anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(seed))
        self.life = PetLife(self.settings, self.anim, rng=random.Random(seed + 10),
                            store=NeedsStore(tmp_path / "s.json"), wall=ELEVEN)
        self.life.needs.give({"energy": 100.0, "food": 100.0, "water": 100.0,
                              "fun": 100.0, "affection": 100.0})
        self.rng = random.Random(seed + 20)
        self.cursor_reactions = CursorReactions(clock=lambda: self.t)
        self.t = 0.0
        self.pos = (700.0, 60.0)
        self.attached = attached
        self.cursor = (600.0, 500.0)
        self.key_idle = 0.0
        self.win = W.Win(7, (300.0, 200.0, 800.0, 500.0), 42, "Code")
        self.next_switch = 300.0
        self.state = State.IDLE
        self.starts: list[tuple[float, str]] = []
        self._last = None

    def step(self):
        rng, dt = self.rng, self.DT
        self.t += dt
        x, y = self.cursor
        self.cursor = (min(1400.0, max(0.0, x + rng.uniform(-30, 30))),
                       min(880.0, max(0.0, y + rng.uniform(-20, 20))))
        self.key_idle = 0.0 if rng.random() < 0.3 else self.key_idle + dt
        if self.t >= self.next_switch:
            self.next_switch = self.t + rng.uniform(120, 900)
            self.win = W.Win(7, self.win.rect, 42, rng.choice(APPS))
        env = Env(now=self.t, wall=ELEVEN + self.t, attached=self.attached,
                  position=None if self.attached else self.pos, size=(124.0, 110.0),
                  screens=[SCREEN], windows=[self.win], world_version=1, cursor=self.cursor,
                  key_idle=self.key_idle, input_idle=min(self.key_idle, 0.5))
        out = self.life.tick(dt, env)
        if out.ground is not None:
            self.pos = out.ground
        anim = self.anim
        offset = (self.cursor[0] - self.pos[0], self.cursor[1] - self.pos[1] - 60.0)
        r = self.cursor_reactions.sample(
            offset, idle=anim.key == "idle" and anim.behaviour is None and not anim.acting,
            asleep=anim.asleep)
        r = self.life.cursor_reaction(r, self.t)
        if r:
            anim.react_to_cursor(r)
        anim.update(dt, self.state)
        if anim.behaviour != self._last:
            if anim.behaviour not in (None, "blink"):
                self.starts.append((self.t, anim.behaviour))
            self._last = anim.behaviour

    def run(self, seconds):
        for _ in range(int(seconds / self.DT)):
            self.step()


@pytest.fixture(scope="module")
def mornings(tmp_path_factory):
    """Three deterministic mornings, 11:00 to 14:00."""
    out = []
    for seed in (1, 2, 3):
        desk = Desk(tmp_path_factory.mktemp(f"desk{seed}"), seed=seed)
        desk.run(3 * HOUR)
        out.append(desk)
    return out


class TestAMorningAtTheDesk:

    def test_the_time_budget(self, mornings):
        total: Counter = Counter()
        for desk in mornings:
            total.update(desk.life.dog.budget(desk.t))
        whole = sum(total.values())
        sleep = (total[D.DEEP] + total[D.DOZE]) / whole
        assert 0.58 <= sleep <= 0.76
        assert 0.17 <= total[D.REST] / whole <= 0.33
        assert 0.03 <= total[D.ACTIVE] / whole <= 0.12
        assert total[D.PLAY] == 0.0

    def test_no_bursts_no_zoomies_no_tricks(self, mornings):
        for desk in mornings:
            names = [n for _t, n in desk.starts]
            assert len(names) / 3.0 <= 20  # things she visibly does per hour, all told
            assert not ENERGETIC & set(names)
            assert not TRICKS & set(names)
            # never more than 4 things in any 2 minutes
            times = [t for t, _n in desk.starts]
            assert all(sum(1 for u in times if t <= u < t + 120) <= 4 for t in times)

    def test_asleep_she_stays_where_she_is(self, tmp_path):
        """Never a walk mid-sleep, whatever the cursor and the apps do."""
        desk = Desk(tmp_path)
        desk.run(5.0)
        desk.life.dog.settle(desk.t, "test")
        desk.life.dog._enter(desk.t, D.DEEP, HOUR, "test")
        desk.run(30.0)
        assert desk.anim.asleep
        start, since = desk.pos, desk.t
        desk.run(20 * MIN)
        assert desk.life.dog.deep and desk.anim.asleep
        assert desk.pos == start
        assert {n for t, n in desk.starts if t > since} <= {"cursor_ear_twitch"}

    def test_a_voice_turn_wakes_her_at_once(self, tmp_path, caplog):
        desk = Desk(tmp_path)
        desk.run(5.0)
        desk.life.dog.settle(desk.t, "test")
        desk.life.dog._enter(desk.t, D.DEEP, HOUR, "test")
        desk.run(20.0)
        assert desk.anim.asleep and desk.life.dog.deep
        with caplog.at_level(logging.INFO, logger="wawi.pet.dog"):
            desk.state = State.LISTENING
            desk.step()
            desk.step()
        assert not desk.anim.asleep
        assert desk.life.dog.state == D.REST
        assert any("Woke up (voice)" in r.getMessage() for r in caplog.records)
        desk.state = State.IDLE
        desk.run(60.0)
        assert not desk.life.dog.asleep  # attentive for a while, not straight back down
