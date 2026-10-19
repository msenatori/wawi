"""pet.liveliness (now an alias for a personality), the owner coming back,
and her life through the dog model (wawi/pet/dog.py): PetLife + a real
animator at local noon and at night."""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime

import pytest

from wawi import mascot
from wawi.pet import dog as D
from wawi.pet import liveliness as L
from wawi.pet import world as W
from wawi.pet.settings import PetSettings
from wawi.reactions import Reaction
from wawi.state import State
from tests.test_pet_life import DT, Rig

NOON = datetime(2027, 1, 15, 12, 0).timestamp()
NIGHT = datetime(2027, 1, 15, 23, 30).timestamp()


class TestSettings:
    def test_default_is_calm(self):
        assert PetSettings.from_dict({}).liveliness == "calm"
        assert PetSettings().liveliness == "calm"

    @pytest.mark.parametrize("raw,level", [("calm", "calm"), ("PLAYFUL", "playful"),
                                           ("tranquila", "calm"), ("inquieta", "playful"),
                                           ("wild", "calm"), (None, "calm"), (3, "calm"),
                                           ("normal", "normal")])
    def test_parsed_forgivingly(self, raw, level):
        assert PetSettings.from_dict({"liveliness": raw}).liveliness == level

    def test_each_level_is_a_personality(self):
        assert [L.personality_for(x) for x in L.LEVELS] == [
            "perra_tranquila", "perra_atenta", "cachorra"]


class TestPresence:
    def test_back_once_after_being_away(self):
        p = L.Presence(300.0)
        assert not p.feed(10.0)
        assert not p.feed(1.0)  # never away: nobody came back
        assert not p.feed(301.0)
        assert not p.feed(900.0)
        assert not p.feed(30.0)  # not yet: no input
        assert p.feed(0.5)
        assert p.away_for == 900.0  # how long they were gone
        assert not p.feed(0.2)

    def test_unknown_idle_or_off_never_greets(self):
        p = L.Presence(300.0)
        for v in (math.inf, 400.0, math.inf, 0.1):
            p.feed(v)
        assert L.Presence(math.inf).feed(0.1) is False
        q = L.Presence(300.0)
        assert not q.feed(math.nan) and not q.feed(math.inf) and not q.feed(0.1)


# -- her life ------------------------------------------------------------------


class LiveRig(Rig):
    """The life rig at a chosen local time, with "seconds since any input"."""

    def __init__(self, tmp_path, wall=NOON, **kw):
        kw.setdefault("attached", True)
        super().__init__(tmp_path, **kw)
        self.wall0 = wall
        self.input_idle = 1.0
        self.meeting = False
        self.life.needs.give({"energy": 100.0, "food": 100.0, "water": 100.0,
                              "fun": 100.0, "affection": 100.0})

    def env(self):
        e = super().env()
        e.wall = self.wall0 + self.t
        e.input_idle = self.input_idle
        e.meeting = self.meeting
        return e

    @property
    def dog(self) -> D.Dog:
        if self.life.dog is None:
            self.step()
        return self.life.dog  # type: ignore[return-value]

    def sleep(self, state=D.DEEP, seconds=3600.0):
        """Put her to sleep now, for a good while."""
        self.step()
        self.dog.settle(self.t, "test")
        self.dog._enter(self.t, state, seconds, "test")
        self.run(3.0)
        assert self.anim.asleep

    def rest(self, seconds=3600.0):
        """Awake and resting, for a good while, nothing scheduled soon."""
        self.step()
        self.dog._enter(self.t, D.REST, seconds, "test")
        self.dog.clock = D._Clock()
        self.run(0.5)


def _logged(caplog, text):
    return any(text in r.getMessage() for r in caplog.records)


class TestDriven:
    def test_her_body_drives_the_animator(self, tmp_path):
        rig = LiveRig(tmp_path)
        assert rig.anim.driven

    def test_she_never_dozes_off_or_gets_up_on_her_own_timer(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.rest()
        rig.run(90.0)
        assert not rig.anim.asleep  # no 10-second doze any more
        rig.sleep()
        seen = rig.run(300.0)
        assert all(f in ("sleeping", "sleeping_b", "ear_twitch") for f, *_ in seen)

    def test_the_bodiless_mascot_is_unchanged(self):
        a = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(1),
                                  sleep_after=2.0)
        for _ in range(int(5 / DT)):
            a.update(DT, State.IDLE)
        assert a.asleep and not a.driven


class TestNoticingTheOwner:
    def test_back_after_an_hour_she_gets_up_and_greets(self, tmp_path, caplog):
        rig = LiveRig(tmp_path)
        rig.cursor = None
        rig.sleep()
        rig.input_idle = 3600.0
        rig.run(2.0)
        assert rig.anim.asleep
        rig.input_idle = 0.3
        with caplog.at_level(logging.INFO, logger="wawi.pet.dog"):
            seen = rig.run(6.0)
        assert not rig.anim.asleep and rig.dog.state in (D.REST, D.PLAY)
        frames = [f for f, *_ in seen]
        assert "stretch" in frames  # got up the way dogs do
        assert {"wag_a", "wag_b"} & set(frames)  # and greeted them
        assert not {"eyeroll", "side_eye"} & set(frames)
        assert _logged(caplog, "Owner back after 60 min: greeting 2")
        assert _logged(caplog, "Woke up (owner_back)")
        rig.run(60.0)
        assert not rig.anim.asleep  # up for a while, not straight back down

    def test_back_after_a_short_break_is_nothing(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.sleep()
        rig.input_idle = 60.0
        rig.run(1.0)
        rig.input_idle = 0.3
        rig.run(2.0)
        assert rig.anim.asleep and rig.dog.deep

    def test_a_voice_turn_wake_is_logged(self, tmp_path, caplog):
        rig = LiveRig(tmp_path)
        rig.sleep()
        with caplog.at_level(logging.INFO, logger="wawi.pet.dog"):
            rig.state = State.LISTENING
            rig.run(0.5)
        assert not rig.anim.asleep
        assert _logged(caplog, "Woke up (voice)")

    def test_a_touch_wakes_her(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.sleep()
        rig.life.touched(rig.t)
        rig.anim.touch("head")
        rig.run(0.5)
        assert rig.dog.state == D.REST and not rig.anim.asleep

    def test_a_background_thought_does_not_wake_her(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.sleep()
        rig.life.react(Reaction(behaviour="side_eye", bubble="mmm", priority=0))
        rig.run(2.0)
        assert rig.anim.asleep and rig.dog.asleep
        assert rig.out.bubble == "mmm"  # the bubble still shows, quietly

    def test_an_event_does(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.sleep()
        rig.life.react(Reaction(behaviour="howl", priority=2))
        rig.run(1.0)
        assert not rig.anim.asleep and rig.dog.state == D.REST


class TestTheCursor:
    def test_asleep_an_ear_at_most_never_a_wake(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.sleep()
        answers = [rig.life.cursor_reaction(r, rig.t + i)
                   for i, r in enumerate(("ear_twitch", "bark", "wake", "ear_twitch"))]
        assert set(answers) <= {"ear_twitch", None}
        assert answers.count("ear_twitch") <= 1
        assert rig.dog.deep

    def test_dozing_a_head_lift_then_back_down(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.sleep(D.DOZE)
        assert rig.life.cursor_reaction("ear_twitch", rig.t) is None
        rig.step()
        assert rig.anim.behaviour == "head_lift" and rig.anim.asleep
        rig.run(12.0)
        assert rig.anim.asleep and rig.dog.state == D.DOZE

    def test_awake_she_wags_once_then_gets_used_to_it(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.rest()
        got = [rig.life.cursor_reaction("wag", rig.t + 5.0 * i) for i in range(10)]
        assert got[0] == "wag" and got[1:].count("wag") == 0


def code_window(owner="Code"):
    return W.Win(7, (300.0, 200.0, 800.0, 500.0), 42, owner)


class TestContextGlances:
    def _ready(self, rig):
        rig.rest()
        rig.life.context.next_at = 0.0
        rig.life.context.since = rig.t - 1000.0  # it has been in front a while

    def test_a_code_editor_gets_a_look_logged_by_category(self, tmp_path, caplog):
        rig = LiveRig(tmp_path, windows=[code_window()])
        self._ready(rig)
        with caplog.at_level(logging.INFO, logger="wawi.pet.life"):
            rig.run(4.0, until=lambda r: r.anim.behaviour == "gesture")
        assert rig.anim.behaviour == "gesture"
        frames = {s.frame for s in rig.anim._oneshot.steps}
        assert frames & {"look_left", "look_right"}
        lines = [r.getMessage() for r in caplog.records if "Context" in r.getMessage()]
        assert lines == ["Context glance: code"]  # the category, never the app or a title

    def test_the_laptop_only_rarely(self, tmp_path):
        rig = LiveRig(tmp_path, windows=[code_window()])
        assert rig.life.personality.context_laptop_chance <= 0.15

    def test_held_back_in_a_meeting_at_night_and_asleep(self, tmp_path):
        rig = LiveRig(tmp_path, windows=[code_window()])
        rig.meeting = True
        self._ready(rig)
        rig.run(4.0)
        assert rig.anim.behaviour != "gesture"
        night = LiveRig(tmp_path / "n", windows=[code_window()], wall=NIGHT)
        self._ready(night)
        night.run(4.0)
        assert night.anim.behaviour != "gesture"
        asleep = LiveRig(tmp_path / "a", windows=[code_window()])
        asleep.sleep(D.DOZE)
        asleep.life.context.next_at = 0.0
        asleep.life.context.since = asleep.t - 1000.0
        asleep.run(4.0)
        assert asleep.anim.behaviour != "gesture"

    def test_a_switch_is_a_glance_toward_the_owner(self, tmp_path):
        rig = LiveRig(tmp_path, attached=False, windows=[code_window("Finder")])
        rig.cursor = (1200.0, 500.0)
        rig.rest()
        rig.life.context.next_at = 0.0
        rig.set_windows([W.Win(8, (900.0, 200.0, 400.0, 300.0), 43, "Safari")])
        rig.run(4.0, until=lambda r: r.anim.behaviour == "gesture")
        assert rig.anim.behaviour == "gesture"
        # The cursor is to her right (she is at x=700).
        assert rig.anim._oneshot.steps[0].frame == "look_right"


class TestAsking:
    def test_hungry_and_the_cursor_comes_she_asks_politely_once(self, tmp_path, caplog):
        rig = LiveRig(tmp_path)
        rig.life.needs.give({"water": -100.0})
        rig.rest()
        rig.run(6.0)
        assert rig.life.mood == "hungry"
        with caplog.at_level(logging.INFO, logger="wawi.pet.life"):
            assert rig.life.ask_need(rig.t)
        frames = [s.frame for s in rig.anim._oneshot.steps]
        assert "hungry" in frames  # a look at the bowl…
        assert frames[0] in ("look_left", "look_right")  # …between looks at you
        rig.step()
        assert rig.out.bubble == "¿Un poco de agua?"
        assert _logged(caplog, "Asking for water")
        rig.run(10.0)
        assert not rig.life.ask_need(rig.t)  # one episode at a time
        rig.life.command("water")
        assert rig.life.requests.need is None  # answered: she stops
        assert not rig.life.ask_need(rig.t + 30 * 60)  # and not again within the hour

    def test_content_she_does_not_ask(self, tmp_path):
        rig = LiveRig(tmp_path)
        rig.rest()
        assert not rig.life.ask_need(rig.t)

    def test_ignored_she_sighs_and_lies_down(self, tmp_path, caplog):
        rig = LiveRig(tmp_path)
        rig.rest()
        rig.life.needs.set_request_state("food", stage=0.0, due=rig.wall0)
        with caplog.at_level(logging.INFO, logger="wawi.pet.life"):
            rig.run(10 * 60.0)
        assert _logged(caplog, "Asking for food")
        assert _logged(caplog, "Request ignored (food): she lies down")
        assert rig.dog.asleep
        assert rig.life.needs.request_state("food")["due"] > rig.wall0 + rig.t


class TestAnimatorHooks:
    def anim(self):
        a = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(1),
                                  sleep_after=1e9)
        a.driven = True
        a.update(DT, State.IDLE)
        return a

    def test_gesture_never_wakes_her_nor_interrupts(self):
        a = self.anim()
        a._fall_asleep()
        assert not a.gesture(("sniff",))
        assert a.asleep
        b = self.anim()
        assert b.react_to_cursor("wag")
        assert not b.gesture(("sniff",))

    def test_gesture_strings_clips_loops_and_timed_ones(self):
        a = self.anim()
        assert a.gesture(("glance", "laptop", "wag@3"), toward=-50.0, seconds=3.0)
        frames = [s.frame for s in a._oneshot.steps]
        assert frames[0] == "look_left"
        assert frames.count("laptop_a") >= 4
        assert frames.count("wag_a") == 2 * 5  # the wag clip (1.5 s) twice
        for _ in range(int(15 / DT)):
            a.update(DT, State.IDLE)
        assert a.behaviour is None and not a.asleep

    def test_lie_down_is_a_sigh_then_sleep(self):
        a = self.anim()
        assert a.lie_down()
        assert a.behaviour == "lie_down" and not a.asleep
        for _ in range(int(3 / DT)):
            a.update(DT, State.IDLE)
        assert a.asleep

    def test_head_lift_keeps_her_asleep(self):
        a = self.anim()
        a._fall_asleep()
        assert a.head_lift(4.0)
        seen = set()
        for _ in range(int(5 / DT)):
            a.update(DT, State.IDLE)
            seen.add(a.frame)
            assert a.asleep
        assert "curl_focus" in seen and a.behaviour is None

    def test_get_up_skips_missing_art(self):
        a = mascot.MascotAnimator(available=set(mascot.FRAME_NAMES) - {"shake_a", "shake_b"},
                                  rng=random.Random(1))
        a.driven = True
        a.update(DT, State.IDLE)
        a._fall_asleep()
        assert a.get_up("dog", ("wake", "shake"))
        assert not a.asleep and a.wake_reason == "dog"
        assert "shake_a" not in {s.frame for s in a._oneshot.steps}

    def test_pick_only_from_the_pool(self):
        a = self.anim()
        picked = []
        for _ in range(30):
            a._stop_oneshot()
            name = a.pick(("yawn", "scratch"))
            picked.append(name)
        assert set(picked) <= {"yawn", "scratch"}
        a._fall_asleep()
        a._stop_oneshot()
        assert a.pick(("yawn",)) is None  # never asleep

    def test_rouse_from_sleep_is_a_real_wake(self):
        a = self.anim()
        a._fall_asleep()
        assert a.rouse("owner_back", then="tilt")
        assert a.wake_reason == "owner_back" and a.idle_for == 0.0

    def test_rouse_waits_out_a_turn(self):
        a = self.anim()
        a._fall_asleep()
        a.update(DT, State.LISTENING)
        assert not a.rouse("owner_back", then="wag")
