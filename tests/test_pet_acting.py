"""Reactions → what she shows, which reaction gets the stage, bubbles, and
the typing/lonely trigger."""

from __future__ import annotations

import random

import pytest

from wawi import mascot
from wawi.pet import acting
from wawi.pet.acting import (
    BUBBLE_MAX,
    Director,
    Plan,
    bubble_seconds,
    bubble_text,
    clip_for_behaviour,
    plan_for,
)
from wawi.pet.watch import TypingDetector, WatchTrigger
from wawi.reactions import Event, Reaction

ALL_CLIPS = set(mascot.CLIPS) | set(mascot.ACT_CLIPS)


class TestReactionToClip:
    @pytest.mark.parametrize("behaviour,clip", [
        ("run", "run"), ("bark", "bark"), ("howl", "howl"), ("alarm", "alarm"),
        ("dig", "dig"), ("newspaper", "newspaper"), ("mail", "mail"),
        ("ball_bring", "ball_bring"), ("spaceship", "ship_board"),
        ("detective", "detective"), ("doctor", "doctor"), ("celebrate", "party"),
        ("sad", "sad"), ("BARK", "bark"), ("tail_chase", "tail_chase"),
    ])
    def test_known_behaviours(self, behaviour, clip):
        assert clip_for_behaviour(behaviour, ALL_CLIPS) == clip

    def test_unknown_degrades_to_a_bark_then_a_tilt(self):
        assert clip_for_behaviour("moonwalk", ALL_CLIPS) == "bark"
        assert clip_for_behaviour("moonwalk", {"tilt", "wag"}) == "tilt"
        assert clip_for_behaviour("moonwalk", set()) is None

    def test_movement_only_and_nothing(self):
        assert clip_for_behaviour("walk_to", ALL_CLIPS) is None
        assert clip_for_behaviour(None, ALL_CLIPS) is None
        assert clip_for_behaviour("", ALL_CLIPS) is None

    def test_every_supported_behaviour_is_playable_with_today_s_frames(self):
        anim = mascot.MascotAnimator(available=mascot.available_frames())
        have = set(anim._clips) | set(anim._acts)
        for b in ("run", "bark", "howl", "alarm", "dig", "newspaper", "mail", "ball_bring",
                  "spaceship", "detective", "doctor", "celebrate", "sad"):
            assert clip_for_behaviour(b, have) == acting.BEHAVIOUR_CLIPS[b], b

    def test_with_the_old_frames_only_acts_borrow_relatives(self):
        old = mascot.FRAME_NAMES[:mascot.FRAME_NAMES.index("walk_a")]
        anim = mascot.MascotAnimator(available=old)
        assert {"bark", "eat", "drink", "watch", "detective"} <= set(anim._acts)
        assert anim._acts["eat"].steps[0].frame == "sniff"
        assert anim._acts["eat"].steps[0].prop == "prop_bowl_food"


class TestPlans:
    def test_costumes_she_has_are_worn_others_ignored(self):
        p = plan_for(Reaction(behaviour="run", costume="firefighter", until_key="web:x"),
                     ALL_CLIPS, ["firefighter"])
        assert p.costume == "firefighter" and p.move == "patrol" and p.until
        p = plan_for(Reaction(costume="tutu"), ALL_CLIPS, [])
        assert p.costume is None and p.empty

    def test_a_costume_alone_is_worn_for_a_while(self):
        p = plan_for(Reaction(costume="umbrella"), ALL_CLIPS, ["umbrella"])
        assert p.hold == acting.COSTUME_ONLY_HOLD and p.clip is None

    def test_spaceship_without_a_time_has_a_default_trip(self):
        assert plan_for(Reaction(behaviour="spaceship"), ALL_CLIPS).hold == acting.SHIP_DEFAULT
        p = plan_for(Reaction(behaviour="spaceship", until_key="fetch:1"), ALL_CLIPS)
        assert p.until and p.hold == 0.0 and p.move == "ship"

    def test_bubble_sound_and_speak(self):
        p = plan_for(Reaction(bubble="  hola\n mundo ", sound="Bark!", speak=True), ALL_CLIPS)
        assert p.bubble == "hola mundo" and p.sound == "bark!" and p.speak
        assert plan_for(Reaction(speak=True), ALL_CLIPS).speak is False

    def test_never_raises(self):
        assert plan_for(Reaction(duration="soon"), ALL_CLIPS) == Plan()  # type: ignore[arg-type]

    def test_costume_frames_are_only_real_ones(self):
        have = set(mascot.FRAME_NAMES)
        assert mascot.costume_frames("firefighter", have) == (
            "firefighter_a", "firefighter_b", "firefighter_hose")
        assert mascot.costume_frames("umbrella", have) == ("umbrella",)
        assert mascot.costume_frames("umbrella", {"idle_a"}) == ()
        assert mascot.costume_frames("../etc", have) == ()
        run = mascot.costume_clip("firefighter", have, running=True)
        assert {s.frame for s in run.steps} == {"firefighter_a", "firefighter_b"} and run.loop
        still = mascot.costume_clip("firefighter", have)
        assert [s.frame for s in still.steps] == ["firefighter_hose"]
        assert mascot.costume_clip("tutu", have) is None


class TestBubbles:
    def test_short_text_untouched(self):
        assert bubble_text("¡Volvió la página!") == "¡Volvió la página!"

    def test_long_text_cut_on_a_word_with_an_ellipsis(self):
        text = "La página de tu proyecto está caída desde hace un rato y no responde ni a palos, ojo"
        out = bubble_text(text)
        assert len(out) <= BUBBLE_MAX and out.endswith("…")
        assert text.startswith(out[:-1])
        assert not out[:-1].endswith(" ")

    def test_one_endless_word_is_cut_hard(self):
        out = bubble_text("a" * 200)
        assert len(out) == BUBBLE_MAX

    def test_nothing_to_say(self):
        assert bubble_text(None) is None and bubble_text("   \n") is None

    def test_how_long_it_stays(self):
        assert bubble_seconds("hola") == acting.BUBBLE_MIN_S
        assert bubble_seconds("x" * 80) <= acting.BUBBLE_MAX_S


def offer(d, now=0.0, **kw):
    r = Reaction(**kw)
    return d.offer(r, plan_for(r, ALL_CLIPS, ["firefighter", "detective"]), now)


class TestDirector:
    def test_first_reaction_takes_the_stage(self):
        d = Director()
        act = offer(d, behaviour="bark")
        assert d.active is act

    def test_higher_priority_preempts_and_a_long_one_resumes(self):
        d = Director()
        fire = offer(d, behaviour="run", costume="firefighter", priority=2, until_key="web:a")
        det = offer(d, behaviour="detective", costume="detective", priority=3, duration=15)
        assert d.active is det and d.suspended == [fire]
        assert d.finish(20.0) is fire

    def test_a_short_one_shot_is_dropped_when_preempted(self):
        d = Director()
        offer(d, behaviour="bark", priority=1)
        big = offer(d, behaviour="howl", priority=3)
        assert d.active is big and d.suspended == []
        assert d.finish(1.0) is None

    def test_lower_priority_waits_its_turn(self):
        d = Director()
        first = offer(d, behaviour="howl", priority=2)
        assert offer(d, behaviour="bark", priority=1) is None
        assert d.active is first
        nxt = d.finish(5.0)
        assert nxt.reaction.behaviour == "bark"

    def test_stale_queued_chatter_is_dropped(self):
        d = Director()
        offer(d, behaviour="howl", priority=2, now=0.0)
        offer(d, behaviour="bark", priority=0, now=0.0)
        assert d.finish(500.0) is None

    def test_cancel_key_ends_the_act_wherever_it_is(self):
        d = Director()
        fire = offer(d, behaviour="run", costume="firefighter", priority=2, until_key="web:a")
        offer(d, behaviour="detective", priority=3, duration=15)
        offer(d, cancel_key="web:a")
        assert fire not in d.suspended and not fire.cancelled  # was suspended: just dropped
        d2 = Director()
        ship = offer(d2, behaviour="spaceship", priority=1, until_key="fetch:1")
        nxt = offer(d2, behaviour="celebrate", priority=1, cancel_key="fetch:1")
        assert ship.cancelled and d2.ended == [ship]
        assert d2.active is nxt and nxt.reaction.behaviour == "celebrate"

    def test_a_cancel_for_nothing_does_nothing(self):
        d = Director()
        act = offer(d, behaviour="howl", priority=1, until_key="a")
        assert offer(d, cancel_key="b") is None
        assert d.active is act

    def test_time_on_stage_and_expiry(self):
        d = Director()
        act = offer(d, behaviour="bark", duration=10)
        d.tick(9.0)
        assert not act.over()
        d.tick(1.5)
        assert act.over()
        until = offer(Director(), behaviour="run", until_key="k")
        until.elapsed = acting.HOLD_MAX + 1
        assert until.over()

    def test_queue_is_bounded(self):
        d = Director()
        offer(d, behaviour="howl", priority=3)
        for i in range(30):
            offer(d, behaviour="bark", priority=0, now=float(i))
        assert len(d.queue) == Director.QUEUE_MAX


class TestSounds:
    def test_missing_sound_is_none_and_logged_once(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setattr(acting, "_sound_missing", set())
        with caplog.at_level("INFO", logger="wawi.pet.acting"):
            assert acting.sound_path("bark", tmp_path) is None
            assert acting.sound_path("bark", tmp_path) is None
        assert len([r for r in caplog.records if "bark" in r.getMessage()]) == 1

    def test_a_present_sound_is_found_and_names_are_sanitised(self, tmp_path):
        (tmp_path / "bark.wav").write_bytes(b"RIFF")
        assert acting.sound_path("Bark", tmp_path) == tmp_path / "bark.wav"
        assert acting.sound_path("../../bark", tmp_path) == tmp_path / "bark.wav"
        assert acting.sound_path(None) is None


class TestTyping:
    def test_sustained_typing(self):
        t = TypingDetector()
        for i in range(20):
            t.feed(float(i), 0.3)
        assert t.typing_for(19.0) == pytest.approx(19.3)
        t.feed(25.0, 6.0)  # a real pause
        assert t.typing_for(25.0) == 0.0

    def test_a_short_gap_between_words_is_still_typing(self):
        t = TypingDetector()
        t.feed(0.0, 0.1)
        t.feed(3.0, 2.5)  # thinking for a moment
        t.feed(4.0, 0.2)
        assert t.typing_for(4.0) == pytest.approx(4.1)


def trigger(**kw):
    kw.setdefault("lonely_after", 1800.0)
    kw.setdefault("rng", random.Random(1))
    return WatchTrigger(**kw)


class TestWatchTrigger:
    def test_not_while_she_has_company(self):
        w = trigger()
        assert not any(w.should_go(float(t), 600.0, 0.2) for t in range(60))

    def test_ignored_and_you_are_typing_she_comes(self):
        w = trigger()
        results = [w.should_go(float(t), 2000.0, 0.2) for t in range(10)]
        assert not results[0] and results[-1]

    def test_ignored_but_not_typing_she_stays(self):
        w = trigger()
        assert not any(w.should_go(float(t), 5000.0, 30.0 + t) for t in range(60))

    def test_she_leaves_after_a_while_or_when_you_stop(self):
        w = trigger(stay=(100.0, 100.0))
        w.started(0.0)
        assert not w.should_leave(50.0, 0.5)
        assert w.should_leave(101.0, 0.5)
        w2 = trigger(stay=(1000.0, 1000.0), bored_after=40.0)
        w2.started(0.0)
        assert w2.should_leave(60.0, 45.0)

    def test_cooldown_after_leaving(self):
        w = trigger(cooldown=600.0)
        w.started(0.0)
        w.left(10.0)
        assert not any(w.should_go(10.0 + t, 5000.0, 0.2) for t in range(0, 500, 5))
        assert any(w.should_go(620.0 + t, 5000.0, 0.2) for t in range(0, 30))

    def test_a_touch_sends_her_off(self):
        w = trigger()
        assert w.touched(0.0) is False
        w.started(0.0)
        assert w.touched(5.0) is True and not w.watching


def test_events_still_carry_data_for_walk_to():
    r = Reaction(behaviour="walk_to", source=Event("x", data={"x": 1, "y": 2}))
    assert plan_for(r, ALL_CLIPS).move == "walk_to"
