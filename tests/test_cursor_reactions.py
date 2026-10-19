"""Near-cursor reactions are based on approach, not hover or touch."""

from wawi import mascot
from wawi.cursor_reactions import CursorReactions
from wawi.mascot import MascotAnimator
from wawi.state import State


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def approach(detector, clock, speed=100.0, **context):
    detector.sample((300.0, 0.0), idle=True)
    clock.advance(0.1)
    return detector.sample((300.0 - speed * 0.1, 0.0), idle=True, **context)


def test_slow_daytime_approach_wags_but_hover_and_touch_do_not():
    clock = Clock()
    detector = CursorReactions(clock)
    assert approach(detector, clock, speed=420.0) == "wag"
    clock.advance(0.1)
    assert detector.sample((253.0, 0.0), idle=True) is None
    clock.advance(0.1)
    assert detector.sample((30.0, 0.0), idle=True) is None
    clock.advance(0.1)
    assert detector.sample((150.0, 0.0), idle=True) is None


def test_fast_approach_barks_only_once_per_two_minutes():
    clock = Clock()
    detector = CursorReactions(clock)
    assert approach(detector, clock, speed=1000.0) == "bark"
    detector.sample((400.0, 0.0), idle=True)
    clock.advance(1.0)
    assert approach(detector, clock, speed=1000.0) is None
    detector.sample((400.0, 0.0), idle=True)
    clock.advance(121.0)
    assert approach(detector, clock, speed=1000.0) == "bark"


def test_guarded_or_night_approach_growls_with_own_cooldown():
    clock = Clock()
    detector = CursorReactions(clock)
    assert approach(detector, clock, speed=1000.0, guarded=True) == "growl"
    detector.sample((400.0, 0.0), idle=True)
    clock.advance(1.0)
    assert approach(detector, clock, speed=1000.0, night=True) is None
    detector.sample((400.0, 0.0), idle=True)
    clock.advance(61.0)
    assert approach(detector, clock, speed=500.0, night=True) == "growl"


def test_out_of_sorts_wags_unless_rushed():
    """Thirsty or lonely is not grumpy: a plain approach wags; only being
    rushed (two fast approaches close together) earns the growl."""
    clock = Clock()
    detector = CursorReactions(clock)
    assert approach(detector, clock, speed=500.0, happy=False) == "wag"
    detector.sample((400.0, 0.0), idle=True)
    clock.advance(1.0)
    assert approach(detector, clock, speed=1000.0, happy=False) == "bark"
    detector.sample((400.0, 0.0), idle=True)
    clock.advance(1.0)
    assert approach(detector, clock, speed=1000.0, happy=False) == "growl"


def test_with_a_wake_cooldown_an_approach_wakes_her_for_real_now_and_then():
    clock = Clock()
    detector = CursorReactions(clock)
    assert approach(detector, clock, speed=500.0, asleep=True, wake_cooldown=240.0) == "wake"
    detector.sample((400.0, 0.0), idle=True, asleep=True)
    clock.advance(30.0)
    # Within the cooldown: back to the ear twitch, she is not a toy.
    assert approach(detector, clock, speed=500.0, asleep=True, wake_cooldown=240.0) == "ear_twitch"
    detector.sample((400.0, 0.0), idle=True, asleep=True)
    clock.advance(240.0)
    assert approach(detector, clock, speed=500.0, asleep=True, wake_cooldown=240.0) == "wake"


def test_animator_cursor_wake_gets_her_up_and_keeps_her_up():
    anim = MascotAnimator(available=mascot.FRAME_NAMES, sleep_after=60.0)
    anim.update(0.1, State.IDLE)
    anim.asleep = True
    anim.idle_for = 999.0
    anim.look_at((170.0, 0.0))
    assert anim.react_to_cursor("wake")
    assert not anim.asleep and anim.idle_for == 0.0
    assert anim.behaviour == "wake"
    frames = [s.frame for s in anim._oneshot.steps]
    assert "follow_right" in frames and "wag_a" in frames
    assert anim.wake_reason == "cursor"
    for _ in range(int(20 / 0.1)):
        anim.update(0.1, State.IDLE)
    assert not anim.asleep  # 20 s later: still up (naps after 60)


def test_bark_from_asleep_starts_the_awake_time_over():
    anim = MascotAnimator(available=mascot.FRAME_NAMES, sleep_after=60.0)
    anim.update(0.1, State.IDLE)
    anim.asleep = True
    anim.idle_for = 999.0
    assert anim.react_to_cursor("bark")
    assert not anim.asleep and anim.idle_for == 0.0


def test_sleeping_ear_twitch_then_repeated_rush_wakes():
    clock = Clock()
    detector = CursorReactions(clock)
    assert approach(detector, clock, speed=1000.0, asleep=True) == "ear_twitch"
    detector.sample((400.0, 0.0), idle=True, asleep=True)
    clock.advance(1.0)
    assert approach(detector, clock, speed=1000.0, asleep=True) == "bark"


def test_stationary_near_cursor_wags_slowly_and_leaving_resets():
    clock = Clock()
    detector = CursorReactions(clock)
    assert detector.sample((150.0, 0.0), idle=True) is None
    clock.advance(0.1)
    assert detector.sample((150.0, 0.0), idle=True) is None
    reactions = []
    for _ in range(17):
        clock.advance(0.1)
        reactions.append(detector.sample((150.0, 0.0), idle=True))
    assert reactions.count("wag") == 1
    clock.advance(0.1)
    assert detector.sample((400.0, 0.0), idle=True) is None
    clock.advance(5.0)
    assert detector.sample((400.0, 0.0), idle=True) is None


def test_non_idle_never_reacts_and_resets_approach():
    clock = Clock()
    detector = CursorReactions(clock)
    detector.sample((300.0, 0.0), idle=True)
    clock.advance(0.1)
    assert detector.sample((200.0, 0.0), idle=False) is None
    clock.advance(0.1)
    assert detector.sample((200.0, 0.0), idle=True) is None


def test_moving_away_or_skipping_directly_onto_her_is_not_an_approach():
    clock = Clock()
    detector = CursorReactions(clock)
    detector.sample((200.0, 0.0), idle=True)
    clock.advance(0.1)
    assert detector.sample((240.0, 0.0), idle=True) is None
    detector.sample((300.0, 0.0), idle=True)
    clock.advance(0.1)
    assert detector.sample((40.0, 0.0), idle=True) is None


def test_animator_cursor_clip_preserves_sleep_and_respects_busy_state():
    frames = {"idle_a", "wag_a", "wag_b", "tail_wag_c", "follow_left", "sleeping",
              "ear_twitch", "growl", "bark_alert", "bark_a"}
    anim = MascotAnimator(available=frames)
    anim.look_at((-170.0, 0.0))
    assert anim.react_to_cursor("wag")
    assert anim.behaviour == "cursor_wag"
    assert anim._oneshot.steps[0].frame == "follow_left"
    anim.update(2.0, State.IDLE)
    anim.asleep = True
    assert anim.react_to_cursor("ear_twitch")
    assert anim.asleep
    anim.update(0.1, State.LISTENING)
    assert not anim.react_to_cursor("bark")
