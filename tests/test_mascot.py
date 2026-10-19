"""Wawi, the mascot: assets, animation parameters, placement and config.

Everything here is pure logic — no window is created. The panel's window and
the offline previews draw exactly the Pose these functions return.
"""

from __future__ import annotations

import logging
import textwrap

import pytest

from wawi import config as config_mod
from wawi import mascot
from wawi.mascot import (
    REST,
    MascotAnimator,
    Pose,
    alone_default_origin,
    alone_origin,
    badge,
    mascot_origin,
    mascot_pose,
    mood,
)
from wawi.state import State

KEYS = ("idle", "listening", "thinking", "speaking")
FRAMES = [i / 30.0 for i in range(0, 30 * 10)]


def poses(key, level=0.0):
    return [mascot_pose(key, t, level) for t in FRAMES]


class TestAssets:
    def test_bundled_assets_ship_with_the_package(self):
        for name in (mascot.SPRITE, mascot.MENUBAR_ICON, mascot.FACE_ICON, mascot.APP_ICON):
            path = mascot.asset_path(name)
            assert path is not None, name
            assert path.parent == mascot.ASSET_DIR

    def test_the_cutout_is_transparent_rgba(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.open(mascot.asset_path(mascot.SPRITE))
        assert img.mode == "RGBA"
        alpha = img.getchannel("A")
        # Corners are background; the middle of her body is not.
        assert alpha.getpixel((0, img.height - 1)) == 0
        assert alpha.getpixel((img.width // 2, img.height // 2)) == 255

    def test_the_original_drawing_is_still_there(self):
        assert (mascot.ASSET_DIR / "wawi_dibujo.jpg").is_file()

    def test_a_missing_asset_is_none_and_logged_once(self, tmp_path, caplog, monkeypatch):
        monkeypatch.setattr(mascot, "_missing_logged", set())
        with caplog.at_level(logging.WARNING, logger="wawi.mascot"):
            assert mascot.asset_path("nope.png", tmp_path) is None
            assert mascot.asset_path("nope.png", tmp_path) is None
        assert len([r for r in caplog.records if "nope.png" in r.getMessage()]) == 1

    def test_a_panel_without_assets_still_works(self, monkeypatch):
        # No AppKit window is built here; the loader must just say "none".
        from wawi.panel import PanelObserver

        monkeypatch.setattr(mascot, "asset_path", lambda *_a, **_k: None)
        obs = PanelObserver(True)
        assert obs._load_image(mascot.SPRITE) is None
        obs._tick_mascot()  # no window: harmless


class TestMood:
    def test_muting_never_changes_the_animation_key(self):
        # Muted = she does not hear. It is NOT a pose: she keeps living.
        for state in State:
            assert mood(state, True) == mood(state, False)

    def test_each_state_has_its_own_key(self):
        assert {mood(s, False) for s in State} == {
            "idle", "listening", "thinking", "speaking"
        }


class TestPoses:
    @pytest.mark.parametrize("key", KEYS)
    def test_motion_stays_within_her_window(self, key):
        # Clear enough to read on a 124-pt sprite, never so big that she is
        # clipped by her own window or turns into a distraction.
        for p in poses(key, level=1.0):
            assert abs(p.dx) <= 3.0
            assert -0.5 <= p.dy <= 9.0
            assert abs(p.rot) <= 9.0
            assert 0.85 <= p.sx <= 1.1 and 0.85 <= p.sy <= 1.12
            assert 0.5 <= p.alpha <= 1.0

    @pytest.mark.parametrize("key", KEYS)
    def test_the_window_has_room_for_every_pose(self, key):
        """The tilted, lifted, stretched sprite stays inside the padding."""
        import math

        w = mascot.SPRITE_WIDTH
        h = w * 552 / 887  # the drawing's aspect
        for p in poses(key, level=1.0):
            r = math.radians(abs(p.rot))
            half = w * p.sx / 2.0
            top = h * p.sy
            reach_x = half * math.cos(r) + top * math.sin(r) + abs(p.dx)
            reach_up = p.dy + top * math.cos(r) + half * math.sin(r)
            assert reach_x <= w / 2.0 + mascot.PAD_X
            assert reach_up <= h + mascot.PAD_TOP

    @pytest.mark.parametrize("key", ("idle", "listening", "thinking", "speaking"))
    def test_she_is_never_dimmed_unless_muted(self, key):
        assert all(p.alpha == 1.0 for p in poses(key))

    def test_idle_breathes_and_is_calm(self):
        ps = poses("idle")
        assert max(p.sy for p in ps) > 1.005  # breathing is visible
        assert all(p.rot == 0.0 for p in ps)  # no head movement at rest
        # The settle bob is occasional: most frames are on the ground.
        grounded = sum(1 for p in ps if p.dy < 0.05)
        assert grounded > 0.8 * len(ps)

    def test_listening_is_head_up_lifted_and_held(self):
        ps = poses("listening")
        assert min(p.dy for p in ps) > 4.0
        assert min(p.sy for p in ps) > 1.04  # stretched tall: ears up
        # Head raised the whole time (negative = head, on the left, lifts).
        assert all(p.rot < -4.0 for p in ps)
        # HELD: barely any vertical motion, unlike talking.
        assert max(p.dy for p in ps) - min(p.dy for p in ps) < 1.0

    def test_listening_follows_the_microphone(self):
        quiet = mascot_pose("listening", 0.0, level=0.0)
        loud = mascot_pose("listening", 0.0, level=1.0)
        assert loud.dy > quiet.dy and loud.sy > quiet.sy

    def test_thinking_tilts_both_ways(self):
        rots = [p.rot for p in poses("thinking")]
        assert min(rots) < -4.0 and max(rots) > 4.0

    def test_speaking_bounces_and_sways(self):
        ps = poses("speaking")
        assert max(p.dy for p in ps) > 5.0
        assert min(p.dy for p in ps) < 0.5  # she lands between syllables
        rots = [p.rot for p in ps]
        assert min(rots) < -3.0 and max(rots) > 3.0

    def test_speaking_has_a_talking_rhythm_without_a_voice(self):
        # About three bobs a second, counted as upward crossings a quarter of the
        # way up (the phrase swell makes some syllables smaller than others).
        dys = [p.dy for p in poses("speaking")]
        mid = min(dys) + 0.25 * (max(dys) - min(dys))
        ups = sum(1 for a, b in zip(dys, dys[1:]) if a < mid <= b)
        assert 25 <= ups <= 35  # over the 10 s sampled

    def test_speaking_follows_her_voice(self):
        quiet = mascot_pose("speaking", 0.3, voice=0.0)
        loud = mascot_pose("speaking", 0.3, voice=1.0)
        assert loud.dy - quiet.dy > 5.0
        assert loud.sy > 1.03 > 0.99 > quiet.sy  # stretch on the syllable

    def test_listening_and_speaking_are_obviously_different(self):
        """The two states that matter most must not be confused at a glance."""
        listen = poses("listening", level=0.3)
        speak = poses("speaking")

        def spread(ps, attr):
            values = [getattr(p, attr) for p in ps]
            return max(values) - min(values)

        # Listening is still; speaking moves a lot.
        assert spread(speak, "dy") > 5 * spread(listen, "dy")
        # Listening holds the head up; speaking sways through upright.
        assert max(p.rot for p in listen) < -4.0
        assert min(p.rot for p in speak) < 0 < max(p.rot for p in speak)

    def test_thinking_stays_on_the_ground(self):
        assert all(abs(p.dy) < 0.01 for p in poses("thinking"))

    def test_no_state_is_dimmed(self):
        for key in KEYS:
            assert all(p.alpha == 1.0 for p in poses(key))

    @pytest.mark.parametrize("key", KEYS)
    def test_reduce_motion_holds_still(self, key):
        still = {mascot_pose(key, t, 1.0, reduce_motion=True) for t in FRAMES[::7]}
        assert still == {REST}


class TestAnimator:
    def test_state_changes_crossfade_instead_of_snapping(self):
        anim = MascotAnimator()
        for _ in range(30):
            anim.update(1 / 30, State.IDLE)
        before = anim.update(1 / 30, State.IDLE)
        first = anim.update(1 / 30, State.LISTENING)
        target = mascot_pose("listening", anim.t, anim.level)
        # One frame in, she is still close to where she was.
        assert abs(first.dy - before.dy) < abs(target.dy - before.dy)
        for _ in range(int(MascotAnimator.FADE * 30) + 2):
            last = anim.update(1 / 30, State.LISTENING)
        assert last == mascot_pose("listening", anim.t, anim.level)

    def test_level_is_smoothed(self):
        anim = MascotAnimator()
        anim.update(1 / 30, State.LISTENING, level=1.0)
        assert 0.0 < anim.level < 1.0

    def test_muted_keeps_the_state_pose_and_only_sets_the_flag(self):
        anim = MascotAnimator()
        for _ in range(30):
            p = anim.update(1 / 30, State.SPEAKING, muted=True)
        assert anim.key == "speaking" and anim.muted and p.alpha == 1.0
        anim.update(1 / 30, State.SPEAKING, muted=False)
        assert not anim.muted

    def test_voice_is_smoothed_but_quick(self):
        anim = MascotAnimator()
        anim.update(1 / 30, State.SPEAKING, voice=0.0)
        anim.update(1 / 30, State.SPEAKING, voice=1.0)
        assert 0.5 < anim.voice < 1.0  # most of a syllable in one frame

    def test_no_voice_falls_back_to_the_rhythm(self):
        anim = MascotAnimator()
        anim.update(1 / 30, State.SPEAKING, voice=0.8)
        anim.update(1 / 30, State.SPEAKING, voice=None)
        assert anim.voice is None

    def test_reduce_motion_is_rest(self):
        anim = MascotAnimator()
        assert anim.update(1 / 30, State.SPEAKING, reduce_motion=True) == REST


class TestPlacement:
    SCREEN = (0.0, 0.0, 1920.0, 1080.0)

    def test_she_sits_outside_the_bottom_left_corner(self):
        panel = (1000.0, 700.0, 460.0, 200.0)
        x, y = mascot_origin(panel, (144.0, 96.0), self.SCREEN)
        assert x < 1000.0 < x + 144.0  # tucked in under the corner
        assert x + 144.0 - 1000.0 == pytest.approx(mascot.OVERLAP)
        assert y == pytest.approx(700.0 - mascot.PAD_BOTTOM)

    def test_against_the_left_edge_she_moves_to_the_right_corner(self):
        panel = (10.0, 700.0, 460.0, 200.0)
        x, _ = mascot_origin(panel, (144.0, 96.0), self.SCREEN)
        assert x == pytest.approx(10.0 + 460.0 - mascot.OVERLAP)

    def test_she_follows_the_panel(self):
        a = mascot_origin((1000.0, 700.0, 460.0, 200.0), (144.0, 96.0), self.SCREEN)
        b = mascot_origin((900.0, 650.0, 460.0, 120.0), (144.0, 96.0), self.SCREEN)
        assert (a[0] - b[0], a[1] - b[1]) == pytest.approx((100.0, 50.0))

    def test_a_drag_offset_is_kept_relative_to_the_panel(self):
        x0, y0 = mascot_origin((1000.0, 700.0, 460.0, 200.0), (144.0, 96.0))
        x1, y1 = mascot_origin((1000.0, 700.0, 460.0, 200.0), (144.0, 96.0),
                               offset=(-30.0, 12.0))
        assert (x1 - x0, y1 - y0) == pytest.approx((-30.0, 12.0))

    def test_status_dot_only_for_listening_and_muted(self):
        assert badge("listening") == "listening"
        assert {badge(k) for k in ("idle", "thinking", "speaking")} == {None}
        # The mute mark holds over anything she is doing.
        assert {badge(k, muted=True) for k in KEYS} == {"muted"}

    def test_her_own_work_shows_dots_unless_something_says_more(self):
        assert badge("idle", musing=True) == "musing"
        assert badge("listening", musing=True) == "listening"
        assert badge("idle", muted=True, musing=True) == "muted"
        # Already showing she is thinking or talking: no second sign.
        assert {badge(k, musing=True) for k in ("thinking", "speaking")} == {None}

    def test_musing_dots_hop_one_after_another(self):
        from wawi.mascot import MUSING_PERIOD_S, musing_dots

        peaks = []
        for i in range(3):
            ts = [MUSING_PERIOD_S * k / 120 for k in range(120)]
            peaks.append(max(ts, key=lambda t: musing_dots(t)[i]))
        assert peaks == sorted(peaks) and len(set(peaks)) == 3
        assert all(0.0 <= h <= 1.0 for t in (0.0, 0.3, 0.9) for h in musing_dots(t))

    def test_pose_lerp_endpoints(self):
        a, b = Pose(dy=0.0), Pose(dy=4.0, alpha=0.5)
        assert a.lerp(b, 0.0) == a and a.lerp(b, 1.0) == b
        assert a.lerp(b, 0.5).dy == pytest.approx(2.0)


class TestPanelHooks:
    def test_menu_title_uses_the_assistant_name(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.title_text = "Wawi"
        assert obs._mascot_menu_title() == "Ocultar a Wawi"
        obs.toggle_mascot()
        assert obs.show_mascot is False
        assert obs._mascot_menu_title() == "Mostrar a Wawi"
        obs.title_text = "wawi"
        assert obs._mascot_menu_title() == "Mostrar a Wawi"


class TestConfig:
    def _load(self, tmp_path, body):
        path = tmp_path / "config.yaml"
        path.write_text(textwrap.dedent(body))
        return config_mod.load(path)

    def test_mascot_defaults_to_on(self, tmp_path):
        assert self._load(tmp_path, "language: es\n").show_mascot is True

    def test_mascot_can_be_turned_off(self, tmp_path):
        cfg = self._load(tmp_path, "ui:\n  mascot: false\n")
        assert cfg.show_mascot is False
        assert cfg.show_panel is True

    def test_the_example_config_documents_it(self):
        from pathlib import Path

        example = Path(__file__).resolve().parent.parent / "config.yaml.example"
        text = example.read_text()
        ui = text[text.index("\nui:"):]
        assert "mascot:" in ui.split("\n\n")[0]


class TestAlonePlacement:
    """Wawi without the panel: her own spot, always on a visible screen."""

    MAIN = (0.0, 0.0, 1440.0, 875.0)      # laptop, visible frame
    SECOND = (1440.0, -200.0, 2560.0, 1415.0)  # external monitor to the right
    SIZE = (156.0, 124.0)

    def test_default_is_the_bottom_right_corner_with_a_margin(self):
        x, y = alone_default_origin(self.SIZE, self.MAIN)
        assert x + self.SIZE[0] == pytest.approx(1440.0 - mascot.ALONE_MARGIN)
        assert y == pytest.approx(mascot.ALONE_MARGIN)

    def test_never_placed_goes_to_the_main_screen_corner(self):
        assert alone_origin(None, self.SIZE, [self.SECOND, self.MAIN], self.MAIN) == \
            alone_default_origin(self.SIZE, self.MAIN)

    def test_a_saved_spot_is_kept(self):
        spot = (300.0, 400.0)
        assert alone_origin(spot, self.SIZE, [self.MAIN, self.SECOND]) == spot

    def test_a_spot_on_the_second_monitor_is_kept_there(self):
        spot = (3000.0, 600.0)
        assert alone_origin(spot, self.SIZE, [self.MAIN, self.SECOND]) == spot

    def test_hanging_off_an_edge_is_nudged_fully_on_screen(self):
        x, y = alone_origin((1300.0, -30.0), self.SIZE, [self.MAIN])
        assert x == pytest.approx(1440.0 - self.SIZE[0])
        assert y == pytest.approx(0.0)

    def test_an_unplugged_monitor_sends_her_home(self):
        # Saved on the external monitor, which is gone now.
        spot = (3000.0, 600.0)
        assert alone_origin(spot, self.SIZE, [self.MAIN], self.MAIN) == \
            alone_default_origin(self.SIZE, self.MAIN)

    def test_no_screen_at_all_is_none(self):
        assert alone_origin((10.0, 10.0), self.SIZE, []) is None


class TestAloneHooks:
    """The panel-side switches, exercised without any window."""

    def test_the_spot_round_trips_and_defaults_to_none(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        assert obs.get_mascot_origin() is None
        obs.set_mascot_origin(120, 340.5)
        assert obs.get_mascot_origin() == (120.0, 340.5)

    def test_she_is_wanted_with_the_panel_hidden(self):
        """Hiding the panel must no longer hide her: no window, no crash."""
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.toggle_panel()
        assert obs.panel_visible is False and obs.show_mascot is True
        obs._tick_mascot()  # no window yet: harmless

    def test_voice_level_only_while_speaking(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.set_voice_level_source(lambda: 0.7)
        obs._current_state = State.LISTENING
        assert obs._voice_level() is None
        obs._current_state = State.SPEAKING
        assert obs._voice_level() == pytest.approx(0.7)

    def test_a_broken_voice_source_is_none(self):
        from wawi.panel import PanelObserver

        def boom():
            raise RuntimeError("audio gone")

        obs = PanelObserver(True)
        obs.set_voice_level_source(boom)
        obs._current_state = State.SPEAKING
        assert obs._voice_level() is None


class TestMascotOnlyConfig:
    def _load(self, tmp_path, body):
        path = tmp_path / "config.yaml"
        path.write_text(textwrap.dedent(body))
        return config_mod.load(path)

    def test_defaults_to_off(self, tmp_path):
        assert self._load(tmp_path, "language: es\n").mascot_only is False

    def test_turning_it_on_implies_the_mascot(self, tmp_path):
        cfg = self._load(tmp_path, "ui:\n  mascot: false\n  mascot_only: true\n")
        assert cfg.mascot_only is True and cfg.show_mascot is True

    def test_the_example_config_documents_it(self):
        from pathlib import Path

        example = Path(__file__).resolve().parent.parent / "config.yaml.example"
        text = example.read_text()
        ui = text[text.index("\nui:"):]
        assert "mascot_only:" in ui.split("\n\n")[0]


# ---------------------------------------------------------------------------
# Frames: clips, the behaviour scheduler, sleep, lip-sync, fallbacks
# ---------------------------------------------------------------------------

import random  # noqa: E402

from wawi.mascot import (  # noqa: E402
    BEHAVIOUR_WEIGHTS,
    CLIPS,
    FRAME_NAMES,
    SLEEP_AFTER,
    Clip,
    Step,
    crossfade_alphas,
    framed_pose,
    motion_overlay,
    resolve_frame,
)

ALL = frozenset(FRAME_NAMES)
DT = 1 / 30


def run(anim, seconds, state=State.IDLE, **kw):
    """Tick *anim* for *seconds*; returns [(pose, frame, behaviour), …]."""
    out = []
    for _ in range(int(round(seconds / DT))):
        pose = anim.update(DT, state, **kw)
        out.append((pose, anim.frame, anim.behaviour))
    return out


def framed(seed=1, frames=ALL, sleep_after=SLEEP_AFTER):
    return MascotAnimator(available=frames, rng=random.Random(seed),
                          sleep_after=sleep_after)


class TestClips:
    def test_a_looping_clip_cycles_through_its_steps(self):
        clip = Clip("x", (Step("a", 0.5), Step("b", 0.25)), loop=True)
        assert clip.duration == pytest.approx(0.75)
        assert [clip.frame_at(t) for t in (0.0, 0.49, 0.5, 0.74, 0.75, 1.3)] == \
            ["a", "a", "b", "b", "a", "b"]
        assert not clip.done(100.0)

    def test_a_one_shot_holds_its_last_frame_and_ends(self):
        clip = Clip("x", (Step("a", 0.2), Step("b", 0.3)))
        assert clip.frame_at(0.1) == "a" and clip.frame_at(0.35) == "b"
        assert clip.frame_at(9.0) == "b"
        assert not clip.done(0.49) and clip.done(0.5)

    def test_missing_frames_are_dropped_from_a_clip(self):
        clip = CLIPS["party"].playable({"idle_a", "happy"})  # no happy_b, no jump
        assert clip is not None
        assert {s.frame for s in clip.steps} == {"happy"}

    def test_a_clip_with_nothing_of_its_own_is_not_playable(self):
        # Sniff without the sniff frame would just be idle_a: not worth it.
        assert CLIPS["sniff"].playable({"idle_a", "idle_b"}) is None
        assert CLIPS["scratch"].playable({"idle_a", "scratch_a"}) is not None

    def test_every_clip_uses_only_known_frames(self):
        for clip in CLIPS.values():
            for step in clip.steps:
                assert step.frame in ALL, (clip.name, step.frame)
                assert step.duration > 0

    def test_behaviours_are_short(self):
        # An idle behaviour is a glance, not a performance.
        for name in BEHAVIOUR_WEIGHTS:
            assert 0.5 <= CLIPS[name].duration <= 3.0, name


class TestFrameFallbacks:
    def test_a_present_frame_is_itself(self):
        assert resolve_frame("listening_b", ALL) == "listening_b"

    def test_missing_frames_walk_to_their_relatives(self):
        have = {"idle_a", "listening", "speaking_a", "speaking_b"}
        assert resolve_frame("listening_b", have) == "listening"
        assert resolve_frame("speaking_c", have) == "speaking_b"
        assert resolve_frame("thinking_b", have) == "idle_a"
        assert resolve_frame("roll", have) == "idle_a"

    def test_without_the_base_frame_there_are_no_frames(self):
        assert resolve_frame("idle_a", {"listening"}) is None
        assert resolve_frame("listening_b", {"listening"}) == "listening"

    def test_without_frames_the_animator_is_the_old_one(self):
        anim = MascotAnimator()
        run(anim, 30.0)
        assert anim.frame is None and anim.behaviour is None
        assert anim.update(DT, State.IDLE) == mascot_pose("idle", anim.t, anim.level)

    def test_a_set_without_the_base_frame_is_ignored(self):
        anim = MascotAnimator(available={"listening", "speaking_a"})
        assert not anim.framed
        run(anim, 1.0, State.LISTENING)
        assert anim.frame is None

    def test_only_the_original_eight_frames_still_animate(self):
        eight = {"idle_a", "idle_b", "listening", "speaking_a", "speaking_b",
                 "thinking", "sleeping", "happy"}
        anim = framed(frames=eight)
        assert anim.playable() == []  # no behaviour has its frames yet
        seen = {f for _, f, _ in run(anim, 60.0)}
        assert seen <= eight and {"idle_a", "idle_b"} <= seen
        anim.celebrate()
        out = run(anim, 2.0)
        assert "happy" in {f for _, f, _ in out}  # cheer, minus happy_b
        for state in (State.LISTENING, State.SPEAKING, State.THINKING):
            assert all(f in eight for _, f, _ in run(anim, 2.0, state, level=1.0))

    def test_available_frames_reads_the_folder_and_logs_once(self, tmp_path, caplog,
                                                            monkeypatch):
        monkeypatch.setattr(mascot, "_missing_logged", set())
        frames = tmp_path / mascot.FRAME_DIR
        frames.mkdir()
        for name in ("idle_a", "listening"):
            (frames / f"{name}.png").write_bytes(b"png")
        with caplog.at_level(logging.INFO, logger="wawi.mascot"):
            assert mascot.available_frames(tmp_path) == {"idle_a", "listening"}
            mascot.available_frames(tmp_path)
        assert len([r for r in caplog.records if "not drawn yet" in r.getMessage()]) == 1

    def test_no_base_frame_on_disk_means_no_frames(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mascot, "_missing_logged", set())
        (tmp_path / mascot.FRAME_DIR).mkdir()
        (tmp_path / mascot.FRAME_DIR / "listening.png").write_bytes(b"png")
        assert mascot.available_frames(tmp_path) == frozenset()

    def test_the_shipped_frames_are_aligned(self):
        """Same size for every shared-box frame, or swapping them would make
        her jump; every loose frame present has an anchor instead."""
        PIL = pytest.importorskip("PIL.Image")
        have = mascot.available_frames()
        assert mascot.BASE_FRAME in have
        shared = [n for n in have if n not in mascot.LOOSE_FRAMES]
        sizes = {PIL.open(mascot.frame_path(n)).size for n in shared}
        assert len(sizes) == 1
        (w, h), = sizes
        assert 200 <= w <= 400  # ~2.5x the on-screen width, not the 1024 source
        anchors = mascot.load_anchors()
        loose = [n for n in have if n in mascot.LOOSE_FRAMES]
        if loose:
            assert anchors["_box"] == (0.0, 0.0, float(w), float(h))
            for n in loose:
                assert n in anchors, n
                assert PIL.open(mascot.frame_path(n)).size == pytest.approx(
                    anchors[n][2:], abs=1.5)


class TestIdleLife:
    def test_she_breathes_and_blinks(self):
        anim = framed(sleep_after=1e9)  # awake idle, not the doze-off
        out = run(anim, 20.0)
        frames = [f for _, f, _ in out]
        assert {"idle_a", "idle_b"} <= set(frames)
        blinks = sum(1 for a, b in zip(frames, frames[1:]) if b == "blink" != a)
        # Every 2–6 s, minus the time spent in behaviours.
        assert 2 <= blinks <= 10

    def test_behaviours_come_every_6_to_20_seconds(self):
        anim = framed(seed=3, sleep_after=1e9)  # awake idle, not the doze-off
        out = run(anim, 100.0)
        starts = [i * DT for i, (_, _, b) in enumerate(out)
                  if b and b != "blink" and (i == 0 or out[i - 1][2] != b)]
        assert len(starts) >= 4
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        # Gap measured start to start: the wait plus the previous behaviour.
        assert all(6.0 <= g <= 20.0 + 3.5 for g in gaps), gaps

    @pytest.mark.parametrize("seed", range(5))
    def test_never_the_same_behaviour_twice_in_a_row(self, seed):
        anim = framed(seed=seed)
        picks = []
        for _ in range(200):
            name = anim._pick_behaviour()
            anim._last_behaviour = name
            picks.append(name)
        assert all(a != b for a, b in zip(picks, picks[1:]))
        # Behaviours, and the idle activities she has the art for.
        assert set(picks) <= set(BEHAVIOUR_WEIGHTS) | set(mascot.ACTIVITIES)
        assert len(set(picks)) >= 7  # variety, not two favourites

    def test_roll_is_rare_and_glances_are_common(self):
        anim = framed(seed=7)
        picks = []
        for _ in range(2000):
            name = anim._pick_behaviour()
            anim._last_behaviour = name
            picks.append(name)
        assert picks.count("roll") < picks.count("look_left") / 3

    def test_the_scheduler_is_reproducible_with_a_seed(self):
        a = [b for _, _, b in run(framed(seed=11), 60.0)]
        b = [b for _, _, b in run(framed(seed=11), 60.0)]
        assert a == b

    def test_a_behaviour_plays_its_frames(self):
        anim = framed()
        assert anim._start("scratch")
        frames = [f for _, f, _ in run(anim, CLIPS["scratch"].duration - DT)]
        assert frames.count("scratch_a") > 0 and frames.count("scratch_b") > 0
        assert anim.behaviour == "scratch"
        run(anim, 0.2)
        assert anim.behaviour != "scratch"

    def test_a_jump_leaves_the_ground(self):
        anim = framed()
        anim.celebrate(big=True)
        out = run(anim, CLIPS["party"].duration)
        jumping = [p.dy for p, f, _ in out if f == "jump"]
        assert jumping and max(jumping) > 5.0


class TestInterruptions:
    def test_a_state_change_stops_a_behaviour_at_once(self):
        anim = framed()
        run(anim, 0.5)
        anim._start("roll")
        run(anim, 0.3)
        assert anim.frame == "roll"
        anim.update(DT, State.LISTENING)
        assert anim.behaviour is None
        assert anim.frame == "listening"  # the very next frame

    def test_no_behaviours_outside_idle(self):
        anim = framed()
        for state in (State.LISTENING, State.THINKING, State.SPEAKING):
            out = run(anim, 40.0, state)
            assert all(b is None for _, _, b in out)

    def test_muted_she_keeps_living(self):
        # Muting the microphone is not turning her off: behaviours, normal
        # opacity, and her usual nap once nobody is around.
        anim = framed(sleep_after=60.0)
        out = run(anim, 50.0, State.IDLE, muted=True)
        assert any(b is not None for _, _, b in out)
        assert all(p.alpha == 1.0 for p, _, _ in out)
        assert not anim.asleep
        run(anim, 30.0, State.IDLE, muted=True)
        assert anim.asleep

    def test_toggling_the_mute_does_not_interrupt_her(self):
        anim = framed()
        run(anim, 0.5)
        anim._start("roll")
        anim.update(DT, State.IDLE, muted=True)
        assert anim.behaviour == "roll"
        anim.update(DT, State.IDLE, muted=False)
        assert anim.behaviour == "roll"

    def test_muted_she_sometimes_folds_her_ears(self):
        anim = framed(frames=ALL | {"ears_down"})
        anim.muted = True
        picks = set()
        for _ in range(600):
            picks.add(anim._pick_behaviour())
            anim._last_behaviour = None
        assert "ears_down" in picks
        anim.muted = False
        assert all(anim._pick_behaviour() != "ears_down" for _ in range(600))

    def test_without_the_ears_down_art_she_just_never_does_it(self):
        anim = framed(frames=ALL - {"ears_down"})
        anim.muted = True
        assert all(anim._pick_behaviour() != "ears_down" for _ in range(600))


class TestSleep:
    def test_she_falls_asleep_after_a_long_idle(self):
        anim = framed()
        run(anim, SLEEP_AFTER - 5.0)
        assert not anim.asleep
        run(anim, 10.0)
        assert anim.asleep
        frames = {f for _, f, _ in run(anim, 10.0)}
        assert frames <= {"sleeping", "sleeping_b"}

    def test_she_yawns_before_lying_down(self):
        anim = framed()
        out = run(anim, SLEEP_AFTER + 3.0)
        lying = [b for _, _, b in out if b == "lie_down"]
        assert lying

    def test_asleep_she_does_nothing_else(self):
        anim = framed()
        run(anim, SLEEP_AFTER + 5.0)
        out = run(anim, 60.0)
        assert all(b is None for _, _, b in out)
        assert all(p.alpha == 1.0 for p, _, _ in out)  # asleep, not muted

    def test_a_state_change_wakes_her_instantly(self):
        anim = framed()
        run(anim, SLEEP_AFTER + 5.0)
        anim.update(DT, State.LISTENING)
        assert not anim.asleep and anim.frame == "listening"
        # Back to idle she is awake, and the sleep timer starts over.
        run(anim, 1.0, State.IDLE)
        assert not anim.asleep and anim.idle_for < 2.0

    def test_a_click_wakes_her_with_a_stretch(self):
        anim = framed()
        run(anim, SLEEP_AFTER + 5.0)
        assert anim.poke()
        assert not anim.asleep and anim.behaviour == "wake"

    def test_no_sleeping_frame_no_sleep(self):
        anim = framed(frames=ALL - {"sleeping", "sleeping_b"})
        run(anim, SLEEP_AFTER + 10.0)
        assert not anim.asleep


class TestListeningAndSpeaking:
    def test_louder_means_ears_up(self):
        anim = framed()
        anim.MEGAPHONE_CHANCE = 0.0  # plain listening this turn
        run(anim, 0.5, State.LISTENING, level=0.0)
        quiet = anim.frame
        # Speaking up (a shout is the volume frames, tested below).
        run(anim, 0.5, State.LISTENING, level=0.6)
        assert quiet == "listening" and anim.frame == "listening_b"
        run(anim, 1.5, State.LISTENING, level=0.0)
        assert anim.frame == "listening"

    @pytest.mark.parametrize("voice, frame", [
        (0.0, "speaking_a"), (0.3, "speaking_c"), (0.9, "speaking_b"),
    ])
    def test_the_mouth_follows_her_voice(self, voice, frame):
        anim = framed()
        run(anim, 0.5, State.SPEAKING, voice=voice)
        assert anim.frame == frame

    def test_the_mouth_moves_on_the_rhythm_without_a_voice(self):
        anim = framed()
        frames = [f for _, f, _ in run(anim, 3.0, State.SPEAKING)]
        assert {"speaking_a", "speaking_b"} <= set(frames)
        changes = sum(1 for a, b in zip(frames, frames[1:]) if a != b)
        assert changes >= 10  # chatter, ~3 syllables a second

    def test_a_missing_half_open_mouth_borrows_the_open_one(self):
        anim = framed(frames=ALL - {"speaking_c"})
        run(anim, 0.5, State.SPEAKING, voice=0.3)
        assert anim.frame == "speaking_b"

    def test_thinking_looks_both_ways(self):
        frames = {f for _, f, _ in run(framed(), 6.0, State.THINKING)}
        assert frames == {"thinking", "thinking_b"}


class TestCelebration:
    def test_a_cheer_after_an_answer(self):
        anim = framed()
        run(anim, 1.0, State.SPEAKING)
        anim.celebrate()
        anim.update(DT, State.IDLE)
        assert anim.behaviour == "cheer"

    def test_a_party_waits_until_she_is_idle(self):
        anim = framed()
        anim.celebrate(big=True)
        out = run(anim, 2.0, State.SPEAKING)
        assert all(b is None for _, _, b in out)
        anim.celebrate()  # the answer ending does not downgrade it
        anim.update(DT, State.IDLE)
        assert anim.behaviour == "party"

    def test_listening_cancels_a_pending_celebration(self):
        anim = framed()
        anim.celebrate(big=True)
        run(anim, 0.5, State.LISTENING)
        run(anim, 0.5, State.IDLE)
        assert anim.behaviour != "party"


class TestPoke:
    def test_a_click_makes_her_react_without_repeating(self):
        anim = framed()
        seen = []
        for _ in range(6):
            assert anim.poke()
            seen.append(anim.behaviour)
            run(anim, 3.0)
        assert all(a != b for a, b in zip(seen, seen[1:]))

    def test_a_double_click_is_one_reaction(self):
        anim = framed()
        assert anim.poke()
        assert not anim.poke()

    def test_clicks_only_while_idle(self):
        anim = framed()
        run(anim, 0.2, State.LISTENING)
        assert not anim.poke()


class TestReduceMotion:
    def test_no_behaviours_no_blinks_no_bob(self):
        anim = framed()
        out = run(anim, SLEEP_AFTER + 30.0, reduce_motion=True)
        assert {f for _, f, _ in out} == {"idle_a"}
        assert all(b is None for _, _, b in out)
        assert all(p == REST for p, _, _ in out)
        assert not anim.asleep

    def test_state_frames_still_swap_without_a_crossfade(self):
        anim = framed()
        for state, frame in ((State.LISTENING, "listening"),
                             (State.SPEAKING, "speaking_c"),
                             (State.THINKING, "thinking")):
            anim.update(DT, state, reduce_motion=True, voice=0.9)
            assert anim.frame == frame
            assert anim.prev_frame is None and anim.frame_mix == 1.0
            out = run(anim, 2.0, state, reduce_motion=True, voice=0.9, level=1.0)
            assert {f for _, f, _ in out} == {frame}  # no lip-sync flicker
        anim.update(DT, State.IDLE, muted=True, reduce_motion=True)
        assert anim.frame == "idle_a"  # muted is a badge, not a pose

    def test_turning_it_on_stops_a_behaviour(self):
        anim = framed()
        anim._start("roll")
        anim.update(DT, State.IDLE, reduce_motion=True)
        assert anim.behaviour is None and anim.frame == "idle_a"

    def test_clicks_and_celebrations_are_ignored(self):
        anim = framed()
        anim.update(DT, State.IDLE, reduce_motion=True)
        assert not anim.poke()
        anim.celebrate(big=True)
        out = run(anim, 2.0, reduce_motion=True)
        assert all(b is None for _, _, b in out)


class TestFrameCrossfade:
    def test_consecutive_frames_dissolve_briefly(self):
        anim = framed()
        run(anim, 0.5)
        anim._start("tilt")
        anim.update(DT, State.IDLE)
        assert anim.frame == "tilt" and anim.prev_frame in ("idle_a", "idle_b")
        assert 0.0 <= anim.frame_mix < 1.0
        run(anim, 0.15)
        assert anim.prev_frame is None and anim.frame_mix == 1.0

    def test_the_dissolve_never_leaves_her_see_through(self):
        for i in range(11):
            prev, cur = crossfade_alphas(i / 10)
            # The new frame is drawn over the old one: combined opacity.
            assert 1 - (1 - prev) * (1 - cur) >= 0.999
        assert crossfade_alphas(0.0) == (1.0, 0.0)
        assert crossfade_alphas(1.0) == (0.0, 1.0)
        assert crossfade_alphas(0.5, 0.55) == pytest.approx((0.55, 0.55))


class TestFramedMotion:
    MOTIONS = ("breathe", "hop", "bounce", "wiggle", "wag", "stretch", "yawn", "tilt",
               "glance", "sniff", "wobble", "dig", "proud", "run_out", "run_back",
               "lean", "roll_eyes", "chin_up", "shrug", "fiddle", "spin", "recoil",
               "sneeze", "huff")

    def test_every_clip_motion_is_covered_here(self):
        used = {s.motion for c in CLIPS.values() for s in c.steps} - {None}
        assert used <= set(self.MOTIONS)

    def _all_poses(self):
        for key in ("idle", "listening", "thinking", "speaking", "asleep"):
            yield from (framed_pose(key, t, 1.0, voice=1.0) for t in FRAMES[::3])
            yield from (framed_pose(key, t, 1.0) for t in FRAMES[::3])
        # Behaviours (and their overlay motion) only ever play while idle.
        base = [framed_pose("idle", t) for t in FRAMES[::15]]
        for m in self.MOTIONS:
            for i in range(21):
                o = motion_overlay(m, i * 0.05, 1.0)
                for p in base:
                    yield mascot._apply(p, o)

    def test_motion_stays_small(self):
        for p in self._all_poses():
            assert abs(p.dx) <= 3.0
            assert -0.5 <= p.dy <= 9.0
            assert abs(p.rot) <= 9.0
            assert 0.9 <= p.sx <= 1.1 and 0.9 <= p.sy <= 1.12

    def test_the_window_has_room_for_every_frame_and_motion(self):
        import math

        PIL = pytest.importorskip("PIL.Image")
        fw, fh = PIL.open(mascot.frame_path(mascot.BASE_FRAME)).size
        w = mascot.SPRITE_WIDTH
        h = w * fh / fw
        for p in self._all_poses():
            r = math.radians(abs(p.rot))
            half = w * p.sx / 2.0
            top = h * p.sy
            assert half * math.cos(r) + top * math.sin(r) + abs(p.dx) <= w / 2 + mascot.PAD_X
            assert p.dy + top * math.cos(r) + half * math.sin(r) <= h + mascot.PAD_TOP

    def test_framed_motion_is_gentler_than_the_single_drawing(self):
        for key in ("listening", "speaking", "thinking"):
            single = max(abs(mascot_pose(key, t, 1.0).rot) for t in FRAMES)
            frames = max(abs(framed_pose(key, t, 1.0).rot) for t in FRAMES)
            assert frames < single


class TestPanelMascotHooks:
    def test_an_answer_ending_queues_a_cheer(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs(State.SPEAKING, State.IDLE)
        assert obs._mascot_celebrate == "cheer"

    def test_a_task_party_is_not_downgraded(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.celebrate(big=True)
        obs(State.SPEAKING, State.IDLE)
        obs.celebrate()
        assert obs._mascot_celebrate == "party"

    def test_other_transitions_do_not_celebrate(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs(State.LISTENING, State.IDLE)
        obs(State.IDLE, State.LISTENING)
        assert obs._mascot_celebrate is None

    def test_a_click_without_a_window_is_harmless(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs._poke_mascot()


# ---------------------------------------------------------------------------
# Personality, listening extras, touch zones, petting, cursor follow
# ---------------------------------------------------------------------------

from wawi.mascot import (  # noqa: E402
    IGNORED_AFTER,
    IRONIC,
    LONG_ANSWER,
    RUN_DISTANCE,
    STILL_HOLD,
    WAKE_BEHAVIOURS,
    WAKE_GAP,
    PetDetector,
    clip_excursion,
    follow_frame,
    listening_frame,
    sprite_uv,
    too_many_touches,
    touch_zone,
)

NEW_BEHAVIOURS = ("dig", "run", "stare", "squint", "eyeroll", "smug", "side_eye",
                  "shrug", "volume_knob", "tail_chase")
ORIGINAL_26 = frozenset(FRAME_NAMES[:26])


def asleep_anim(seed=1, frames=ALL):
    anim = framed(seed=seed, frames=frames)
    run(anim, SLEEP_AFTER + 3.0)
    # An idle activity picked just before may still be going: she lies down
    # once it is over.
    for _ in range(40):
        if anim.asleep:
            break
        run(anim, 1.0)
    assert anim.asleep
    return anim


class TestNewAssets:
    def test_every_drawn_frame_ships(self):
        # The body/costume frames (walk_a onwards) arrive later, in any
        # subset; the 62 drawn before them must always be there.
        drawn = frozenset(FRAME_NAMES[:FRAME_NAMES.index("walk_a")])
        assert len(drawn) == 62
        assert drawn <= mascot.available_frames()

    def test_the_set_stays_small(self):
        present = [n for n in FRAME_NAMES if mascot.frame_path(n).is_file()]
        total = sum(mascot.frame_path(n).stat().st_size for n in present)
        assert total < 25_000 * len(present)  # palette PNGs, ~12 KB each

    def test_an_outlier_frame_does_not_blow_up_the_shared_box(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "make_wawi_assets.py"
        spec = importlib.util.spec_from_file_location("make_wawi_assets", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        boxes = {"a": (20, 260, 1010, 922), "b": (30, 180, 1000, 922),
                 "c": (60, 100, 980, 922)}
        m = mod._MARGIN
        assert mod.shared_box(boxes, (1024, 1024)) == (20 - m, 100 - m, 1010 + m, 922 + m)
        boxes["huge"] = (0, 0, 1024, 1024)  # drawn at the wrong scale
        assert mod.shared_box(boxes, (1024, 1024)) == (20 - m, 100 - m, 1010 + m, 922 + m)


class TestNewBehaviours:
    def test_all_of_them_are_playable_with_the_full_set(self):
        assert set(NEW_BEHAVIOURS) <= set(framed().playable())

    def test_with_the_old_frames_only_they_simply_never_happen(self):
        anim = framed(frames=ORIGINAL_26)
        assert not set(NEW_BEHAVIOURS) & set(anim.playable())
        for _ in range(300):
            name = anim._pick_behaviour()
            anim._last_behaviour = name
            assert name not in NEW_BEHAVIOURS

    def test_dig_paws_three_times_then_shows_off_the_hole(self):
        anim = framed()
        assert anim._start("dig")
        frames = [f for _, f, _ in run(anim, CLIPS["dig"].duration - DT)]
        assert {"dig_a", "dig_b", "dig_c"} <= set(frames)
        assert frames[-1] == "dig_hole"
        assert frames.index("dig_hole") > max(i for i, f in enumerate(frames) if f == "dig_c")

    def test_the_run_goes_out_a_little_and_comes_home(self):
        clip = CLIPS["run"]
        xs = [clip_excursion(clip, i * DT) for i in range(int(clip.duration / DT) + 2)]
        assert min(xs) == pytest.approx(-RUN_DISTANCE, abs=0.5)  # the way she faces
        assert max(xs) <= 0.0
        assert clip_excursion(clip, clip.duration) == pytest.approx(0.0, abs=1e-6)

    def test_running_back_she_is_mirrored_and_only_then(self):
        anim = framed()
        anim._start("run")
        out = []
        for _ in range(int(CLIPS["run"].duration / DT) + 3):
            anim.update(DT, State.IDLE)
            out.append((anim.frame, anim.mirrored, anim.excursion))
        assert any(m for _, m, _ in out) and not out[0][1]
        assert all(f.startswith("run_") for f, m, _ in out if m)
        assert all(x >= -RUN_DISTANCE - 0.5 for _, _, x in out)
        assert out[-1][2] == 0.0 and not out[-1][1]  # home, facing as drawn

    def test_a_run_without_its_stop_frame_still_comes_home(self):
        clip = CLIPS["run"].playable(ALL - {"run_stop_look"})
        assert clip_excursion(clip, clip.duration) == pytest.approx(0.0, abs=1e-6)
        assert CLIPS["run"].playable(ALL - {"run_a", "run_b", "run_c", "run_d",
                                            "run_stop_look"}) is None

    def test_a_state_change_ends_the_run_at_once(self):
        anim = framed()
        anim._start("run")
        run(anim, 0.4)
        assert anim.excursion < 0.0
        anim.update(DT, State.LISTENING)
        assert anim.excursion == 0.0 and not anim.mirrored

    def test_ignored_for_a_while_she_gets_ironic(self):
        anim = framed()
        anim.ignored_for = IGNORED_AFTER + 1.0
        picks = set()
        for _ in range(100):
            name = anim._pick_behaviour()
            anim._last_behaviour = name
            picks.add(name)
        assert picks <= set(IRONIC) and len(picks) >= 2

    def test_talking_to_her_or_touching_her_is_attention(self):
        anim = framed()
        anim.ignored_for = IGNORED_AFTER + 1.0
        anim.update(DT, State.LISTENING)
        assert anim.ignored_for < 1.0
        anim = framed()
        anim.ignored_for = IGNORED_AFTER + 1.0
        anim.touch("head")
        assert anim.ignored_for == 0.0

    def test_after_a_long_answer_she_shrugs_or_smirks(self):
        anim = framed()
        run(anim, LONG_ANSWER + 0.5, State.SPEAKING)
        anim.celebrate()
        anim.update(DT, State.IDLE)
        assert anim.behaviour in ("shrug", "smug")
        # The next short answer is a plain cheer again.
        run(anim, 1.0, State.SPEAKING)
        anim.celebrate()
        anim.update(DT, State.IDLE)
        assert anim.behaviour == "cheer"

    def test_a_long_answer_does_not_spoil_a_party(self):
        anim = framed()
        run(anim, LONG_ANSWER + 0.5, State.SPEAKING)
        anim.celebrate(big=True)
        anim.update(DT, State.IDLE)
        assert anim.behaviour == "party"


class TestWakingOnHerOwn:
    def test_now_and_then_she_gets_up_for_something_rare_then_sleeps_again(self):
        anim = asleep_anim(seed=5)
        out = run(anim, WAKE_GAP[1] + 25.0)
        did = [b for i, (_, _, b) in enumerate(out) if b and (i == 0 or out[i - 1][2] != b)]
        # The big ones, or one of her idle activities.
        assert did and set(did) <= set(WAKE_BEHAVIOURS) | set(mascot.ACTIVITIES)
        assert anim.asleep  # and back to the resting look
        assert "lie_down" not in did  # the yawn is part of the same one-shot

    def test_not_too_often(self):
        anim = asleep_anim(seed=2)
        out = run(anim, 600.0)
        starts = [i * DT for i, (_, _, b) in enumerate(out)
                  if b and (i == 0 or out[i - 1][2] is None)]
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        assert 2 <= len(starts) <= 600 / WAKE_GAP[0] + 1
        assert all(g >= WAKE_GAP[0] for g in gaps)

    def test_long_ignored_she_wakes_up_just_to_be_ironic(self):
        anim = asleep_anim(seed=4)
        anim.ignored_for = IGNORED_AFTER + 1.0
        anim._next_wake = 0.0
        anim.update(DT, State.IDLE)
        assert anim.behaviour in IRONIC

    def test_muted_she_still_gets_up_now_and_then(self):
        anim = asleep_anim(seed=5)
        out = run(anim, WAKE_GAP[1] + 10.0, State.IDLE, muted=True)
        assert any(b is not None for _, _, b in out)

    def test_never_under_reduce_motion(self):
        anim = asleep_anim()
        out = run(anim, WAKE_GAP[1] + 5.0, reduce_motion=True)
        assert all(b is None for _, _, b in out)


class TestListeningExtras:
    @pytest.mark.parametrize("args, frame", [
        (dict(level=0.97, ears_up=True, t=1.0, quiet_for=0.0), "volume_b"),
        (dict(level=0.85, ears_up=True, t=1.0, quiet_for=0.0), "volume_a"),
        (dict(level=0.85, ears_up=True, t=1.0, quiet_for=0.0, megaphone=True), "volume_a"),
        (dict(level=0.4, ears_up=True, t=1.0, quiet_for=0.0), "listening_b"),
        (dict(level=0.0, ears_up=False, t=1.0, quiet_for=0.5), "listening"),
        (dict(level=0.0, ears_up=False, t=0.1, quiet_for=0.1, megaphone=True), "listening"),
        (dict(level=0.3, ears_up=False, t=1.0, quiet_for=0.0, megaphone=True), "megaphone_a"),
        (dict(level=0.0, ears_up=False, t=1.0, quiet_for=0.9, megaphone=True), "megaphone_b"),
        (dict(level=0.0, ears_up=False, t=2.5, quiet_for=1.0, megaphone=True), "listening"),
        (dict(level=0.0, ears_up=False, t=3.0, quiet_for=3.0, ear_cup=True), "ear_cup"),
        (dict(level=0.0, ears_up=False, t=3.0, quiet_for=3.0), "listening"),
    ])
    def test_frame_by_level(self, args, frame):
        assert listening_frame(**args) == frame

    def test_the_first_frame_is_still_plain_listening(self):
        anim = framed()
        anim.MEGAPHONE_CHANCE = 1.0
        anim.update(DT, State.LISTENING)
        assert anim.frame == "listening"  # instant, the trumpet comes after

    def test_the_trumpet_then_leaning_in_when_you_are_quiet(self):
        anim = framed()
        anim.MEGAPHONE_CHANCE, anim.EAR_CUP_CHANCE = 1.0, 0.0
        frames = [f for _, f, _ in run(anim, 2.5, State.LISTENING, level=0.0)]
        assert frames.index("megaphone_a") < frames.index("megaphone_b")
        assert frames[-1] in ("listening", "listening_b")  # trumpet down

    def test_quiet_she_cups_her_ear(self):
        anim = framed()
        anim.MEGAPHONE_CHANCE, anim.EAR_CUP_CHANCE = 0.0, 1.0
        run(anim, 3.0, State.LISTENING, level=0.0)
        assert anim.frame == "ear_cup"
        out = run(anim, 0.5, State.LISTENING, level=0.6)  # speaking up
        assert out[-1][1] == "listening_b"

    def test_a_shout_blows_her_ears_back_quickly(self):
        anim = framed()
        anim.MEGAPHONE_CHANCE = 0.0
        run(anim, 0.3, State.LISTENING, level=0.0)
        out = run(anim, 0.4, State.LISTENING, level=1.0)
        frames = [f for _, f, _ in out]
        assert "volume_a" in frames and frames[-1] == "volume_b"
        assert frames.index("volume_b") * DT < 0.4
        out = run(anim, 2.0, State.LISTENING, level=0.0)
        assert out[-1][1] in ("listening", "listening_b", "ear_cup")

    def test_missing_extras_fall_back_to_plain_listening(self):
        extras = {"megaphone_a", "megaphone_b", "ear_cup", "volume_a", "volume_b"}
        anim = framed(frames=ALL - extras)
        anim.MEGAPHONE_CHANCE = anim.EAR_CUP_CHANCE = 1.0
        seen = {f for _, f, _ in run(anim, 3.0, State.LISTENING, level=0.0)}
        seen |= {f for _, f, _ in run(anim, 1.0, State.LISTENING, level=1.0)}
        assert seen <= {"listening", "listening_b"}

    def test_reduce_motion_keeps_one_listening_frame(self):
        anim = framed()
        anim.MEGAPHONE_CHANCE = 1.0
        out = run(anim, 3.0, State.LISTENING, level=1.0, reduce_motion=True)
        assert {f for _, f, _ in out} == {"listening"}


class TestTouchZones:
    @pytest.mark.parametrize("u, v, zone", [
        (0.29, 0.40, "nose"), (0.10, 0.62, "ear"), (0.57, 0.58, "ear"),
        (0.33, 0.64, "head"), (0.33, 0.50, "head"), (0.14, 0.07, "paw"),
        (0.30, 0.23, "belly"), (0.93, 0.20, "tail"), (0.60, 0.28, "back"),
        (0.92, 0.90, None), (1.2, 0.5, None), (0.5, -0.1, None),
    ])
    def test_sitting(self, u, v, zone):
        assert touch_zone(u, v) == zone

    def test_mirrored_her_head_is_on_the_right(self):
        assert touch_zone(1 - 0.29, 0.40, mirrored=True) == "nose"
        assert touch_zone(1 - 0.93, 0.20, mirrored=True) == "tail"
        assert touch_zone(0.93, 0.20, mirrored=True) != "tail"

    @pytest.mark.parametrize("u, v, zone", [
        (0.26, 0.20, "nose"), (0.33, 0.34, "head"), (0.90, 0.21, "tail"),
        (0.24, 0.06, "paw"), (0.5, 0.85, None),
    ])
    def test_lying(self, u, v, zone):
        assert touch_zone(u, v, lying=True) == zone

    def test_every_zone_is_on_her_in_the_drawing(self):
        """Each zone's centre lands on paint, not on the transparent canvas."""
        PIL = pytest.importorskip("PIL.Image")
        for frame, table in (("idle_a", mascot._ZONES_SITTING),
                             ("sleeping", mascot._ZONES_LYING)):
            img = PIL.open(mascot.frame_path(frame)).convert("RGBA")
            w, h = img.size
            for zone, u0, v0, u1, v1 in table:
                u, v = (u0 + u1) / 2, (v0 + v1) / 2
                assert img.getpixel((int(u * (w - 1)), int((1 - v) * (h - 1))))[3] > 200, \
                    (frame, zone)

    def test_view_points_map_to_the_drawn_frame(self):
        w, h = 124.0, 110.0
        W = w + 2 * mascot.PAD_X
        assert sprite_uv((W / 2, mascot.PAD_BOTTOM), W, (w, h), REST) == \
            pytest.approx((0.5, 0.0))
        assert sprite_uv((W / 2 - w / 2, mascot.PAD_BOTTOM + h), W, (w, h), REST) == \
            pytest.approx((0.0, 1.0))
        # Hopped up 7 pt: the same view point is lower on her.
        hop = Pose(dy=7.0)
        assert sprite_uv((W / 2, mascot.PAD_BOTTOM + 7.0), W, (w, h), hop)[1] == \
            pytest.approx(0.0)

    def test_the_mapping_undoes_a_tilt_and_a_squash(self):
        import math

        w, h = 124.0, 110.0
        W = w + 2 * mascot.PAD_X
        pose = Pose(dx=1.0, dy=2.0, sx=1.03, sy=0.97, rot=6.0)
        u, v = 0.3, 0.6
        # Forward, the way the view draws: scale, rotate, translate.
        x, y = (u - 0.5) * w * pose.sx, v * h * pose.sy
        r = math.radians(pose.rot)
        x, y = x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
        point = (x + W / 2 + pose.dx, y + mascot.PAD_BOTTOM + pose.dy)
        assert sprite_uv(point, W, (w, h), pose) == pytest.approx((u, v))


class TestTouchReactions:
    @pytest.mark.parametrize("zone, allowed", [
        ("head", {"pet"}), ("back", {"pet"}), ("nose", {"boop", "sneeze"}),
        ("belly", {"tickle"}), ("tail", {"tail_grab", "tail_chase"}),
        ("ear", {"ear_flop"}), ("paw", {"paw_up"}),
    ])
    def test_each_zone_has_its_reaction(self, zone, allowed):
        anim = framed()
        assert anim.touch(zone)
        assert anim.behaviour in allowed

    def test_the_nose_sometimes_sneezes_and_the_tail_sometimes_spins(self):
        nose, tail = set(), set()
        for seed in range(40):
            a = framed(seed=seed)
            a.touch("nose")
            nose.add(a.behaviour)
            b = framed(seed=seed)
            b.touch("tail")
            tail.add(b.behaviour)
        assert nose == {"boop", "sneeze"} and tail == {"tail_grab", "tail_chase"}

    def test_a_zone_without_its_frames_gets_a_generic_reaction(self):
        anim = framed(frames=ALL - {"tickle_a", "tickle_b"})
        assert anim.touch("belly")
        assert anim.behaviour in mascot.POKE_REACTIONS

    def test_asleep_a_touch_wakes_her_then_she_reacts(self):
        anim = asleep_anim()
        assert anim.touch("belly")
        assert not anim.asleep and anim.behaviour == "wake"
        frames = [f for _, f, _ in run(anim, 4.5)]
        assert frames.index("stretch") < frames.index("tickle_a")

    def test_a_double_click_hops(self):
        anim = framed()
        anim.touch("head")
        assert anim.double_click()
        assert anim.behaviour == "hop"
        out = run(anim, CLIPS["hop"].duration)
        assert max(p.dy for p, f, _ in out if f == "jump") > 5.0

    def test_a_double_click_wakes_her_into_a_hop(self):
        anim = asleep_anim()
        assert anim.double_click()
        assert "jump" in {f for _, f, _ in run(anim, 4.0)}

    def test_touches_only_while_idle(self):
        anim = framed()
        run(anim, 0.2, State.SPEAKING)
        assert not anim.touch("head") and not anim.double_click()


class TestGrumpy:
    def test_threshold(self):
        assert not too_many_touches([0, 1, 2, 3, 4, 5], 5.0)
        assert too_many_touches([0, 1, 2, 3, 4, 5, 6], 6.0)
        assert not too_many_touches([0, 1, 2, 3, 4, 5, 6], 8.5)  # 0 is too old

    def _poke(self, anim, n, gap):
        for _ in range(n):
            anim.touch("head")
            run(anim, gap)

    def test_too_many_touches_and_she_is_fed_up(self):
        anim = framed()
        self._poke(anim, 7, 0.6)
        assert anim.behaviour == "grumpy"
        # Fed up means fed up: more touches are ignored until it passes.
        assert not anim.touch("nose")
        assert anim.behaviour == "grumpy"

    def test_six_is_still_fine(self):
        anim = framed()
        seen = []
        for _ in range(6):
            anim.touch("nose")
            seen.append(anim.behaviour)
            run(anim, 0.6)
        assert "grumpy" not in seen

    def test_slow_touches_never_annoy_her(self):
        anim = framed()
        seen = []
        for _ in range(12):
            anim.touch("belly")
            seen.append(anim.behaviour)
            run(anim, 2.0)
        assert "grumpy" not in seen


class TestPetting:
    def _stroke(self, det, t0, secs, speed=0.6, lo=0.15, hi=0.5, zone="head"):
        """Back and forth between lo and hi (u), sampled at 60 Hz."""
        t, u, d = t0, lo, 1
        while t < t0 + secs:
            det.feed(t, u, zone)
            u += d * speed / 60
            if u >= hi or u <= lo:
                d = -d
            t += 1 / 60
        return t

    def test_back_and_forth_over_her_head_is_petting(self):
        det = PetDetector()
        t = self._stroke(det, 0.0, 2.0)
        assert det.stroking(t)

    def test_passing_over_her_once_is_not(self):
        det = PetDetector()
        for i in range(60):
            det.feed(i / 60, 0.1 + i / 100, "head")
        assert not det.stroking(1.0)

    def test_a_shaky_hand_is_not(self):
        det = PetDetector()
        for i in range(120):
            det.feed(i / 60, 0.3 + (0.01 if i % 2 else 0.0), "head")
        assert not det.stroking(2.0)

    def test_it_stops_when_the_strokes_do(self):
        det = PetDetector()
        t = self._stroke(det, 0.0, 2.0)
        assert det.stroking(t)
        assert not det.stroking(t + PetDetector.STOP_AFTER + 0.1)

    def test_strokes_on_her_tail_are_not_petting_her_head(self):
        det = PetDetector()
        t = self._stroke(det, 0.0, 2.0, zone="tail")
        assert not det.stroking(t)

    def test_she_melts_while_petted_and_is_happy_after(self):
        anim = framed()
        anim.set_petting(True)
        frames = {f for _, f, _ in run(anim, SLEEP_AFTER + 5.0)}
        assert frames == {"pet_head", "pet_head_b"}
        assert not anim.asleep  # petting keeps her awake
        anim.set_petting(False)
        assert anim.behaviour == "petted"

    def test_asleep_she_enjoys_it_and_sleeps_on(self):
        anim = asleep_anim()
        anim.set_petting(True)
        assert anim.asleep and anim.behaviour == "petting"
        run(anim, 2.0)
        assert anim.frame in ("pet_head", "pet_head_b")
        anim.set_petting(False)
        run(anim, 1.0)
        assert anim.asleep and anim.frame in ("sleeping", "sleeping_b")

    def test_reduce_motion_petting_is_one_still_frame(self):
        anim = framed()
        anim.update(DT, State.IDLE, reduce_motion=True)
        anim.set_petting(True)
        out = run(anim, 3.0, reduce_motion=True)
        assert {f for _, f, _ in out} == {"pet_head"}
        assert all(p == REST for p, _, _ in out)
        anim.set_petting(False)
        anim.update(DT, State.IDLE, reduce_motion=True)
        assert anim.frame == "idle_a"


class TestCursorFollow:
    @pytest.mark.parametrize("dx, dy, frame", [
        (-100, 0, "follow_left"), (100, 10, "follow_right"), (-60, -40, "follow_left"),
        (10, 100, "follow_up"), (0, -100, None), (500, 0, None), (5, 5, None),
    ])
    def test_frame_choice(self, dx, dy, frame):
        assert follow_frame(dx, dy) == frame

    def test_awake_her_eyes_follow_the_cursor(self):
        anim = framed(sleep_after=1e9)
        anim.look_at((-100.0, 0.0))
        run(anim, 0.2)
        assert anim.frame == "follow_left"
        anim.look_at((0.0, 120.0))
        run(anim, 0.5)
        assert anim.frame == "follow_up"
        anim.look_at(None)
        run(anim, 0.1)
        assert anim.frame in ("idle_a", "idle_b")

    def test_a_cursor_wobbling_on_the_line_does_not_flicker(self):
        anim = framed(sleep_after=1e9)
        frames = []
        for i in range(60):
            anim.look_at((3.0 if i % 2 else -3.0, 60.0 if i % 4 < 2 else 20.0))
            anim.update(DT, State.IDLE)
            frames.append(anim.frame)
        changes = sum(1 for a, b in zip(frames, frames[1:]) if a != b)
        assert changes <= 2 / mascot.MascotAnimator.FOLLOW_HOLD + 1

    def test_asleep_she_does_not_follow(self):
        anim = asleep_anim()
        anim.look_at((-100.0, 0.0))
        out = run(anim, 2.0)
        assert {f for _, f, _ in out} <= {"sleeping", "sleeping_b"}

    def test_without_the_follow_frames_she_just_sits(self):
        anim = framed(frames=ALL - {"follow_left", "follow_right", "follow_up"},
                      sleep_after=1e9)
        anim.look_at((-100.0, 0.0))
        run(anim, 0.5)
        assert anim.frame in ("idle_a", "idle_b")


class TestReduceMotionTouch:
    def test_a_zone_shows_one_frame_without_motion(self):
        anim = framed()
        anim.update(DT, State.IDLE, reduce_motion=True)
        assert anim.touch("belly")
        out = run(anim, STILL_HOLD - 0.1, reduce_motion=True)
        assert {f for _, f, _ in out} == {"tickle_a"}  # no kick loop
        assert all(p == REST for p, _, _ in out)
        assert all(b is None for _, _, b in out)
        assert anim.excursion == 0.0 and not anim.mirrored
        run(anim, 0.3, reduce_motion=True)
        assert anim.frame == "idle_a"

    def test_no_double_click_hop(self):
        anim = framed()
        anim.update(DT, State.IDLE, reduce_motion=True)
        assert not anim.double_click()


class TestPanelTouchHooks:
    def test_touch_and_hover_without_a_window_are_harmless(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs._touch_mascot((10.0, 10.0), 1)
        obs._touch_mascot(None, 2)
        obs._hover_mascot((10.0, 10.0))
        obs._hover_mascot(None)
        assert obs._mascot_cursor_offset() is None
