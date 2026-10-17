"""The mascot engine's body hooks: anchored frames, the window canvas, props,
acts and locomotion over idle life and sleep, mood bias, mirroring."""

from __future__ import annotations

import json
import random

import pytest

from wawi import mascot
from wawi.mascot import ACT_CLIPS, BODY_CLIPS, Clip, MascotAnimator, Step
from wawi.state import State

DT = 1 / 30
ALL = mascot.FRAME_NAMES


def anim(frames=ALL, seed=1):
    return MascotAnimator(available=frames, rng=random.Random(seed))


def run(a, seconds, state=State.IDLE):
    out = []
    for _ in range(int(seconds / DT)):
        a.update(DT, state)
        out.append(a.frame)
    return out


class TestAnchors:
    ANCH = {"_box": (0.0, 0.0, 320.0, 286.0), "sit_edge": (8.75, -55.94, 302.81, 243.12),
            "prop_ball": (124.0, 0.31, 72.0, 70.6)}

    def test_shared_frames_fill_the_sprite(self):
        assert mascot.frame_rect("idle_a", (124.0, 110.0), self.ANCH) == (-62.0, 0.0, 124.0, 110.0)
        assert mascot.frame_rect("sit_edge", (124.0, 110.0), {}) == (-62.0, 0.0, 124.0, 110.0)

    def test_a_loose_frame_hangs_below_the_ledge_at_the_same_scale(self):
        x, y, w, h = mascot.frame_rect("sit_edge", (124.0, 110.0), self.ANCH)
        k = 124.0 / 320.0
        assert y == pytest.approx(-55.94 * k) and y < 0
        assert w == pytest.approx(302.81 * k) and x == pytest.approx(-62.0 + 8.75 * k)

    def test_canvas_extent_covers_the_frames_and_is_capped(self):
        sprite = (124.0, 110.0)
        rects = [mascot.frame_rect(n, sprite, self.ANCH) for n in ("idle_a", "sit_edge")]
        side, below, above = mascot.canvas_extent(rects, sprite)
        assert below == pytest.approx(55.94 * 124 / 320) and side == 0.0 and above == 0.0
        huge = [(-1000.0, -1000.0, 3000.0, 3000.0)]
        side, below, above = mascot.canvas_extent(huge, sprite, cap=0.5)
        assert side == 62.0 and below == 55.0 and above == 55.0

    def test_props_stand_on_the_ground_in_front_of_her(self):
        x, y, w, h = mascot.prop_rect("prop_ball", (124.0, 110.0), self.ANCH)
        assert y == 0.0 and w == pytest.approx(72.0 * 124 / 320)
        assert x + w / 2 < 0  # her head side (left)
        x, y, w, h = mascot.prop_rect("prop_bone", (124.0, 110.0), {}, (100.0, 50.0))
        assert w == pytest.approx(0.32 * 124) and h == pytest.approx(w / 2)

    def test_a_broken_anchors_file_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mascot, "_missing_logged", set())
        d = tmp_path / mascot.FRAME_DIR
        d.mkdir()
        (d / mascot.ANCHORS_FILE).write_text("{oops")
        assert mascot.load_anchors(tmp_path) == {}
        (d / mascot.ANCHORS_FILE).write_text(json.dumps({"_box": [0, 0, 3, 4], "bad": [1]}))
        assert mascot.load_anchors(tmp_path) == {"_box": (0.0, 0.0, 3.0, 4.0)}
        assert mascot.load_anchors(tmp_path / "nowhere") == {}

    def test_shipped_props_exist_and_are_anchored(self):
        anchors = mascot.load_anchors()
        for p in mascot.available_props():
            assert p in anchors


class TestImporterAnchors:
    def _mod(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "scripts" / "make_wawi_assets.py"
        spec = importlib.util.spec_from_file_location("make_wawi_assets", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_anchor_math(self):
        mod = self._mod()
        box = (0, 100, 1000, 930)
        # Same bottom as the box: y 0; half the scale.
        assert mod.anchor_for((100, 500, 300, 930), box, 0.5) == [50.0, 0.0, 100.0, 215.0]
        # sit_edge: ground at 700 moved onto the 922 baseline (shift 222).
        a = mod.anchor_for((0, 200, 1000, 880), box, 0.5, shift=222)
        assert a[1] == pytest.approx((930 - (880 + 222)) * 0.5) and a[1] < 0

    def test_import_crops_loose_frames_separately(self, tmp_path):
        from PIL import Image

        mod = self._mod()
        src = tmp_path / "src"
        src.mkdir()
        for name, (x0, y0, x1, y1) in {
            "idle_a": (100, 300, 900, 922), "idle_b": (110, 310, 890, 922),
            "ship_fly_a": (50, 20, 1000, 700), "prop_ball": (450, 850, 560, 922),
        }.items():
            img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
            img.paste((200, 50, 50, 255), (x0, y0, x1, y1))
            img.save(src / f"{name}.png")
        dest = tmp_path / "out"
        mod.import_frames(src, dest, width=320, quantize=False)
        anchors = json.loads((dest / "anchors.json").read_text())
        assert set(anchors) == {"_box", "ship_fly_a", "prop_ball"}
        bw, bh = anchors["_box"][2:]
        assert Image.open(dest / "idle_a.png").size == (bw, bh)
        assert Image.open(dest / "prop_ball.png").size[0] < 60
        assert anchors["ship_fly_a"][1] > 0  # up in the sky


class TestActsAndLocomotion:
    def test_resolved_clips_borrow_relatives(self):
        clip = Clip("x", (Step("walk_a", 0.1), Step("nope", 0.1)))
        r = clip.resolved({"idle_a", "run_a"})
        assert [s.frame for s in r.steps] == ["run_a", "idle_a"]
        assert Clip("y", (Step("idle_b", 1.0),)).resolved({"idle_a"}) is None

    def test_an_act_wakes_her_and_plays(self):
        a = anim()
        run(a, 15.0)
        assert a.asleep
        assert a.act("bark")
        frames = run(a, ACT_CLIPS["bark"].duration + 0.2)
        assert {"bark_a", "bark_b"} <= set(frames)
        assert not a.acting and not a.asleep

    def test_a_looping_act_holds_off_sleep(self):
        a = anim()
        a.act("watch", loop=True)
        frames = run(a, 30.0)
        assert not a.asleep and set(frames) <= {"watch", "watch_b"}
        a.stop_act()
        run(a, 15.0)
        assert a.asleep  # the resting look comes back on its own

    def test_a_voice_turn_shows_over_an_act_and_pauses_it(self):
        a = anim()
        a.act("howl")
        run(a, 0.5)
        t = a._act_t
        frames = run(a, 2.0, State.SPEAKING)
        assert "howl" not in frames and a._act_t == t
        assert run(a, 0.2)[-1] == "howl"

    def test_locomotion_wins_even_over_a_voice_turn(self):
        a = anim()
        a.set_locomotion("fall")
        assert run(a, 0.5, State.LISTENING)[-1] == "fall"
        a.set_locomotion("walk")
        assert set(run(a, 1.0)) <= {"walk_a", "walk_b", "walk_c", "walk_d"}
        a.set_locomotion(None)
        assert a.locomotion is None

    def test_mirrored_follows_her_walk(self):
        a = anim()
        a.set_locomotion("walk")
        a.facing_right = True
        run(a, 0.1)
        assert a.mirrored
        a.facing_right = False
        assert not a.mirrored

    def test_unknown_locomotion_and_acts_are_refused(self):
        a = anim()
        a.set_locomotion("teleport")
        assert a.locomotion is None
        assert a.act("moonwalk") is False
        assert MascotAnimator().act("bark") is False  # no frames at all

    def test_costume_clip_acts(self):
        a = anim()
        clip = mascot.costume_clip("umbrella", a.frames)
        assert a.act(clip, loop=True)
        assert set(run(a, 3.0)) == {"umbrella"}

    def test_touches_do_not_interrupt_an_act(self):
        a = anim()
        run(a, 1.0)
        a.act("doctor")
        assert a.touch("nose") is False
        assert a.act_name == "doctor"

    def test_the_run_act_by_the_panel_moves_and_mirrors(self):
        a = anim()
        a.act(Clip("run", mascot.CLIPS["run"].steps, loop=True))
        xs, mirrored = [], []
        for _ in range(int(mascot.CLIPS["run"].duration / DT)):
            a.update(DT, State.IDLE)
            xs.append(a.excursion)
            mirrored.append(a.mirrored)
        assert min(xs) < -20 and True in mirrored and False in mirrored

    def test_sit_edge_is_her_awake_resting_clip_on_a_window(self):
        a = anim()
        a.idle_clip = "sit_edge"
        frames = run(a, 2.0)
        assert "sit_edge" in frames and "idle_a" not in frames and "idle_b" not in frames

    def test_every_body_and_act_clip_is_playable_with_today_s_frames(self):
        a = MascotAnimator(available=mascot.available_frames())
        assert set(BODY_CLIPS) <= set(a._body_clips)
        assert set(ACT_CLIPS) <= set(a._acts)


class TestPanelPetHooks:
    """No window is built here: the hooks must be harmless before start()."""

    def test_configure_and_commands_without_a_window(self):
        from wawi.panel import PanelObserver
        from wawi.pet.settings import PetSettings

        obs = PanelObserver(True)
        obs.configure_pet({"enabled": True, "lonely_after_min": 5})
        assert isinstance(obs.pet_settings, PetSettings)
        assert obs.pet_settings.lonely_after == 300
        assert obs.pet_command("feed") is False  # no body yet
        obs._tick_mascot()
        obs._stop_pet()

    def test_bad_settings_keep_the_plain_mascot(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.configure_pet("nonsense")
        assert obs.pet_settings is not None and obs.pet_settings.enabled
        obs.configure_pet({"enabled": False})
        assert obs._ensure_pet() is None

    def test_no_body_without_frames(self):
        from wawi.panel import PanelObserver

        obs = PanelObserver(True)
        obs.configure_pet({})
        assert not obs._mascot_animator.framed
        assert obs._ensure_pet() is None
