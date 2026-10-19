"""What the owner is doing, from the front app's NAME only: which category it
is, and how rarely she makes a gesture about it."""

from __future__ import annotations

import random

import pytest

from wawi.pet import context as C


class TestClassify:
    @pytest.mark.parametrize("owner,category", [
        ("Code", "code"), ("Cursor", "code"), ("iTerm2", "code"), ("Terminal", "code"),
        ("Ghostty", "code"), ("PyCharm CE", "code"), ("Xcode", "code"),
        ("QuickTime Player", "video"), ("IINA", "video"), ("VLC", "video"), ("TV", "video"),
        ("Slack", "chat"), ("Messages", "chat"), ("WhatsApp", "chat"), ("Discord", "chat"),
        ("Figma", "design"), ("Sketch", "design"), ("Adobe Photoshop 2025", "design"),
    ])
    def test_known_apps(self, owner, category):
        assert C.classify(owner) == category

    @pytest.mark.parametrize("owner", ["Finder", "Safari", "Mail", "", None, "TVision Pro",
                                       "Codex Helper", "Notes"])
    def test_most_apps_have_no_category(self, owner):
        assert C.classify(owner) is None

    def test_every_category_has_a_gesture(self):
        assert set(C.APPS) | {"switch"} == set(C.GESTURES)


def gestures(gap=(600.0, 600.0)):
    return C.ContextGestures(gap, rng=random.Random(1))


class TestRateLimit:
    def test_nothing_for_the_app_open_at_launch_until_half_a_gap(self):
        g = gestures()
        assert g.sample(0.0, "Code", True) is None
        assert g.sample(C.DWELL + 1, "Code", True) is None  # dwell, but too soon
        assert g.sample(301.0, "Code", True) == "code"

    def test_a_switch_is_a_glance_after_it_settles(self):
        g = gestures()
        g.sample(0.0, "Finder", True)
        assert g.sample(400.0, "Safari", True) is None  # just switched
        assert g.sample(400.5, "Safari", True) is None
        assert g.sample(400.0 + C.SETTLE, "Safari", True) == "switch"

    def test_a_switch_to_a_known_app_is_its_gesture(self):
        g = gestures()
        g.sample(0.0, "Finder", True)
        g.sample(400.0, "Slack", True)
        assert g.sample(402.0, "Slack", True) == "chat"

    def test_at_most_one_per_gap(self):
        g = gestures()
        g.sample(0.0, "Finder", True)
        out = []
        t = 400.0
        apps = ["Safari", "Slack", "Code", "Figma"]
        for i in range(600):  # an hour of switching every 6 s
            t += 6.0
            out.append(g.sample(t, apps[i % len(apps)], True))
            out.append(g.sample(t + 2.0, apps[i % len(apps)], True))
        fired = [c for c in out if c is not None]
        assert 4 <= len(fired) <= 7  # ~3600 s / 600 s

    def test_not_allowed_only_tracks_and_a_stale_switch_is_dropped(self):
        g = gestures()
        g.sample(0.0, "Finder", True)
        g.sample(400.0, "Safari", False)
        assert g.sample(402.0, "Safari", False) is None  # in a meeting, say
        assert g.sample(400.0 + C.STALE + 1, "Safari", True) is None

    def test_a_long_stay_in_a_known_app_gets_its_gesture(self):
        g = gestures()
        g.sample(0.0, "Finder", True)
        g.sample(400.0, "IINA", False)  # switched during a meeting
        assert g.sample(420.0, "IINA", True) is None
        assert g.sample(400.0 + C.DWELL, "IINA", True) == "video"

    def test_no_front_app_no_gesture(self):
        g = gestures()
        g.sample(0.0, "Finder", True)
        assert g.sample(400.0, None, True) is None
        assert g.sample(405.0, None, True) is None
