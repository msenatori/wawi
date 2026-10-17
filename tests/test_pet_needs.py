"""Her needs: decay over real time, persistence with catch-up, moods, the
Spanish summary, mood → behaviour weights, settings and the thread-safe API."""

from __future__ import annotations

import json
import random
import threading
from collections import Counter

import pytest

from wawi import mascot
from wawi.pet import api
from wawi.pet.needs import (
    FLOOR,
    NEEDS,
    Needs,
    NeedsStore,
    Snapshot,
    describe,
    mood_bias,
    mood_of,
)
from wawi.pet.settings import PetSettings

H = 3600.0
T0 = 1_800_000_000.0


def needs(**v):
    base = {n: 90.0 for n in NEEDS}
    base.update(v)
    return Needs(base, now=T0)


class TestDecay:
    def test_needs_drop_over_hours_not_minutes(self):
        n = needs()
        n.advance(T0 + 10 * 60)
        s = n.snapshot()
        assert all(s.get(k) > 88.0 for k in NEEDS)

    def test_a_working_day_awake_makes_her_hungry(self):
        n = needs()
        n.advance(T0 + 10 * H)
        assert n.snapshot().food < 35.0
        assert n.mood() in ("hungry", "lonely", "bored")

    def test_she_never_drops_below_the_floor(self):
        n = needs()
        n.advance(T0 + 3 * 24 * H)
        s = n.snapshot()
        assert all(s.get(k) >= FLOOR for k in NEEDS if k != "energy")
        assert min(s.get(k) for k in NEEDS) >= FLOOR

    def test_sleeping_restores_energy_and_slows_the_rest(self):
        awake, asleep = needs(energy=40.0), needs(energy=40.0)
        awake.advance(T0 + 4 * H)
        asleep.advance(T0 + 4 * H, asleep=True)
        assert asleep.snapshot().energy > 40.0 > awake.snapshot().energy
        assert asleep.snapshot().food > awake.snapshot().food

    def test_a_clock_going_backwards_changes_nothing(self):
        n = needs()
        n.advance(T0 - 5 * H)
        assert n.snapshot().food == pytest.approx(90.0)

    def test_interactions_raise_needs_and_cap_at_100(self):
        n = needs(food=20.0, water=20.0, fun=20.0, affection=20.0)
        n.feed()
        n.water()
        n.play()
        n.touched()
        n.petted(4.0)
        s = n.snapshot()
        assert s.food > 70 and s.water > 70 and s.fun > 55 and s.affection > 35
        for _ in range(10):
            n.feed()
        assert n.snapshot().food == 100.0


class TestPersistence:
    def test_round_trip(self, tmp_path):
        store = NeedsStore(tmp_path / "pet_state.json")
        n = needs(food=42.0)
        assert store.save(n, T0, force=True)
        back = store.load(T0)
        assert back.snapshot().food == pytest.approx(42.0)

    def test_restart_catches_up_the_time_away_as_rest(self, tmp_path):
        store = NeedsStore(tmp_path / "pet_state.json")
        store.save(needs(food=80.0, energy=30.0), T0, force=True)
        back = store.load(T0 + 6 * H)
        s = back.snapshot()
        assert s.food < 80.0          # she got hungrier while away…
        assert s.energy > 30.0        # …but she slept
        assert s.food > 80.0 - 6 * 6.0  # resting decays slower than awake

    def test_catch_up_is_capped(self, tmp_path):
        store = NeedsStore(tmp_path / "pet_state.json")
        store.save(needs(), T0, force=True)
        back = store.load(T0 + 365 * 24 * H)
        assert min(back.snapshot().get(k) for k in NEEDS) >= FLOOR

    def test_missing_or_broken_file_gives_fresh_needs(self, tmp_path):
        p = tmp_path / "pet_state.json"
        assert NeedsStore(p).load(T0).mood() in ("happy", "content")
        p.write_text("{not json")
        assert NeedsStore(p).load(T0).snapshot().food == pytest.approx(80.0)

    def test_saves_are_throttled_unless_something_changed(self, tmp_path):
        store = NeedsStore(tmp_path / "s.json")
        n = needs()
        assert store.save(n, T0)
        assert not store.save(n, T0 + 10)
        n.feed()
        assert store.save(n, T0 + 11)
        assert store.save(n, T0 + 11 + store.SAVE_EVERY)

    def test_the_file_is_plain_json(self, tmp_path):
        p = tmp_path / "s.json"
        NeedsStore(p).save(needs(), T0, force=True)
        data = json.loads(p.read_text())
        assert set(data["needs"]) == set(NEEDS) and data["saved_at"] == T0

    def test_no_path_never_writes(self):
        assert NeedsStore(None).save(needs(), T0, force=True) is False


class TestMood:
    @pytest.mark.parametrize("vals,mood", [
        ({}, "happy"),
        ({"food": 20.0}, "hungry"),
        ({"water": 25.0}, "hungry"),
        ({"energy": 15.0}, "sleepy"),
        ({"affection": 10.0}, "lonely"),
        ({"fun": 20.0}, "bored"),
        ({"food": 50.0}, "content"),
        ({"fun": 10.0, "food": 25.0}, "bored"),  # the most pressing wins
    ])
    def test_mood_of(self, vals, mood):
        base = {n: 90.0 for n in NEEDS}
        base.update(vals)
        assert mood_of(Snapshot(**base)) == mood

    def test_spanish_summary(self):
        assert describe(Snapshot(90, 90, 90, 90, 90)).startswith("Está feliz")
        text = describe(Snapshot(20, 25, 90, 90, 90))
        assert text.startswith("Tiene hambre") and "sed" in text
        assert "tranquila" in describe(Snapshot(60, 90, 90, 90, 90))

    def test_levels_for_the_voice_brain(self):
        lv = needs(food=25.0).levels()
        assert lv["mood"] == "hungry"
        assert lv["hunger"] == pytest.approx(0.75)
        assert 0.0 <= lv["thirst"] <= 1.0 and lv["energy"] == pytest.approx(0.9)
        assert isinstance(lv["summary"], str)


class TestMoodBias:
    def _counts(self, mood, n=3000):
        anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(3))
        anim.bias = mood_bias(mood)
        c = Counter()
        for _ in range(n):
            name = anim._pick_behaviour()
            c[name] += 1
            anim._last_behaviour = None
        return c

    def test_hunger_brings_in_the_empty_bowl(self):
        assert self._counts("content")["hungry_bowl"] == 0
        assert self._counts("hungry")["hungry_bowl"] > 100

    def test_a_happy_dog_rolls_more_and_sulks_less(self):
        happy, content = self._counts("happy"), self._counts("content")
        assert happy["roll"] > content["roll"] * 1.5
        assert happy["eyeroll"] < content["eyeroll"]

    def test_a_sleepy_dog_yawns(self):
        assert self._counts("sleepy")["yawn"] > self._counts("content")["yawn"] * 2

    def test_unknown_mood_is_neutral(self):
        assert mood_bias("zen") == {}


class TestSettings:
    def test_defaults(self):
        s = PetSettings.from_dict({})
        assert s.enabled and s.wander and s.lonely_after == 30 * 60
        assert s.wander_every == (8 * 60, 25 * 60)
        assert str(s.state_file) == "pet_state.json"

    def test_garbage_falls_back(self):
        s = PetSettings.from_dict({"wander_every_min": "often", "lonely_after_min": "x",
                                   "state_file": None})
        assert s.wander_every == (8 * 60, 25 * 60)
        assert s.lonely_after == 30 * 60 and s.state_file is None

    def test_range_is_ordered(self):
        assert PetSettings.from_dict({"wander_every_min": [20, 5]}).wander_every == (300, 1200)

    def test_config_carries_the_section(self, tmp_path):
        from wawi import config

        p = tmp_path / "config.yaml"
        p.write_text("language: es\npet:\n  enabled: false\n  lonely_after_min: 5\n")
        cfg = config.load(p)
        s = PetSettings.from_dict(cfg.pet)
        assert s.enabled is False and s.lonely_after == 300


class TestApi:
    def setup_method(self):
        api.attach(None)

    def teardown_method(self):
        api.attach(None)

    def test_not_running(self):
        assert api.feed() is False
        assert "no está" in api.status()
        assert api.needs() is None

    def test_commands_from_any_thread_queue_for_the_main_thread(self):
        n = needs()
        api.attach(n.status, n.levels)
        threads = [threading.Thread(target=f) for f in (api.feed, api.water, api.play) * 5]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        got = api.drain()
        assert Counter(got) == {"feed": 5, "water": 5, "play": 5}
        assert api.drain() == []
        assert api.status().startswith("Está")
        assert api.needs()["mood"] == "happy"

    def test_a_broken_source_does_not_raise(self):
        def boom():
            raise RuntimeError

        api.attach(boom, boom)
        assert "No sé" in api.status()
        assert api.needs() is None
