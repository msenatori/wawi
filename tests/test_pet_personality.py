"""Personalities (wawi/pet/personality.py, files in personalities/): the
built-ins load, missing keys fall back to the default, bad values are
clamped and never crash, and pet.liveliness is an alias."""

from __future__ import annotations

import dataclasses
import logging

import pytest
import yaml

from wawi.pet import personality as P
from wawi.pet.settings import PetSettings

BUILT_INS = ("perra_tranquila", "perra_atenta", "cachorra")


class TestBuiltIns:
    def test_all_three_ship_in_the_repo_folder(self):
        assert P.FOLDER.name == "personalities" and P.FOLDER.parent.name != "wawi"
        assert set(BUILT_INS) <= set(P.available())

    @pytest.mark.parametrize("name", BUILT_INS)
    def test_each_loads_clean_with_its_metadata(self, name, caplog):
        with caplog.at_level(logging.WARNING, logger="wawi.pet.personality"):
            p = P.load(name)
        assert not caplog.records  # no unknown keys, nothing out of range
        assert p.name == name and p.description and p.author and p.version

    def test_the_default_file_says_what_the_code_defaults_say(self):
        """perra_tranquila.yaml is complete and matches Personality()."""
        raw = P.flatten(yaml.safe_load(P.find(P.DEFAULT).read_text()))
        assert set(P.SPEC) <= set(raw)  # every parameter is in the file
        assert dataclasses.replace(P.load(P.DEFAULT), description="") \
            == dataclasses.replace(P.Personality(name=P.DEFAULT, author="Wawi", version="1"),
                                   description="")

    def test_the_levels_differ_the_right_way(self):
        calm, alert, young = (P.load(n) for n in BUILT_INS)
        assert alert.rest_minutes[0] > calm.rest_minutes[0]
        assert alert.rest_glance_every_min[1] < calm.rest_glance_every_min[1]
        assert young.sleep_deep_median_min < calm.sleep_deep_median_min
        assert young.play_per_day == 3 and calm.play_per_day == 1
        assert calm.play_cooldown_h >= 8.0


class TestFallbacks:
    def test_missing_keys_come_from_the_default(self, tmp_path):
        (tmp_path / "tiny.yaml").write_text("name: tiny\nrest:\n  minutes: [1, 2]\n")
        p = P.load("tiny", search=(tmp_path, P.FOLDER))
        base = P.load(P.DEFAULT)
        assert p.name == "tiny" and p.rest_minutes == (1.0, 2.0)
        assert p.sleep_deep_median_min == base.sleep_deep_median_min
        assert p.greeting_full_after_min == base.greeting_full_after_min

    def test_unknown_name_is_the_default(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wawi.pet.personality"):
            p = P.load("no_such_dog")
        assert p.name == P.DEFAULT
        assert any("no_such_dog" in r.getMessage() for r in caplog.records)

    def test_a_path_works_and_json_too(self, tmp_path):
        f = tmp_path / "mine.json"
        f.write_text('{"name": "mine", "play": {"per_day": 2}}')
        p = P.load(str(f))
        assert p.name == "mine" and p.play_per_day == 2

    @pytest.mark.parametrize("text", ["", "- a list\n", "name: [unclosed\n", "42\n"])
    def test_broken_files_never_crash(self, tmp_path, text):
        (tmp_path / "bad.yaml").write_text(text)
        p = P.load("bad", search=(tmp_path, P.FOLDER))
        assert p.sleep_deep_median_min == P.load(P.DEFAULT).sleep_deep_median_min


class TestClamping:
    def test_out_of_range_is_clamped_and_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wawi.pet.personality"):
            p = P.from_dict({"active": {"chance": 7}, "habituation": {"drop": -1},
                             "rest": {"minutes": [500, -3]}, "play": {"per_day": 2.6}})
        assert p.active_chance == 1.0 and p.habituation_drop == 0.0
        assert p.rest_minutes == (0.5, 240.0)  # each end clamped, then sorted
        assert p.play_per_day == 3
        assert len(caplog.records) >= 3

    def test_wrong_types_are_dropped(self, caplog):
        with caplog.at_level(logging.WARNING, logger="wawi.pet.personality"):
            p = P.from_dict({"sleep": {"deep_sigma": "lots", "doze_in_min": ["a", 2]},
                             "wake": {"stretch": True}, "nope": 1})
        base = P.Personality()
        assert p.sleep_deep_sigma == base.sleep_deep_sigma
        assert p.sleep_doze_in_min == base.sleep_doze_in_min
        assert p.wake_stretch == base.wake_stretch
        assert any("unknown key nope" in r.getMessage() for r in caplog.records)

    def test_a_single_number_for_a_range_is_both_ends(self):
        assert P.from_dict({"rest": {"minutes": 12}}).rest_minutes == (12.0, 12.0)

    def test_min_above_max_is_fixed(self):
        p = P.from_dict({"sleep": {"deep_min_min": 50, "deep_max_min": 20}})
        assert p.sleep_deep_min_min <= p.sleep_deep_max_min


class TestConfig:
    def test_default_is_the_calm_dog(self):
        assert PetSettings().personality == "perra_tranquila"
        s = PetSettings.from_dict({})
        assert s.liveliness == "calm" and s.personality == "perra_tranquila"

    @pytest.mark.parametrize("level,name", [("calm", "perra_tranquila"),
                                            ("normal", "perra_atenta"),
                                            ("playful", "cachorra"),
                                            ("juguetona", "cachorra"),
                                            ("wild", "perra_tranquila")])
    def test_liveliness_is_an_alias(self, level, name):
        assert PetSettings.from_dict({"liveliness": level}).personality == name

    def test_personality_wins_over_liveliness(self):
        s = PetSettings.from_dict({"liveliness": "playful", "personality": "perra_atenta"})
        assert s.personality == "perra_atenta"

    def test_aliases_load_too(self):
        assert P.load("playful").name == "cachorra"
        assert P.load("").name == P.DEFAULT
