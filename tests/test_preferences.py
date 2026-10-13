"""Tests for durable, user-taught preferences."""

from __future__ import annotations

from wawi.preferences import MAX_CHARS, MAX_PREFERENCES, Preference, Preferences


class TestStoring:
    def test_a_preference_is_remembered(self, tmp_path):
        p = Preferences(tmp_path / "prefs.txt")
        p.add("Sé más breve")
        assert [i.text for i in p.items()] == ["Sé más breve"]

    def test_empty_input_stores_nothing(self, tmp_path):
        p = Preferences(tmp_path / "prefs.txt")
        assert p.add("") is None
        assert p.add("   ") is None
        assert len(p) == 0

    def test_whitespace_is_normalized(self, tmp_path):
        p = Preferences(tmp_path / "prefs.txt")
        p.add("  sé   más\n breve  ")
        assert p.items()[0].text == "sé más breve"

    def test_an_overlong_preference_is_clipped(self, tmp_path):
        p = Preferences(tmp_path / "prefs.txt")
        p.add("x" * (MAX_CHARS * 2))
        assert len(p.items()[0].text) <= MAX_CHARS

    def test_repeating_yourself_reinforces_rather_than_duplicates(self, tmp_path):
        p = Preferences(tmp_path / "prefs.txt")
        p.add("sé breve")
        p.add("no uses markdown")
        p.add("SÉ BREVE")
        assert len(p) == 2
        # Restated, so it is now the most recent — and last wins on conflict.
        assert p.items()[-1].text == "SÉ BREVE"

    def test_the_oldest_is_dropped_when_full(self, tmp_path):
        p = Preferences(tmp_path / "prefs.txt", max_items=3)
        for i in range(5):
            p.add(f"preferencia {i}")
        assert [i.text for i in p.items()] == [
            "preferencia 2",
            "preferencia 3",
            "preferencia 4",
        ]

    def test_default_cap(self, tmp_path):
        assert Preferences(tmp_path / "p.txt").max_items == MAX_PREFERENCES


class TestPersistence:
    def test_preferences_survive_a_restart(self, tmp_path):
        path = tmp_path / "prefs.txt"
        Preferences(path).add("hablá más lento")
        assert [i.text for i in Preferences(path).items()] == ["hablá más lento"]

    def test_the_file_is_editable_by_hand(self, tmp_path):
        """A misheard preference must not need the assistant to remove it."""
        path = tmp_path / "prefs.txt"
        path.write_text("# comentario\nno uses markdown\nsé breve\n")
        assert [i.text for i in Preferences(path).items()] == [
            "no uses markdown",
            "sé breve",
        ]

    def test_comments_and_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "prefs.txt"
        path.write_text("\n# nota\n\n  \nsé breve\n")
        assert len(Preferences(path)) == 1

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert len(Preferences(tmp_path / "nope.txt")) == 0

    def test_an_unwritable_path_never_raises(self, tmp_path):
        blocked = tmp_path / "afile"
        blocked.write_text("not a directory")
        p = Preferences(blocked / "nested" / "prefs.txt")
        p.add("sé breve")  # must not raise
        assert len(p) == 1

    def test_disabled_when_no_path_is_given(self):
        p = Preferences(None)
        assert not p.enabled
        p.add("sé breve")
        assert len(p) == 1  # still usable in memory for the session


class TestRemoval:
    def test_remove_by_the_number_the_user_hears(self, tmp_path):
        p = Preferences(tmp_path / "prefs.txt")
        p.add("uno")
        p.add("dos")
        assert p.remove(1).text == "uno"
        assert [i.text for i in p.items()] == ["dos"]

    def test_removing_a_number_that_does_not_exist_is_safe(self, tmp_path):
        p = Preferences(tmp_path / "prefs.txt")
        p.add("uno")
        assert p.remove(0) is None
        assert p.remove(9) is None
        assert len(p) == 1

    def test_clear_forgets_everything_including_on_disk(self, tmp_path):
        path = tmp_path / "prefs.txt"
        p = Preferences(path)
        p.add("uno")
        p.clear()
        assert len(Preferences(path)) == 0


class TestPromptRendering:
    def test_nothing_learned_costs_no_tokens(self, tmp_path):
        assert Preferences(tmp_path / "p.txt").as_prompt_block() == ""

    def test_every_preference_reaches_the_prompt(self, tmp_path):
        p = Preferences(tmp_path / "p.txt")
        p.add("sé breve")
        p.add("no leas listas enteras")
        block = p.as_prompt_block()
        assert "sé breve" in block
        assert "no leas listas enteras" in block

    def test_the_prompt_says_the_newest_wins(self, tmp_path):
        """Two contradictory corrections must resolve, not confuse."""
        p = Preferences(tmp_path / "p.txt")
        p.add("sé breve")
        assert "gana la última" in p.as_prompt_block()

    def test_spoken_list_is_numbered_for_removal(self, tmp_path):
        p = Preferences(tmp_path / "p.txt")
        p.add("uno")
        p.add("dos")
        spoken = p.spoken_list()
        assert spoken.startswith("1: uno")
        assert "2: dos" in spoken

    def test_spoken_list_says_so_when_empty(self, tmp_path):
        assert "no me enseñaste" in Preferences(tmp_path / "p.txt").spoken_list()


class TestPreferenceLine:
    def test_round_trips_through_a_line(self):
        pref = Preference("sé breve", "2026-08-09")
        assert Preference.from_line(pref.to_line()) == pref

    def test_a_line_without_a_date_still_parses(self):
        assert Preference.from_line("sé breve").text == "sé breve"
