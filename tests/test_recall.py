"""Tests for the durable rolling memory that survives a restart."""

from __future__ import annotations

from wawi.prompts import system_prompt
from wawi.recall import (
    DEFAULT_MAX_ENTRIES,
    MAX_CHARS_PER_SIDE,
    Exchange,
    Recall,
)


class TestStoring:
    def test_an_exchange_is_remembered(self, tmp_path):
        r = Recall(tmp_path / "recall.txt")
        r.add("cómo va el deploy", "ya salió")
        assert [(e.user, e.assistant) for e in r.entries()] == [
            ("cómo va el deploy", "ya salió")
        ]

    def test_an_empty_exchange_is_not_worth_a_slot(self, tmp_path):
        r = Recall(tmp_path / "recall.txt")
        assert r.add("", "") is None
        assert r.add("   ", "\n") is None
        assert len(r) == 0

    def test_half_an_exchange_is_still_worth_keeping(self, tmp_path):
        """A dropped answer still says what the user was asking about."""
        r = Recall(tmp_path / "recall.txt")
        assert r.add("cómo va el deploy", "") is not None
        assert len(r) == 1

    def test_whitespace_is_collapsed(self, tmp_path):
        r = Recall(tmp_path / "recall.txt")
        r.add("  cómo\n va  ", "  ya\tsalió ")
        assert r.entries()[0].user == "cómo va"
        # A tab in the text would shift the columns of every stored field.
        assert "\t" not in r.entries()[0].assistant

    def test_each_side_is_clipped(self, tmp_path):
        r = Recall(tmp_path / "recall.txt")
        r.add("x" * 900, "y" * 900)
        entry = r.entries()[0]
        assert len(entry.user) <= MAX_CHARS_PER_SIDE
        assert len(entry.assistant) <= MAX_CHARS_PER_SIDE


class TestEviction:
    """Newest kept, oldest dropped — the whole point of a rolling cache."""

    def test_exactly_at_the_bound_nothing_is_dropped(self, tmp_path):
        r = Recall(tmp_path / "recall.txt", max_entries=3)
        for i in range(3):
            r.add(f"pregunta {i}", f"respuesta {i}")
        assert [e.user for e in r.entries()] == [
            "pregunta 0",
            "pregunta 1",
            "pregunta 2",
        ]

    def test_one_past_the_bound_drops_exactly_the_oldest(self, tmp_path):
        r = Recall(tmp_path / "recall.txt", max_entries=3)
        for i in range(4):
            r.add(f"pregunta {i}", f"respuesta {i}")
        assert [e.user for e in r.entries()] == [
            "pregunta 1",
            "pregunta 2",
            "pregunta 3",
        ]

    def test_the_bound_holds_across_a_restart(self, tmp_path):
        """Loading must re-apply the cap, or a shrunk config never takes effect."""
        path = tmp_path / "recall.txt"
        wide = Recall(path, max_entries=10)
        for i in range(10):
            wide.add(f"pregunta {i}", f"respuesta {i}")

        narrow = Recall(path, max_entries=2)
        assert [e.user for e in narrow.entries()] == ["pregunta 8", "pregunta 9"]

    def test_the_bound_is_configurable_and_defaults_small(self, tmp_path):
        assert Recall(tmp_path / "r.txt").max_entries == DEFAULT_MAX_ENTRIES
        assert Recall(tmp_path / "r.txt", max_entries=2).max_entries == 2


class TestPersistence:
    def test_exchanges_survive_a_restart(self, tmp_path):
        path = tmp_path / "recall.txt"
        Recall(path).add("cómo va el deploy", "ya salió")
        assert [e.user for e in Recall(path).entries()] == ["cómo va el deploy"]

    def test_the_file_is_editable_by_hand(self, tmp_path):
        """A transcript of your house must be fixable without asking the machine."""
        path = tmp_path / "recall.txt"
        r = Recall(path)
        r.add("uno", "dos")
        text = path.read_text(encoding="utf-8")
        assert "uno" in text and "dos" in text
        assert text.lstrip().startswith("#")

    def test_the_write_leaves_no_temp_file_behind(self, tmp_path):
        path = tmp_path / "recall.txt"
        Recall(path).add("uno", "dos")
        assert [p.name for p in tmp_path.iterdir()] == ["recall.txt"]

    def test_clearing_deletes_the_file(self, tmp_path):
        path = tmp_path / "recall.txt"
        r = Recall(path)
        r.add("algo privado", "sí")
        r.clear()
        assert not path.exists()
        assert len(r) == 0
        assert r.as_prompt_block("Wawi") == ""

    def test_clearing_an_absent_store_is_not_an_error(self, tmp_path):
        Recall(tmp_path / "never-written.txt").clear()


class TestDegradation:
    """A half-written store costs the memory, never the assistant."""

    def test_a_torn_last_line_keeps_everything_before_it(self, tmp_path):
        path = tmp_path / "recall.txt"
        r = Recall(path)
        r.add("uno", "dos")
        r.add("tres", "cuatro")
        with path.open("a", encoding="utf-8") as fh:
            fh.write("2026-01-01 10:00\tuna pregunta a med")

        assert [e.user for e in Recall(path).entries()] == ["uno", "tres"]

    def test_binary_garbage_degrades_to_no_memory(self, tmp_path):
        path = tmp_path / "recall.txt"
        path.write_bytes(b"\x00\x81\xfe\xff not a store at all \x00")
        r = Recall(path)
        assert r.entries() == []
        assert r.as_prompt_block("Wawi") == ""

    def test_an_unreadable_file_degrades_to_no_memory(self, tmp_path):
        path = tmp_path / "recall.txt"
        path.mkdir()  # a directory where a file was expected
        assert Recall(path).entries() == []

    def test_an_unwritable_path_does_not_raise(self, tmp_path):
        path = tmp_path / "recall.txt"
        path.mkdir()
        r = Recall(path)
        assert r.add("uno", "dos") is not None  # the turn survives the failed write


class TestDisabled:
    """An empty config value disables it, the way conversation_log does."""

    def test_no_path_means_no_store_and_no_file(self, tmp_path):
        r = Recall(None)
        assert not r.enabled
        assert r.add("uno", "dos") is None
        assert r.as_prompt_block("Wawi") == ""
        assert list(tmp_path.iterdir()) == []

    def test_zero_entries_also_disables_it(self, tmp_path):
        path = tmp_path / "recall.txt"
        r = Recall(path, max_entries=0)
        assert not r.enabled
        assert r.add("uno", "dos") is None
        assert not path.exists()

    def test_a_negative_bound_is_not_a_slice_that_keeps_everything(self, tmp_path):
        assert Recall(tmp_path / "r.txt", max_entries=-3).max_entries == 0


class TestReachesThePrompt:
    def test_what_was_stored_last_run_is_in_the_block(self, tmp_path):
        path = tmp_path / "recall.txt"
        Recall(path).add("cómo se llama el proyecto", "wawi")

        block = Recall(path).as_prompt_block("Wawi")
        assert "cómo se llama el proyecto" in block
        assert "Wawi: wawi" in block

    def test_the_block_lands_in_the_system_prompt(self, tmp_path):
        path = tmp_path / "recall.txt"
        Recall(path).add("cómo se llama el proyecto", "wawi")

        prompt = system_prompt("Wawi", recall=Recall(path).as_prompt_block("Wawi"))
        assert "cómo se llama el proyecto" in prompt

    def test_the_prompt_warns_that_it_is_old_and_possibly_misheard(self):
        """Unwarned, a misheard fragment reads to the model as an established fact."""
        es = system_prompt("Wawi", recall="Usuario: hola", language="es")
        assert "ANTERIORES" in es
        assert "errores" in es
        en = system_prompt("Wawi", recall="Usuario: hola", language="en")
        assert "PREVIOUS" in en
        assert "errors" in en

    def test_the_warning_comes_before_the_transcript(self):
        prompt = system_prompt("Wawi", recall="Usuario: algo dudoso")
        assert prompt.index("ANTERIORES") < prompt.index("algo dudoso")

    def test_an_empty_store_adds_nothing_to_the_prompt(self):
        assert system_prompt("Wawi") == system_prompt("Wawi", recall="")

    def test_the_block_is_frozen_for_the_life_of_the_process(self, tmp_path):
        """Re-rendering it per turn would reconnect the live SDK session per turn.

        The system prompt is part of ``brain.Engine``'s connection signature, so
        a block that changed after every answer would tear down the connection
        after every answer — losing the in-session conversation this exists to
        protect.
        """
        path = tmp_path / "recall.txt"
        r = Recall(path)
        r.add("de la sesión anterior", "sí")
        r = Recall(path)  # a fresh run

        before = r.as_prompt_block("Wawi")
        r.add("dicho recién", "ajá")

        assert r.as_prompt_block("Wawi") == before
        assert "dicho recién" not in r.as_prompt_block("Wawi")
        # ...but it IS on disk, so the NEXT run sees it.
        assert "dicho recién" in path.read_text(encoding="utf-8")


class TestExchangeLines:
    def test_a_comment_line_is_not_an_exchange(self):
        assert Exchange.from_line("# escribí lo que quieras acá") is None

    def test_a_blank_line_is_not_an_exchange(self):
        assert Exchange.from_line("   ") is None

    def test_a_line_with_the_wrong_field_count_is_skipped(self):
        assert Exchange.from_line("solo un campo") is None
        assert Exchange.from_line("uno\tdos\ttres\tcuatro") is None

    def test_a_round_trip_keeps_both_sides(self):
        entry = Exchange("preguntá", "respondí", "2026-01-01 10:00")
        assert Exchange.from_line(entry.to_line()) == entry
