"""The decoder seed, and the bug that made it necessary.

Whisper continues the STYLE of the text it is seeded with. Left unseeded it
transcribed the accented "si" as bare "si" -- which in Spanish is the word
"if", not "yes". That is not a cosmetic difference here: the approval parser
in :mod:`wawi.lang` reads the opening words of a spoken answer to decide
whether the executor may run a shell command, and "mira si te parece" reaching
it as an unaccented "si" is a denial arriving as consent.

These tests pin the seed itself, the wiring that carries it to mlx-whisper,
and the one property that makes it worth having: that it is re-read per turn,
because the HUD can switch language mid-session.
"""

from __future__ import annotations

import numpy as np
import pytest

from wawi import lang
from wawi.stt import STT


class TestSeedText:
    def test_every_supported_language_has_a_seed(self):
        # A missing entry falls back to Spanish, which would steer an English
        # turn toward Spanish orthography -- worse than sending no seed at all.
        for code in lang.SUPPORTED:
            assert lang.STT_PROMPT[code].strip()

    def test_spanish_seed_shows_the_accent_it_exists_to_teach(self):
        # The whole point. Without a written "sí" in the seed the decoder has
        # no reason to produce one, and affirmative() sees "si" = "if".
        assert "sí" in lang.STT_PROMPT["es"].lower()

    def test_spanish_seed_survives_the_approval_parser_as_a_yes(self):
        # Guards the round trip rather than the spelling: whatever the seed
        # teaches must still read as consent on the way back in.
        assert lang.affirmative("sí, dale") is True
        assert lang.affirmative("mirá si te parece") is not True

    def test_seed_contains_no_question_whisper_could_echo(self):
        # Whisper falls back to its seed on silence. "¿Está bien así?" in the
        # seed came back as "¿Está bien?" -- a turn nobody said.
        for code in lang.SUPPORTED:
            assert "?" not in lang.STT_PROMPT[code]
            assert "bien así" not in lang.STT_PROMPT[code].lower()

    def test_assistant_name_is_appended_when_given(self):
        # A wake word the decoder has never seen comes back as whatever it
        # sounds like, and every downstream matcher then misses it.
        assert lang.stt_prompt("es", "Wawi").endswith("Wawi.")

    def test_no_name_leaves_the_seed_untouched(self):
        assert lang.stt_prompt("en") == lang.STT_PROMPT["en"]

    def test_blank_name_does_not_leave_a_dangling_period(self):
        assert not lang.stt_prompt("es", "   ").endswith(" .")

    def test_unknown_language_falls_back_instead_of_raising(self):
        # Mid-turn KeyError here would be spoken as a connection error.
        assert lang.stt_prompt("de") == lang.STT_PROMPT[lang.DEFAULT]


class TestSttWiring:
    @staticmethod
    def _spy(monkeypatch):
        """Replace mlx_whisper.transcribe and capture the kwargs it receives."""
        seen: dict = {}

        class _FakeModule:
            @staticmethod
            def transcribe(audio, **kwargs):
                seen.update(kwargs)
                return {"text": " ok "}

        monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", _FakeModule)
        return seen

    def test_seed_reaches_mlx_whisper(self, monkeypatch):
        seen = self._spy(monkeypatch)
        STT().transcribe(np.zeros(16, dtype=np.float32), "es", "Sí, dale.")
        assert seen["initial_prompt"] == "Sí, dale."

    def test_no_seed_omits_the_argument_entirely(self, monkeypatch):
        # Not passed rather than passed empty: an empty prompt is not the same
        # as no prompt, and an older mlx-whisper need never see the argument.
        seen = self._spy(monkeypatch)
        STT().transcribe(np.zeros(16, dtype=np.float32), "es")
        assert "initial_prompt" not in seen

    def test_empty_seed_is_treated_as_no_seed(self, monkeypatch):
        seen = self._spy(monkeypatch)
        STT().transcribe(np.zeros(16, dtype=np.float32), "es", "")
        assert "initial_prompt" not in seen

    def test_language_is_still_forwarded(self, monkeypatch):
        # Regression guard: the seed was added alongside language, not instead.
        seen = self._spy(monkeypatch)
        STT().transcribe(np.zeros(16, dtype=np.float32), "en", "Yes.")
        assert seen["language"] == "en"

    def test_result_is_still_stripped(self, monkeypatch):
        self._spy(monkeypatch)
        assert STT().transcribe(np.zeros(16, dtype=np.float32)) == "ok"


class TestOrchestratorSeedSelection:
    """Which seed a turn actually uses, and why it is decided per turn."""

    @staticmethod
    def _orch(language="es", override="", name="Wawi"):
        from wawi.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch._language = language
        orch._cfg = type(
            "Cfg", (), {"stt_initial_prompt": override, "assistant_name": name}
        )()
        return orch

    def test_it_uses_the_seed_for_the_active_language(self):
        assert self._orch("en")._stt_prompt() == lang.stt_prompt("en", "Wawi")

    def test_the_language_toggle_changes_the_seed_mid_session(self):
        # Read per turn, not cached at startup. A Spanish seed over English
        # audio steers the decoder toward the language you stopped speaking.
        orch = self._orch("es")
        spanish = orch._stt_prompt()
        orch._language = "en"
        assert orch._stt_prompt() != spanish

    def test_a_config_override_wins(self):
        assert self._orch(override="solo esto")._stt_prompt() == "solo esto"

    def test_a_whitespace_override_is_not_an_override(self):
        # An override of spaces would silently disable the accent seeding that
        # the approval parser depends on.
        assert self._orch(override="   ")._stt_prompt() == lang.stt_prompt("es", "Wawi")
