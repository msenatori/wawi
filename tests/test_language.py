"""The language switch, and the four things it has to move at once.

Switching language is the kind of feature that half-works convincingly: change
one setting and the assistant still answers, just wrongly — English words in a
Spanish voice, or a Spanish prompt over English audio. Each test here pins one
of the pieces that must travel together, because any one of them left behind
produces output that sounds like a bug in something else entirely.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from wawi import config, lang
from wawi.prompts import system_prompt, system_prompt_no_tools


class TestLangTable:
    def test_every_supported_language_has_every_string(self):
        # A missing entry would be a KeyError mid-turn, spoken as "no pude
        # conectarme" — i.e. it would look like a network failure.
        tables = (
            lang.VOICE,
            lang.ERROR_TEXT,
            lang.NO_AUDIO_TEXT,
            lang.NO_AUDIO_NOTE,
            lang.DEGRADED_SUFFIX,
            lang.TTS_LANG,
        )
        for table in tables:
            for code in lang.SUPPORTED:
                assert code in table, (table, code)

    def test_an_unknown_language_degrades_instead_of_raising(self):
        assert lang.normalise("pt") == lang.DEFAULT
        assert lang.normalise(None) == lang.DEFAULT
        assert lang.get(lang.ERROR_TEXT, "klingon") == lang.ERROR_TEXT[lang.DEFAULT]

    def test_each_language_gets_a_voice_of_its_own(self):
        # ef_dora reading English is the loudest possible half-switch.
        assert lang.voice_for("es") != lang.voice_for("en")


class TestPrompt:
    def test_the_prompt_orders_the_right_language(self):
        assert "READ ALOUD" in system_prompt("Wawi", language="en")
        assert "EN VOZ ALTA" in system_prompt("Wawi", language="es")

    def test_english_prompt_carries_no_spanish(self):
        prompt = system_prompt("Wawi", language="en")
        for spanish in ("español", "Respondé", "Contestá", "usuario."):
            assert spanish not in prompt, spanish

    def test_both_languages_keep_the_path_rule(self):
        # capture_screen exists in every mode and every language, so the rule
        # that keeps an absolute path off the speaker must too.
        assert "never say them out loud" in system_prompt("Wawi", language="en")
        assert "nunca los digas en voz alta" in system_prompt("Wawi", language="es")

    def test_both_languages_ask_for_the_shortest_answer(self):
        assert "MINIMUM" in system_prompt("Wawi", language="en")
        assert "MÍNIMO" in system_prompt("Wawi", language="es")

    def test_the_no_tools_prompt_follows_too(self):
        assert "no tools" in system_prompt_no_tools("Wawi", language="en")
        assert "herramientas" in system_prompt_no_tools("Wawi", language="es")


class TestConfig:
    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(textwrap.dedent(body))
        return path

    def test_auto_voice_follows_the_language(self, tmp_path):
        es = config.load(self._write(tmp_path, "language: es\ntts:\n  voice: auto\n"))
        assert es.tts_voice == lang.voice_for("es")

        path = tmp_path / "en.yaml"
        path.write_text("language: en\ntts:\n  voice: auto\n")
        assert config.load(path).tts_voice == lang.voice_for("en")

    def test_an_explicit_voice_always_wins(self, tmp_path):
        # Someone who wrote em_alex meant it.
        path = self._write(tmp_path, "language: en\ntts:\n  voice: em_alex\n")
        assert config.load(path).tts_voice == "em_alex"

    def test_an_unspeakable_language_is_refused_at_load(self, tmp_path):
        # Whisper understands Portuguese; the assistant has no prompt, voice or
        # spoken strings for it. Failing here beats answering Spanish to a
        # Portuguese question.
        path = self._write(tmp_path, "language: pt\n")
        with pytest.raises(config.ConfigError) as exc:
            config.load(path)
        assert "pt" in str(exc.value)


class TestRuntimeSwitch:
    """The ES/EN pill changes the language mid-session.

    The risk here is a HALF switch: the STT still told "es" while the voice is
    English, or the prompt switched but the voice did not. Each test pins one
    piece to the same source of truth.
    """

    def test_the_voice_switches_without_reloading_the_model(self, monkeypatch):
        # Kokoro takes voice and lang as CALL arguments, which is what makes
        # this cheap enough to do between turns. If that ever changes, this
        # test fails rather than the panel silently taking a second to respond.
        from wawi.voice import Voice

        loads = []

        class _FakeKokoro:
            def __init__(self, *a):
                loads.append(1)

        monkeypatch.setattr("kokoro_onnx.Kokoro", _FakeKokoro)
        voice = Voice(Path("/nowhere"), "ef_dora")
        assert len(loads) == 1

        voice.set_language("en")
        assert voice._voice == lang.voice_for("en")
        assert voice._kokoro_lang == lang.TTS_LANG["en"]
        assert len(loads) == 1  # nothing was reloaded

        voice.set_language("es")
        assert voice._voice == lang.voice_for("es")
        assert voice._kokoro_lang == lang.TTS_LANG["es"]
        assert len(loads) == 1

    def test_an_unknown_code_is_normalised_not_stored_raw(self):
        from wawi.voice import Voice

        voice = Voice.__new__(Voice)
        voice.set_language("klingon")
        assert voice._voice == lang.voice_for(lang.DEFAULT)

    def test_the_kokoro_code_is_not_our_code(self):
        # "en-us" vs "en": passing ours to Kokoro does not fail, it reads
        # English with the wrong phonetics — a silent, audible bug.
        assert lang.TTS_LANG["en"] != "en"
        assert set(lang.TTS_LANG) == set(lang.SUPPORTED)


class TestSpokenYesNo:
    """The yes/no parser gates every command the executor wants to run.

    It is the one place in Wawi where being wrong in the permissive
    direction is expensive, so what is pinned here is mostly the refusals.
    """

    @pytest.mark.parametrize(
        "answer",
        ["sí", "si", "dale", "dale, hacelo", "ok", "yes", "sure", "obvio", "Sí."],
    )
    def test_an_affirmative_is_read_as_yes(self, answer):
        assert lang.affirmative(answer) is True

    @pytest.mark.parametrize(
        "answer", ["no", "no, pará", "nope", "ni loco no", "cancelá", "stop"]
    )
    def test_a_negative_is_read_as_no(self, answer):
        assert lang.affirmative(answer) is False

    @pytest.mark.parametrize(
        "answer", ["", "   ", "fijate vos", "en el proyecto de tienda", "¿qué?"]
    )
    def test_anything_else_is_neither(self, answer):
        # None, not False: the caller must know it did not get an answer, so it
        # can ask again instead of silently treating a shrug as a refusal.
        assert lang.affirmative(answer) is None

    def test_a_yes_buried_mid_sentence_is_not_an_answer(self):
        """"no, en el proyecto sí está" is not approval of anything."""
        assert lang.affirmative("no, en el proyecto viejo sí está") is False
        assert lang.affirmative("mirá si te parece que sí conviene") is None

    def test_both_languages_are_accepted_at_once(self):
        # People answering a machine out loud code-switch constantly, and the
        # cost of missing a yes is only a repeated question.
        assert lang.affirmative("yes") is True
        assert lang.affirmative("sí") is True
