"""Tests for the speech funnel and its incremental splitter.

Kokoro and sounddevice are never imported: ``Voice`` is built without running
``__init__`` (which loads a 310 MB model and opens an audio device) and the two
things it calls out to — synthesis and playback — are replaced. What is under
test is the policy around them: which text is spoken, in what pieces, and what
happens after a barge-in.
"""

from __future__ import annotations

import threading

import pytest

from wawi import voice as voice_mod
from wawi.voice import Voice, take_complete_sentences


class FakeVoice(Voice):
    """A Voice that records what it was asked to speak instead of speaking it."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.spoken: list[str] = []
        self._abort = threading.Event()
        self._fail_on = fail_on

    def _synth_and_play(self, text: str) -> None:  # type: ignore[override]
        if self._fail_on is not None and self._fail_on in text:
            raise RuntimeError("Kokoro fell over")
        self.spoken.append(text)


class TestIncrementalSplitting:
    """One splitter for both paths, so the streamed pieces match the whole."""

    def test_a_finished_sentence_is_separated_from_the_rest(self):
        finished, rest = take_complete_sentences("Uno. Dos. Tr")
        assert finished == ["Uno.", "Dos."]
        assert rest == "Tr"

    def test_a_sentence_ends_only_once_whitespace_follows_it(self):
        """Otherwise "Son las 3." is spoken before "3.15" turns out to be one
        number."""
        finished, rest = take_complete_sentences("Son las 3.")
        assert finished == []
        assert rest == "Son las 3."

    def test_nothing_finished_yet(self):
        assert take_complete_sentences("Estoy escri") == ([], "Estoy escri")

    def test_the_pieces_always_add_back_up(self):
        text = "Uno. Dos! ¿Tres? Cuatro… Cin"
        finished, rest = take_complete_sentences(text)
        assert " ".join([*finished, rest]) == text

    def test_the_whole_splitter_agrees_with_the_incremental_one(self):
        text = "Uno. Dos! ¿Tres? Cuatro."
        finished, rest = take_complete_sentences(text)
        expected = [*finished, rest.strip()] if rest.strip() else finished
        assert voice_mod._split_sentences(text) == expected

    def test_text_without_punctuation_is_still_one_sentence(self):
        assert voice_mod._split_sentences("hola qué tal") == ["hola qué tal"]


class TestSayPart:
    """Streaming speech: same funnel, one sentence at a time."""

    def test_each_piece_is_spoken_in_order(self):
        v = FakeVoice()
        assert v.say_part("Hay tres agentes.", first=True) is True
        assert v.say_part("Dos terminaron.", first=False) is True
        assert v.spoken == ["Hay tres agentes.", "Dos terminaron."]

    def test_a_barge_in_stops_the_rest_of_the_answer(self):
        v = FakeVoice()
        v.say_part("Hay tres agentes.", first=True)
        v._abort.set()  # what interrupt() does, without sounddevice

        assert v.say_part("Dos terminaron.", first=False) is False
        assert v.spoken == ["Hay tres agentes."]

    def test_only_the_first_piece_may_clear_a_barge_in(self):
        """Otherwise every sentence undoes the interruption of the one before
        it, and a cut-off answer stops for a moment and then keeps talking."""
        v = FakeVoice()
        v._abort.set()

        assert v.say_part("Sigo hablando.", first=False) is False
        assert v.spoken == []

        assert v.say_part("Turno nuevo.", first=True) is True
        assert v.spoken == ["Turno nuevo."]

    def test_a_failed_piece_never_raises(self):
        """Voice.say has never propagated an exception; nor does this."""
        v = FakeVoice(fail_on="rota")
        assert v.say_part("Esta está rota.", first=True) is True
        assert v.say_part("Esta no.", first=False) is True
        assert v.spoken == ["Esta no."]

    def test_a_piece_with_nothing_speakable_is_silent(self):
        """The "I left the code on screen" line belongs to a whole answer made
        of code, not to one sentence of a good one."""
        v = FakeVoice()
        v.say_part("42.", first=True)
        assert v.spoken == []

    def test_say_still_announces_an_answer_that_was_all_code(self):
        v = FakeVoice()
        v.say("```python\nx = 1\n```")
        assert v.spoken == ["Te dejé el código en pantalla."]

    def test_say_clears_a_stale_barge_in(self):
        v = FakeVoice()
        v._abort.set()
        v.say("Turno nuevo.")
        assert v.spoken == ["Turno nuevo."]


class TestSpeechLockIsShared:
    def test_both_entry_points_take_the_one_voice_lock(self):
        """Two utterances overlapping is the failure this lock exists for."""
        v = FakeVoice()
        with voice_mod.VOICE_LOCK:
            done = threading.Event()

            def _speak() -> None:
                v.say_part("Hola.", first=True)
                done.set()

            threading.Thread(target=_speak, daemon=True).start()
            assert not done.wait(0.2)
            assert v.spoken == []

        assert done.wait(2.0)
        assert v.spoken == ["Hola."]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("**Listo**", "Listo"),
        ("Mirá `el archivo`", "Mirá el archivo"),
        ("Hecho 🎉", "Hecho"),
    ],
)
def test_cleaning_is_unchanged_for_streamed_pieces(raw, expected):
    assert voice_mod._clean_for_tts(raw, code_fallback=False) == expected


class TestOutputLevel:
    """What Wawi bobs to while she talks: the level of the sentence playing."""

    def test_envelope_follows_loudness_and_is_scaled_to_one(self):
        import numpy as np

        sr = 1000
        loud = np.full(200, 0.5, dtype=np.float32)
        soft = np.full(200, 0.05, dtype=np.float32)
        env = voice_mod.speech_envelope(np.concatenate([soft, loud, soft]), sr)
        assert len(env) == 30  # 20 ms frames over 0.6 s
        assert env.max() == pytest.approx(1.0)
        assert env[15] > 5 * env[2]

    def test_silence_and_tiny_input_are_flat(self):
        import numpy as np

        assert not voice_mod.speech_envelope(np.zeros(1000, dtype=np.float32), 1000).any()
        assert len(voice_mod.speech_envelope(np.zeros(3, dtype=np.float32), 1000)) == 0

    def test_level_is_none_when_nothing_plays(self):
        assert FakeVoice().output_level() is None

    def test_level_indexes_the_playing_sentence_by_time(self, monkeypatch):
        import numpy as np

        v = FakeVoice()
        v._playing = (np.array([0.1, 0.9, 0.4], dtype=np.float32), 100.0)
        monkeypatch.setattr(voice_mod.time, "monotonic", lambda: 100.025)
        assert v.output_level() == pytest.approx(0.9)
        monkeypatch.setattr(voice_mod.time, "monotonic", lambda: 101.0)
        assert v.output_level() is None  # past the end: silent


class TestWarm:
    def _voice(self):
        from wawi.voice import Voice

        v = object.__new__(Voice)
        v.calls = []
        v._synth_raw = lambda text: v.calls.append(text)
        return v

    def test_warms_a_cold_model_once(self):
        import time as _t

        v = self._voice()
        v.warm()
        v.warm()  # second call inside the window: nothing to do
        for _ in range(50):
            if v.calls:
                break
            _t.sleep(0.01)
        assert v.calls == ["Sí."]

    def test_skips_a_warm_model(self):
        import time as _t

        v = self._voice()
        v._last_synth = _t.monotonic()
        v.warm()
        _t.sleep(0.05)
        assert v.calls == []
