"""The post-Whisper junk filter (wawi/speechgate.py). Pure functions only."""

from __future__ import annotations

import numpy as np
import pytest

from wawi import lang
from wawi.speechgate import (
    FilterConfig,
    SpeechEvidence,
    SttMetrics,
    judge,
    normalize,
)

PROMPT = lang.stt_prompt("es", "Wawi")

# Numbers measured on this machine with whisper-large-v3-turbo (see the
# speechgate docstring): real speech vs the same model fed noise.
CONFIDENT = SttMetrics(avg_logprob=-0.25, no_speech_prob=0.0, compression_ratio=0.6, segments=1)
NOISE_LIKE = SttMetrics(avg_logprob=-0.55, no_speech_prob=0.0, compression_ratio=0.6, segments=1)
STRONG = SpeechEvidence(voiced_ms=1200, active_ms=1500)
WEAK = SpeechEvidence(voiced_ms=270, active_ms=600)


class TestNormalize:
    def test_folds_case_accents_and_punctuation(self):
        assert normalize("¡Suscríbete al canal!") == "suscribete al canal"
        assert normalize("  ¿Está   bien? ") == "esta bien"

    def test_empty(self):
        assert normalize("") == ""
        assert normalize("...") == ""


class TestAlwaysJunk:
    @pytest.mark.parametrize("text", [
        "¡Suscríbete al canal!",
        "Gracias por ver el video.",
        "Subtítulos realizados por la comunidad de Amara.org",
        "Thanks for watching!",
    ])
    def test_subtitle_credits_are_dropped_whatever_the_evidence(self, text):
        assert judge(text, CONFIDENT, STRONG, PROMPT) is not None


class TestWeakJunk:
    def test_gracias_from_noise_is_dropped(self):
        # The log's most common phantom turn: "Gracias." at rms 0.003.
        reason = judge("Gracias.", NOISE_LIKE, WEAK, PROMPT)
        assert reason is not None and "weak evidence" in reason

    def test_gracias_said_clearly_is_kept(self):
        assert judge("Gracias.", CONFIDENT, STRONG, PROMPT) is None

    def test_little_voiced_speech_alone_is_enough_to_drop(self):
        assert judge("Thank you.", CONFIDENT, WEAK, PROMPT) is not None

    def test_low_confidence_alone_is_enough_to_drop(self):
        assert judge("Amén.", NOISE_LIKE, STRONG, PROMPT) is not None


class TestPromptEcho:
    def test_the_prompt_parroted_back_from_noise_is_dropped(self):
        # "¿Está bien así?" used to be in the seed and Whisper returned
        # "¿Está bien?" for silence, white noise, clicks and tones alike. It is
        # out of the seed now; the weak-junk rule still drops the phrase.
        reason = judge("¿Está bien?", NOISE_LIKE, WEAK, PROMPT)
        assert reason is not None

    def test_the_same_words_with_strong_evidence_are_kept(self):
        assert judge("¿Está bien?", CONFIDENT, STRONG, PROMPT) is None

    def test_a_longer_piece_of_the_prompt_is_also_an_echo(self):
        assert judge("Abrí el archivo y revisá el código.", NOISE_LIKE, WEAK, PROMPT)

    def test_a_real_sentence_is_not_an_echo(self):
        assert judge("Abrí el navegador.", NOISE_LIKE, WEAK, PROMPT) is None

    def test_partial_words_do_not_count_as_an_echo(self):
        assert judge("Revi", NOISE_LIKE, WEAK, PROMPT) is None


class TestProtectedReplies:
    @pytest.mark.parametrize("text", ["Sí.", "No.", "Dale.", "Sí, dale.", "Ok.", "Listo."])
    def test_short_genuine_replies_survive_weak_evidence(self, text):
        # Several of these are IN the prompt; they are still real answers.
        assert judge(text, NOISE_LIKE, WEAK, PROMPT) is None

    def test_but_not_garbage_confidence(self):
        awful = SttMetrics(avg_logprob=-1.6, no_speech_prob=0.0, compression_ratio=0.5, segments=1)
        assert judge("Sí.", awful, STRONG, PROMPT) is not None


class TestWhisperConfidence:
    def test_looping_decoder_is_dropped(self):
        loop = SttMetrics(avg_logprob=-0.2, no_speech_prob=0.0, compression_ratio=3.1, segments=1)
        reason = judge("ok ok ok ok ok ok ok ok ok ok ok ok", loop, STRONG, PROMPT)
        assert reason and "compression_ratio" in reason

    def test_very_low_logprob_is_dropped(self):
        low = SttMetrics(avg_logprob=-1.3, no_speech_prob=0.0, compression_ratio=1.0, segments=1)
        reason = judge("vamos a la playa mañana", low, STRONG, PROMPT)
        assert reason and "avg_logprob" in reason

    def test_no_speech_needs_a_mediocre_logprob_too(self):
        silent = SttMetrics(avg_logprob=-0.7, no_speech_prob=0.9, compression_ratio=1.0, segments=1)
        assert judge("vamos a la playa", silent, STRONG, PROMPT)
        sure = SttMetrics(avg_logprob=-0.2, no_speech_prob=0.9, compression_ratio=1.0, segments=1)
        assert judge("vamos a la playa", sure, STRONG, PROMPT) is None

    def test_normal_sentence_is_kept(self):
        assert judge("¿Qué tiempo hace hoy?", CONFIDENT, STRONG, PROMPT) is None


class TestFallbacks:
    def test_nothing_measured_keeps_the_old_behaviour(self):
        # Fake STTs in tests and the fallback recorder: no metrics, no VAD.
        assert judge("Gracias.", None, None, PROMPT) is None
        assert judge("¡Suscríbete al canal!", None, None, PROMPT) is not None

    def test_push_to_talk_is_judged_on_confidence_only(self):
        forced = SpeechEvidence(voiced_ms=0, active_ms=0, forced=True)
        assert judge("Gracias.", CONFIDENT, forced, PROMPT) is None
        assert judge("Gracias.", NOISE_LIKE, forced, PROMPT) is not None

    def test_the_master_switch(self):
        assert judge("¡Suscríbete al canal!", NOISE_LIKE, WEAK, PROMPT,
                     FilterConfig(enabled=False)) is None

    def test_empty_text_is_not_judged(self):
        assert judge("", NOISE_LIKE, WEAK, PROMPT) is None


class TestSttMetrics:
    def test_aggregates_weighted_by_duration(self):
        m = SttMetrics.from_segments([
            {"start": 0.0, "end": 1.0, "avg_logprob": -0.2, "no_speech_prob": 0.1,
             "compression_ratio": 1.0},
            {"start": 1.0, "end": 4.0, "avg_logprob": -0.6, "no_speech_prob": 0.3,
             "compression_ratio": 2.0},
        ])
        assert m is not None
        assert m.avg_logprob == pytest.approx(-0.5)
        assert m.no_speech_prob == pytest.approx(0.3)
        assert m.compression_ratio == pytest.approx(2.0)
        assert m.segments == 2

    def test_unusable_segments_give_none(self):
        assert SttMetrics.from_segments([]) is None
        assert SttMetrics.from_segments([{"text": "hola"}]) is None


class TestSpeechEvidence:
    def test_ratio(self):
        assert SpeechEvidence(300, 600).ratio == pytest.approx(0.5)
        assert SpeechEvidence(0, 0).ratio == 0.0


class TestSttDetailed:
    def test_metrics_come_back_with_the_text(self, monkeypatch):
        from wawi.stt import STT

        class _FakeModule:
            @staticmethod
            def transcribe(audio, **kwargs):
                assert type(audio) is np.ndarray  # never the listener subclass
                return {"text": " Gracias. ", "segments": [
                    {"start": 0.0, "end": 2.0, "avg_logprob": -0.5,
                     "no_speech_prob": 0.0, "compression_ratio": 0.5},
                ]}

        monkeypatch.setitem(__import__("sys").modules, "mlx_whisper", _FakeModule)
        from wawi.listener import CapturedAudio

        audio = np.zeros(16, dtype=np.float32).view(CapturedAudio)
        text, metrics = STT().transcribe_detailed(audio, "es")
        assert text == "Gracias."
        assert metrics is not None and metrics.avg_logprob == pytest.approx(-0.5)
        assert STT().transcribe(audio, "es") == "Gracias."
