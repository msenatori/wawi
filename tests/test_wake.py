"""Tests for transcription-based wake phrase detection.

No model, no network, no audio: the transcription call is replaced by a stub
so every assertion is about the matching and rejection logic, which is where
the real risk lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from wawi.wake import (
    WakeError,
    WhisperWakeDetector,
    looks_like_hallucination,
    match_trigger,
    normalize,
    strip_accents,
)

SR = 16000


def audio(seconds: float = 1.0) -> np.ndarray:
    return np.full(int(SR * seconds), 0.2, dtype=np.float32)


def detector_returning(text: str, phrases=("viernes", "bruno")) -> WhisperWakeDetector:
    """A detector whose transcription is fixed, so only matching is under test."""
    det = WhisperWakeDetector(phrases, sample_rate=SR)
    det._transcribe = lambda a, **kw: {"text": text}  # type: ignore[method-assign]
    return det


class TestNormalization:
    def test_strips_accents(self):
        assert strip_accents("Nébula") == "Nebula"
        assert strip_accents("qué hora es") == "que hora es"

    def test_normalize_lowercases_and_drops_punctuation(self):
        assert normalize("¡Viernes, qué hora es?") == "viernes que hora es"

    def test_normalize_collapses_whitespace(self):
        assert normalize("  bruno   \n dale  ") == "bruno dale"


class TestHallucinationGuard:
    @pytest.mark.parametrize(
        "text",
        [
            "end end end end end end",
            "buesta buesta buesta buesta",
            "bu esta bu esta bu esta bu esta",
        ],
    )
    def test_detects_token_loops(self, text):
        """The real whisper-tiny failure mode on short or ambiguous audio."""
        assert looks_like_hallucination(text)

    @pytest.mark.parametrize(
        "text",
        [
            "viernes que hora es",
            "bruno abri la terminal por favor",
            "no no gracias",  # a repeat, but only twice
        ],
    )
    def test_leaves_real_speech_alone(self, text):
        assert not looks_like_hallucination(text)

    def test_repeating_the_wake_word_still_wakes_it(self):
        """Observed in real use: people repeat the name when ignored.

        The loop guard used to eat this, so the assistant was deafest exactly
        when the user was most frustrated. A repeat of the PHRASE is a summon;
        a repeat of anything else is still a hallucination.
        """
        assert match_trigger("Bruno, Bruno, Bruno, Bruno.", ["bruno"]) == "bruno"
        assert match_trigger("viernes viernes viernes viernes", ["viernes"]) == "viernes"
        # The window cut the last one in half (seen live, 2026-10-16 10:07).
        assert match_trigger("Wawi, Wawi, Wawi, Wawi, Wawi, W", ["wawi"]) == "wawi"

    def test_a_loop_of_something_else_is_still_rejected(self):
        assert match_trigger("bueno bueno bueno bueno bueno", ["bruno"]) is None
        assert match_trigger("end end end end end", ["bruno"]) is None


class TestMatching:
    def test_exact_match_at_the_start(self):
        assert match_trigger("Viernes, qué hora es?", ["viernes"]) == "viernes"

    def test_accents_do_not_decide_the_match(self):
        """Whisper returns the same word with and without accents."""
        assert match_trigger("nebula que hora es", ["nébula"]) == "nébula"
        assert match_trigger("Nébula que hora es", ["nebula"]) == "nebula"

    def test_tolerates_a_small_transcription_slip(self):
        assert match_trigger("viernez que hora es", ["viernes"]) == "viernes"
        # Observed live: whisper returned "bruna" for "Bruno".
        assert match_trigger("bruna podemos ver el proyecto", ["bruno"]) == "bruno"
        # A transposition is one slip to a transcriber, not two.
        assert match_trigger("burno dale", ["bruno"]) == "bruno"

    @pytest.mark.parametrize(
        "said",
        ["bueno dale", "uno dos tres", "trueno en el patio", "bono de guerra"],
    )
    def test_common_words_never_wake_it(self, said):
        """"bueno" scores 0.800 against "bruno" — identical to the real slip
        "bruna". A similarity RATIO cannot separate them; edit distance can,
        and "bueno" being the commonest filler in Spanish makes this the
        difference between usable and unusable."""
        assert match_trigger(said, ["bruno"]) is None

    def test_rejects_an_unrelated_word(self):
        assert match_trigger("martes que hora es", ["viernes"]) is None

    def test_any_configured_phrase_matches(self):
        for said, expected in [("bruno dale", "bruno"), ("viernes dale", "viernes")]:
            assert match_trigger(said, ["viernes", "bruno"]) == expected

    def test_trigger_must_be_near_the_start(self):
        """Saying the word mid-sentence must not wake Wawi."""
        late = "ayer le conte a bruno que el proyecto estaba listo"
        assert match_trigger(late, ["bruno"]) is None

    def test_a_leading_filler_word_still_matches(self):
        assert match_trigger("eh, viernes, abrime la terminal", ["viernes"]) == "viernes"

    def test_multi_word_phrases_must_be_contiguous(self):
        assert match_trigger("che viernes dale", ["che viernes"]) == "che viernes"
        assert match_trigger("che dale viernes", ["che viernes"]) is None

    def test_empty_transcript_matches_nothing(self):
        assert match_trigger("", ["viernes"]) is None
        assert match_trigger("   ...  ", ["viernes"]) is None


class TestDetector:
    def test_requires_at_least_one_phrase(self):
        with pytest.raises(WakeError, match="No wake phrases"):
            WhisperWakeDetector([])
        with pytest.raises(WakeError):
            WhisperWakeDetector(["", "   "])

    def test_blank_phrases_are_dropped_not_kept(self):
        det = WhisperWakeDetector(["viernes", "", "  ", "bruno"])
        assert det.phrases == ["viernes", "bruno"]

    def test_check_reports_the_matched_phrase(self):
        det = detector_returning("Viernes, qué hora es?")
        result = det.check(audio(1.0))
        assert result.ok
        assert result.matched == "viernes"
        assert result.text == "Viernes, qué hora es?"

    def test_check_reports_no_match_without_the_trigger(self):
        result = detector_returning("pasame la sal").check(audio(1.0))
        assert not result.ok
        assert result.matched is None

    def test_very_short_segments_are_skipped_without_transcribing(self):
        """Where whisper-tiny hallucinates most, and never a real trigger."""
        calls = []
        det = WhisperWakeDetector(["viernes"], sample_rate=SR)
        det._transcribe = lambda a, **kw: calls.append(a) or {"text": "viernes"}

        assert not det.check(audio(0.1)).ok
        assert calls == []

    def test_a_transcription_crash_is_not_fatal(self):
        det = WhisperWakeDetector(["viernes"], sample_rate=SR)

        def boom(a, **kw):
            raise RuntimeError("model exploded")

        det._transcribe = boom
        result = det.check(audio(1.0))
        assert not result.ok  # reported as no match, not raised

    def test_check_before_load_does_not_crash(self):
        det = WhisperWakeDetector(["viernes"], sample_rate=SR)
        assert not det.check(audio(1.0)).ok

    def test_missing_mlx_whisper_gives_a_pip_hint(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mlx_whisper":
                raise ImportError("no mlx_whisper here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(WakeError, match="pip install mlx-whisper"):
            WhisperWakeDetector(["viernes"]).load()
