"""Transcription-based wake phrase detection.

The alternative to a trained wake-word model. Instead of a neural net that
recognises one fixed English phrase, this transcribes a segment of speech with
a small Whisper model and checks whether the text starts with one of YOUR
trigger phrases.

Why this exists
---------------
``openwakeword`` ships six pretrained phrases, all trained on English speech.
Measured on this project, ``hey_jarvis`` scores 0.9989 spoken by an English
voice and 0.0014 spoken with Spanish phonetics -- unusable. Only ``alexa``
survives a Spanish accent. Training a custom model is possible but expensive:
it needs torch, several GB of background/RIR datasets, ~11 hours of
false-positive validation audio, and 50k training steps on a GPU.

Transcription sidesteps all of it. Whisper already runs in this project, it is
multilingual, and it needs no training: the trigger is a string in
``config.yaml``. Change the phrase by editing one line.

The cost, honestly
------------------
A dedicated wake model costs ~1.4 ms per 80 ms chunk and runs continuously.
This costs ~40 ms per *speech segment* and runs only when the VAD says someone
is talking. In silence it costs nothing at all. On Apple Silicon with
``whisper-tiny`` that is a comfortable trade; on a slower machine, or with a
larger model, it is not. Measure before switching.

The failure mode that matters
-----------------------------
Small Whisper models hallucinate on short or ambiguous audio, and they do it in
a very specific way: they emit a token loop, e.g. ``"end end end end end…"`` or
``"buesta buesta bu…"``. Measured here, a clean one-word trigger transcribes in
~40 ms while a hallucinating one takes 500-850 ms -- the latency gives it away
as reliably as the text does. :func:`looks_like_hallucination` rejects those
transcripts -- with one carve-out learned from real use: a loop made of the
wake phrase itself ("Bruno, Bruno, Bruno") is a person repeating themselves
because they are being ignored, and rejecting it makes the assistant deafest
exactly when the user is most frustrated. See :func:`_is_repeated_summon`.

Choosing a trigger phrase
-------------------------
Not every word survives transcription. Verified across three synthetic voices,
"viernes", "bruno", "mariano" and "cosmos" came back verbatim every time, while
"edith", "astro" and "nico" collapsed into token loops. Prefer a word that is
natural in the speaker's own language, two syllables or more, and not a
homophone of a common word. Verify a candidate with ``scripts/diag_wake.py``
rather than trusting intuition.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Default trigger transcription model. Tiny is enough to recognise a known
# short word and is ~3x faster than base; the full-accuracy transcription of
# the actual request still happens downstream with the large model.
DEFAULT_WAKE_MODEL = "mlx-community/whisper-tiny-mlx"

# A trigger must appear within this many words of the start. Prevents "…and
# then I told Bruno about it" mid-sentence from waking Wawi.
_TRIGGER_WINDOW_WORDS = 3

# A token repeated at least this many times consecutively marks the transcript
# as a hallucination loop rather than speech.
_LOOP_REPEATS = 4

# Edit distance tolerated on a trigger word, by word length.
#
# Ratio-based similarity CANNOT do this job. Measured on the real phrase
# "bruno": the genuine mis-transcription "bruna" scores 0.800 and so does
# "bueno" -- the single most common filler word in Spanish. Any ratio
# threshold either rejects the real slip or wakes on every "bueno".
#
# Edit distance separates them cleanly: "bruna" is 1 edit away, "bueno" is 2.
# Transpositions count as one ("burno"), because that is what a transcriber
# actually produces.
_MAX_EDITS_SHORT = 1     # words of _MIN_FUZZY_LEN..7 characters
_MAX_EDITS_LONG = 2      # words of 8+ characters, where 2 edits stay distinctive
_MIN_FUZZY_LEN = 5       # below this, one edit is most of the word

# Segments shorter than this cannot contain a real trigger and are exactly
# where small Whisper models hallucinate most. Skipped without transcribing.
_MIN_SEGMENT_S = 0.35


class WakeError(RuntimeError):
    """Raised for a missing dependency or an unusable configuration."""


def strip_accents(text: str) -> str:
    """Return *text* with diacritics removed (``vérnes`` -> ``vernes``).

    Whisper is inconsistent about accents on short words -- "Nébula" comes back
    as both "nébula" and "nebula" from the same phrase in different voices --
    so accents must never decide whether the trigger matched.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize(text: str) -> str:
    """Lowercase, drop accents and punctuation, collapse whitespace."""
    cleaned = strip_accents(text.lower())
    cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def looks_like_hallucination(text: str) -> bool:
    """True when *text* is a Whisper token loop rather than speech.

    Small models fail by repeating one token many times. A real request never
    repeats the same word four times in a row, so this is safe to reject
    outright.
    """
    words = normalize(text).split()
    if len(words) < _LOOP_REPEATS:
        return False

    run = 1
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if cur == prev else 1
        if run >= _LOOP_REPEATS:
            return True

    # A loop can also alternate ("bu esta bu esta bu esta"). Very low variety
    # across many words means the same thing.
    if len(words) >= 8 and len(set(words)) <= 2:
        return True
    return False


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Damerau-Levenshtein distance, capped at *limit* + 1.

    Capped because the answer is only ever compared against a small threshold,
    and bailing out early keeps this cheap enough to run per word per segment.
    """
    if abs(len(a) - len(b)) > limit:
        return limit + 1

    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            # Transposition: "burno" for "bruno" is one slip, not two.
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


def _similar(a: str, b: str) -> bool:
    """True when two words match closely enough to survive a transcription slip.

    Uses edit distance, not a similarity ratio -- see _MAX_EDITS_SHORT for why
    the ratio approach cannot separate "bruna" (a real slip) from "bueno" (a
    common word that must never wake anything).
    """
    if a == b:
        return True
    longest = max(len(a), len(b))
    if longest < _MIN_FUZZY_LEN:
        return False
    limit = _MAX_EDITS_LONG if longest >= 8 else _MAX_EDITS_SHORT
    return _edit_distance(a, b, limit) <= limit


def echo_overlap(heard: str, spoken: str) -> float:
    """Fraction of *heard* whose words also appear in *spoken*, 0.0..1.0.

    Text-level echo cancellation. Amplitude thresholds cannot do this job: the
    assistant's own voice through the speakers routinely exceeds any floor that
    still lets a person be heard, so it transcribes itself, answers itself, and
    holds a capture open for the full utterance cap while the user waits.

    Comparing what was HEARD against what is being SAID is exact where a
    threshold is a guess -- the assistant knows its own words.
    """
    heard_words = set(normalize(heard).split())
    if not heard_words:
        return 0.0
    spoken_words = set(normalize(spoken).split())
    if not spoken_words:
        return 0.0
    return len(heard_words & spoken_words) / len(heard_words)


def _is_repeated_summon(words: list[str], phrases: Sequence[str]) -> bool:
    """True when the transcript is just the wake phrase said several times.

    People repeat the name when they are not being answered -- which is
    precisely the moment the assistant must not ignore them. Left to the
    hallucination guard, "Bruno, Bruno, Bruno" looks exactly like a token loop
    and gets thrown away, so the assistant is deafest when the user is most
    frustrated. Observed in real use, not hypothesised.
    """
    if not words:
        return False
    targets = {normalize(p) for p in phrases if normalize(p)}
    return " ".join(dict.fromkeys(words)) in targets or set(words) <= targets


def explain_trigger(text: str, phrases: Sequence[str]) -> tuple[str | None, str]:
    """Return ``(matched_phrase_or_None, reason)``.

    The reason exists because "no match" is useless when you are trying to work
    out why the assistant ignored you. Every rejection path here used to look
    identical from outside; the interesting one is ``too-late``, where the word
    WAS said but landed outside the leading window -- usually because the
    pre-roll captured a couple of seconds of earlier talk and pushed it back.
    """
    words = normalize(text).split()
    if not words:
        return None, "empty-transcript"

    if looks_like_hallucination(text):
        if _is_repeated_summon(words, phrases):
            pass  # a person repeating the name, not a token loop
        else:
            return None, "hallucination-loop"

    for phrase in phrases:
        target = normalize(phrase).split()
        if not target:
            continue
        # Where does it appear at ALL, window aside?
        for start in range(max(1, len(words) - len(target) + 1)):
            window = words[start : start + len(target)]
            if len(window) == len(target) and all(
                _similar(w, t) for w, t in zip(window, target)
            ):
                if start < _TRIGGER_WINDOW_WORDS:
                    return phrase, "matched"
                return None, f"too-late (word {start + 1}, window {_TRIGGER_WINDOW_WORDS})"
    return None, "phrase-absent"


def match_trigger(text: str, phrases: Sequence[str]) -> str | None:
    """Return the trigger phrase found near the start of *text*, or None.

    Matching is accent-insensitive, punctuation-insensitive and tolerant of a
    small transcription slip. A multi-word phrase must appear contiguously.
    """
    words = normalize(text).split()
    if looks_like_hallucination(text) and not _is_repeated_summon(words, phrases):
        return None

    if not words:
        return None

    for phrase in phrases:
        target = normalize(phrase).split()
        if not target:
            continue
        # The phrase must START the utterance (within a small window), so a
        # trigger mentioned mid-sentence does not wake Wawi.
        last_start = min(_TRIGGER_WINDOW_WORDS, max(1, len(words) - len(target) + 1))
        for start in range(last_start):
            window = words[start : start + len(target)]
            if len(window) == len(target) and all(
                _similar(w, t) for w, t in zip(window, target)
            ):
                return phrase
    return None


@dataclass(frozen=True)
class WakeMatch:
    """Outcome of one trigger check, for logging and diagnostics."""

    matched: str | None
    text: str
    elapsed_s: float
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.matched is not None


class WhisperWakeDetector:
    """Detects a spoken trigger phrase by transcribing a speech segment.

    Usage:
        detector = WhisperWakeDetector(["wawi"])
        detector.load()                    # optional; warms the model
        if detector.check(segment).ok:
            ...

    ``check`` is called on the listener's worker thread and blocks it for the
    duration of the transcription (~40 ms with the default model), so keep the
    model small.
    """

    def __init__(
        self,
        phrases: Sequence[str],
        model: str = DEFAULT_WAKE_MODEL,
        language: str = "es",
        sample_rate: int = 16000,
    ) -> None:
        cleaned = [p.strip() for p in phrases if p and p.strip()]
        if not cleaned:
            raise WakeError(
                "No wake phrases configured. Set wake.phrases in config.yaml, "
                'for example:  phrases: ["wawi"]'
            )
        self._phrases = cleaned
        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        self._transcribe = None

    @property
    def phrases(self) -> list[str]:
        return list(self._phrases)

    @property
    def model(self) -> str:
        return self._model

    def load(self) -> None:
        """Import mlx-whisper and warm the model, or raise WakeError.

        Called from ``Listener.start()`` so a missing dependency surfaces at
        startup with an actionable message rather than on the first word
        spoken.
        """
        if self._transcribe is not None:
            return
        try:
            import mlx_whisper
        except ImportError as exc:
            raise WakeError(
                "mlx-whisper is not installed, so the whisper wake engine "
                "cannot run. Install it with:  pip install mlx-whisper"
            ) from exc

        self._transcribe = mlx_whisper.transcribe
        try:
            self._transcribe(
                np.zeros(self._sample_rate // 2, dtype=np.float32),
                path_or_hf_repo=self._model,
                language=self._language,
            )
        except Exception as exc:
            self._transcribe = None
            raise WakeError(
                f"Could not load the wake transcription model {self._model!r}: {exc}"
            ) from exc
        log.info(
            "Whisper wake engine ready: model=%s phrases=%s",
            self._model,
            self._phrases,
        )

    def check(self, segment: np.ndarray) -> WakeMatch:
        """Transcribe *segment* and report whether it starts with a trigger.

        Never raises: a transcription failure is logged and reported as no
        match, because one bad segment must not take the listener down.
        """
        duration = segment.size / self._sample_rate if segment.size else 0.0
        if duration < _MIN_SEGMENT_S:
            log.info(
                "wake: segment too short (%.2fs < %.2fs) — not transcribed",
                duration, _MIN_SEGMENT_S,
            )
            return WakeMatch(None, "", 0.0, "segment-too-short")
        if self._transcribe is None:
            log.warning("Wake detector used before load(); treating as no match")
            return WakeMatch(None, "", 0.0, "not-loaded")

        started = time.perf_counter()
        try:
            result = self._transcribe(
                np.asarray(segment, dtype=np.float32),
                path_or_hf_repo=self._model,
                language=self._language,
            )
            text = str(result.get("text", ""))
        except Exception:
            log.warning("Wake transcription failed; treating as no match", exc_info=True)
            return WakeMatch(None, "", time.perf_counter() - started, "transcribe-failed")

        elapsed = time.perf_counter() - started
        matched, reason = explain_trigger(text, self._phrases)
        # INFO, not DEBUG: this line is the whole answer to "why did it ignore
        # me", and it is worthless if you have to reproduce the problem with a
        # flag turned on.
        log.info(
            "wake: %.2fs audio, %.0fms, %r -> %s",
            duration, elapsed * 1000, text.strip()[:120], reason,
        )
        return WakeMatch(matched, text.strip(), elapsed, reason)
