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
import threading
import time
import unicodedata
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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

# Tokens decoded for the early trigger look (Listener, wake.window_s): the
# phrase must be among the first _TRIGGER_WINDOW_WORDS words, which never
# takes more than this, and the cap bounds a token loop to a few ms.
_QUICK_TOKENS = 16

# Above this no-speech probability, a transcript that IS the prompt is the
# prompt recited back over near-silence, not someone saying the name twice.
_PROMPT_ECHO_NO_SPEECH = 0.5

#: How Whisper writes a trigger that is not a dictionary word, per phrase.
#:
#: "Wawi" is four letters, below _MIN_FUZZY_LEN, so it gets no edit-distance
#: slack at all -- and it needs some: measured on Kokoro clips (see
#: scripts/wakeword/measure_phrase.py), whisper-tiny WITHOUT a prompt wrote it
#: "Wavis", "Wabies", "Wowy", "Chewawi" and only 2/30 times "Wawi". With the
#: decoder prompt (:func:`wake_prompt`) it comes back verbatim almost always,
#: and these are the residual spellings. Deliberately NOT in the list, because
#: each is an everyday word and would wake her on its own: "guau" (her own
#: bark, and Spanish "wow"), "agua", "bueno", "wifi", "huawei", "y vos".
KNOWN_ALIASES: dict[str, tuple[str, ...]] = {
    "wawi": (
        "wawy", "wawie", "waui", "wauy", "wawis", "guagui", "guaui", "uaui",
        "huawi", "wavi", "wowie", "wowy", "waway", "wawii",
    ),
}

#: An interjection Whisper glues onto the name ("Chewawi", "Xiewawi"): the
#: rest of the word is checked against the phrase on its own.
_GLUED_PREFIXES = ("che", "chee", "chie", "chew", "xe", "xie", "she", "shhe", "sche")


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
    if len(words) > 2 and any(t.startswith(words[-1]) for t in targets):
        # The window cut the last one in half: "Wawi, Wawi, Wawi, W".
        words = words[:-1]
    return " ".join(dict.fromkeys(words)) in targets or set(words) <= targets


def wake_prompt(phrases: Sequence[str]) -> str:
    """The decoder prompt that makes Whisper spell the trigger as configured.

    Whisper's initial prompt is a vocabulary hint: the decoder continues text
    that already contains the word, so a name it has never seen stops being
    split into the nearest real words. Measured with whisper-tiny on 30
    Kokoro clips of "Wawi" (alone, "che Wawi", "Wawi, ¿qué hora es?"): 2/30
    verbatim without it, 30/30 with it -- and 6x faster, because a decoder
    that is not guessing does not fall into token loops. The false-positive
    cost is measured by the same script; see scripts/wakeword/README.md.
    """
    names = [p.strip() for p in phrases if p and p.strip()]
    if not names:
        return ""
    spoken = ", ".join(n[:1].upper() + n[1:] for n in names)
    return f"{spoken}, {spoken}."


def _targets(phrase: str, aliases: dict[str, Sequence[str]] | None) -> list[list[str]]:
    """Every word sequence that counts as *phrase*: itself and its aliases."""
    key = normalize(phrase)
    table = KNOWN_ALIASES if aliases is None else aliases
    out = [key.split()]
    for alias in table.get(key, ()):
        words = normalize(alias).split()
        if words and words not in out:
            out.append(words)
    return [t for t in out if t]


def _word_is(word: str, target: str, alternatives: set[str]) -> bool:
    if _similar(word, target) or word in alternatives:
        return True
    for prefix in _GLUED_PREFIXES:
        if word.startswith(prefix) and len(word) > len(prefix):
            rest = word[len(prefix):]
            if rest == target or rest in alternatives:
                return True
    return False


def _find(words: list[str], phrase: str,
          aliases: dict[str, Sequence[str]] | None) -> tuple[int, int] | None:
    """(start, length) of the first occurrence of *phrase* in *words*."""
    canonical, *alternates = _targets(phrase, aliases)
    # Only the configured phrase gets edit-distance slack. An alias is a
    # spelling Whisper was SEEN to produce, matched exactly: fuzzy aliases
    # reach everyday words ("guaui" is one edit from "guau", "huawi" one
    # from "huawei").
    singles = {t[0] for t in alternates if len(t) == 1}
    best: tuple[int, int] | None = None
    for i, target in enumerate([canonical] + [t for t in alternates if len(t) > 1]):
        n = len(target)
        for start in range(max(1, len(words) - n + 1)):
            window = words[start:start + n]
            if len(window) != n:
                continue
            if n == 1:
                hit = _word_is(window[0], target[0], singles)
            elif i == 0:
                hit = all(_similar(w, t) for w, t in zip(window, target))
            else:
                hit = window == target
            if hit:
                if best is None or start < best[0]:
                    best = (start, n)
                break
    return best


def split_request(
    text: str,
    phrases: Sequence[str],
    aliases: dict[str, Sequence[str]] | None = None,
) -> tuple[str | None, str]:
    """``(matched_phrase, request)``: the wake word taken off the front.

    "Wawi, ¿qué hora es?" said in one breath is ONE turn: the trigger is
    stripped and the rest, with its original casing and punctuation, is the
    request. ``request`` is "" for a bare summon ("Wawi", "che Wawi"). When
    no trigger is found near the start, returns ``(None, text)`` untouched.
    """
    spans = [(m.start(), m.end(), normalize(m.group(0))) for m in re.finditer(r"\w+", text)]
    spans = [s for s in spans if s[2]]
    words = [s[2] for s in spans]
    for phrase in phrases:
        if not normalize(phrase):
            continue
        found = _find(words, phrase, aliases)
        if found is None or found[0] >= _TRIGGER_WINDOW_WORDS:
            continue
        start, n = found
        end = spans[start + n - 1][1]
        rest = text[end:].lstrip(" \t\n,.;:!¡-—…")
        # "¿Wawi, qué hora es?" leaves "qué hora es?": fine as it is. A lone
        # closing mark ("Wawi?") is no request at all.
        if not any(ch.isalnum() for ch in rest):
            rest = ""
        return phrase, rest.strip()
    return None, text


def explain_trigger(
    text: str,
    phrases: Sequence[str],
    aliases: dict[str, Sequence[str]] | None = None,
) -> tuple[str | None, str]:
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

    late: int | None = None
    for phrase in phrases:
        if not normalize(phrase):
            continue
        # Where does it appear at ALL, window aside?
        found = _find(words, phrase, aliases)
        if found is None:
            continue
        if found[0] < _TRIGGER_WINDOW_WORDS:
            return phrase, "matched"
        late = found[0] if late is None else min(late, found[0])
    if late is not None:
        return None, f"too-late (word {late + 1}, window {_TRIGGER_WINDOW_WORDS})"
    return None, "phrase-absent"


def match_trigger(
    text: str,
    phrases: Sequence[str],
    aliases: dict[str, Sequence[str]] | None = None,
) -> str | None:
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
        if not normalize(phrase):
            continue
        # The phrase must START the utterance (within a small window), so a
        # trigger mentioned mid-sentence does not wake Wawi.
        found = _find(words, phrase, aliases)
        if found is not None and found[0] < _TRIGGER_WINDOW_WORDS:
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
        aliases: dict[str, Sequence[str]] | None = None,
        prompt: str | None = None,
    ) -> None:
        cleaned = [p.strip() for p in phrases if p and p.strip()]
        if not cleaned:
            raise WakeError(
                "No wake phrases configured. Set wake.phrases in config.yaml, "
                'for example:  phrases: ["wawi"]'
            )
        self._phrases = cleaned
        # Built-in spellings plus the configured ones (wake.aliases).
        table = {k: list(v) for k, v in KNOWN_ALIASES.items()}
        for key, extra in (aliases or {}).items():
            table.setdefault(normalize(key), []).extend(str(e) for e in extra)
        self._aliases = {k: tuple(v) for k, v in table.items()}
        # None = the default prompt naming the phrases; "" = no prompt at all.
        self._prompt = wake_prompt(cleaned) if prompt is None else prompt
        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        # The decode hook: (audio, **options) -> {"text": ...}. Set by load();
        # tests replace it with a stub.
        self._transcribe = None
        self._executor: ThreadPoolExecutor | None = None
        self._whisper = None

    @property
    def phrases(self) -> list[str]:
        return list(self._phrases)

    @property
    def model(self) -> str:
        return self._model

    @property
    def aliases(self) -> dict[str, tuple[str, ...]]:
        return dict(self._aliases)

    def split(self, text: str) -> tuple[str | None, str]:
        """:func:`split_request` with this detector's phrases and aliases."""
        return split_request(text, self._phrases, self._aliases)

    def _decode_options(self) -> dict:
        return {"initial_prompt": self._prompt} if self._prompt else {}

    def load(self) -> None:
        """Load this detector's own Whisper model and warm it, or raise WakeError.

        Called from ``Listener.start()`` when a trigger is needed, so a missing
        dependency surfaces at startup with an actionable message rather than
        on the first word spoken. Safe to call again (no-op) and from any
        thread: the model lives on the detector's own MLX thread.

        The model is loaded PRIVATELY rather than through
        ``mlx_whisper.transcribe``: that function keeps ONE model per process
        (``ModelHolder``), so a tiny wake model and the large STT model
        evicted each other on every turn -- measured ~0.5 s of reloading the
        large model before each answer, and ~0.25 s for the tiny one before
        the next wake check.
        """
        if self._transcribe is not None:
            return
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:
            raise WakeError(
                "mlx-whisper is not installed, so the whisper wake engine "
                "cannot run. Install it with:  pip install mlx-whisper"
            ) from exc
        try:
            self._mlx().submit(self._load_private).result()
            self._transcribe = self._mlx_transcribe
            self._transcribe(
                np.zeros(self._sample_rate // 2, dtype=np.float32),
                path_or_hf_repo=self._model,
                language=self._language,
                **self._decode_options(),
            )
        except Exception as exc:
            self._transcribe = None
            raise WakeError(
                f"Could not load the wake transcription model {self._model!r}: {exc}"
            ) from exc
        log.info(
            "Whisper wake engine ready: model=%s phrases=%s prompt=%r",
            self._model,
            self._phrases,
            self._prompt,
        )

    def load_async(self) -> None:
        """Start loading in the background (the live mode switch): never blocks."""
        if self._transcribe is not None:
            return

        def _go() -> None:
            try:
                self.load()
            except Exception:
                log.warning("Wake model failed to load in the background", exc_info=True)

        threading.Thread(target=_go, name="wawi-wake-load", daemon=True).start()

    @property
    def loaded(self) -> bool:
        return self._transcribe is not None

    # -- the private MLX model ------------------------------------------------

    def _mlx(self) -> ThreadPoolExecutor:
        # MLX streams are thread-local (see wawi/stt.py): load and decode on
        # ONE thread of our own, never the listener's or the STT's.
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-wake")
        return self._executor

    def _load_private(self) -> None:
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        self._whisper = load_model(self._model, dtype=mx.float16)

    def _mlx_transcribe(self, audio: np.ndarray, **options) -> dict:
        return self._mlx().submit(self._decode, np.asarray(audio, np.float32), options).result()

    def _decode(self, audio: np.ndarray, options: dict) -> dict:
        """One greedy pass over the first 30 s, capped at *sample_len* tokens.

        No temperature fallback: ``transcribe`` re-decodes up to five more
        times when the first pass looks unsure, which is exactly the case on
        a non-word like "Wawi" -- and a trigger check has no use for the
        "better" guesses those passes make. The token cap bounds the token
        loops small models fall into (they cost 500-850 ms each, measured).
        """
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions

        model = self._whisper
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        result = model.decode(mel, DecodingOptions(
            language=options.get("language") or self._language,
            prompt=options.get("initial_prompt") or None,
            sample_len=options.get("sample_len"),
            without_timestamps=True,
            fp16=True,
        ))
        return {"text": result.text, "no_speech_prob": float(result.no_speech_prob),
                "avg_logprob": float(result.avg_logprob)}

    # -- checking ---------------------------------------------------------------

    def check(self, segment: np.ndarray, quick: bool = False) -> WakeMatch:
        """Transcribe *segment* and report whether it starts with a trigger.

        *quick* caps the decode at a few tokens: enough for the trigger
        window (the phrase must be among the first words), not for the whole
        sentence. The listener uses it for the early look at the first
        ``wake.window_s`` of speech; the echo check needs the full text.

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
        options = self._decode_options()
        if quick:
            options["sample_len"] = _QUICK_TOKENS
        try:
            result = self._transcribe(
                np.asarray(segment, dtype=np.float32),
                path_or_hf_repo=self._model,
                language=self._language,
                **options,
            )
            text = str(result.get("text", ""))
            no_speech = result.get("no_speech_prob")
            if (self._prompt and no_speech is not None and no_speech > _PROMPT_ECHO_NO_SPEECH
                    and normalize(text) == normalize(self._prompt)):
                # Whisper's known failure with a prompt: on audio it thinks
                # holds no words it recites the prompt back. Real speech
                # saying "Wawi, Wawi" has a low no-speech probability.
                log.info("wake: transcript is the prompt echoed back — ignored")
                return WakeMatch(None, text.strip(), time.perf_counter() - started,
                                 "prompt-echo")
        except Exception:
            log.warning("Wake transcription failed; treating as no match", exc_info=True)
            return WakeMatch(None, "", time.perf_counter() - started, "transcribe-failed")

        elapsed = time.perf_counter() - started
        matched, reason = explain_trigger(text, self._phrases, self._aliases)
        # INFO, not DEBUG: this line is the whole answer to "why did it ignore
        # me", and it is worthless if you have to reproduce the problem with a
        # flag turned on.
        log.info(
            "wake: %.2fs audio%s, %.0fms, %r -> %s",
            duration, " (window)" if quick else "", elapsed * 1000, text.strip()[:120], reason,
        )
        return WakeMatch(matched, text.strip(), elapsed, reason)


class AcousticVerifier:
    """Second opinion on a whisper match: the trained "wawi" model's score.

    Whisper with the name as a prompt hears "Wawi" in almost every word that
    sounds a bit like it -- measured, about half of "Huawei", "Wally", "wow",
    "guay", "guau" clips. The openWakeWord model (wawi/assets/wakeword) was
    trained on exactly those near-misses as negatives and scores them near 0,
    so a match only stands when the audio ALSO scores >= *threshold* on it.
    The two fail differently, which is the point of asking both.

    ~1.2 ms per 80 ms of audio; a 3 s candidate costs ~45 ms, and it runs only
    after Whisper has already said yes. Never raises: an error means "no veto".
    """

    def __init__(self, model_path: str, threshold: float = 0.5,
                 inference_framework: str = "onnx") -> None:
        import os

        self._path = model_path
        self._key = os.path.splitext(os.path.basename(model_path))[0]
        self._threshold = float(threshold)
        self._framework = inference_framework
        self._model = None
        self._lock = threading.Lock()

    @property
    def threshold(self) -> float:
        return self._threshold

    def load(self) -> None:
        if self._model is None:
            import openwakeword

            self._model = openwakeword.Model(
                wakeword_models=[self._path], inference_framework=self._framework)

    def score(self, audio: np.ndarray) -> float:
        """Peak score over *audio* (float32 16 kHz), with a second of lead-in.

        The model reads 1.28 s of context; the lead-in of silence lets a
        name said in the very first frames still fill its window.
        """
        try:
            with self._lock:
                self.load()
                model = self._model
                model.reset()
                pcm = np.clip(np.asarray(audio, np.float32), -1.0, 1.0)
                pcm = np.concatenate([np.zeros(16000, np.float32), pcm,
                                      np.zeros(4000, np.float32)])
                pcm = (pcm * 32767).astype(np.int16)
                best = 0.0
                for i in range(0, pcm.size - 1279, 1280):
                    s = float(model.predict(pcm[i:i + 1280]).get(self._key, 0.0))
                    best = max(best, s)
                model.reset()
                return best
        except Exception:
            log.warning("Acoustic verifier failed; not vetoing", exc_info=True)
            return 1.0

    def accepts(self, audio: np.ndarray) -> tuple[bool, float]:
        s = self.score(audio)
        return s >= self._threshold, s
