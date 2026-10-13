"""Tell a person talking apart from noise, AFTER Whisper has had its say.

The listener (:mod:`wawi.listener`) already refuses segments that never had
enough voiced speech in them. What survives that still reaches Whisper, and
Whisper has one well-known failure: handed audio with no words in it, it does
not return nothing -- it returns the most likely sentence from its training
data. Measured on this machine with ``whisper-large-v3-turbo``:

    silence, white noise, a clap, a 440 Hz tone, keyboard clicks
        -> "Gracias."  /  "¡Suscríbete al canal!"        (no initial prompt)
        -> "¿Está bien?"                                  (with our prompt)

The second line is the decoder parroting its own seed (:data:`wawi.lang.
STT_PROMPT` used to contain "¿Está bien así?"; it no longer does, and "esta
bien" stays in :data:`WEAK_JUNK`). In the log that exact phrase became a turn
the user never said, again and again, at rms 0.002.

Whisper's per-segment confidence does NOT save us here: on this mlx-whisper
build ``no_speech_prob`` comes back 0.0 for pure silence, and the avg_logprob
of a hallucination (-0.42 .. -0.6) overlaps a genuine short reply (-0.19 ..
-0.5). So the rules lean on what the LISTENER measured -- how much voiced
speech the segment really had -- and use Whisper's numbers only where they are
unambiguous (a decoder stuck in a loop, or confidence far below anything real).

Pure functions, no model, no I/O: cheap to call on every turn and to test.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

# Phrases nobody says to a voice assistant, but Whisper produces from noise
# because they close millions of subtitled videos. Matched as SUBSTRINGS of the
# normalised transcript and dropped whatever the evidence: there is no
# legitimate turn that consists of a subtitle credit.
ALWAYS_JUNK: tuple[str, ...] = (
    "suscribete al canal",
    "suscribanse al canal",
    "no olvides suscribirte",
    "dale like",
    "gracias por ver",
    "gracias por mirar",
    "subtitulos por",
    "subtitulos realizados por",
    "subtitulado por",
    "amara org",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
)

# Short transcripts Whisper also invents from noise, but that a person CAN say.
# Matched EXACTLY (after normalising) and dropped only when the evidence that
# someone spoke is weak -- see :func:`judge`.
WEAK_JUNK: frozenset[str] = frozenset({
    "gracias", "muchas gracias", "gracias gracias", "muchisimas gracias",
    "thank you", "thanks", "thank you very much", "you",
    "amen", "chau", "adios", "bye", "bye bye",
    "musica", "music", "aplausos", "applause", "risas", "laughter",
    "silencio", "silence",
    "y", "eh", "ah", "oh", "mm", "mmm", "hmm", "uh", "um",
    "esta bien", "nos vemos", "hasta luego", "hasta la proxima",
})

# Genuine short replies. Never dropped as junk or as a prompt echo -- "sí",
# "no" and "dale" all appear in the STT prompt, and they are exactly the
# answers a confirmation question is waiting for. Only the hard confidence
# rules (logprob floor, compression ratio) can still reject them.
PROTECTED: frozenset[str] = frozenset({
    "si", "no", "dale", "si dale", "dale si", "ok", "okay", "okey", "oka",
    "bueno", "listo", "claro", "perfecto", "de una", "obvio", "no gracias",
    "si gracias", "yes", "yeah", "yep", "nope", "sure", "no thanks",
    "si por favor", "para", "basta", "cancela", "cancelar", "segui", "stop",
})

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = _PUNCT.sub(" ", folded.lower())
    return _SPACE.sub(" ", folded).strip()


@dataclass(frozen=True)
class SttMetrics:
    """Whisper's own confidence for one transcription, aggregated.

    Attributes:
        avg_logprob: Mean token log-probability, weighted by segment length.
            Closer to 0 is more confident.
        no_speech_prob: Highest per-segment probability of "no speech".
        compression_ratio: Highest per-segment gzip ratio; above ~2.4 the
            decoder is repeating itself.
        segments: How many segments the numbers come from.
    """

    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    segments: int

    @classmethod
    def from_segments(cls, segments: Iterable[dict]) -> SttMetrics | None:
        """Aggregate mlx-whisper's ``result["segments"]``; None if unusable."""
        rows = []
        for seg in segments or ():
            try:
                rows.append((
                    float(seg["avg_logprob"]),
                    float(seg["no_speech_prob"]),
                    float(seg["compression_ratio"]),
                    max(1e-3, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        if not rows:
            return None
        total = sum(r[3] for r in rows)
        return cls(
            avg_logprob=sum(r[0] * r[3] for r in rows) / total,
            no_speech_prob=max(r[1] for r in rows),
            compression_ratio=max(r[2] for r in rows),
            segments=len(rows),
        )


@dataclass(frozen=True)
class SpeechEvidence:
    """What the listener measured about the segment, before any transcription.

    Attributes:
        voiced_ms: Milliseconds of frames the VAD called speech.
        active_ms: Milliseconds from the first to the last voiced frame.
        forced: Opened by push-to-talk; the VAD was not consulted, so
            ``voiced_ms`` means nothing and the key press is the evidence.
    """

    voiced_ms: int
    active_ms: int
    forced: bool = False

    @property
    def ratio(self) -> float:
        return self.voiced_ms / self.active_ms if self.active_ms > 0 else 0.0


@dataclass(frozen=True)
class FilterConfig:
    """Knobs for :func:`judge`. See config.yaml ``stt:`` for the user's view.

    Attributes:
        enabled: Master switch. False lets every non-empty transcript through.
        min_avg_logprob: Below this the transcript is dropped outright.
        max_compression_ratio: Above this the decoder is looping; dropped.
        max_no_speech_prob: At or above this AND a mediocre avg_logprob
            (< -0.5), Whisper itself thinks nothing was said.
        junk_min_voiced_ms: A WEAK_JUNK phrase or a prompt echo is kept only
            if the listener heard at least this much voiced speech...
        junk_min_logprob: ...and Whisper was at least this confident.
    """

    enabled: bool = True
    min_avg_logprob: float = -1.0
    max_compression_ratio: float = 2.4
    max_no_speech_prob: float = 0.6
    junk_min_voiced_ms: int = 700
    junk_min_logprob: float = -0.45


def _echoes_prompt(norm: str, prompt: str) -> bool:
    """True when the whole transcript is a piece of the decoder's seed."""
    if not norm or not prompt:
        return False
    p = normalize(prompt)
    # Whole-word containment, so "si" does not match inside "revisa".
    return f" {norm} " in f" {p} "


def judge(
    text: str,
    metrics: SttMetrics | None = None,
    evidence: SpeechEvidence | None = None,
    prompt: str = "",
    cfg: FilterConfig | None = None,
) -> str | None:
    """Return why *text* should be discarded, or None to keep it.

    Args:
        text: The transcript, as Whisper returned it.
        metrics: Whisper's confidence, when the STT exposed it.
        evidence: The listener's voiced-speech measurement, when available.
            None (push-to-talk fallback recorder, tests) is treated as "no
            VAD evidence either way": only Whisper's confidence decides.
        prompt: The initial prompt the decoder was seeded with.
        cfg: Thresholds; defaults when None.
    """
    cfg = cfg or FilterConfig()
    if not cfg.enabled:
        return None
    norm = normalize(text)
    if not norm:
        return None  # empty is handled upstream as "nothing said"

    for phrase in ALWAYS_JUNK:
        if phrase in norm:
            return f"known Whisper hallucination ({phrase!r})"

    if metrics is not None:
        if metrics.compression_ratio > cfg.max_compression_ratio:
            return (
                f"decoder looping (compression_ratio {metrics.compression_ratio:.2f}"
                f" > {cfg.max_compression_ratio:.2f})"
            )
        if metrics.avg_logprob < cfg.min_avg_logprob:
            return (
                f"low confidence (avg_logprob {metrics.avg_logprob:.2f}"
                f" < {cfg.min_avg_logprob:.2f})"
            )
        if metrics.no_speech_prob >= cfg.max_no_speech_prob and metrics.avg_logprob < -0.5:
            return (
                f"Whisper heard no speech (no_speech_prob {metrics.no_speech_prob:.2f},"
                f" avg_logprob {metrics.avg_logprob:.2f})"
            )

    if norm in PROTECTED:
        return None

    suspect = None
    if norm in WEAK_JUNK:
        suspect = "common Whisper hallucination"
    elif _echoes_prompt(norm, prompt):
        suspect = "echo of the STT prompt"
    if suspect is None:
        return None

    # A suspect phrase stays only on strong evidence that a person said it.
    weak: list[str] = []
    if evidence is not None and not evidence.forced:
        if evidence.voiced_ms < cfg.junk_min_voiced_ms:
            weak.append(f"voiced {evidence.voiced_ms}ms < {cfg.junk_min_voiced_ms}ms")
    if metrics is not None and metrics.avg_logprob < cfg.junk_min_logprob:
        weak.append(f"avg_logprob {metrics.avg_logprob:.2f} < {cfg.junk_min_logprob:.2f}")
    if metrics is None and (evidence is None or evidence.forced):
        # Nothing measured at all: trust the key press / the caller.
        return None
    if weak:
        return f"{suspect}, weak evidence ({', '.join(weak)})"
    return None
