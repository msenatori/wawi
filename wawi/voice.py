"""Kokoro ONNX TTS synthesis and playback with a global voice lock.

Uses kokoro-onnx (ONNX Runtime backend) which is fully thread-safe: ONNX
Runtime sessions hold no GPU/MPS stream state, so inference can run from any
thread without the RuntimeError("There is no Stream(gpu) in current thread")
that affected mlx-based engines.

VOICE_LOCK is the module-level singleton that serializes all TTS output — two
utterances can never overlap (REQ-F1-05, REQ-F1-10).

say() is the single speech funnel:
  1. Sanitize text (strip markdown/emojis — they make espeak-ng slow or hang).
  2. Normal: synth via Kokoro → numpy float32 array → sounddevice playback.
     Synthesis of the next sentence is pipelined on a background thread while
     the current sentence plays (overlapping synth/playback reduces latency).
  3. Kokoro failure: log WARNING, print text to stderr.
  4. Never propagates an exception to the caller.

say_part() is the same funnel for an answer that is still being generated:
the caller hands over one finished sentence at a time and speech starts on
the first one instead of on the last.

Barge-in: call interrupt() from any thread to abort the current utterance
immediately. say() clears the abort flag at the top so it is safe to call
say() again immediately after interrupt().
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Tuple

import numpy as np

log = logging.getLogger(__name__)

# The one serialization lock for all TTS output.
# VOICE_LOCK is a LEAF lock — never acquired while holding orchestrator._turn_lock
# would cause a cycle; the design guarantees turn_lock → VOICE_LOCK ordering.
VOICE_LOCK = threading.Lock()

# Fenced code blocks: ```[lang]\n...\n``` (possibly unterminated at EOF).
_FENCED_CODE = re.compile(r"```[^\n`]*\n.*?(?:```|$)", re.DOTALL)
# Inline code spans: `word` — strip backticks, keep inner text.
_INLINE_CODE = re.compile(r"`([^`\n]*)`")

# Markdown emphasis/structure characters that should never be spoken.
_MD_CHARS = re.compile(r"[*_`#>~|]+")
# Emoji / pictographic / symbol / dingbat ranges + variation selectors.
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff"   # symbols, emoticons, pictographs, supplemental
    "\U00002600-\U000027bf"    # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"    # regional indicators (flags)
    "←-⇿"            # arrows
    "⬀-⯿"            # misc symbols and arrows
    "️]"                  # variation selector-16
)

# Hard cap on characters fed to TTS — a runaway long response should not freeze
# the assistant in "speaking" for a minute. Truncate at a sentence-ish boundary.
_MAX_TTS_CHARS = 600

# Trailing silence appended to each synthesized sentence before playback.
# Prevents Bluetooth / AirPods from closing the output buffer early and
# clipping the final syllable of the sentence.
_TAIL_PAD_S = 0.3


def _clean_for_tts(text: str, *, code_fallback: bool = True) -> str:
    """Strip code blocks, markdown markers, and emojis; collapse whitespace; cap length.

    Processing order:
      1. Remove fenced code blocks entirely (they are shown on-screen).
      2. Strip backticks from inline code spans, keeping the inner token.
      3. Remove remaining markdown emphasis/structure characters.
      4. Remove emojis.
      5. Collapse whitespace.
      6. If nothing useful remains, return a short Spanish spoken fallback.
      7. Apply the existing length cap.

    espeak-ng (Kokoro's Spanish phonemizer) chokes on '*', emojis, and very
    long unbroken input — sanitizing keeps synthesis fast and the speech clean.

    Args:
        code_fallback: When False, step 6 returns "" instead of the spoken
            "I left the code on screen" line. Streamed answers arrive one
            sentence at a time, and a sentence that happens to hold no letters
            ("42.") is not an answer made of code — announcing one mid-answer
            would interrupt a perfectly good explanation with a non-sequitur.
    """
    # Step 1: remove fenced code blocks (shown on screen, not spoken).
    text = _FENCED_CODE.sub("", text)
    # Step 2: inline code — strip backticks, keep inner text.
    text = _INLINE_CODE.sub(r"\1", text)
    # Step 3-5: existing markdown/emoji cleanup + whitespace collapse.
    text = _MD_CHARS.sub("", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Step 6: if code removal left nothing meaningful, use a spoken fallback.
    if not re.search(r"[A-Za-záéíóúüñÁÉÍÓÚÜÑ]", text):
        return "Te dejé el código en pantalla." if code_fallback else ""
    # Step 7: existing length cap.
    if len(text) > _MAX_TTS_CHARS:
        clipped = text[:_MAX_TTS_CHARS]
        # Prefer cutting at the last sentence end for a natural stop.
        cut = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "))
        text = clipped[: cut + 1] if cut > 200 else clipped
    return text


# Split on sentence-ending punctuation followed by whitespace. Short chunks
# phonemize reliably in Spanish (long inputs lose their tail to espeak word
# mismatches). Falls back to the whole text if no sentence boundary is found.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def take_complete_sentences(text: str) -> Tuple[list[str], str]:
    """Split *text* into the sentences that are FINISHED and the rest.

    A sentence counts as finished only when its terminal punctuation is
    followed by whitespace inside *text*, which is what makes this safe to run
    on a buffer that is still growing: "Son las 3." could still turn out to be
    "Son las 3.15", and "3.15" read as two sentences is worse than waiting.

    The one splitter, shared with :func:`_split_sentences`. A second one drifts,
    and the drift is audible: the streamer would hand Kokoro a fragment the
    non-streaming path would never have produced.

    Returns:
        (finished sentences, trailing remainder). The remainder is returned
        raw — the caller owns it and must keep feeding it.
    """
    parts = _SENTENCE_SPLIT.split(text)
    finished = [p.strip() for p in parts[:-1] if p.strip()]
    return finished, parts[-1]


#: Frame length of the speech envelope the mascot bobs to. 20 ms resolves
#: syllables (~200 ms) comfortably and costs nothing to compute.
_ENVELOPE_HOP_S = 0.02


def speech_envelope(samples: np.ndarray, sr: int, hop_s: float = _ENVELOPE_HOP_S) -> np.ndarray:
    """Loudness of *samples* per *hop_s* frame, scaled to 0..1.

    Scaled by the sentence's own 95th percentile rather than a fixed constant,
    so a quiet voice and a loud one both use the whole range: what the mascot
    shows is the RHYTHM of the speech, not how loud the speakers are.
    """
    hop = max(1, int(sr * hop_s))
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size < hop:
        return np.zeros(0, dtype=np.float32)
    frames = data[: data.size // hop * hop].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    top = float(np.percentile(rms, 95))
    if top <= 1e-6:
        return np.zeros(len(rms), dtype=np.float32)
    return np.clip(rms / top, 0.0, 1.0).astype(np.float32)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into trimmed sentences; never returns an empty list."""
    finished, remainder = take_complete_sentences(text)
    parts = [*finished, remainder.strip()] if remainder.strip() else finished
    return parts or [text]


# ----------------------------------------------------------------------
# Voice character: blend, pitch, brightness
# ----------------------------------------------------------------------
#
# All three are optional and default to "off": with tts.voice a plain name,
# pitch 0 and brightness 0 the audio is exactly what it was before they
# existed. They are cheap (one polyphase resample, which already runs when the
# output device is not at 24 kHz, plus one first-difference filter) and need
# nothing beyond numpy (scipy is used when present, as for the device resample).

#: Beyond an octave it stops sounding like a voice; and Kokoro refuses speeds
#: outside 0.5..2.0, which is what a shift is synthesized at.
MAX_PITCH_SEMITONES = 12.0

#: Named looks for the voice. A preset only fills in what the user left unset:
#: an explicit tts.voice / pitch_semitones / brightness in config.yaml wins.
TTS_PRESETS: dict[str, dict[str, object]] = {
    # Small playful dog. Dora stays the base (the only Spanish female voice,
    # so the Spanish stays clean), 30% of Heart softens and rounds it, and 4
    # semitones up makes it smaller. Picked from an audition of 17 variants,
    # checked with Whisper: as intelligible as plain Dora. (A Bella blend was
    # brighter but turned an isolated "¡Hola!" into "Bullet".)
    "cachorra": {"voice": "ef_dora:0.7,af_heart:0.3", "pitch_semitones": 4.0, "brightness": 0.2},
}


def parse_voice_spec(spec: str) -> list[tuple[str, float]]:
    """Parse ``"ef_dora"`` or ``"ef_dora:0.7,af_heart:0.3"`` into (name, weight)s.

    Weights are normalised to add up to 1, so ``"a:7,b:3"`` and
    ``"a:0.7,b:0.3"`` are the same blend. A name without a weight counts as 1.
    Names are NOT checked against the model here (that needs the voices file);
    :class:`Voice` does it on load.

    Raises:
        ValueError: on an empty spec, a malformed name or weight, a weight
            that is not > 0, or a voice named twice.
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("tts.voice is empty")
    parts: list[tuple[str, float]] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            raise ValueError(f"tts.voice {spec!r}: empty entry in the blend")
        name, sep, weight_s = raw.partition(":")
        name = name.strip()
        if not re.fullmatch(r"[a-z]{2}_[a-z0-9_]+", name):
            raise ValueError(f"tts.voice {spec!r}: {name!r} is not a Kokoro voice name")
        try:
            weight = float(weight_s) if sep else 1.0
        except ValueError:
            raise ValueError(f"tts.voice {spec!r}: weight {weight_s.strip()!r} is not a number") from None
        if not (0.0 < weight < float("inf")):
            raise ValueError(f"tts.voice {spec!r}: the weight of {name} must be > 0")
        if any(n == name for n, _ in parts):
            raise ValueError(f"tts.voice {spec!r}: {name} appears twice")
        parts.append((name, weight))
    total = sum(w for _, w in parts)
    return [(n, w / total) for n, w in parts]


def primary_voice(spec: str) -> str:
    """The heaviest voice of a blend (the first one on a tie)."""
    return max(parse_voice_spec(spec), key=lambda nw: nw[1])[0]


def _voice_language(spec: str) -> str:
    """Kokoro's language letter for *spec*'s main voice ("e" for ef_dora)."""
    try:
        return primary_voice(spec)[:1]
    except ValueError:
        return ""


def retune(samples: np.ndarray, sr: int, pitch_semitones: float, out_sr: int | None = None) -> np.ndarray:
    """Shift the pitch of *samples* by *pitch_semitones* and resample to *out_sr*.

    The shift is a change of playback rate: the samples are read as if they
    had been recorded at ``sr * ratio`` and resampled from there to *out_sr*.
    That shortens the audio by ``ratio`` too; the caller makes up for it by
    synthesizing at ``speed / ratio``, so the duration comes out unchanged.
    Formants move with the pitch, which is the point: it is what makes the
    voice sound smaller and younger instead of just higher.

    One resample does both jobs (pitch and device rate), so pitching costs
    nothing extra on a 48 kHz output. Polyphase with scipy when present, an
    FFT resample (brick-wall low-pass, so no aliasing) otherwise.
    """
    out_sr = int(out_sr or sr)
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    src = sr * 2.0 ** (float(pitch_semitones) / 12.0)
    if abs(src - out_sr) < 1e-6 or data.size == 0:
        return data
    from fractions import Fraction

    frac = Fraction(out_sr / src).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    try:
        from scipy.signal import resample_poly  # type: ignore[import]
    except ImportError:
        return _fft_resample(data, up, down)
    return resample_poly(data, up, down).astype(np.float32)


def _fft_resample(data: np.ndarray, up: int, down: int) -> np.ndarray:
    """Resample by up/down in the frequency domain (numpy only)."""
    n_out = max(1, int(round(data.size * up / down)))
    spec = np.fft.rfft(data)
    keep = n_out // 2 + 1
    if keep <= spec.size:
        spec = spec[:keep]
    else:
        spec = np.concatenate([spec, np.zeros(keep - spec.size, dtype=spec.dtype)])
    return (np.fft.irfft(spec, n_out) * (n_out / data.size)).astype(np.float32)


def brighten(samples: np.ndarray, amount: float) -> np.ndarray:
    """A gentle treble tilt: add *amount* of the first difference (+6 dB/octave).

    0 is a no-op; 0.2-0.4 is "a bit brighter". Never louder at the peak than
    the input (or 0.99), so it cannot clip. One vector op.
    """
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if amount <= 0 or data.size < 2:
        return data
    out = data + float(amount) * np.diff(data, prepend=data[0])
    limit = max(float(np.max(np.abs(data))), 0.99)
    peak = float(np.max(np.abs(out)))
    if peak > limit:
        out *= limit / peak
    return out.astype(np.float32)


class Voice:
    """Synthesize and play text using Kokoro ONNX TTS.

    Args:
        voices_dir: Directory containing kokoro-v1.0.onnx and voices-v1.0.bin.
        voice: Kokoro voice name, e.g. "ef_dora" (Spanish female).
        speed: TTS synthesis speed multiplier (1.0 = normal, 1.05 = slightly faster).
        pitch_semitones: Character pitch shift (0 = the voice as Kokoro made it).
        brightness: Treble tilt 0..1 (0 = off).

    *voice* may be a blend, ``"ef_dora:0.7,af_heart:0.3"``: the style vectors
    are mixed by weight. The phonemiser stays the language's, so a blend
    changes the timbre, not what is pronounced.
    """

    def _mark_play(self) -> None:
        """Note when a sentence starts coming out of the speakers (latency log)."""
        marks = self.__dict__.setdefault("_play_marks", [])
        marks.append(time.perf_counter())
        del marks[:-32]

    def first_play_since(self, t: float) -> float | None:
        """perf_counter of the first sentence played at or after ``t``, if any."""
        for mark in self.__dict__.get("_play_marks", []):
            if mark >= t:
                return mark
        return None

    def __init__(
        self,
        voices_dir: Path,
        voice: str,
        speed: float = 1.05,
        pitch_semitones: float = 0.0,
        brightness: float = 0.0,
        home_language: str | None = None,
    ) -> None:
        from kokoro_onnx import Kokoro  # type: ignore[import]

        self._voice: str | np.ndarray = voice
        # The voice as configured, so a language round trip (es -> en -> es)
        # comes back to it instead of to the language's stock voice.
        self._configured_voice = voice
        # The language the configured voice was chosen for. A voice from
        # another family (jf_alpha speaking Spanish) has no letter in common
        # with its language, so the round trip also keys on this.
        self._home_language = home_language
        self._speed = speed
        self._pitch = max(-MAX_PITCH_SEMITONES, min(MAX_PITCH_SEMITONES, float(pitch_semitones)))
        self._brightness = max(0.0, min(1.0, float(brightness)))
        # Kokoro takes both as CALL arguments, so switching language costs
        # nothing: the 310 MB model stays loaded and only these two change.
        self._kokoro_lang = "es"
        self._kokoro = Kokoro(
            str(voices_dir / "kokoro-v1.0.onnx"),
            str(voices_dir / "voices-v1.0.bin"),
        )
        self._voice = self._style_for(voice)
        # Barge-in abort event. Set by interrupt(); cleared at the top of say()
        # so a previous interrupt never silences a fresh utterance.
        self._abort = threading.Event()
        # What is coming out of the speakers right now, for the mascot:
        # (envelope, started_at). Replaced as one tuple, so a reader on another
        # thread sees either the old sentence or the new one, never half.
        self._playing: tuple[np.ndarray, float] | None = None

        # Output device native sample rate. Kokoro synthesizes at 24 kHz; if the
        # device runs at another rate (e.g. 48 kHz on Mac speakers), PortAudio's
        # built-in sample-rate conversion sounds gritty/low-quality. We resample
        # to the device rate ourselves (high-quality polyphase) so PortAudio does
        # NO conversion. None = couldn't detect → play at Kokoro's native rate.
        self._device_sr: int | None = None
        try:
            import sounddevice as sd  # type: ignore[import]

            self._device_sr = int(sd.query_devices(kind="output")["default_samplerate"])
            log.info("TTS output device native sample rate: %d Hz", self._device_sr)
        except Exception:
            log.warning("Could not detect output sample rate; no resampling", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_language(self, code: str) -> None:
        """Switch the spoken language, voice and phonemiser together.

        All three or none: a Spanish voice reading English text, or an English
        voice fed the Spanish phonemiser, are both worse than not switching.
        Cheap enough to call between turns — Kokoro takes them per call, so
        nothing is reloaded.
        """
        from wawi import lang

        code = lang.normalise(code)
        voice = lang.voice_for(code)
        configured = getattr(self, "_configured_voice", None)
        home = getattr(self, "_home_language", None)
        if configured and (
            _voice_language(configured) == voice[:1]
            or (home is not None and lang.normalise(home) == code)
        ):
            voice = configured
        self._voice = self._style_for(voice)
        self._kokoro_lang = lang.TTS_LANG[code]

    def _style_for(self, spec: str) -> "str | np.ndarray":
        """What to hand Kokoro for *spec*: the name itself, or a blended style.

        A single voice stays a NAME, so the default path is exactly what it
        always was. A blend is the weighted sum of the style vectors. An
        unknown voice is left out of the blend with a warning (all unknown:
        the first name is passed through and Kokoro's own error reports it).
        """
        try:
            parts = parse_voice_spec(spec)
        except ValueError:
            return spec  # Kokoro reports it per sentence, as it always did
        kokoro = getattr(self, "_kokoro", None)
        if len(parts) == 1 or kokoro is None:
            return parts[0][0]
        known = [(n, w) for n, w in parts if n in kokoro.voices]
        for name, _ in parts:
            if name not in kokoro.voices:
                log.warning("tts.voice: unknown Kokoro voice %r left out of the blend", name)
        if not known:
            return parts[0][0]
        total = sum(w for _, w in known)
        style = sum((w / total) * np.asarray(kokoro.get_voice_style(n), dtype=np.float32) for n, w in known)
        return np.asarray(style, dtype=np.float32)

    def output_level(self) -> float | None:
        """Level 0..1 of the speech playing right now, or None when silent.

        Read by the panel's render tick, on the main thread, while this object
        plays on another: it only reads one tuple and indexes an array.
        """
        playing = getattr(self, "_playing", None)
        if playing is None:
            return None
        envelope, started = playing
        i = int((time.monotonic() - started) / _ENVELOPE_HOP_S)
        if i < 0 or i >= len(envelope):
            return None
        return float(envelope[i])

    def interrupt(self) -> None:
        """Signal the current utterance to stop immediately.

        Sets the abort event and calls sounddevice.stop() to kill in-flight
        playback. Safe to call from any thread at any time, including when
        say() is not currently running.
        """
        import sounddevice as sd  # type: ignore[import]

        log.info("TTS interrupt requested")
        self._abort.set()
        try:
            sd.stop()
        except Exception:
            log.warning("sounddevice.stop() failed during interrupt", exc_info=True)

    def say(self, text: str) -> None:
        """Synthesize *text* and play it.

        Acquires VOICE_LOCK for the entire synth+playback duration so two
        utterances never overlap. Falls back silently to stderr print on failure.

        Clears the abort flag at the top so a prior interrupt() call never
        suppresses a fresh utterance.

        Args:
            text: UTF-8 text to synthesize. If empty, returns immediately.
        """
        # Clear any stale abort from a previous barge-in BEFORE acquiring the lock
        # so that we never enter _synth_and_play in an already-aborted state.
        self._abort.clear()
        self._speak_clean(_clean_for_tts(text))

    #: Idle this long and the model's pages have gone cold: the first answer
    #: after a pause took 3-4 s to synthesize instead of under one.
    WARM_AFTER_S = 90.0

    def warm(self) -> None:
        """Page the model back in, off the caller's thread, if it has idled.

        Called the moment someone starts talking, so the synthesis it costs
        overlaps with them finishing the sentence and with transcription.
        Plays nothing. Never raises.
        """
        now = time.monotonic()
        if now - self.__dict__.get("_last_synth", 0.0) < self.WARM_AFTER_S:
            return
        self._last_synth = now

        def _run() -> None:
            try:
                self._synth_raw("Sí.")
            except Exception:
                log.debug("TTS warm-up failed", exc_info=True)

        threading.Thread(target=_run, name="wawi-tts-warm", daemon=True).start()

    def say_part(self, text: str, *, first: bool) -> bool:
        """Speak one piece of an answer that is still being generated.

        The streaming counterpart of :meth:`say`. It exists because
        :meth:`say` clears the abort flag at the top: calling it once per
        sentence would make every sentence undo the barge-in that silenced the
        one before it, so cutting a streamed answer off would be impossible —
        it would stop for a moment and then keep talking.

        Args:
            first: True for the opening piece of an utterance, which is the
                only one allowed to clear a previous barge-in.

        Returns:
            False once the utterance has been aborted, so the caller stops
            feeding it. Never raises, like :meth:`say`.
        """
        if first:
            self._abort.clear()
        elif self._abort.is_set():
            return False
        self._speak_clean(_clean_for_tts(text, code_fallback=False))
        return not self._abort.is_set()

    def _speak_clean(self, clean: str) -> None:
        """Play already-sanitized text under VOICE_LOCK, swallowing failures.

        The single place both entry points converge on, so the guarantee that
        speech never raises at the caller is made once.
        """
        if not clean:
            return

        with VOICE_LOCK:
            try:
                self._synth_and_play(clean)
            except Exception:
                log.warning(
                    "Kokoro TTS failed; falling back to stderr print",
                    exc_info=True,
                )
                print(clean, file=sys.stderr)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _synth_sentence(self, sentence: str) -> Tuple[np.ndarray, int]:
        """Synthesize a single sentence and return (samples, sample_rate).

        Appends _TAIL_PAD_S seconds of silence to prevent Bluetooth buffer
        early-close clipping. Raises on Kokoro failure (caller handles).
        """
        samples, sample_rate = self._synth_raw(sentence)
        pitch = getattr(self, "_pitch", 0.0)
        if pitch:
            # Pitch and device rate in ONE resample (see retune()).
            target = self._device_sr or sample_rate
            samples, sample_rate = retune(samples, sample_rate, pitch, target), target
        else:
            # Resample to the output device's native rate so PortAudio does no
            # (low-quality) conversion of its own. No-op when rates already match.
            samples, sample_rate = self._resample_to_device(samples, sample_rate)
        if getattr(self, "_brightness", 0.0):
            samples = brighten(samples, self._brightness)
        # Append trailing silence to prevent BT buffer early-close
        pad_samples = int(sample_rate * _TAIL_PAD_S)
        silence = np.zeros(pad_samples, dtype=samples.dtype)
        samples = np.concatenate([samples, silence])
        return samples, sample_rate

    def _synth_raw(self, sentence: str) -> Tuple[np.ndarray, int]:
        """Kokoro's own output for *sentence*, before any character transform.

        With a pitch shift the sentence is synthesized slower by the shift
        ratio, so that :func:`retune` (which shortens it by the same ratio)
        gives back the configured speed.
        """
        self._last_synth = time.monotonic()
        speed = self._speed
        pitch = getattr(self, "_pitch", 0.0)
        if pitch:
            speed = min(2.0, max(0.5, speed / 2.0 ** (pitch / 12.0)))
        return self._kokoro.create(sentence, voice=self._voice, speed=speed, lang=self._kokoro_lang)

    def _resample_to_device(self, samples: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """Resample *samples* from *sr* to the device's native rate (polyphase).

        Returns the audio unchanged if the device rate is unknown, already
        matches, or resampling fails — always safe to play the result at the
        returned rate.
        """
        target = self._device_sr
        if not target or target == sr:
            return samples, sr
        try:
            from math import gcd

            from scipy.signal import resample_poly  # type: ignore[import]

            g = gcd(int(target), int(sr))
            up, down = int(target) // g, int(sr) // g
            out = resample_poly(samples, up, down).astype(np.float32)
            return out, target
        except Exception:
            log.warning("Resample %d->%d Hz failed; playing at native rate", sr, target, exc_info=True)
            return samples, sr

    def _synth_and_play(self, text: str) -> None:
        """Internal: synthesize and play *text* one sentence at a time, SERIALLY.

        Strictly serial: synthesize one sentence, play it to completion, then
        synthesize the next. Synthesis (heavy ONNX inference) NEVER runs while
        audio is playing — concurrent synth saturates the CPU and starves the
        PortAudio callback, which produces underruns heard as noise/glitches/
        crackle during playback. Serial costs a little first-audio latency per
        sentence but keeps playback clean.

        Abort: checks self._abort before each synth AND before each play. If set,
        breaks out cleanly (barge-in path). sd.stop() has already been called by
        interrupt(), so playback is already halted.

        Raises:
            Exception: only if EVERY sentence fails (caught by say()).
        """
        import sounddevice as sd  # type: ignore[import]

        sentences = _split_sentences(text)
        log.info("TTS speaking %d chars in %d sentence(s)", len(text), len(sentences))
        t_start = time.perf_counter()
        spoken = 0

        for i, sentence in enumerate(sentences):
            # Abort before synthesizing this sentence (barge-in).
            if self._abort.is_set():
                log.info("TTS interrupted by barge-in (before sentence %d/%d)", i + 1, len(sentences))
                break

            # Synthesize this sentence fully (blocking) — nothing else runs.
            try:
                samples, sample_rate = self._synth_sentence(sentence)
            except Exception:
                log.warning(
                    "TTS failed on sentence %d/%d: %r",
                    i + 1, len(sentences), sentence[:60], exc_info=True,
                )
                continue

            # Abort after synth, before playing (synth can take a moment).
            if self._abort.is_set():
                log.info("TTS interrupted by barge-in (before play sentence %d/%d)", i + 1, len(sentences))
                break

            # Play to completion. With no concurrent synth, the callback never
            # underruns, so the audio stays clean.
            try:
                try:
                    envelope = speech_envelope(samples, sample_rate)
                except Exception:
                    envelope = None
                self._mark_play()
                sd.play(samples, samplerate=sample_rate)
                if envelope is not None:
                    self._playing = (envelope, time.monotonic())
                try:
                    sd.wait()  # block until this sentence finishes (or sd.stop() fires)
                finally:
                    self._playing = None
                spoken += 1
            except Exception:
                log.warning(
                    "TTS playback failed on sentence %d/%d: %r",
                    i + 1, len(sentences), sentence[:60], exc_info=True,
                )
                continue

        log.info(
            "TTS done: %d/%d sentence(s) in %.1fs",
            spoken, len(sentences), time.perf_counter() - t_start,
        )
        if spoken == 0 and sentences and not self._abort.is_set():
            raise RuntimeError("all sentences failed to synthesize")
