"""Sine-wave earcon observer for state transitions.

Plays a short ascending beep on LISTENING (mic opens) and a descending beep
on THINKING (mic closes). Generated in code — no asset files required.

Parameters (from design assumption #4):
  Sample rate : 44 100 Hz
  Amplitude   : 0.25  (prevents clipping)
  Tone length : 90 ms each
  Fade        : 8 ms linear in/out per tone (kills clicks)
  Ascending   : 660 Hz → 990 Hz  (LISTENING)
  Descending  : 880 Hz → 587 Hz  (THINKING)
  Playback    : non-blocking (sd.play without sd.wait)
"""

from __future__ import annotations

import logging

import numpy as np

from wawi.state import State

log = logging.getLogger(__name__)

_SR = 44_100          # sample rate (Hz)
_AMP = 0.25           # amplitude (0–1)
_TONE_MS = 90         # duration of each tone in milliseconds
_FADE_MS = 8          # linear fade in/out in milliseconds

# Earcon frequency pairs (Hz): each tuple is (tone_A, tone_B)
_ASCENDING = (660, 990)   # LISTENING — mic opens
_DESCENDING = (880, 587)  # THINKING  — mic closes


def _sine_tone(freq: float, ms: int, sample_rate: int, amplitude: float, fade_ms: int) -> np.ndarray:
    """Generate a single sine tone with linear fade in/out.

    Args:
        freq: Frequency in Hz.
        ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.
        amplitude: Peak amplitude (0–1).
        fade_ms: Fade in/out duration in milliseconds.

    Returns:
        Float32 numpy array.
    """
    n = int(sample_rate * ms / 1000)
    t = np.linspace(0, ms / 1000, n, endpoint=False, dtype=np.float32)
    tone = amplitude * np.sin(2 * np.pi * freq * t)

    # Apply fade
    fade_n = int(sample_rate * fade_ms / 1000)
    if fade_n > 0 and 2 * fade_n <= n:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        tone[:fade_n] *= ramp
        tone[-fade_n:] *= ramp[::-1]

    return tone


def _build_earcon(freq_a: float, freq_b: float) -> np.ndarray:
    """Concatenate two sine tones into a stereo-ready float32 array."""
    tone_a = _sine_tone(freq_a, _TONE_MS, _SR, _AMP, _FADE_MS)
    tone_b = _sine_tone(freq_b, _TONE_MS, _SR, _AMP, _FADE_MS)
    return np.concatenate([tone_a, tone_b])


# Pre-build earcons at import time (cheap, avoids per-turn allocation)
_EARCON_LISTENING = _build_earcon(*_ASCENDING)
_EARCON_THINKING = _build_earcon(*_DESCENDING)


class EarconObserver:
    """State observer that plays sine earcons on state transitions.

    Args:
        enabled: If False, __call__ is a no-op (ui.sound_cues=false path).
        sample_rate: Playback sample rate. Must match the generated earcon arrays.
    """

    def __init__(self, enabled: bool, sample_rate: int = _SR) -> None:
        self.enabled = enabled
        self._sample_rate = sample_rate

    def __call__(self, old: State, new: State) -> None:  # noqa: ARG002
        """React to (old, new) state transitions.

        Plays ascending earcon on → LISTENING, descending on → THINKING.
        Non-blocking: sd.play() is called without sd.wait() so the pipeline
        is not stalled by earcon playback.
        """
        if not self.enabled:
            return

        if new is State.LISTENING:
            self._play(_EARCON_LISTENING)
        elif new is State.THINKING:
            self._play(_EARCON_THINKING)

    def _play(self, buf: np.ndarray) -> None:
        """Play *buf* non-blocking via sounddevice.

        Any playback error is logged as WARNING and suppressed so a broken
        audio device never kills the pipeline.
        """
        try:
            import sounddevice as sd  # type: ignore[import]

            sd.play(buf, samplerate=self._sample_rate)
            # No sd.wait() — earcons play in the background
        except Exception:
            log.warning("Earcon playback failed", exc_info=True)
