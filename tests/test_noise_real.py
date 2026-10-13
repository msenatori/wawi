"""The noise gates against the REAL Silero VAD, on synthetic noise and real speech.

The hermetic listener tests prove the logic with a fake VAD; these prove the
production thresholds (config.py defaults) against the actual model. Skipped
when openwakeword or its Silero weights are not installed. No microphone: the
listener's worker step is driven synchronously.

The speech fixtures are Kokoro TTS (voice em_alex), 16 kHz mono int16.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from wawi.config import _DEFAULTS
from wawi.listener import Listener, ListenerConfig

oww = pytest.importorskip("openwakeword")

SR = 16000
FIXTURES = Path(__file__).parent / "fixtures"
W = _DEFAULTS["wake"]


def _real_vad():
    try:
        return oww.VAD()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Silero VAD weights unavailable: {exc}")


def _wav(name: str) -> np.ndarray:
    with wave.open(str(FIXTURES / name)) as w:
        assert w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767


def _run(signal: np.ndarray) -> tuple[list[np.ndarray], Listener]:
    """Feed *signal* (with room tone around it) through a production-tuned listener."""
    captured: list[np.ndarray] = []
    listener = Listener(
        on_utterance=captured.append,
        cfg=ListenerConfig(
            engine="whisper", require_phrase=False, sample_rate=SR,
            silence_ms=600, vad_threshold=W["vad_threshold"],
            vad_onset_threshold=W["vad_onset_threshold"],
            min_onset_ms=W["min_onset_ms"], onset_window_ms=W["onset_window_ms"],
            min_voiced_ms=W["min_voiced_ms"], min_voiced_ratio=W["min_voiced_ratio"],
        ),
    )
    listener._vad = _real_vad()
    listener._reset_capture_state()
    rng = np.random.default_rng(7)
    room = lambda s: rng.normal(0, 0.001, int(s * SR)).astype(np.float32)  # noqa: E731
    audio = np.concatenate([room(1.0), signal.astype(np.float32), room(1.5)])
    for i in range(0, len(audio), 1024):
        listener._pump(audio[i:i + 1024])
    return captured, listener


def _noise(kind: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = SR * 2
    if kind == "white_loud":
        return rng.normal(0, 0.05, n)
    if kind == "white_soft":
        return rng.normal(0, 0.008, n)
    if kind == "keyboard":
        x = np.zeros(n)
        for i in range(0, n, 1600):  # a keystroke every 100 ms
            x[i:i + 80] = rng.normal(0, 0.5, 80)
        return x
    if kind == "clap":
        x = np.zeros(SR // 2)
        x[:800] = rng.normal(0, 0.6, 800) * np.exp(-np.arange(800) / 150)
        return x
    if kind == "music":
        t = np.arange(n) / SR
        return (0.1 * np.sin(2 * np.pi * 440 * t) + 0.07 * np.sin(2 * np.pi * 660 * t)
                + 0.05 * np.sin(2 * np.pi * 550 * t * (1 + 0.01 * np.sin(6 * t))))
    if kind == "silence":
        return np.zeros(n)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["white_loud", "white_soft", "keyboard", "clap", "music", "silence"])
def test_noise_never_becomes_a_turn(kind):
    captured, _ = _run(_noise(kind))
    assert captured == [], f"{kind} was dispatched as speech"


@pytest.mark.parametrize("name", ["speech_si.wav", "speech_frase.wav"])
def test_real_speech_is_dispatched(name):
    captured, _ = _run(_wav(name))
    assert len(captured) == 1
    assert captured[0].speech.voiced_ms >= W["min_voiced_ms"]


def test_a_short_reply_survives_quiet_delivery():
    # A soft "sí" -- Silero scores shape, not loudness, so it must still pass.
    captured, _ = _run(_wav("speech_si.wav") * 0.15)
    assert len(captured) == 1
