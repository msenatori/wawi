#!/usr/bin/env python3
"""Make Wawi's three little sounds — bark, whine, howl — from nothing but math.

    python scripts/make_wawi_sounds.py          # writes wawi/assets/sounds/*.wav

Why synthesise instead of recording: there was no recording of the real dog
yet, and a downloaded "dog bark" sample comes with a licence to check and
sounds like somebody else's dog. These are deliberately small and cartoonish:
a soft "wuf", a short whine, a short howl, all quiet (peak about -12 dBFS) and
short (bark and whine < 0.6 s, howl < 1.5 s). A pet that makes LOUD noises on its own is
a pet that gets muted.

How they are built (each is a sum of a few harmonics on a moving pitch, i.e.
a tiny additive voice, shaped like the real thing):

- bark: a fast pitch DROP (~520 -> 300 Hz) under a 6 ms attack and a quick
  exponential decay, plus a breath of filtered noise at the start — the drop
  plus the noise burst is what reads as "woof" rather than "beep". Two
  harmonics are boosted around 1 kHz, roughly where a dog's "o" formant sits.
- whine: a thin, mostly-sine voice that rises and falls (~750 -> 1050 ->
  820 Hz) with a 6 Hz wobble and a slow attack: the soft nasal "hnnn".
- howl: "aooo" — a slow rise (~380 -> 620 Hz), a held note with vibrato,
  and a sag at the end; darker harmonics so it stays gentle.

Everything is deterministic (fixed seed), so running the script twice gives
byte-identical files and the tests can check that.

Replacing them with the real dog
--------------------------------
Record short clips and save them over these files, same names:
``wawi/assets/sounds/bark.wav``, ``whine.wav``, ``howl.wav``. Any format
NSSound plays works if you keep the name and change the extension (``.aiff``,
``.caf``, ``.m4a``, ``.mp3``), but then delete the ``.wav`` of the same name:
``.aiff`` is looked up before ``.wav`` (see ``wawi/pet/acting.py``,
``sound_path``). Keep them short and quiet. ``pet: {sounds: false}`` in
config.yaml silences them all; quiet mode (meetings, quiet hours) already
does.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

RATE = 22050
#: Peak level, linear (0.25 ≈ -12 dBFS): audible on laptop speakers, never
#: loud. The sustained whine and howl carry far more energy per peak than the
#: clipped bark, so they get a lower peak to land at a similar loudness (RMS).
PEAK = 0.25
PEAK_SUSTAINED = 0.13
SEED = 7
OUT_DIR = Path(__file__).resolve().parent.parent / "wawi" / "assets" / "sounds"


def _time(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * RATE)) / RATE


def _voice(freq: np.ndarray, weights: list[float]) -> np.ndarray:
    """Additive voice: harmonics k*f with the given weights, on a gliding f."""
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    out = np.zeros_like(freq)
    for k, w in enumerate(weights, start=1):
        if w:
            out += w * np.sin(k * phase)
    return out


def _lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    """One-pole low-pass: takes the hiss off the noise and the edge off harmonics."""
    a = np.exp(-2 * np.pi * cutoff / RATE)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def _fade(x: np.ndarray, fade_in: float, fade_out: float) -> np.ndarray:
    """Raised-cosine edges, so no file starts or ends with a click."""
    n_in, n_out = max(1, int(fade_in * RATE)), max(1, int(fade_out * RATE))
    env = np.ones_like(x)
    env[:n_in] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_in))
    env[-n_out:] = np.minimum(env[-n_out:], 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n_out)))
    return x * env


def _normalise(x: np.ndarray, level: float = PEAK) -> np.ndarray:
    peak = float(np.max(np.abs(x))) or 1.0
    return x * (level / peak)


def bark() -> np.ndarray:
    """A soft cartoon "wuf": 0.32 s."""
    rng = np.random.default_rng(SEED)
    t = _time(0.32)
    # Pitch falls fast then levels off: the shape of a bark's vowel.
    freq = 300 + 220 * np.exp(-t / 0.05)
    # Formant-ish weights: strong 2nd-3rd harmonics (~0.6-1.5 kHz).
    tone = _voice(freq, [0.6, 1.0, 0.7, 0.35, 0.15, 0.06])
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.075)
    # Low-passed twice: a soft breath, not hiss.
    noise = _lowpass(_lowpass(rng.standard_normal(len(t)), 1200.0), 1200.0)
    breath = noise / (np.max(np.abs(noise)) or 1.0) * np.exp(-t / 0.03) * 0.35
    return _normalise(_fade(tone * env + breath, 0.002, 0.04))


def whine() -> np.ndarray:
    """A short questioning whine: 0.55 s."""
    t = _time(0.55)
    shape = np.sin(np.pi * np.clip(t / 0.55, 0, 1)) ** 0.8  # up, then back down
    freq = 760 + 290 * shape - 60 * (t / 0.55)
    freq *= 1 + 0.012 * np.sin(2 * np.pi * 6.0 * t)  # the wobble
    tone = _voice(freq, [1.0, 0.18, 0.07])
    env = (1 - np.exp(-t / 0.05)) * (1 - 0.35 * (t / 0.55))
    return _normalise(_fade(tone * env, 0.03, 0.12), PEAK_SUSTAINED)


def howl() -> np.ndarray:
    """A small "aooo": 1.35 s."""
    t = _time(1.35)
    rise = 1 - np.exp(-t / 0.18)
    sag = np.clip((t - 1.0) / 0.35, 0, 1) ** 2
    freq = 380 + 240 * rise - 120 * sag
    # Vibrato grows in as the note is held, like a real howl.
    freq *= 1 + 0.018 * np.clip((t - 0.3) / 0.4, 0, 1) * np.sin(2 * np.pi * 5.0 * t)
    tone = _voice(freq, [1.0, 0.45, 0.2, 0.08])  # dark "ooo"
    env = (1 - np.exp(-t / 0.12)) * (1 - 0.5 * sag)
    return _normalise(_fade(tone * env, 0.05, 0.3), PEAK_SUSTAINED)


SOUNDS = {"bark": bark, "whine": whine, "howl": howl}


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).round().astype("<i2").tobytes()


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(to_pcm16(x))


def main(out_dir: Path = OUT_DIR) -> list[Path]:
    written = []
    for name, make in SOUNDS.items():
        path = out_dir / f"{name}.wav"
        write_wav(path, make())
        written.append(path)
        print(f"{path}  ({len(make()) / RATE:.2f} s)")
    return written


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR)
