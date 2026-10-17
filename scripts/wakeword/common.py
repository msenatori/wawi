"""Shared pieces for the "Wawi" wake-word scripts: synthesis, augmentation, IO.

Everything here works on 16 kHz mono float32 in -1..1, the format the
listener, Silero VAD, openWakeWord and Whisper all consume.

Kokoro (kokoro-onnx, already a Wawi dependency) is the synthetic voice. Its
weights are expected where Wawi keeps them, ``~/.local/share/kokoro``
(override with ``KOKORO_DIR``). Kokoro has only three Spanish voices, so the
generators also borrow the Portuguese, Italian, French and English voices with
the Spanish phonemiser: they read Spanish text with a foreign accent, which is
exactly the speaker variety a wake word needs to survive.

No network access happens here. Scripts that download data say so.
"""

from __future__ import annotations

import os
import random
import wave
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000

#: Spanish voices first: they are what the owner's ears (and Whisper) expect.
SPANISH_VOICES = ("ef_dora", "em_alex", "em_santa")
#: Accented readers of the same Spanish text (lang="es" phonemiser).
ACCENT_VOICES = (
    "pf_dora", "pm_alex", "pm_santa", "if_sara", "im_nicola", "ff_siwis",
    "af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael",
    "am_puck", "bf_emma", "bm_george", "bm_lewis", "hf_alpha", "hm_omega",
)

#: How the owner actually says it, and how the name gets written down. Every
#: variant is read by every voice; the punctuation changes Kokoro's prosody
#: (a call, a question, a name in passing), which matters more than spelling.
POSITIVE_TEXTS = (
    "Wawi.", "¡Wawi!", "¿Wawi?", "Wawi...", "Guagui.", "¡Guaui!", "Uaui.",
    "Wawi, ¿qué hora es?", "Wawi, poné música.", "Wawi, ¿cómo está el clima?",
    "Che Wawi.", "Che, Wawi, ¿me escuchás?", "Hola Wawi.", "Ey, Wawi.",
    "Wawi, vení.", "Wawi, ¿qué error tiró el blog?", "Wawi, recordame algo.",
)

#: Words and phrases a careless matcher (or model) would confuse with "Wawi":
#: same vowels, same rhythm, or a common word in a Rioplatense kitchen.
CONFUSABLE_TEXTS = (
    "guau", "guau guau", "wifi", "el wifi no anda", "bueno", "bueno, dale",
    "agua", "un vaso de agua", "Huawei", "tengo un Huawei", "y vos", "¿y vos?",
    "wawa", "guagua", "huevo", "huevos", "guay", "hawaii", "Hawái", "walkie",
    "Wally", "wow", "uy", "uh", "guapa", "aguante", "ahí va", "jaguar",
)

#: Ordinary talk near the machine: none of it may wake her.
RANDOM_TEXTS = (
    "¿Qué hacemos hoy para comer?", "Mañana tengo una reunión temprano.",
    "Pasame la sal, por favor.", "El colectivo vino lleno otra vez.",
    "No sé dónde dejé las llaves.", "¿Viste el partido de anoche?",
    "Hay que comprar pan y leche.", "Me voy a dormir que estoy re cansado.",
    "La compu está lentísima hoy.", "Llamame cuando llegues.",
    "Che, ¿cerraste la puerta?", "Está lloviendo un montón.",
    "Hoy hace un calor terrible.", "¿Me prestás el cargador?",
    "El perro ladra cada vez que pasa alguien.", "Tengo que mandar ese mail.",
    "Vamos a pedir una pizza.", "Esta canción me encanta.",
    "¿A qué hora empieza la película?", "Mirá lo que encontré en internet.",
    "The meeting starts at nine.", "I think the build is broken again.",
)


def kokoro_dir() -> Path:
    return Path(os.environ.get("KOKORO_DIR", Path.home() / ".local/share/kokoro"))


_kokoro = None


def kokoro():
    """The Kokoro engine, loaded once. Raises with the fix if it is missing."""
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        d = kokoro_dir()
        model, voices = d / "kokoro-v1.0.onnx", d / "voices-v1.0.bin"
        if not model.is_file() or not voices.is_file():
            raise SystemExit(
                f"Kokoro weights not found in {d}. Put kokoro-v1.0.onnx and "
                "voices-v1.0.bin there (same files Wawi's voice uses), or set KOKORO_DIR."
            )
        _kokoro = Kokoro(str(model), str(voices))
    return _kokoro


def resample(x: np.ndarray, sr_from: int, sr_to: int = SR) -> np.ndarray:
    """Polyphase resampling (scipy), float32 out."""
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(np.asarray(x, dtype=np.float64), sr_to // g, sr_from // g).astype(
        np.float32
    )


def synth(text: str, voice: str, speed: float = 1.0) -> np.ndarray:
    """*text* spoken by Kokoro *voice*, at 16 kHz."""
    samples, sr = kokoro().create(text, voice=voice, speed=float(speed), lang="es")
    return resample(np.asarray(samples, dtype=np.float32), sr)


def pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Cheap pitch shift by resampling: also changes the length a little.

    That coupling is acceptable here (and even welcome: people who speak
    higher often speak faster); what matters is that the formants move, so a
    model does not learn three voices instead of a word.
    """
    if not semitones:
        return x
    factor = 2.0 ** (semitones / 12.0)
    shifted = resample(x, int(round(SR * factor)), SR)
    return shifted.astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float | None, rng: np.random.Generator,
              noise: np.ndarray | None = None) -> np.ndarray:
    """Mix in *noise* (or pink-ish noise) at *snr_db*. None = clean."""
    if snr_db is None:
        return x
    if noise is None or noise.size == 0:
        white = rng.normal(0.0, 1.0, x.size)
        # A one-pole low-pass makes it room-like rather than hiss.
        from scipy.signal import lfilter

        noise = lfilter([0.03], [1.0, -0.97], white)
    else:
        if noise.size < x.size:
            noise = np.tile(noise, int(np.ceil(x.size / noise.size)))
        start = int(rng.integers(0, noise.size - x.size + 1))
        noise = noise[start:start + x.size]
    p_sig = float(np.mean(x ** 2)) + 1e-12
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10.0)))
    return (x + scale * noise).astype(np.float32)


def reverb(x: np.ndarray, rng: np.random.Generator, rt60: float = 0.3) -> np.ndarray:
    """A synthetic room: exponentially decaying noise as the impulse response."""
    from scipy.signal import fftconvolve

    n = max(1, int(rt60 * SR))
    ir = rng.normal(0, 1, n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0 / 0.3
    y = fftconvolve(x, ir * 0.3)[: x.size]
    peak = float(np.abs(y).max()) or 1.0
    return (y / peak * float(np.abs(x).max() or 1.0)).astype(np.float32)


def pad(x: np.ndarray, before_s: float, after_s: float, rng: np.random.Generator,
        floor: float = 0.001) -> np.ndarray:
    """Room tone around a clip, like a microphone hears it."""
    def room(s: float) -> np.ndarray:
        return rng.normal(0, floor, int(s * SR)).astype(np.float32)
    return np.concatenate([room(before_s), x.astype(np.float32), room(after_s)])


def gain(x: np.ndarray, db: float) -> np.ndarray:
    y = x * (10 ** (db / 20.0))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class Variant:
    text: str
    voice: str
    speed: float
    semitones: float
    snr_db: float | None
    reverb: bool

    @property
    def tag(self) -> str:
        snr = "clean" if self.snr_db is None else f"snr{int(self.snr_db)}"
        return f"{self.voice}_s{self.speed:.2f}_p{self.semitones:+.0f}_{snr}{'_rv' if self.reverb else ''}"


def variants(texts: Sequence[str], voices: Sequence[str], n: int, seed: int,
             speeds: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.25),
             semitones: Sequence[float] = (-2, -1, 0, 0, 1, 2),
             snrs: Sequence[float | None] = (None, None, 20, 10, 5),
             reverb_p: float = 0.3) -> list[Variant]:
    """*n* random (text, voice, speed, pitch, noise) combinations, reproducible."""
    r = random.Random(seed)
    return [
        Variant(r.choice(list(texts)), r.choice(list(voices)), r.choice(list(speeds)),
                r.choice(list(semitones)), r.choice(list(snrs)), r.random() < reverb_p)
        for _ in range(n)
    ]


def render(v: Variant, rng: np.random.Generator, noise: np.ndarray | None = None) -> np.ndarray:
    """The audio for one variant, padded with room tone."""
    x = synth(v.text, v.voice, v.speed)
    x = pitch_shift(x, v.semitones)
    if v.reverb:
        x = reverb(x, rng)
    x = gain(x / (float(np.abs(x).max()) or 1.0) * 0.5, float(rng.uniform(-12, 0)))
    x = add_noise(x, v.snr_db, rng, noise)
    return pad(x, float(rng.uniform(0.2, 0.6)), float(rng.uniform(0.3, 0.8)), rng)


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    """16-bit mono WAV at 16 kHz -> float32. Other formats via soundfile."""
    try:
        with wave.open(str(path)) as w:
            if w.getsampwidth() == 2 and w.getnchannels() == 1:
                data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                return resample(data.astype(np.float32) / 32767, w.getframerate())
    except wave.Error:
        pass
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), sr)


def wavs(folder: Path) -> Iterable[Path]:
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in exts)
