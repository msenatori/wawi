"""TTS diagnostic: synth a sentence EXACTLY as Wawi does and save it to WAV.

Goal: isolate whether the "low-quality / noisy" sound is in the AUDIO Kokoro
produces (model / G2P problem) or in how sounddevice plays it back (24k->48k
resampling). Run this, then `afplay tts_diag.wav` (macOS native player resamples
cleanly). If the WAV ALSO sounds bad -> it's the model/voice. If the WAV sounds
clean -> it's the sounddevice playback path.

Usage:  .venv/bin/python scripts/tts_diag.py   (writes tts_diag.wav in the cwd)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wawi import config as jconfig  # noqa: E402
from wawi.voice import Voice  # noqa: E402

SENTENCE = (
    "Hola, soy Wawi. Esto es una prueba de calidad de audio para ver si la "
    "voz se escucha limpia o con ruido."
)


def main() -> None:
    cfg = jconfig.load()
    print(f"voice={cfg.tts_voice!r}  voices_dir={cfg.tts_voices_dir}  speed={cfg.tts_speed}")

    v = Voice(cfg.tts_voices_dir, cfg.tts_voice, speed=cfg.tts_speed)

    # _synth_sentence is exactly what production plays (speed + tail padding).
    samples, sr = v._synth_sentence(SENTENCE)
    samples = np.asarray(samples, dtype=np.float32)

    peak = float(np.abs(samples).max())
    clipped = int(np.sum(np.abs(samples) > 0.999))
    print(f"sample_rate={sr} Hz  dtype={samples.dtype}  n={len(samples)}  "
          f"dur={len(samples)/sr:.2f}s  peak={peak:.3f}  clipped_samples={clipped}")
    if clipped:
        print("  !! CLIPPING DETECTED — distortion is in the synthesized signal")

    out = "tts_diag.wav"
    try:
        import soundfile as sf  # type: ignore[import]
        sf.write(out, samples, sr)
    except Exception:
        # Fallback: scipy int16 WAV
        from scipy.io import wavfile  # type: ignore[import]
        pcm = np.clip(samples, -1.0, 1.0)
        wavfile.write(out, sr, (pcm * 32767).astype(np.int16))
    print(f"\nWrote {out} at {sr} Hz. Now listen with the native player:\n"
          f"    afplay {out}\n"
          "If THAT sounds clean -> the bug is in sounddevice playback (resampling).\n"
          "If THAT also sounds low-quality -> the bug is the Kokoro voice/G2P.")


if __name__ == "__main__":
    main()
