"""Throwaway diagnostic: isolate Kokoro SYNTHESIS from sounddevice PLAYBACK.

    .venv/bin/python scripts/tts_test.py
    afplay tts_test.wav

Reads the printed peak amplitude:
  - peak ~0.0      -> Kokoro generated SILENCE -> synthesis is broken.
  - peak ~0.1-1.0  -> audio is real. If afplay is silent it's a player/format
                      issue; if afplay sounds cut it's Kokoro quality.
WAV is written as PCM_16 (universal) so afplay always plays it.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

VOICES_DIR = Path.home() / ".local/share/kokoro"

TEXT = (
    "Hola, soy Wawi. Estoy mirando tu pantalla y te respondo lo que necesites. "
    "Esta es una prueba larga para ver si el audio se corta o no al final."
)


def main() -> None:
    kokoro = Kokoro(
        str(VOICES_DIR / "kokoro-v1.0.onnx"),
        str(VOICES_DIR / "voices-v1.0.bin"),
    )
    samples, sample_rate = kokoro.create(TEXT, voice="ef_dora", speed=1.0, lang="es")
    samples = np.asarray(samples, dtype=np.float32)

    dur = len(samples) / sample_rate
    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    print(f"samples={len(samples)}  dur={dur:.1f}s  sr={sample_rate}  peak={peak:.4f}")
    if peak < 0.01:
        print("  -> SILENT: Kokoro produced no real audio (synthesis problem)")

    # Write as 16-bit PCM so afplay/QuickTime always open it.
    sf.write("tts_test.wav", samples, sample_rate, subtype="PCM_16")
    print("wrote tts_test.wav  —  now run:  afplay tts_test.wav")


if __name__ == "__main__":
    main()
