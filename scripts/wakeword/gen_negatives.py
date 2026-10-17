#!/usr/bin/env python3
"""Synthetic negatives for the wake-word model: near-misses, talk, and noise.

Three folders under --out:

    hard/    the words a "Wawi" detector confuses ("guau", "Huawei", "wow",
             "Wally", "guay", "uy", "Hawái", "wifi", "agua", "bueno"…), alone
             and inside sentences. train.py weights these up.
    speech/  ordinary Rioplatense talk (templated sentences, many voices).
    noise/   no Kokoro at all: white/pink/brown noise, hum, clicks, tones,
             chords and silence, at many levels.

Real recorded speech (LibriSpeech, FLEURS) is the other half of the negative
data and comes from fetch_background.py.

    python scripts/wakeword/gen_negatives.py --out data/neg_train --hard 2500 --speech 2500 --noise 600 --seed 2
    python scripts/wakeword/gen_negatives.py --out data/neg_test --hard 400 --speech 300 --noise 100 --seed 98
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from synth_pool import render_all  # noqa: E402

#: The confusable words again, plus more of them and in context. Each is a
#: real risk: "guau" is her own bark and the Spanish "wow", "uy" and "bueno"
#: are said all day, "Huawei" and "wifi" are tech talk near a computer.
HARD_TEXTS = C.CONFUSABLE_TEXTS + (
    "¡guau!", "¡uy!", "uy, no", "¡wow!", "guau, qué lindo", "mi Huawei se apagó",
    "se cortó el wifi", "¿hay wifi?", "Walter", "Wanda", "Waldo", "Hawái es lindo",
    "¡guay!", "una guagua", "un huevito", "Javi", "Lavi", "Davi", "Gabi",
    "Yaguarón", "guarani", "Paraguay", "Uruguay", "wasabi", "Wakanda", "Kiwi",
    "Tavi", "Bawi", "Mauri", "¡Mauri!", "Rawi", "kawaii", "Malawi", "wagyu",
    "guacho", "guaso", "ahí", "¡ay!", "ay, Dios", "huy", "¿y vos qué?",
    "¡vamos!", "¡bien!", "¡guau, guau!", "awww", "jajaja", "¿qué?",
)

_SUBJECTS = ("Yo", "Mi hermana", "El jefe", "Mi vieja", "Juan", "La vecina",
             "Nosotros", "Ustedes", "El técnico", "Mi amigo")
_VERBS = ("quiere", "necesita", "va a", "no puede", "tiene que", "prefiere",
          "se olvidó de", "empezó a", "dejó de", "piensa")
_ACTIONS = ("comprar pan", "arreglar la compu", "llamar al médico", "ver la serie",
            "pagar la luz", "cocinar fideos", "salir a correr", "mandar el mail",
            "ordenar la pieza", "lavar los platos", "estudiar para el examen",
            "pasear al perro", "subir las fotos", "cambiar la contraseña")
_TAILS = ("hoy", "mañana", "a la tarde", "antes de las ocho", "el fin de semana",
          "otra vez", "si llueve", "cuando pueda", "", "")


def sentences(seed: int, n: int) -> list[str]:
    """*n* templated sentences, plus the fixed ones in common.RANDOM_TEXTS."""
    combos = list(itertools.product(_SUBJECTS, _VERBS, _ACTIONS, _TAILS))
    random.Random(seed).shuffle(combos)
    out = [f"{s} {v} {a} {t}".strip() + "." for s, v, a, t in combos[:n]]
    return list(C.RANDOM_TEXTS) + out


def noise_clips(out: Path, n: int, seed: int) -> None:
    """Non-speech negatives, rendered with numpy only."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    kinds = ("white", "pink", "brown", "hum", "clicks", "tone", "chord", "silence")
    for i in range(n):
        kind = kinds[i % len(kinds)]
        secs = float(rng.uniform(1.5, 4.0))
        t = np.arange(int(secs * C.SR)) / C.SR
        w = rng.normal(0, 1, t.size)
        if kind == "white":
            x = w
        elif kind == "pink":
            x = lfilter([0.05], [1, -0.95], w)
        elif kind == "brown":
            x = lfilter([0.01], [1, -0.995], w)
        elif kind == "hum":
            f0 = rng.choice([50.0, 60.0])
            x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        elif kind == "clicks":
            x = np.zeros(t.size)
            for p in rng.integers(0, t.size - 200, int(secs * rng.uniform(3, 15))):
                x[p:p + 120] += rng.normal(0, 1, 120) * np.exp(-np.arange(120) / 25)
        elif kind == "tone":
            f = rng.uniform(200, 2000)
            x = np.sin(2 * np.pi * f * t + 3 * np.sin(2 * np.pi * rng.uniform(0.5, 6) * t))
        elif kind == "chord":
            base = rng.uniform(110, 440)
            x = sum(np.sin(2 * np.pi * base * r * t) for r in (1, 1.26, 1.5))
            x *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1, 4) * t) ** 2
        else:
            x = np.zeros(t.size)
        x = np.asarray(x, dtype=np.float32)
        peak = float(np.abs(x).max()) or 1.0
        x = x / peak * float(10 ** (rng.uniform(-40, -6) / 20))
        x = x + rng.normal(0, 0.001, x.size).astype(np.float32)
        C.write_wav(out / f"{i:05d}_{kind}.wav", x)
    print(f"{n} noise clips in {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hard", type=int, default=2500)
    ap.add_argument("--speech", type=int, default=2500)
    ap.add_argument("--noise", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--noise-bg", type=str, default=None,
                    help="float32 .npy of background audio at 16 kHz to mix in")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    voices = C.SPANISH_VOICES * 6 + C.ACCENT_VOICES
    common_aug = dict(speeds=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                      semitones=(-3, -2, -1, 0, 0, 1, 2, 3),
                      snrs=(None, None, 25, 20, 15, 10, 5), reverb_p=0.35)
    if args.hard:
        items = C.variants(HARD_TEXTS, voices, args.hard, args.seed, **common_aug)
        render_all(items, args.out / "hard", args.seed, args.workers, args.threads, args.noise_bg)
    if args.speech:
        texts = sentences(args.seed, 400)
        items = C.variants(texts, voices, args.speech, args.seed + 1, **common_aug)
        render_all(items, args.out / "speech", args.seed + 1, args.workers, args.threads,
                   args.noise_bg)
    if args.noise:
        noise_clips(args.out / "noise", args.noise, args.seed + 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
