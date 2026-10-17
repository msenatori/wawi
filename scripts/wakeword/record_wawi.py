#!/usr/bin/env python3
"""Record your own "Wawi" samples for the wake-word model. Run on the Mac.

Synthetic voices get the model started; YOUR voice, your microphone and your
room are what make it reliable for you. 50-100 takes are plenty. Each take
is 2 s: wait for the prompt, say the line once, the way you would call her.
Vary it on purpose: louder and softer, closer and farther from the mic, tired,
in a hurry, from across the room, with the TV on.

Keys after each take: Enter = keep, r = redo, p = play it back, q = quit.

    .venv/bin/python scripts/wakeword/record_wawi.py --out data/pos_owner --n 80
    .venv/bin/python scripts/wakeword/record_wawi.py --out data/pos_owner_test --n 20

The first folder trains (run_all.sh picks up data/pos_owner automatically);
keep a separate held-out folder (pos_owner_test) so the evaluation also
reports recall on your real voice. Recordings stay local: *.wav is gitignored.
Optionally record NEGATIVES too (--prompts negatives): the words you say a lot
that sound close ("guau", "uy", "bueno"...); put them in data/neg_owner and add
it to train.py --hard.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

POSITIVE_PROMPTS = (
    "Wawi", "Wawi (más fuerte)", "Wawi (bajito)", "¿Wawi?", "Che, Wawi",
    "Wawi (desde lejos)", "Hola, Wawi", "Wawi (rápido)", "Wawi (cansado)",
    "Ey, Wawi", "Wawi (llamándola)", "Wawi, Wawi",
)
NEGATIVE_PROMPTS = (
    "guau", "uy", "bueno", "wifi", "Huawei", "wow", "agua", "y vos", "guay",
    "dale", "¿qué?", "Javi",
)


def record(seconds: float) -> np.ndarray:
    import sounddevice as sd

    x = sd.rec(int(seconds * C.SR), samplerate=C.SR, channels=1, dtype="float32")
    sd.wait()
    return x.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/pos_owner"))
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--prompts", choices=("positives", "negatives"), default="positives")
    args = ap.parse_args()

    import sounddevice as sd

    prompts = POSITIVE_PROMPTS if args.prompts == "positives" else NEGATIVE_PROMPTS
    args.out.mkdir(parents=True, exist_ok=True)
    start = len(list(args.out.glob("*.wav")))
    print(f"Grabando en {args.out} (ya hay {start}). Enter = guardar, r = repetir, "
          "p = escuchar, q = salir.\n")
    taken = 0
    for prompt in itertools.cycle(prompts):
        if taken >= args.n:
            break
        while True:
            input(f"[{start + taken + 1}/{start + args.n}] Decí: «{prompt}» — Enter y hablá…")
            time.sleep(0.15)
            x = record(args.seconds)
            peak = float(np.abs(x).max())
            warn = "  (muy bajo, ¿hablaste?)" if peak < 0.02 else "  (saturado)" if peak > 0.98 else ""
            ans = input(f"  pico {peak:.2f}{warn}  [Enter/r/p/q] ").strip().lower()
            while ans == "p":
                sd.play(x, C.SR)
                sd.wait()
                ans = input("  [Enter/r/q] ").strip().lower()
            if ans == "q":
                print(f"Listo: {taken} tomas nuevas en {args.out}")
                return 0
            if ans == "r":
                continue
            slug = "".join(ch for ch in prompt.lower() if ch.isalnum())[:20] or "take"
            C.write_wav(args.out / f"owner_{start + taken:04d}_{slug}.wav", x)
            taken += 1
            break
    print(f"Listo: {taken} tomas nuevas en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
