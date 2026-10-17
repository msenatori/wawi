#!/usr/bin/env python3
"""Synthetic "Wawi" clips for training the wake-word model (Kokoro).

Every clip is the wake word said on its own or after a short call ("che",
"hola", "ey"), across 21 Kokoro voices (Spanish plus accented readers of the
same Spanish text), five speeds, +-2 semitones, reverb, and noise at 20/10/5
dB SNR -- with real speech babble as the noise when --noise is given (see
features.py --make-noise). The model learns the END of the word: the
feature step aligns each clip so the word finishes near the end of its
window, which is when the streaming detector must fire.

The owner's own recordings (record_wawi.py) matter more than any amount of
these; train.py takes both folders.

    python scripts/wakeword/gen_positives.py --out data/pos_train --n 4000 --seed 1
    python scripts/wakeword/gen_positives.py --out data/pos_test --n 400 --seed 99 --eval
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from synth_pool import render_all  # noqa: E402

#: What a training positive says. The word must END the clip: anything after
#: it would put the wrong audio at the end of the model's window.
TRAIN_TEXTS = (
    "Wawi.", "Wawi.", "¡Wawi!", "¡Wawi!", "¿Wawi?", "Wawi...", "Wawi, Wawi.",
    "Che, Wawi.", "Che Wawi.", "Hola, Wawi.", "Ey, Wawi.", "Dale, Wawi.",
    "Oíme, Wawi.", "¡Eh, Wawi!", "Uaui.", "¡Guaui!",
)

#: Held-out positives include the single-breath form ("Wawi, ¿qué hora es?"):
#: the evaluator streams the whole clip and the word is in the middle of it.
EVAL_TEXTS = TRAIN_TEXTS + (
    "Wawi, ¿qué hora es?", "Wawi, poné música.", "Che, Wawi, ¿me escuchás?",
    "Wawi, ¿cómo está el clima?",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval", action="store_true", help="held-out set (adds sentence forms)")
    ap.add_argument("--noise", type=str, default=None,
                    help="float32 .npy of background audio at 16 kHz to mix in")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    voices = C.SPANISH_VOICES * 6 + C.ACCENT_VOICES
    texts = EVAL_TEXTS if args.eval else TRAIN_TEXTS
    items = C.variants(texts, voices, args.n, args.seed,
                       speeds=(0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3),
                       semitones=(-3, -2, -1, 0, 0, 1, 2, 3),
                       snrs=(None, None, 25, 20, 15, 10, 5), reverb_p=0.35)
    render_all(items, args.out, args.seed, args.workers, args.threads, args.noise)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
