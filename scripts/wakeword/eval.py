#!/usr/bin/env python3
"""Evaluate a wake-word model: recall and false accepts per hour.

Everything goes through the REAL ``openwakeword.Model`` the listener uses,
fed 1280-sample (80 ms) int16 chunks, so the numbers are what Wawi would see:

  positives   each held-out clip streamed on its own, with 1 s of room tone
              before and after; a hit is any score >= threshold.
  negatives   held-out audio streamed as one long recording; an activation is
              a score crossing the threshold, debounced by --refractory
              seconds (the listener opens one capture per wake). Reported
              per hour of audio.
  confusable  the hard-negative clips ("guau", "Huawei", "wow"…), each on its
              own like the positives: the share that would wake her.

    python scripts/wakeword/eval.py --model wawi/assets/wakeword/wawi.onnx \\
        --pos data/pos_test --confusable data/neg_test/hard \\
        --neg data/bg/fleurs-es419-test data/bg/librispeech-test-clean data/neg_test/speech \\
        --neg-hours 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)
CHUNK = 1280
_model = None
_key = ""


def _init(model_path: str) -> None:
    import openwakeword

    global _model, _key
    _model = openwakeword.Model(wakeword_models=[model_path], inference_framework="onnx")
    _key = Path(model_path).stem


def _stream(x: np.ndarray) -> np.ndarray:
    _model.reset()
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    n = pcm.size // CHUNK
    return np.array([_model.predict(pcm[i * CHUNK:(i + 1) * CHUNK])[_key] for i in range(n)],
                    dtype=np.float32)


def _clip_scores(path: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(path)) % 2**32)
    x = C.pad(C.read_wav(Path(path)), 1.0, 1.0, rng)
    return _stream(x)


def sustained(scores: np.ndarray, patience: int) -> np.ndarray:
    """Scores held for *patience* consecutive 80 ms frames (their running min).

    patience=1 is the raw score. openwakeword.Model's ``patience`` argument
    does the same: a single-frame spike stops counting as a wake.
    """
    if patience <= 1 or scores.size < patience:
        return scores if patience <= 1 else np.zeros(0, np.float32)
    n = scores.size - patience + 1
    return np.minimum.reduce([scores[i:i + n] for i in range(patience)])


def _peak(scores: np.ndarray, patience: int) -> float:
    s = sustained(scores, patience)
    return float(s.max()) if s.size else 0.0


def _long_scores(paths: list[str]) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(0)
    parts = []
    for p in paths:
        try:
            parts.append(C.read_wav(Path(p)))
        except Exception:
            continue
        parts.append(rng.normal(0, 0.0008, int(0.3 * C.SR)).astype(np.float32))
    x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return _stream(x), x.size / C.SR


def activations(scores: np.ndarray, thr: float, refractory_frames: int) -> int:
    count, last = 0, -10**9
    for i in np.nonzero(scores >= thr)[0]:
        if i - last > refractory_frames:
            count += 1
            last = i
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pos", type=Path, nargs="+", required=True)
    ap.add_argument("--neg", type=Path, nargs="+", required=True)
    ap.add_argument("--confusable", type=Path, nargs="*", default=[])
    ap.add_argument("--neg-hours", type=float, default=3.0, help="cap on negative audio streamed")
    ap.add_argument("--refractory", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--patience", type=int, nargs="+", default=[1, 2, 3],
                    help="consecutive frames >= threshold needed to wake (reported for each)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    pos = [str(f) for d in args.pos for f in C.wavs(d)]
    conf = [str(f) for d in args.confusable for f in C.wavs(d)]
    rng = np.random.default_rng(1)
    # Each negative source gets an equal share of the budget and is reported
    # on its own: "5 false wakes an hour" means nothing until you know whether
    # they came from Spanish talk, English talk or a fan.
    budget = args.neg_hours * 3600 * C.SR * 2 / max(1, len(args.neg))
    jobs: list[tuple[str, list[str]]] = []
    for d in args.neg:
        files = [str(f) for f in C.wavs(d)]
        rng.shuffle(files)
        chosen, total = [], 0
        for f in files:
            chosen.append(f)
            # Size as a proxy for duration (16-bit WAV; FLAC is ~half, so
            # compressed sources get somewhat more audio). Measured exactly below.
            total += os.path.getsize(f) * (2 if f.endswith(".flac") else 1)
            if total >= budget:
                break
        jobs += [(d.name, chosen[i::args.workers]) for i in range(args.workers)]

    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
        pos_s = list(pool.map(_clip_scores, pos, chunksize=16))
        conf_s = list(pool.map(_clip_scores, conf, chunksize=16)) if conf else []
        long = list(pool.map(_long_scores, [paths for _, paths in jobs]))
    sources = [name for name, _ in jobs]
    hours = sum(h for _, h in long) / 3600
    by_src_h: dict[str, float] = {}
    for name, (_, h) in zip(sources, long):
        by_src_h[name] = by_src_h.get(name, 0.0) + h / 3600
    refr = int(args.refractory / 0.08)

    rows = []
    print(f"model {args.model}  positives {len(pos)}  confusable {len(conf)}  negatives {hours:.2f} h")
    print("  " + ", ".join(f"{k} {v:.2f} h" for k, v in by_src_h.items()))
    conf_max = np.array([_peak(s, 1) for s in conf_s]) if conf_s else np.zeros(0)
    for pat in args.patience:
        pos_max = np.array([_peak(s, pat) for s in pos_s])
        cmax = np.array([_peak(s, pat) for s in conf_s]) if conf_s else np.zeros(0)
        print(f"\npatience {pat} ({pat * 80} ms above threshold)")
        print(f"{'thr':>5} {'recall':>7} {'FA/h':>7} {'confusable':>10}  per source FA/h")
        for thr in THRESHOLDS:
            fa_src: dict[str, int] = {}
            for name, (sc, _) in zip(sources, long):
                fa_src[name] = fa_src.get(name, 0) + activations(sustained(sc, pat), thr, refr)
            fa = sum(fa_src.values())
            row = {"patience": pat, "threshold": thr, "recall": float(np.mean(pos_max >= thr)),
                   "fa_per_hour": fa / hours if hours else float("nan"),
                   "fa_per_hour_by_source": {k: v / by_src_h[k] for k, v in fa_src.items()
                                             if by_src_h[k]},
                   "confusable_rate": float(np.mean(cmax >= thr)) if cmax.size else None}
            rows.append(row)
            cr = "-" if row["confusable_rate"] is None else f"{row['confusable_rate']:.1%}"
            per = " ".join(f"{k}={v:.1f}" for k, v in row["fa_per_hour_by_source"].items())
            print(f"{thr:5.2f} {row['recall']:7.1%} {row['fa_per_hour']:7.2f} {cr:>10}  {per}")
    if conf:
        # Which confusable words wake her, at the middle threshold.
        by_word: dict[str, list[float]] = {}
        man = {}
        for d in args.confusable:
            mf = Path(d) / "manifest.jsonl"
            if mf.is_file():
                for line in mf.read_text().splitlines():
                    r = json.loads(line)
                    if "file" in r:
                        man[str(Path(d) / r["file"])] = r["text"]
        for f, s in zip(conf, conf_max):
            by_word.setdefault(man.get(f, Path(f).stem), []).append(float(s))
        worst = sorted(by_word.items(), key=lambda kv: -np.mean(kv[1]))[:12]
        print("most wake-like confusables (mean max score):")
        for w, s in worst:
            print(f"   {np.mean(s):.2f}  {w!r} (n={len(s)})")
    print(f"eval took {time.time() - t0:.0f}s")
    if args.json:
        args.json.write_text(json.dumps({"hours": hours, "hours_by_source": by_src_h, "positives": len(pos),
                                         "confusable": len(conf), "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
