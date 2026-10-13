#!/usr/bin/env python3
"""Offline check of the Smart Turn end-of-turn model. Never opens the mic.

Answers three questions, each with real numbers from THIS machine:

    --bench        how long does one inference take on an 8 s window?
                   (median / p90 over N runs, preprocessing included)
    --parity       does wawi.turn's numpy log-mel match the reference
                   transformers.WhisperFeatureExtractor upstream uses?
                   Needs `pip install transformers` in a SCRATCH env; Wawi
                   itself does not depend on it.
    WAV files      P(complete) for each file, with silence appended the way the
                   listener sees it at the moment it asks (--pause-ms). Files
                   whose name starts with "complete" / "incomplete" are scored
                   against the threshold and an accuracy line is printed.

Usage:
    .venv/bin/python scripts/bench_smart_turn.py --bench
    .venv/bin/python scripts/bench_smart_turn.py --parity clips/*.wav
    .venv/bin/python scripts/bench_smart_turn.py --pause-ms 300 clips/*.wav

The model must be downloaded first:
    python -c 'from wawi.turn import download_model; download_model()'
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from wawi import turn  # noqa: E402

SR = turn.SAMPLE_RATE


def _load(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype(np.float32)
    return audio


def _trim(audio: np.ndarray, level: float = 0.005) -> np.ndarray:
    idx = np.flatnonzero(np.abs(audio) >= level)
    return audio[idx[0] : idx[-1] + 1] if idx.size else audio


def bench(model: turn.SmartTurn, runs: int) -> None:
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(turn.WINDOW_S * SR) * 0.05).astype(np.float32)
    feats_ms, total_ms = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        turn.whisper_features(audio)
        feats_ms.append((time.perf_counter() - t0) * 1000)
        _, ms = model.timed_predict(audio)
        total_ms.append(ms)
    q = statistics.quantiles(total_ms, n=10)
    print(
        f"inference on an 8 s window, {runs} runs: median {statistics.median(total_ms):.1f} ms, "
        f"p90 {q[-1]:.1f} ms, min {min(total_ms):.1f} ms "
        f"(of which log-mel median {statistics.median(feats_ms):.1f} ms)"
    )


def parity(model: turn.SmartTurn, paths: list[str]) -> None:
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(chunk_length=8)
    worst_feat, worst_p = 0.0, 0.0
    for path in paths:
        audio = turn.fit_window(_load(path))
        ref = fe(
            audio, sampling_rate=SR, return_tensors="np", padding="max_length",
            max_length=8 * SR, truncation=True, do_normalize=True,
        ).input_features.astype(np.float32)
        ours = turn.whisper_features(audio)
        d_feat = float(np.abs(ref - ours).max())
        p_ref = float(model._session.run(None, {model._input_name: ref})[0].reshape(-1)[0])
        p_ours = model.predict(audio)
        worst_feat = max(worst_feat, d_feat)
        worst_p = max(worst_p, abs(p_ref - p_ours))
        print(f"{os.path.basename(path):24s} max|Δfeat|={d_feat:.2e}  P ref={p_ref:.4f} ours={p_ours:.4f}")
    print(f"parity over {len(paths)} files: worst |Δfeature| {worst_feat:.2e}, worst |ΔP| {worst_p:.2e}")


def score(model: turn.SmartTurn, paths: list[str], pause_ms: int, threshold: float) -> None:
    pad = np.zeros(int(SR * pause_ms / 1000), dtype=np.float32)
    lead = np.zeros(SR // 2, dtype=np.float32)  # stands in for the pre-roll
    right = total = 0
    for path in paths:
        audio = np.concatenate([lead, _trim(_load(path)), pad])
        p, ms = model.timed_predict(audio)
        name = os.path.basename(path)
        verdict = "complete" if p >= threshold else "incomplete"
        mark = ""
        for label in ("incomplete", "complete"):
            if name.startswith(label):
                total += 1
                ok = verdict == label
                right += ok
                mark = "ok" if ok else "WRONG"
                break
        print(f"{name:24s} P={p:.3f}  {verdict:10s} {mark:5s} ({ms:.0f} ms)")
    if total:
        print(f"accuracy at pause {pause_ms} ms, threshold {threshold}: {right}/{total}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--pause-ms", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    model = turn.SmartTurn(threads=args.threads)
    model.load()
    print(f"model: {model.path}")
    if args.bench:
        bench(model, args.runs)
    if args.parity and args.wavs:
        parity(model, args.wavs)
    elif args.wavs:
        score(model, args.wavs, args.pause_ms, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
