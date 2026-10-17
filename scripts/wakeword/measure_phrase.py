#!/usr/bin/env python3
"""Measure the whisper phrase mode ("Wawi" by transcription) on synthetic clips.

Generates positives ("Wawi", "che Wawi", "Wawi, ¿qué hora es?"…) and negatives
(confusable words: "guau", "wifi", "bueno", "agua", "Huawei", "y vos"…, plus
ordinary sentences) with Kokoro across voices, speeds, pitches, noise and
reverb, then runs them through:

    old        mlx_whisper.transcribe(whisper-tiny), no prompt, whole clip:
               what phrase mode did before this change
    detector   wawi.wake.WhisperWakeDetector as shipped: private model,
               greedy decode, decoder prompt "Wawi, Wawi.", whole clip
    window     the same, quick decode of only the first pre-roll + window_s
    verified   window + the trained "wawi" model as a second opinion
               (wake.verify_model; needs wawi/assets/wakeword/wawi.onnx)
    listener   end to end: the real Listener (real Silero VAD, onset/voiced
               gates, the early window, endpointing) with require_phrase on;
               counts what would have become a turn
    turbo      whisper-large-v3-turbo + prompt via transcribe (reference)

Prints recall, false positives split into confusable words and ordinary
sentences, and latency; writes every transcript to --out (JSON) so new
aliases can be read off the misses. Apple Silicon (mlx-whisper); no network
beyond the first model download.

    .venv/bin/python scripts/wakeword/measure_phrase.py --pos 120 --neg 160
    .venv/bin/python scripts/wakeword/measure_phrase.py --configs detector,listener
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import common as C  # noqa: E402
from wawi.wake import (  # noqa: E402
    AcousticVerifier,
    WhisperWakeDetector,
    explain_trigger,
    wake_prompt,
)

TINY = "mlx-community/whisper-tiny-mlx"
TURBO = "mlx-community/whisper-large-v3-turbo"
PHRASES = ["wawi"]


def clip_set(n_pos: int, n_neg: int, seed: int) -> list[tuple[str, bool, str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    voices = C.SPANISH_VOICES * 4 + C.ACCENT_VOICES  # Spanish weighted up
    pos = C.variants(C.POSITIVE_TEXTS, voices, n_pos, seed)
    neg = C.variants(C.CONFUSABLE_TEXTS * 2 + C.RANDOM_TEXTS, voices, n_neg, seed + 1)
    out = []
    for label, items in ((True, pos), (False, neg)):
        for v in items:
            out.append((v.text, label, v.tag, C.render(v, rng)))
    return out


def run_transcribe(clips, model: str, prompt: bool) -> list[dict]:
    import mlx_whisper

    opts = {"initial_prompt": wake_prompt(PHRASES)} if prompt else {}
    mlx_whisper.transcribe(np.zeros(8000, np.float32), path_or_hf_repo=model, language="es")
    rows = []
    for text, label, tag, audio in clips:
        t0 = time.perf_counter()
        heard = str(mlx_whisper.transcribe(audio, path_or_hf_repo=model, language="es",
                                           **opts).get("text", "")).strip()
        rows.append(_row(text, label, tag, heard, explain_trigger(heard, PHRASES)[0],
                         time.perf_counter() - t0))
    return rows


def run_detector(clips, detector, window_s: float | None, pre_roll: float,
                 verifier=None) -> list[dict]:
    rows = []
    for text, label, tag, audio in clips:
        if window_s is not None:
            audio = audio[: int((pre_roll + window_s) * C.SR)]
        t0 = time.perf_counter()
        r = detector.check(audio, quick=window_s is not None)
        matched = r.matched
        if matched is not None and verifier is not None and not verifier.accepts(audio)[0]:
            matched = None
        rows.append(_row(text, label, tag, r.text, matched, time.perf_counter() - t0))
    return rows


def run_listener(clips, detector, window_s: float, verifier=None) -> list[dict]:
    """Each clip through a fresh production-tuned Listener with real VAD."""
    import openwakeword

    from wawi.config import _DEFAULTS
    from wawi.listener import Listener, ListenerConfig

    w = _DEFAULTS["wake"]
    vad = openwakeword.VAD()
    rows = []
    for text, label, tag, audio in clips:
        got: list[np.ndarray] = []
        listener = Listener(
            on_utterance=got.append, wake_detector=detector, verifier=verifier,
            cfg=ListenerConfig(
                engine="whisper", require_phrase=True, sample_rate=C.SR,
                pre_roll_s=w["pre_roll_s"], silence_ms=w["silence_ms"],
                vad_threshold=w["vad_threshold"], vad_onset_threshold=w["vad_onset_threshold"],
                min_onset_ms=w["min_onset_ms"], onset_window_ms=w["onset_window_ms"],
                min_voiced_ms=w["min_voiced_ms"], min_voiced_ratio=w["min_voiced_ratio"],
                wake_window_s=window_s,
            ),
        )
        listener._vad = vad
        vad.reset_states() if hasattr(vad, "reset_states") else None
        listener._reset_capture_state()
        tail = np.zeros(int(1.5 * C.SR), np.float32)
        stream = np.concatenate([np.zeros(int(1.0 * C.SR), np.float32), audio, tail])
        t0 = time.perf_counter()
        for i in range(0, stream.size, 1024):
            listener._pump(stream[i:i + 1024])
        woke = got[0].wake if got else None
        rows.append(_row(text, label, tag, "", woke, time.perf_counter() - t0))
    return rows


def _row(text, label, tag, heard, matched, seconds) -> dict:
    return {"text": text, "positive": label, "tag": tag, "heard": heard,
            "woke": matched is not None, "ms": round(seconds * 1000, 1)}


def report(name: str, rows: list[dict]) -> None:
    pos = [r for r in rows if r["positive"]]
    conf = [r for r in rows if not r["positive"] and r["text"] in C.CONFUSABLE_TEXTS]
    sent = [r for r in rows if not r["positive"] and r["text"] in C.RANDOM_TEXTS]

    def rate(rs):
        n = sum(r["woke"] for r in rs)
        return f"{n / max(1, len(rs)):6.1%} ({n}/{len(rs)})"

    ms = [r["ms"] for r in rows]
    print(f"{name:9} recall {rate(pos)}  FP confusables {rate(conf)}  "
          f"FP sentences {rate(sent)}  {np.mean(ms):6.0f} ms mean, p95 {np.percentile(ms, 95):.0f}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", type=int, default=120)
    ap.add_argument("--neg", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--pre-roll", type=float, default=0.6)
    ap.add_argument("--configs", default="old,detector,window,verified,listener,listener+verify")
    ap.add_argument("--verify-threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("wakeword_measure.json"))
    args = ap.parse_args()

    print(f"Generating {args.pos} positives + {args.neg} negatives with Kokoro…", flush=True)
    clips = clip_set(args.pos, args.neg, args.seed)
    detector = WhisperWakeDetector(PHRASES)

    def verifier():
        from wawi.listener import resolve_wake_model

        return AcousticVerifier(resolve_wake_model("wawi")[1], threshold=args.verify_threshold)
    results: dict[str, list[dict]] = {}
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        if name == "old":
            rows = run_transcribe(clips, TINY, prompt=False)
        elif name == "turbo":
            rows = run_transcribe(clips, TURBO, prompt=True)
        else:
            detector.load()
            if name == "detector":
                rows = run_detector(clips, detector, None, args.pre_roll)
            elif name == "window":
                rows = run_detector(clips, detector, args.window, args.pre_roll)
            elif name == "verified":
                rows = run_detector(clips, detector, args.window, args.pre_roll, verifier())
            elif name == "listener":
                rows = run_listener(clips, detector, args.window)
            elif name == "listener+verify":
                rows = run_listener(clips, detector, args.window, verifier())
            else:
                raise SystemExit(f"unknown config {name!r}")
        results[name] = rows
        report(name, rows)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"Transcripts: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
