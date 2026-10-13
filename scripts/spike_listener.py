#!/usr/bin/env python3
"""Standalone demo of the always-on listener (openWakeWord engine).

Runs wawi.listener.Listener against your real microphone and prints one line
per captured utterance: duration, peak level, and the pre-roll that was
prepended. Nothing is transcribed and nothing leaves the machine.

Usage:
    python scripts/spike_listener.py

    # first run only: fetch the ~8 MB of model weights
    python scripts/spike_listener.py --download-models

    # tune the two knobs that matter
    python scripts/spike_listener.py --pre-roll 2.0 --silence-ms 600

Requires:
    pip install openwakeword sounddevice

No account and no access key: openWakeWord's "hey_jarvis" model is pretrained
(its weights are CC BY-NC-SA 4.0, non-commercial). The model weights are NOT bundled in the wheel, so the
first run needs --download-models (or `wawi --check`, which reports the same
thing).

Say "hey jarvis, <something>" (openWakeWord's phrase) in one breath and watch the line appear when you
stop talking. Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Allow running the script directly from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wawi.listener import (  # noqa: E402
    Listener,
    ListenerConfig,
    ListenerError,
    download_models,
    missing_model_files,
    models_ready,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Listen continuously for openWakeWord's "hey jarvis" and report captures.'
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Fetch the openWakeWord model weights (~8 MB) and exit. "
        "Needed once; the wheel ships code only.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=1.5,
        help="Seconds of audio kept from before the wake word fired (default: 1.5).",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=800,
        help="Silence that ends an utterance. Lower feels snappier but truncates "
        "mid-sentence pauses; higher feels sluggish (default: 800).",
    )
    parser.add_argument("--max-utterance", type=float, default=30.0)
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero speech probability 0..1 above which a frame counts as speech. "
        "Higher is stricter.",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=0.5,
        help="openWakeWord score 0..1 required to trigger. Higher means fewer "
        "false wakes and more missed ones.",
    )
    parser.add_argument(
        "--framework",
        default="onnx",
        choices=("onnx", "tflite"),
        help="Inference framework. onnx is the only option on macOS (tflite-runtime "
        "publishes Linux wheels only).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.download_models:
        try:
            download_models()
        except ListenerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print("Models are ready.")
        return 0

    if not models_ready(args.framework):
        try:
            missing = [Path(p).name for p in missing_model_files(args.framework)]
            detail = ", ".join(missing)
        except ListenerError as exc:
            # openwakeword itself is not installed.
            print(f"error: {exc}", file=sys.stderr)
            print("  pip install openwakeword sounddevice", file=sys.stderr)
            return 2
        print(
            f"error: openWakeWord model weights are missing ({detail}).\n"
            "  They are not bundled with the package. Fetch them once with:\n"
            "    python scripts/spike_listener.py --download-models",
            file=sys.stderr,
        )
        return 2

    cfg = ListenerConfig(
        sample_rate=args.sample_rate,
        pre_roll_s=args.pre_roll,
        silence_ms=args.silence_ms,
        max_utterance_s=args.max_utterance,
        vad_threshold=args.vad_threshold,
        wake_threshold=args.wake_threshold,
        inference_framework=args.framework,
    )

    count = 0
    started = time.monotonic()

    def on_utterance(buf: np.ndarray) -> None:
        # Runs on the listener's worker thread — keep it short.
        nonlocal count
        count += 1
        duration = buf.size / cfg.sample_rate
        peak = float(np.abs(buf).max()) if buf.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(buf)))) if buf.size else 0.0
        print(
            f"[{time.monotonic() - started:7.1f}s] "
            f"utterance #{count}: {duration:5.2f}s  "
            f"peak={peak:.3f}  rms={rms:.4f}  "
            f"samples={buf.size}  (pre-roll {cfg.pre_roll_s:.1f}s included)",
            flush=True,
        )

    try:
        listener = Listener(on_utterance=on_utterance, cfg=cfg)
        listener.start()
    except ListenerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Listening at {cfg.sample_rate} Hz — pre-roll {cfg.pre_roll_s:.1f}s, "
        f"endpoint after {cfg.silence_ms} ms of silence, cap {cfg.max_utterance_s:.0f}s.\n"
        'Say "hey jarvis, <something>" in one breath. Ctrl-C to quit.',
        flush=True,
    )

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping…", flush=True)
    finally:
        listener.stop()

    print(f"Captured {count} utterance(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
