#!/usr/bin/env python3
"""Live diagnostic for the wake cascade. Answers ONE question:

    when you say the wake phrase, what does each stage actually see?

Runs the real Listener against the real microphone, using the SAME engine
config.yaml selects, and prints one line every half second:

    mic=0.081  vad=0.94 SPEECH   ...

    mic  — peak amplitude of the audio arriving. Flat 0.000 means the mic is
           not delivering anything and nothing downstream can possibly work.
    vad  — Silero speech probability. Should jump near 1.0 while you talk.

The last column depends on the engine:

    openwakeword — `wake=` is the model score for its pretrained phrase.
                   Every detection is reported, including the ones the speech
                   gate discards; that path is invisible at INFO level in the
                   app and is a common reason a wake word "does nothing".

    whisper      — each speech segment is transcribed and shown with what it
                   matched, so you can see exactly how your trigger word came
                   out. This is the tool for judging a candidate word: some
                   transcribe verbatim, others collapse into a token loop.

Usage:
    .venv/bin/python scripts/diag_wake.py                  # uses config.yaml
    .venv/bin/python scripts/diag_wake.py --seconds 60
    .venv/bin/python scripts/diag_wake.py --engine whisper --phrases viernes,bruno
    .venv/bin/python scripts/diag_wake.py --engine openwakeword --wake-threshold 0.3

Say your trigger phrase a few times, in your normal voice, at your normal
distance from the machine. Ctrl-C to stop early.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wawi.listener import Listener, ListenerConfig, ListenerError  # noqa: E402


def _load_defaults():
    """Read wake settings from config.yaml; fall back to library defaults."""
    try:
        from wawi import config

        cfg = config.load()
        return {
            "engine": cfg.wake_engine,
            "phrases": list(cfg.wake_phrases),
            "whisper_model": cfg.wake_whisper_model,
            "language": cfg.language,
            "vad_threshold": cfg.wake_vad_threshold,
            "min_speech_level": cfg.wake_min_speech_level,
            "wake_threshold": cfg.wake_threshold,
        }
    except Exception as exc:  # config is optional for a diagnostic
        print(f"(config.yaml unavailable: {exc}; using defaults)", file=sys.stderr)
        return {
            "engine": "openwakeword",
            "phrases": ["viernes"],
            "whisper_model": "mlx-community/whisper-tiny-mlx",
            "language": "es",
            "vad_threshold": 0.5,
            "min_speech_level": 0.0,
            "wake_threshold": 0.5,
        }


def measure_noise(seconds: float) -> int:
    """Measure the room's noise floor and recommend a min_speech_level.

    The amplitude gate only works if the room is genuinely quiet while this
    runs -- it is measuring what Wawi should learn to ignore.
    """
    import numpy as np
    import sounddevice as sd

    peaks: list[float] = []

    def cb(indata, frames, t, status):  # noqa: ARG001
        peaks.append(float(np.abs(indata[:, 0]).max()))

    print(f"Measuring {seconds:.0f}s of room noise — STAY QUIET, do not speak…")
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
                        blocksize=1024, callback=cb):
        time.sleep(seconds)

    p = np.array(peaks)
    if p.size == 0:
        print("No audio arrived at all — check the microphone permission.")
        return 1

    floor = float(np.percentile(p, 95))
    # Sit above the noise, well under speech (which peaks around 0.1-0.3).
    recommended = round(min(max(floor * 1.5, 0.01), 0.10), 3)
    print(f"\n  noise peak: mean={p.mean():.4f}  p95={floor:.4f}  max={p.max():.4f}")
    print(f"\n  Recommended in config.yaml:\n\n    wake:\n      min_speech_level: {recommended}\n")
    if floor > 0.08:
        print("  WARNING: this room is loud. Anything above ~0.08 starts to overlap")
        print("  with normal speech, so the gate may also swallow quiet words.")
        print("  A closer or directional microphone will help more than any setting.")
    print("  Re-run without --measure-noise to confirm the trigger now works.")
    return 0


def main() -> int:
    d = _load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--engine", choices=["openwakeword", "whisper"], default=d["engine"])
    ap.add_argument("--phrases", default=",".join(d["phrases"]),
                    help="comma-separated trigger phrases (whisper engine)")
    ap.add_argument("--whisper-model", default=d["whisper_model"])
    ap.add_argument("--wake-threshold", type=float, default=d["wake_threshold"])
    ap.add_argument("--vad-threshold", type=float, default=d["vad_threshold"])
    ap.add_argument("--min-speech-level", type=float, default=d["min_speech_level"])
    ap.add_argument("--measure-noise", action="store_true",
                    help="measure the room's noise floor and recommend min_speech_level")
    args = ap.parse_args()

    if args.measure_noise:
        return measure_noise(args.seconds if args.seconds < 20 else 8.0)

    phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]
    cfg = ListenerConfig(
        engine=args.engine,
        wake_threshold=args.wake_threshold,
        vad_threshold=args.vad_threshold,
        min_speech_level=args.min_speech_level,
        silence_ms=800,
    )

    stats = {"vad": 0.0, "wake": 0.0, "detections": 0, "gated": 0, "checks": 0,
             "saw_speech": False, "gated_by_level": 0}
    detector = None

    if args.engine == "whisper":
        from wawi.wake import WakeError, WhisperWakeDetector

        try:
            detector = WhisperWakeDetector(
                phrases, model=args.whisper_model, language=d["language"]
            )
        except WakeError as exc:
            print(f"Bad configuration: {exc}", file=sys.stderr)
            return 1

    utterances: list[float] = []
    listener = Listener(
        on_utterance=lambda buf: utterances.append(buf.size / cfg.sample_rate),
        cfg=cfg,
        wake_detector=detector,
    )

    # Instrument the stages in place, so we measure the real code path rather
    # than a reimplementation of it.
    def is_speech(frame):
        import numpy as np

        score = float(listener._vad.predict(frame, frame_size=listener._vad_frame_len))
        stats["vad"] = max(stats["vad"], score)
        # Mirror the real _is_speech, amplitude gate included, or the
        # diagnostic would report behaviour the app does not have.
        loud_enough = (
            listener._min_speech_amplitude == 0
            or int(np.abs(frame).max()) >= listener._min_speech_amplitude
        )
        speech = loud_enough and score >= cfg.vad_threshold
        if score >= cfg.vad_threshold and not loud_enough:
            stats["gated_by_level"] += 1
        stats["saw_speech"] = stats["saw_speech"] or speech
        return score if loud_enough else -1.0

    # The listener reads scores, not booleans, since the onset gate needs the
    # raw number (vad_onset_threshold); patch the method it actually calls.
    listener._speech_score = is_speech  # type: ignore[method-assign]

    if args.engine == "openwakeword":
        def detect_wake(frame):
            scores = listener._wake_model.predict(frame)
            score = float(scores.get("hey_jarvis", 0.0))
            stats["wake"] = max(stats["wake"], score)
            if score >= cfg.wake_threshold:
                stats["detections"] += 1
                gated = not any(listener._recent_speech)
                stats["gated"] += int(gated)
                print(
                    f"\n  >>> WAKE FIRED  score={score:.3f}"
                    + ("   BUT THE SPEECH GATE DISCARDED IT (VAD saw no speech)"
                       if gated else "   ACCEPTED")
                )
                return True
            return False

        listener._detect_wake = detect_wake  # type: ignore[method-assign]
    else:
        real_check = detector.check

        def check(segment):
            result = real_check(segment)
            stats["checks"] += 1
            if result.ok:
                stats["detections"] += 1
            mark = f"MATCHED {result.matched!r}" if result.ok else f"NO — {result.reason}"
            print(
                f"\n  >>> heard {result.text!r}  ({result.elapsed_s*1000:.0f}ms)  {mark}"
            )
            return result

        detector.check = check  # type: ignore[method-assign]

    try:
        listener.start()
    except ListenerError as exc:
        print(f"Could not start the listener: {exc}", file=sys.stderr)
        return 1

    if args.engine == "whisper":
        print(f'Listening for {args.seconds:.0f}s — say one of: {phrases}')
        print(f"engine=whisper  model={args.whisper_model}  vad>={args.vad_threshold}\n")
    else:
        print(f'Listening for {args.seconds:.0f}s — say "hey jarvis, what time is it?" (openWakeWord's phrase)')
        print(f"engine=openwakeword  wake>={args.wake_threshold}  vad>={args.vad_threshold}\n")

    deadline = time.monotonic() + args.seconds
    peak_mic = 0.0
    try:
        while time.monotonic() < deadline:
            time.sleep(0.5)
            mic = listener.current_level()
            peak_mic = max(peak_mic, mic)
            speech = "SPEECH" if stats["vad"] >= cfg.vad_threshold else "      "
            tail = "" if args.engine == "whisper" else f"  wake={stats['wake']:.3f}"
            print(f"  mic={mic:.3f}  vad={stats['vad']:.2f} {speech}{tail}")
            stats["vad"] = 0.0
            stats["wake"] = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

    print("\n--- summary ---")
    print(f"  microphone peak over the run : {peak_mic:.4f}")
    if args.min_speech_level > 0:
        print(f"  frames cut by the level gate : {stats['gated_by_level']}"
              f"  (min_speech_level={args.min_speech_level})")
    if args.engine == "whisper":
        print(f"  speech segments transcribed  : {stats['checks']}")
    print(f"  wake detections              : {stats['detections']}")
    if args.engine == "openwakeword":
        print(f"  discarded by the speech gate : {stats['gated']}")
    print(f"  utterances dispatched        : {len(utterances)}")

    listener_stats = listener.stats()
    if listener_stats:
        print("\n  listener counters:")
        for key, value in sorted(listener_stats.items()):
            print(f"    {key:32s} {value}")

    if peak_mic == 0.0:
        print("\n  VERDICT: the microphone delivered pure silence. Nothing else matters")
        print("  until that is fixed — check System Settings > Privacy > Microphone.")
    elif args.engine == "whisper" and stats["checks"] == 0:
        if stats["saw_speech"]:
            # The VAD DID see speech — the problem is that it never stops
            # seeing it, so the utterance never endpoints and nothing is
            # ever transcribed.
            print("\n  VERDICT: the VAD heard speech but never enough SILENCE to close")
            print("  the utterance, so nothing was ever transcribed. The room's noise")
            print("  floor is scoring as speech. Measure it and set an amplitude gate:")
            print("      python scripts/diag_wake.py --measure-noise")
        else:
            print("\n  VERDICT: the VAD never registered speech at all. Retry with")
            print("  --vad-threshold 0.2, and check you are close enough to the mic.")
    elif stats["detections"] == 0 and args.engine == "whisper":
        print("\n  VERDICT: speech was transcribed but never matched a phrase. Read the")
        print("  'heard' lines above: if your word comes out mangled every time, pick")
        print("  another one — some words transcribe reliably and some do not.")
    elif stats["detections"] == 0:
        print("\n  VERDICT: the mic works but the wake model never cleared the bar.")
        print("  Retry with --wake-threshold 0.3 (LOWER is more eager).")
    elif args.engine == "openwakeword" and stats["gated"] == stats["detections"]:
        print("\n  VERDICT: the wake word fires but the VAD speech gate kills every")
        print("  detection. Retry with --vad-threshold 0.2.")
    else:
        print("\n  VERDICT: the cascade works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
