#!/usr/bin/env python3
"""Turn audio into openWakeWord feature windows: (N, 16, 96) float16.

The model trained here is a small head on openWakeWord's FROZEN front-end
(melspectrogram.onnx -> embedding_model.onnx, Google's speech embedding).
openwakeword.Model feeds a custom model the last 16 embedding frames (each
80 ms, so 1.28 s of context over a ~2 s receptive field), which is exactly
one 2-second clip through AudioFeatures.embed_clips. So:

  --mode aligned   short clips (the wake word, a confusable word). The END of
                   the speech is placed near the end of a 2 s window, with
                   --jitters random offsets each: that is the moment a
                   streaming detector must fire (positives) or must not
                   (hard negatives).
  --mode stream    long or many clips (talk, noise, background corpora),
                   concatenated and cut into 10 s chunks; every window with
                   a --stride (in 80 ms frames) is kept.

    python scripts/wakeword/features.py --mode aligned --in data/pos_train --out feats/pos_train.npy --jitters 3
    python scripts/wakeword/features.py --mode stream --in data/bg/librispeech-dev-clean --out feats/bg_libri.npy --stride 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

WINDOW_FRAMES = 16           # what openwakeword.Model feeds a custom model
CLIP_SAMPLES = 32000         # 2 s -> exactly 16 embedding frames
CHUNK_SAMPLES = 160000       # 10 s chunks for stream mode
_FRAME = 160                 # 10 ms


def feature_models() -> tuple[str, str]:
    """Paths of openWakeWord's front-end ONNX files (env OWW_MODELS overrides)."""
    d = os.environ.get("OWW_MODELS")
    if not d:
        import openwakeword

        d = str(Path(openwakeword.__file__).parent / "resources" / "models")
    mel, emb = Path(d) / "melspectrogram.onnx", Path(d) / "embedding_model.onnx"
    if not mel.is_file() or not emb.is_file():
        raise SystemExit(f"openWakeWord feature models not found in {d}. Run once: "
                         "python -c 'import openwakeword.utils as u; u.download_models([])' "
                         "or set OWW_MODELS to a folder with melspectrogram.onnx and embedding_model.onnx")
    return str(mel), str(emb)


def extractor(ncpu: int):
    from openwakeword.utils import AudioFeatures

    mel, emb = feature_models()
    return AudioFeatures(melspec_model_path=mel, embedding_model_path=emb,
                         ncpu=ncpu, inference_framework="onnx")


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def speech_end(x: np.ndarray) -> int:
    """Sample index where the clip's speech ends (energy over 10 ms frames)."""
    n = x.size // _FRAME
    if n == 0:
        return x.size
    rms = np.sqrt(np.mean(x[: n * _FRAME].reshape(n, _FRAME) ** 2, axis=1))
    floor = float(np.percentile(rms, 10))
    loud = np.nonzero(rms > max(floor * 4, rms.max() * 0.08))[0]
    return int((loud[-1] + 1) * _FRAME) if loud.size else x.size


def align(x: np.ndarray, rng: np.random.Generator, jitter_s: tuple[float, float]) -> np.ndarray:
    """A 2 s clip whose speech ends jitter seconds before the end."""
    end = speech_end(x)
    tail = int(rng.uniform(*jitter_s) * C.SR)
    start = end + tail - CLIP_SAMPLES
    floor = rng.normal(0, 0.0008, CLIP_SAMPLES).astype(np.float32)
    out = floor.copy()
    src_from = max(0, start)
    dst_from = src_from - start
    piece = x[src_from:end + tail][: CLIP_SAMPLES - dst_from]
    # Whatever the clip does not cover (before its start, past its end)
    # stays room tone.
    out[dst_from:dst_from + piece.size] = piece
    return out


def aligned_windows(files: list[Path], jitters: int, seed: int, ncpu: int,
                    jitter_s: tuple[float, float]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clips = []
    for f in files:
        x = C.read_wav(f)
        for _ in range(jitters):
            clips.append(to_int16(align(x, rng, jitter_s)))
    if not clips:
        return np.zeros((0, WINDOW_FRAMES, 96), np.float16)
    feats = extractor(ncpu).embed_clips(np.stack(clips), batch_size=256, ncpu=ncpu)
    return feats[:, -WINDOW_FRAMES:, :].astype(np.float16)


def stream_windows(files: list[Path], stride: int, ncpu: int, max_hours: float | None,
                   seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    parts, total = [], 0
    limit = int(max_hours * 3600 * C.SR) if max_hours else None
    for f in files:
        try:
            x = C.read_wav(f)
        except Exception:
            continue
        parts.append(x)
        parts.append(rng.normal(0, 0.0008, int(0.3 * C.SR)).astype(np.float32))
        total += x.size
        if limit and total >= limit:
            break
    audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    hop = CHUNK_SAMPLES - CLIP_SAMPLES  # 2 s overlap: no window lost at a seam
    chunks = [audio[i:i + CHUNK_SAMPLES] for i in range(0, max(1, audio.size - CLIP_SAMPLES), hop)]
    chunks = [np.pad(c, (0, CHUNK_SAMPLES - c.size)) for c in chunks if c.size]
    if not chunks:
        return np.zeros((0, WINDOW_FRAMES, 96), np.float16)
    fx = extractor(ncpu)
    out = []
    for b in range(0, len(chunks), 64):
        emb = fx.embed_clips(to_int16(np.stack(chunks[b:b + 64])), batch_size=256, ncpu=ncpu)
        n = emb.shape[1]
        idx = np.arange(0, n - WINDOW_FRAMES + 1, stride)
        win = np.stack([emb[:, i:i + WINDOW_FRAMES, :] for i in idx], axis=1)
        out.append(win.reshape(-1, WINDOW_FRAMES, 96).astype(np.float16))
    print(f"  {audio.size / C.SR / 3600:.2f} h of audio", flush=True)
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("aligned", "stream"), required=True)
    ap.add_argument("--in", dest="inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jitters", type=int, default=2)
    ap.add_argument("--jitter-s", type=float, nargs=2, default=(0.0, 0.3))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ncpu", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    files = [f for d in args.inputs for f in (C.wavs(d) if d.is_dir() else [d])]
    print(f"{args.mode}: {len(files)} files from {[str(d) for d in args.inputs]}", flush=True)
    if args.mode == "aligned":
        feats = aligned_windows(files, args.jitters, args.seed, args.ncpu, tuple(args.jitter_s))
    else:
        feats = stream_windows(files, args.stride, args.ncpu, args.max_hours, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, feats)
    print(f"{feats.shape} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
