"""Parallel Kokoro rendering for the generators (positives and negatives).

Kokoro's ONNX session grabs every core by default, so N worker processes
would fight over N x cores threads. Each worker here gets its own session
pinned to ``threads`` intra-op threads instead, which is what makes a
16-core CPU box (or a 12-core Mac) actually render N clips at once.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common as C

_noise: np.ndarray | None = None


def _init(threads: int, noise_path: str | None) -> None:
    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, threads)
    opts.inter_op_num_threads = 1
    d = C.kokoro_dir()
    session = ort.InferenceSession(str(d / "kokoro-v1.0.onnx"), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    C._kokoro = Kokoro.from_session(session, str(d / "voices-v1.0.bin"))
    global _noise
    if noise_path:
        _noise = np.load(noise_path, mmap_mode="r")


def _job(args: tuple[int, C.Variant, str, int]) -> dict | None:
    i, v, out_dir, seed = args
    rng = np.random.default_rng(seed + i)
    noise = None
    if _noise is not None and v.snr_db is not None and rng.random() < 0.6:
        # Real background (speech babble, room) for most noisy clips; the
        # synthetic pink noise in common.add_noise for the rest.
        n = int(rng.integers(2, 6) * C.SR)
        start = int(rng.integers(0, max(1, _noise.shape[0] - n)))
        noise = np.asarray(_noise[start:start + n], dtype=np.float32)
    try:
        audio = C.render(v, rng, noise)
    except Exception as exc:  # a voice/text Kokoro cannot read: skip it
        return {"i": i, "error": str(exc)}
    name = f"{i:06d}_{v.tag}.wav"
    C.write_wav(Path(out_dir) / name, audio)
    return {"i": i, "file": name, "text": v.text, "voice": v.voice, "speed": v.speed,
            "semitones": v.semitones, "snr_db": v.snr_db, "reverb": v.reverb}


def render_all(items: Sequence[C.Variant], out_dir: Path, seed: int, workers: int,
               threads: int = 2, noise_path: str | None = None) -> list[dict]:
    """Render *items* into *out_dir* (16 kHz WAVs) and write manifest.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, v, str(out_dir), seed) for i, v in enumerate(items)]
    rows: list[dict] = []
    if workers <= 1:
        _init(max(threads, os.cpu_count() or 1), noise_path)
        rows = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(threads, noise_path)) as pool:
            for k, row in enumerate(pool.map(_job, jobs, chunksize=8), 1):
                rows.append(row)
                if k % 250 == 0:
                    print(f"  {k}/{len(jobs)}", flush=True)
    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bad = sum(1 for r in rows if r and "error" in r)
    print(f"{len(rows) - bad} clips in {out_dir} ({bad} skipped)")
    return rows
