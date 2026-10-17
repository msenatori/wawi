#!/usr/bin/env python3
"""Train the "wawi" wake-word head and export it as an openWakeWord ONNX model.

Input: feature windows from features.py, (N, 16, 96). Output: an ONNX model
with input (1, 16, 96) float32 and output (1, 1) sigmoid score -- the shape
openwakeword.Model expects of a custom model; its prediction key is the file
stem, so save it as ``wawi.onnx``.

The head is small on purpose (a ~200k-parameter MLP, like openWakeWord's own
custom models): the frozen embedding does the hearing, the head only decides.
It trains in minutes on a CPU; --device picks cuda/mps when available.

Hard negatives (the confusable words) get --hard-weight in the loss, and
negatives in general get a growing weight late in training: a missed
"Wawi" costs you a repeat, a false wake costs a paid turn and an interruption.

    python scripts/wakeword/train.py \\
        --pos feats/pos_train.npy [feats/pos_owner.npy ...] \\
        --hard feats/neg_hard.npy \\
        --neg feats/neg_speech.npy feats/neg_noise.npy feats/bg_libri.npy feats/bg_fleurs.npy \\
        --out wawi/assets/wakeword/wawi.onnx --steps 30000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Head(nn.Module):
    """openWakeWord's custom-model shape: flatten -> MLP -> sigmoid."""

    def __init__(self, frames: int = 16, dim: int = 96, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(frames * dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load(paths: list[Path]) -> np.ndarray:
    arrs = [np.load(p, mmap_mode="r") for p in paths]
    arrs = [a for a in arrs if a.shape[0]]
    if not arrs:
        return np.zeros((0, 16, 96), np.float16)
    return np.concatenate([np.asarray(a) for a in arrs])


def split(x: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(x.shape[0])
    k = int(x.shape[0] * frac)
    return x[idx[k:]], x[idx[:k]]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def scores(model: Head, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, x.shape[0], 8192):
        out.append(model(torch.from_numpy(np.asarray(x[i:i + 8192], np.float32)).to(device)).cpu())
    model.train()
    return torch.cat(out).numpy().reshape(-1) if out else np.zeros(0)


def export(model: Head, out: Path) -> None:
    model = model.cpu().eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 16, 96)
    kwargs = dict(input_names=["input"], output_names=["score"], opset_version=13)
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:  # torch < 2.5 has no dynamo argument
        torch.onnx.export(model, dummy, str(out), **kwargs)
    print(f"exported {out} ({out.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", type=Path, nargs="+", required=True)
    ap.add_argument("--hard", type=Path, nargs="*", default=[])
    ap.add_argument("--neg", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--hard-weight", type=float, default=3.0)
    ap.add_argument("--neg-weight-max", type=float, default=10.0,
                    help="negative loss weight ramps from 1 to this over training")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads for torch: the model is tiny, more threads only contend")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    pos_tr, pos_va = split(load(args.pos), args.val_frac, rng)
    hard_tr, hard_va = split(load(args.hard), args.val_frac, rng)
    neg_tr, neg_va = split(load(args.neg), args.val_frac, rng)
    print(f"device={device} pos={len(pos_tr)}/{len(pos_va)} hard={len(hard_tr)}/{len(hard_va)} "
          f"neg={len(neg_tr)}/{len(neg_va)}", flush=True)

    model = Head(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    bce = nn.BCELoss(reduction="none")
    # A batch: 1/4 positives, 1/4 hard negatives (if any), the rest negatives.
    n_pos = args.batch // 4
    n_hard = args.batch // 4 if len(hard_tr) else 0
    n_neg = args.batch - n_pos - n_hard
    t0 = time.time()
    for step in range(1, args.steps + 1):
        parts = [pos_tr[rng.integers(0, len(pos_tr), n_pos)]]
        if n_hard:
            parts.append(hard_tr[rng.integers(0, len(hard_tr), n_hard)])
        parts.append(neg_tr[rng.integers(0, len(neg_tr), n_neg)])
        x = torch.from_numpy(np.concatenate(parts).astype(np.float32)).to(device)
        # Feature-space augmentation: a little noise and gain, so the head
        # does not memorise Kokoro's exact embeddings.
        x = x * (1 + 0.05 * torch.randn(x.shape[0], 1, 1, device=device))
        x = x + 0.02 * torch.randn_like(x)
        y = torch.cat([torch.ones(n_pos), torch.zeros(n_hard + n_neg)]).to(device)
        ramp = 1 + (args.neg_weight_max - 1) * min(1.0, step / (0.7 * args.steps))
        w = torch.cat([torch.ones(n_pos), torch.full((n_hard,), args.hard_weight * ramp),
                       torch.full((n_neg,), ramp)]).to(device)
        loss = (bce(model(x).reshape(-1), y) * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == args.steps:
            sp, sh, sn = (scores(model, a, device) for a in (pos_va, hard_va, neg_va))
            line = [f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s"]
            for thr in (0.5, 0.7, 0.9):
                line.append(f"t{thr}: rec {np.mean(sp >= thr):.3f} hardFP {np.mean(sh >= thr) if sh.size else 0:.3f} "
                            f"negFP {np.mean(sn >= thr):.5f}")
            print(" | ".join(line), flush=True)

    export(model, args.out)
    sp, sh, sn = (scores(model, a, device) for a in (pos_va, hard_va, neg_va))
    summary = {"val_recall": {t: float(np.mean(sp >= t)) for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)},
               "val_hard_fp": {t: float(np.mean(sh >= t)) if sh.size else 0 for t in (0.5, 0.7, 0.9)},
               "val_neg_fp_window": {t: float(np.mean(sn >= t)) for t in (0.5, 0.7, 0.9)},
               "steps": args.steps, "train_seconds": round(time.time() - t0, 1),
               "counts": {"pos": len(pos_tr), "hard": len(hard_tr), "neg": len(neg_tr)}}
    args.out.with_suffix(".train.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
