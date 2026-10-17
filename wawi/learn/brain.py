"""The tiny brain: a numpy MLP that learns the owner's rhythm from her life log.

The bandit learns *which* behaviours the owner likes; the brain learns
*when*: Monday mornings with the owner typing hard are not Friday evenings.
It is deliberately small and deliberately humble:

* **Input** — 20 context features: time of day and weekday (sin/cos), a
  weekend flag, her mood (one-hot), her five needs, how long since anyone
  touched or talked to her, whether the owner is typing, how many alerts
  she showed lately (:func:`features`).
* **Network** — 20 → 32 (tanh) → one output per behaviour plus one "alert
  now" head. Each output predicts the reward of that action in that context
  (direct-method contextual bandit: the loss only sees the action taken).
* **Data** — every resolved decision lands in a rolling JSONL life log
  (:class:`LifeLog`): a few hundred bytes each, capped by age and size.
* **Training** — mini-batch Adam for a fixed number of steps on a background
  thread with background QoS, only when the machine has been idle for a
  while and at most every few hours (:meth:`Brain.due`). Thousands of rows
  train in well under a second on one core; nothing runs otherwise.
* **Influence** — the rules and the utility weights stay the prior. Below
  ``min_samples`` decisions the brain only whispers (a tenth of its
  ceiling); after that its weight follows its skill on a chronological
  holdout (beat the per-action average or say nothing), and never exceeds
  ``max_weight``. Its multipliers are clamped like the bandit's.

Memory: the weights are ~4 KB; a training run on a full log is a few MB.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

log = logging.getLogger(__name__)

MOODS = ("happy", "content", "bored", "hungry", "sleepy", "lonely")
NEEDS = ("food", "water", "energy", "fun", "affection")
N_FEATURES = 20
HIDDEN = 32
ALERT = "__alert__"
#: Brain multipliers before blending, like the bandit's.
FACTOR_MIN, FACTOR_MAX = 0.5, 2.0
#: Skill (1 - holdout MSE / baseline MSE) at which the weight is maxed out.
FULL_SKILL = 0.3
#: Fraction of the ceiling the brain may use before it has enough data.
WHISPER = 0.1
#: New decisions needed before another training run is worth it.
RETRAIN_NEW = 20
#: Records read for training at most (the newest ones).
MAX_TRAIN = 20_000


def features(ctx: dict[str, Any]) -> np.ndarray:
    """The context as 20 floats, all roughly in [-1, 1]. Missing keys get
    neutral values, so a half-known context is still a valid input."""
    wall = float(ctx.get("wall") or time.time())
    lt = time.localtime(wall)
    hour = lt.tm_hour + lt.tm_min / 60.0
    day = lt.tm_wday + hour / 24.0
    x = [
        math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * day / 7.0), math.cos(2 * math.pi * day / 7.0),
        1.0 if lt.tm_wday >= 5 else 0.0,
    ]
    mood = str(ctx.get("mood") or "content")
    x += [1.0 if mood == m else 0.0 for m in MOODS]
    needs = ctx.get("needs") or {}
    for n in NEEDS:
        try:
            x.append(min(1.0, max(0.0, float(needs.get(n, 60.0)) / 100.0)))
        except (TypeError, ValueError):
            x.append(0.6)
    ignored_min = max(0.0, float(ctx.get("ignored_for") or 0.0)) / 60.0
    x.append(min(1.0, math.log1p(ignored_min) / math.log1p(240.0)))
    key_idle = ctx.get("key_idle")
    key_idle = 1e9 if key_idle is None or not math.isfinite(float(key_idle)) else float(key_idle)
    x.append(1.0 if key_idle < 30.0 else 0.0)
    x.append(min(1.0, math.log1p(key_idle / 60.0) / math.log1p(120.0)))
    x.append(min(1.0, float(ctx.get("recent_alerts") or 0) / 5.0))
    return np.asarray(x, dtype=np.float32)


# --------------------------------------------------------------------------- #
# The life log
# --------------------------------------------------------------------------- #


class LifeLog:
    """Append-only JSONL of resolved decisions, rotated by age and size.

    One line: ``{"t": wall, "h": "b"|"a", "a": action, "r": reward, "x": [...]}``.
    Appends are one short write; rotation rewrites the file only when it
    outgrows ``max_bytes`` (keeping the newest half that is not too old).
    """

    def __init__(self, path: Path | None, *, max_days: float = 30.0,
                 max_bytes: int = 2_000_000) -> None:
        self.path = path
        self.max_age_s = max_days * 86400.0
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def append(self, wall: float, head: str, action: str, reward: float,
               x: Sequence[float]) -> None:
        if self.path is None:
            return
        line = json.dumps({"t": round(wall, 1), "h": head, "a": action,
                           "r": round(float(reward), 3),
                           "x": [round(float(v), 3) for v in x]},
                          ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                if self.path.stat().st_size > self.max_bytes:
                    self._rotate(wall)
            except OSError:
                log.debug("learn: life log append failed", exc_info=True)

    def _rotate(self, wall: float) -> None:
        rows = self._read_lines()
        keep: list[str] = []
        size = 0
        for line in reversed(rows):
            try:
                t = float(json.loads(line).get("t", 0))
            except (ValueError, AttributeError):
                continue
            if wall - t > self.max_age_s:
                break
            size += len(line.encode("utf-8")) + 1
            if size > self.max_bytes // 2:
                break
            keep.append(line)
        keep.reverse()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")  # type: ignore[union-attr]
        tmp.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")
        os.replace(tmp, self.path)  # type: ignore[arg-type]

    def _read_lines(self) -> list[str]:  # rotation only: the file is small then
        try:
            return [ln for ln in self.path.read_text(encoding="utf-8").splitlines()  # type: ignore[union-attr]
                    if ln.strip()]
        except (OSError, AttributeError):
            return []

    def records(self, wall: float | None = None) -> list[dict[str, Any]]:
        """Valid records younger than ``max_days``, oldest first."""
        return list(self._iter(wall))

    def _iter(self, wall: float | None = None) -> Iterator[dict[str, Any]]:
        wall = time.time() if wall is None else wall
        if self.path is None:
            return
        with self._lock:
            try:
                with open(self.path, encoding="utf-8") as fh:
                    # Only the newest MAX_TRAIN lines, without holding the file.
                    lines: deque[str] = deque(fh, maxlen=MAX_TRAIN)
            except OSError:
                return
        for line in lines:
            try:
                rec = json.loads(line)
                if wall - float(rec["t"]) > self.max_age_s:
                    continue
                if len(rec["x"]) != N_FEATURES or rec["h"] not in ("a", "b"):
                    continue
                float(rec["r"])
            except (ValueError, KeyError, TypeError):
                continue
            yield rec

    def arrays(self, wall: float | None, index: dict[str, int], alert: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X, action index, reward) for training, built row by row so a full
        log never exists as Python objects all at once (a few MB, not 80)."""
        X = np.empty((MAX_TRAIN, N_FEATURES), np.float32)
        A = np.empty(MAX_TRAIN, np.int64)
        R = np.empty(MAX_TRAIN, np.float32)
        n = 0
        for rec in self._iter(wall):
            if rec["h"] == "b":
                a = index.get(rec["a"])
                if a is None:
                    continue  # a behaviour that no longer exists
            else:
                a = alert
            X[n] = rec["x"]
            A[n] = a
            R[n] = float(rec["r"])
            n += 1
        return X[:n].copy(), A[:n].copy(), np.clip(R[:n], -1.0, 1.0)


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #


class MLP:
    """x → tanh(W1 x + b1) → W2 h + b2. Trained with a masked MSE."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((n_in, n_hidden)) / math.sqrt(n_in)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, np.float32)
        self.W2 = (rng.standard_normal((n_hidden, n_out)) * 0.1 / math.sqrt(n_hidden)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2

    def fit(self, X: np.ndarray, A: np.ndarray, R: np.ndarray, *, steps: int = 1500,
            batch: int = 256, lr: float = 0.01, l2: float = 1e-4, seed: int = 0) -> float:
        """Mini-batch Adam on (features, action index, reward). Returns the
        MSE of the last batch on the actions taken.

        A fixed number of small steps, whatever the log size: the cost is
        bounded (~1 s on one efficiency core) and so is memory, since no
        step ever holds more than *batch* rows of activations.
        """
        n = len(X)
        rng = np.random.default_rng(seed)
        m = [np.zeros_like(p) for p in self.params()]
        v = [np.zeros_like(p) for p in self.params()]
        b1, b2, eps = 0.9, 0.999, 1e-8
        loss = 0.0
        size = min(batch, n)
        rows = np.arange(size)
        for step in range(1, steps + 1):
            idx = rng.integers(0, n, size) if size < n else np.arange(n)
            Xb, Ab, Rb = X[idx], A[idx], R[idx]
            H = np.tanh(Xb @ self.W1 + self.b1)
            out = H @ self.W2 + self.b2
            err = out[rows, Ab] - Rb
            loss = float(np.mean(err ** 2))
            dout = np.zeros_like(out)
            dout[rows, Ab] = 2.0 * err / size
            gW2 = H.T @ dout + l2 * self.W2
            gb2 = dout.sum(0)
            dH = (dout @ self.W2.T) * (1.0 - H ** 2)
            gW1 = Xb.T @ dH + l2 * self.W1
            gb1 = dH.sum(0)
            for i, (p, g) in enumerate(zip(self.params(), (gW1, gb1, gW2, gb2))):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** step)
                vh = v[i] / (1 - b2 ** step)
                p -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        return loss


def _background_qos() -> None:
    """Run this thread at macOS background QoS (efficiency cores, yields to
    everything). Best effort: elsewhere it is simply a normal thread.
    Never the main thread: that one draws her, and it would stay demoted."""
    if threading.current_thread() is threading.main_thread():
        return
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
        libc.pthread_set_qos_class_self_np(ctypes.c_uint(0x09), ctypes.c_int(0))
    except Exception:
        pass


class Brain:
    def __init__(
        self,
        actions: Iterable[str],
        *,
        path: Path | None = None,
        lifelog: LifeLog | None = None,
        min_samples: int = 300,
        max_weight: float = 0.5,
        retrain_every_s: float = 12 * 3600.0,
        retrain_idle_s: float = 600.0,
        seed: int = 0,
        steps: int = 1500,
    ) -> None:
        self.actions = list(dict.fromkeys(list(actions) + [ALERT]))
        self.index = {a: i for i, a in enumerate(self.actions)}
        self.path = path
        self.log = lifelog or LifeLog(None)
        self.min_samples = int(min_samples)
        self.max_weight = float(max_weight)
        self.retrain_every_s = float(retrain_every_s)
        self.retrain_idle_s = float(retrain_idle_s)
        self.seed = seed
        self.steps = steps
        self.net: MLP | None = None
        self.samples = 0
        self.skill = 0.0
        self.trained_at: float | None = None
        self.new_since_train = 0
        self._training = False
        self._lock = threading.Lock()

    # -- recording ------------------------------------------------------------
    def record(self, wall: float, head: str, action: str, reward: float, x: np.ndarray) -> None:
        self.log.append(wall, head, action, reward, x)
        self.new_since_train += 1

    # -- using it -----------------------------------------------------------------
    @property
    def weight(self) -> float:
        """How much of the vote the brain gets right now, in [0, max_weight]."""
        if self.net is None or self.skill <= 0.0:
            return 0.0
        whisper = WHISPER * self.max_weight
        if self.samples < self.min_samples:
            return whisper * self.samples / self.min_samples
        k = min(1.0, self.skill / FULL_SKILL)
        return whisper + (self.max_weight - whisper) * k

    def _predict(self, x: np.ndarray) -> np.ndarray | None:
        with self._lock:
            net = self.net
        if net is None:
            return None
        return net.forward(np.asarray(x, np.float32)[None, :])[0]

    def factors(self, x: np.ndarray, arms: Iterable[str]) -> dict[str, float]:
        """Blended multipliers per behaviour: 1.0 when the brain has no say."""
        arms = list(arms)
        w = self.weight
        out = pred = None
        if w > 0.0:
            pred = self._predict(x)
        if pred is None:
            return {a: 1.0 for a in arms}
        known = [a for a in arms if a in self.index]
        if not known:
            return {a: 1.0 for a in arms}
        vals = np.array([pred[self.index[a]] for a in known])
        rel = vals - vals.mean()
        out = {a: 1.0 for a in arms}
        for a, r in zip(known, rel):
            f = float(np.clip(math.exp(2.0 * float(r)), FACTOR_MIN, FACTOR_MAX))
            out[a] = f ** w
        return out

    def alert(self, x: np.ndarray) -> float | None:
        """Predicted reward of showing an alert in this context, in [-1, 1]."""
        if self.weight <= 0.0:
            return None
        pred = self._predict(x)
        if pred is None:
            return None
        return float(np.clip(pred[self.index[ALERT]], -1.0, 1.0))

    # -- training -------------------------------------------------------------------
    def due(self, wall: float, key_idle: float) -> bool:
        """Worth training now? Only when idle, rarely, and with new data."""
        if self._training or key_idle < self.retrain_idle_s:
            return False
        if self.new_since_train < RETRAIN_NEW and self.trained_at is not None:
            return False
        return self.trained_at is None or wall - self.trained_at >= self.retrain_every_s

    def train(self, wall: float | None = None) -> bool:
        """One training run, synchronously. True if a model came out of it."""
        wall = time.time() if wall is None else wall
        X, A, R = self.log.arrays(wall, self.index, self.index[ALERT])
        self.trained_at = wall
        self.new_since_train = 0
        if len(X) < 30:
            return False
        cut = int(len(X) * 0.8)
        net = MLP(N_FEATURES, HIDDEN, len(self.actions), seed=self.seed)
        net.fit(X[:cut], A[:cut], R[:cut], steps=self.steps, seed=self.seed)
        skill = self._skill(net, X, A, R, cut)
        # Final model on everything; the holdout only decides its weight.
        final = MLP(N_FEATURES, HIDDEN, len(self.actions), seed=self.seed)
        final.fit(X, A, R, steps=self.steps, seed=self.seed)
        with self._lock:
            self.net = final
            self.samples = len(X)
            self.skill = skill
        self.save()
        log.info("learn: brain trained on %d decisions (skill %.2f, weight %.2f)",
                 len(X), skill, self.weight)
        return True

    def _skill(self, net: MLP, X: np.ndarray, A: np.ndarray, R: np.ndarray, cut: int) -> float:
        """1 - holdout MSE / MSE of the per-action training average."""
        if len(X) - cut < 10:
            return 0.0
        means = np.zeros(len(self.actions), np.float32)
        overall = float(R[:cut].mean())
        for i in range(len(self.actions)):
            sel = A[:cut] == i
            means[i] = float(R[:cut][sel].mean()) if sel.any() else overall
        Xh, Ah, Rh = X[cut:], A[cut:], R[cut:]
        base = float(np.mean((means[Ah] - Rh) ** 2))
        model = float(np.mean((net.forward(Xh)[np.arange(len(Xh)), Ah] - Rh) ** 2))
        if base <= 1e-9:
            return 0.0
        return float(max(-1.0, 1.0 - model / base))

    def maybe_train(self, wall: float, key_idle: float,
                    spawn: Callable[[Callable[[], None]], Any] | None = None) -> bool:
        """Start a background training run if one is due. Returns at once."""
        if not self.due(wall, key_idle):
            return False
        self._training = True

        def run() -> None:
            _background_qos()
            try:
                self.train(wall)
            except Exception:
                log.warning("learn: brain training failed", exc_info=True)
            finally:
                self._training = False

        if spawn is not None:
            spawn(run)
        else:
            threading.Thread(target=run, name="wawi-learn-train", daemon=True).start()
        return True

    # -- persistence -------------------------------------------------------------------
    def save(self) -> None:
        if self.path is None or self.net is None:
            return
        try:
            tmp = self.path.with_name(self.path.name + ".tmp.npz")
            n = self.net
            np.savez(tmp, W1=n.W1, b1=n.b1, W2=n.W2, b2=n.b2,
                     actions=np.asarray(self.actions),
                     meta=np.asarray([self.samples, self.skill, self.trained_at or 0.0]))
            os.replace(tmp, self.path)
        except OSError:
            log.debug("learn: could not save the brain", exc_info=True)

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as z:
                if [str(a) for a in z["actions"]] != self.actions \
                        or z["W1"].shape != (N_FEATURES, HIDDEN):
                    log.info("learn: brain file is for other actions; retraining later")
                    return
                net = MLP(N_FEATURES, HIDDEN, len(self.actions))
                net.W1, net.b1, net.W2, net.b2 = (z[k].astype(np.float32)
                                                  for k in ("W1", "b1", "W2", "b2"))
                samples, skill, trained = (float(v) for v in z["meta"])
            self.net, self.samples, self.skill = net, int(samples), skill
            self.trained_at = trained or None
        except Exception:
            log.warning("learn: brain file unreadable; starting fresh", exc_info=True)
