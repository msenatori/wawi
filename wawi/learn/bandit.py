"""Which of her idle behaviours the owner enjoys: a contextual bandit.

Thompson sampling with a Beta posterior per (behaviour, context), where the
context is the part of the day times her mood ("tarde:bored"). Each posterior
borrows a little from the behaviour's evidence across all contexts, so a
behaviour seen five times in the morning is not a blank page in the evening.

The bandit never chooses a behaviour itself. The animator still rolls its
weighted dice over BEHAVIOUR_WEIGHTS x the mood bias; the bandit only hands
back a multiplier per behaviour, clamped to [factor_min, factor_max] — so a
behaviour the owner never reacts to becomes rarer, never extinct, and the
sampling noise keeps exploration alive by itself.

Evidence decays with a half-life (weeks), so a taste that changed is
re-learned instead of being outvoted by last year forever.

Pure numpy, no I/O: the learner persists :meth:`to_dict`.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

#: How much of a behaviour's all-context evidence each context borrows.
POOL = 0.25
#: A behaviour that got no reaction at all: weak evidence she was not
#: noticed. Most behaviours get none, so it must be small, or everything
#: would sink together.
NEUTRAL_FAIL = 0.25
#: Below this the (decayed) evidence of a cell is forgotten entirely.
FORGET = 0.02


def part_of_day(hour: int) -> str:
    if 6 <= hour < 12:
        return "mañana"
    if 12 <= hour < 20:
        return "tarde"
    if 20 <= hour < 24:
        return "noche"
    return "madrugada"


def context(hour: int, mood: str) -> str:
    return f"{part_of_day(int(hour) % 24)}:{mood or 'content'}"


class BehaviourBandit:
    def __init__(
        self,
        *,
        factor_min: float = 0.5,
        factor_max: float = 2.0,
        half_life_days: float = 21.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.factor_min = float(factor_min)
        self.factor_max = float(factor_max)
        self.half_life_s = float(half_life_days) * 86400.0
        self.rng = rng if rng is not None else np.random.default_rng()
        #: "arm|context" -> [successes, failures] (weighted, decayed).
        self.cells: dict[str, list[float]] = {}
        #: arm -> [successes, failures] across contexts.
        self.arms: dict[str, list[float]] = {}
        self.decayed_at: float | None = None
        self.updates = 0

    # -- learning ---------------------------------------------------------
    @staticmethod
    def outcome(reward: float) -> tuple[float, float]:
        """A reward in [-1, 1] as (success, failure) increments."""
        if reward > 0:
            return float(reward), 0.0
        if reward < 0:
            return 0.0, float(-reward)
        return 0.0, NEUTRAL_FAIL

    def update(self, arm: str, ctx: str, reward: float) -> None:
        s, f = self.outcome(reward)
        for table, key in ((self.cells, f"{arm}|{ctx}"), (self.arms, arm)):
            cell = table.setdefault(key, [0.0, 0.0])
            cell[0] += s
            cell[1] += f
        self.updates += 1

    def decay(self, wall: float) -> None:
        """Fade all evidence by the time since the last decay (half-life)."""
        if self.decayed_at is None or wall < self.decayed_at:
            self.decayed_at = wall
            return
        dt = wall - self.decayed_at
        if dt < 60.0:
            return
        k = 0.5 ** (dt / self.half_life_s)
        self.decayed_at = wall
        for table in (self.cells, self.arms):
            for key in list(table):
                cell = table[key]
                cell[0] *= k
                cell[1] *= k
                if cell[0] + cell[1] < FORGET:
                    del table[key]

    # -- using it ---------------------------------------------------------
    def posterior(self, arm: str, ctx: str) -> tuple[float, float]:
        cs, cf = self.cells.get(f"{arm}|{ctx}", (0.0, 0.0))
        gs, gf = self.arms.get(arm, (0.0, 0.0))
        return 1.0 + cs + POOL * gs, 1.0 + cf + POOL * gf

    def factors(self, arms: Iterable[str], ctx: str) -> dict[str, float]:
        """One Thompson draw per arm, relative to the draws' mean, clamped.

        With no evidence every arm draws from the same Beta(1, 1): the
        factors scatter around 1 and the animator's own weights rule.
        """
        names = list(dict.fromkeys(arms))
        if not names:
            return {}
        draws = np.array([self.rng.beta(*self.posterior(a, ctx)) for a in names])
        mean = float(draws.mean()) or 1e-9
        out = {}
        for name, d in zip(names, draws):
            out[name] = float(min(self.factor_max, max(self.factor_min, d / mean)))
        return out

    def expected(self, arm: str, ctx: str | None = None) -> float:
        """Posterior mean engagement (all contexts when *ctx* is None)."""
        if ctx is None:
            s, f = self.arms.get(arm, (0.0, 0.0))
            return (1.0 + s) / (2.0 + s + f)
        a, b = self.posterior(arm, ctx)
        return a / (a + b)

    def evidence(self, arm: str) -> float:
        s, f = self.arms.get(arm, (0.0, 0.0))
        return s + f

    def favourites(self, n: int = 3, min_evidence: float = 2.0) -> list[tuple[str, float]]:
        """The behaviours the owner reacts to most, best first."""
        known = [(a, self.expected(a)) for a in self.arms if self.evidence(a) >= min_evidence]
        return sorted(known, key=lambda kv: -kv[1])[:n]

    def least(self, n: int = 2, min_evidence: float = 3.0) -> list[tuple[str, float]]:
        known = [(a, self.expected(a)) for a in self.arms if self.evidence(a) >= min_evidence]
        return sorted(known, key=lambda kv: kv[1])[:n]

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        r = lambda c: [round(c[0], 4), round(c[1], 4)]  # noqa: E731
        return {
            "cells": {k: r(v) for k, v in self.cells.items()},
            "arms": {k: r(v) for k, v in self.arms.items()},
            "decayed_at": self.decayed_at,
            "updates": self.updates,
        }

    def load(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            return

        def table(x: Any) -> dict[str, list[float]]:
            out: dict[str, list[float]] = {}
            for k, v in (x or {}).items() if isinstance(x, dict) else ():
                try:
                    s, f = float(v[0]), float(v[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(s) and math.isfinite(f) and s >= 0 and f >= 0:
                    out[str(k)] = [s, f]
            return out

        self.cells = table(raw.get("cells"))
        self.arms = table(raw.get("arms"))
        at = raw.get("decayed_at")
        self.decayed_at = float(at) if isinstance(at, (int, float)) else None
        try:
            self.updates = int(raw.get("updates", 0))
        except (TypeError, ValueError):
            self.updates = 0
