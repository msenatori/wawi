"""Which alerts the owner cares about, learned from what they do with them.

Per event kind (``weather.hot``) and, where it means something, per source
(the RSS feed, the website, the service), a Beta posterior over "the owner
engages with this". Outcomes, all decided by the learner from timing:

* **engaged** — touched or petted her, talked to her, within the reward
  window after the bubble appeared: a full success;
* **dismissed** — clicked the bubble away within a few seconds: a full
  failure ("not now, not this");
* **ignored** — the bubble expired with no reaction at all: a weak failure,
  because most people simply keep working, and that is fine;
* a bubble clicked away after it was read counts as nothing.

:meth:`AlertLearner.score` maps the posterior to [-1, 1] (0 = no idea) with
the amount of evidence behind it. The reflex layer uses it to batch the
info/notice kinds the owner keeps brushing away. Urgent never asks.
"""

from __future__ import annotations

import math
from typing import Any

#: Outcome -> (success, failure) increments.
OUTCOMES: dict[str, tuple[float, float]] = {
    "engaged": (1.0, 0.0),
    "dismissed": (0.0, 1.0),
    "ignored": (0.0, 0.3),
    "read": (0.0, 0.0),
}
#: Prior pseudo-counts: Beta(2, 2), a mild "no idea" centred on 0.5.
PRIOR = 2.0
#: Weight of the kind's evidence when scoring one source of that kind.
KIND_POOL = 0.5
FORGET = 0.02


def source_of(event: Any) -> str:
    """The thing an event is about, at the grain the owner has tastes for:
    the feed for news (each item has its own key), the key otherwise."""
    data = getattr(event, "data", None) or {}
    kind = str(getattr(event, "kind", "") or "")
    if kind.startswith("rss."):
        return str(data.get("feed") or "")
    return str(getattr(event, "key", "") or data.get("url") or "")


class AlertLearner:
    def __init__(self, *, half_life_days: float = 21.0) -> None:
        self.half_life_s = float(half_life_days) * 86400.0
        self.kinds: dict[str, list[float]] = {}
        self.sources: dict[str, list[float]] = {}
        self.decayed_at: float | None = None

    def record(self, event: Any, outcome: str) -> None:
        s, f = OUTCOMES.get(outcome, (0.0, 0.0))
        if s == 0.0 and f == 0.0:
            return
        kind = str(getattr(event, "kind", "") or "")
        if not kind:
            return
        cell = self.kinds.setdefault(kind, [0.0, 0.0])
        cell[0] += s
        cell[1] += f
        src = source_of(event)
        if src:
            cell = self.sources.setdefault(f"{kind}|{src}", [0.0, 0.0])
            cell[0] += s
            cell[1] += f

    def score(self, event: Any) -> tuple[float, float]:
        """(score in [-1, 1], evidence). Positive = the owner engages."""
        kind = str(getattr(event, "kind", "") or "")
        ks, kf = self.kinds.get(kind, (0.0, 0.0))
        ss, sf = self.sources.get(f"{kind}|{source_of(event)}", (0.0, 0.0))
        # The source's own evidence first, the kind's as a softer prior.
        s = ss + KIND_POOL * ks
        f = sf + KIND_POOL * kf
        mean = (PRIOR + s) / (2 * PRIOR + s + f)
        return 2.0 * mean - 1.0, ss + sf + KIND_POOL * (ks + kf)

    def decay(self, wall: float) -> None:
        if self.decayed_at is None or wall < self.decayed_at:
            self.decayed_at = wall
            return
        dt = wall - self.decayed_at
        if dt < 60.0:
            return
        k = 0.5 ** (dt / self.half_life_s)
        self.decayed_at = wall
        for table in (self.kinds, self.sources):
            for key in list(table):
                table[key][0] *= k
                table[key][1] *= k
                if sum(table[key]) < FORGET:
                    del table[key]

    def liked(self, n: int = 2, min_evidence: float = 3.0) -> list[tuple[str, float]]:
        return self._ranked(n, min_evidence, reverse=True)

    def disliked(self, n: int = 2, min_evidence: float = 3.0) -> list[tuple[str, float]]:
        return self._ranked(n, min_evidence, reverse=False)

    def _ranked(self, n: int, min_evidence: float, reverse: bool) -> list[tuple[str, float]]:
        out = []
        for kind, (s, f) in self.kinds.items():
            if s + f >= min_evidence:
                out.append((kind, 2.0 * (PRIOR + s) / (2 * PRIOR + s + f) - 1.0))
        out.sort(key=lambda kv: kv[1], reverse=reverse)
        return [kv for kv in out if (kv[1] > 0.1 if reverse else kv[1] < -0.1)][:n]

    def to_dict(self) -> dict[str, Any]:
        r = lambda c: [round(c[0], 4), round(c[1], 4)]  # noqa: E731
        return {"kinds": {k: r(v) for k, v in self.kinds.items()},
                "sources": {k: r(v) for k, v in self.sources.items()},
                "decayed_at": self.decayed_at}

    def load(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            return

        def table(x: Any) -> dict[str, list[float]]:
            out: dict[str, list[float]] = {}
            for k, v in (x.items() if isinstance(x, dict) else ()):
                try:
                    s, f = float(v[0]), float(v[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(s) and math.isfinite(f) and s >= 0 and f >= 0:
                    out[str(k)] = [s, f]
            return out

        self.kinds = table(raw.get("kinds"))
        self.sources = table(raw.get("sources"))
        at = raw.get("decayed_at")
        self.decayed_at = float(at) if isinstance(at, (int, float)) else None
