"""Benchmark the mind's small local model on this machine.

Usage (needs mlx-lm, which Wawi treats as optional):

    pip install mlx-lm
    python scripts/bench_mind.py [mlx-community/Qwen3-0.6B-4bit ...]

Prints load time, first and warm per-call latency, and what the bubble and
triage parsers make of the output -- the same code paths Wawi runs, so a bad
model shows up here as "fallback" lines instead of on the desktop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wawi.mind import local  # noqa: E402
from wawi.mind.settings import SUGGESTED_LOCAL_MODEL  # noqa: E402
from wawi.reactions import Event  # noqa: E402

SUMMARIES = (
    local.StateSummary(16, 30, {"mood": "aburrida", "hunger": 0.5}, ("Empezó a llover",)),
    local.StateSummary(9, 5, {"hunger": 0.9}, ()),
    local.StateSummary(23, 40, {"energy": 0.1}, ("El disco está casi lleno",)),
)
EVENTS = (
    Event("rss.item", "info", title="Salió Python 3.15 beta 2"),
    Event("weather.rain", "notice", title="Llueve en 15 minutos"),
    Event("clock.hour", "info", title="Son las 15"),
    Event("battery.low", "notice", title="Batería al 15%"),
)


def main(repos: list[str]) -> None:
    for repo in repos or [SUGGESTED_LOCAL_MODEL]:
        model = local.LocalModel(local._MLXRunner(repo), name=repo, load_timeout_s=600, call_timeout_s=30)
        t0 = time.perf_counter()
        model.warm_up()
        model.generate("x", "x", max_tokens=1)
        print(f"\n== {repo}: load {model.stats.get('load_s', time.perf_counter() - t0):.2f}s")
        lat = []
        for summary in SUMMARIES * 2:
            t = time.perf_counter()
            raw = model.generate(local.BUBBLE_SYSTEM, summary.render(), max_tokens=40,
                                 temperature=0.8, timeout=30)
            lat.append(time.perf_counter() - t)
            parsed = local.parse_bubble(raw)
            print(f"  bubble {lat[-1]*1000:5.0f}ms  {'OK ' if parsed else 'FALLBACK'} {raw!r}"[:120])
        print(f"  bubble warm avg {sum(lat[1:]) / (len(lat) - 1) * 1000:.0f}ms")
        triage = local.Triage(model, timeout_s=30)
        for ev in EVENTS:
            t = time.perf_counter()
            verdict = triage.decide(ev)
            print(f"  triage {(time.perf_counter() - t)*1000:5.0f}ms  {ev.kind:13s} -> {verdict}")
        model.close()


if __name__ == "__main__":
    main(sys.argv[1:])
