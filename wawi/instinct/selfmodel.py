"""Who Wawi is, what she noticed today, what she knows about the owner.

A compact, bounded summary for the optional local model (the mind's bubble
generator), so a bubble can reference real things -- "Hoy codeaste 3 horas
seguidas, ¿pausa?" -- instead of the hour of the day only. It also yields a
few ready-made lines for the canned path, so the same awareness shows with
no model at all.

Everything here is short on purpose: a 1.7B model reads a few lines well and
a page badly, and the prompt is re-read on every bubble.
"""

from __future__ import annotations

import time
from typing import Any

from wawi.instinct.categories import FOCUS, word
from wawi.instinct.observer import ActivityObserver, Context
from wawi.instinct.patterns import PatternMiner, _duration

IDENTITY = "Sos Wawi, una perrita que vive en el escritorio de tu dueño."
#: Bounds: at most this many facts of each kind, each at most this long.
MAX_NOTES = 4
MAX_FACTS = 3
MAX_CHARS = 80


def _short(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1].rstrip() + "…"


def today_notes(ctx: Context | None) -> list[str]:
    """What she noticed today, from the live context."""
    if ctx is None:
        return []
    notes: list[str] = []
    if ctx.active_min_today >= 30:
        notes.append(f"Hoy tu dueño lleva {_duration(ctx.active_min_today)} en la compu")
    if ctx.focus_s >= 20 * 60:
        notes.append(f"Está concentrado hace {_duration(ctx.focus_s / 60)}"
                     + (f" en {word(ctx.category)}" if ctx.category in FOCUS else ""))
    elif ctx.focus_ended and ctx.wall - ctx.focus_ended[0] < 3600:
        notes.append(f"Recién terminó un bloque de {_duration(ctx.focus_ended[1] / 60)}")
    if ctx.active and ctx.since_break_s >= 60 * 60:
        notes.append(f"Lleva {_duration(ctx.since_break_s / 60)} sin pausa")
    if ctx.returned and ctx.wall - ctx.returned[0] < 600 and ctx.returned[1] != float("inf") \
            and ctx.returned[1] >= 30 * 60:
        notes.append(f"Volvió hace poco después de {_duration(ctx.returned[1] / 60)} afuera")
    if ctx.switches >= 14:
        notes.append("Está saltando mucho de app en app")
    return [_short(n) for n in notes[:MAX_NOTES]]


def ready_lines(ctx: Context | None) -> list[str]:
    """Canned bubbles grounded in today (each at most 60 characters)."""
    if ctx is None or not ctx.active:
        return []
    lines: list[str] = []
    if ctx.since_break_s >= 90 * 60:
        h = ctx.since_break_s / 3600
        what = "codeaste" if ctx.category in ("code", "terminal") else "estás"
        lines.append(f"Hoy {what} {_duration(h * 60)} seguidas, ¿pausa?"
                     if h >= 2 else f"Hace {_duration(h * 60)} que no parás, ¿pausa?")
    if ctx.focus_s >= 45 * 60:
        lines.append("Shh, estás en racha. Yo vigilo.")
    if ctx.active_min_today >= 8 * 60:
        lines.append("Hoy le diste un montón. Yo también, eh.")
    return [ln for ln in lines if len(ln) <= 60]


def known_facts(miner: PatternMiner | None, now: float | None = None,
                min_conf: float = 0.5) -> list[str]:
    if miner is None:
        return []
    try:
        pats = miner.patterns(now, min_conf)
    except Exception:
        return []
    return [_short(p.text) for p in pats[:MAX_FACTS]]


def self_notes(observer: ActivityObserver | None, miner: PatternMiner | None,
               now: float | None = None) -> dict[str, Any]:
    """{"notes": [...], "facts": [...], "lines": [...]} -- all short Spanish."""
    now = time.time() if now is None else now
    ctx = None
    if observer is not None:
        try:
            ctx = observer.context(now)
        except Exception:
            ctx = None
    return {"notes": today_notes(ctx), "facts": known_facts(miner, now),
            "lines": ready_lines(ctx)}


def render(summary: dict[str, Any], pet: dict[str, Any] | None = None) -> str:
    """The self-model as a few lines of prompt."""
    parts = [IDENTITY]
    mood = (pet or {}).get("mood") if isinstance(pet, dict) else None
    if mood:
        parts.append(f"Tu ánimo: {str(mood)[:20]}.")
    notes = summary.get("notes") or []
    if notes:
        parts.append("Hoy notaste: " + "; ".join(notes) + ".")
    facts = summary.get("facts") or []
    if facts:
        parts.append("Sabés de tu dueño: " + "; ".join(facts) + ".")
    return " ".join(parts)
