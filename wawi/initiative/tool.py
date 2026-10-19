"""The voice tool: "¿qué cosas hacés sola?" and "no me avises más de eso".

Answered locally from the initiative's own switches; no Claude call needed
beyond picking the tool. "Eso" is the last thing she showed on her own
(within half an hour). Kept here so wawi/tools.py only needs one small block.
"""

from __future__ import annotations

from typing import Any

from wawi.initiative.core import KINDS

SCHEMA: dict[str, Any] = {
    "name": "initiative",
    "animation": "notebook",
    # Terse: every schema is re-sent (and paid for) on every turn.
    "description": "¿Qué hacés sola? (list); no me avises más / avisame (off/on; "
                   "kind vacío = lo último que mostraste, all = todo)",
    "input_schema": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["list", "off", "on"]},
        "kind": {"type": "string", "enum": [*KINDS, "all"]}}},
}

#: How each level reads in a sentence.
HOW = {"quiet": "en silencio", "offer": "con un globito", "say": "y te lo digo en voz alta",
       "auto": "y lo hago directo"}


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


def listing(engine: Any) -> str:
    if engine is None or not engine.enabled:
        return "Ahora no hago nada sola: la iniciativa está apagada (la prendés en el menú, Iniciativa)."
    on, off = [], []
    for _kind, title, level in engine.describe():
        if level == "off":
            off.append(title)
        else:
            on.append(f"{title} {HOW.get(level, '')}".strip())
    text = "Sola hago esto: " + _join(on) + "." if on else "Ahora no hago nada sola."
    if off:
        text += " Apagado: " + _join(off) + "."
    return text


def run(engine: Any, action: str = "list", kind: str = "") -> tuple[bool, str, dict]:
    """(ok, speak, data) for the tool."""
    action = (action or "list").strip().lower()
    kind = (kind or "").strip()
    if engine is None:
        return False, "La iniciativa está apagada en la configuración.", {"error": "off"}
    if action == "list":
        return True, listing(engine), {"kinds": engine.describe(), "enabled": engine.enabled}
    on = action == "on"
    if action not in ("on", "off"):
        return False, "No entendí qué hacer con eso.", {"error": "bad_action"}
    if kind == "all":
        engine.set_enabled(on)
        text = "Listo, vuelvo a hacer cosas sola." if on else "Listo, no hago nada sola hasta que me digas."
        return True, text, {"enabled": on}
    if not on and not engine.enabled:
        return True, "Ya no estoy haciendo nada sola.", {"enabled": False}
    if not kind:
        kind = engine.recent_kind() or ""
        if not kind:
            return False, "¿De qué no te aviso más? No mostré nada mío hace poco.", {"error": "no_recent"}
    if kind not in KINDS:
        return False, "No conozco ese aviso.", {"error": "unknown_kind"}
    if on and not engine.enabled:
        engine.set_enabled(True)
    engine.set_kind(kind, on)
    title = KINDS[kind].title
    if on:
        return True, f"Listo, de nuevo {title}.", {"kind": kind, "on": True}
    return True, f"Listo, ya no {title}.", {"kind": kind, "on": False}
