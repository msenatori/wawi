"""The model-facing half: two terse schemas (they are re-sent every turn).

- ``learn_capability``: "No sé hacer eso todavía, me lo armo" -- files a
  capability request in the issue journal. Active whenever the journal is.
- ``self_fix``: "arreglate lo de…", "deshacé el último arreglo", "¿qué te
  arreglaste?", and Aplicar/Descartar a pending change. Active while the
  service is installed.

Handlers return ``(ok, speak, data)`` like the rest of wawi/tools.py.
"""

from __future__ import annotations

from typing import Any

LEARN_SPEC: dict[str, Any] = {
    "name": "learn_capability",
    "animation": "notebook",
    "example": "Wawi, aprendé a leerme el clima de mañana",
    "ability": "Aprender algo nuevo que no sé hacer",
    "description": ("Si te piden algo que no podés hacer, o te dicen «aprendé a…», anotalo para "
                    "armártelo sola. what: la capacidad, corta. from_user: true si él pidió "
                    "aprenderlo."),
    "input_schema": {"type": "object", "properties": {
        "what": {"type": "string"}, "detail": {"type": "string"},
        "from_user": {"type": "boolean"}}, "required": ["what"]},
}

SELFFIX_SPEC: dict[str, Any] = {
    "name": "self_fix",
    "animation": "busy",
    "example": "Wawi, ¿qué te arreglaste hoy?",
    "ability": "Arreglarme sola y contarte qué me arreglé",
    "description": ("Tu autoarreglo. action: fix (arreglate lo de <what>), undo (deshacé el último "
                    "arreglo), report (¿qué te arreglaste?), history (mostralo en la pizarra), "
                    "apply/discard (el arreglo que espera su ok), requests (qué necesitás de él)."),
    "input_schema": {"type": "object", "properties": {
        "action": {"type": "string",
                   "enum": ["fix", "undo", "report", "history", "apply", "discard", "requests"]},
        "what": {"type": "string"}}, "required": ["action"]},
}


def learn_capability(journal: Any, what: str, detail: str = "",
                     from_user: bool = False) -> tuple[bool, str, dict]:
    what = " ".join((what or "").split())[:120]
    if not what:
        return False, "Decime qué querés que aprenda.", {"error": "empty"}
    if journal is None or not getattr(journal, "enabled", False):
        return False, "No tengo dónde anotarlo.", {"error": "journal_disabled"}
    from wawi.selfimprove import KIND_CAPABILITY

    issue = journal.record(KIND_CAPABILITY, f"Aprender: {what}", detail=detail or what,
                           subject=what.lower(),
                           context={"source": "user" if from_user else "model"})
    speak = ("Dale, lo aprendo en un rato libre." if from_user
             else "No sé hacer eso todavía, me lo armo.")
    return True, speak, {"title": what, "count": getattr(issue, "count", 1)}


def self_fix(service: Any, action: str, what: str = "") -> tuple[bool, str, dict]:
    if service is None:
        return False, "No tengo el autoarreglo disponible.", {"error": "selffix_off"}
    action = (action or "").strip().lower()
    if action == "fix":
        ok, speak = service.fix_now(what or "")
        return ok, speak, {}
    if action == "undo":
        ok, speak = service.undo_last()
        return ok, speak, {}
    if action in ("apply", "discard"):
        ok, speak = service.decide_pending(action == "apply")
        return ok, speak, {}
    if action == "history":
        shown = service.show_history()
        return shown, ("Te dejé el historial en la pizarra." if shown
                       else "No tengo la pizarra a mano."), {}
    if action == "requests":
        rows = service.owner_requests()
        if not rows:
            return True, "Por ahora no necesito nada tuyo.", {"requests": []}
        return True, rows[0]["text"], {"requests": rows}
    if action == "report":
        return True, service.spoken_report(), {"markdown": service.today_markdown()[:3000]}
    return False, "No entendí qué hago con el autoarreglo.", {"error": "bad_action"}
