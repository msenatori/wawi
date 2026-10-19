"""What she tells the owner about fixing herself: the day's section and the
history page. Plain markdown, Spanish, no internals he did not ask for."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

#: outcome -> how it reads in a sentence about «title».
_LINES = {
    "applied": "Me arreglé «{title}».",
    "watching": "Me arreglé «{title}»; lo estoy vigilando.",
    "rolled_back": "Intenté arreglar «{title}», pero lo deshice: {reason}.",
    "undone": "Deshiciste mi arreglo de «{title}».",
    "revert_blocked": "Mi arreglo de «{title}» falló y no pude deshacerlo solo ({reason}). "
                      "Necesito que lo mires.",
    "rejected": "Intenté arreglar «{title}» y no pasó mis controles: {reason}.",
    "error": "Intenté arreglar «{title}» y algo se rompió en el camino: {reason}.",
    "pending_owner": "Tengo listo un arreglo para «{title}», pero te toca decidir a vos.",
    "discarded": "Descartaste mi arreglo de «{title}».",
    "running": "Me estoy arreglando «{title}».",
    "waiting_clean": "Tengo «{title}» listo; espero a que tu repo esté limpio para aplicarlo.",
}


def attempt_line(row: dict) -> str:
    template = _LINES.get(str(row.get("outcome")), "«{title}»: {reason}.")
    reasons = row.get("reasons") or []
    reason = str(reasons[0] if reasons else row.get("outcome", "")).rstrip(".")
    line = template.format(title=row.get("title", "?"), reason=reason)
    commit = str(row.get("commit") or "")
    if row.get("outcome") in ("applied", "watching") and commit:
        line += f" (commit `{commit[:8]}`)"
    return line


def today_markdown(attempts: Iterable[dict], owner_requests: Iterable[dict],
                   today: str | None = None) -> str:
    """The "lo que me arreglé hoy" section; "" when there is nothing to say."""
    day = today or datetime.now().strftime("%Y-%m-%d")
    rows = [a for a in attempts
            if str(a.get("started", "")).startswith(day) or str(a.get("finished", "")).startswith(day)]
    requests = list(owner_requests)
    if not rows and not requests:
        return ""
    out = ["## Lo que me arreglé hoy", ""]
    out += [f"- {attempt_line(a)}" for a in rows] or ["- Hoy no toqué nada de mí."]
    if requests:
        out += ["", "**Para seguir necesito algo tuyo:**", ""]
        out += [f"- {r.get('text', '')} _(por «{r.get('issue', '')}»)_" for r in requests]
    return "\n".join(out) + "\n"


def history_markdown(attempts: list[dict], pending: dict | None, owner_requests: list[dict],
                     enabled: bool, quota: tuple[int, int]) -> str:
    used, total = quota
    out = ["# Autoarreglo", "",
           f"Estado: **{'prendido' if enabled else 'apagado'}** · intentos de hoy: {used} de {total}.",
           ""]
    if pending:
        out += ["## Esperando tu decisión", "",
                f"«{pending.get('title', '?')}»: {', '.join(pending.get('ask_reasons') or [])}.",
                "Decime «aplicá el arreglo» o «descartalo», o usá el menú Autoarreglo.", ""]
    if owner_requests:
        out += ["## Lo que necesito de vos", ""]
        out += [f"- {r.get('text', '')} _(por «{r.get('issue', '')}»)_" for r in owner_requests]
        out.append("")
    out += ["## Historial", ""]
    rows = list(reversed(attempts))[:30]
    if not rows:
        out.append("Todavía no intenté arreglarme nada.")
    for row in rows:
        when = str(row.get("started", ""))[:16]
        out.append(f"- `{when}` {attempt_line(row)}")
    return "\n".join(out) + "\n"


def pending_markdown(pending: dict, diff: str) -> str:
    """The board page for an ``ask`` change: what, why it needs him, the diff."""
    reasons = pending.get("ask_reasons") or []
    return "\n".join([
        f"# Arreglo listo: {pending.get('title', '?')}",
        "",
        str(pending.get("summary") or "").strip(),
        "",
        "**Te pregunto porque:** " + (", ".join(reasons) if reasons else "prefiero que lo veas") + ".",
        "",
        "Pasó los tests y la revisión. Decime **«aplicá el arreglo»** o **«descartalo»** "
        "(o desde el menú Autoarreglo: Aplicar / Descartar).",
        "",
        "```diff",
        diff[:150_000],
        "```",
    ]) + "\n"
