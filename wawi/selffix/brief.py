"""The brief a coding agent gets: tight, scrubbed, and with the rules in it.

Everything that comes from the room (the issue's words, the log excerpt)
goes through :func:`wawi.selffix.scrub.scrub` first: it leaves the Mac for a
model provider.
"""

from __future__ import annotations

import re
from pathlib import Path

from wawi.selffix.issues import IssueRecord, plain
from wawi.selffix.scrub import scrub

#: Paths a self-fix may never touch. Mirrored by the forbidden-path gate.
FORBIDDEN_PATHS = (
    "config.yaml",
    "*_local.json",
    "private/*",
    "wawi/selffix/*",
    "scripts/wawictl",
    "LICENSE",
    ".git/*",
)

TEST_COMMAND = ("-m pytest -q -p no:cacheprovider --deselect "
                "tests/test_listener.py::TestLifecycle::test_stop_does_not_hang_with_a_full_queue")


def log_excerpt(log_path: Path | None, issue: IssueRecord, lines: int = 40,
                *, max_bytes: int = 4_000_000) -> str:
    """The last *lines* lines of the log about *issue*, scrubbed.

    Matched by fingerprint first (the journal logs ``fp=``), then by the
    issue's distinctive words; plus the ERROR/traceback lines around them.
    """
    if log_path is None or lines <= 0:
        return ""
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as fh:
            fh.seek(max(0, size - max_bytes))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    words = [w for w in re.findall(r"[a-z0-9]{5,}", plain(issue.title))][:6]
    picked = []
    for line in text.splitlines():
        low = plain(line)
        if (issue.fingerprint and f"fp={issue.fingerprint}" in line) or (
                words and sum(1 for w in words if w in low) >= min(2, len(words))):
            picked.append(line)
    return scrub("\n".join(picked[-lines:]))


def build_brief(issue: IssueRecord, *, excerpt: str = "", previous_failures: list[str] | None = None,
                python: str = ".venv/bin/python", max_lines: int = 400, max_files: int = 12) -> str:
    """The author's prompt, in Spanish like the repo's conversation."""
    kind = "capacidad nueva" if issue.is_capability else "bug"
    out = [
        "Sos el agente que arregla a Wawi (asistente de voz de escritorio para macOS, "
        "Python + PyObjC). Trabajás SOLO en este worktree.",
        "",
        f"## Tarea ({kind})",
        f"Título: {scrub(issue.title)}",
        f"Tipo de issue: {issue.kind} · ocurrencias: {issue.count}",
    ]
    if issue.detail:
        out += ["", "Qué pasó:", scrub(issue.detail)[:2000]]
    if excerpt:
        out += ["", "Últimas líneas del log relacionadas (datos personales ya tapados):",
                "```", excerpt[-6000:], "```"]
    if previous_failures:
        out += ["", "Intentos anteriores fallaron por esto; no repitas el error:"]
        out += [f"- {scrub(reason)[:300]}" for reason in previous_failures[-6:]]
    out += ["", "## Cómo"]
    if issue.is_capability:
        out += [
            "- Es una capacidad nueva: armala como un agregado (wawi/plugins.py) o un módulo "
            "nuevo y aislado bajo wawi/, con sus tests. Nada de reescribir el núcleo.",
            "- Si necesita una dependencia, solo una pura-Python y fijada con == en "
            "requirements.txt. Si no hace falta, no agregues ninguna.",
        ]
    else:
        out += [
            "- Primero escribí un test que reproduzca el bug y FALLE. Después arreglalo y "
            "dejá ese test pasando.",
            "- No agregues dependencias.",
        ]
    out += [
        "- Seguí el estilo del código que tocás: código y comentarios en inglés, textos para "
        "el usuario en español rioplatense.",
        f"- Corré la suite entera antes de terminar: `{python} {TEST_COMMAND}`. Tiene que "
        "quedar verde.",
        f"- Cambio chico: como mucho {max_lines} líneas y {max_files} archivos.",
        "- No hagas commit, merge ni push: el commit lo hace Wawi después de revisarlo.",
        "- No toques nunca: " + ", ".join(FORBIDDEN_PATHS) + ", ni reglas de secretos de "
        ".gitignore.",
        "- Nada de tokens, claves, e-mails ni rutas personales (/Users/...) en el código.",
        "- Si para hacerlo bien hace falta algo del dueño (grabar su voz, un permiso de macOS, "
        "instalar algo, un token, elegir entre opciones), no lo hagas: decilo en el reporte.",
        "",
        "## Reporte final (tu último mensaje)",
        "- Una línea `COMMIT: <mensaje de commit en español, una línea>`.",
        "- Dos o tres oraciones de qué cambiaste y por qué.",
        "- Una línea `OWNER_IMPACT: none` si el cambio solo afecta el comportamiento interno de "
        "Wawi; si no, `OWNER_IMPACT: interaction` (cambia cómo le habla o lo interrumpe), "
        "`accounts` (sus cuentas, servicios, agregados, credenciales u otros repos), "
        "`data` (datos que salen de la Mac) o `needs_owner` (necesita algo de él).",
    ]
    return "\n".join(out) + "\n"


def build_review(issue: IssueRecord, diff: str, author_report: str) -> str:
    """The reviewer's prompt: strict, about correctness and regressions only."""
    return "\n".join([
        "Sos el revisor de un cambio que otro agente hizo en Wawi (asistente de voz para macOS, "
        "Python). Juzgá SOLO corrección y regresiones: ¿arregla el problema sin romper nada? "
        "¿Los tests prueban de verdad el arreglo? ¿Hay efectos colaterales, condiciones de "
        "carrera, excepciones no manejadas o código que no corre?",
        "No opines de estilo. Podés leer el repo del worktree; no edites nada.",
        "",
        f"## Issue\n{scrub(issue.title)}\n{scrub(issue.detail)[:1500]}",
        "",
        f"## Reporte del autor\n{scrub(author_report)[-2000:]}",
        "",
        "## Diff",
        "```diff",
        diff[:60_000],
        "```",
        "",
        "Terminá con UNA línea exactamente así: `VERDICT: APPROVE` o `VERDICT: REJECT`, "
        "y antes, las razones en pocas viñetas.",
    ]) + "\n"


_VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVE|REJECT)\b", re.I)


def parse_verdict(text: str) -> tuple[bool, str]:
    """(approved, reasons). Anything but a final explicit APPROVE is a reject."""
    matches = _VERDICT_RE.findall(text or "")
    reasons = _VERDICT_RE.sub("", text or "").strip()[-1500:]
    if not matches:
        return False, reasons or "el revisor no dio veredicto"
    return matches[-1].upper() == "APPROVE", reasons


_COMMIT_RE = re.compile(r"^\s*COMMIT:\s*(.+)$", re.M)


def parse_report(text: str) -> tuple[str, str]:
    """(commit subject, summary) from the author's final message."""
    m = _COMMIT_RE.search(text or "")
    subject = m.group(1).strip()[:100] if m else ""
    summary = re.sub(r"^\s*(COMMIT|OWNER_IMPACT):.*$", "", text or "", flags=re.M).strip()
    return subject, summary[:1200]
