"""Tool layer exposed to the model.

Every function here answers one spoken request and returns a
:class:`ToolResult` carrying two very different payloads:

* ``speak`` — a SHORT sentence, already fit for TTS.
* ``data`` — the structured detail the model may summarize itself. It is never
  spoken verbatim.

Two rules shape this module:

1. **Cost.** ``brain.py`` runs the SDK in lean mode (no CLAUDE.md, no MCP, no
   built-in tool schemas) so a turn costs about a cent. Every tool and every
   word of every description is re-sent on every turn, so :data:`TOOLS` is
   deliberately four terse schemas and nothing more.
2. **Blast radius.** The four always-on tools cannot change code, run a
   command, or type into anything: they write a screenshot, a preference line,
   an issue file and the clipboard, all recoverable with an editor or a second
   copy. Wawi is something you
   ASK, and the surface it can act on is kept small enough that a
   mistranscription cannot cost you anything.

   The four ``TASK_TOOLS`` are the deliberate exception, and they are off by
   default. They do not act themselves — they drive a separate executor agent
   (:mod:`wawi.task`) that does, and every command that agent wants to run is
   read out loud for approval first. That approval is what replaces the small
   blast radius; it is not an addition to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

MAX_SPEAK_CHARS = 240
"""Hard cap on spoken text. Feeding a long answer to TTS talks for minutes."""

MAX_DATA_CHARS = 4000
"""Cap on any single free-text blob returned in ``data``. Bounds token cost."""


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool call.

    Attributes:
        ok: False when nothing was accomplished. The model should relay
            ``speak`` and stop.
        speak: Short sentence, ready for TTS. Clipped to
            :data:`MAX_SPEAK_CHARS` on construction — the invariant holds even
            if a caller passes something long.
        data: Structured detail for the model, or None.
    """

    ok: bool
    speak: str
    data: dict | None = None

    def __post_init__(self) -> None:
        clipped = _clip(self.speak, MAX_SPEAK_CHARS)
        if clipped != self.speak:
            object.__setattr__(self, "speak", clipped)


def _clip(text: str, limit: int) -> str:
    """Return *text* trimmed to *limit* characters, with an ellipsis when cut."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


_ES_UNKNOWN_TOOL = "No conozco esa acción."


# --------------------------------------------------------------------------- #
# Installed collaborators
# --------------------------------------------------------------------------- #

_SHOTS_FALLBACK_DIR = "screenshots"
_SCREEN_MONITOR = 1
_PREFERENCES: Any = None
_JOURNAL: Any = None
_TASK_RUNNER: Any = None


def set_screen_options(directory: str, monitor: int) -> None:
    """Point screenshots at a directory and a monitor (from config)."""
    global _SHOTS_FALLBACK_DIR, _SCREEN_MONITOR
    _SHOTS_FALLBACK_DIR = directory or "screenshots"
    _SCREEN_MONITOR = int(monitor)


def set_preferences(prefs: Any) -> None:
    """Install the store the remember_preference tool writes to."""
    global _PREFERENCES
    _PREFERENCES = prefs


def set_journal(journal: Any) -> None:
    """Install the issue journal the report_issue tool writes to."""
    global _JOURNAL
    _JOURNAL = journal


def set_task_runner(runner: Any) -> None:
    """Install the executor the four task tools drive, or None to remove them.

    Installing it is what makes those four schemas ACTIVE — see
    :func:`active_tools`. Nothing is paid for them until a user turns
    ``tasks.enabled`` on.
    """
    global _TASK_RUNNER
    _TASK_RUNNER = runner


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def remember_preference(text: str) -> ToolResult:
    """Store a standing instruction about HOW to answer, from the user.

    This is self-improvement that changes behaviour without changing code: the
    preference lands in the system prompt of every later turn.
    """
    if _PREFERENCES is None or not getattr(_PREFERENCES, "enabled", False):
        return ToolResult(
            False,
            "No tengo dónde guardarlo: las preferencias están apagadas.",
            {"error": "preferences_disabled"},
        )
    pref = _PREFERENCES.add(text)
    if pref is None:
        return ToolResult(False, "Decime qué querés que recuerde.", {"error": "empty"})
    return ToolResult(True, "Listo, lo voy a tener en cuenta.", {"text": pref.text})


def capture_screen(note: str = "") -> ToolResult:
    """Screenshot the screen and save it as a file.

    Distinct from Wawi LOOKING at the screen to answer you — that is the
    screen tool in :mod:`wawi.brain`, which sends the image inline to the
    model and is gone at the end of the turn. This one leaves a file behind,
    which is what you want when the screenshot itself is the deliverable.
    """
    from pathlib import Path

    from wawi import screen

    try:
        img = screen.capture(_SCREEN_MONITOR)
        path = screen.save_jpeg(img, Path(_SHOTS_FALLBACK_DIR))
    except Exception as exc:
        log.warning("capture_screen failed: %s", exc)
        return ToolResult(
            False,
            "No pude capturar la pantalla. Puede ser el permiso de grabación de pantalla.",
            {"error": "capture_failed", "detail": str(exc)[:200]},
        )

    return ToolResult(
        True,
        "Listo, saqué la captura.",
        {"path": str(path), "note": note.strip()},
    )


def report_issue(title: str, detail: str = "") -> ToolResult:
    """File a problem the USER reported. Writes an issue; never touches code.

    This is the signal telemetry cannot produce: a turn that succeeded
    mechanically and was still wrong. Without it there is nothing to improve
    against.
    """
    if _JOURNAL is None or not getattr(_JOURNAL, "enabled", False):
        return ToolResult(
            False,
            "No tengo dónde anotarlo: el registro de problemas está apagado.",
            {"error": "journal_disabled"},
        )
    title = (title or "").strip()
    if not title:
        return ToolResult(False, "Decime qué anoto.", {"error": "empty_title"})

    from wawi.selfimprove import KIND_USER_REPORTED

    issue = _JOURNAL.record(
        KIND_USER_REPORTED,
        title,
        detail=detail,
        subject=title,
        context={"source": "user"},
    )
    count = getattr(issue, "count", 1)
    speak = "Anotado." if count == 1 else f"Anotado, ya van {count} veces."
    return ToolResult(True, speak, {"title": title, "count": count})


# --------------------------------------------------------------------------- #
# Delegation — the conversational half of the two-agent split
# --------------------------------------------------------------------------- #
#
# These four are how Wawi stays in charge of an executor that is doing real
# work (see wawi/task.py). Every one of them returns immediately: the model
# calling them is holding the turn lock, and a turn that waits on the executor
# is a microphone that stopped working.
#
# They also break the "nothing here can run a command" rule at the top of this
# module — indirectly, which is the only way it can be broken. start_task hands
# a goal to an agent that CAN run commands, and every one of those commands is
# read out loud for approval before it runs. That approval is the replacement
# for the small blast radius, not an addition to it.


#: pbcopy decodes stdin with the locale's encoding. Started from launchd or a
#: bare background shell there is often NO locale, and then the accents in
#: "acción" can land on the clipboard mangled. Forcing UTF-8 is the whole fix.
_PBCOPY_ENV_OVERRIDES = {"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}
_PBCOPY_TIMEOUT_S = 5.0


def copy_to_clipboard(text: str) -> ToolResult:
    """Put *text* on the macOS clipboard, verbatim.

    The voice path to what the panel's "copiar" pill does: "copiá eso" after
    an answer, a command or a task result. Harmless by construction — it only
    replaces the clipboard — so it is pre-approved like every tool here and
    never asks.

    pbcopy rather than NSPasteboard: tools run on the brain's worker thread,
    and AppKit objects belong to the main thread. A child process has no
    thread to get wrong. No shell, and the text goes in on stdin, so nothing
    in it is ever interpreted.
    """
    import os
    import subprocess

    # Not stripped beyond the edges: a command's internal spacing and line
    # breaks are exactly what the user wants to paste.
    text = (text or "").strip("\n\r ")
    if not text:
        return ToolResult(False, "Decime qué querés que copie.", {"error": "empty"})
    try:
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            env={**os.environ, **_PBCOPY_ENV_OVERRIDES},
            timeout=_PBCOPY_TIMEOUT_S,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("copy_to_clipboard failed: %s", exc)
        return ToolResult(
            False,
            "No pude copiarlo al portapapeles.",
            {"error": "copy_failed", "detail": str(exc)[:200]},
        )
    log.info("Copied %d chars to the clipboard by voice", len(text))
    return ToolResult(True, "Listo, copiado.", {"chars": len(text)})


def _no_runner() -> ToolResult:
    return ToolResult(
        False,
        "No puedo delegar trabajo: las tareas están apagadas.",
        {"error": "tasks_disabled"},
    )


def start_task(goal: str) -> ToolResult:
    """Hand a goal to the executor agent and return at once."""
    if _TASK_RUNNER is None:
        return _no_runner()
    goal = (goal or "").strip()
    if not goal:
        return ToolResult(False, "Decime qué querés que haga.", {"error": "empty_goal"})
    if _TASK_RUNNER.is_running:
        current = _TASK_RUNNER.current
        return ToolResult(
            False,
            "Ya hay una tarea en curso. Cortala primero si querés cambiar.",
            {
                "error": "already_running",
                "id": getattr(current, "id", ""),
                "goal": getattr(current, "goal", ""),
            },
        )
    task = _TASK_RUNNER.start(goal)
    if task is None:  # pragma: no cover - lost a race with another turn
        return ToolResult(False, "No pude arrancarla.", {"error": "not_started"})
    return ToolResult(True, "Arranqué con eso.", {"id": task.id, "goal": task.goal})


def task_status() -> ToolResult:
    """Report what the executor is doing, so "¿cómo va?" is answerable.

    Reads state the executor already published; it never wakes it. That is the
    whole reason the mailbox holds progress instead of the executor pushing it
    at the user: asking must be free.
    """
    if _TASK_RUNNER is None:
        return _no_runner()
    status = _TASK_RUNNER.status()
    if status.get("task", True) is None or not status.get("id"):
        return ToolResult(True, "No hay ninguna tarea corriendo.", {"running": False})
    if status.get("waiting_for_you"):
        speak = "Está esperando que le contestes."
    elif status["running"]:
        speak = status.get("progress", [""])[-1] if status.get("progress") else "Trabajando."
    else:
        speak = "Ya terminó."
    return ToolResult(True, speak, status)


def answer_task(text: str) -> ToolResult:
    """Pass the user's reply to the executor waiting on it.

    This is the seam that lets the microphone stay on the conversational path:
    the answer arrives as an ordinary turn and the model routes it here.
    """
    if _TASK_RUNNER is None:
        return _no_runner()
    if not (text or "").strip():
        return ToolResult(False, "No entendí qué contestarle.", {"error": "empty"})
    if not _TASK_RUNNER.mailbox.answer(text):
        return ToolResult(
            False,
            "No hay ninguna pregunta esperando respuesta.",
            {"error": "nothing_pending"},
        )
    return ToolResult(True, "Se lo paso.", {"answer": text.strip()})


def stop_task() -> ToolResult:
    """Cancel the executor. Also releases anything blocked on an answer."""
    if _TASK_RUNNER is None:
        return _no_runner()
    if not _TASK_RUNNER.stop():
        return ToolResult(False, "No hay nada corriendo.", {"error": "nothing_running"})
    return ToolResult(True, "Listo, la corté.", {"stopped": True})


# --------------------------------------------------------------------------- #
# Schemas handed to the model
# --------------------------------------------------------------------------- #

TOOLS: list[dict[str, Any]] = [
    {
        "name": "capture_screen",
        "description": (
            "Sacá una foto de la pantalla y guardala como archivo. Usala cuando "
            "te pidan capturar/sacar una captura, screenshot o pantallazo. "
            "Devuelve el path del archivo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
        },
    },
    {
        "name": "remember_preference",
        "description": "Guardá una instrucción del usuario sobre CÓMO responder, para siempre.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "report_issue",
        "description": "Anotá un problema o sugerencia del usuario. Solo escribe un issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "detail": {"type": "string"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "copy_to_clipboard",
        # The verbatim rule lives HERE, not in the style rules: those tell the
        # model to talk without paths, URLs or markdown, and a command copied
        # under those rules would arrive paraphrased and useless.
        "description": (
            "Copiá texto al portapapeles. Usala si piden copiar algo («copiá "
            "eso», «pasámelo al portapapeles»). Pasá el texto EXACTO (tu última "
            "respuesta, un comando, el resultado de una tarea), sin adornos "
            "hablados; de código o comandos, solo el comando, tal cual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


#: Kept OUT of :data:`TOOLS` so they cost nothing until someone opts in. Four
#: more schemas on every turn of every user is a real bill, and most users will
#: never delegate anything.
_START_TASK_DESCRIPTION = (
    "Delegá un trabajo largo a un agente ejecutor. Volvés enseguida; el agente "
    "avisa cuando termina. Pasale el objetivo completo, con todo el contexto."
)
"""Placeholder. The live text is built by :func:`_start_task_description`."""

#: What each executor tool actually lets it do, said the way the conversational
#: model needs to hear it: in terms of what it may PROMISE the user.
_TOOL_POWERS = {
    "Read": "leer archivos",
    "Glob": "buscar archivos por nombre",
    "Grep": "buscar texto dentro de archivos",
    "Write": "crear archivos",
    "Edit": "modificar archivos",
    "Bash": "correr comandos del sistema",
}


def _start_task_description(tool_names: "tuple[str, ...] | list[str]") -> str:
    """Describe the executor that is installed RIGHT NOW.

    The list comes from config, so it changes per install. Naming what the
    executor CANNOT do matters more than naming what it can: the failure this
    prevents is the assistant agreeing out loud to an action, delegating it,
    and reporting success for work that was never possible.
    """
    powers = [_TOOL_POWERS[name] for name in tool_names if name in _TOOL_POWERS]
    can = ", ".join(powers) if powers else "nada"
    line = (
        "Delegá un trabajo largo a un agente ejecutor. Volvés enseguida; el "
        "agente avisa cuando termina. Pasale el objetivo completo, con todo el "
        f"contexto que tengas. El agente SOLO puede: {can}."
    )
    if "Bash" not in tool_names:
        line += (
            " NO puede correr comandos ni abrir aplicaciones: si te piden algo "
            "así, decilo en vez de delegarlo."
        )
    if "Write" not in tool_names and "Edit" not in tool_names:
        line += " NO puede modificar ni crear archivos."
    return line


TASK_TOOLS: list[dict[str, Any]] = [
    {
        "name": "start_task",
        # The capability sentence is filled in per turn by active_tools() from
        # the executor that is actually installed. Hardcoding "puede leer,
        # escribir y correr comandos" made this tool LIE: with a read-only
        # tools list the model still promised to open an application, said
        # "listo, lo estoy abriendo", and the task reported DONE having done
        # nothing. A tool description is a capability claim; a false one is a
        # bug with a voice.
        "description": _START_TASK_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {"goal": {"type": "string"}},
            "required": ["goal"],
        },
    },
    {
        "name": "task_status",
        "description": "Qué está haciendo el agente ahora. Usalo si te preguntan cómo va.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "answer_task",
        "description": (
            "Pasale al agente la respuesta del usuario a la pregunta que hizo. "
            "Usalo cuando el usuario esté contestando algo que el agente preguntó."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "stop_task",
        "description": "Cortá el trabajo del agente.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def active_tools() -> list[dict[str, Any]]:
    """The schemas handed to the model this turn.

    The task schemas appear only once an executor is installed. Every schema
    here is re-sent, and re-paid for, on every single turn — so a capability
    nobody turned on must not be in the list.
    """
    if _TASK_RUNNER is None:
        return list(TOOLS)
    # Rebuilt per turn rather than at install: the description is a claim about
    # the executor, and the only copy that matters is the one the model reads
    # on the turn it decides whether to promise something.
    tool_names = getattr(getattr(_TASK_RUNNER, "_options", None), "tools", ())
    task_tools = []
    for schema in TASK_TOOLS:
        if schema["name"] == "start_task":
            schema = {**schema, "description": _start_task_description(tool_names)}
        task_tools.append(schema)
    return [*TOOLS, *task_tools]


_DISPATCH: dict[str, Callable[..., ToolResult]] = {
    "capture_screen": lambda a: capture_screen(str(a.get("note", ""))),
    "remember_preference": lambda a: remember_preference(str(a.get("text", ""))),
    "report_issue": lambda a: report_issue(
        str(a.get("title", "")), str(a.get("detail", ""))
    ),
    "copy_to_clipboard": lambda a: copy_to_clipboard(str(a.get("text", ""))),
    "start_task": lambda a: start_task(str(a.get("goal", ""))),
    "task_status": lambda a: task_status(),
    "answer_task": lambda a: answer_task(str(a.get("text", ""))),
    "stop_task": lambda a: stop_task(),
}


def dispatch(name: str, args: dict | None = None) -> ToolResult:
    """Route a model tool call to its implementation.

    Unknown names and malformed arguments produce a spoken failure rather than
    an exception: the agent loop has no way to say a traceback.
    """
    handler = _DISPATCH.get(name)
    if handler is None:
        log.warning("dispatch: unknown tool %r", name)
        return ToolResult(False, _ES_UNKNOWN_TOOL, {"error": "unknown_tool", "name": name})
    try:
        return handler(args or {})
    except Exception:  # pragma: no cover - defensive
        log.exception("dispatch: tool %r raised", name)
        return ToolResult(False, "Algo falló al ejecutar esa acción.", {"error": "tool_failed"})
