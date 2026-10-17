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
import html
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
_MIND: Any = None
_BOARD: Any = None
_CONNECTORS: Any = None


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


def set_board(board: Any) -> None:
    """Install the visual board. None removes its two voice tools."""
    global _BOARD
    _BOARD = board


def set_connectors(connectors: Any) -> None:
    """Expose enabled borrowed MCP capabilities in the local abilities view."""
    global _CONNECTORS
    _CONNECTORS = connectors


def show_board(title: str, content: str, format: str = "markdown") -> ToolResult:
    if _BOARD is None:
        return ToolResult(False, "La pizarra no está disponible.")
    try:
        page = _BOARD.show(title, content, format)
    except ValueError as exc:
        return ToolResult(False, str(exc), {"error": "invalid_board"})
    return ToolResult(True, "Te lo dejo en la pizarra.", {"title": page.title})


def show_image(query: str, url: str = "", title: str = "") -> ToolResult:
    if _BOARD is None:
        return ToolResult(False, "La pizarra no está disponible.")
    from wawi.images import ImageError, find_image
    try:
        card = find_image(query, url, title)
        _BOARD.show_image(card)
    except ImageError as exc:
        return ToolResult(False, str(exc), {"error": "image_not_found"})
    return ToolResult(True, "Te la dejo en la pizarra.",
                      {"title": card.title, "source": card.source_url,
                       "credit": card.credit, "license": card.license})


def hide_board() -> ToolResult:
    if _BOARD is None:
        return ToolResult(False, "La pizarra no está disponible.")
    _BOARD.hide()
    return ToolResult(True, "Cerré la pizarra.")


def previous_board() -> ToolResult:
    if _BOARD is None:
        return ToolResult(False, "La pizarra no está disponible.")
    page = _BOARD.previous()
    if page is None:
        return ToolResult(False, "No hay una pizarra anterior.")
    return ToolResult(True, "Acá está la pizarra de antes.", {"title": page.title})


def show_abilities() -> ToolResult:
    """Show only tools available on this turn, never a stale capabilities list."""
    if _BOARD is None:
        return ToolResult(False, "La pizarra no está disponible.")
    groups = ability_groups()
    from wawi.pet import api
    scenic = api.window_available()
    if hasattr(_BOARD, "show_abilities"):
        page = _BOARD.show_abilities(groups, scenic=scenic)
        if page.format == "abilities":
            api.trick("ladder")
    else:
        lines = [f"## {heading}\n" + "\n".join(
            f"- {html.escape(description)} — «{html.escape(example)}»" for _, description, example in entries)
            for heading, entries in groups]
        _BOARD.show("Todo lo que sé hacer", "\n\n".join(lines))
    return ToolResult(True, "Mirá todo lo que sé hacer.")


def set_mind(mind: Any) -> None:
    """Install Wawi's mind (wawi/mind/), or None to remove its tools.

    Like the task runner, installing it is what makes :data:`MIND_TOOLS` (and
    the pet / sensors tools, when those modules exist) ACTIVE.
    """
    global _MIND
    _MIND = mind


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
# The mind, the pet and the sensors
# --------------------------------------------------------------------------- #
#
# Same blast radius as the four always-on tools: fetch_page READS a page the
# user named (READ tier -- no approval, like a search; file:// and private
# addresses are refused in wawi/mind/fetch.py), whats_new and the status tools
# read state, and the pet tools only change a cartoon dog's needs.


def _mind_off() -> ToolResult:
    return ToolResult(False, "Eso todavía no lo sé hacer.", {"error": "mind_disabled"})


def fetch_page(url: str, question: str = "") -> ToolResult:
    """"Andá a ver esta página": the spaceship flies there and back."""
    if _MIND is None:
        return _mind_off()
    if not (url or "").strip():
        return ToolResult(False, "Decime qué página.", {"error": "empty_url"})
    ok, speak, data = _MIND.fetch_page(url, question or "")
    return ToolResult(ok, speak, data)


def whats_new() -> ToolResult:
    """"¿Qué hay de nuevo?": feeds and sensors, summarized."""
    if _MIND is None:
        return _mind_off()
    ok, speak, data = _MIND.digest()
    return ToolResult(ok, speak, data)


#: What to say when the body has not learned this yet.
_PET_MISSING = {
    "feed": "Todavía no sé comer.",
    "water": "Todavía no sé tomar agua.",
    "play": "Todavía no sé jugar.",
    "status": "Todavía no sé decirte cómo estoy.",
}

_NEED_WORDS = {"hunger": "hambre", "thirst": "sed", "boredom": "aburrimiento"}


def _pet_speak(value: Any, fallback: str) -> tuple[str, dict | None]:
    """A pet API answer as (spoken sentence, data). str and dict both work."""
    if isinstance(value, str) and value.strip():
        return value.strip(), None
    if isinstance(value, dict):
        for key in ("speak", "message", "text"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key].strip(), value
        return fallback, value
    return fallback, None


def _pet_action(action: str) -> ToolResult:
    from wawi.mind import peers

    try:
        value = peers.call("wawi.pet.api", action)
    except Exception as exc:
        log.warning("pet %s failed: %s", action, exc)
        return ToolResult(False, "Uy, algo me salió mal.", {"error": "pet_failed"})
    if value is peers.MISSING:
        return ToolResult(False, _PET_MISSING[action], {"error": "pet_missing"})
    speak, data = _pet_speak(value, "¡Gracias!")
    return ToolResult(True, speak, data)


def pet_status() -> ToolResult:
    """How the pet is, from wawi.pet.api.status() (a dict or a sentence)."""
    from wawi.mind import peers

    try:
        value = peers.call("wawi.pet.api", "status")
    except Exception as exc:
        log.warning("pet status failed: %s", exc)
        return ToolResult(False, "No sé bien cómo estoy.", {"error": "pet_failed"})
    if value is peers.MISSING:
        return ToolResult(False, _PET_MISSING["status"], {"error": "pet_missing"})
    if isinstance(value, dict) and not any(k in value for k in ("speak", "message", "text")):
        bits = []
        if isinstance(value.get("mood"), str) and value["mood"].strip():
            bits.append(f"estoy {value['mood'].strip()}")
        for key, word in _NEED_WORDS.items():
            level = value.get(key)
            if isinstance(level, (int, float)) and level >= 0.66:
                bits.append(f"tengo mucha {word}")
        energy = value.get("energy")
        if isinstance(energy, (int, float)) and energy < 0.33:
            bits.append("estoy cansada")
        speak = (", ".join(bits) + ".") if bits else "estoy bien."
        return ToolResult(True, speak[:1].upper() + speak[1:], value)
    speak, data = _pet_speak(value, "Estoy bien.")
    return ToolResult(True, speak, data)


# --------------------------------------------------------------------------- #
# Schemas handed to the model
# --------------------------------------------------------------------------- #

TOOLS: list[dict[str, Any]] = [
    {
        "name": "capture_screen",
        "animation": "camera",
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
        "animation": "notebook",
        "description": "Guardá una instrucción del usuario sobre CÓMO responder, para siempre.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "report_issue",
        "animation": "wrench",
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
        "animation": "carry_paper",
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


BOARD_TOOLS: list[dict[str, Any]] = [
    {"name": "show_abilities", "animation": "present", "example": "Wawi, ¿qué sabés hacer?",
     "ability": "Mostrar todo lo que sé hacer",
     "description": "¿Qué podés hacer?/¿qué sabés hacer?: mostrá tus capacidades actuales, agrupadas y con ejemplos; decí solo una frase.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "show_board", "animation": "present",
     "ability": "Mostrar listas, tablas, recetas o código en la pizarra",
     "description": "Mostrá una pizarra si ver una tabla, receta, lista, código o mini app es mejor que oírla; nunca para respuestas triviales. Hablá breve.",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string"}, "content": {"type": "string"},
         "format": {"type": "string", "enum": ["markdown", "html"]}},
         "required": ["title", "content"]}},
    {"name": "show_image", "animation": "ship",
     "ability": "Buscar una foto y mostrarla con su info al lado",
     "example": "Wawi, mostrame una foto de un perro",
     "description": "Buscá una imagen libre y mostrala en la pizarra con descripción, fuente y crédito. Para una página concreta, pasá url; si no, buscá por query.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string"}, "url": {"type": "string"}, "title": {"type": "string"}},
         "required": ["query"]}},
    {"name": "hide_board", "animation": None,
     "description": "Cerrá la pizarra cuando te lo pidan.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "previous_board", "animation": "present",
     "description": "Mostrá la pizarra anterior cuando te la pidan.",
     "input_schema": {"type": "object", "properties": {}}},
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
        "animation": "laptop",
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
        "animation": "laptop",
        "description": "Qué está haciendo el agente ahora. Usalo si te preguntan cómo va.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "answer_task",
        "animation": "laptop",
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
        "animation": "laptop",
        "description": "Cortá el trabajo del agente.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


_EMPTY: dict[str, Any] = {"type": "object", "properties": {}}

#: Active only while a mind is installed with ``mind.tools`` on.
MIND_TOOLS: list[dict[str, Any]] = [
    {
        "name": "fetch_page",
        "animation": "ship",
        "description": (
            "Andá a leer una página web que te nombren («andá a ver esta "
            "página», «fijate qué dice X») y contestá la pregunta. Devuelve "
            "la respuesta para decir."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "question": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "whats_new",
        "animation": "newspaper",
        "description": "«¿Qué hay de nuevo?»: novedades de los feeds y avisos de los sensores.",
        "input_schema": _EMPTY,
    },
]

#: Active only when the body's wawi.pet.api exists.
#: The pet tools need no "working" animation: feeding, water and the ball are
#: acted out by her body (animation: None).
PET_TOOLS: list[dict[str, Any]] = [
    {"name": "pet_feed", "animation": None,
     "description": "Darte de comer (sos la mascota).", "input_schema": _EMPTY},
    {"name": "pet_water", "animation": None, "description": "Darte agua.", "input_schema": _EMPTY},
    {"name": "pet_play", "animation": None, "description": "Jugar con vos.", "input_schema": _EMPTY},
    {"name": "pet_status", "animation": None,
     "description": "Cómo estás: ánimo, hambre, sed.", "input_schema": _EMPTY},
]


def mind_active() -> bool:
    """True when the mind's tools are handed to the model (see prompts.py)."""
    return _MIND is not None and bool(getattr(getattr(_MIND, "settings", None), "tools", True))


def _mind_tools() -> list[dict[str, Any]]:
    """The mind's schemas for this turn: none unless a mind is installed."""
    if not mind_active():
        return []
    from wawi.mind import peers

    out = list(MIND_TOOLS)
    if peers.pet_available():
        out += PET_TOOLS
    return out


def _board_tools() -> list[dict[str, Any]]:
    return [for_model(spec) for spec in BOARD_TOOLS] if _BOARD is not None else []


def active_tools() -> list[dict[str, Any]]:
    """The schemas handed to the model this turn.

    The task schemas appear only once an executor is installed. Every schema
    here is re-sent, and re-paid for, on every single turn — so a capability
    nobody turned on must not be in the list.
    """
    if _TASK_RUNNER is None:
        return [*TOOLS, *_board_tools(), *_sensor_tools(), *_mind_tools(), *_instinct_tools()]
    # Rebuilt per turn rather than at install: the description is a claim about
    # the executor, and the only copy that matters is the one the model reads
    # on the turn it decides whether to promise something.
    tool_names = getattr(getattr(_TASK_RUNNER, "_options", None), "tools", ())
    task_tools = []
    for schema in TASK_TOOLS:
        if schema["name"] == "start_task":
            schema = {**schema, "description": _start_task_description(tool_names)}
        task_tools.append(schema)
    return [*TOOLS, *_board_tools(), *_sensor_tools(), *task_tools, *_mind_tools(), *_instinct_tools()]


_DISPATCH: dict[str, Callable[..., ToolResult]] = {
    "capture_screen": lambda a: capture_screen(str(a.get("note", ""))),
    "remember_preference": lambda a: remember_preference(str(a.get("text", ""))),
    "report_issue": lambda a: report_issue(
        str(a.get("title", "")), str(a.get("detail", ""))
    ),
    "copy_to_clipboard": lambda a: copy_to_clipboard(str(a.get("text", ""))),
    "show_board": lambda a: show_board(str(a.get("title", "")), str(a.get("content", "")), str(a.get("format", "markdown"))),
    "show_image": lambda a: show_image(str(a.get("query", "")), str(a.get("url", "")), str(a.get("title", ""))),
    "hide_board": lambda a: hide_board(),
    "previous_board": lambda a: previous_board(),
    "show_abilities": lambda a: show_abilities(),
    "start_task": lambda a: start_task(str(a.get("goal", ""))),
    "task_status": lambda a: task_status(),
    "answer_task": lambda a: answer_task(str(a.get("text", ""))),
    "stop_task": lambda a: stop_task(),
    "fetch_page": lambda a: fetch_page(str(a.get("url", "")), str(a.get("question", "") or "")),
    "whats_new": lambda a: whats_new(),
    "pet_feed": lambda a: _pet_action("feed"),
    "pet_water": lambda a: _pet_action("water"),
    "pet_play": lambda a: _pet_action("play"),
    "pet_status": lambda a: pet_status(),
}


def dispatch(name: str, args: dict | None = None) -> ToolResult:
    """Route a model tool call to its implementation.

    Unknown names and malformed arguments produce a spoken failure rather than
    an exception: the agent loop has no way to say a traceback.
    """
    handler = _DISPATCH.get(name) or _plugin_handler(name)
    if handler is None:
        log.warning("dispatch: unknown tool %r", name)
        return ToolResult(False, _ES_UNKNOWN_TOOL, {"error": "unknown_tool", "name": name})
    try:
        return handler(args or {})
    except Exception:  # pragma: no cover - defensive
        log.exception("dispatch: tool %r raised", name)
        return ToolResult(False, "Algo falló al ejecutar esa acción.", {"error": "tool_failed"})


# --------------------------------------------------------------------------- #
# Sensor status (wawi/sensors). Self-contained: the schema is only sent while
# sensors are running, so it costs nothing for anyone who never enables them.
# --------------------------------------------------------------------------- #

SENSOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "sensor_status",
        "animation": "detective",
        "description": (
            "Estado de lo que vigilás: webs, disco, batería, servicios, clima. "
            "Usala si preguntan «¿cómo está todo?» o por alguna de esas cosas."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]
_SENSORS_ACTIVE = False


def set_sensors_active(active: bool) -> None:
    """Expose (or hide) :data:`SENSOR_TOOLS`; the orchestrator calls this."""
    global _SENSORS_ACTIVE
    _SENSORS_ACTIVE = bool(active)


def _sensor_tools() -> list[dict[str, Any]]:
    return list(SENSOR_TOOLS) if _SENSORS_ACTIVE else []


def sensor_status() -> ToolResult:
    """Read-only summary of the sensors' last readings. Never does I/O."""
    from wawi import sensors

    text = sensors.status()
    return ToolResult(True, text, {"status": text})


_DISPATCH["sensor_status"] = lambda a: sensor_status()


# --------------------------------------------------------------------------- #
# What she learned from the owner (wawi/learn): «¿qué aprendiste de mí?».
# One schema, sent with the mind's tools (only while a mind is installed),
# answered from what the learner already knows.
# --------------------------------------------------------------------------- #

def what_learned() -> ToolResult:
    from wawi.learn import tool as _learn_tool

    ok, text = _learn_tool.what_learned()
    return ToolResult(ok, text, {"summary": text})


from wawi.learn.tool import SCHEMA as _LEARN_SCHEMA  # noqa: E402

MIND_TOOLS.append(_LEARN_SCHEMA)
_DISPATCH["what_learned"] = lambda a: what_learned()


# --------------------------------------------------------------------------- #
# Configuring the senses by voice (wawi/sensors/live.py). Same exposure as
# sensor_status: sent only while the sensors run.
#
# READ tier on purpose (no approval prompt): these only change WHAT Wawi
# watches — a URL to check every two minutes, a port to knock on, a city for
# the forecast. Every target is validated (http/https only, ports 1-65535,
# plain names) and the change is one sentence away from being undone. The
# web sensor may watch localhost and the LAN: "vigilá mi servidor de
# desarrollo" is a legitimate request, made by the user out loud.
# --------------------------------------------------------------------------- #

SENSOR_TOOLS.append({
    "name": "watch",
    "animation": "binoculars",
    "description": (
        "Cambiar lo que vigilás: add (vigilá X, avisame del clima de X, sumá "
        "este feed), remove (dejá de vigilar X) o list (qué vigilás). "
        "kind: web|feed|port|container|process|weather."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "kind": {"type": "string"},
            "target": {"type": "string"},
        },
        "required": ["action"],
    },
})


def watch(action: str, kind: str = "", target: str = "") -> ToolResult:
    """Add, remove or list what the sensors watch. Live, and persisted."""
    from wawi import sensors

    action = (action or "").strip().lower()
    if action == "list":
        ok, speak, data = sensors.watching()
        return ToolResult(ok, speak, data)
    if action not in ("add", "remove"):
        return ToolResult(False, "¿Querés que vigile algo o que deje de vigilarlo?",
                          {"error": "bad_action"})
    if kind != "weather" and not (target or "").strip():
        return ToolResult(False, "Decime qué querés que vigile.", {"error": "empty_target"})
    if action == "add":
        ok, speak, data = sensors.watch(kind, target)
    else:
        ok, speak, data = sensors.unwatch(kind, target)
    return ToolResult(ok, speak, data)


_DISPATCH["watch"] = lambda a: watch(
    str(a.get("action", "")), str(a.get("kind", "") or ""), str(a.get("target", "") or "")
)


# --------------------------------------------------------------------------- #
# Her instincts (wawi/instinct): «¿qué sabés de mí?», «¿qué patrones viste?»,
# «olvidá lo que aprendiste». One schema, sent only while instincts are
# installed, answered locally from the pattern miner -- no Claude call. The
# forget path wipes local files only (her observations and what she learned).
# --------------------------------------------------------------------------- #

from wawi.instinct.tool import SCHEMA as _INSTINCT_SCHEMA  # noqa: E402

INSTINCT_TOOLS: list[dict[str, Any]] = [_INSTINCT_SCHEMA]


def _instinct_tools() -> list[dict[str, Any]]:
    from wawi import instinct

    return list(INSTINCT_TOOLS) if instinct.get() is not None else []


def about_me(forget: bool = False) -> ToolResult:
    from wawi.instinct import tool as _instinct_tool

    ok, text = _instinct_tool.about_me(forget)
    return ToolResult(ok, text, {"summary": text, "forgot": bool(forget)})


_DISPATCH["about_me"] = lambda a: about_me(bool(a.get("forget", False)))


# --------------------------------------------------------------------------- #
# Games against her (wawi/games). One compact schema for starting, playing a
# move and stopping, sent only while a games manager is installed. Most moves
# never get here: the voice turn hands short game replies ("piedra", "más
# alto", "dale") to the games before the brain sees them (orchestrator).
# --------------------------------------------------------------------------- #

GAME_TOOLS: list[dict[str, Any]] = [
    {
        "name": "play_game",
        "animation": None,
        "description": (
            "Jugar con vos. name: ppt|tateti|numero|adivino|pelota|escondidas|"
            "trivia, o stop para terminar. move: la jugada, si ya juegan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "move": {"type": "string"}},
            "required": ["name"],
        },
    },
]


def _game_tools() -> list[dict[str, Any]]:
    from wawi import games

    return list(GAME_TOOLS) if games.get() is not None else []


def play_game(name: str, move: str = "") -> ToolResult:
    from wawi import games

    ok, text = games.play(name, move)
    return ToolResult(ok, text, {"game": name})


_DISPATCH["play_game"] = lambda a: play_game(str(a.get("name", "")), str(a.get("move", "") or ""))

_base_active_tools = active_tools


def active_tools() -> list[dict[str, Any]]:  # noqa: F811 - extends the one above
    """The schemas for this turn, plus the game's while games are on, as the
    MODEL sees them: without the mascot's ``animation`` field."""
    return [for_model(s) for s in (*_base_active_tools(), *_game_tools())]


# --------------------------------------------------------------------------- #
# Orca (wawi/orca.py): coding tasks for an Orca agent in its own worktree.
# Sent only while a bridge is installed (orca.enabled and a CLI found).
#
# send_to_orca and cancel_orca_task never act by themselves: they leave a
# question pending, and only the owner's NEXT words, read before the brain
# runs (wawi.orca.hear, called by the orchestrator), can say yes. A paid agent
# must never start because the model decided the owner meant it.
# --------------------------------------------------------------------------- #

ORCA_TOOLS: list[dict[str, Any]] = [
    {
        "name": "send_to_orca",
        "description": (
            "Mandá una tarea de código a un agente de Orca, que trabaja en su "
            "propio worktree. Usala cuando el dueño pida «mandalo a Orca», «que "
            "lo haga un agente», «pasáselo a Orca» sobre algo que charlaron. "
            "La transcripción suele escribir Orca como «ahorca», «a orca» u "
            "«orka»: es lo mismo. Si piden mandar issues anotados, leé el "
            "issue y usalo como base del spec. "
            "title: corto. spec: autocontenido, con secciones Target:, Change:, "
            "Constraints:, Ownership: y Acceptance: (cómo se comprueba). repo: "
            "solo si nombran otro repo. NO se manda todavía: decí la pregunta "
            "que devuelve y esperá el sí."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "spec": {"type": "string"},
                "repo": {"type": "string"},
                "agent": {"type": "string"},
            },
            "required": ["title", "spec"],
        },
        "animation": "laptop",
    },
    {
        "name": "orca_status",
        "description": "«¿Cómo van las tareas?», «¿qué hace Orca?»: estado de las tareas en Orca.",
        "input_schema": _EMPTY,
        "animation": "binoculars",
    },
    {
        "name": "answer_orca",
        "description": "Pasale a Orca la respuesta del dueño a la pregunta que hizo un agente.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "animation": "laptop",
    },
    {
        "name": "cancel_orca_task",
        "description": "«Cortá la tarea de Orca»: pide confirmación y la corta. title: cuál.",
        "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
        "animation": "wrench",
    },
]

# Visible handoff in the current checkout; unlike send_to_orca, it creates no Run.
ORCA_TERMINAL_TOOLS: list[dict[str, Any]] = [
    {"name": "open_terminal", "description": "Abrí terminal visible en Orca. Claude/Codex y cualquier prompt requieren un sí hablado; shell vacía no. repo: nombre conocido por Orca.",
     "input_schema": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["claude", "codex", "shell"]}, "repo": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["kind"]}, "animation": "laptop"},
    {"name": "terminal_read", "description": "Leé y resumí las últimas líneas de una terminal abierta por Wawi. which: nombre o handle; omití si hay una sola.",
     "input_schema": {"type": "object", "properties": {"which": {"type": "string"}}}, "animation": "laptop"},
    {"name": "terminal_send", "description": "Escribí texto en una terminal abierta por Wawi; requiere un sí hablado. which: nombre o handle.",
     "input_schema": {"type": "object", "properties": {"which": {"type": "string"}, "text": {"type": "string"}}, "required": ["text"]}, "animation": "laptop"},
    {"name": "list_terminals", "description": "Listá las terminales que abrió Wawi en Orca.",
     "input_schema": _EMPTY, "animation": "laptop"},
]
ORCA_TOOLS.extend(ORCA_TERMINAL_TOOLS)


def _orca() -> Any:
    from wawi import orca

    return orca.get()


def _orca_off() -> ToolResult:
    return ToolResult(False, "No tengo Orca conectada.", {"error": "orca_disabled"})


def _orca_tools() -> list[dict[str, Any]]:
    """The Orca schemas for this turn; answer_orca carries the pending question."""
    bridge = _orca()
    if bridge is None:
        return []
    out = []
    for schema in ORCA_TOOLS:
        if schema["name"] in {"open_terminal", "terminal_read", "terminal_send", "list_terminals"} and not bridge.settings.terminal_handoff_enabled:
            continue
        if schema["name"] == "answer_orca" and bridge.state.questions:
            q = bridge.state.questions[0]
            schema = {**schema, "description": schema["description"] + (
                f" Pregunta pendiente sobre «{q.get('title')}»: {str(q.get('text'))[:200]}"
            )}
        out.append(schema)
    return out


def send_to_orca(title: str, spec: str, repo: str = "", agent: str = "") -> ToolResult:
    bridge = _orca()
    if bridge is None:
        return _orca_off()
    ok, speak, data = bridge.draft(title, spec, repo or None, agent or None)
    return ToolResult(ok, speak, data)


def orca_status() -> ToolResult:
    bridge = _orca()
    if bridge is None:
        return _orca_off()
    speak, data = bridge.status_summary()
    return ToolResult(True, speak, data)


def answer_orca(text: str) -> ToolResult:
    bridge = _orca()
    if bridge is None:
        return _orca_off()
    ok, speak, data = bridge.answer(text)
    return ToolResult(ok, speak, data)


def cancel_orca_task(title: str = "") -> ToolResult:
    bridge = _orca()
    if bridge is None:
        return _orca_off()
    ok, speak, data = bridge.ask_cancel(title)
    return ToolResult(ok, speak, data)


_DISPATCH["send_to_orca"] = lambda a: send_to_orca(
    str(a.get("title", "")), str(a.get("spec", "")),
    str(a.get("repo", "") or ""), str(a.get("agent", "") or ""),
)
_DISPATCH["orca_status"] = lambda a: orca_status()
_DISPATCH["answer_orca"] = lambda a: answer_orca(str(a.get("text", "")))
_DISPATCH["cancel_orca_task"] = lambda a: cancel_orca_task(str(a.get("title", "") or ""))


def _terminal_tool(method: str, *args: str) -> ToolResult:
    bridge = _orca()
    if bridge is None:
        return _orca_off()
    ok, speak, data = getattr(bridge, method)(*args)
    return ToolResult(ok, speak, data)


_DISPATCH["open_terminal"] = lambda a: _terminal_tool(
    "open_terminal", str(a.get("kind", "")), str(a.get("repo", "") or ""), str(a.get("prompt", "") or ""))
_DISPATCH["terminal_read"] = lambda a: _terminal_tool("terminal_read", str(a.get("which", "") or ""))
_DISPATCH["terminal_send"] = lambda a: _terminal_tool(
    "terminal_send", str(a.get("which", "") or ""), str(a.get("text", "")))
_DISPATCH["list_terminals"] = lambda a: _terminal_tool("list_terminals")

_games_active_tools = active_tools


def active_tools() -> list[dict[str, Any]]:  # noqa: F811 - extends the one above
    """The schemas for this turn, plus Orca's while a bridge is installed."""
    return [*_games_active_tools(), *(for_model(s) for s in _orca_tools())]


SHORTCUT_TOOLS: list[dict[str, Any]] = [
    {"name": "list_shortcuts", "description": "Listá los atajos de macOS disponibles.",
     "input_schema": _EMPTY, "animation": "binoculars"},
    {"name": "run_shortcut", "description": "Corré un atajo de macOS por nombre; input es texto opcional. Si pregunta confirmación, esperá el próximo sí del dueño.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "input": {"type": "string"}}, "required": ["name"]},
     "animation": "busy"},
]


def _shortcut_bridge() -> Any:
    from wawi import shortcuts
    return shortcuts.get()


def _shortcut_result(method: str, *args: str) -> ToolResult:
    bridge = _shortcut_bridge()
    if bridge is None:
        return ToolResult(False, "No tengo Atajos disponible en esta Mac.", {"error": "shortcuts_disabled"})
    return ToolResult(*getattr(bridge, method)(*args))


_DISPATCH["list_shortcuts"] = lambda a: _shortcut_result("list")
_DISPATCH["run_shortcut"] = lambda a: _shortcut_result(
    "run", str(a.get("name", "")), str(a.get("input", "") or ""))

_shortcuts_base_active_tools = active_tools


def active_tools() -> list[dict[str, Any]]:  # noqa: F811 - extends the one above
    """Expose Shortcuts only when its bridge is installed."""
    base = _shortcuts_base_active_tools()
    if _shortcut_bridge() is None:
        return base
    return [*base, *(for_model(s) for s in SHORTCUT_TOOLS)]


# --------------------------------------------------------------------------- #
# How she looks while a tool runs (wawi/working.py, wawi/mascot.WORK_CLIPS).
# Each tool declares it where it is defined: an optional "animation" field in
# its spec (a WORK_CLIPS name), None for "deliberately none" (the pet and game
# tools animate themselves). It never reaches the model (for_model). A tool
# or skill added later brings its own the same way, or register_animation().
# --------------------------------------------------------------------------- #

#: Fields of a spec that are ours, not the model's.
_LOCAL_FIELDS = ("animation", "example", "ability")
#: Animations registered at runtime (a connector by server name, Orca…).
_EXTRA_ANIMATIONS: dict[str, str | None] = {}


def for_model(spec: dict[str, Any]) -> dict[str, Any]:
    """*spec* without the fields the model must never see."""
    if not any(k in spec for k in _LOCAL_FIELDS):
        return spec
    return {k: v for k, v in spec.items() if k not in _LOCAL_FIELDS}


def all_specs() -> list[dict[str, Any]]:
    """Every tool spec this module knows, active or not."""
    return [*TOOLS, *BOARD_TOOLS, *TASK_TOOLS, *MIND_TOOLS, *PET_TOOLS, *SENSOR_TOOLS,
            *INSTINCT_TOOLS, *GAME_TOOLS, *ORCA_TOOLS, *SHORTCUT_TOOLS]


def register_animation(tool: str, animation: str | None) -> None:
    """Give a tool defined elsewhere (a connector, Orca's send_to_orca) its
    animation. Any thread; last one wins."""
    _EXTRA_ANIMATIONS[str(tool)] = animation


def animation_for(tool: str) -> str | None:
    """The animation *tool* declared; None when it has deliberately none or
    is unknown (the mascot then shows its generic "busy" clip, see
    wawi/working.py — except for "deliberately none", see has_no_animation)."""
    if tool in _EXTRA_ANIMATIONS:
        return _EXTRA_ANIMATIONS[tool]
    for spec in all_specs():
        if spec.get("name") == tool:
            value = spec.get("animation")
            return str(value) if value else None
    return None


def has_no_animation(tool: str) -> bool:
    """*tool* declares ``"animation": None`` on purpose: show nothing extra."""
    if tool in _EXTRA_ANIMATIONS:
        return _EXTRA_ANIMATIONS[tool] is None
    for spec in all_specs():
        if spec.get("name") == tool:
            return "animation" in spec and spec["animation"] is None
    return False

# Local reminders and owner-approved recurring instructions. The names and
# descriptions are intentionally short: every active schema is sent per turn.
REMINDER_TOOLS: list[dict[str, Any]] = [
    {"name": "set_timer", "animation": "notebook", "description": "Temporizador en segundos o minutos; label opcional.",
     "input_schema": {"type": "object", "properties": {"seconds": {"type": "integer"}, "minutes": {"type": "integer"}, "label": {"type": "string"}}}},
    {"name": "remind_me", "animation": "notebook", "description": "Recordame algo: when en español, p. ej. «en 20 minutos», «a las 18», «mañana a las 9».",
     "input_schema": {"type": "object", "properties": {"when": {"type": "string"}, "text": {"type": "string"}}, "required": ["when", "text"]}},
    {"name": "list_reminders", "animation": "notebook", "description": "Listar recordatorios y temporizadores activos.", "input_schema": _EMPTY},
    {"name": "cancel_reminder", "animation": "notebook", "description": "Cancelar por id o texto exacto.",
     "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "add_routine", "animation": "notebook", "description": "Rutina diaria o semanal: schedule «todas las mañanas a las 9» o «los viernes a las 18». instruction: qué hacer. Pide sí al dueño; no la ejecutes todavía.",
     "input_schema": {"type": "object", "properties": {"schedule": {"type": "string"}, "instruction": {"type": "string"}}, "required": ["schedule", "instruction"]}},
    {"name": "list_routines", "animation": "notebook", "description": "Listar rutinas programadas.", "input_schema": _EMPTY},
    {"name": "remove_routine", "animation": "notebook", "description": "Quitar rutina por id o instrucción exacta.",
     "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]


def _reminder_action(name: str, args: dict) -> ToolResult:
    from wawi import reminders

    service = reminders.get()
    if service is None:
        return ToolResult(False, "Los recordatorios están desactivados.", {"error": "reminders_disabled"})
    try:
        if name == "set_timer":
            item = service.set_timer(seconds=args.get("seconds", 0), minutes=args.get("minutes", 0), label=str(args.get("label", "")))
            return ToolResult(True, "Listo, puse el temporizador.", item)
        if name == "remind_me":
            item = service.remind_me(str(args.get("when", "")), str(args.get("text", "")))
            return ToolResult(True, "Listo, te lo recuerdo.", item)
        if name in ("list_reminders", "list_routines"):
            kind = "reminder" if name == "list_reminders" else "routine"
            items = service.list_items(kind)
            return ToolResult(True, f"Tenés {len(items)} {'recordatorios' if kind == 'reminder' else 'rutinas'}.", {"items": items})
        if name in ("cancel_reminder", "remove_routine"):
            kind = "reminder" if name == "cancel_reminder" else "routine"
            removed = service.remove(str(args.get("id", "")), kind)
            return ToolResult(removed, "Listo, lo quité." if removed else "No encontré uno solo con ese nombre o id.")
        draft = service.add_routine(str(args.get("schedule", "")), str(args.get("instruction", "")))
        return ToolResult(True, f"¿Programo la rutina «{draft['schedule']}» para {draft['text']}?", {**draft, "note": "Esperá el sí del dueño en su próxima frase; no llames otra herramienta."})
    except (ValueError, TypeError) as exc:
        return ToolResult(False, str(exc), {"error": "invalid_reminder"})


for _name in ("set_timer", "remind_me", "list_reminders", "cancel_reminder",
              "add_routine", "list_routines", "remove_routine"):
    _DISPATCH[_name] = lambda args, name=_name: _reminder_action(name, args)

_previous_active_tools = active_tools
_previous_all_specs = all_specs


def active_tools() -> list[dict[str, Any]]:  # noqa: F811
    from wawi import reminders

    return [*_previous_active_tools(), *(for_model(s) for s in REMINDER_TOOLS)] if reminders.get() else _previous_active_tools()


def all_specs() -> list[dict[str, Any]]:  # noqa: F811
    return [*_previous_all_specs(), *REMINDER_TOOLS]


# Local display metadata never leaves for_model(). New active tools without an
# example still appear in the scroll with a description-derived voice prompt.
_ABILITY_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Charlar y pensar", ("capture_screen", "copy_to_clipboard", "report_issue")),
    ("Mostrar en la pizarra", ("show_abilities", "show_board", "hide_board", "previous_board")),
    ("Recordatorios y rutinas", ("set_timer", "remind_me", "list_reminders", "cancel_reminder",
                              "add_routine", "list_routines", "remove_routine")),
    ("Atajos", ("list_shortcuts", "run_shortcut")),
    ("Mirar páginas y novedades", ("fetch_page", "whats_new")),
    ("Vigilar servicios y clima", ("sensor_status", "watch")),
    ("Orca y terminales", ("send_to_orca", "orca_status", "answer_orca", "cancel_orca_task",
                          "open_terminal", "terminal_read", "terminal_send", "list_terminals",
                          "start_task", "task_status", "answer_task", "stop_task")),
    ("Juegos", ("play_game",)),
    ("Mimos, comida y agua", ("pet_feed", "pet_water", "pet_play", "pet_status")),
    ("Memoria y Obsidian", ("remember_preference", "what_learned", "about_me")),
)

_ABILITY_EXAMPLES: dict[str, str] = {
    "capture_screen": "Wawi, sacá una captura de pantalla",
    "copy_to_clipboard": "Wawi, copiá eso",
    "report_issue": "Wawi, anotá este problema",
    "show_abilities": "Wawi, ¿qué sabés hacer?",
    "show_board": "Wawi, mostrame esta lista en la pizarra",
    "hide_board": "Wawi, cerrá la pizarra",
    "previous_board": "Wawi, mostrame la pizarra anterior",
    "set_timer": "Wawi, poné un timer de diez minutos",
    "remind_me": "Wawi, recordame llamar a Ana mañana",
    "list_reminders": "Wawi, ¿qué recordatorios tengo?",
    "cancel_reminder": "Wawi, cancelá el recordatorio de Ana",
    "add_routine": "Wawi, todos los días a las nueve recordame estirar",
    "list_routines": "Wawi, ¿qué rutinas tengo?",
    "remove_routine": "Wawi, quitá la rutina de estirar",
    "list_shortcuts": "Wawi, ¿qué atajos tengo?",
    "run_shortcut": "Wawi, corré el atajo Modo trabajo",
    "fetch_page": "Wawi, mirá esta página y contame qué dice",
    "whats_new": "Wawi, ¿qué hay de nuevo?",
    "sensor_status": "Wawi, ¿cómo están mis servicios y el clima?",
    "watch": "Wawi, vigilá esta página",
    "send_to_orca": "Wawi, mandá esta tarea a Orca",
    "orca_status": "Wawi, ¿cómo van las tareas de Orca?",
    "answer_orca": "Wawi, decile a Orca que sí",
    "cancel_orca_task": "Wawi, cancelá la tarea de Orca",
    "open_terminal": "Wawi, abrí una terminal en Orca",
    "terminal_read": "Wawi, leé esa terminal",
    "terminal_send": "Wawi, escribí esto en la terminal",
    "list_terminals": "Wawi, ¿qué terminales abriste?",
    "start_task": "Wawi, hacé esta tarea",
    "task_status": "Wawi, ¿cómo va la tarea?",
    "answer_task": "Wawi, respondé que sí a la tarea",
    "stop_task": "Wawi, detené la tarea",
    "play_game": "Wawi, juguemos al tatetí",
    "pet_feed": "Wawi, tomá comida",
    "pet_water": "Wawi, tomá agua",
    "pet_play": "Wawi, juguemos con la pelota",
    "pet_status": "Wawi, ¿cómo estás?",
    "remember_preference": "Wawi, recordá que prefiero respuestas cortas",
    "what_learned": "Wawi, ¿qué aprendiste de mí?",
    "about_me": "Wawi, ¿qué sabés de mí?",
}

_ABILITY_LABELS = {
    "remember_preference": "Recordar cómo preferís que responda",
    "about_me": "Contarte qué patrones aprendí de vos",
    "pet_feed": "Recibir comida",
    "pet_water": "Tomar agua",
    "pet_play": "Jugar con vos",
    "play_game": "Jugar al tatetí, trivia, adivinanzas y más",
}

def ability_groups() -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Group the live model registry, including newly installed tool specs."""
    active = {spec["name"]: spec for spec in active_tools()}
    local = {spec["name"]: spec for spec in all_specs()}
    seen: set[str] = set()

    def entry(name: str) -> tuple[str, str, str]:
        spec = local.get(name, active[name])
        description = str(spec.get("ability") or _ABILITY_LABELS.get(name)
                          or spec.get("description") or name).split(". ")[0].split("; ")[0].strip(" .")
        description = description[:90] + ("…" if len(description) > 90 else "")
        example = str(spec.get("example") or _ABILITY_EXAMPLES.get(name)
                      or f"Wawi, {description[:70].lower()}").strip()
        return name, description, example

    groups: list[tuple[str, list[tuple[str, str, str]]]] = []
    for heading, names in _ABILITY_SECTIONS:
        entries = [entry(name) for name in names if name in active]
        if entries:
            groups.append((heading, entries))
            seen.update(name for name, _, _ in entries)
    from wawi import plugins

    services = [entry(name) for name in active if plugins.is_plugin_tool(name)]
    seen.update(name for name, _, _ in services)
    manager = plugins.get()
    if manager is not None:
        services += manager.ability_entries()
    if services:
        groups.append(("Tus servicios (agregados)", services))
    extra = [entry(name) for name in active if name not in seen]
    if _CONNECTORS:
        borrowed: dict[str, list[tuple[str, str, str]]] = {}
        for name in _CONNECTORS.allowed_tools():
            parts = name.split("__", 2)
            if len(parts) < 2 or parts[0] != "mcp" or parts[1].startswith(plugins.MCP_PREFIX):
                continue
            server = parts[1]
            action = parts[2] if len(parts) > 2 else "consult"
            label = action.replace(server + "_", "").replace("_", " ").capitalize()
            borrowed.setdefault(server, []).append((name, f"{label} en {server.title()}",
                                                      f"Wawi, buscá esto en {server.title()}"))
        for server, entries in sorted(borrowed.items()):
            if server in ("obsidian", "engram"):
                index = next((i for i, (heading, _) in enumerate(groups)
                              if heading == "Memoria y Obsidian"), None)
                if index is not None:
                    groups[index][1].extend(entries)
                    continue
            groups.append((f"Conector {server.title()}", entries))
    if extra:
        groups.append(("Otras cosas", extra))
    return groups


# --------------------------------------------------------------------------- #
# Agregados (wawi/plugins.py): the owner's own services. ``my_services`` while
# any exists, plus ONE ``svc_<id>`` per enabled REST plugin. MCP plugins are
# MCP servers, not schemas here (wawi/connectors.py gates them).
# --------------------------------------------------------------------------- #


def _plugin_specs() -> list[dict[str, Any]]:
    from wawi import plugins

    manager = plugins.get()
    if manager is None:
        return []
    try:
        return manager.tool_specs()
    except Exception:
        log.warning("plugins: could not build their tools", exc_info=True)
        return []


def _plugin_handler(name: str) -> Callable[[dict], ToolResult] | None:
    from wawi import plugins

    manager = plugins.get()
    if manager is None or not plugins.is_plugin_tool(name):
        return None
    return lambda args: ToolResult(*manager.dispatch(name, args))


_plugins_previous_active_tools = active_tools
_plugins_previous_all_specs = all_specs


def active_tools() -> list[dict[str, Any]]:  # noqa: F811 - extends the one above
    return [*_plugins_previous_active_tools(), *(for_model(s) for s in _plugin_specs())]


def all_specs() -> list[dict[str, Any]]:  # noqa: F811 - extends the one above
    return [*_plugins_previous_all_specs(), *_plugin_specs()]
