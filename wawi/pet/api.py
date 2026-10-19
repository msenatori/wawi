"""The thread-safe door to Wawi's body, for the voice brain and the menus.

``feed()``, ``water()`` and ``play()`` may be called from ANY thread: they
only enqueue a command, which the body takes on the main thread at its next
frame (AppKit and the animator live there). ``status()`` (a short Spanish
sentence) and ``needs()`` (numbers) read her needs under their lock, from any
thread too.

With the pet not running (no panel, ``pet.enabled: false``) commands are
dropped — they return False — and ``status()`` says so instead of guessing.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable

#: Commands waiting for the main thread.
_commands: "queue.SimpleQueue[str]" = queue.SimpleQueue()
_lock = threading.Lock()
#: Set by the running body (wawi/pet/life.py).
_status_source: Callable[[], str] | None = None
_levels_source: Callable[[], dict[str, Any]] | None = None
_window_source: Callable[[], bool] | None = None
_at_window_source: Callable[[], bool] | None = None

COMMANDS = ("feed", "water", "play")


def attach(status_source: Callable[[], str] | None,
           levels_source: Callable[[], dict[str, Any]] | None = None,
           window_source: Callable[[], bool] | None = None,
           at_window_source: Callable[[], bool] | None = None) -> None:
    """The body is alive (or gone, with None). Called by the body itself."""
    global _status_source, _levels_source, _window_source, _at_window_source
    with _lock:
        _status_source = status_source
        _levels_source = levels_source if status_source is not None else None
        _window_source = window_source if status_source is not None else None
        _at_window_source = at_window_source if status_source is not None else None
    if status_source is None:
        drain()


def running() -> bool:
    with _lock:
        return _status_source is not None


def window_available() -> bool:
    """Whether a reachable real window can host Wawi's ladder right now."""
    with _lock:
        source = _window_source
    try:
        return bool(source and source())
    except Exception:
        return False


def at_window() -> bool:
    """True after the climb ends on an actual window's top edge."""
    with _lock:
        source = _at_window_source
    try:
        return bool(source and source())
    except Exception:
        return False


def _send(command: str) -> bool:
    if not running():
        return False
    _commands.put(command)
    return True


def feed() -> bool:
    """Give her food (the bowl appears, she eats). Any thread."""
    return _send("feed")


def water() -> bool:
    """Give her water. Any thread."""
    return _send("water")


def play() -> bool:
    """Play ball with her. Any thread."""
    return _send("play")


#: Spanish names for her activities (wawi/mascot.ACTIVITIES), for trick().
TRICK_ALIASES = {
    "zoomies": "zoomies", "correr": "zoomies", "carrera": "zoomies",
    "saltos": "jump_tries", "saltar": "jump_tries", "salto": "jump_tries",
    "porrazo": "jump_flop", "caida": "jump_flop",
    "paracaidas": "parachute", "paracaídas": "parachute",
    "hueso": "play_chew", "pelota": "play_ball",
    "pajaro": "bird_chase", "pájaro": "bird_chase",
    "mariposa": "butterfly", "hoja": "catch_leaf",
    "sacudirse": "shake", "sacudida": "shake",
    # Not an activity: her body climbs onto a window with the ladder.
    "escalera": "ladder", "escalar": "ladder", "trepar": "ladder",
}


def trick(name: str) -> bool:
    """Do one of her activities as soon as she is idle (``parachute``,
    ``bird_chase``… or a Spanish name: ``paracaidas``, ``pajaro``; and
    ``escalera``: up onto a window with her ladder). For the
    owner to see each one on demand. Any thread."""
    key = (name or "").strip().lower()
    return _send("trick:" + TRICK_ALIASES.get(key, key)) if key else False


#: The personality she is running with (set by the body), for menus/tools.
_personality_name: str | None = None


def set_personality(name: str) -> bool:
    """Become the personality *name* (already resolved: see
    wawi.pet.personality.match) from the next frame on. Any thread."""
    name = (name or "").strip()
    return _send("personality:" + name) if name else False


def personality() -> str | None:
    """The personality she has now, or None when she is not running."""
    with _lock:
        return _personality_name if _status_source is not None else None


def _set_personality_name(name: str | None) -> None:
    global _personality_name
    with _lock:
        _personality_name = name


def status() -> str:
    """A short Spanish summary of how she is. Any thread."""
    with _lock:
        source = _status_source
    if source is None:
        return "Wawi no está en pantalla ahora."
    try:
        return source()
    except Exception:
        return "No sé bien cómo está Wawi ahora."


def needs() -> dict[str, Any] | None:
    """Her mood and needs as numbers, or None when she is not running::

        {"mood": "hungry", "hunger": 0.7, "thirst": 0.4, "energy": 0.8,
         "boredom": 0.3, "loneliness": 0.5, "summary": "Tiene hambre: …"}

    hunger/thirst/boredom/loneliness: 0 = fine, 1 = badly wanted; energy:
    1 = well rested. Any thread.
    """
    with _lock:
        source = _levels_source
    if source is None:
        return None
    try:
        return source()
    except Exception:
        return None


def drain() -> list[str]:
    """Take every waiting command (main thread, once per frame)."""
    out: list[str] = []
    while True:
        try:
            out.append(_commands.get_nowait())
        except queue.Empty:
            return out
