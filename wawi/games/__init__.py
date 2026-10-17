"""Games against Wawi — and the times she asks you to play.

Piedra, papel o tijera (she learns your habits), ta-te-tí on a little board
beside her, adivinar el número both ways round, fetch with a ball you throw
with the mouse, escondidas behind your windows, and a dog trivia. Played by
voice, by clicking her bubble's buttons, or with the mouse.

* :mod:`.core` — the game contract and reading spoken Spanish.
* :mod:`.rps`, :mod:`.tictactoe`, :mod:`.guess`, :mod:`.fetch`,
  :mod:`.hide`, :mod:`.trivia` — the games, all pure.
* :mod:`.propose` — when she proposes one, and learning from the answer.
* :mod:`.manager` — one game at a time, scores, reactions.
* :mod:`.ui` — the AppKit side (bubble with buttons, board, ball), driven by
  the panel through :class:`.ui.GamesHost`.

Module-level helpers below are the thread-safe door for the voice turn and
the ``play_game`` tool; they do nothing (and answer None) when no manager is
installed (``games.enabled: false``, or no panel).
"""

from __future__ import annotations

import logging
from typing import Any

from wawi.games.settings import GamesSettings

log = logging.getLogger(__name__)

_MANAGER: Any = None

__all__ = ["GamesSettings", "build", "get", "hear", "install", "play", "stop"]


def get() -> Any:
    return _MANAGER


def install(manager: Any) -> None:
    global _MANAGER
    _MANAGER = manager


def build(cfg: Any) -> Any:
    """A GameManager for *cfg* (its raw ``games:`` section), or None when
    ``games.enabled`` is off. Never raises: a game is never worth a crash."""
    try:
        settings = GamesSettings.from_dict(getattr(cfg, "games", None) or {})
        if not settings.enabled:
            return None
        from wawi.games.manager import GameManager

        return GameManager(settings)
    except Exception:
        log.warning("games: could not start; Wawi does not play today", exc_info=True)
        return None


def hear(text: str) -> str | None:
    """Offer a transcript to the games. The answer to say, or None when it
    is not for them (then it goes to the brain as usual). Any thread."""
    m = _MANAGER
    if m is None:
        return None
    try:
        return m.hear(text)
    except Exception:
        log.debug("games: hear failed", exc_info=True)
        return None


def play(name: str, move: str = "") -> tuple[bool, str]:
    """The ``play_game`` tool: start *name* ("stop" ends the game); with a
    *move* while that game runs, play it."""
    m = _MANAGER
    if m is None:
        return False, "Ahora no puedo jugar: los juegos están apagados."
    try:
        n = (name or "").strip().lower()
        if n in ("stop", "parar", "terminar", "basta", "fin"):
            return True, m.stop()
        if move and m.playing is not None:
            said = m.hear(move)
            if said is not None:
                return True, said
        return True, m.start(n)
    except Exception:
        log.warning("games: play failed", exc_info=True)
        return False, "Uy, se me trabó el juego."


def stop() -> str:
    m = _MANAGER
    return "No estamos jugando a nada." if m is None else m.stop()
