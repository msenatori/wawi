"""Iniciativa: Wawi notices what the owner needs and acts on her own.

The instincts (wawi/instinct) make her a pet with a life; the initiative
makes her an assistant that does not wait to be asked. A small loop (every
~25 s, a background thread, cheap) turns local signals into **moments** and
moments into **acts**, each at a level the owner shapes without being asked:
``quiet`` (done silently), ``offer`` (a bubble with a link, silent), ``say``
(spoken, rare) or ``auto`` (done for them). See :mod:`.core` for the loop and
the budget, :mod:`.learning` for how levels move.

The moments today, one module each:

* :mod:`.meeting` -- **en reunión → se calla sola.** Google Meet in a running
  browser, Zoom in a call, a meeting app in front. While it lasts the
  process-wide :func:`in_meeting` flag silences every speaker (reflexes'
  quiet mode, so the reaction voice, the instincts and the mind; the
  curiosities; the dance; her barks; spoken reactions). After it: "¿Anoto lo
  que quedó?" → a note (wawi/notes).
* :mod:`.orca_watch` -- **un agente de Orca terminó o te espera.** She walks
  over and says so in a bubble; the link brings that terminal forward.
* :mod:`.morning` -- **resumen de la mañana.** On the first activity of the
  day, a "Buen día" board page prepared quietly, then offered.

Later moments, each a small watcher registered with
:func:`wawi.initiative.core.register` plus a row in ``KINDS``:

* a new file in Downloads → offer to open / move / summarise it;
* long uncommitted work in a git repo → offer a commit message or a stash;
* end of the day → a summary note of what got done (notes, Orca, git);
* stuck on the same thing for a long while → opt-in "¿te ayudo?" with a
  screen look (the only one that would need a brain turn, and only on yes).

Privacy: everything is local. No window titles or screen contents are read;
the only browser look is whether a Meet call tab exists (URLs never logged
or kept). Nothing goes to Claude: none of these acts runs a brain turn.

Public surface, safe to call when nothing is installed:

* :func:`build` / :func:`install` / :func:`get`.
* :func:`in_meeting` -- the quiet flag for every speaker.
* :func:`bubble_text` / :func:`bubble_link` / :func:`owns` / :func:`click` --
  the panel's bubble.
* :func:`menu_item` -- the "Iniciativa ▸" submenu.
* :func:`run_tool` -- the voice tool (:mod:`.tool`).
"""

from __future__ import annotations

import logging
from typing import Any

from wawi.initiative.meeting import in_meeting

log = logging.getLogger(__name__)

__all__ = ["build", "bubble_link", "bubble_text", "click", "get", "in_meeting", "install",
           "menu_item", "owns", "run_tool"]

_ENGINE: Any = None


def get() -> Any:
    return _ENGINE


def install(engine: Any) -> None:
    global _ENGINE
    _ENGINE = engine


def _load_watchers() -> None:
    """Import the moment modules: each registers its watcher on import
    (with its polling order). meeting.py is already in, through in_meeting."""
    from wawi.initiative import morning, orca_watch  # noqa: F401


def build(cfg: Any, hooks: Any = None) -> Any:
    """A (not started) engine for *cfg*'s ``initiative:`` section, or None
    when it is off. Never raises: initiative is never worth a crash."""
    try:
        from wawi.initiative.core import Initiative, InitiativeSettings
        from wawi.initiative.learning import LearningStore

        settings = InitiativeSettings.from_dict(getattr(cfg, "initiative", None) or {})
        if not settings.enabled:
            return None
        _load_watchers()
        try:
            from wawi.pet.settings import PetSettings

            pet = PetSettings.from_dict(getattr(cfg, "pet", {}) or {})
            from dataclasses import replace

            settings = replace(settings, night_start=pet.night_start, night_end=pet.night_end)
        except Exception:
            pass
        from wawi import paths as _paths

        store = LearningStore(_paths.resolve(settings.local_file) if settings.local_file else None)
        engine = Initiative(settings, store=store, hooks=hooks)
        log.info("initiative: %s (%s)", "on" if engine.enabled else "off (menu)",
                 ", ".join(type(w).__name__ for w in engine.watchers))
        return engine
    except Exception:
        log.warning("initiative: could not start; Wawi runs without it", exc_info=True)
        return None


def bubble_text() -> str | None:
    engine = _ENGINE
    if engine is None:
        return None
    try:
        return engine.bubble_text()
    except Exception:
        return None


def bubble_link(text: str | None) -> str | None:
    engine = _ENGINE
    if engine is None:
        return None
    try:
        return engine.bubble_link(text)
    except Exception:
        return None


def owns(text: str | None) -> bool:
    engine = _ENGINE
    try:
        return engine is not None and engine.owns(text)
    except Exception:
        return False


def click(on_link: bool) -> None:
    engine = _ENGINE
    if engine is not None:
        try:
            engine.click(on_link)
        except Exception:
            log.debug("initiative: click failed", exc_info=True)


def menu_item() -> Any:
    """The "Iniciativa ▸" NSMenuItem, or None (nothing installed / no AppKit)."""
    engine = _ENGINE
    if engine is None:
        return None
    try:
        from wawi.initiative.menu import build_item

        return build_item(engine)
    except Exception:
        log.debug("initiative: no menu", exc_info=True)
        return None


def run_tool(action: str = "list", kind: str = "") -> tuple[bool, str, dict]:
    from wawi.initiative.tool import run

    return run(_ENGINE, action, kind)
