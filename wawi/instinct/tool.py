"""The voice tool: "Wawi, ¿qué sabés de mí?" and "olvidá lo que aprendiste".

Answered locally from the pattern miner -- no Claude call is needed to say
what she saw. Kept here so wawi/tools.py only needs one small block.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA: dict[str, Any] = {
    "name": "about_me",
    "animation": "notebook",
    # Terse: every schema is re-sent (and paid for) on every turn.
    "description": "¿Qué sabés de mí?/¿qué patrones viste? forget=true: olvidá lo que aprendiste",
    "input_schema": {"type": "object", "properties": {"forget": {"type": "boolean"}}},
}

#: Spanish for "she learned you like it when she…", per instinct.
LIKES = {
    "welcome": "te salude cuando volvés", "focus_company": "te acompañe cuando te concentrás",
    "focus_done": "festeje tus bloques largos", "break_ball": "te traiga la pelota para la pausa",
    "late_night": "te avise cuando se hace tarde", "frenzy": "te mire de reojo",
    "hungry": "te pida comida", "lunch": "te pregunte si comiste",
    "noticed": "te diga lo que noto de tus rutinas",
}


def _learner() -> Any:
    try:
        from wawi import learn

        return learn.get()
    except Exception:
        return None


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def summary(now: float | None = None, *, show: bool = True) -> str:
    """A short spoken summary of the owner's patterns (and liked instincts)."""
    from wawi import instinct

    inst = instinct.get()
    if inst is None:
        return "No estoy mirando tus rutinas: los instintos están apagados en la configuración."
    if not inst.settings.observe:
        return "No miro cómo va tu día: lo tenés apagado en la configuración (instinct.observe)."
    pats = _pick(inst.miner.patterns(now, 0.35))
    if show:
        _notebook(inst)
    if not pats:
        days = sum(1 for _, d in inst.store.recent() if sum(d.get("act", ())) >= 3600)
        if days == 0:
            return "Todavía nada: recién empiezo a mirarte. Dame unos días y te cuento."
        word = "día" if days == 1 else "días"
        return (f"Todavía te estoy conociendo: llevo {days} {word} mirándote. "
                "En una semana te cuento tus rutinas.")
    lines = []
    for p in pats:
        text = p.text if p.confidence >= 0.5 else "creo que " + _lower_first(p.text)
        lines.append(text[:1].upper() + text[1:])
    text = "Esto vi: " + ". ".join(lines) + "."
    liked = _liked()
    if liked:
        text += f" Y aprendí que te gusta que {liked}."
    return text


#: What is worth saying first: the routines people recognise themselves in.
ORDER = ("start", "lunch", "app:", "focus", "breaks", "late", "end", "start_weekend",
         "switching", "hours")
#: The spoken answer is clipped at 240 characters (tools.MAX_SPEAK_CHARS):
#: stop adding facts before that.
MAX_SPOKEN = 200


def _rank(key: str) -> int:
    return next((i for i, k in enumerate(ORDER) if key == k or
                 (k.endswith(":") and key.startswith(k))), len(ORDER))


def _pick(pats: list[Any]) -> list[Any]:
    out, size = [], len("Esto vi: ")
    for p in sorted(pats, key=lambda p: (_rank(p.key), -p.confidence)):
        if size + len(p.text) + 12 > MAX_SPOKEN or len(out) >= 4:
            continue
        out.append(p)
        size += len(p.text) + 12
    return out


def _liked() -> str:
    lr = _learner()
    if lr is None or not hasattr(lr, "instinct_likes"):
        return ""
    try:
        names = [n for n in lr.instinct_likes(2) if n in LIKES]
    except Exception:
        return ""
    words = [LIKES[n] for n in names]
    return " y que ".join(words)


def _notebook(inst: Any) -> None:
    """She writes in her notebook while she tells you (a frame from the
    instinct art; the body falls back to a head tilt without it)."""
    try:
        from wawi.reactions import Reaction

        inst.engine.bus.push(Reaction(behaviour="pattern_notebook", priority=1, duration=4.0))
    except Exception:
        log.debug("instinct: notebook reaction failed", exc_info=True)


def forget_all() -> str:
    """Wipe what she observed and what she learned. Local files only."""
    from wawi import instinct
    from wawi.instinct.settings import InstinctSettings

    inst = instinct.get()
    if inst is not None:
        inst.forget()
    else:
        _unlink(InstinctSettings().state_file)
    lr = _learner()
    if lr is not None and hasattr(lr, "forget"):
        lr.forget()
    elif lr is None:
        from wawi.learn.settings import LearnSettings

        s = LearnSettings()
        for p in (s.state_file, s.brain_file, s.life_log):
            _unlink(p)
    return ("Listo: me olvidé de tus horarios, de tus rutinas y de lo que aprendí de tus "
            "gustos. Eran datos que estaban solo en esta compu. Empiezo de cero.")


def _unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        log.warning("instinct: could not delete %s", path, exc_info=True)


def about_me(forget: bool = False) -> tuple[bool, str]:
    if forget:
        return True, forget_all()
    from wawi import instinct

    return instinct.get() is not None, summary()
