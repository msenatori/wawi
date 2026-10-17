"""Wawi learns from her owner, with tiny local models and nothing else.

No LLM, no GPU, no download needed: a contextual bandit over her idle
behaviours (:mod:`wawi.learn.bandit`), per-kind alert engagement
(:mod:`wawi.learn.alerts`), headline relevance from keywords and learned
words (:mod:`wawi.learn.relevance`, with optional static embeddings), and a
small numpy MLP that learns the owner's rhythm while the machine idles
(:mod:`wawi.learn.brain`). All of it is numpy, which Wawi already needs.

Public surface, safe to call when nothing is installed (it then does
nothing and answers None):

* :func:`build` / :func:`install` / :func:`get` -- the process-wide learner.
* :func:`signal` -- the owner did something (touch, pet, drag, hide…).
* :func:`triage` -- the reflex layer's hook: "now" / "later" / "ignore" / None.
* :func:`summary` -- "¿qué aprendiste de mí?".
"""

from __future__ import annotations

import logging
from typing import Any

from wawi.learn.settings import LearnSettings

log = logging.getLogger(__name__)

_LEARNER: Any = None

__all__ = ["LearnSettings", "actions", "build", "get", "install", "signal", "summary", "triage"]


def get() -> Any:
    return _LEARNER


def install(learner: Any) -> None:
    global _LEARNER
    _LEARNER = learner


def actions() -> list[str]:
    """Every idle behaviour the animator may pick: its own weights, plus the
    acts only a mood brings in (the brain needs a fixed list of outputs)."""
    from wawi.mascot import BEHAVIOUR_WEIGHTS
    from wawi.pet.needs import MOOD_BIAS

    names = list(BEHAVIOUR_WEIGHTS)
    for bias in MOOD_BIAS.values():
        names += [n for n in bias if n not in names]
    return names


def build(cfg: Any) -> Any:
    """A loaded Learner for *cfg* (its raw ``learn:`` section), or None when
    ``learn.enabled`` is off. Never raises: learning is never worth a crash."""
    try:
        from wawi.learn.learner import Learner

        settings = LearnSettings.from_dict(getattr(cfg, "learn", None) or {})
        if not settings.enabled:
            return None
        learner = Learner(settings, actions=actions())
        learner.load()
        log.info("learn: on (behaviours %s, alerts %s, brain %s, embeddings %s)",
                 settings.behaviours, settings.alerts, settings.brain,
                 "loading" if learner.relevance.embedder is not None else "no")
        return learner
    except Exception:
        log.warning("learn: could not start; Wawi runs without learning", exc_info=True)
        return None


def signal(kind: str, now: float | None = None) -> None:
    learner = _LEARNER
    if learner is None:
        return
    try:
        learner.signal(kind, now)
    except Exception:
        log.debug("learn: signal %s failed", kind, exc_info=True)


def triage(event: Any) -> str | None:
    learner = _LEARNER
    if learner is None:
        return None
    try:
        return learner.triage(event)
    except Exception:
        log.debug("learn: triage failed", exc_info=True)
        return None


def summary() -> str:
    learner = _LEARNER
    if learner is None:
        return "No estoy aprendiendo nada todavía: el aprendizaje está apagado."
    return learner.summary()
