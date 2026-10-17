"""Wawi's mind: a little judgement on top of rules that already work.

Public surface (kept small so the sensors and the body can call it without
knowing anything else about this package):

* :func:`build` -- make a :class:`~wawi.mind.core.Mind` from the app config.
* :func:`install` / :func:`get` -- the process-wide mind (or None).
* :func:`triage` -- "now" / "later" / "ignore" for an info/notice event, or
  None meaning "use your own rule". Safe to call when no mind is installed.

Importing this package is cheap: no model, no SDK, no network until used.
"""

from __future__ import annotations

import logging
from typing import Any

from wawi.mind.settings import MindSettings, parse

log = logging.getLogger(__name__)

_MIND: Any = None

__all__ = ["MindSettings", "build", "get", "install", "parse", "triage"]


def get() -> Any:
    return _MIND


def install(mind: Any) -> None:
    global _MIND
    _MIND = mind


def triage(event: Any, timeout: float | None = None) -> str | None:
    """Ask the installed mind to triage *event*; None when there is no mind."""
    mind = _MIND
    if mind is None:
        return None
    return mind.triage(event, timeout)


def build(cfg: Any) -> Any:
    """A started-but-idle Mind for *cfg*, or None when ``mind.enabled`` is off."""
    from wawi.mind.claude import ClaudeMind, RateLimiter
    from wawi.mind.core import Mind
    from wawi.mind.local import build_local_model

    settings: MindSettings = getattr(cfg, "mind", None) or MindSettings()
    if not settings.enabled:
        return None
    local = build_local_model(
        settings.local_model,
        settings.local_backend,
        brain_backend=getattr(cfg, "backend", "claude"),
        ollama_url=getattr(cfg, "ollama_url", "http://localhost:11434/api/chat"),
        load_timeout_s=settings.local_load_timeout_s,
        call_timeout_s=settings.local_timeout_s,
    )
    claude = None
    if getattr(cfg, "backend", "claude") == "claude":
        claude = ClaudeMind(
            model=settings.claude_model or getattr(cfg, "model_on_demand", ""),
            timeout_s=settings.claude_timeout_s,
            limiter=RateLimiter(
                min_interval_s=settings.claude_min_interval_s,
                max_per_hour=settings.claude_max_per_hour,
            ),
        )
    log.info(
        "mind: on (local model: %s, Claude for: %s)",
        settings.local_model if local else "none",
        ", ".join(settings.claude_events) if claude else "nothing",
    )
    return Mind(settings, claude=claude, local=local)
