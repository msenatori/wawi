"""Guarded access to the parts of Wawi the mind does not own.

The body (``wawi.pet.api``) and the sensors (``wawi.sensors``,
``wawi.reflexes``) are built in parallel and may not exist in a given
checkout. Every lookup here is an import at CALL time, wrapped so a missing
module or a missing function reads as "not available" instead of a crash.
Nothing else in the mind imports those modules directly.

Expected APIs (all optional):

* ``wawi.pet.api.feed() / water() / play() / status()`` -- each returns a
  short Spanish sentence (str) or a dict; status() a dict such as
  ``{"mood": "contenta", "hunger": 0.3, "thirst": 0.1, "energy": 0.8}``.
* ``wawi.sensors.status()`` -- a dict or short text describing what the
  sensors see right now.
* ``wawi.reflexes.is_quiet()`` -- True during quiet mode (meeting, full
  screen, quiet hours). ``quiet()`` / ``BUDGET.is_quiet()`` are also tried.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

#: Sentinel: the peer module or function does not exist in this checkout.
MISSING = object()


def _function(module: str, name: str) -> Callable[..., Any] | None:
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    except Exception:  # a peer module that crashes on import is just "missing"
        log.warning("mind: importing %s failed", module, exc_info=True)
        return None
    fn = getattr(mod, name, None)
    return fn if callable(fn) else None


def call(module: str, name: str, *args: Any) -> Any:
    """Call ``module.name(*args)``; :data:`MISSING` if it does not exist.

    Exceptions raised by the peer propagate: "exists and failed" is a
    different answer from "does not exist", and the caller words them
    differently.
    """
    fn = _function(module, name)
    if fn is None:
        return MISSING
    return fn(*args)


def available(module: str, name: str) -> bool:
    return _function(module, name) is not None


def pet_available() -> bool:
    return available("wawi.pet.api", "status")


def sensors_available() -> bool:
    return available("wawi.sensors", "status")


def pet_status() -> Any:
    """The pet's status dict, or None when there is no pet (or it failed)."""
    try:
        value = call("wawi.pet.api", "status")
    except Exception:
        log.debug("mind: pet status failed", exc_info=True)
        return None
    return None if value is MISSING else value


def sensors_status() -> Any:
    try:
        value = call("wawi.sensors", "status")
    except Exception:
        log.debug("mind: sensors status failed", exc_info=True)
        return None
    return None if value is MISSING else value


def is_quiet() -> bool | None:
    """True/False from the reflex layer's quiet mode, or None when unknown.

    Callers decide what "unknown" means: a bubble is harmless and goes
    ahead; a spoken sentence is not and stays silent.
    """
    for name in ("is_quiet", "quiet"):
        fn = _function("wawi.reflexes", name)
        if fn is not None:
            try:
                return bool(fn())
            except Exception:
                log.debug("mind: reflexes.%s failed", name, exc_info=True)
                return None
    try:
        mod = importlib.import_module("wawi.reflexes")
        budget = getattr(mod, "BUDGET", None)
        fn = getattr(budget, "is_quiet", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    return None


def self_model() -> dict[str, Any] | None:
    """Her self-model from wawi.instinct (what she noticed today, what she
    knows of the owner's routines), or None when instincts are not there."""
    try:
        value = call("wawi.instinct", "self_notes")
    except Exception:
        log.debug("mind: self model failed", exc_info=True)
        return None
    return value if isinstance(value, dict) else None
