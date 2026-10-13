"""State machine and observer registry for Wawi.

The orchestrator calls set_state(). UI and earcon modules register as observers
and react to (old, new) transitions without the orchestrator knowing about them.
"""

from __future__ import annotations

import enum
import logging
import threading
from typing import Protocol

log = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class StateObserver(Protocol):
    """Any callable that accepts (old_state, new_state)."""

    def __call__(self, old: State, new: State) -> None: ...


class StateMachine:
    """Thread-safe state machine with a subscriber registry.

    Notification contract:
    - Observers are snapshotted under the lock, then called outside it.
    - This prevents an observer re-entrantly calling set_state() from deadlocking.
    - Observer exceptions are caught and logged; a broken observer never kills the pipeline.
    """

    def __init__(self) -> None:
        self._state = State.IDLE
        self._observers: list[StateObserver] = []
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def subscribe(self, obs: StateObserver) -> None:
        """Register an observer to receive (old, new) state transition events."""
        with self._lock:
            self._observers.append(obs)

    def set_state(self, new: State) -> None:
        """Transition to *new* and notify all observers with (old, new).

        The internal lock is released before any observer is called to prevent
        re-entrant deadlocks. Observers run synchronously on the calling thread.
        """
        with self._lock:
            old = self._state
            self._state = new
            snapshot = list(self._observers)

        for obs in snapshot:
            try:
                obs(old, new)
            except Exception:
                log.warning("Observer %r raised an exception; ignoring", obs, exc_info=True)
